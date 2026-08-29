import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from calibration_tool.camera import load_capture_plan


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "geometry_experiment.py"
SPEC = importlib.util.spec_from_file_location("geometry_experiment", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
GEOMETRY_EXPERIMENT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GEOMETRY_EXPERIMENT)


class GeometryExperimentTests(unittest.TestCase):
    def _write_complete_dataset(self, root: Path, config_id: str, exposure_us: float) -> Path:
        dataset = root / config_id
        frames = []
        csv_rows = []
        csv_fields = (
            "task_id", "index", "filename", "quality_passed", "quality_warnings",
            "transport_warnings", "exposure_us", "gain_db", "pixel_format",
            "width", "height", "offset_x", "offset_y",
        )
        for task_id in ("reference", "multiheight"):
            image_dir = dataset / "images" / task_id
            image_dir.mkdir(parents=True, exist_ok=True)
            for index in range(1, 51):
                relative = f"images/{task_id}/{index:04d}.tif"
                (dataset / relative).write_bytes(b"unchanged-image-bytes")
                frames.append({"task_id": task_id, "index": index, "filename": relative})
                csv_rows.append({
                    "task_id": task_id,
                    "index": index,
                    "filename": relative,
                    "quality_passed": index != 1,
                    "quality_warnings": "dynamic_range_low" if index == 1 else "",
                    "transport_warnings": "",
                    "exposure_us": exposure_us,
                    "gain_db": 0.0,
                    "pixel_format": "Mono8",
                    "width": 2448,
                    "height": 2048,
                    "offset_x": 0,
                    "offset_y": 0,
                })
        manifest = {
            "status": "completed",
            "plan": {"metadata": {"config_id": config_id}},
            "tasks": {
                task_id: {
                    "status": "completed",
                    "frames_expected": 50,
                    "frames_captured": 50,
                }
                for task_id in ("reference", "multiheight")
            },
            "frames": frames,
        }
        (dataset / "dataset_manifest.yaml").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        with (dataset / "frames.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=csv_fields)
            writer.writeheader()
            writer.writerows(csv_rows)
        return dataset

    def test_init_creates_fixed_matrix_and_blank_mutable_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            experiment_dir = Path(temporary) / "geometry_baseline_angle"
            master_path = GEOMETRY_EXPERIMENT.initialize_experiment(experiment_dir)

            for relative_dir in ("configs", "configs/generated", "data", "results"):
                self.assertTrue((experiment_dir / relative_dir).is_dir())

            with master_path.open(encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                rows = list(reader)

            self.assertEqual(tuple(reader.fieldnames or ()), GEOMETRY_EXPERIMENT.CSV_FIELDS)
            self.assertEqual(len(rows), 12)
            self.assertEqual(
                [row["config_id"] for row in rows],
                [
                    "B00_A05", "B00_A10", "B00_A15", "B00_A20",
                    "B05_A05", "B05_A10", "B05_A15", "B05_A20",
                    "B12p5_A05", "B12p5_A10", "B12p5_A15", "B12p5_A20",
                ],
            )
            self.assertEqual(
                [row["baseline_scale_reading"] for row in rows],
                ["0"] * 4 + ["5"] * 4 + ["12.5"] * 4,
            )
            for row in rows:
                for field in (*GEOMETRY_EXPERIMENT.MANUAL_FIELDS, *GEOMETRY_EXPERIMENT.AUTOMATED_FIELDS):
                    self.assertEqual(row[field], "")

    def test_existing_master_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            experiment_dir = Path(temporary) / "geometry_baseline_angle"
            master_path = GEOMETRY_EXPERIMENT.initialize_experiment(experiment_dir)
            original = master_path.read_bytes()

            with self.assertRaises(FileExistsError):
                GEOMETRY_EXPERIMENT.initialize_experiment(experiment_dir)

            self.assertEqual(master_path.read_bytes(), original)

    def test_make_plan_generates_two_tasks_per_configuration(self):
        with tempfile.TemporaryDirectory() as temporary:
            experiment_dir = Path(temporary) / "geometry_baseline_angle"
            master_path = GEOMETRY_EXPERIMENT.initialize_experiment(experiment_dir)
            original_master = master_path.read_bytes()

            paths = GEOMETRY_EXPERIMENT.make_capture_plans(experiment_dir)

            self.assertEqual(len(paths), 12)
            self.assertEqual(master_path.read_bytes(), original_master)
            first = load_capture_plan(paths[0])
            last = load_capture_plan(paths[-1])
            self.assertEqual([task.task_id for task in first.tasks], ["reference", "multiheight"])
            self.assertEqual([task.frames for task in first.tasks], [50, 50])
            self.assertEqual([task.quality_mode for task in first.tasks], ["laser", "laser"])
            self.assertEqual([task.config.exposure_us for task in first.tasks], [1500.0, 1500.0])
            self.assertEqual(first.tasks[0].relative_path(1).as_posix(), "images/reference/0001.tif")
            self.assertEqual(first.tasks[1].relative_path(50).as_posix(), "images/multiheight/0050.tif")
            self.assertEqual(first.metadata["baseline_scale_reading"], 0)
            self.assertIsNone(first.metadata["baseline_actual_mm"])
            self.assertEqual(first.metadata["laser_angle_deg"], 5)
            self.assertEqual(last.metadata["baseline_scale_reading"], 12.5)
            self.assertEqual(last.metadata["laser_angle_deg"], 20)
            self.assertEqual([task.config.exposure_us for task in last.tasks], [1900.0, 1900.0])
            self.assertEqual(first.metadata["working_distance_nominal_mm"], 1000)
            self.assertFalse(first.metadata["working_distance_calibrated"])

    def test_make_plan_preserves_manual_master_fields_and_refuses_plan_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            experiment_dir = Path(temporary) / "geometry_baseline_angle"
            master_path = GEOMETRY_EXPERIMENT.initialize_experiment(experiment_dir)
            rows = GEOMETRY_EXPERIMENT._load_master_rows(master_path)
            rows[0]["baseline_actual_mm"] = "42.75"
            rows[0]["manual_notes"] = "人工测量"
            with master_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=GEOMETRY_EXPERIMENT.CSV_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            original_master = master_path.read_bytes()

            paths = GEOMETRY_EXPERIMENT.make_capture_plans(experiment_dir)
            first = load_capture_plan(paths[0])
            self.assertEqual(first.metadata["baseline_actual_mm"], 42.75)
            with self.assertRaises(FileExistsError):
                GEOMETRY_EXPERIMENT.make_capture_plans(experiment_dir)
            self.assertEqual(master_path.read_bytes(), original_master)

    def test_audit_captures_handles_invalid_fov_and_preserves_images_and_manual_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            experiment_dir = Path(temporary) / "geometry_baseline_angle"
            master_path = GEOMETRY_EXPERIMENT.initialize_experiment(experiment_dir)
            fieldnames, master_rows = GEOMETRY_EXPERIMENT._read_master_table(master_path)
            master_rows[0]["baseline_actual_mm"] = "42.5"
            master_rows[0]["manual_notes"] = "视场外，人工确认"
            GEOMETRY_EXPERIMENT._write_csv(master_path, fieldnames, master_rows)

            data_root = experiment_dir / "data"
            expected_ids = [row["config_id"] for row in GEOMETRY_EXPERIMENT.build_initial_rows()]
            for config_id in expected_ids:
                if config_id == "B00_A05":
                    continue
                exposure = 1900.0 if config_id == "B12p5_A20" else 1500.0
                self._write_complete_dataset(data_root, config_id, exposure)
            sample_image = data_root / "B00_A10" / "images" / "reference" / "0001.tif"
            original_image = sample_image.read_bytes()

            result = GEOMETRY_EXPERIMENT.audit_captures(data_root, master_path)

            self.assertEqual(result["summary"], {
                "expected_conditions": 12,
                "captured_conditions": 11,
                "invalid_fov": 1,
                "complete_datasets": 11,
                "incomplete_datasets": 0,
            })
            self.assertFalse(result["camera_consistency"]["consistent"])
            self.assertEqual(sample_image.read_bytes(), original_image)
            self.assertTrue((experiment_dir / "results" / "capture_audit.csv").is_file())
            self.assertTrue((experiment_dir / "results" / "capture_audit.json").is_file())

            _, updated_rows = GEOMETRY_EXPERIMENT._read_master_table(master_path)
            invalid = updated_rows[0]
            self.assertEqual(invalid["status"], "invalid_fov")
            self.assertEqual(invalid["capture_complete"], "false")
            self.assertEqual(invalid["exclude_reason"], "laser_out_of_fov")
            self.assertEqual(invalid["phaseA_selected"], "false")
            self.assertEqual(invalid["baseline_actual_mm"], "42.5")
            self.assertEqual(invalid["manual_notes"], "视场外，人工确认")
            self.assertTrue(all(invalid[field] == "NaN" for field in GEOMETRY_EXPERIMENT.ANALYSIS_NAN_FIELDS))
            self.assertTrue(all(row["status"] == "captured" for row in updated_rows[1:]))
            first_captured = next(
                record for record in result["datasets"] if record["config_id"] == "B00_A10"
            )
            self.assertEqual(first_captured["quality_warning_frame_count"], 2)
            self.assertEqual(first_captured["quality_warning_occurrence_count"], 2)

    def test_reference_analysis_limits_surface_trims_segments_and_preserves_steger_detail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "B12p5_A20"
            image_dir = dataset / "images" / "reference"
            image_dir.mkdir(parents=True)
            frame_rows = []
            for index in range(1, 51):
                image = np.full((24, 32), 5, dtype=np.uint8)
                image[10:13, :] = 180
                ok, encoded = cv2.imencode(".tif", image)
                self.assertTrue(ok)
                filename = f"images/reference/{index:04d}.tif"
                (dataset / filename).write_bytes(encoded.tobytes())
                frame_rows.append({
                    "task_id": "reference",
                    "index": index,
                    "filename": filename,
                    "pixel_format": "Mono8",
                })
            (dataset / "dataset_manifest.yaml").write_text(
                json.dumps({
                    "status": "completed",
                    "plan": {"metadata": {"config_id": "B12p5_A20"}},
                }),
                encoding="utf-8",
            )
            with (dataset / "frames.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=("task_id", "index", "filename", "pixel_format")
                )
                writer.writeheader(); writer.writerows(frame_rows)
            steger_config = root / "realtime_steger.yaml"
            steger_config.write_text("steger: {}\n", encoding="utf-8")
            analysis_config = root / "analysis.yaml"
            analysis_config.write_text(
                "schema_version: 1\nsteger_config: realtime_steger.yaml\n"
                "reference:\n  valid_frame_fraction_min: 0.8\n  max_interp_gap_px: 1\n"
                "reference_surface:\n  x_range: [1, 30]\n  segment_edge_trim_px: 1\n"
                "  smooth_spline_basis_count: 12\n  smooth_spline_penalty: 1.0\n"
                "  robust_huber_delta: 1.5\n  robust_max_iterations: 10\n"
                "  cv_min_segment_width_px: 5\n",
                encoding="utf-8",
            )

            valid = np.ones(32, dtype=bool)
            valid[8] = False  # one-column gap: local interpolation
            valid[18:22] = False  # larger gap: smooth model only
            fake_realtime = SimpleNamespace(
                load_steger_options=lambda _path: {"sigma": 1.5},
                extract_steger_columns=lambda _image, _options: SimpleNamespace(
                    u_px=np.arange(32, dtype=np.float64),
                    v_px=np.where(valid, 11.25, np.nan),
                    valid=valid,
                    response=np.where(valid, 2.0, np.nan),
                    offset_px=np.where(valid, 0.1, np.nan),
                    normal_y_abs=np.where(valid, 1.0, np.nan),
                ),
            )
            fake_src = root / "calibration_src"
            fake_src.mkdir()
            (fake_src / "realtime_steger.py").write_text("# formal stub\n", encoding="utf-8")
            with patch.object(
                GEOMETRY_EXPERIMENT, "_load_realtime_steger", return_value=fake_realtime
            ):
                summary = GEOMETRY_EXPERIMENT.analyze_reference(
                    dataset, analysis_config, fake_src
                )

            self.assertEqual(summary["source_counts"], {
                "observed": 19,
                "short_gap_interpolated": 1,
                "smooth_model_filled": 4,
                "segment_edge_excluded": 6,
                "outside_reference_surface": 2,
                "invalid": 0,
            })
            self.assertEqual(summary["cross_validation"]["eligible_segment_count"], 3)
            self.assertEqual(summary["cross_validation"]["interior_segment_count"], 1)
            self.assertEqual(summary["cross_validation"]["boundary_segment_count"], 2)
            self.assertEqual(
                summary["cross_validation"]["formal_metrics_scope"],
                "interior_segments_only",
            )
            self.assertFalse(summary["cross_validation"]["used_for_final_model_fit"])
            self.assertFalse(summary["global_line_fit_applied"])
            self.assertFalse(summary["model_extrapolated_outside_reference_surface"])
            self.assertFalse(summary["multiheight_analyzed"])
            detail_path = dataset / "analysis" / "reference_frame_columns.csv"
            detail_before = detail_path.read_bytes()
            with patch.object(
                GEOMETRY_EXPERIMENT, "_load_realtime_steger", return_value=fake_realtime
            ):
                second = GEOMETRY_EXPERIMENT.analyze_reference(
                    dataset, analysis_config, fake_src, overwrite=True
                )
            self.assertTrue(second["per_frame_steger_output_preserved"])
            self.assertEqual(detail_path.read_bytes(), detail_before)
            with (dataset / "analysis" / "reference_by_column.csv").open(
                encoding="utf-8", newline=""
            ) as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[8]["source"], "short_gap_interpolated")
            self.assertEqual(
                [rows[index]["source"] for index in range(18, 22)],
                ["smooth_model_filled"] * 4,
            )
            self.assertEqual([rows[index]["source"] for index in (0, 31)], ["outside_reference_surface"] * 2)
            self.assertTrue(all(rows[index]["y_ref_smooth_px"] == "NaN" for index in (0, 31)))
            self.assertTrue(all(rows[index]["y_ref_smooth_px"] != "NaN" for index in range(1, 31)))
            self.assertEqual(rows[1]["source"], "segment_edge_excluded")
            with (dataset / "analysis" / "reference_cv_by_segment.csv").open(
                encoding="utf-8", newline=""
            ) as stream:
                cv_rows = list(csv.DictReader(stream))
            self.assertEqual(len(cv_rows), 3)
            self.assertEqual(
                [row["segment_role"] for row in cv_rows],
                ["boundary", "interior", "boundary"],
            )
            self.assertIn("reference_cv_interior_rmse_px", summary["statistics"])
            self.assertIn("reference_cv_boundary_rmse_px", summary["statistics"])
            self.assertTrue((dataset / "analysis" / "reference_cv_residual.png").is_file())
            self.assertFalse((dataset / "images" / "multiheight").exists())

    def test_preview_reference_roi_accepts_null_surface_without_building_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "B12p5_A20"
            image_dir = dataset / "images" / "reference"
            image_dir.mkdir(parents=True)
            rows = []
            for index in range(1, 51):
                image = np.full((12, 10), 5, dtype=np.uint8)
                image[6, :] = 200
                ok, encoded = cv2.imencode(".tif", image)
                self.assertTrue(ok)
                filename = f"images/reference/{index:04d}.tif"
                (dataset / filename).write_bytes(encoded.tobytes())
                rows.append({
                    "task_id": "reference", "index": index,
                    "filename": filename, "pixel_format": "Mono8",
                })
            (dataset / "dataset_manifest.yaml").write_text(
                json.dumps({
                    "status": "completed",
                    "plan": {"metadata": {"config_id": "B12p5_A20"}},
                }),
                encoding="utf-8",
            )
            with (dataset / "frames.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=("task_id", "index", "filename", "pixel_format")
                )
                writer.writeheader(); writer.writerows(rows)
            (root / "realtime_steger.yaml").write_text("steger: {}\n", encoding="utf-8")
            config = root / "analysis.yaml"
            config.write_text(
                "schema_version: 1\nsteger_config: realtime_steger.yaml\n"
                "reference:\n  valid_frame_fraction_min: 0.8\n  max_interp_gap_px: 8\n"
                "reference_surface:\n  x_range: null\n  segment_edge_trim_px: 2\n",
                encoding="utf-8",
            )
            valid = np.ones(10, dtype=bool)
            fake_realtime = SimpleNamespace(
                load_steger_options=lambda _path: {"sigma": 1.5},
                extract_steger_columns=lambda _image, _options: SimpleNamespace(
                    u_px=np.arange(10, dtype=np.float64),
                    v_px=np.full(10, 6.0),
                    valid=valid,
                    response=np.ones(10),
                    offset_px=np.zeros(10),
                    normal_y_abs=np.ones(10),
                ),
            )
            fake_src = root / "calibration_src"
            fake_src.mkdir()
            (fake_src / "realtime_steger.py").write_text("# formal stub\n", encoding="utf-8")
            with patch.object(
                GEOMETRY_EXPERIMENT, "_load_realtime_steger", return_value=fake_realtime
            ):
                summary = GEOMETRY_EXPERIMENT.preview_reference_roi(dataset, config, fake_src)
            self.assertIsNone(summary["reference_surface_x_range"])
            self.assertFalse(summary["reference_model_built"])
            self.assertTrue((dataset / "analysis" / "reference_roi_preview.png").is_file())
            with self.assertRaises(ValueError):
                GEOMETRY_EXPERIMENT.analyze_reference(dataset, config, fake_src)
            self.assertFalse((dataset / "images" / "multiheight").exists())

    def test_multiheight_roi_preview_uses_same_steger_and_never_computes_height(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "B12p5_A20"
            image_dir = dataset / "images" / "multiheight"
            image_dir.mkdir(parents=True)
            frame_rows = []
            for index in range(1, 51):
                image = np.full((12, 10), 5, dtype=np.uint8)
                image[6, :] = 200
                ok, encoded = cv2.imencode(".tif", image)
                self.assertTrue(ok)
                filename = f"images/multiheight/{index:04d}.tif"
                (dataset / filename).write_bytes(encoded.tobytes())
                frame_rows.append({
                    "task_id": "multiheight",
                    "index": index,
                    "filename": filename,
                    "pixel_format": "Mono8",
                })
            sample_image = dataset / frame_rows[0]["filename"]
            original_image = sample_image.read_bytes()
            (dataset / "dataset_manifest.yaml").write_text(
                json.dumps({
                    "status": "completed",
                    "plan": {"metadata": {"config_id": "B12p5_A20"}},
                }),
                encoding="utf-8",
            )
            with (dataset / "frames.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=("task_id", "index", "filename", "pixel_format")
                )
                writer.writeheader(); writer.writerows(frame_rows)
            steger_config = root / "realtime_steger.yaml"
            steger_config.write_text("steger: {}\n", encoding="utf-8")
            analysis_config = root / "analysis.yaml"
            analysis_config.write_text(
                "schema_version: 1\nsteger_config: realtime_steger.yaml\n"
                "reference:\n  valid_frame_fraction_min: 0.8\n  max_interp_gap_px: 8\n"
                "reference_surface:\n  x_range: [1, 8]\n  segment_edge_trim_px: 2\n",
                encoding="utf-8",
            )
            registry = root / "roi_registry.yaml"
            registry.write_text(
                "schema_version: 1\ntemplate_config: B12p5_A20\nconfigs:\n"
                "  B12p5_A20:\n    reference_surface:\n      x_range: [1, 8]\n"
                "      method: manual_confirmed\n    multiheight:\n"
                "      h001: {height_mm: 1.0, x_range: [1, 4]}\n"
                "      h010: {height_mm: 10.0, x_range: [5, 8]}\n"
                "      h030: {height_mm: 30.0, x_range: null}\n",
                encoding="utf-8",
            )
            output_dir = dataset / "analysis"
            output_dir.mkdir()
            (output_dir / "reference_analysis.json").write_text(
                json.dumps({
                    "centre_extractor": "realtime_steger.extract_steger_columns",
                    "steger_config_sha256": GEOMETRY_EXPERIMENT.sha256_file(steger_config),
                    "reference_surface_x_range": [1, 8],
                    "model_extrapolated_outside_reference_surface": False,
                    "multiheight_analyzed": False,
                }),
                encoding="utf-8",
            )
            with (output_dir / "reference_by_column.csv").open(
                "w", encoding="utf-8", newline=""
            ) as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=("u", "y_ref_smooth_px", "source")
                )
                writer.writeheader()
                for u in range(10):
                    inside = 1 <= u <= 8
                    writer.writerow({
                        "u": u,
                        "y_ref_smooth_px": 6.0 if inside else "NaN",
                        "source": "observed" if inside else "outside_reference_surface",
                    })
            valid = np.ones(10, dtype=bool)
            fake_realtime = SimpleNamespace(
                load_steger_options=lambda _path: {"sigma": 1.5},
                extract_steger_columns=lambda _image, _options, **_kwargs: SimpleNamespace(
                    u_px=np.arange(10, dtype=np.float64),
                    v_px=np.full(10, 6.0),
                    valid=valid,
                    response=np.ones(10),
                    offset_px=np.zeros(10),
                    normal_y_abs=np.ones(10),
                    metadata={},
                ),
            )
            fake_src = root / "calibration_src"
            fake_src.mkdir()
            (fake_src / "realtime_steger.py").write_text("# formal stub\n", encoding="utf-8")
            with patch.object(
                GEOMETRY_EXPERIMENT, "_load_realtime_steger", return_value=fake_realtime
            ):
                summary = GEOMETRY_EXPERIMENT.annotate_multiheight_rois(
                    dataset,
                    registry,
                    analysis_config,
                    fake_src,
                    preview_only=True,
                )
            self.assertEqual(summary["frame_count"], 50)
            self.assertEqual(summary["rois"]["h001"]["width_px"], 4)
            self.assertEqual(summary["rois"]["h001"]["steger_valid_column_fraction"], 1.0)
            self.assertTrue(summary["rois"]["h010"]["fully_inside_reference_surface"])
            self.assertIsNone(summary["rois"]["h030"]["x_range"])
            self.assertFalse(summary["delta_y_computed"])
            self.assertFalse(summary["reference_model_modified"])
            self.assertEqual(sample_image.read_bytes(), original_image)
            self.assertFalse((dataset / "images" / "reference").exists())
            self.assertTrue((output_dir / "multiheight_median.png").is_file())
            self.assertTrue((output_dir / "multiheight_roi_preview.png").is_file())
            _, _, config_entry = GEOMETRY_EXPERIMENT._load_roi_registry(
                registry, "B12p5_A20"
            )
            with self.assertRaises(ValueError):
                GEOMETRY_EXPERIMENT._validate_roi_candidate(
                    config_entry, "h030", (4, 6), 10
                )

    def test_plateau_detection_erodes_stable_run_and_never_falls_back_to_selected_roi(self):
        config = {
            "multiheight_valid_fraction_min": 0.8,
            "plateau_median_window_px": 5,
            "plateau_max_gradient_px_per_column": 0.08,
            "plateau_max_step_px": 0.25,
            "plateau_sigma_mad_scale": 4.0,
            "plateau_sigma_floor_limit_px": 0.03,
            "plateau_erosion_px": 2,
            "min_stable_width_px": 12,
        }
        delta = np.full(48, np.nan, dtype=np.float64)
        delta[5:9] = [-2.0, -1.7, -1.4, -1.1]
        delta[9:28] = -1.0 + 0.01 * np.sin(np.arange(19))
        delta[28:32] = [-1.1, -1.4, -1.7, -2.0]
        sigma = np.full(48, 0.01, dtype=np.float64)
        valid_fraction = np.ones(48, dtype=np.float64)

        detected = GEOMETRY_EXPERIMENT._detect_stable_plateau(
            (5, 31), delta, sigma, valid_fraction, config
        )
        self.assertEqual(detected["status"], "ok")
        self.assertIsNotNone(detected["auto_stable_x_range"])
        self.assertIsNotNone(detected["analysis_x_range"])
        auto_left, auto_right = detected["auto_stable_x_range"]
        final_left, final_right = detected["analysis_x_range"]
        self.assertEqual((final_left, final_right), (auto_left + 2, auto_right - 2))
        self.assertGreaterEqual(final_right - final_left + 1, 12)

        steep = np.arange(48, dtype=np.float64) * 0.5
        review = GEOMETRY_EXPERIMENT._detect_stable_plateau(
            (5, 31), steep, sigma, valid_fraction, config
        )
        self.assertEqual(review["status"], "needs_manual_review")
        self.assertIsNone(review["analysis_x_range"])

    def test_roi_trim_sensitivity_is_diagnostic_only_and_uses_raw_delta(self):
        delta = np.full(40, np.nan, dtype=np.float64)
        delta[5:35] = np.linspace(-10.05, -9.95, 30)
        valid_fraction = np.ones(40, dtype=np.float64)
        config = {
            "roi_trim_values_px": (0, 2, 3, 4),
            "roi_trim_max_relative_change": 0.02,
            "multiheight_valid_fraction_min": 0.8,
        }
        rows, stable = GEOMETRY_EXPERIMENT._roi_trim_sensitivity_rows(
            "h010", 10.0, (5, 34), delta, valid_fraction, config
        )
        self.assertTrue(stable)
        self.assertEqual([row["trim_px"] for row in rows], [0, 2, 3, 4])
        self.assertTrue(all(row["roi_stable"] for row in rows))
        self.assertAlmostEqual(rows[0]["delta_y_median_px"], -10.0)
        self.assertAlmostEqual(rows[0]["sensitivity_median_px_per_mm"], 1.0)

    def test_roi_trim_sensitivity_reports_unavailable_without_raising(self):
        delta = np.full(40, np.nan, dtype=np.float64)
        valid_fraction = np.zeros(40, dtype=np.float64)
        config = {
            "roi_trim_values_px": (0, 2, 3, 4),
            "roi_trim_max_relative_change": 0.02,
            "multiheight_valid_fraction_min": 0.8,
        }

        rows, stable = GEOMETRY_EXPERIMENT._roi_trim_sensitivity_rows(
            "h001", 1.0, (5, 34), delta, valid_fraction, config
        )

        self.assertIsNone(stable)
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(not row["available"] for row in rows))
        self.assertTrue(all(row["unavailable_reason"] == "no_valid_delta_y" for row in rows))
        self.assertTrue(all(row["delta_y_median_px"] is None for row in rows))

    def test_geometry_summary_ignores_h1_failure_for_primary_status(self):
        matrix_row = {
            "config_id": "B05_A10",
            "baseline_scale_reading": "5",
            "laser_angle_deg": "10",
        }
        reference = {
            "statistics": {
                "reference_cv_interior_rmse_px": 0.1,
                "reference_cv_interior_p95_px": 0.2,
            }
        }
        h1 = {
            "formal_statistics_allowed": False,
            "warnings": ["no_object_columns", "no_common_columns"],
            "sensitivity_median_px_per_mm": None,
            "sigma_pixel_p95_px": None,
            "sigma_z_pred_p95_mm": None,
            "roi_trim_max_relative_change": None,
        }
        primary = {
            "formal_statistics_allowed": True,
            "warnings": ["stable_plateau_width_below_min"],
            "sensitivity_median_px_per_mm": 0.8,
            "sigma_pixel_p95_px": 0.02,
            "sigma_z_pred_p95_mm": 0.025,
            "roi_trim_max_relative_change": 0.01,
        }
        multiheight = {
            "rois": {"h001": h1, "h010": dict(primary), "h030": dict(primary)},
            "sensitivity_combined_px_per_mm": 0.8,
            "sigma_z_pred_combined_mm": 0.025,
            # A stale aggregate flag must not let an H1-only warning request review.
            "needs_manual_review": True,
        }

        row = GEOMETRY_EXPERIMENT._geometry_summary_row(
            matrix_row, reference, multiheight
        )
        self.assertEqual(row["status"], "warning")
        self.assertFalse(row["needs_manual_review"])
        self.assertEqual(row["sensitivity_h1"], "NaN")
        self.assertAlmostEqual(row["sensitivity_combined_px_per_mm"], 0.8)

        multiheight["rois"]["h010"]["formal_statistics_allowed"] = False
        failed = GEOMETRY_EXPERIMENT._geometry_summary_row(
            matrix_row, reference, multiheight
        )
        self.assertEqual(failed["status"], "failed")
        self.assertTrue(failed["needs_manual_review"])

    def test_steger_sweep_statistics_separates_per_frame_and_column_gates(self):
        valid = np.asarray([
            [True, True, True],
            [True, True, True],
            [True, True, False],
            [True, False, False],
        ])
        y = np.where(valid, 10.0 + np.arange(4)[:, None] * 0.01, np.nan)

        statistics, columns = GEOMETRY_EXPERIMENT._steger_sweep_roi_statistics(
            y, valid, (0, 2)
        )

        np.testing.assert_array_equal(columns["detected_count"], [4, 3, 2])
        np.testing.assert_allclose(columns["detected_fraction"], [1.0, 0.75, 0.5])
        self.assertAlmostEqual(statistics["per_frame_detection_rate_mean"], 0.75)
        self.assertAlmostEqual(statistics["valid_column_fraction_at_0_8"], 1 / 3)
        self.assertAlmostEqual(statistics["valid_column_fraction_at_0_6"], 2 / 3)
        self.assertAlmostEqual(statistics["valid_column_fraction_at_0_5"], 1.0)

    def test_steger_h1_failure_classifier_prefers_safe_per_frame_recovery(self):
        def parameter_rows(parameter_id, *, formal, h1_08, h1_05, h1_rate, background):
            common = {
                "parameter_id": parameter_id,
                "formal_parameters": formal,
                "threshold": 30.0 if formal else 20.0,
                "deriv_thresh": 0.5 if formal else 0.4,
            }
            return [
                {**common, "roi_id": "h001", "valid_column_fraction_at_0_8": h1_08,
                 "valid_column_fraction_at_0_5": h1_05,
                 "per_frame_detection_rate_mean": h1_rate,
                 "center_shift_vs_formal_px": None, "sigma_p95_ratio_vs_formal": None},
                {**common, "roi_id": "h010", "center_shift_vs_formal_px": 0.02,
                 "sigma_p95_ratio_vs_formal": 1.05},
                {**common, "roi_id": "h030", "center_shift_vs_formal_px": -0.03,
                 "sigma_p95_ratio_vs_formal": 1.10},
                {**common, "roi_id": "background", "per_frame_detection_rate_mean": background},
            ]

        rows = parameter_rows(
            "T30_D0.5", formal=True, h1_08=0.0, h1_05=0.0, h1_rate=0.0,
            background=0.0,
        ) + parameter_rows(
            "T20_D0.4", formal=False, h1_08=0.8, h1_05=1.0, h1_rate=0.9,
            background=0.01,
        )

        verdict = GEOMETRY_EXPERIMENT._classify_steger_h1_failure(rows)
        self.assertEqual(verdict["verdict"], "A. mainly_per_frame_steger_threshold")
        self.assertEqual(verdict["candidate_parameter_id"], "T20_D0.4")

    def test_realtime_steger_diagnostic_path_preserves_formal_output(self):
        realtime = GEOMETRY_EXPERIMENT._load_realtime_steger(
            SCRIPT_PATH.parents[1] / "calibration" / "src"
        )
        image = np.zeros((96, 128), dtype=np.uint8)
        image[46:49, 12:116] = 180
        options = {
            "sigma": 1.5,
            "threshold": 30.0,
            "deriv_thresh": 0.5,
            "roi_margin": 12,
            "roi_max_height": 64,
            "scan_axis": "column",
        }
        formal = realtime.extract_steger_columns(image, options)
        diagnostic = realtime.extract_steger_columns(image, options, diagnostic=True)

        self.assertIsNone(formal.diagnostics)
        self.assertIsNotNone(diagnostic.diagnostics)
        for field in (
            "u_px", "v_px", "valid", "response", "offset_px", "normal_y_abs",
            "corrected_signal",
        ):
            np.testing.assert_array_equal(
                getattr(formal, field), getattr(diagnostic, field)
            )
        self.assertEqual(formal.metadata, diagnostic.metadata)

    def test_realtime_steger_diagnostic_identifies_peak_outside_detected_band(self):
        realtime = GEOMETRY_EXPERIMENT._load_realtime_steger(
            SCRIPT_PATH.parents[1] / "calibration" / "src"
        )
        image = np.zeros((160, 64), dtype=np.uint8)
        image[48:51, 20:60] = 180
        image[118:121, 2:12] = 80
        options = {
            "sigma": 1.5,
            "threshold": 30.0,
            "deriv_thresh": 0.5,
            "roi_margin": 8,
            "roi_max_height": 64,
            "scan_axis": "column",
        }
        result = realtime.extract_steger_columns(image, options, diagnostic=True)
        diagnostics = result.diagnostics
        self.assertIsNotNone(diagnostics)
        self.assertFalse(result.valid[5])
        self.assertTrue(diagnostics.intensity_peak_outside_detected_band[5])
        self.assertEqual(
            diagnostics.rejection_reason[5],
            "intensity_peak_outside_detected_band",
        )
        expanded = realtime.extract_steger_columns(
            image,
            options,
            additional_band_bounds=(100, 135),
        )
        self.assertTrue(expanded.valid[5])
        self.assertEqual(expanded.metadata["original_band_top_px"], 40.0)
        self.assertEqual(expanded.metadata["raw_candidate_top"], 48)
        self.assertEqual(expanded.metadata["raw_candidate_bottom"], 51)
        self.assertEqual(expanded.metadata["margin_before_clip"], [40, 59])
        self.assertEqual(expanded.metadata["margin_after_clip"], [40, 59])
        self.assertFalse(expanded.metadata["roi_max_height_applied"])
        self.assertEqual(expanded.metadata["final_band_top"], 40)
        self.assertEqual(expanded.metadata["final_band_bottom"], 59)
        self.assertEqual(expanded.metadata["reference_envelope_top_px"], 100.0)
        self.assertEqual(expanded.metadata["reference_envelope_bottom_exclusive_px"], 135.0)
        self.assertEqual(expanded.metadata["final_band_top_px"], 40.0)
        self.assertEqual(expanded.metadata["final_band_bottom_exclusive_px"], 135.0)

    def test_reference_vertical_envelope_uses_only_surface_and_clips_to_image(self):
        y_ref = np.full(12, np.nan, dtype=np.float64)
        y_ref[2:10] = np.asarray([1.2, 2.0, 3.0, 4.0, 5.0, 6.0, 7.2, 8.1])
        self.assertEqual(
            GEOMETRY_EXPERIMENT._reference_vertical_envelope(
                y_ref, (2, 9), margin_px=3, image_height=10
            ),
            (0, 10),
        )
        y_ref[2:10] = np.linspace(10.1, 11.2, 8)
        self.assertEqual(
            GEOMETRY_EXPERIMENT._reference_vertical_envelope(
                y_ref, (2, 9), margin_px=3, image_height=30
            ),
            (7, 15),
        )

    def test_raw_profile_metrics_use_local_background_and_contiguous_half_max(self):
        profile = np.asarray([10, 10, 10, 10, 20, 60, 100, 60, 20, 10, 10, 10, 10])
        metrics = GEOMETRY_EXPERIMENT._local_raw_profile_metrics(profile)
        self.assertEqual(metrics["peak_intensity"], 100.0)
        self.assertEqual(metrics["local_background"], 10.0)
        self.assertEqual(metrics["peak_minus_background"], 90.0)
        self.assertEqual(metrics["fwhm_px"], 3.0)

    def test_root_cause_classifier_prefers_sigma_recovery_over_other_gates(self):
        profiles = [
            {"roi_id": "h001", "peak_intensity": 100.0, "peak_minus_background": 80.0},
            {"roi_id": "h010", "peak_intensity": 120.0, "peak_minus_background": 90.0},
            {"roi_id": "h030", "peak_intensity": 130.0, "peak_minus_background": 100.0},
        ]
        rejection = {"rois": {"h001": {"rejection_reason_counts": {
            "accepted": 0, "ridge_response_below_threshold": 100,
        }}}}
        rows = []
        for sigma, recovery in ((1.5, 0.0), (1.0, 0.8)):
            for roi_id in ("h001", "h010", "h030"):
                rows.append({
                    "sigma": sigma,
                    "roi_id": roi_id,
                    "valid_column_fraction_at_0_8": recovery if roi_id == "h001" else 1.0,
                    "center_shift_vs_formal_px": None if roi_id == "h001" else 0.02,
                    "sigma_p95_ratio_vs_formal": None if roi_id == "h001" else 1.05,
                })
            rows.append({
                "sigma": sigma,
                "roi_id": "background",
                "per_frame_detection_rate_mean": 0.0,
            })
        verdict = GEOMETRY_EXPERIMENT._classify_steger_root_cause(
            profiles, rejection, rows
        )
        self.assertEqual(verdict["verdict"], "B. steger_scale_mismatch")
        self.assertEqual(verdict["candidate_sigma"], 1.0)

    def test_root_cause_classifier_reports_detected_band_gate(self):
        profiles = [
            {"roi_id": "h001", "peak_intensity": 41.0, "peak_minus_background": 39.0},
            {"roi_id": "h010", "peak_intensity": 52.0, "peak_minus_background": 47.0},
            {"roi_id": "h030", "peak_intensity": 47.0, "peak_minus_background": 41.0},
        ]
        rejection = {"rois": {"h001": {"rejection_reason_counts": {
            "accepted": 0, "intensity_peak_outside_detected_band": 1450,
        }}}}
        rows = []
        for sigma in (0.8, 1.5, 2.5):
            for roi_id in ("h001", "h010", "h030"):
                rows.append({
                    "sigma": sigma,
                    "roi_id": roi_id,
                    "valid_column_fraction_at_0_8": 0.0 if roi_id == "h001" else 1.0,
                    "center_shift_vs_formal_px": None if roi_id == "h001" else 0.0,
                    "sigma_p95_ratio_vs_formal": None if roi_id == "h001" else 1.0,
                })
            rows.append({
                "sigma": sigma,
                "roi_id": "background",
                "per_frame_detection_rate_mean": 0.0,
            })
        verdict = GEOMETRY_EXPERIMENT._classify_steger_root_cause(
            profiles, rejection, rows
        )
        self.assertEqual(verdict["verdict"], "C. other_steger_rejection_rule")
        self.assertEqual(
            verdict["h1_dominant_rejection_reason"],
            "intensity_peak_outside_detected_band",
        )

    def test_repeatability_and_combined_metrics_use_h10_h30_only(self):
        repeat_valid, statistics = GEOMETRY_EXPERIMENT._repeatability_statistics(
            np.asarray([0.01, 0.02, 0.03]),
            np.asarray([1.0, 2.0, 3.0]),
            np.asarray([True, True, True]),
        )
        self.assertTrue(np.all(repeat_valid))
        self.assertAlmostEqual(statistics["sigma_pixel_p50_px"], 0.02)
        self.assertAlmostEqual(statistics["sigma_z_pred_p50_mm"], 0.01)
        self.assertAlmostEqual(statistics["sigma_z_pred_p95_mm"], 0.01)

        rois = {
            "h001": {
                "formal_statistics_allowed": True,
                "sensitivity_median_px_per_mm": 99.0,
                "sigma_z_pred_p95_mm": 99.0,
            },
            "h010": {
                "formal_statistics_allowed": True,
                "sensitivity_median_px_per_mm": 0.8,
                "sigma_z_pred_p95_mm": 0.02,
            },
            "h030": {
                "formal_statistics_allowed": True,
                "sensitivity_median_px_per_mm": 1.0,
                "sigma_z_pred_p95_mm": 0.04,
            },
        }
        sensitivity, sigma_z = GEOMETRY_EXPERIMENT._phase_a_combined_metrics(rois)
        self.assertAlmostEqual(sensitivity, 0.9)
        self.assertAlmostEqual(sigma_z, 0.03)
        rois["h030"]["formal_statistics_allowed"] = False
        self.assertEqual(
            GEOMETRY_EXPERIMENT._phase_a_combined_metrics(rois),
            (None, None),
        )

    def test_roi_registry_matrix_initialization_preserves_verified_template(self):
        with tempfile.TemporaryDirectory() as temporary:
            registry = Path(temporary) / "roi_registry.yaml"
            template = {
                "reference_surface": {"x_range": [935, 2236], "method": "manual_confirmed"},
                "multiheight": {
                    "h001": {"height_mm": 1.0, "selected_x_range": [1711, 1744]},
                    "h010": {"height_mm": 10.0, "selected_x_range": [1771, 1807]},
                    "h030": {"height_mm": 30.0, "selected_x_range": [1836, 1862]},
                },
            }
            GEOMETRY_EXPERIMENT._write_roi_registry(registry, {
                "schema_version": 1,
                "experiment_type": "geometry_baseline_angle",
                "template_config": "B12p5_A20",
                "configs": {"B12p5_A20": template},
            })
            _path, document = GEOMETRY_EXPERIMENT.ensure_roi_registry(registry)
            self.assertEqual(len(document["configs"]), 12)
            preserved = document["configs"]["B12p5_A20"]
            self.assertEqual(preserved["reference_surface"]["x_range"], [935, 2236])
            self.assertEqual(
                preserved["multiheight"]["h010"]["selected_x_range"], [1771, 1807]
            )
            self.assertEqual(preserved["status"], "confirmed")
            self.assertTrue(preserved["manual_confirmed"])
            self.assertEqual(document["configs"]["B00_A05"]["status"], "invalid_fov")
            self.assertEqual(document["configs"]["B00_A10"]["status"], "pending")

    def test_roi_audit_checks_reference_containment_overlap_and_trim3_width(self):
        entry = {
            "reference_surface": {"x_range": [10, 100]},
            "multiheight": {
                "h001": {"height_mm": 1.0, "selected_x_range": [20, 40]},
                "h010": {"height_mm": 10.0, "selected_x_range": [45, 65]},
                "h030": {"height_mm": 30.0, "selected_x_range": [70, 90]},
            },
        }
        self.assertEqual(
            GEOMETRY_EXPERIMENT._roi_entry_issues(entry, 120, 3, 12), []
        )
        entry["multiheight"]["h030"]["selected_x_range"] = [60, 105]
        issues = GEOMETRY_EXPERIMENT._roi_entry_issues(entry, 120, 3, 12)
        self.assertIn("h030_outside_reference", issues)
        self.assertIn("roi_overlap:h010:h030", issues)


if __name__ == "__main__":
    unittest.main()
