import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from calibration_tool.calibration_results import (
    DETAILS_LOADED,
    DETAILS_UNAVAILABLE,
    load_laser_surface_details,
    summarize_calibration_run,
)
from calibration_tool.calibration_run import CalibrationRun, CalibrationStage
from calibration_tool.calibration_run_io import load_calibration_run
from calibration_tool.io_utils import dump_yaml


class LaserSurfaceResultsTests(unittest.TestCase):
    @staticmethod
    def _run(
        root: Path,
        *,
        result_file: Path,
        metrics: dict | None = None,
    ) -> CalibrationRun:
        return CalibrationRun(
            run_id="run-001",
            project_id="demo",
            workflow_path=root / "workflow.yaml",
            started_utc=datetime(2026, 8, 28, 3, 24, tzinfo=timezone.utc),
            completed_utc=datetime(2026, 8, 28, 3, 26, tzinfo=timezone.utc),
            status="completed",
            laser_orientation="horizontal",
            stages=[
                CalibrationStage(
                    stage="laser_surface_models",
                    status="completed",
                    output_dir=result_file.parent,
                    result_file=result_file,
                    metrics=metrics
                    or {
                        "model_type": "circular_cone",
                        "validation_rmse_mm": 0.14,
                        "validation_p95_mm": 0.25,
                        "validation_valid_rate": 0.98,
                    },
                )
            ],
            overall="pass",
        )

    @staticmethod
    def _write_result(path: Path) -> None:
        path.write_text(
            "model_type: circular_cone\n"
            "model_selection:\n"
            "  default_model: circular_cone\n"
            "metrics:\n"
            "  train:\n"
            "    board_rmse_mm: 0.12\n"
            "    board_p95_abs_mm: 0.22\n"
            "    board_max_abs_mm: 0.50\n"
            "    valid_rate: 0.99\n"
            "  validation:\n"
            "    board_rmse_mm: 0.14\n"
            "    board_p95_abs_mm: 0.25\n"
            "    board_max_abs_mm: 0.60\n"
            "    valid_rate: 0.98\n"
            "axis_unit_camera: [0.0, 1.0, 0.0]\n"
            "apex_camera_mm: [1.0, 2.0, 3.0]\n"
            "half_apex_angle_deg: 20.0\n",
            encoding="utf-8",
        )

    @staticmethod
    def _write_models(root: Path, *, include_quadratic: bool = True) -> None:
        models = root / "models"
        models.mkdir(parents=True, exist_ok=True)
        (models / "global_plane.yaml").write_text(
            "model_type: global_plane\nplane_abcd: [1.0, 2.0, 3.0, 4.0]\n",
            encoding="utf-8",
        )
        (models / "circular_cone.yaml").write_text(
            "model_type: circular_cone\n"
            "axis_unit_camera: [0.0, 1.0, 0.0]\n"
            "apex_camera_mm: [1.0, 2.0, 3.0]\n"
            "half_apex_angle_deg: 20.0\n",
            encoding="utf-8",
        )
        if include_quadratic:
            (models / "quadratic_graph.yaml").write_text(
                "model_type: quadratic_graph\n"
                "coefficients: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]\n",
                encoding="utf-8",
            )

    @staticmethod
    def _write_comparison(root: Path, *, models: tuple[str, ...]) -> None:
        path = root / "model_comparison.csv"
        lines = [
            "model,split,board_rmse_mm,board_p95_abs_mm,board_max_abs_mm,valid_rate"
        ]
        values = {
            "global_plane": ("0.30", "0.40", "0.80", "0.90"),
            "quadratic_graph": ("0.20", "0.30", "0.70", "0.95"),
            "circular_cone": ("0.12", "0.25", "0.60", "0.98"),
        }
        for model in models:
            rmse, p95, max_value, rate = values[model]
            lines.append(f"{model},train,{rmse},{p95},{max_value},{rate}")
            lines.append(f"{model},validation,{rmse},{p95},{max_value},{rate}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_complete_run_reads_three_models_metrics_parameters_and_plots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "laser_model"
            root.mkdir(parents=True)
            result_file = root / "laser_model.yaml"
            self._write_result(result_file)
            self._write_models(root)
            self._write_comparison(
                root,
                models=("global_plane", "quadratic_graph", "circular_cone"),
            )
            for name in (
                "validation_error_vs_u.png",
                "validation_error_vs_v.png",
                "validation_error_vs_depth.png",
            ):
                (root / name).write_bytes(b"existing plot")

            details = load_laser_surface_details(
                summarize_calibration_run(self._run(root, result_file=result_file)).laser_surface
            )

            self.assertEqual(details.status, DETAILS_LOADED)
            self.assertEqual(details.selected_model, "circular_cone")
            self.assertEqual(len(details.model_comparisons), 3)
            self.assertAlmostEqual(details.model_comparisons[0].validation_rmse_mm, 0.30)
            self.assertAlmostEqual(details.model_comparisons[1].train_rmse_mm, 0.20)
            self.assertAlmostEqual(details.validation_max_mm, 0.60)
            self.assertEqual(details.selected_parameters.apex_camera_mm, (1.0, 2.0, 3.0))
            self.assertEqual(
                details.model_comparisons[0].parameters.plane_abcd,
                (1.0, 2.0, 3.0, 4.0),
            )
            self.assertEqual(
                details.model_comparisons[1].parameters.coefficients,
                (1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
            )
            self.assertIsNotNone(details.error_vs_u)
            self.assertIsNotNone(details.error_vs_v)
            self.assertIsNotNone(details.error_vs_depth)

    def test_missing_comparison_csv_keeps_selected_result_available(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "laser_model"
            root.mkdir(parents=True)
            result_file = root / "laser_model.yaml"
            self._write_result(result_file)

            details = load_laser_surface_details(
                summarize_calibration_run(self._run(root, result_file=result_file)).laser_surface
            )

            self.assertEqual(details.status, DETAILS_LOADED)
            self.assertIn("model_comparison.csv", " ".join(details.notes))
            selected = next(item for item in details.models if item.model == "circular_cone")
            self.assertTrue(selected.available)
            self.assertAlmostEqual(selected.validation_rmse_mm, 0.14)
            self.assertFalse(
                next(item for item in details.models if item.model == "global_plane").available
            )

    def test_partial_model_comparison_is_kept_without_failing_the_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "laser_model"
            root.mkdir(parents=True)
            result_file = root / "laser_model.yaml"
            self._write_result(result_file)
            self._write_comparison(root, models=("global_plane", "circular_cone"))

            details = load_laser_surface_details(
                summarize_calibration_run(self._run(root, result_file=result_file)).laser_surface
            )

            quadratic = next(item for item in details.models if item.model == "quadratic_graph")
            self.assertFalse(quadratic.available)
            self.assertIsNone(quadratic.validation_rmse_mm)
            self.assertTrue(
                next(item for item in details.models if item.model == "global_plane").available
            )

    def test_surface_metric_column_names_are_supported_as_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "laser_model"
            root.mkdir(parents=True)
            result_file = root / "laser_model.yaml"
            self._write_result(result_file)
            (root / "model_comparison.csv").write_text(
                "model,split,surface_rmse_mm,surface_p95_abs_mm,surface_max_abs_mm,valid_rate\n"
                "circular_cone,train,0.10,0.20,0.40,0.90\n"
                "circular_cone,validation,0.11,0.21,0.41,0.91\n",
                encoding="utf-8",
            )

            details = load_laser_surface_details(
                summarize_calibration_run(self._run(root, result_file=result_file)).laser_surface
            )
            selected = next(item for item in details.models if item.model == "circular_cone")

            self.assertAlmostEqual(selected.validation_rmse_mm, 0.11)
            self.assertAlmostEqual(selected.validation_p95_mm, 0.21)
            self.assertAlmostEqual(selected.validation_max_mm, 0.41)

    def test_corrupt_laser_result_is_reported_without_raising(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "laser_model"
            root.mkdir(parents=True)
            result_file = root / "laser_model.yaml"
            result_file.write_text("model_type: [", encoding="utf-8")

            details = load_laser_surface_details(
                summarize_calibration_run(self._run(root, result_file=result_file)).laser_surface
            )

            self.assertEqual(details.status, DETAILS_UNAVAILABLE)
            self.assertIn("读取失败", details.error or "")
            self.assertEqual(len(details.model_comparisons), 3)

    def test_legacy_workflow_report_reads_existing_laser_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            laser_root = root / "laser_model"
            laser_root.mkdir(parents=True)
            result_file = laser_root / "laser_model.yaml"
            self._write_result(result_file)
            self._write_models(laser_root)
            self._write_comparison(
                laser_root,
                models=("global_plane", "quadratic_graph", "circular_cone"),
            )
            report = root / "projects" / "haikang" / "reports" / "workflow_20260828.yaml"
            dump_yaml(
                report,
                {
                    "workflow": "../plans/workflow_haikang.yaml",
                    "status": "completed",
                    "overall": "pass",
                    "stages": [
                        {
                            "stage": "laser_surface_models",
                            "status": "completed",
                            "output_dir": str(laser_root),
                            "result_file": str(result_file),
                            "metrics": {
                                "model_type": "circular_cone",
                                "validation_p95_mm": 0.25,
                            },
                        }
                    ],
                },
            )

            run = load_calibration_run(report)
            details = load_laser_surface_details(summarize_calibration_run(run).laser_surface)

            self.assertEqual(details.selected_model, "circular_cone")
            self.assertEqual(len(details.models), 3)
            self.assertTrue(all(item.available for item in details.models))
            self.assertAlmostEqual(details.validation_p95_mm, 0.25)


if __name__ == "__main__":
    unittest.main()
