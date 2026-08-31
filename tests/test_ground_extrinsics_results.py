import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from calibration_tool.calibration_results import (
    DETAILS_LOADED,
    DETAILS_UNAVAILABLE,
    NOT_EXECUTED,
    load_ground_extrinsics_details,
    summarize_calibration_run,
)
from calibration_tool.calibration_run import CalibrationRun, CalibrationStage
from calibration_tool.calibration_run_io import load_calibration_run
from calibration_tool.io_utils import dump_yaml


class GroundExtrinsicsResultsTests(unittest.TestCase):
    @staticmethod
    def _run(stage: CalibrationStage) -> CalibrationRun:
        return CalibrationRun(
            run_id="run-001",
            project_id="demo",
            workflow_path=Path("workflow.yaml"),
            started_utc=datetime(2026, 8, 28, 3, 24, tzinfo=timezone.utc),
            completed_utc=datetime(2026, 8, 28, 3, 26, tzinfo=timezone.utc),
            status="completed",
            stages=[stage],
            overall="pass",
        )

    @staticmethod
    def _write_result(root: Path, *, with_validation: bool = True) -> Path:
        validation = ""
        if with_validation:
            validation = (
                "validation:\n"
                "  detected_frame_count: 1\n"
                "  metrics: {rmse_mm: 0.06, p95_abs_mm: 0.12, max_abs_mm: 0.15}\n"
                "  frames:\n"
                "  - image: chess 009.tif\n"
                "    detection_method: SB\n"
                "    pnp_rmse_px: 0.07\n"
                "    zg_rmse_mm: 0.045\n"
                "    zg_p95_abs_mm: 0.08\n"
                "    zg_max_abs_mm: 0.10\n"
                "    zg_std_mm: 0.03\n"
                "    zg_point_count: 88\n"
            )
        result = root / "camera_ground_extrinsics.yaml"
        result.write_text(
            "schema_version: 2\n"
            "method: checkerboard_plane_only\n"
            "units: mm\n"
            "T_ground_from_camera:\n"
            "- [1.0, 0.0, 0.0, 10.0]\n"
            "- [0.0, 1.0, 0.0, 20.0]\n"
            "- [0.0, 0.0, 1.0, 30.0]\n"
            "- [0.0, 0.0, 0.0, 1.0]\n"
            "quality_checks:\n"
            "  fit_zg: {rmse_mm: 0.07}\n"
            "  validation_zg: {rmse_mm: 0.06, p95_abs_mm: 0.12, max_abs_mm: 0.15}\n"
            + validation,
            encoding="utf-8",
        )
        return result

    def test_legacy_report_reads_board_only_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = root / "projects" / "haikang" / "reports" / "workflow_20260828.yaml"
            dump_yaml(
                report,
                {
                    "workflow": "../plans/workflow_haikang.yaml",
                    "status": "completed",
                    "overall": "pass",
                    "stages": [
                        {
                            "stage": "ground_extrinsics_board_only",
                            "status": "completed",
                            "metrics": {
                                "validation_rmse_mm": 0.0656368254396831,
                                "validation_p95_mm": 0.1228353097992283,
                            },
                        }
                    ],
                },
            )

            summary = summarize_calibration_run(load_calibration_run(report))

            self.assertEqual(summary.ground_extrinsics.stage, "ground_extrinsics_board_only")
            self.assertAlmostEqual(
                summary.ground_extrinsics.validation_rmse_mm,
                0.0656368254396831,
            )
            self.assertAlmostEqual(
                summary.ground_extrinsics.validation_p95_mm,
                0.1228353097992283,
            )

    def test_shared_steger_stage_is_read_without_mixing_board_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self._write_result(root)
            stage = CalibrationStage(
                stage="ground_extrinsics_shared_steger",
                status="completed",
                output_dir=root,
                result_file=result,
                metrics={"validation_rmse_mm": 0.06, "validation_p95_mm": 0.12},
            )

            details = load_ground_extrinsics_details(
                summarize_calibration_run(self._run(stage)).ground_extrinsics
            )

            self.assertEqual(details.stage, "ground_extrinsics_shared_steger")
            self.assertEqual(details.status, DETAILS_LOADED)
            self.assertEqual(details.validation_frame_count, 1)
            self.assertEqual(details.rotation_matrix[1][1], 1.0)

    def test_missing_stage_has_no_pose_or_validation(self) -> None:
        stage = CalibrationStage(stage="intrinsics", status="completed")
        summary = summarize_calibration_run(self._run(stage))

        details = load_ground_extrinsics_details(summary.ground_extrinsics)

        self.assertEqual(summary.ground_extrinsics.status, NOT_EXECUTED)
        self.assertEqual(details.status, NOT_EXECUTED)
        self.assertEqual(details.rotation_matrix, ())
        self.assertEqual(details.validation_rows, ())

    def test_missing_or_corrupt_result_is_unavailable_without_raising(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing = CalibrationStage(
                stage="ground_extrinsics_board_only",
                status="completed",
                output_dir=root,
                result_file=root / "missing.yaml",
            )
            missing_details = load_ground_extrinsics_details(
                summarize_calibration_run(self._run(missing)).ground_extrinsics
            )
            self.assertEqual(missing_details.status, DETAILS_UNAVAILABLE)
            self.assertIn("不存在", missing_details.error or "")

            corrupt = root / "corrupt.yaml"
            corrupt.write_text("T_ground_from_camera: [", encoding="utf-8")
            corrupt_stage = CalibrationStage(
                stage="ground_extrinsics_board_only",
                status="completed",
                output_dir=root,
                result_file=corrupt,
            )
            corrupt_details = load_ground_extrinsics_details(
                summarize_calibration_run(self._run(corrupt_stage)).ground_extrinsics
            )
            self.assertEqual(corrupt_details.status, DETAILS_UNAVAILABLE)
            self.assertIn("读取失败", corrupt_details.error or "")

    def test_existing_validation_csv_is_mapped_and_no_validation_degrades(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self._write_result(root, with_validation=False)
            (root / "fit_frames.csv").write_text(
                "image,zg_rmse_mm\nfit.tif,0.07\n", encoding="utf-8"
            )
            (root / "validation_frames.csv").write_text(
                "image,pnp_rmse_px,zg_rmse_mm,zg_p95_abs_mm,zg_max_abs_mm,zg_std_mm,zg_point_count\n"
                "valid.tif,0.08,0.05,0.09,0.11,0.03,88\n",
                encoding="utf-8",
            )
            stage = CalibrationStage(
                stage="ground_extrinsics_board_only",
                status="completed",
                output_dir=root,
                result_file=result,
            )
            details = load_ground_extrinsics_details(
                summarize_calibration_run(self._run(stage)).ground_extrinsics
            )
            self.assertEqual(details.validation_frame_count, 1)
            self.assertEqual(details.validation_rows[0].frame, "valid.tif")
            self.assertAlmostEqual(details.validation_rows[0].rmse_mm, 0.05)

            no_validation = self._write_result(root, with_validation=False)
            for filename in ("validation_frames.csv",):
                (root / filename).unlink()
            no_validation_details = load_ground_extrinsics_details(
                summarize_calibration_run(
                    self._run(
                        CalibrationStage(
                            stage="ground_extrinsics_board_only",
                            status="completed",
                            output_dir=root,
                            result_file=no_validation,
                        )
                    )
                ).ground_extrinsics
            )
            self.assertEqual(no_validation_details.status, DETAILS_LOADED)
            self.assertIsNone(no_validation_details.validation_frame_count)
            self.assertEqual(no_validation_details.validation_rows, ())
            self.assertIn("validation", " ".join(no_validation_details.notes).lower())


if __name__ == "__main__":
    unittest.main()
