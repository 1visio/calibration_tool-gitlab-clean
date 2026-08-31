import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from calibration_tool.calibration_results import (
    DETAILS_LOADED,
    DETAILS_UNAVAILABLE,
    NOT_EXECUTED,
    load_intrinsics_details,
    summarize_calibration_run,
)
from calibration_tool.calibration_run import CalibrationRun, CalibrationStage
from calibration_tool.calibration_run_io import load_calibration_run
from calibration_tool.io_utils import dump_yaml


class CalibrationResultsTests(unittest.TestCase):
    @staticmethod
    def _run(stages: list[CalibrationStage]) -> CalibrationRun:
        return CalibrationRun(
            run_id="run-001",
            project_id="demo",
            workflow_path=Path("workflow.yaml"),
            started_utc=datetime(2026, 8, 28, 3, 24, tzinfo=timezone.utc),
            completed_utc=datetime(2026, 8, 28, 3, 26, tzinfo=timezone.utc),
            status="completed",
            laser_orientation="horizontal",
            stages=stages,
            overall="pass",
        )

    def test_complete_run_is_mapped_to_structured_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            intrinsics_dir = root / "intrinsics"
            laser_dir = root / "laser_model"
            ground_dir = root / "ground_extrinsics"
            bias_dir = root / "ground_bias"
            run = self._run(
                [
                    CalibrationStage(
                        stage="intrinsics",
                        status="completed",
                        output_dir=intrinsics_dir,
                        result_file=intrinsics_dir / "calibration_result.yaml",
                        metrics={
                            "fit_rmse_px": 0.12,
                            "test_rmse_px": 0.14,
                            "fit_image_count": 16,
                            "test_image_count": 6,
                        },
                    ),
                    CalibrationStage(
                        stage="laser_surface_models",
                        status="completed",
                        output_dir=laser_dir,
                        result_file=laser_dir / "laser_model.yaml",
                        metrics={
                            "model_type": "circular_cone",
                            "validation_rmse_mm": 0.13,
                            "validation_p95_mm": 0.25,
                            "validation_valid_rate": 1.0,
                        },
                    ),
                    CalibrationStage(
                        stage="ground_extrinsics_shared_steger",
                        status="completed",
                        output_dir=ground_dir,
                        result_file=ground_dir / "ground.yaml",
                        metrics={
                            "validation_rmse_mm": 0.06,
                            "validation_p95_mm": 0.12,
                        },
                    ),
                    CalibrationStage(
                        stage="ground_bias",
                        status="completed",
                        output_dir=bias_dir,
                        metrics={
                            "loaded_frame_count": 12,
                            "independent_validation_frame_count": 4,
                        },
                    ),
                ]
            )

            summary = summarize_calibration_run(run)

            self.assertEqual(summary.run_id, "run-001")
            self.assertEqual(summary.overall, "pass")
            self.assertEqual(summary.intrinsics.fit_image_count, 16)
            self.assertEqual(summary.intrinsics.test_image_count, 6)
            self.assertEqual(summary.intrinsics.fit_rmse_px, 0.12)
            self.assertEqual(summary.laser_surface.model_type, "circular_cone")
            self.assertEqual(summary.laser_surface.validation_valid_rate, 1.0)
            self.assertEqual(summary.ground_extrinsics.validation_p95_mm, 0.12)
            self.assertEqual(summary.ground_bias.loaded_frame_count, 12)
            self.assertEqual(summary.laser_surface.output_dir, laser_dir)
            self.assertEqual(
                summary.intrinsics.result_file,
                intrinsics_dir / "calibration_result.yaml",
            )

    def test_missing_stages_are_reported_as_not_executed(self) -> None:
        run = self._run(
            [
                CalibrationStage(
                    stage="intrinsics",
                    status="completed",
                    metrics={"fit_rmse_px": 0.12},
                )
            ]
        )

        summary = summarize_calibration_run(run)

        self.assertEqual(summary.intrinsics.status, "completed")
        self.assertEqual(summary.intrinsics.fit_rmse_px, 0.12)
        self.assertEqual(summary.laser_surface.status, NOT_EXECUTED)
        self.assertEqual(summary.ground_extrinsics.status, NOT_EXECUTED)
        self.assertEqual(summary.ground_bias.status, NOT_EXECUTED)
        self.assertIsNone(summary.laser_surface.output_dir)

    def test_legacy_workflow_report_is_supported_by_the_same_adapter(self) -> None:
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
                            "stage": "intrinsics",
                            "status": "completed",
                            "metrics": {
                                "fit_image_count": 16,
                                "test_rmse_px": 0.10029572248458862,
                            },
                        },
                        {
                            "stage": "laser_surface_models",
                            "status": "completed",
                            "metrics": {
                                "model_type": "circular_cone",
                                "validation_p95_mm": 0.25176227466754425,
                            },
                        },
                        {
                            "stage": "ground_extrinsics_board_only",
                            "status": "completed",
                            "metrics": {"validation_rmse_mm": 0.0656368254396831},
                        },
                    ],
                },
            )

            summary = summarize_calibration_run(load_calibration_run(report))

            self.assertEqual(summary.run_id, "workflow_20260828")
            self.assertEqual(summary.project_id, "haikang")
            self.assertEqual(summary.intrinsics.fit_image_count, 16)
            self.assertAlmostEqual(summary.intrinsics.test_rmse_px, 0.10029572248458862)
            self.assertEqual(summary.laser_surface.model_type, "circular_cone")
            self.assertAlmostEqual(summary.laser_surface.validation_p95_mm, 0.25176227466754425)
            self.assertAlmostEqual(summary.ground_extrinsics.validation_rmse_mm, 0.0656368254396831)
            self.assertEqual(summary.ground_bias.status, NOT_EXECUTED)

    def test_intrinsics_details_reads_existing_result_and_per_image_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result_file = root / "calibration_result.yaml"
            result_file.write_text(
                "camera_matrix:\n"
                "- [100.0, 0.0, 50.0]\n"
                "- [0.0, 101.0, 60.0]\n"
                "- [0.0, 0.0, 1.0]\n"
                "dist_coeffs: [-0.1, 0.02, 0.003, -0.004, 0.0]\n"
                "fit_metrics: {image_count: 1, overall_reprojection_rmse: 0.1}\n"
                "test_metrics: {image_count: 1, overall_reprojection_rmse: 0.2}\n",
                encoding="utf-8",
            )
            (root / "fit_images.csv").write_text(
                "image,status,per_image_rmse,mean_euclidean_reprojection_error,error_stage\n"
                "fit.tif,used,0.12,0.08,final_fit\n",
                encoding="utf-8",
            )
            (root / "test_images.csv").write_text(
                "image,status,per_image_rmse,mean_euclidean_reprojection_error,error_stage\n"
                "test.tif,evaluated,0.15,0.09,frozen_intrinsics\n",
                encoding="utf-8",
            )
            run = self._run(
                [
                    CalibrationStage(
                        stage="intrinsics",
                        status="completed",
                        result_file=result_file,
                        metrics={},
                    )
                ]
            )

            summary = summarize_calibration_run(run)
            details = load_intrinsics_details(summary.intrinsics)

            self.assertEqual(details.status, DETAILS_LOADED)
            self.assertEqual(details.camera_matrix[0][0], 100.0)
            self.assertEqual(details.camera_matrix[1][2], 60.0)
            self.assertEqual(details.dist_coeffs[3], -0.004)
            self.assertEqual(details.fit_rmse_px, 0.1)
            self.assertEqual(details.test_image_count, 1)
            self.assertEqual([row.split for row in details.reprojection_rows], ["fit", "test"])
            self.assertEqual(details.reprojection_rows[0].rmse_px, 0.12)
            self.assertEqual(details.reprojection_rows[1].error_stage, "frozen_intrinsics")

    def test_intrinsics_details_reports_missing_result_file_without_raising(self) -> None:
        run = self._run(
            [
                CalibrationStage(
                    stage="intrinsics",
                    status="completed",
                    result_file=Path("missing-calibration-result.yaml"),
                    metrics={"fit_rmse_px": 0.1},
                )
            ]
        )

        details = load_intrinsics_details(summarize_calibration_run(run).intrinsics)

        self.assertEqual(details.status, DETAILS_UNAVAILABLE)
        self.assertIn("不存在", details.error or "")


if __name__ == "__main__":
    unittest.main()
