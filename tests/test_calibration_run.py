import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from calibration_tool import CalibrationRun as ExportedCalibrationRun
from calibration_tool import CalibrationStage as ExportedCalibrationStage
from calibration_tool import infer_calibration_run_root as exported_infer_run_root
from calibration_tool import load_calibration_run as exported_load_calibration_run
from calibration_tool.calibration_run import CalibrationRun, CalibrationStage
from calibration_tool.calibration_run_io import (
    infer_calibration_run_root,
    load_calibration_run,
    save_calibration_run,
)
from calibration_tool.io_utils import dump_yaml


class CalibrationRunTests(unittest.TestCase):
    def test_public_package_exports(self) -> None:
        self.assertIs(ExportedCalibrationRun, CalibrationRun)
        self.assertIs(ExportedCalibrationStage, CalibrationStage)
        self.assertIs(exported_infer_run_root, infer_calibration_run_root)
        self.assertIs(exported_load_calibration_run, load_calibration_run)

    def test_infers_common_run_root_from_completed_stage_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_root = root / "runs" / "20260828"
            report = {
                "stages": [
                    {"stage": "intrinsics", "status": "completed", "output_dir": str(run_root / "intrinsics")},
                    {"stage": "laser", "status": "completed", "output_dir": str(run_root / "laser_model")},
                    {"stage": "ground", "status": "completed", "output_dir": str(run_root / "ground_extrinsics")},
                    {"stage": "ignored", "status": "failed", "output_dir": str(root / "runs" / "other")},
                ],
            }

            self.assertEqual(infer_calibration_run_root(report), run_root)

    def test_does_not_guess_root_from_insufficient_or_mixed_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = {
                "stages": [
                    {"stage": "intrinsics", "status": "completed", "output_dir": str(root / "runs" / "one" / "intrinsics")},
                ],
            }
            self.assertIsNone(infer_calibration_run_root(report))

            report["stages"].append({
                "stage": "laser",
                "status": "completed",
                "output_dir": str(root / "runs" / "two" / "laser_model"),
            })
            self.assertIsNone(infer_calibration_run_root(report))

    def test_round_trip_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = CalibrationRun(
                run_id="run-001",
                project_id="haikang",
                workflow_path=root / "workflow.yaml",
                started_utc=datetime(2026, 8, 28, 3, 24, tzinfo=timezone.utc),
                completed_utc=datetime(2026, 8, 28, 3, 26, tzinfo=timezone.utc),
                status="completed",
                laser_orientation="horizontal",
                stages=[
                    CalibrationStage(
                        stage="intrinsics",
                        status="completed",
                        output_dir=root / "runs" / "intrinsics",
                        result_file=root / "runs" / "intrinsics" / "calibration_result.yaml",
                        metrics={"test_rmse_px": 0.1},
                        quality_gates=[
                            {"id": "intrinsics.test_rmse", "status": "pass"}
                        ],
                        module="calibrate_chessboard_opencv_reusable",
                    )
                ],
                gates=[{"id": "intrinsics.test_rmse", "status": "pass"}],
                counts={"pass": 1, "warn": 0, "fail": 0},
                overall="pass",
                extra={"operator": "test"},
            )

            path = save_calibration_run(run, root / "calibration_run.yaml")
            loaded = load_calibration_run(path)

            self.assertEqual(loaded, run)
            self.assertEqual(loaded.to_dict()["run_id"], "run-001")
            self.assertEqual(loaded.stages[0].result_file, run.stages[0].result_file)

    def test_loads_existing_haikang_workflow_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = root / "projects" / "haikang" / "reports" / "workflow_20260828.yaml"
            dump_yaml(
                report,
                {
                    "workflow": "../plans/workflow_haikang.yaml",
                    "status": "completed",
                    "overall": "pass",
                    "laser": {"orientation": "horizontal"},
                    "stages": [
                        {
                            "stage": "intrinsics",
                            "status": "completed",
                            "metrics": {"fit_image_count": 16},
                        },
                        {
                            "stage": "laser_surface_models",
                            "status": "completed",
                            "metrics": {"model_type": "circular_cone"},
                        },
                        {
                            "stage": "ground_extrinsics_board_only",
                            "status": "completed",
                            "metrics": {"validation_rmse_mm": 0.06},
                        },
                    ],
                    "counts": {"pass": 7, "warn": 0, "fail": 0},
                },
            )

            run = load_calibration_run(report)

            self.assertEqual(run.run_id, "workflow_20260828")
            self.assertEqual(run.project_id, "haikang")
            self.assertEqual(run.workflow_path.name, "workflow_haikang.yaml")
            self.assertEqual(run.status, "completed")
            self.assertEqual(run.overall, "pass")
            self.assertEqual(run.laser_orientation, "horizontal")
            self.assertEqual([stage.stage for stage in run.stages], [
                "intrinsics",
                "laser_surface_models",
                "ground_extrinsics_board_only",
            ])
            self.assertEqual(run.stages[0].metrics["fit_image_count"], 16)
            self.assertEqual(run.stages[1].metrics["model_type"], "circular_cone")
            self.assertEqual(run.counts, {"pass": 7, "warn": 0, "fail": 0})

    def test_from_workflow_report_accepts_missing_run_identity_and_failed_stage_fields(self) -> None:
        report = {
            "workflow": "plans/workflow.yaml",
            "status": "failed",
            "laser": {"orientation": "vertical"},
            "stages": [{
                "stage": "intrinsics",
                "status": "failed",
                "metrics": None,
                "quality_gates": None,
                "arguments": None,
            }],
        }
        run = CalibrationRun.from_workflow_report(
            report,
            source_path="/tmp/project/reports/workflow_legacy.yaml",
        )

        self.assertEqual(run.run_id, "workflow_legacy")
        self.assertEqual(run.project_id, "project")
        self.assertEqual(run.laser_orientation, "vertical")
        self.assertEqual(run.stages[0].metrics, {})
        self.assertEqual(run.stages[0].quality_gates, [])
        self.assertEqual(run.overall, "fail")


if __name__ == "__main__":
    unittest.main()
