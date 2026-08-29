import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from calibration_tool.workflow import run_workflow


class WorkflowReportTests(unittest.TestCase):
    def test_laser_surface_stage_receives_project_orientation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = root / "workflow.yaml"
            plan.write_text(yaml.safe_dump({
                "schema_version": 1,
                "calibration_src": ".",
                "stages": [{"name": "laser_surface_models", "options": {"config": "fit.yaml"}}],
            }), encoding="utf-8")
            record = {"stage": "laser_surface_models", "status": "completed", "quality_gates": []}
            with patch("calibration_tool.workflow.ComputationService") as service:
                service.return_value.run.return_value = record
                report = run_workflow(plan, laser_orientation="vertical")

            arguments = service.return_value.run.call_args.args[1]
            self.assertEqual(
                arguments[arguments.index("--laser-orientation") + 1],
                "vertical",
            )
            self.assertEqual(report["laser"], {"orientation": "vertical"})

    def test_report_aggregates_stage_gates_for_acceptance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = root / "workflow.yaml"
            plan.write_text(yaml.safe_dump({
                "schema_version": 1,
                "calibration_src": ".",
                "report": "workflow_report.yaml",
                "stages": [{"name": "intrinsics", "options": {}}],
            }), encoding="utf-8")
            record = {
                "stage": "intrinsics",
                "status": "completed",
                "quality_gates": [{
                    "id": "intrinsics.fit_rmse",
                    "status": "pass",
                    "actual": 0.1,
                    "expected": "<= 0.30 px",
                }],
            }
            with patch("calibration_tool.workflow.ComputationService") as service:
                service.return_value.run.return_value = record
                report = run_workflow(plan)

            self.assertEqual(report["overall"], "pass")
            self.assertEqual(report["counts"], {"pass": 1, "warn": 0, "fail": 0})
            self.assertEqual(
                report["gates"][0]["id"],
                "intrinsics.fit_rmse",
            )
            persisted = yaml.safe_load((root / "workflow_report.yaml").read_text(encoding="utf-8"))
            self.assertEqual(persisted["overall"], "pass")


if __name__ == "__main__":
    unittest.main()
