import json
import tempfile
import unittest
from pathlib import Path

import yaml

from calibration_tool.gui.acceptance_plan import (
    ensure_default_acceptance_plan,
    update_acceptance_plan_from_workflow,
)
from calibration_tool.io_utils import load_document


ROOT = Path(__file__).resolve().parents[1]


class AcceptancePlanTests(unittest.TestCase):
    def test_default_plan_is_created_with_formal_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary) / "project"
            plan_path = ensure_default_acceptance_plan(
                workspace,
                "laser plane demo",
                repo_root=ROOT,
            )
            self.assertEqual(plan_path, workspace / "plans" / "acceptance_plan.yaml")
            document = load_document(plan_path)
            self.assertEqual(document["schema_version"], 1)
            self.assertEqual(
                (plan_path.parent / document["policy"]).resolve(),
                (ROOT / "configs" / "acceptance_policy.yaml").resolve(),
            )
            self.assertEqual(document["inputs"]["workflow_report"], None)
            self.assertFalse((workspace / "reports" / "acceptance").exists())

    def test_workflow_completion_fills_only_empty_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan_path = ensure_default_acceptance_plan(root / "project", "demo", repo_root=ROOT)
            workflow_path = root / "workflow.yaml"
            workflow_path.write_text(
                "schema_version: 1\n"
                "report: reports/workflow_run.yaml\n"
                "stages: []\n",
                encoding="utf-8",
            )
            compensation_dir = root / "runs" / "ground_bias"
            compensation_dir.mkdir(parents=True)
            compensation_path = compensation_dir / "compensation_metrics.json"
            compensation_path.write_text(json.dumps({"loaded_frame_count": 3}), encoding="utf-8")
            (compensation_dir / "ground_bias_table.csv").write_text("u,bias\n1,0\n", encoding="utf-8")
            result = {
                "schema_version": 1,
                "workflow": str(workflow_path),
                "status": "completed",
                "stages": [{"stage": "ground_bias", "result_file": str(compensation_path)}],
                "overall": "pass",
                "counts": {"pass": 1, "warn": 0, "fail": 0},
                "gates": [],
            }
            update = update_acceptance_plan_from_workflow(plan_path, workflow_path, result)
            self.assertTrue(update.changed)
            document = load_document(plan_path)
            inputs = document["inputs"]
            report_path = root / "reports" / "workflow_run.yaml"
            self.assertTrue(report_path.is_file())
            self.assertEqual((plan_path.parent / inputs["workflow_report"]).resolve(), report_path.resolve())
            self.assertEqual(inputs["quality_reports"], [inputs["workflow_report"]])
            self.assertEqual((plan_path.parent / inputs["compensation_metrics"]).resolve(), compensation_path.resolve())
            self.assertEqual(len(inputs["artifacts"]), 1)

            inputs["runtime_config"] = "manual/runtime.yaml"
            inputs["quality_reports"] = ["manual/quality.yaml"]
            plan_path.write_text(
                yaml.safe_dump(document, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            second = update_acceptance_plan_from_workflow(plan_path, workflow_path, result)
            self.assertFalse(second.changed)
            preserved = load_document(plan_path)["inputs"]
            self.assertEqual(preserved["runtime_config"], "manual/runtime.yaml")
            self.assertEqual(preserved["quality_reports"], ["manual/quality.yaml"])


if __name__ == "__main__":
    unittest.main()
