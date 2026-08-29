import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "sweep_steger_sigma.py"
SPEC = importlib.util.spec_from_file_location("sweep_steger_sigma", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
SWEEP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SWEEP)


class StegerSigmaSweepTests(unittest.TestCase):
    def test_sigma_list_is_parsed_in_declared_order(self):
        values = SWEEP.parse_sigma_values([0.8, "1.0", 1.5, 3])
        self.assertEqual(values, [0.8, 1.0, 1.5, 3.0])
        with self.assertRaises(ValueError):
            SWEEP.parse_sigma_values([1.5, 1.5])
        with self.assertRaises(ValueError):
            SWEEP.parse_sigma_values([0.0])
        self.assertEqual(SWEEP.sigma_slug(1.0), "sigma_1p0")
        self.assertEqual(SWEEP.sigma_slug(1.5), "sigma_1p5")

    def test_config_copy_changes_only_sigma_and_preserves_vertical(self):
        original = {
            "intrinsics": "camera.yaml",
            "laser": {"orientation": "vertical"},
            "board": {}, "patterns": {}, "datasets": {}, "models": {},
            "extraction": {"method": "steger", "sigma": 1.5, "min_response": 0.8},
        }
        snapshot = copy.deepcopy(original)
        varied = SWEEP.config_for_sigma(original, 2.0)
        SWEEP.assert_only_sigma_changed(original, varied, 2.0)
        self.assertEqual(original, snapshot)
        self.assertEqual(varied["laser"]["orientation"], "vertical")
        self.assertEqual(varied["extraction"]["min_response"], 0.8)

    def test_sweep_yaml_resolves_paths_and_contains_baseline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = {
                "base_config": "base.yaml", "output_dir": "../runs/sweep",
                "sigma_values": [0.8, 1.5, 2.0], "baseline": {"sigma": 1.5},
            }
            path = root / "configs" / "sweep.yaml"
            path.parent.mkdir()
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            loaded = SWEEP.load_sweep_config(path)
            self.assertEqual(loaded["sigma_values"], [0.8, 1.5, 2.0])
            self.assertEqual(Path(loaded["base_config"]), (path.parent / "base.yaml").resolve())

    def test_baseline_gate_passes_exact_formal_metrics_and_fails_drift(self):
        comparison = pd.DataFrame([
            {"model": "circular_cone", "split": "validation", "board_rmse_mm": 0.08321982635298517},
            {"model": "quadratic_graph", "split": "validation", "board_rmse_mm": 0.0801910803822904},
        ])
        points = pd.DataFrame(
            [{"split": "train", "image_id": index % 18 + 1} for index in range(16102)]
            + [{"split": "validation", "image_id": index % 6 + 19} for index in range(5400)]
        )
        baseline = {
            "rmse_abs_tolerance_mm": 1.0e-4,
            "point_count_relative_tolerance": 0.01,
            "expected_train_point_count": 16102,
            "expected_validation_point_count": 5400,
            "expected_validation_board_rmse_mm": {
                "circular_cone": 0.08321982635298517,
                "quadratic_graph": 0.0801910803822904,
            },
        }
        result = SWEEP.baseline_check(comparison, points, baseline, 18, 6, "vertical")
        self.assertEqual(result["status"], "PASS")
        drifted = comparison.copy()
        drifted.loc[drifted["model"] == "circular_cone", "board_rmse_mm"] += 2.0e-4
        result = SWEEP.baseline_check(drifted, points, baseline, 18, 6, "vertical")
        self.assertEqual(result["status"], "FAIL")

    def test_stable_range_requires_three_consecutive_eligible_points(self):
        summary = pd.DataFrame([
            {"sigma": 1.4, "validation_board_rmse_mm": 1.015, "eligible_for_ranking": True},
            {"sigma": 1.5, "validation_board_rmse_mm": 1.010, "eligible_for_ranking": True},
            {"sigma": 1.6, "validation_board_rmse_mm": 1.000, "eligible_for_ranking": True},
            {"sigma": 1.8, "validation_board_rmse_mm": 0.999, "eligible_for_ranking": False},
        ])
        self.assertEqual(SWEEP.stable_ranges(summary, 2.0, 3), [(1.4, 1.6)])
        summary.loc[summary["sigma"] == 1.5, "eligible_for_ranking"] = False
        self.assertEqual(SWEEP.stable_ranges(summary, 2.0, 3), [])


if __name__ == "__main__":
    unittest.main()
