from __future__ import annotations

import unittest

import numpy as np

from scripts.search_region_expansion_characterization import (
    CASES,
    FrameLevel,
    infer_recommendation,
    pair_samples,
)


def _frame(centers: list[float], valid: list[bool], responses: list[float]) -> FrameLevel:
    return FrameLevel(
        center_px=np.asarray(centers, dtype=np.float64),
        valid=np.asarray(valid, dtype=bool),
        response=np.asarray(responses, dtype=np.float64),
        region_start_px=0.0,
        region_end_px=40.0,
    )


class ExpansionCharacterizationTests(unittest.TestCase):
    def test_pair_metrics_use_same_floor_candidate_definition(self) -> None:
        paired = pair_samples(
            [_frame([10.2, 11.9, 14.0], [True, True, False], [2.0, 4.0, np.nan])],
            [_frame([10.3, 12.1, 14.0], [True, True, True], [3.0, 8.0, 5.0])],
        )

        self.assertEqual(paired["paired_count"], 2)
        self.assertAlmostEqual(paired["shift_p50_px"], 0.15)
        self.assertAlmostEqual(paired["shift_p95_px"], 0.195)
        self.assertAlmostEqual(paired["shift_max_px"], 0.2)
        self.assertEqual(paired["same_floor_fraction"], 0.5)
        self.assertEqual(paired["response_ratio_p50"], 1.75)

    def test_recommendation_uses_first_level_with_all_later_transitions_stable(self) -> None:
        level_rows = []
        adjacent_rows = []
        clearances = [13.35, 35.13, 64.9, 24.07]
        for case, clearance in zip(CASES, clearances, strict=True):
            level_rows.append({
                "case_id": case.case_id,
                "expansion_each_side_px": 32,
                "boundary_clearance_p05_px": clearance,
            })
            for before, after, shift, valid_delta in (
                (8, 16, 0.02, 0.0),
                (16, 32, 0.001, 0.02),
                (32, 48, 0.0, 0.0),
            ):
                adjacent_rows.append({
                    "case_id": case.case_id,
                    "from_expansion_px": before,
                    "to_expansion_px": after,
                    "center_shift_p95_px": shift,
                    "valid_fraction_delta": valid_delta,
                    "same_floor_candidate_fraction": 1.0,
                })

        result = infer_recommendation(level_rows, adjacent_rows)

        self.assertTrue(result["characterization_complete"])
        self.assertEqual(result["minimum_stable_expansion_each_side_px"], 32)
        self.assertEqual(result["recommended_minimum_safe_clearance_px"], 14)


if __name__ == "__main__":
    unittest.main()
