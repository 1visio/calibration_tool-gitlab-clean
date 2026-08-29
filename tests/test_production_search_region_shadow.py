from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


CALIBRATION_SRC = Path(__file__).resolve().parents[1] / "calibration" / "src"
if str(CALIBRATION_SRC) not in sys.path:
    sys.path.insert(0, str(CALIBRATION_SRC))

import realtime_steger as steger  # noqa: E402


OPTIONS = {
    "sigma": 1.5,
    "threshold": 30.0,
    "deriv_thresh": 0.5,
    "roi_margin": 4,
    "roi_max_height": 80,
    "scan_axis": "column",
}


def two_horizontal_bands() -> np.ndarray:
    image = np.zeros((80, 64), dtype=np.uint8)
    image[20:23, :] = 200
    image[60:63, :32] = 100
    return image


def assert_formal_arrays_equal(
    case: unittest.TestCase,
    first: steger.StegerExtraction,
    second: steger.StegerExtraction,
) -> None:
    for name in (
        "u_px",
        "v_px",
        "valid",
        "response",
        "offset_px",
        "normal_y_abs",
        "corrected_signal",
    ):
        with case.subTest(field=name):
            np.testing.assert_array_equal(getattr(first, name), getattr(second, name))


class ProductionSearchRegionShadowTests(unittest.TestCase):
    def test_column_detector_summary_preserves_all_normal_axis_components(self) -> None:
        extracted = steger.extract_steger(two_horizontal_bands(), OPTIONS)
        summary = extracted.detector_summary

        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertEqual(summary.normal_axis, "v")
        self.assertEqual(summary.normal_axis_extent, 80)
        self.assertEqual(summary.row_peak.shape, (80,))
        self.assertEqual(summary.row_sum.shape, (80,))
        self.assertEqual(summary.seed, 20)
        self.assertEqual(summary.adaptive_threshold, 60.0)
        self.assertEqual(summary.active_intervals, ((20, 23), (60, 63)))
        self.assertEqual(summary.seed_active_interval, (20, 23))
        self.assertEqual(summary.margin_before_clip, (16, 27))
        self.assertEqual(summary.margin_after_clip, (16, 27))
        self.assertFalse(summary.margin_clamped_start)
        self.assertFalse(summary.margin_clamped_end)
        self.assertFalse(summary.roi_max_height_applied)
        self.assertEqual(
            (summary.auto_search_region.start_px, summary.auto_search_region.end_px),
            (16, 27),
        )
        np.testing.assert_array_equal(
            np.flatnonzero(summary.active_mask),
            np.r_[20:23, 60:63],
        )

    def test_shadow_proposal_records_outside_evidence_without_changing_formal_band(self) -> None:
        extracted = steger.extract_steger(two_horizontal_bands(), OPTIONS)

        self.assertEqual(extracted.metadata["final_search_region_start_px"], 16.0)
        self.assertEqual(extracted.metadata["final_search_region_end_px"], 27.0)
        self.assertEqual(extracted.corrected_signal.shape[0], 11)
        self.assertEqual(extracted.metadata["outside_region_active_intervals_px"], [[60, 63]])
        self.assertEqual(extracted.metadata["outside_region_peak_intervals_px"], [[60, 63]])
        self.assertTrue(extracted.metadata["shadow_would_expand"])
        self.assertEqual(
            extracted.metadata["shadow_reason"],
            "significant_intensity_outside_current_region",
        )
        self.assertEqual(
            (
                extracted.metadata["shadow_proposed_search_region_start_px"],
                extracted.metadata["shadow_proposed_search_region_end_px"],
            ),
            (6.0, 77.0),
        )

    def test_row_mode_uses_original_u_as_normal_axis(self) -> None:
        horizontal = two_horizontal_bands()
        vertical = np.ascontiguousarray(horizontal.T)
        column = steger.extract_steger(horizontal, OPTIONS)
        row = steger.extract_steger(vertical, dict(OPTIONS, scan_axis="row"))
        column_summary = column.detector_summary
        row_summary = row.detector_summary

        self.assertIsNotNone(column_summary)
        self.assertIsNotNone(row_summary)
        assert column_summary is not None and row_summary is not None
        self.assertEqual(row_summary.normal_axis, "u")
        self.assertEqual(row_summary.normal_axis_extent, vertical.shape[1])
        np.testing.assert_array_equal(row_summary.row_peak, column_summary.row_peak)
        np.testing.assert_array_equal(row_summary.row_sum, column_summary.row_sum)
        np.testing.assert_array_equal(row_summary.active_mask, column_summary.active_mask)
        self.assertEqual(row_summary.active_intervals, column_summary.active_intervals)
        self.assertEqual(row_summary.seed_active_interval, column_summary.seed_active_interval)
        self.assertEqual(
            row.metadata["shadow_proposed_search_region_start_px"],
            column.metadata["shadow_proposed_search_region_start_px"],
        )
        self.assertEqual(
            row.metadata["shadow_proposed_search_region_end_px"],
            column.metadata["shadow_proposed_search_region_end_px"],
        )

    def test_safe_active_interval_does_not_propose_expansion(self) -> None:
        image = np.zeros((80, 64), dtype=np.uint8)
        image[30:33, :] = 200
        extracted = steger.extract_steger(image, dict(OPTIONS, roi_margin=20))

        self.assertFalse(extracted.metadata["shadow_would_expand"])
        self.assertEqual(
            extracted.metadata["shadow_reason"],
            "current_region_has_safe_intensity_clearance",
        )
        self.assertEqual(
            extracted.metadata["shadow_proposed_search_region_start_px"],
            extracted.metadata["final_search_region_start_px"],
        )
        self.assertEqual(
            extracted.metadata["shadow_proposed_search_region_end_px"],
            extracted.metadata["final_search_region_end_px"],
        )

    def test_detector_summary_records_margin_clamp_and_max_height(self) -> None:
        image = np.zeros((80, 64), dtype=np.uint8)
        image[:40, :] = 200
        extracted = steger.extract_steger(
            image,
            dict(OPTIONS, roi_margin=4, roi_max_height=20),
        )
        summary = extracted.detector_summary

        self.assertIsNotNone(summary)
        assert summary is not None and summary.auto_search_region is not None
        self.assertEqual(summary.seed_active_interval, (0, 40))
        self.assertEqual(summary.margin_before_clip, (-4, 44))
        self.assertEqual(summary.margin_after_clip, (0, 44))
        self.assertTrue(summary.margin_clamped_start)
        self.assertFalse(summary.margin_clamped_end)
        self.assertTrue(summary.roi_max_height_applied)
        self.assertEqual(
            (summary.auto_search_region.start_px, summary.auto_search_region.end_px),
            (0, 20),
        )

    def test_resolver_return_value_cannot_change_formal_extraction(self) -> None:
        image = two_horizontal_bands()
        baseline = steger.extract_steger(image, OPTIONS)
        forced = steger.ProductionSearchRegionResolution(
            steger.LaserSearchRegion(0, 80, "test-shadow-only"),
            True,
            "forced_test_proposal",
        )

        with mock.patch.object(
            steger,
            "resolve_production_search_region",
            return_value=forced,
        ):
            shadow_changed = steger.extract_steger(image, OPTIONS)

        assert_formal_arrays_equal(self, baseline, shadow_changed)
        self.assertEqual(
            shadow_changed.metadata["final_search_region_start_px"],
            baseline.metadata["final_search_region_start_px"],
        )
        self.assertEqual(
            shadow_changed.metadata["final_search_region_end_px"],
            baseline.metadata["final_search_region_end_px"],
        )
        self.assertEqual(
            shadow_changed.metadata["shadow_proposed_search_region_end_px"],
            80.0,
        )


if __name__ == "__main__":
    unittest.main()
