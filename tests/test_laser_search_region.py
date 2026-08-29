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
    "roi_margin": 8,
    "roi_max_height": 80,
    "scan_axis": "column",
}


def horizontal_stripe(
    width: int = 160,
    height: int = 96,
    center_v: float = 32.25,
    sigma: float = 1.8,
) -> np.ndarray:
    rows = np.arange(height, dtype=np.float64)[:, None]
    profile = 220.0 * np.exp(-((rows - center_v) ** 2) / (2.0 * sigma**2))
    return np.broadcast_to(profile, (height, width)).astype(np.uint8).copy()


def assert_extraction_arrays_equal(
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


class LaserSearchRegionTests(unittest.TestCase):
    def test_column_without_extra_region_matches_legacy_apis_exactly(self) -> None:
        image = horizontal_stripe()
        legacy = steger.extract_steger_columns(image, OPTIONS)
        unified = steger.extract_steger(image, OPTIONS)

        assert_extraction_arrays_equal(self, unified, legacy)
        np.testing.assert_array_equal(unified.pixels, legacy.pixels)
        np.testing.assert_array_equal(steger.steger_backend(image, OPTIONS), legacy.pixels)

    def test_column_extra_region_matches_additional_band_bounds(self) -> None:
        image = horizontal_stripe()
        bounds = (20, 52)
        region = steger.LaserSearchRegion(*bounds, source="test-reference")
        legacy = steger.extract_steger_columns(
            image, OPTIONS, additional_band_bounds=bounds
        )
        modern_columns = steger.extract_steger_columns(
            image, OPTIONS, additional_search_region=region
        )
        unified = steger.extract_steger(image, OPTIONS, search_region=region)

        assert_extraction_arrays_equal(self, modern_columns, legacy)
        assert_extraction_arrays_equal(self, unified, legacy)
        np.testing.assert_array_equal(unified.pixels, legacy.pixels)

    def test_row_without_extra_region_matches_legacy_transpose_backend(self) -> None:
        horizontal = horizontal_stripe()
        vertical = np.ascontiguousarray(horizontal.T)
        row_options = dict(OPTIONS, scan_axis="row")
        legacy_transposed = steger.extract_steger_columns(
            np.ascontiguousarray(vertical.T), OPTIONS
        ).pixels[:, ::-1]
        unified = steger.extract_steger(vertical, row_options)

        np.testing.assert_array_equal(unified.pixels, legacy_transposed)
        np.testing.assert_array_equal(
            steger.steger_backend(vertical, row_options), legacy_transposed
        )

    def test_row_extra_region_uses_original_u_axis(self) -> None:
        horizontal = horizontal_stripe(center_v=32.25)
        vertical = np.ascontiguousarray(horizontal.T)
        row_options = dict(OPTIONS, scan_axis="row")
        region = steger.LaserSearchRegion(24, 42, source="original-u-reference")

        with mock.patch.object(steger, "_detect_steger_band", return_value=None):
            extracted = steger.extract_steger(
                vertical, row_options, search_region=region
            )

        self.assertGreater(len(extracted.pixels), 120)
        np.testing.assert_allclose(extracted.pixels[:, 0], 32.25, atol=0.08)
        self.assertEqual(extracted.metadata["normal_axis"], "u")
        self.assertEqual(extracted.metadata["final_search_region_start_px"], 24.0)
        self.assertEqual(extracted.metadata["final_search_region_end_px"], 42.0)
        self.assertEqual(
            extracted.metadata["additional_search_region_source"],
            "original-u-reference",
        )

    def test_horizontal_vertical_transpose_centres_are_exactly_symmetric(self) -> None:
        horizontal = horizontal_stripe()
        vertical = np.ascontiguousarray(horizontal.T)
        horizontal_points = steger.extract_steger(horizontal, OPTIONS).pixels
        vertical_points = steger.extract_steger(
            vertical, dict(OPTIONS, scan_axis="row")
        ).pixels

        np.testing.assert_array_equal(vertical_points, horizontal_points[:, ::-1])

    def test_invalid_or_empty_search_region_is_rejected(self) -> None:
        image = horizontal_stripe()
        with self.assertRaisesRegex(ValueError, "start_px < end_px"):
            steger.LaserSearchRegion(10, 10, source="empty")
        with self.assertRaisesRegex(ValueError, "source 不能为空"):
            steger.LaserSearchRegion(10, 20, source="")
        with self.assertRaisesRegex(ValueError, "裁剪到图像法向轴范围后为空"):
            steger.extract_steger(
                image,
                OPTIONS,
                search_region=steger.LaserSearchRegion(200, 220, source="outside"),
            )
        with self.assertRaisesRegex(ValueError, "不能同时指定"):
            steger.extract_steger_columns(
                image,
                OPTIONS,
                additional_search_region=steger.LaserSearchRegion(
                    20, 52, source="new"
                ),
                additional_band_bounds=(20, 52),
            )


if __name__ == "__main__":
    unittest.main()
