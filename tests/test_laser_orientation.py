import unittest

import numpy as np

from scripts.fit_laser_models_from_triplets import (
    select_one_per_column,
    select_one_per_scanline,
)


class LaserOrientationPostprocessTests(unittest.TestCase):
    def test_horizontal_path_matches_frozen_legacy_column_result(self):
        x = np.array([0.1, 0.2, 1.1, 1.2, 2.1, 3.1], dtype=float)
        y = np.array([5.0, 4.0, 5.1, 9.0, 5.2, 5.3], dtype=float)
        response = np.array([1.0, 4.0, 5.0, 2.0, 3.0, 6.0], dtype=float)
        legacy = select_one_per_column(x, y, response, 2, 2.0, 900)
        generic = select_one_per_scanline(
            x, y, response, 2, 2.0, 900, orientation="horizontal"
        )
        expected = (
            np.array([0.2, 1.1, 2.1, 3.1]),
            np.array([4.0, 5.1, 5.2, 5.3]),
            np.array([4.0, 5.0, 3.0, 6.0]),
        )
        for actual, frozen in zip(legacy, expected, strict=True):
            np.testing.assert_array_equal(actual, frozen)
        for expected, actual in zip(legacy, generic, strict=True):
            np.testing.assert_array_equal(actual, expected)

    def test_vertical_keeps_strongest_per_row_sorts_y_and_fits_x_of_y(self):
        y = np.repeat(np.arange(20, dtype=float), 2)
        x = np.repeat(2.0 * np.arange(20, dtype=float) + 1.0, 2)
        x[1::2] += 20.0
        response = np.tile([5.0, 1.0], 20)
        x[20] = 100.0  # strongest candidate on row 10, removed by x=f(y) continuity
        selected_x, selected_y, selected_response = select_one_per_scanline(
            x,
            y,
            response,
            poly_degree=1,
            outlier_threshold_px=0.5,
            max_points=900,
            orientation="vertical",
        )
        self.assertTrue(np.all(np.diff(selected_y) > 0))
        self.assertNotIn(10.0, selected_y)
        np.testing.assert_allclose(selected_x, 2.0 * selected_y + 1.0)
        np.testing.assert_array_equal(selected_response, np.full(selected_y.size, 5.0))


if __name__ == "__main__":
    unittest.main()
