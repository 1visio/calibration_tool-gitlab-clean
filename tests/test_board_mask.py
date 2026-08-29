from __future__ import annotations

import unittest

import cv2
import numpy as np

from calibration_tool.board_mask import (
    full_board_object_corners,
    full_board_physical_mask,
    projected_board_boundary,
)
from scripts.fit_laser_models_from_triplets import (
    BoardPose,
    board_mask_for_pose,
    board_inner_mask,
    make_object_points,
)


class BoardMaskTests(unittest.TestCase):
    @staticmethod
    def camera_setup() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        camera_matrix = np.array(
            [[1000.0, 0.0, 100.0], [0.0, 1000.0, 100.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros(5, dtype=np.float64)
        rvec = np.zeros(3, dtype=np.float64)
        tvec = np.array([0.0, 0.0, 1000.0], dtype=np.float64)
        return camera_matrix, dist_coeffs, rvec, tvec

    def test_11x8_20mm_boundary_is_12x9_board(self) -> None:
        corners = full_board_object_corners(11, 8, 20.0)
        np.testing.assert_allclose(
            corners,
            np.array(
                [
                    [-20.0, -20.0, 0.0],
                    [220.0, -20.0, 0.0],
                    [220.0, 160.0, 0.0],
                    [-20.0, 160.0, 0.0],
                ]
            ),
        )

    def test_projected_boundary_uses_pnp_and_zero_inset(self) -> None:
        camera_matrix, dist_coeffs, rvec, tvec = self.camera_setup()
        projected = projected_board_boundary(
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs,
            pattern_cols=11,
            pattern_rows=8,
            square_size_mm=20.0,
        )
        np.testing.assert_allclose(
            projected,
            np.array([[80.0, 80.0], [320.0, 80.0], [320.0, 260.0], [80.0, 260.0]]),
            atol=1.0e-12,
        )

    def test_mask_covers_outer_squares_but_not_outside_board(self) -> None:
        camera_matrix, dist_coeffs, rvec, tvec = self.camera_setup()
        mask = full_board_physical_mask(
            (400, 400),
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs,
            pattern_cols=11,
            pattern_rows=8,
            square_size_mm=20.0,
        )
        # Four outer-square representative points: physical (-10/-10), etc.
        inside_uv = cv2.projectPoints(
            np.array([[-10.0, -10.0, 0.0], [210.0, -10.0, 0.0], [210.0, 150.0, 0.0], [-10.0, 150.0, 0.0]]),
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs,
        )[0].reshape(-1, 2)
        outside_uv = cv2.projectPoints(
            np.array([[-21.0, -10.0, 0.0], [221.0, -10.0, 0.0], [210.0, 161.0, 0.0]]),
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs,
        )[0].reshape(-1, 2)
        self.assertTrue(np.all(mask[np.rint(inside_uv[:, 1]).astype(int), np.rint(inside_uv[:, 0]).astype(int)]))
        self.assertFalse(np.any(mask[np.rint(outside_uv[:, 1]).astype(int), np.rint(outside_uv[:, 0]).astype(int)]))

    def test_formal_mode_defaults_to_full_and_legacy_mode_is_explicit(self) -> None:
        camera_matrix, dist_coeffs, rvec, tvec = self.camera_setup()
        inner_corners = cv2.projectPoints(
            make_object_points(11, 8, 20.0),
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs,
        )[0].reshape(-1, 2)
        pose = BoardPose(
            rvec=rvec,
            tvec=tvec,
            normal=np.array([0.0, 0.0, -1.0]),
            d=1000.0,
            reprojection_rmse_px=0.0,
            corners=inner_corners,
        )
        board_cfg = {"pattern_cols": 11, "pattern_rows": 8, "square_size_mm": 20.0}
        full = board_mask_for_pose((400, 400), pose, camera_matrix, dist_coeffs, board_cfg, {})
        legacy = board_mask_for_pose(
            (400, 400),
            pose,
            camera_matrix,
            dist_coeffs,
            board_cfg,
            {"board_mask_mode": "inner_corner_hull", "board_mask_margin_px": 0},
        )
        self.assertTrue(full[90, 90])
        self.assertFalse(legacy[90, 90])
        np.testing.assert_array_equal(
            legacy,
            board_inner_mask((400, 400), inner_corners, margin_px=0),
        )


if __name__ == "__main__":
    unittest.main()
