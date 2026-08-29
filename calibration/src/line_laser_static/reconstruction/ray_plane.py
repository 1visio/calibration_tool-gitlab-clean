from __future__ import annotations

import numpy as np

from ..calibration import LaserCalibration
from ..models import PointCloud, StripeProfile


class RayPlaneReconstructor:
    def __init__(self, calibration: LaserCalibration, parallel_epsilon: float = 1e-9) -> None:
        if np.any(np.abs(calibration.distortion) > 1e-12):
            raise ValueError(
                "基线重建器只接受已去畸变像素；真实非零畸变请先接入 OpenCV 去畸变"
            )
        self.calibration = calibration
        self.parallel_epsilon = parallel_epsilon
        self._inverse_camera_matrix = np.linalg.inv(calibration.camera_matrix)

    def reconstruct(self, profile: StripeProfile) -> PointCloud:
        pixels = np.vstack(
            [profile.u_px, profile.v_px, np.ones(profile.u_px.size, dtype=np.float64)]
        )
        rays = self._inverse_camera_matrix @ pixels
        denominator = self.calibration.plane_normal @ rays
        scale = np.full(profile.u_px.size, np.nan, dtype=np.float64)
        stable = np.abs(denominator) > self.parallel_epsilon
        scale[stable] = -self.calibration.plane_offset / denominator[stable]
        valid = profile.valid & stable & np.isfinite(scale) & (scale > 0)

        points = rays * scale
        points[:, ~valid] = np.nan
        return PointCloud(
            x_mm=points[0],
            y_mm=points[1],
            z_mm=points[2],
            intensity=profile.intensity.copy(),
            confidence=profile.confidence.copy(),
            valid=valid,
        )
