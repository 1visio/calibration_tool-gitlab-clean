from __future__ import annotations

import numpy as np

from ..models import Frame, StripeProfile


class CentroidStripeExtractor:
    """逐列峰值邻域灰度重心；仅作为可替换的基线实现。"""

    def __init__(self, window_radius: int = 4, min_contrast: float = 30.0) -> None:
        if window_radius < 1:
            raise ValueError("window_radius 必须大于等于 1")
        self.window_radius = window_radius
        self.min_contrast = min_contrast

    def extract(self, frame: Frame) -> StripeProfile:
        image = frame.image.astype(np.float64, copy=False)
        baseline = image.min(axis=0)
        signal = image - baseline
        peak_rows = signal.argmax(axis=0)
        columns = np.arange(frame.metadata.width)
        peak_signal = signal[peak_rows, columns]
        peak_intensity = image[peak_rows, columns]

        offsets = np.arange(-self.window_radius, self.window_radius + 1)[:, None]
        raw_rows = peak_rows[None, :] + offsets
        inside = (raw_rows >= 0) & (raw_rows < frame.metadata.height)
        sample_rows = np.clip(raw_rows, 0, frame.metadata.height - 1)
        weights = np.take_along_axis(signal, sample_rows, axis=0) * inside
        weight_sum = weights.sum(axis=0)

        v_px = np.full(frame.metadata.width, np.nan, dtype=np.float64)
        nonzero = weight_sum > 0
        v_px[nonzero] = (
            (weights[:, nonzero] * raw_rows[:, nonzero]).sum(axis=0)
            / weight_sum[nonzero]
        )
        confidence = np.clip(peak_signal / 255.0, 0.0, 1.0)
        valid = nonzero & (peak_signal >= self.min_contrast) & np.isfinite(v_px)

        return StripeProfile(
            u_px=np.arange(frame.metadata.width, dtype=np.float64),
            v_px=v_px,
            intensity=peak_intensity.astype(np.float64),
            confidence=confidence.astype(np.float64),
            valid=valid,
        )
