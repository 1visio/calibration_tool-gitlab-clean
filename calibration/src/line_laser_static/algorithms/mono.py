from __future__ import annotations

import re

import numpy as np

from ..models import Frame, StripeProfile


class MonoStripeExtractor:
    """面向 450 nm + 黑白相机的逐列亚像素光条提取器。"""

    def __init__(
        self,
        window_radius: int = 5,
        background_percentile: float = 50.0,
        min_contrast: float = 30.0,
        min_snr: float = 10.0,
        noise_floor: float = 1.0,
        min_fwhm_px: float = 2.0,
        max_fwhm_px: float = 10.0,
        saturation_fraction: float = 0.98,
        reject_saturated: bool = True,
        sensor_max_value: float | None = None,
        roi_x_start: int | None = None,
        roi_x_end: int | None = None,
        roi_y_start: int | None = None,
        roi_y_end: int | None = None,
    ) -> None:
        if window_radius < 1:
            raise ValueError("window_radius 必须大于等于 1")
        if not 0.0 <= background_percentile <= 100.0:
            raise ValueError("background_percentile 必须位于 0 到 100")
        if min_contrast < 0.0 or min_snr < 0.0:
            raise ValueError("min_contrast 和 min_snr 不能为负数")
        if noise_floor <= 0.0:
            raise ValueError("noise_floor 必须大于零")
        if min_fwhm_px <= 0.0 or max_fwhm_px <= min_fwhm_px:
            raise ValueError("FWHM 范围必须满足 0 < min_fwhm_px < max_fwhm_px")
        if not 0.0 < saturation_fraction <= 1.0:
            raise ValueError("saturation_fraction 必须位于 (0, 1]")
        if sensor_max_value is not None and sensor_max_value <= 0.0:
            raise ValueError("sensor_max_value 必须大于零")

        self.window_radius = int(window_radius)
        self.background_percentile = float(background_percentile)
        self.min_contrast = float(min_contrast)
        self.min_snr = float(min_snr)
        self.noise_floor = float(noise_floor)
        self.min_fwhm_px = float(min_fwhm_px)
        self.max_fwhm_px = float(max_fwhm_px)
        self.saturation_fraction = float(saturation_fraction)
        self.reject_saturated = bool(reject_saturated)
        self.sensor_max_value = sensor_max_value
        self.roi_x_start = roi_x_start
        self.roi_x_end = roi_x_end
        self.roi_y_start = roi_y_start
        self.roi_y_end = roi_y_end

    def extract(self, frame: Frame) -> StripeProfile:
        if not np.issubdtype(frame.image.dtype, np.number):
            raise TypeError("黑白相机图像必须是数值型二维数组")

        x_start, x_end = _resolve_interval(
            self.roi_x_start, self.roi_x_end, frame.metadata.width, "ROI x"
        )
        y_start, y_end = _resolve_interval(
            self.roi_y_start, self.roi_y_end, frame.metadata.height, "ROI y"
        )
        sensor_max = _resolve_sensor_max(frame, self.sensor_max_value)

        image = frame.image.astype(np.float32, copy=False)
        if not np.isfinite(image).all():
            raise ValueError("黑白相机图像包含 NaN 或无穷值")
        if float(image.min()) < 0.0 or float(image.max()) > sensor_max:
            raise ValueError(
                f"图像灰度必须位于 [0, {sensor_max:g}]；请核对 pixel_format 或 sensor_max_value"
            )

        roi = image[y_start:y_end, x_start:x_end]
        column_background = np.percentile(
            roi, self.background_percentile, axis=0
        ).astype(np.float32, copy=False)
        signal = roi - column_background[None, :]
        np.maximum(signal, 0.0, out=signal)

        filtered = signal.copy()
        if roi.shape[0] >= 3:
            filtered[1:-1] = (
                signal[:-2] + 2.0 * signal[1:-1] + signal[2:]
            ) * 0.25

        roi_columns = np.arange(roi.shape[1])
        peak_rows = filtered.argmax(axis=0)
        peak_intensity = roi[peak_rows, roi_columns]

        profile_radius = max(
            self.window_radius, int(np.ceil(self.max_fwhm_px)) + 1
        )
        offsets = np.arange(-profile_radius, profile_radius + 1)[:, None]
        raw_rows = peak_rows[None, :] + offsets
        inside = (raw_rows >= 0) & (raw_rows < roi.shape[0])
        sample_rows = np.clip(raw_rows, 0, roi.shape[0] - 1)
        profiles = np.take_along_axis(signal, sample_rows, axis=0) * inside
        profiles -= profiles.min(axis=0, keepdims=True)

        centroid_mask = np.abs(offsets) <= self.window_radius
        centroid_weights = profiles * centroid_mask
        weight_sum = centroid_weights.sum(axis=0)
        v_local = np.full(roi.shape[1], np.nan, dtype=np.float64)
        nonzero = weight_sum > 0.0
        v_local[nonzero] = (
            (centroid_weights[:, nonzero] * raw_rows[:, nonzero]).sum(axis=0)
            / weight_sum[nonzero]
        )

        profile_peak_rows = profiles.argmax(axis=0)
        contrast = profiles[profile_peak_rows, roi_columns].astype(np.float64)
        column_median = np.median(roi, axis=0)
        mad = np.median(np.abs(roi - column_median[None, :]), axis=0)
        noise_sigma = np.maximum(1.4826 * mad, self.noise_floor)
        snr = contrast / noise_sigma
        fwhm_px = _estimate_fwhm_columns(profiles, profile_peak_rows)
        saturated = peak_intensity >= sensor_max * self.saturation_fraction

        width_valid = (
            np.isfinite(fwhm_px)
            & (fwhm_px >= self.min_fwhm_px)
            & (fwhm_px <= self.max_fwhm_px)
        )
        valid = (
            nonzero
            & np.isfinite(v_local)
            & (contrast >= self.min_contrast)
            & (snr >= self.min_snr)
            & width_valid
        )
        if self.reject_saturated:
            valid &= ~saturated

        contrast_score = _threshold_score(contrast, self.min_contrast)
        snr_score = _threshold_score(snr, self.min_snr)
        width_midpoint = 0.5 * (self.min_fwhm_px + self.max_fwhm_px)
        width_half_range = 0.5 * (self.max_fwhm_px - self.min_fwhm_px)
        width_score = np.clip(
            1.0 - 0.5 * np.abs(fwhm_px - width_midpoint) / width_half_range,
            0.0,
            1.0,
        )
        width_score[~np.isfinite(width_score)] = 0.0
        confidence = np.minimum(np.minimum(contrast_score, snr_score), width_score)
        if self.reject_saturated:
            confidence[saturated] = 0.0

        return StripeProfile(
            u_px=np.arange(x_start, x_end, dtype=np.float64),
            v_px=v_local + float(y_start),
            intensity=peak_intensity.astype(np.float64),
            confidence=confidence.astype(np.float64),
            valid=valid,
            contrast=contrast,
            snr=snr.astype(np.float64),
            fwhm_px=fwhm_px,
            saturated=saturated,
        )


def _resolve_interval(
    start: int | None, end: int | None, size: int, label: str
) -> tuple[int, int]:
    resolved_start = 0 if start is None else int(start)
    resolved_end = size if end is None else int(end)
    if not 0 <= resolved_start < resolved_end <= size:
        raise ValueError(
            f"{label} 必须满足 0 <= start < end <= {size}，"
            f"实际为 [{resolved_start}, {resolved_end})"
        )
    return resolved_start, resolved_end


def _resolve_sensor_max(frame: Frame, configured: float | None) -> float:
    if configured is not None:
        return float(configured)

    match = re.search(r"Mono(\d+)", frame.metadata.pixel_format, re.IGNORECASE)
    if match is not None:
        bits = int(match.group(1))
        if not 1 <= bits <= 32:
            raise ValueError(f"不支持的像素格式 {frame.metadata.pixel_format!r}")
        return float((1 << bits) - 1)

    if np.issubdtype(frame.image.dtype, np.integer):
        return float(np.iinfo(frame.image.dtype).max)
    raise ValueError("浮点图像必须显式设置 sensor_max_value")


def _threshold_score(values: np.ndarray, threshold: float) -> np.ndarray:
    if threshold <= 0.0:
        return np.ones(values.shape, dtype=np.float64)
    return np.clip(values / (2.0 * threshold), 0.0, 1.0)


def _estimate_fwhm_columns(
    profiles: np.ndarray, peak_rows: np.ndarray
) -> np.ndarray:
    widths = np.full(profiles.shape[1], np.nan, dtype=np.float64)
    for column, peak_row in enumerate(peak_rows):
        profile = profiles[:, column]
        peak = float(profile[peak_row])
        if peak <= 0.0:
            continue
        half = 0.5 * peak

        left = int(peak_row)
        while left > 0 and profile[left] >= half:
            left -= 1
        if profile[left] >= half:
            continue
        left_crossing = left + (half - profile[left]) / (
            profile[left + 1] - profile[left]
        )

        right = int(peak_row)
        last = profile.size - 1
        while right < last and profile[right] >= half:
            right += 1
        if profile[right] >= half:
            continue
        right_crossing = right - (half - profile[right]) / (
            profile[right - 1] - profile[right]
        )
        widths[column] = float(right_crossing - left_crossing)
    return widths
