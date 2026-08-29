from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .models import StripeProfile


def summarize_profile_quality(profile: StripeProfile) -> dict[str, int | float]:
    summary: dict[str, int | float] = {
        "point_count": int(profile.u_px.size),
        "valid_count": int(profile.valid.sum()),
        "valid_ratio": float(profile.valid.mean()) if profile.valid.size else 0.0,
    }
    _add_distribution(summary, "contrast", profile.contrast)
    _add_distribution(summary, "snr", profile.snr)
    _add_distribution(summary, "fwhm_px", profile.fwhm_px)

    if profile.saturated is not None:
        summary["saturated_count"] = int(profile.saturated.sum())
        summary["saturated_ratio"] = (
            float(profile.saturated.mean()) if profile.saturated.size else 0.0
        )

    adjacent = (
        profile.valid[1:]
        & profile.valid[:-1]
        & np.isclose(np.diff(profile.u_px), 1.0)
    )
    if np.any(adjacent):
        steps = np.abs(np.diff(profile.v_px)[adjacent])
        summary["center_step_abs_px_p95"] = float(np.percentile(steps, 95.0))
    return summary


def summarize_repeatability(
    profiles: Sequence[StripeProfile],
) -> dict[str, int | float]:
    if len(profiles) < 2:
        raise ValueError("重复性至少需要两帧 StripeProfile")

    reference_u = profiles[0].u_px
    if any(
        profile.u_px.shape != reference_u.shape
        or not np.array_equal(profile.u_px, reference_u)
        for profile in profiles[1:]
    ):
        raise ValueError("重复性统计要求各帧使用相同的 u_px 列坐标")

    centers = np.vstack([profile.v_px for profile in profiles])
    valid = np.vstack(
        [profile.valid & np.isfinite(profile.v_px) for profile in profiles]
    )
    valid_counts = valid.sum(axis=0)
    repeatable = valid_counts >= 2
    center_std = np.full(reference_u.size, np.nan, dtype=np.float64)
    for column in np.flatnonzero(repeatable):
        center_std[column] = np.std(centers[valid[:, column], column], ddof=1)

    finite_std = center_std[np.isfinite(center_std)]
    summary: dict[str, int | float] = {
        "frame_count": len(profiles),
        "column_count": int(reference_u.size),
        "repeatable_column_count": int(repeatable.sum()),
        "repeatable_column_ratio": float(repeatable.mean()) if repeatable.size else 0.0,
        "mean_frame_valid_ratio": float(
            np.mean([profile.valid.mean() for profile in profiles])
        ),
    }
    if finite_std.size:
        summary.update(
            {
                "center_std_px_median": float(np.median(finite_std)),
                "center_std_px_p95": float(np.percentile(finite_std, 95.0)),
                "center_std_px_max": float(np.max(finite_std)),
            }
        )
    return summary


def _add_distribution(
    summary: dict[str, int | float],
    name: str,
    values: np.ndarray | None,
) -> None:
    if values is None:
        return
    finite = values[np.isfinite(values)]
    if not finite.size:
        return
    summary[f"{name}_median"] = float(np.median(finite))
    summary[f"{name}_p05"] = float(np.percentile(finite, 5.0))
    summary[f"{name}_p95"] = float(np.percentile(finite, 95.0))
