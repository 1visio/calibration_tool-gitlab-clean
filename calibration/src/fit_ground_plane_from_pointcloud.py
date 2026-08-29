#!/usr/bin/env python3
"""Fit Zg = a*Xg + b*Yg + c to an explicitly selected stripe region.

The script also reports the geometric degeneracy of fitting a ground plane from
one reconstructed laser stripe and exports signed residual profiles.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--u-min", type=float, default=500.0)
    parser.add_argument("--u-max", type=float, default=2300.0)
    return parser.parse_args()


def natural_key(path: Path) -> tuple[int, str]:
    suffix = path.parent.name.removeprefix("pose_1")
    if not suffix:
        return (1, path.parent.name)
    try:
        return (int(suffix.removeprefix("_")), path.parent.name)
    except ValueError:
        return (10**9, path.parent.name)


def load_selected_points(input_dir: Path, u_min: float, u_max: float) -> pd.DataFrame:
    paths = sorted(input_dir.glob("*/points_ground.csv"), key=natural_key)
    if not paths:
        raise FileNotFoundError(f"No */points_ground.csv found under {input_dir}")

    frames: list[pd.DataFrame] = []
    required = {"u", "Xg_mm", "Yg_mm", "Zg_mm"}
    for path in paths:
        frame = pd.read_csv(path)
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        frame = frame.loc[
            frame["u"].between(u_min, u_max, inclusive="both"),
            ["u", "Xg_mm", "Yg_mm", "Zg_mm"],
        ].copy()
        frame.insert(0, "frame", path.parent.name)
        frames.append(frame)

    selected = pd.concat(frames, ignore_index=True)
    finite = np.isfinite(selected[["u", "Xg_mm", "Yg_mm", "Zg_mm"]]).all(axis=1)
    selected = selected.loc[finite].reset_index(drop=True)
    if len(selected) < 3:
        raise ValueError("Fewer than three finite selected points")
    return selected


def fit_z_plane(points: pd.DataFrame) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    xyz = points[["Xg_mm", "Yg_mm", "Zg_mm"]].to_numpy(dtype=np.float64)
    design = np.column_stack([xyz[:, 0], xyz[:, 1], np.ones(len(xyz))])
    coeff, _, rank, singular = np.linalg.lstsq(design, xyz[:, 2], rcond=None)
    vertical = xyz[:, 2] - design @ coeff

    normal_raw = np.array([-coeff[0], -coeff[1], 1.0], dtype=np.float64)
    normal = normal_raw / np.linalg.norm(normal_raw)
    if normal[2] < 0.0:
        normal = -normal
    signed_orthogonal = vertical / np.linalg.norm(normal_raw)
    angle_deg = math.degrees(math.acos(float(np.clip(normal[2], -1.0, 1.0))))

    centred_xy = xyz[:, :2] - xyz[:, :2].mean(axis=0)
    xy_singular = np.linalg.svd(centred_xy, full_matrices=False, compute_uv=False)
    result: dict[str, object] = {
        "model": "Zg_mm = a*Xg_mm + b*Yg_mm + c_mm",
        "a": float(coeff[0]),
        "b": float(coeff[1]),
        "c_mm": float(coeff[2]),
        "normal_unit_positive_z": [float(value) for value in normal],
        "angle_with_positive_Zg_axis_deg": angle_deg,
        "vertical_residual_rms_mm": float(np.sqrt(np.mean(vertical**2))),
        "vertical_residual_max_abs_mm": float(np.max(np.abs(vertical))),
        "orthogonal_residual_rms_mm": float(np.sqrt(np.mean(signed_orthogonal**2))),
        "orthogonal_residual_max_abs_mm": float(np.max(np.abs(signed_orthogonal))),
        "design_rank": int(rank),
        "design_singular_values": [float(value) for value in singular],
        "design_condition_number": float(singular[0] / singular[-1]),
        "centered_XY_singular_values_mm": [float(value) for value in xy_singular],
        "centered_XY_aspect_ratio": float(xy_singular[0] / xy_singular[-1]),
    }
    return result, vertical, signed_orthogonal


def fit_z_plane_robust(
    points: pd.DataFrame,
    *,
    mad_threshold: float = 3.5,
    max_iterations: int = 8,
    min_points: int = 6,
) -> tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray]:
    """Robustly fit ``Zg = a*Xg + b*Yg + c`` using iterative MAD rejection.

    The returned residual arrays cover every input row. ``inlier_mask`` marks
    the points retained by the final fit. Vertical residual is the quantity to
    subtract from ``Zg``; orthogonal residual is supplied for diagnostics.
    """

    required = ["Xg_mm", "Yg_mm", "Zg_mm"]
    missing = [name for name in required if name not in points.columns]
    if missing:
        raise ValueError(f"Plane-fit input is missing columns: {missing}")
    if mad_threshold <= 0:
        raise ValueError("mad_threshold must be > 0")
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")
    if min_points < 3:
        raise ValueError("min_points must be >= 3")

    xyz = points[required].to_numpy(dtype=np.float64)
    finite = np.all(np.isfinite(xyz), axis=1)
    if np.count_nonzero(finite) < min_points:
        raise ValueError(
            f"Fewer than {min_points} finite points are available for plane fitting"
        )

    inlier = finite.copy()
    iteration_count = 0
    for iteration_count in range(1, max_iterations + 1):
        fit_points = pd.DataFrame(xyz[inlier], columns=required)
        fit_result, _, _ = fit_z_plane(fit_points)
        if int(fit_result["design_rank"]) < 3:
            raise ValueError("Per-frame plane fit is rank deficient")

        predicted = (
            float(fit_result["a"]) * xyz[:, 0]
            + float(fit_result["b"]) * xyz[:, 1]
            + float(fit_result["c_mm"])
        )
        vertical = xyz[:, 2] - predicted
        center = float(np.median(vertical[inlier]))
        mad = float(np.median(np.abs(vertical[inlier] - center)))
        robust_sigma = 1.4826 * mad
        if not np.isfinite(robust_sigma) or robust_sigma <= 1.0e-12:
            break

        updated = finite & (
            np.abs(vertical - center) <= mad_threshold * robust_sigma
        )
        if np.count_nonzero(updated) < min_points:
            raise ValueError("MAD rejection left too few points for plane fitting")
        if np.array_equal(updated, inlier):
            break
        inlier = updated

    final_points = pd.DataFrame(xyz[inlier], columns=required)
    result, _, _ = fit_z_plane(final_points)
    if int(result["design_rank"]) < 3:
        raise ValueError("Per-frame plane fit is rank deficient")

    predicted = (
        float(result["a"]) * xyz[:, 0]
        + float(result["b"]) * xyz[:, 1]
        + float(result["c_mm"])
    )
    vertical = xyz[:, 2] - predicted
    normal_norm = math.sqrt(
        float(result["a"]) ** 2 + float(result["b"]) ** 2 + 1.0
    )
    signed_orthogonal = vertical / normal_norm
    result = dict(result)
    result.update(
        {
            "fit_method": "iterative_mad",
            "mad_threshold": float(mad_threshold),
            "max_iterations": int(max_iterations),
            "iteration_count": int(iteration_count),
            "input_point_count": int(len(points)),
            "finite_point_count": int(np.count_nonzero(finite)),
            "inlier_point_count": int(np.count_nonzero(inlier)),
        }
    )
    return result, vertical, signed_orthogonal, inlier


def fit_z_vs_x(points: pd.DataFrame) -> tuple[dict[str, float], np.ndarray]:
    x = points["Xg_mm"].to_numpy(dtype=np.float64)
    z = points["Zg_mm"].to_numpy(dtype=np.float64)
    design = np.column_stack([x, np.ones(len(x))])
    coeff, *_ = np.linalg.lstsq(design, z, rcond=None)
    residual = z - design @ coeff
    return {
        "a": float(coeff[0]),
        "c_mm": float(coeff[1]),
        "residual_rms_mm": float(np.sqrt(np.mean(residual**2))),
        "residual_max_abs_mm": float(np.max(np.abs(residual))),
        "residual_peak_to_valley_mm": float(np.ptp(residual)),
    }, residual


def aggregate_profile(points: pd.DataFrame) -> pd.DataFrame:
    profile_input = points.copy()
    profile_input["u_bin"] = np.rint(profile_input["u"]).astype(int)
    return profile_input.groupby("u_bin", as_index=False).agg(
        u_mean_px=("u", "mean"),
        Xg_mean_mm=("Xg_mm", "mean"),
        plane_vertical_residual_mean_mm=("plane_vertical_residual_mm", "mean"),
        plane_vertical_residual_std_mm=("plane_vertical_residual_mm", "std"),
        plane_orthogonal_residual_mean_mm=("plane_orthogonal_residual_mm", "mean"),
        plane_orthogonal_residual_std_mm=("plane_orthogonal_residual_mm", "std"),
        linear_x_residual_mean_mm=("linear_x_residual_mm", "mean"),
        linear_x_residual_std_mm=("linear_x_residual_mm", "std"),
        sample_count=("frame", "count"),
    )


def plot_plane_residual(points: pd.DataFrame, profile: pd.DataFrame, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(12, 5.5))
    for _, frame in points.groupby("frame", sort=False):
        axis.plot(
            frame["u"],
            frame["plane_orthogonal_residual_mm"],
            color="#7aa6d8",
            alpha=0.28,
            linewidth=0.65,
        )
    axis.plot(
        profile["u_mean_px"],
        profile["plane_orthogonal_residual_mean_mm"],
        color="#c62828",
        linewidth=1.25,
        label="10-frame mean",
    )
    axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    axis.set_xlabel("image column u / px")
    axis.set_ylabel("signed orthogonal plane residual / mm")
    axis.set_title("Residual after removing fitted Zg = aXg + bYg + c")
    axis.grid(True, linestyle="--", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_linear_x_residual(points: pd.DataFrame, profile: pd.DataFrame, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(12, 5.5))
    for _, frame in points.groupby("frame", sort=False):
        axis.plot(
            frame["Xg_mm"],
            frame["linear_x_residual_mm"],
            color="#79a878",
            alpha=0.25,
            linewidth=0.65,
        )
    axis.plot(
        profile["Xg_mean_mm"],
        profile["linear_x_residual_mean_mm"],
        color="#6a1b9a",
        linewidth=1.25,
        label="10-frame mean",
    )
    axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    axis.set_xlabel("Xg / mm")
    axis.set_ylabel("residual after Zg = aXg + c / mm")
    axis.set_title("Observable one-stripe curvature after linear detrending")
    axis.grid(True, linestyle="--", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def markdown_report(metrics: dict[str, object]) -> str:
    fit = metrics["requested_plane_fit"]
    alternative = metrics["observable_one_stripe_check"]
    assert isinstance(fit, dict)
    assert isinstance(alternative, dict)
    normal = fit["normal_unit_positive_z"]
    return f"""# 地面激光线点云平面拟合报告

## 选点

- 输入帧数：{metrics['frame_count']}
- 明确地面区域：`u = [{metrics['u_min_px']:.0f}, {metrics['u_max_px']:.0f}] px`
- 选择点数：{metrics['selected_point_count']}
- 选择理由：已检测物体位于左端约 `u = 0–441 px`；从 500 px 起留出过渡安全边界，并在右端留出 147 px 图像边界。

## 按指定模型拟合

`Zg = a*Xg + b*Yg + c`，坐标单位为 mm：

- `a = {fit['a']:.12g}`
- `b = {fit['b']:.12g}`
- `c = {fit['c_mm']:.12g} mm`
- 单位法向量（取 `+Zg` 分量为正）：`[{normal[0]:.12g}, {normal[1]:.12g}, {normal[2]:.12g}]`
- 法向量与 `+Zg` 轴夹角：`{fit['angle_with_positive_Zg_axis_deg']:.9f} deg`
- 正交距离 RMS：`{fit['orthogonal_residual_rms_mm']:.12g} mm`
- 最大绝对正交残差：`{fit['orthogonal_residual_max_abs_mm']:.12g} mm`
- Z 方向残差 RMS：`{fit['vertical_residual_rms_mm']:.12g} mm`
- 最大绝对 Z 方向残差：`{fit['vertical_residual_max_abs_mm']:.12g} mm`

## 几何有效性说明

该结果不能解释为由这批数据独立测得的真实地面法向。点云由相机射线与同一个标定激光平面求交生成，因此所有重建点天然位于该激光平面；单条地面激光线只给出地面与激光平面的近似一维交线，无法唯一约束地面二维平面。这里拟合出的法向实际恢复的是重建所使用的激光平面法向，接近零的残差主要是重建约束本身，而不是地面平整度。

- 设计矩阵条件数：`{fit['design_condition_number']:.9g}`
- 中心化 XY 覆盖长宽比：`{fit['centered_XY_aspect_ratio']:.9g}`
- 单激光线可观测的 `Zg = a*Xg + c` 去趋势残差 RMS：`{alternative['residual_rms_mm']:.9f} mm`
- 对应最大绝对残差：`{alternative['residual_max_abs_mm']:.9f} mm`
- 对应峰谷值：`{alternative['residual_peak_to_valley_mm']:.9f} mm`

若目标是获得真实地面平面，应采集至少 3 条不共线的地面激光线（例如移动相机/激光器，并将每次点云正确统一到同一世界坐标系），或直接使用棋盘/平面靶标上的二维角点集合。

## 输出文件

- `metrics.json`：全部拟合数值和诊断量。
- `selected_ground_points_with_residuals.csv`：所有选中地面点及逐点残差。
- `residual_profile.csv`：按图像列聚合的 10 帧残差曲线数据。
- `plane_residual_curve.png`：用户指定的去拟合平面残差曲线。
- `one_stripe_linear_detrended_residual_curve.png`：单激光线实际可观测的弯曲残差参考图。
"""


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    points = load_selected_points(args.input_dir, args.u_min, args.u_max)
    plane_fit, vertical, orthogonal = fit_z_plane(points)
    linear_x_fit, linear_x_residual = fit_z_vs_x(points)

    points["plane_vertical_residual_mm"] = vertical
    points["plane_orthogonal_residual_mm"] = orthogonal
    points["linear_x_residual_mm"] = linear_x_residual
    profile = aggregate_profile(points)

    metrics: dict[str, object] = {
        "input_dir": str(args.input_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "selection_definition": "Explicit central ground stripe; obstacle and edge margins excluded.",
        "u_min_px": float(args.u_min),
        "u_max_px": float(args.u_max),
        "frame_count": int(points["frame"].nunique()),
        "selected_point_count": int(len(points)),
        "coordinate_unit": "mm",
        "requested_plane_fit": plane_fit,
        "observable_one_stripe_check": {
            "model": "Zg_mm = a*Xg_mm + c_mm",
            **linear_x_fit,
        },
        "interpretation": (
            "The requested free 3D plane fit recovers the laser plane imposed by reconstruction, "
            "not an independently observable ground plane. A single stripe is insufficient to "
            "identify the true 2D ground plane."
        ),
    }

    points.to_csv(
        args.output_dir / "selected_ground_points_with_residuals.csv",
        index=False,
        float_format="%.12g",
    )
    profile.to_csv(args.output_dir / "residual_profile.csv", index=False, float_format="%.12g")
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "ground_plane_fit_report.md").write_text(
        markdown_report(metrics), encoding="utf-8"
    )
    plot_plane_residual(points, profile, args.output_dir / "plane_residual_curve.png")
    plot_linear_x_residual(
        points,
        profile,
        args.output_dir / "one_stripe_linear_detrended_residual_curve.png",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
