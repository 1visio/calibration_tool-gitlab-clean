#!/usr/bin/env python3
"""Separate first-order tilt from ground extrinsics and the laser model.

The board-only extrinsics are validated with checkerboard corners that were not
used for fitting.  Laser points from an empty-board frame are then compared in
the same ground coordinate system and, independently, against the calibrated
laser plane after intersecting their camera rays with the checkerboard plane.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import calibrate_ground_extrinsics_board_only as board_calib  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intrinsics", type=Path, required=True)
    parser.add_argument("--board-extrinsics", type=Path, required=True)
    parser.add_argument("--board-fit-dir", type=Path, required=True)
    parser.add_argument("--board-validation-dir", type=Path, required=True)
    parser.add_argument("--laser-plane", type=Path, required=True)
    parser.add_argument("--laser-points", type=Path, required=True)
    parser.add_argument("--comparison-extrinsics", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pattern-cols", type=int, default=11)
    parser.add_argument("--pattern-rows", type=int, default=8)
    parser.add_argument("--square-size-mm", type=float, default=20.0)
    parser.add_argument("--u-min", type=float, default=450.0)
    parser.add_argument("--u-max", type=float, default=1950.0)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_transform(document: dict[str, Any]) -> np.ndarray:
    transform = np.asarray(document["T_ground_from_camera"], dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("T_ground_from_camera must be a finite 4x4 matrix")
    return transform


def transform_points(points_camera: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points_camera @ transform[:3, :3].T + transform[:3, 3]


def residual_metrics(residual: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(residual, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    return {
        "point_count": int(values.size),
        "mean_mm": float(np.mean(values)),
        "std_mm": float(np.std(values)),
        "rms_mm": float(np.sqrt(np.mean(values**2))),
        "mean_abs_mm": float(np.mean(np.abs(values))),
        "p95_abs_mm": float(np.percentile(np.abs(values), 95)),
        "max_abs_mm": float(np.max(np.abs(values))),
        "min_mm": float(np.min(values)),
        "max_mm": float(np.max(values)),
        "peak_to_valley_mm": float(np.ptp(values)),
    }


def fit_z_plane(xyz: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    points = np.asarray(xyz, dtype=np.float64)
    design = np.column_stack([points[:, 0], points[:, 1], np.ones(len(points))])
    coeff, _, rank, singular = np.linalg.lstsq(design, points[:, 2], rcond=None)
    vertical = points[:, 2] - design @ coeff
    normal_raw = np.asarray([-coeff[0], -coeff[1], 1.0])
    normal = normal_raw / np.linalg.norm(normal_raw)
    orthogonal = vertical / np.linalg.norm(normal_raw)
    angle_deg = math.degrees(math.acos(float(np.clip(normal[2], -1.0, 1.0))))
    result = {
        "model": "Zg = a*Xg + b*Yg + c",
        "a": float(coeff[0]),
        "b": float(coeff[1]),
        "c_mm": float(coeff[2]),
        "normal_unit_positive_z": [float(value) for value in normal],
        "tilt_from_Zg_deg": angle_deg,
        "slope_x_angle_deg": math.degrees(math.atan(float(coeff[0]))),
        "slope_y_angle_deg": math.degrees(math.atan(float(coeff[1]))),
        "vertical_residual": residual_metrics(vertical),
        "orthogonal_residual": residual_metrics(orthogonal),
        "raw_z_about_zero": residual_metrics(points[:, 2]),
        "design_rank": int(rank),
        "design_condition_number": float(singular[0] / singular[-1]),
    }
    return result, vertical


def fit_linear(x: np.ndarray, y: np.ndarray, coordinate: str) -> tuple[dict[str, Any], np.ndarray]:
    independent = np.asarray(x, dtype=np.float64).reshape(-1)
    dependent = np.asarray(y, dtype=np.float64).reshape(-1)
    design = np.column_stack([independent, np.ones(len(independent))])
    coeff, *_ = np.linalg.lstsq(design, dependent, rcond=None)
    residual = dependent - design @ coeff
    return {
        "model": f"value = slope*{coordinate} + intercept",
        "slope": float(coeff[0]),
        "intercept": float(coeff[1]),
        "slope_angle_deg": math.degrees(math.atan(float(coeff[0]))),
        "residual": residual_metrics(residual),
        "raw_value": residual_metrics(dependent),
    }, residual


def scan_board_points(
    directory: Path,
    split: str,
    intrinsics: board_calib.Intrinsics,
    object_points: np.ndarray,
    pattern: tuple[int, int],
    transform: np.ndarray,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    point_frames: list[pd.DataFrame] = []
    observations: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.tif")):
        observation = board_calib.detect_chessboard(
            path, intrinsics, object_points, pattern, max_pnp_rmse_px=0.25
        )
        points_ground = transform_points(observation.points_camera, transform)
        plane, residual = fit_z_plane(points_ground)
        frame = pd.DataFrame(
            {
                "split": split,
                "frame": path.name,
                "corner_index": np.arange(len(points_ground)),
                "Xc_mm": observation.points_camera[:, 0],
                "Yc_mm": observation.points_camera[:, 1],
                "Zc_mm": observation.points_camera[:, 2],
                "Xg_mm": points_ground[:, 0],
                "Yg_mm": points_ground[:, 1],
                "Zg_mm": points_ground[:, 2],
                "frame_plane_vertical_residual_mm": residual,
            }
        )
        point_frames.append(frame)
        observations.append(
            {
                "split": split,
                "frame": path.name,
                "detection_method": observation.detection_method,
                "pnp_reprojection_rmse_px": observation.reprojection_rmse_px,
                "plane": plane,
            }
        )
    if not point_frames:
        raise FileNotFoundError(f"No checkerboard TIFF files found in {directory}")
    return pd.concat(point_frames, ignore_index=True), observations


def load_laser_plane(path: Path) -> tuple[np.ndarray, float]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    plane = document.get("plane", document)
    normal = np.asarray([plane["a"], plane["b"], plane["c"]], dtype=np.float64)
    norm = float(np.linalg.norm(normal))
    return normal / norm, float(plane["d"]) / norm


def rays_to_ground_plane(
    uv: np.ndarray,
    intrinsics: board_calib.Intrinsics,
    ground_normal_camera: np.ndarray,
    ground_d_mm: float,
) -> np.ndarray:
    normalized = cv2.undistortPoints(
        uv.reshape(-1, 1, 2).astype(np.float64),
        intrinsics.camera_matrix,
        intrinsics.dist_coeffs,
    ).reshape(-1, 2)
    rays = np.column_stack([normalized, np.ones(len(normalized))])
    denominator = rays @ ground_normal_camera
    if np.any(np.abs(denominator) < 1.0e-9):
        raise ValueError("Some laser rays are parallel to the checkerboard plane")
    depth = -ground_d_mm / denominator
    return rays * depth[:, None]


def plot_board_validation(points: pd.DataFrame, plane: dict[str, Any], output: Path) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    coeff = np.asarray([plane["a"], plane["b"], plane["c_mm"]])
    for frame_name, frame in points.groupby("frame", sort=False):
        predicted = (
            coeff[0] * frame["Xg_mm"].to_numpy()
            + coeff[1] * frame["Yg_mm"].to_numpy()
            + coeff[2]
        )
        axes[0].scatter(frame["Xg_mm"], frame["Zg_mm"], s=13, label=frame_name)
        axes[1].scatter(
            frame["Xg_mm"], frame["Zg_mm"].to_numpy() - predicted, s=13
        )
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_ylabel("checkerboard corner Zg / mm")
    axes[0].set_title("Independent checkerboard validation in board-only ground coordinates")
    axes[0].legend(fontsize=8)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Xg / mm")
    axes[1].set_ylabel("residual after validation plane / mm")
    for axis in axes:
        axis.grid(True, linestyle="--", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_laser_selection(all_points: pd.DataFrame, selected: pd.DataFrame, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(12, 5.5))
    axis.scatter(all_points["u"], all_points["Zg_mm"], s=6, color="#b0bec5", label="all extracted")
    axis.scatter(selected["u"], selected["Zg_mm"], s=8, color="#1565c0", label="selected board surface")
    axis.axvline(float(selected["u"].min()), color="#455a64", linestyle="--", linewidth=0.8)
    axis.axvline(float(selected["u"].max()), color="#455a64", linestyle="--", linewidth=0.8)
    axis.set_xlabel("image column u / px")
    axis.set_ylabel("reconstructed Zg / mm")
    axis.set_title("Empty-board laser frame: explicit central pattern-surface selection")
    axis.grid(True, linestyle="--", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_laser_trend(points: pd.DataFrame, laser_fit: dict[str, Any], output: Path) -> None:
    order = np.argsort(points["Xg_mm"].to_numpy())
    x = points["Xg_mm"].to_numpy()[order]
    z = points["Zg_mm"].to_numpy()[order]
    board_z = points["board_validation_plane_Zg_mm"].to_numpy()[order]
    fit_z = laser_fit["slope"] * x + laser_fit["intercept"]
    residual = points["laser_linear_residual_mm"].to_numpy()[order]

    figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].scatter(x, z, s=8, color="#1565c0", alpha=0.75, label="laser reconstruction")
    axes[0].plot(x, fit_z, color="#c62828", linewidth=1.4, label="first-order fit")
    axes[0].plot(x, board_z, color="#2e7d32", linewidth=1.4, label="independent board plane")
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_ylabel("Zg / mm")
    axes[0].set_title("Same physical plane: board validation versus laser reconstruction")
    axes[0].legend()
    axes[1].plot(x, residual, color="#6a1b9a", linewidth=1.0)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Xg / mm")
    axes[1].set_ylabel("laser residual after linear trend / mm")
    for axis in axes:
        axis.grid(True, linestyle="--", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_laser_plane_distance(points: pd.DataFrame, fit: dict[str, Any], output: Path) -> None:
    order = np.argsort(points["true_ground_Xg_mm"].to_numpy())
    x = points["true_ground_Xg_mm"].to_numpy()[order]
    distance = points["laser_plane_signed_distance_at_true_ground_mm"].to_numpy()[order]
    predicted = fit["slope"] * x + fit["intercept"]
    figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].scatter(x, distance, s=8, color="#ef6c00", alpha=0.75)
    axes[0].plot(x, predicted, color="#4e342e", linewidth=1.4)
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_ylabel("signed distance to calibrated laser plane / mm")
    axes[0].set_title("Observed laser on checkerboard, constrained by board-only ground plane")
    axes[1].plot(x, distance - predicted, color="#00838f", linewidth=1.0)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("true ground Xg / mm")
    axes[1].set_ylabel("distance residual after linear trend / mm")
    for axis in axes:
        axis.grid(True, linestyle="--", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_attribution(metrics: dict[str, Any], output: Path) -> None:
    attribution = metrics["first_order_tilt_attribution"]
    labels = [
        "board validation\n(best estimate)",
        "board validation\n(per-frame upper bound)",
        "laser, board-only\nextrinsics",
        "laser, previous\nextrinsics",
    ]
    values = [
        attribution["board_projected_slope_angle_deg"],
        attribution["board_projected_per_frame_max_abs_angle_deg"],
        metrics["laser_empty_board"]["linear_Zg_vs_Xg"]["slope_angle_deg"],
        metrics["comparison_extrinsics"]["linear_Zg_vs_Xg"]["slope_angle_deg"],
    ]
    colors = ["#2e7d32", "#66bb6a", "#c62828", "#ef6c00"]
    figure, axis = plt.subplots(figsize=(10, 5.5))
    bars = axis.bar(labels, values, color=colors)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_ylabel("first-order slope angle along laser X span / deg")
    axis.set_title("First-order tilt attribution")
    axis.grid(True, axis="y", linestyle="--", alpha=0.3)
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + (0.035 if value >= 0 else -0.035),
            f"{value:.4f} deg",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=9,
        )
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def build_report(metrics: dict[str, Any]) -> str:
    board = metrics["board_validation"]
    laser = metrics["laser_empty_board"]
    mismatch = metrics["laser_minus_board_validation_plane"]
    attribution = metrics["first_order_tilt_attribution"]
    model_distance = metrics["laser_model_check"]["signed_distance_to_laser_plane"]
    comparison = metrics["comparison_extrinsics"]
    verdict = attribution["verdict"]
    return f"""# 棋盘外参与激光模型一阶倾斜归因报告

## 数据边界

- 外参：仅由 6 张棋盘图拟合；没有使用激光点。
- 独立验证：2 张未参与拟合的棋盘图，共 {board['point_count']} 个角点。
- 激光对比：棋盘采集后约 1 分钟拍摄的空棋盘激光帧 `{laser['frame']}`。
- 明确平面区域：空棋盘中央图案表面 `u=[{laser['u_min_px']:.0f}, {laser['u_max_px']:.0f}] px`，共 {laser['point_count']} 点；避开外框和棋盘外区域。

## 1. 独立棋盘验证外参

验证角点拟合：

`Zg = {board['plane']['a']:.12g}*Xg + {board['plane']['b']:.12g}*Yg + {board['plane']['c_mm']:.12g}`

- 平面总倾角：`{board['plane']['tilt_from_Zg_deg']:.6f} deg`
- X 向倾角：`{board['plane']['slope_x_angle_deg']:.6f} deg`
- Y 向倾角：`{board['plane']['slope_y_angle_deg']:.6f} deg`
- 相对 `Zg=0` 的 RMS：`{board['plane']['raw_z_about_zero']['rms_mm']:.6f} mm`
- 去掉自身拟合平面后的 RMS：`{board['plane']['vertical_residual']['rms_mm']:.6f} mm`
- 最大绝对残差：`{board['plane']['vertical_residual']['max_abs_mm']:.6f} mm`

## 2. 同一棋盘平面的激光重建点

沿激光跨度的一阶模型：

`Zg = {laser['linear_Zg_vs_Xg']['slope']:.12g}*Xg + {laser['linear_Zg_vs_Xg']['intercept']:.12g}`

- 一阶倾角：`{laser['linear_Zg_vs_Xg']['slope_angle_deg']:.6f} deg`
- 去一阶趋势后 RMS：`{laser['linear_Zg_vs_Xg']['residual']['rms_mm']:.6f} mm`
- 去趋势最大绝对残差：`{laser['linear_Zg_vs_Xg']['residual']['max_abs_mm']:.6f} mm`
- 去趋势峰谷值：`{laser['linear_Zg_vs_Xg']['residual']['peak_to_valley_mm']:.6f} mm`

将独立验证棋盘平面投影到完全相同的激光 XY 采样位置后，激光减棋盘的一阶倾角仍为 `{mismatch['slope_angle_deg']:.6f} deg`。

## 3. 直接检查当前激光平面模型

对每个实测激光像素，先用 board-only 地面平面求相机射线交点，再计算该真实地面点到标定激光平面的有符号距离：

- 距离 RMS：`{model_distance['raw_value']['rms_mm']:.6f} mm`
- 距离范围：`[{model_distance['raw_value']['min_mm']:.6f}, {model_distance['raw_value']['max_mm']:.6f}] mm`
- 距离随 X 的斜率：`{model_distance['slope']:.9f} mm/mm`
- 去一阶趋势后的距离 RMS：`{model_distance['residual']['rms_mm']:.6f} mm`

这说明当前实测激光片与所加载激光平面之间不仅有固定偏移，还有稳定的一阶角度不一致。

## 4. 归因

- 独立棋盘平面投影到激光方向的一阶角：`{attribution['board_projected_slope_angle_deg']:.6f} deg`
- 两张验证图逐帧结果给出的保守外参上界：`{attribution['board_projected_per_frame_max_abs_angle_deg']:.6f} deg`
- 激光重建一阶角：`{laser['linear_Zg_vs_Xg']['slope_angle_deg']:.6f} deg`
- 使用之前外参时，同一批相机系激光点的一阶角：`{comparison['linear_Zg_vs_Xg']['slope_angle_deg']:.6f} deg`
- 更换为 board-only 外参仅改变：`{comparison['angle_change_to_board_only_deg']:.6f} deg`
- 按保守外参上界，至少 `{attribution['minimum_laser_model_fraction_percent']:.2f}%` 的一阶斜率无法由外参解释。

**结论：{verdict}**

这里的“激光模型”包括激光平面参数本身、7 月 29 日标定到 7 月 30 日测量之间的激光器机械漂移，以及与激光中心提取相关的系统误差；本次数据可以排除外参为主要来源，但不能仅凭单条地面激光线继续区分这三者。

## 输出

- `board_validation_residual.png`：独立棋盘角点及去平面残差。
- `laser_selection_profile.png`：空棋盘地面点选择。
- `laser_tilt_and_residual.png`：棋盘平面与激光重建的一阶趋势/残差。
- `laser_plane_model_distance.png`：真实地面射线交点到激光平面的距离。
- `tilt_attribution.png`：一阶倾斜来源对比。
- `summary_metrics.json`、`board_points.csv`、`laser_ground_points.csv`：完整数值与逐点数据。
"""


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    intrinsics = board_calib.load_intrinsics(args.intrinsics)
    board_document = read_json(args.board_extrinsics)
    board_transform = load_transform(board_document)
    object_points = board_calib.chessboard_object_points(
        args.pattern_cols, args.pattern_rows, args.square_size_mm
    )
    pattern = (args.pattern_cols, args.pattern_rows)

    fit_points, fit_observations = scan_board_points(
        args.board_fit_dir,
        "fit",
        intrinsics,
        object_points,
        pattern,
        board_transform,
    )
    validation_points, validation_observations = scan_board_points(
        args.board_validation_dir,
        "validation",
        intrinsics,
        object_points,
        pattern,
        board_transform,
    )
    fit_plane, fit_residual = fit_z_plane(
        fit_points[["Xg_mm", "Yg_mm", "Zg_mm"]].to_numpy()
    )
    validation_plane, validation_residual = fit_z_plane(
        validation_points[["Xg_mm", "Yg_mm", "Zg_mm"]].to_numpy()
    )
    fit_points["pooled_plane_vertical_residual_mm"] = fit_residual
    validation_points["pooled_plane_vertical_residual_mm"] = validation_residual
    board_points = pd.concat([fit_points, validation_points], ignore_index=True)

    laser_all = pd.read_csv(args.laser_points)
    required = {"u", "v", "Xc_mm", "Yc_mm", "Zc_mm", "Xg_mm", "Yg_mm", "Zg_mm"}
    missing = required.difference(laser_all.columns)
    if missing:
        raise ValueError(f"Laser point CSV is missing: {sorted(missing)}")
    laser = laser_all.loc[laser_all["u"].between(args.u_min, args.u_max)].copy()
    if len(laser) < 100:
        raise ValueError("Fewer than 100 laser points in the selected empty-board range")

    laser_linear, laser_linear_residual = fit_linear(
        laser["Xg_mm"].to_numpy(), laser["Zg_mm"].to_numpy(), "Xg"
    )
    laser["laser_linear_residual_mm"] = laser_linear_residual

    board_coeff = np.asarray(
        [validation_plane["a"], validation_plane["b"], validation_plane["c_mm"]]
    )
    board_predicted = (
        board_coeff[0] * laser["Xg_mm"].to_numpy()
        + board_coeff[1] * laser["Yg_mm"].to_numpy()
        + board_coeff[2]
    )
    laser["board_validation_plane_Zg_mm"] = board_predicted
    board_projected_fit, _ = fit_linear(
        laser["Xg_mm"].to_numpy(), board_predicted, "Xg"
    )
    mismatch = laser["Zg_mm"].to_numpy() - board_predicted
    mismatch_fit, mismatch_residual = fit_linear(
        laser["Xg_mm"].to_numpy(), mismatch, "Xg"
    )
    laser["laser_minus_board_plane_mm"] = mismatch
    laser["laser_minus_board_linear_residual_mm"] = mismatch_residual

    per_frame_projected_angles: list[float] = []
    for observation in validation_observations:
        plane = observation["plane"]
        predicted = (
            plane["a"] * laser["Xg_mm"].to_numpy()
            + plane["b"] * laser["Yg_mm"].to_numpy()
            + plane["c_mm"]
        )
        projected, _ = fit_linear(laser["Xg_mm"].to_numpy(), predicted, "Xg")
        observation["projected_on_laser_X_span"] = projected
        per_frame_projected_angles.append(float(projected["slope_angle_deg"]))

    ground_plane = board_document["ground_plane_in_camera"]
    ground_normal_camera = np.asarray(ground_plane["normal"], dtype=np.float64)
    ground_d_mm = float(ground_plane["D_mm"])
    true_ground_camera = rays_to_ground_plane(
        laser[["u", "v"]].to_numpy(),
        intrinsics,
        ground_normal_camera,
        ground_d_mm,
    )
    true_ground = transform_points(true_ground_camera, board_transform)
    laser_normal, laser_d = load_laser_plane(args.laser_plane)
    signed_laser_plane_distance = true_ground_camera @ laser_normal + laser_d
    model_distance_fit, model_distance_residual = fit_linear(
        true_ground[:, 0], signed_laser_plane_distance, "true_ground_Xg"
    )
    laser["true_ground_Xc_mm"] = true_ground_camera[:, 0]
    laser["true_ground_Yc_mm"] = true_ground_camera[:, 1]
    laser["true_ground_Zc_mm"] = true_ground_camera[:, 2]
    laser["true_ground_Xg_mm"] = true_ground[:, 0]
    laser["true_ground_Yg_mm"] = true_ground[:, 1]
    laser["true_ground_Zg_mm"] = true_ground[:, 2]
    laser["laser_plane_signed_distance_at_true_ground_mm"] = signed_laser_plane_distance
    laser["laser_plane_distance_linear_residual_mm"] = model_distance_residual

    if args.comparison_extrinsics is None:
        raise ValueError("--comparison-extrinsics is required for attribution")
    comparison_document = read_json(args.comparison_extrinsics)
    comparison_transform = load_transform(comparison_document)
    camera_points = laser[["Xc_mm", "Yc_mm", "Zc_mm"]].to_numpy()
    comparison_ground = transform_points(camera_points, comparison_transform)
    comparison_fit, comparison_residual = fit_linear(
        comparison_ground[:, 0], comparison_ground[:, 2], "Xg"
    )
    laser["comparison_Xg_mm"] = comparison_ground[:, 0]
    laser["comparison_Yg_mm"] = comparison_ground[:, 1]
    laser["comparison_Zg_mm"] = comparison_ground[:, 2]
    laser["comparison_linear_residual_mm"] = comparison_residual

    board_best_angle = abs(float(board_projected_fit["slope_angle_deg"]))
    board_upper_angle = max(abs(value) for value in per_frame_projected_angles)
    mismatch_angle = abs(float(mismatch_fit["slope_angle_deg"]))
    minimum_laser_fraction = max(0.0, 1.0 - board_upper_angle / mismatch_angle) * 100.0
    verdict = (
        "一阶倾斜主要来自当前激光平面模型与实测激光片不一致，而不是 board-only 外参姿态。"
    )

    metrics: dict[str, Any] = {
        "inputs": {
            "intrinsics": str(args.intrinsics.resolve()),
            "board_extrinsics": str(args.board_extrinsics.resolve()),
            "board_fit_dir": str(args.board_fit_dir.resolve()),
            "board_validation_dir": str(args.board_validation_dir.resolve()),
            "laser_plane": str(args.laser_plane.resolve()),
            "laser_points": str(args.laser_points.resolve()),
            "comparison_extrinsics": str(args.comparison_extrinsics.resolve()),
        },
        "board_fit": {
            "frame_count": int(fit_points["frame"].nunique()),
            "point_count": int(len(fit_points)),
            "plane": fit_plane,
            "per_frame": fit_observations,
            "note": "These frames defined the board-only extrinsics and are not independent validation.",
        },
        "board_validation": {
            "frame_count": int(validation_points["frame"].nunique()),
            "point_count": int(len(validation_points)),
            "plane": validation_plane,
            "per_frame": validation_observations,
            "note": "These checkerboard frames were not used to fit the board-only extrinsics.",
        },
        "laser_empty_board": {
            "frame": args.laser_points.parent.name,
            "point_count": int(len(laser)),
            "u_min_px": float(args.u_min),
            "u_max_px": float(args.u_max),
            "selection": "Central checkerboard pattern surface in an empty-board laser frame.",
            "linear_Zg_vs_Xg": laser_linear,
        },
        "laser_minus_board_validation_plane": mismatch_fit,
        "laser_model_check": {
            "definition": (
                "Intersect each observed laser pixel ray with the board-only checkerboard plane, "
                "then measure signed orthogonal distance to the calibrated laser plane."
            ),
            "true_ground_Zg_max_abs_mm": float(np.max(np.abs(true_ground[:, 2]))),
            "signed_distance_to_laser_plane": model_distance_fit,
        },
        "comparison_extrinsics": {
            "path": str(args.comparison_extrinsics.resolve()),
            "linear_Zg_vs_Xg": comparison_fit,
            "angle_change_to_board_only_deg": float(
                laser_linear["slope_angle_deg"] - comparison_fit["slope_angle_deg"]
            ),
        },
        "first_order_tilt_attribution": {
            "board_projected_slope_angle_deg": float(board_projected_fit["slope_angle_deg"]),
            "board_projected_per_frame_angles_deg": per_frame_projected_angles,
            "board_projected_per_frame_max_abs_angle_deg": board_upper_angle,
            "laser_minus_board_abs_angle_deg": mismatch_angle,
            "best_estimate_external_fraction_percent": 100.0 * board_best_angle / mismatch_angle,
            "minimum_laser_model_fraction_percent": minimum_laser_fraction,
            "verdict": verdict,
        },
    }

    board_points.to_csv(args.output_dir / "board_points.csv", index=False, float_format="%.12g")
    laser.to_csv(args.output_dir / "laser_ground_points.csv", index=False, float_format="%.12g")
    pd.DataFrame(fit_observations + validation_observations).to_json(
        args.output_dir / "board_per_frame_planes.json",
        orient="records",
        force_ascii=False,
        indent=2,
    )
    (args.output_dir / "summary_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "board_vs_laser_tilt_report.md").write_text(
        build_report(metrics), encoding="utf-8"
    )

    plot_board_validation(
        validation_points, validation_plane, args.output_dir / "board_validation_residual.png"
    )
    plot_laser_selection(laser_all, laser, args.output_dir / "laser_selection_profile.png")
    plot_laser_trend(laser, laser_linear, args.output_dir / "laser_tilt_and_residual.png")
    plot_laser_plane_distance(
        laser, model_distance_fit, args.output_dir / "laser_plane_model_distance.png"
    )
    plot_attribution(metrics, args.output_dir / "tilt_attribution.png")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
