#!/usr/bin/env python3
"""Plot paper-style laser-line extraction and pixel-domain curvature diagnostics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import pandas as pd
import yaml

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import ConnectionPatch, Rectangle

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None


REQUIRED_COLUMNS = {
    "image_id",
    "split",
    "x_px",
    "y_px",
    "status",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Draw a paper-style laser stripe figure and residual plots to judge "
            "whether extracted laser centers are already curved in image space."
        )
    )
    parser.add_argument("--centres-csv", required=True, type=Path)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--intrinsics-yaml",
        type=Path,
        default=None,
        help=(
            "Camera intrinsics YAML. When set, the script also writes undistorted-pixel "
            "centerline/residual figures and a raw-vs-undistorted comparison CSV."
        ),
    )
    parser.add_argument(
        "--image-pattern",
        default="laser {id:03d}.tif",
        help="Pattern relative to image-dir. Use {id} for image id.",
    )
    parser.add_argument("--split", default="train", choices=["train", "validation", "all"])
    parser.add_argument("--poses", nargs="*", type=int, default=None)
    parser.add_argument("--example-pose", type=int, default=None)
    parser.add_argument("--smooth-window", type=int, default=51)
    parser.add_argument(
        "--extractor",
        choices=("csv", "measurement-tool", "shared"),
        default="csv",
        help=(
            "Point source. csv reads centres-csv; measurement-tool uses the same "
            "default Steger backend as reconstruct_ground_pointcloud_cloudcompare_v4.py; "
            "shared uses calibration/src/steger_laser_center.py."
        ),
    )
    parser.add_argument(
        "--right-panel",
        choices=("centerline", "residual"),
        default="centerline",
        help="Content of the right subplot in the paper-style example figure.",
    )
    parser.add_argument("--image-point-size", type=float, default=2.2)
    parser.add_argument("--plot-point-size", type=float, default=5.0)
    parser.add_argument("--image-line-width", type=float, default=0.8)
    parser.add_argument("--plot-line-width", type=float, default=0.8)
    parser.add_argument("--roi-pad-px", type=float, default=90.0)
    parser.add_argument(
        "--right-y-pad-px",
        type=float,
        default=5.0,
        help="Y padding for right centerline panel. Smaller values make slight curvature easier to see.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def configure_matplotlib() -> None:
    candidates = [
        "Arial",
        "Segoe UI",
        "Microsoft YaHei",
        "SimHei",
        "DejaVu Sans",
    ]
    chosen = "DejaVu Sans"
    for family in candidates:
        try:
            path = font_manager.findfont(family, fallback_to_default=False)
            if path and Path(path).exists():
                chosen = family
                break
        except Exception:
            continue

    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [chosen, "DejaVu Sans"],
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8,
        }
    )


def read_centres(path: Path, split: str, poses: Sequence[int] | None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError("centres csv missing columns: " + ", ".join(sorted(missing)))

    for column in ("image_id", "x_px", "y_px"):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    mask = df["status"].astype(str).eq("accepted_final")
    mask &= df["x_px"].notna() & df["y_px"].notna()
    if split != "all":
        mask &= df["split"].astype(str).str.lower().eq(split)
    if poses:
        mask &= df["image_id"].isin(poses)

    selected = df.loc[mask].copy()
    if selected.empty:
        raise ValueError("No accepted laser centers after filtering.")

    selected["image_id"] = selected["image_id"].astype(int)
    selected = selected.sort_values(["image_id", "x_px"]).reset_index(drop=True)
    return selected


def load_intrinsics(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
    try:
        camera_matrix = np.asarray(data["camera_matrix"], dtype=np.float64).reshape(3, 3)
        dist_coeffs = np.asarray(data["dist_coeffs"], dtype=np.float64).reshape(-1)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Cannot read camera_matrix/dist_coeffs from {path}") from exc
    return camera_matrix, dist_coeffs


def add_undistorted_pixel_columns(
    points: pd.DataFrame,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> pd.DataFrame:
    result = points.copy()
    pixels = result[["x_px", "y_px"]].to_numpy(dtype=np.float64)
    undistorted = cv2.undistortPoints(
        pixels.reshape(-1, 1, 2),
        camera_matrix,
        dist_coeffs,
        P=camera_matrix,
    ).reshape(-1, 2)
    result["x_undist_px"] = undistorted[:, 0]
    result["y_undist_px"] = undistorted[:, 1]
    return result


def _read_gray_image(path: Path) -> np.ndarray:
    if Image is None:
        raise ImportError("Pillow is required to read image files.")
    image = Image.open(path)
    array = np.asarray(image)
    if array.ndim == 3:
        array = array[..., 0]
    return np.ascontiguousarray(array)


def _load_measurement_tool_steger() -> tuple[object, dict]:
    import yaml

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[1]
    tool_root = project_root / "linelaser_tool" / "laser_measurement_tool"
    if str(tool_root) not in sys.path:
        sys.path.insert(0, str(tool_root))

    from laser.backends import steger_backend

    config_path = tool_root / "configs" / "measure_tool.yaml"
    with config_path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream) or {}
    options = dict((document.get("extraction") or {}).get("steger") or {})
    if not options:
        raise ValueError(f"{config_path} has no extraction.steger options.")
    return steger_backend, options


def _load_shared_steger() -> tuple[object, dict]:
    import steger_laser_center as steger

    settings = steger.StegerSettings()

    def backend(gray: np.ndarray, _options: dict) -> np.ndarray:
        extracted = steger.extract_steger_columns(gray, settings)
        return extracted.pixels

    return backend, {"settings": settings}


def extract_centres_from_images(
    image_dir: Path,
    image_pattern: str,
    image_ids: Sequence[int],
    extractor: str,
    split: str,
) -> pd.DataFrame:
    if extractor == "measurement-tool":
        backend, options = _load_measurement_tool_steger()
    elif extractor == "shared":
        backend, options = _load_shared_steger()
    else:
        raise ValueError(f"Unsupported extractor: {extractor}")

    rows: list[dict[str, float | int | str]] = []
    for image_id in image_ids:
        image_path = resolve_image_path(image_dir, image_pattern, image_id)
        if image_path is None:
            raise FileNotFoundError(f"Cannot find laser image for pose {image_id} in {image_dir}")
        gray = _read_gray_image(image_path)
        pixels = np.asarray(backend(gray, options), dtype=np.float64).reshape(-1, 2)
        for x_px, y_px in pixels:
            rows.append(
                {
                    "image_id": int(image_id),
                    "split": split,
                    "x_px": float(x_px),
                    "y_px": float(y_px),
                    "status": "accepted_final",
                }
            )

    if not rows:
        raise ValueError(f"{extractor} extracted no laser centers.")
    return pd.DataFrame(rows).sort_values(["image_id", "x_px"]).reset_index(drop=True)


def resolve_image_path(image_dir: Path, pattern: str, image_id: int) -> Path | None:
    candidates = [
        image_dir / pattern.format(id=image_id),
        image_dir / f"laser {image_id:03d}.tif",
        image_dir / f"laser {image_id}.tif",
        image_dir / f"pose_{image_id}.tif",
        image_dir / f"pose_{image_id:03d}.tif",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def smooth(values: np.ndarray, requested_window: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if requested_window <= 1 or values.size < 3:
        return values.copy()

    window = min(int(requested_window), values.size)
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return values.copy()

    kernel = np.ones(window, dtype=float) / float(window)
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def line_fit_metrics(
    group: pd.DataFrame,
    smooth_window: int,
    x_col: str = "x_px",
    y_col: str = "y_px",
) -> dict[str, float]:
    x = group[x_col].to_numpy(dtype=float)
    y = group[y_col].to_numpy(dtype=float)
    x_centered = x - float(np.mean(x))

    line_coef = np.polyfit(x_centered, y, deg=1)
    line_y = np.polyval(line_coef, x_centered)
    residual = y - line_y
    residual_smooth = smooth(residual, smooth_window)

    quad_coef = np.polyfit(x_centered, y, deg=2)
    quad_y = np.polyval(quad_coef, x_centered)
    quad_residual = y - quad_y

    return {
        "point_count": int(len(group)),
        "x_min_px": float(np.min(x)),
        "x_max_px": float(np.max(x)),
        "y_min_px": float(np.min(y)),
        "y_max_px": float(np.max(y)),
        "y_span_px": float(np.max(y) - np.min(y)),
        "line_slope_px_per_px": float(line_coef[0]),
        "line_intercept_y_px": float(line_coef[1]),
        "line_residual_rms_px": float(np.sqrt(np.mean(residual**2))),
        "line_residual_abs_p95_px": float(np.percentile(np.abs(residual), 95)),
        "line_residual_abs_max_px": float(np.max(np.abs(residual))),
        "smooth_residual_min_px": float(np.min(residual_smooth)),
        "smooth_residual_max_px": float(np.max(residual_smooth)),
        "curvature_span_px": float(np.max(residual_smooth) - np.min(residual_smooth)),
        "quadratic_a_px_per_px2": float(quad_coef[0]),
        "quadratic_residual_rms_px": float(np.sqrt(np.mean(quad_residual**2))),
    }


def save_curvature_summary(
    points: pd.DataFrame,
    output_dir: Path,
    smooth_window: int,
    output_name: str = "laser_line_curvature_summary.csv",
    x_col: str = "x_px",
    y_col: str = "y_px",
) -> pd.DataFrame:
    rows = []
    for image_id, group in points.groupby("image_id", sort=True):
        metrics = line_fit_metrics(group, smooth_window, x_col=x_col, y_col=y_col)
        metrics["image_id"] = int(image_id)
        rows.append(metrics)

    summary = pd.DataFrame(rows)
    front = ["image_id", "point_count", "curvature_span_px", "line_residual_rms_px"]
    summary = summary[front + [c for c in summary.columns if c not in front]]
    summary.to_csv(output_dir / output_name, index=False, encoding="utf-8-sig")
    return summary


def load_image(path: Path) -> np.ndarray:
    if Image is None:
        raise ImportError("Pillow is required to read image files.")
    image = Image.open(path)
    array = np.asarray(image)
    if array.ndim == 3:
        array = array[..., 0]
    return array.astype(float)


def contrast_limits(image: np.ndarray) -> tuple[float, float]:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return 0.0, 1.0
    low = float(np.percentile(finite, 1.0))
    high = float(np.percentile(finite, 99.7))
    if high <= low:
        high = low + 1.0
    return low, high


def padded_bounds(
    group: pd.DataFrame,
    image_shape: tuple[int, int],
    x_pad: float,
    y_pad: float,
    x_col: str = "x_px",
    y_col: str = "y_px",
) -> tuple[float, float, float, float]:
    height, width = image_shape
    x_min = max(0.0, float(group[x_col].min()) - x_pad)
    x_max = min(float(width - 1), float(group[x_col].max()) + x_pad)
    y_min = max(0.0, float(group[y_col].min()) - y_pad)
    y_max = min(float(height - 1), float(group[y_col].max()) + y_pad)
    return x_min, x_max, y_min, y_max


def draw_example_figure(
    points: pd.DataFrame,
    image_dir: Path,
    image_pattern: str,
    output_dir: Path,
    example_pose: int,
    right_panel: str,
    image_point_size: float,
    plot_point_size: float,
    image_line_width: float,
    plot_line_width: float,
    roi_pad_px: float,
    right_y_pad_px: float,
    smooth_window: int,
    dpi: int,
    x_col: str = "x_px",
    y_col: str = "y_px",
    coordinate_label: str = "raw pixel",
    output_suffix: str = "",
    camera_matrix: np.ndarray | None = None,
    dist_coeffs: np.ndarray | None = None,
) -> Path:
    group = points.loc[points["image_id"].eq(example_pose)].sort_values(x_col)
    if group.empty:
        raise ValueError(f"Example pose {example_pose} has no accepted points.")

    image_path = resolve_image_path(image_dir, image_pattern, example_pose)
    if image_path is None:
        raise FileNotFoundError(f"Cannot find laser image for pose {example_pose} in {image_dir}")

    image = load_image(image_path)
    if camera_matrix is not None and dist_coeffs is not None:
        image = cv2.undistort(image.astype(np.float32), camera_matrix, dist_coeffs)
    vmin, vmax = contrast_limits(image)
    roi_x_min, roi_x_max, roi_y_min, roi_y_max = padded_bounds(
        group,
        image.shape,
        x_pad=roi_pad_px,
        y_pad=roi_pad_px,
        x_col=x_col,
        y_col=y_col,
    )
    x_min, x_max, y_min, y_max = padded_bounds(
        group,
        image.shape,
        x_pad=roi_pad_px,
        y_pad=right_y_pad_px,
        x_col=x_col,
        y_col=y_col,
    )

    fig = plt.figure(figsize=(9.0, 3.7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.18)
    ax_img = fig.add_subplot(gs[0, 0])
    ax_line = fig.add_subplot(gs[0, 1])

    ax_img.imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
    ax_img.plot(group[x_col], group[y_col], color="#00c8ff", linewidth=image_line_width, alpha=0.9)
    ax_img.scatter(
        group[x_col],
        group[y_col],
        s=image_point_size,
        c="#fff06a",
        edgecolors="none",
        alpha=0.9,
    )
    rect = Rectangle(
        (roi_x_min, roi_y_min),
        roi_x_max - roi_x_min,
        roi_y_max - roi_y_min,
        fill=False,
        edgecolor="#e23b3b",
        linewidth=1.5,
    )
    ax_img.add_patch(rect)
    ax_img.set_xlim(0, image.shape[1] - 1)
    ax_img.set_ylim(image.shape[0] - 1, 0)
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    ax_img.set_title("Laser stripe on checkerboard")

    if right_panel == "centerline":
        ax_line.scatter(
            group[x_col],
            group[y_col],
            s=plot_point_size,
            color="#214eea",
            alpha=0.65,
            label="Stripe points",
        )
        ax_line.plot(
            group[x_col],
            group[y_col],
            color="#214eea",
            linewidth=plot_line_width,
            alpha=0.55,
        )
        ax_line.set_xlim(x_min, x_max)
        ax_line.set_ylim(y_min, y_max)
        ax_line.set_xlabel(f"X value / {coordinate_label}")
        ax_line.set_ylabel(f"Y value / {coordinate_label}")
        ax_line.set_title("Extracted centerline")
    else:
        x = group[x_col].to_numpy(dtype=float)
        y = group[y_col].to_numpy(dtype=float)
        x_centered = x - float(np.mean(x))
        line_coef = np.polyfit(x_centered, y, deg=1)
        residual = y - np.polyval(line_coef, x_centered)
        residual_smooth = smooth(residual, smooth_window)
        ax_line.scatter(
            x,
            residual,
            s=plot_point_size,
            color="#214eea",
            alpha=0.28,
            label="Raw residual",
        )
        ax_line.plot(
            x,
            residual_smooth,
            color="#e23b3b",
            linewidth=max(plot_line_width, 1.2),
            label="Smoothed residual",
        )
        ax_line.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        span = float(np.nanmax(residual_smooth) - np.nanmin(residual_smooth))
        ax_line.set_xlim(x_min, x_max)
        ax_line.set_xlabel(f"X value / {coordinate_label}")
        ax_line.set_ylabel(f"Residual / {coordinate_label}")
        ax_line.set_title(f"Residual after line fit, span={span:.2f}px")
    ax_line.grid(True, alpha=0.28)
    ax_line.legend(loc="best", frameon=True)

    for y_corner, ax_y in ((roi_y_min, 1.0), (roi_y_max, 0.0)):
        con = ConnectionPatch(
            xyA=(roi_x_max, y_corner),
            coordsA=ax_img.transData,
            xyB=(0.0, ax_y),
            coordsB=ax_line.transAxes,
            color="#e23b3b",
            linewidth=1.0,
        )
        fig.add_artist(con)

    fig.text(0.27, 0.015, "(a)", ha="center", va="center", fontsize=12)
    fig.text(0.73, 0.015, "(b)", ha="center", va="center", fontsize=12)
    fig.suptitle(
        f"Laser line extraction result, pose {example_pose:03d} ({coordinate_label})",
        y=0.98,
    )
    fig.subplots_adjust(bottom=0.15, top=0.86)

    path = output_dir / f"pose_{example_pose:03d}{output_suffix}_paper_style_laser_line.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def draw_residual_diagnostics(
    points: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
    smooth_window: int,
    dpi: int,
    x_col: str = "x_px",
    y_col: str = "y_px",
    coordinate_label: str = "pixel",
    output_name: str = "laser_line_curvature_residuals.png",
) -> Path:
    poses = summary.sort_values("curvature_span_px", ascending=False)["image_id"].astype(int).tolist()
    top_poses = poses[: min(8, len(poses))]

    fig, axes = plt.subplots(2, 1, figsize=(8.4, 6.4), sharex=False)
    ax_raw, ax_res = axes

    for image_id in top_poses:
        group = points.loc[points["image_id"].eq(image_id)].sort_values(x_col)
        x = group[x_col].to_numpy(dtype=float)
        y = group[y_col].to_numpy(dtype=float)
        x_centered = x - float(np.mean(x))
        line_coef = np.polyfit(x_centered, y, deg=1)
        line_y = np.polyval(line_coef, x_centered)
        residual = y - line_y
        residual_smooth = smooth(residual, smooth_window)
        normalized_x = (x - x.min()) / max(float(x.max() - x.min()), 1.0)

        ax_raw.plot(x, y, linewidth=1.2, label=f"{image_id:03d}")
        ax_res.plot(normalized_x, residual_smooth, linewidth=1.35, label=f"{image_id:03d}")

    ax_raw.set_title("Extracted laser centerlines")
    ax_raw.set_xlabel(f"X value / {coordinate_label}")
    ax_raw.set_ylabel(f"Y value / {coordinate_label}")
    ax_raw.grid(True, alpha=0.25)
    ax_raw.legend(title="pose", ncol=4, loc="best")

    ax_res.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax_res.set_title("Residual after best straight-line fit")
    ax_res.set_xlabel("Normalized X along extracted stripe")
    ax_res.set_ylabel(f"Residual / {coordinate_label}")
    ax_res.grid(True, alpha=0.25)
    ax_res.legend(title="pose", ncol=4, loc="best")

    fig.tight_layout()
    path = output_dir / output_name
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def save_raw_vs_undistorted_comparison(
    raw_summary: pd.DataFrame,
    undistorted_summary: pd.DataFrame,
    output_dir: Path,
) -> Path:
    raw = raw_summary.add_prefix("raw_").rename(columns={"raw_image_id": "image_id"})
    undist = undistorted_summary.add_prefix("undist_").rename(columns={"undist_image_id": "image_id"})
    comparison = raw.merge(undist, on="image_id", how="inner")
    comparison["curvature_span_delta_px"] = (
        comparison["undist_curvature_span_px"] - comparison["raw_curvature_span_px"]
    )
    comparison["curvature_span_ratio"] = (
        comparison["undist_curvature_span_px"] / comparison["raw_curvature_span_px"]
    )
    comparison["line_residual_rms_delta_px"] = (
        comparison["undist_line_residual_rms_px"] - comparison["raw_line_residual_rms_px"]
    )
    path = output_dir / "laser_line_curvature_raw_vs_undistorted.csv"
    comparison.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def draw_raw_vs_undistorted_comparison(
    points: pd.DataFrame,
    output_dir: Path,
    example_pose: int,
    smooth_window: int,
    dpi: int,
) -> Path:
    group = points.loc[points["image_id"].eq(example_pose)].sort_values("x_px")
    if group.empty:
        raise ValueError(f"Example pose {example_pose} has no accepted points.")

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.0))
    axes = axes.reshape(2, 2)
    coordinate_sets = [
        ("Raw pixel", "x_px", "y_px"),
        ("Undistorted pixel", "x_undist_px", "y_undist_px"),
    ]
    for column, (title, x_col, y_col) in enumerate(coordinate_sets):
        x = group[x_col].to_numpy(dtype=float)
        y = group[y_col].to_numpy(dtype=float)
        x_centered = x - float(np.mean(x))
        line = np.polyfit(x_centered, y, deg=1)
        residual = y - np.polyval(line, x_centered)
        residual_smooth = smooth(residual, smooth_window)
        span = float(np.max(residual_smooth) - np.min(residual_smooth))

        ax_line = axes[0, column]
        ax_res = axes[1, column]
        ax_line.scatter(x, y, s=2.0, color="#214eea", alpha=0.55)
        ax_line.plot(x, y, color="#214eea", linewidth=0.8, alpha=0.55)
        ax_line.set_title(f"{title} centerline")
        ax_line.set_xlabel("X / pixel")
        ax_line.set_ylabel("Y / pixel")
        ax_line.grid(True, alpha=0.25)

        ax_res.scatter(x, residual, s=2.0, color="#214eea", alpha=0.22)
        ax_res.plot(x, residual_smooth, color="#e23b3b", linewidth=1.25)
        ax_res.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        ax_res.set_title(f"Residual span={span:.2f}px")
        ax_res.set_xlabel("X / pixel")
        ax_res.set_ylabel("Residual / pixel")
        ax_res.grid(True, alpha=0.25)

    fig.suptitle(f"Raw vs undistorted laser centerline, pose {example_pose:03d}", y=0.99)
    fig.tight_layout()
    path = output_dir / f"pose_{example_pose:03d}_raw_vs_undistorted_curvature.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> int:
    args = parse_args()
    configure_matplotlib()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.extractor == "csv":
        points = read_centres(args.centres_csv, args.split, args.poses)
    else:
        if args.poses:
            image_ids = args.poses
        elif args.example_pose is not None:
            image_ids = [args.example_pose]
        else:
            image_ids = sorted(read_centres(args.centres_csv, args.split, None)["image_id"].unique().tolist())
        points = extract_centres_from_images(
            args.image_dir,
            args.image_pattern,
            image_ids,
            args.extractor,
            args.split,
        )
    camera_matrix = None
    dist_coeffs = None
    if args.intrinsics_yaml is not None:
        camera_matrix, dist_coeffs = load_intrinsics(args.intrinsics_yaml)
        points = add_undistorted_pixel_columns(points, camera_matrix, dist_coeffs)

    summary = save_curvature_summary(points, args.output_dir, args.smooth_window)

    if args.example_pose is None:
        example_pose = int(
            summary.sort_values("curvature_span_px", ascending=False).iloc[0]["image_id"]
        )
    else:
        example_pose = args.example_pose

    example_path = draw_example_figure(
        points,
        args.image_dir,
        args.image_pattern,
        args.output_dir,
        example_pose,
        args.right_panel,
        args.image_point_size,
        args.plot_point_size,
        args.image_line_width,
        args.plot_line_width,
        args.roi_pad_px,
        args.right_y_pad_px,
        args.smooth_window,
        args.dpi,
    )
    residual_path = draw_residual_diagnostics(
        points,
        summary,
        args.output_dir,
        args.smooth_window,
        args.dpi,
    )

    summary_path = args.output_dir / "laser_line_curvature_summary.csv"
    undistorted_example_path = None
    undistorted_residual_path = None
    comparison_csv_path = None
    comparison_figure_path = None
    undistorted_summary = None
    if camera_matrix is not None and dist_coeffs is not None:
        undistorted_summary = save_curvature_summary(
            points,
            args.output_dir,
            args.smooth_window,
            output_name="laser_line_curvature_summary_undistorted.csv",
            x_col="x_undist_px",
            y_col="y_undist_px",
        )
        undistorted_example_path = draw_example_figure(
            points,
            args.image_dir,
            args.image_pattern,
            args.output_dir,
            example_pose,
            args.right_panel,
            args.image_point_size,
            args.plot_point_size,
            args.image_line_width,
            args.plot_line_width,
            args.roi_pad_px,
            args.right_y_pad_px,
            args.smooth_window,
            args.dpi,
            x_col="x_undist_px",
            y_col="y_undist_px",
            coordinate_label="undistorted pixel",
            output_suffix="_undistorted",
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
        )
        undistorted_residual_path = draw_residual_diagnostics(
            points,
            undistorted_summary,
            args.output_dir,
            args.smooth_window,
            args.dpi,
            x_col="x_undist_px",
            y_col="y_undist_px",
            coordinate_label="undistorted pixel",
            output_name="laser_line_curvature_residuals_undistorted.png",
        )
        comparison_csv_path = save_raw_vs_undistorted_comparison(
            summary,
            undistorted_summary,
            args.output_dir,
        )
        comparison_figure_path = draw_raw_vs_undistorted_comparison(
            points,
            args.output_dir,
            example_pose,
            args.smooth_window,
            args.dpi,
        )

    top = summary.sort_values("curvature_span_px", ascending=False).head(8)
    print(f"Example figure: {example_path}")
    print(f"Residual figure: {residual_path}")
    print(f"Summary CSV: {summary_path}")
    if undistorted_example_path is not None:
        print(f"Undistorted example figure: {undistorted_example_path}")
        print(f"Undistorted residual figure: {undistorted_residual_path}")
        print(f"Raw-vs-undistorted figure: {comparison_figure_path}")
        print(f"Raw-vs-undistorted CSV: {comparison_csv_path}")
    print("Top curvature spans:")
    for row in top.itertuples(index=False):
        print(
            f"  pose {int(row.image_id):03d}: "
            f"span={row.curvature_span_px:.3f}px, "
            f"rms={row.line_residual_rms_px:.3f}px, "
            f"points={int(row.point_count)}"
        )
    if undistorted_summary is not None:
        comparison = pd.read_csv(comparison_csv_path)
        print("Raw vs undistorted curvature spans:")
        for row in comparison.sort_values("raw_curvature_span_px", ascending=False).head(8).itertuples(index=False):
            print(
                f"  pose {int(row.image_id):03d}: "
                f"raw={row.raw_curvature_span_px:.3f}px, "
                f"undist={row.undist_curvature_span_px:.3f}px, "
                f"ratio={row.curvature_span_ratio:.3f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
