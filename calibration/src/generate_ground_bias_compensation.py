#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a column-wise ground-bias lookup table (LUT) from repeated flat-ground
line-laser scans, then visualize the Z profile and point cloud before/after
compensation.

The script supports two workflows:

1. Reconstructed point files (recommended and directly runnable)
   Each frame must provide image column u and ground-frame coordinates Xg/Yg/Zg.
   Supported formats: CSV/TXT, NPY, NPZ, and ASCII PCD.

2. Raw image files through a user adapter
   Pass --adapter path/to/reconstruction_adapter.py. The adapter must expose
   reconstruct_frame(image_path) and return u/x/y/z arrays.

Outputs:
- ground_bias_table.npy
- ground_bias_table.csv
- z_profile_before_after.png
- pointcloud_before_after.png
- ground_cloud_before.npz / ground_cloud_after.npz
- ground_cloud_before.ply / ground_cloud_after.ply (unless disabled)
- compensation_metrics.json
- ground_flatness_compensation_report.md

The default correction preserves the best-fit linear ground trend and removes
only the repeatable column-wise residual:

    bias(u) = mean_Z(u) - linear_trend(u)
    Z_corrected = Z_raw - bias(u)

This is an engineering compensation for the current fixed mechanical state. It
must be rebuilt after changing camera/laser pose, focus, aperture, baseline, or
working distance.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fit_ground_plane_from_pointcloud import fit_z_plane_robust


SUPPORTED_POINT_SUFFIXES = {".csv", ".txt", ".npy", ".npz", ".pcd"}
SUPPORTED_IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".bmp", ".jpg", ".jpeg"}

COLUMN_ALIASES = {
    "u": ("u", "col", "column", "pixel_u", "image_u", "img_u", "laser_u"),
    "v": ("v", "row", "pixel_v", "image_v", "img_v", "laser_v"),
    "x": ("x", "xg", "xg_mm", "ground_x", "robot_x", "world_x"),
    "y": ("y", "yg", "yg_mm", "ground_y", "robot_y", "world_y"),
    "z": ("z", "zg", "zg_mm", "ground_z", "robot_z", "world_z", "height"),
}


@dataclass
class FramePoints:
    path: Path
    u: np.ndarray
    xyz: np.ndarray
    compensation_axis: str = "u"

    def validate(self) -> "FramePoints":
        if self.compensation_axis not in {"u", "v"}:
            raise ValueError(
                f"{self.path}: compensation_axis must be 'u' or 'v', "
                f"got {self.compensation_axis!r}"
            )
        self.u = np.asarray(self.u, dtype=np.float64).reshape(-1)
        self.xyz = np.asarray(self.xyz, dtype=np.float64)
        if self.xyz.ndim != 2 or self.xyz.shape[1] != 3:
            raise ValueError(f"{self.path}: xyz must have shape (N, 3), got {self.xyz.shape}")
        if self.u.shape[0] != self.xyz.shape[0]:
            raise ValueError(
                f"{self.path}: u length {self.u.shape[0]} != xyz rows {self.xyz.shape[0]}"
            )
        finite = np.isfinite(self.u) & np.all(np.isfinite(self.xyz), axis=1)
        self.u = self.u[finite]
        self.xyz = self.xyz[finite]
        if self.u.size == 0:
            raise ValueError(f"{self.path}: no finite points remain")
        return self


@dataclass
class ProfileGrid:
    columns: np.ndarray
    x_by_frame: np.ndarray
    z_by_frame: np.ndarray


@dataclass
class BiasTable:
    columns: np.ndarray
    xg_mm: np.ndarray
    raw_mean_z_mm: np.ndarray
    trend_z_mm: np.ndarray
    bias_mm: np.ndarray
    sample_count: np.ndarray
    repeatability_sigma_mm: np.ndarray
    metadata: dict[str, Any]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and validate a column-wise Z compensation table from repeated flat-ground scans.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="Folder containing 31 frames or reconstructed point files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Folder for LUT, figures, point clouds, and report.")
    parser.add_argument(
        "--glob",
        dest="glob_pattern",
        default="*",
        help="Input filename glob, e.g. '*.csv', '*.npz', or '*.tif'.",
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=None,
        help="Python adapter for raw images. It must define reconstruct_frame(image_path).",
    )
    parser.add_argument(
        "--compensation-axis",
        choices=("u", "v"),
        default="u",
        help="Image coordinate used to build and apply the ground-bias LUT.",
    )
    parser.add_argument(
        "--column-order",
        default="u,x,y,z",
        help="Column order for headerless TXT/CSV or plain NPY arrays.",
    )
    parser.add_argument("--column-bin-width", type=float, default=1.0, help="Image-column bin width in pixels.")
    parser.add_argument("--min-samples-per-column", type=int, default=5, help="Minimum frame samples needed to keep a LUT column.")
    parser.add_argument(
        "--aggregate",
        choices=("mean", "median"),
        default="mean",
        help="Across-frame aggregation used to form the reference profile.",
    )
    parser.add_argument(
        "--trend",
        choices=("linear", "constant", "none"),
        default="linear",
        help="Trend retained after correction. 'linear' removes only curvature; 'constant' flattens tilt; 'none' subtracts the full mean profile.",
    )
    parser.add_argument(
        "--residual-mode",
        choices=("aggregate-trend", "per-frame-plane"),
        default="aggregate-trend",
        help=(
            "Residual definition used to build the LUT. The legacy default fits one "
            "trend after across-frame aggregation; per-frame-plane robustly fits "
            "Zg=a*Xg+b*Yg+c in each frame before aggregation."
        ),
    )
    parser.add_argument(
        "--plane-fit-mad-threshold",
        type=float,
        default=3.5,
        help="MAD rejection threshold used inside each per-frame plane fit.",
    )
    parser.add_argument(
        "--plane-fit-max-iterations",
        type=int,
        default=8,
        help="Maximum robust iterations for each per-frame plane fit.",
    )
    parser.add_argument(
        "--mad-threshold",
        type=float,
        default=3.5,
        help="Per-column MAD outlier threshold across frames. Set <=0 to disable.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="Odd moving-average window applied to the LUT. 1 disables smoothing.",
    )
    parser.add_argument(
        "--validation-count",
        type=int,
        default=0,
        help="Reserve the last N frames for validation. Use 0 to build/evaluate with all frames.",
    )
    parser.add_argument("--u-min", type=float, default=None, help="Optional minimum image column to keep.")
    parser.add_argument("--u-max", type=float, default=None, help="Optional maximum image column to keep.")
    parser.add_argument("--x-min", type=float, default=None, help="Optional minimum Xg in mm to keep.")
    parser.add_argument("--x-max", type=float, default=None, help="Optional maximum Xg in mm to keep.")
    parser.add_argument("--y-min", type=float, default=None, help="Optional minimum Yg in mm to keep.")
    parser.add_argument("--y-max", type=float, default=None, help="Optional maximum Yg in mm to keep.")
    parser.add_argument("--z-min", type=float, default=None, help="Optional minimum Zg in mm to keep.")
    parser.add_argument("--z-max", type=float, default=None, help="Optional maximum Zg in mm to keep.")
    parser.add_argument(
        "--max-plot-points",
        type=int,
        default=120000,
        help="Maximum points sampled for the 3D comparison figure.",
    )
    parser.add_argument(
        "--no-ply",
        action="store_true",
        help="Do not export ASCII PLY point clouds.",
    )
    parser.add_argument("--dpi", type=int, default=240, help="Figure resolution.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for point-cloud plot sampling.")
    return parser.parse_args(argv)


def discover_inputs(input_dir: Path, pattern: str) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    files = sorted(p for p in input_dir.glob(pattern) if p.is_file())
    if not files:
        raise FileNotFoundError(f"No files matched {pattern!r} in {input_dir}")
    return files


def normalize_key(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def resolve_named_array(mapping: Mapping[str, Any], canonical: str) -> np.ndarray:
    normalized = {normalize_key(str(k)): k for k in mapping.keys()}
    for alias in COLUMN_ALIASES[canonical]:
        key = normalized.get(normalize_key(alias))
        if key is not None:
            return np.asarray(mapping[key])
    raise KeyError(f"Missing '{canonical}' field. Available fields: {list(mapping.keys())}")


def parse_column_order(text: str, compensation_axis: str = "u") -> list[str]:
    order = [normalize_key(x) for x in text.split(",") if x.strip()]
    expected = {compensation_axis, "x", "y", "z"}
    if set(order) != expected or len(order) != 4:
        names = ",".join((compensation_axis, "x", "y", "z"))
        raise ValueError(
            f"--column-order must contain {names} exactly once for "
            f"--compensation-axis {compensation_axis}."
        )
    return order


def frame_from_mapping(
    path: Path, mapping: Mapping[str, Any], compensation_axis: str = "u"
) -> FramePoints:
    coordinate = resolve_named_array(mapping, compensation_axis)
    x = resolve_named_array(mapping, "x")
    y = resolve_named_array(mapping, "y")
    z = resolve_named_array(mapping, "z")
    xyz = np.column_stack((x, y, z))
    return FramePoints(
        path=path,
        u=coordinate,
        xyz=xyz,
        compensation_axis=compensation_axis,
    ).validate()


def frame_from_plain_array(
    path: Path,
    array: np.ndarray,
    order: Sequence[str],
    compensation_axis: str = "u",
) -> FramePoints:
    arr = np.asarray(array)
    if arr.ndim != 2 or arr.shape[1] < 4:
        raise ValueError(f"{path}: expected a 2-D array with at least four columns, got {arr.shape}")
    indices = {
        name: order.index(name) for name in (compensation_axis, "x", "y", "z")
    }
    return FramePoints(
        path=path,
        u=arr[:, indices[compensation_axis]],
        xyz=arr[:, [indices["x"], indices["y"], indices["z"]]],
        compensation_axis=compensation_axis,
    ).validate()


def has_header(first_line: str) -> bool:
    tokens = [t.strip() for t in first_line.replace(";", ",").split(",")]
    if len(tokens) <= 1:
        tokens = first_line.split()
    for token in tokens:
        try:
            float(token)
        except ValueError:
            return True
    return False


def load_csv_or_txt(
    path: Path, order: Sequence[str], compensation_axis: str = "u"
) -> FramePoints:
    first_line = path.read_text(encoding="utf-8-sig", errors="ignore").splitlines()[0]
    delimiter = "," if "," in first_line else None
    if has_header(first_line):
        data = np.genfromtxt(path, names=True, delimiter=delimiter, encoding="utf-8-sig")
        if data.dtype.names is None:
            raise ValueError(f"{path}: failed to parse header")
        mapping = {name: data[name] for name in data.dtype.names}
        return frame_from_mapping(path, mapping, compensation_axis)
    array = np.loadtxt(path, delimiter=delimiter)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    return frame_from_plain_array(path, array, order, compensation_axis)


def load_npy(
    path: Path, order: Sequence[str], compensation_axis: str = "u"
) -> FramePoints:
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.dtype.names:
        mapping = {name: data[name] for name in data.dtype.names}
        return frame_from_mapping(path, mapping, compensation_axis)
    if isinstance(data, np.ndarray) and data.dtype == object and data.shape == ():
        item = data.item()
        if isinstance(item, Mapping):
            return frame_from_mapping(path, item, compensation_axis)
    return frame_from_plain_array(path, np.asarray(data), order, compensation_axis)


def load_npz(
    path: Path, order: Sequence[str], compensation_axis: str = "u"
) -> FramePoints:
    with np.load(path, allow_pickle=True) as data:
        keys = list(data.keys())
        normalized = {normalize_key(k): k for k in keys}
        if all(
            any(normalize_key(alias) in normalized for alias in COLUMN_ALIASES[c])
            for c in (compensation_axis, "x", "y", "z")
        ):
            mapping = {k: data[k] for k in keys}
            return frame_from_mapping(path, mapping, compensation_axis)
        if "points" in data:
            points = np.asarray(data["points"])
            if compensation_axis in data:
                coordinate = np.asarray(data[compensation_axis])
                if points.ndim != 2 or points.shape[1] < 3:
                    raise ValueError(f"{path}: 'points' must be Nx3 or wider")
                return FramePoints(
                    path=path,
                    u=coordinate,
                    xyz=points[:, :3],
                    compensation_axis=compensation_axis,
                ).validate()
            return frame_from_plain_array(path, points, order, compensation_axis)
        if len(keys) == 1:
            return frame_from_plain_array(
                path, np.asarray(data[keys[0]]), order, compensation_axis
            )
        raise ValueError(
            f"{path}: could not identify {compensation_axis}/x/y/z arrays. Keys: {keys}"
        )


def load_ascii_pcd(path: Path, compensation_axis: str = "u") -> FramePoints:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    fields: list[str] | None = None
    data_start = None
    data_mode = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith("FIELDS "):
            fields = stripped.split()[1:]
        elif upper.startswith("DATA "):
            data_mode = stripped.split()[1].lower()
            data_start = i + 1
            break
    if fields is None or data_start is None:
        raise ValueError(f"{path}: invalid PCD header")
    if data_mode != "ascii":
        raise ValueError(f"{path}: only ASCII PCD is supported; DATA={data_mode}")
    array = np.loadtxt(lines[data_start:])
    if array.ndim == 1:
        array = array.reshape(1, -1)
    mapping = {field: array[:, i] for i, field in enumerate(fields)}
    return frame_from_mapping(path, mapping, compensation_axis)


def load_adapter(adapter_path: Path) -> Callable[..., Any]:
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter does not exist: {adapter_path}")
    spec = importlib.util.spec_from_file_location("ground_reconstruction_adapter", adapter_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load adapter: {adapter_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, "reconstruct_frame", None)
    if not callable(fn):
        raise AttributeError(f"Adapter {adapter_path} must define callable reconstruct_frame(image_path)")
    return fn


def frame_from_adapter_result(
    path: Path, result: Any, compensation_axis: str = "u"
) -> FramePoints:
    if isinstance(result, FramePoints):
        result.path = path
        return result.validate()
    if isinstance(result, Mapping):
        return frame_from_mapping(path, result, compensation_axis)
    if isinstance(result, tuple) and len(result) == 2:
        coordinate, xyz = result
        return FramePoints(
            path=path,
            u=coordinate,
            xyz=xyz,
            compensation_axis=compensation_axis,
        ).validate()
    array = np.asarray(result)
    return frame_from_plain_array(
        path,
        array,
        (compensation_axis, "x", "y", "z"),
        compensation_axis,
    )


def load_frame(
    path: Path,
    order: Sequence[str],
    adapter_fn: Callable[..., Any] | None,
    compensation_axis: str = "u",
) -> FramePoints:
    suffix = path.suffix.lower()
    if suffix in SUPPORTED_IMAGE_SUFFIXES:
        if adapter_fn is None:
            raise ValueError(
                f"{path}: raw image input requires --adapter. See reconstruction_adapter_template.py."
            )
        return frame_from_adapter_result(path, adapter_fn(path), compensation_axis)
    if suffix == ".csv" or suffix == ".txt":
        return load_csv_or_txt(path, order, compensation_axis)
    if suffix == ".npy":
        return load_npy(path, order, compensation_axis)
    if suffix == ".npz":
        return load_npz(path, order, compensation_axis)
    if suffix == ".pcd":
        return load_ascii_pcd(path, compensation_axis)
    raise ValueError(f"Unsupported input format: {path}")


def filter_frame(frame: FramePoints, args: argparse.Namespace) -> FramePoints:
    u = frame.u
    xyz = frame.xyz
    mask = np.ones(u.shape[0], dtype=bool)
    if args.u_min is not None:
        mask &= u >= args.u_min
    if args.u_max is not None:
        mask &= u <= args.u_max
    for axis, low, high in (
        (0, args.x_min, args.x_max),
        (1, args.y_min, args.y_max),
        (2, args.z_min, args.z_max),
    ):
        if low is not None:
            mask &= xyz[:, axis] >= low
        if high is not None:
            mask &= xyz[:, axis] <= high
    filtered = FramePoints(
        frame.path, u[mask], xyz[mask], frame.compensation_axis
    ).validate()
    return filtered


def bin_frame(frame: FramePoints, bin_width: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if bin_width <= 0:
        raise ValueError("--column-bin-width must be > 0")
    bins = np.round(frame.u / bin_width) * bin_width
    unique = np.unique(bins)
    x_values = np.full(unique.shape, np.nan, dtype=np.float64)
    z_values = np.full(unique.shape, np.nan, dtype=np.float64)
    for i, col in enumerate(unique):
        m = bins == col
        ground_axis = 0 if frame.compensation_axis == "u" else 1
        x_values[i] = np.median(frame.xyz[m, ground_axis])
        z_values[i] = np.median(frame.xyz[m, 2])
    return unique, x_values, z_values


def build_profile_grid(frames: Sequence[FramePoints], bin_width: float) -> ProfileGrid:
    binned = [bin_frame(frame, bin_width) for frame in frames]
    all_cols = np.unique(np.concatenate([item[0] for item in binned]))
    x_matrix = np.full((len(frames), all_cols.size), np.nan, dtype=np.float64)
    z_matrix = np.full_like(x_matrix, np.nan)
    col_to_idx = {float(c): i for i, c in enumerate(all_cols)}
    for frame_idx, (cols, xs, zs) in enumerate(binned):
        for c, x, z in zip(cols, xs, zs):
            idx = col_to_idx[float(c)]
            x_matrix[frame_idx, idx] = x
            z_matrix[frame_idx, idx] = z
    return ProfileGrid(all_cols, x_matrix, z_matrix)


def frame_plane_residual(
    frame: FramePoints, args: argparse.Namespace
) -> tuple[FramePoints, dict[str, Any]]:
    """Replace a frame's Z values with signed vertical residuals to its own plane."""

    points = pd.DataFrame(
        {
            "Xg_mm": frame.xyz[:, 0],
            "Yg_mm": frame.xyz[:, 1],
            "Zg_mm": frame.xyz[:, 2],
        }
    )
    fit, vertical_residual, _, inlier = fit_z_plane_robust(
        points,
        mad_threshold=float(getattr(args, "plane_fit_mad_threshold", 3.5)),
        max_iterations=int(getattr(args, "plane_fit_max_iterations", 8)),
    )
    residual_xyz = frame.xyz[inlier].copy()
    residual_xyz[:, 2] = vertical_residual[inlier]
    residual_frame = FramePoints(
        path=frame.path,
        u=frame.u[inlier].copy(),
        xyz=residual_xyz,
        compensation_axis=frame.compensation_axis,
    ).validate()
    diagnostic_keys = (
        "a",
        "b",
        "c_mm",
        "design_rank",
        "design_condition_number",
        "centered_XY_aspect_ratio",
        "input_point_count",
        "finite_point_count",
        "inlier_point_count",
        "iteration_count",
    )
    diagnostic = {key: fit[key] for key in diagnostic_keys}
    diagnostic["source_file"] = str(frame.path)
    return residual_frame, diagnostic


def build_residual_profile_grid(
    frames: Sequence[FramePoints], args: argparse.Namespace
) -> tuple[ProfileGrid, list[dict[str, Any]]]:
    residual_mode = str(getattr(args, "residual_mode", "aggregate-trend"))
    if residual_mode == "aggregate-trend":
        return build_profile_grid(frames, args.column_bin_width), []
    if residual_mode != "per-frame-plane":
        raise ValueError(f"Unknown residual mode: {residual_mode!r}")
    residual_frames: list[FramePoints] = []
    diagnostics: list[dict[str, Any]] = []
    for frame in frames:
        residual_frame, diagnostic = frame_plane_residual(frame, args)
        residual_frames.append(residual_frame)
        diagnostics.append(diagnostic)
    return build_profile_grid(residual_frames, args.column_bin_width), diagnostics


def mad_filter(matrix: np.ndarray, threshold: float) -> np.ndarray:
    if threshold <= 0:
        return matrix.copy()
    filtered = matrix.copy()
    med = np.nanmedian(filtered, axis=0)
    abs_dev = np.abs(filtered - med)
    mad = np.nanmedian(abs_dev, axis=0)
    robust_sigma = 1.4826 * mad
    for j in range(filtered.shape[1]):
        if not np.isfinite(robust_sigma[j]) or robust_sigma[j] <= 1e-12:
            continue
        bad = abs_dev[:, j] > threshold * robust_sigma[j]
        filtered[bad, j] = np.nan
    return filtered


def aggregate_matrix(matrix: np.ndarray, method: str) -> np.ndarray:
    if method == "mean":
        return np.nanmean(matrix, axis=0)
    return np.nanmedian(matrix, axis=0)


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    if window % 2 == 0:
        raise ValueError("--smooth-window must be odd")
    if window > values.size:
        raise ValueError("--smooth-window cannot exceed the number of valid LUT columns")
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(padded, kernel, mode="valid")


def fit_trend(coord: np.ndarray, z: np.ndarray, mode: str) -> np.ndarray:
    valid = np.isfinite(coord) & np.isfinite(z)
    if np.count_nonzero(valid) < 2:
        raise ValueError("Not enough valid columns to fit a trend")
    if mode == "none":
        return np.zeros_like(z)
    if mode == "constant":
        return np.full_like(z, np.nanmedian(z[valid]))
    if np.unique(coord[valid]).size < 2:
        raise ValueError("Trend coordinate is degenerate")
    coeff = np.polyfit(coord[valid], z[valid], deg=1)
    return np.polyval(coeff, coord)


def build_bias_table(
    frames: Sequence[FramePoints],
    args: argparse.Namespace,
) -> tuple[BiasTable, ProfileGrid, np.ndarray]:
    residual_mode = str(getattr(args, "residual_mode", "aggregate-trend"))
    grid, plane_fits = build_residual_profile_grid(frames, args)
    z_filtered = mad_filter(grid.z_by_frame, args.mad_threshold)
    x_filtered = mad_filter(grid.x_by_frame, args.mad_threshold)
    counts = np.sum(np.isfinite(z_filtered), axis=0)
    keep = counts >= args.min_samples_per_column
    if np.count_nonzero(keep) < 10:
        raise ValueError(
            f"Only {np.count_nonzero(keep)} columns meet min samples={args.min_samples_per_column}; "
            "check input data, filters, or --column-bin-width."
        )
    columns = grid.columns[keep]
    z_kept = z_filtered[:, keep]
    x_kept = x_filtered[:, keep]
    raw_mean = aggregate_matrix(z_kept, args.aggregate)
    x_mean = np.nanmedian(x_kept, axis=0)
    trend_coord = x_mean.copy()
    if np.count_nonzero(np.isfinite(trend_coord)) < max(2, int(0.8 * trend_coord.size)):
        trend_coord = columns.copy()
    else:
        missing_x = ~np.isfinite(trend_coord)
        if np.any(missing_x):
            trend_coord[missing_x] = np.interp(
                columns[missing_x], columns[~missing_x], trend_coord[~missing_x]
            )
    if residual_mode == "per-frame-plane":
        trend = np.zeros_like(raw_mean)
    else:
        trend = fit_trend(trend_coord, raw_mean, args.trend)
    bias = raw_mean - trend
    bias = moving_average(bias, args.smooth_window)
    repeatability = np.nanstd(z_kept, axis=0, ddof=1)
    compensation_axis = str(getattr(args, "compensation_axis", "u"))
    metadata = {
        "version": 1,
        "compensation_axis": compensation_axis,
        "trend_coordinate": "xg_mm" if compensation_axis == "u" else "yg_mm",
        "trend_mode": args.trend,
        "residual_mode": residual_mode,
        "residual_definition": (
            "signed_vertical_Zg_residual_to_each_frame_plane"
            if residual_mode == "per-frame-plane"
            else "aggregate_profile_minus_retained_trend"
        ),
        "aggregate": args.aggregate,
        "column_bin_width_px": args.column_bin_width,
        "min_samples_per_column": args.min_samples_per_column,
        "mad_threshold": args.mad_threshold,
        "smooth_window": args.smooth_window,
        "build_frame_count": len(frames),
        "source_files": [str(f.path) for f in frames],
    }
    if residual_mode == "per-frame-plane":
        metadata["per_frame_plane_model"] = "Zg_mm = a*Xg_mm + b*Yg_mm + c_mm"
        metadata["plane_fit_method"] = "iterative_mad"
        metadata["plane_fit_mad_threshold"] = float(
            getattr(args, "plane_fit_mad_threshold", 3.5)
        )
        metadata["plane_fit_max_iterations"] = int(
            getattr(args, "plane_fit_max_iterations", 8)
        )
        metadata["per_frame_plane_fits"] = plane_fits
    table = BiasTable(
        columns=columns,
        xg_mm=x_mean,
        raw_mean_z_mm=raw_mean,
        trend_z_mm=trend,
        bias_mm=bias,
        sample_count=counts[keep],
        repeatability_sigma_mm=repeatability,
        metadata=metadata,
    )
    return table, ProfileGrid(columns, x_kept, z_kept), z_kept


def interpolate_bias(coordinate: np.ndarray, table: BiasTable) -> np.ndarray:
    values = np.asarray(coordinate, dtype=np.float64)
    outside = (values < table.columns[0]) | (values > table.columns[-1])
    if np.any(outside):
        axis = str(table.metadata.get("compensation_axis", "u"))
        warnings.warn(
            f"{np.count_nonzero(outside)} point(s) are outside the ground-bias "
            f"{axis} range [{table.columns[0]:g}, {table.columns[-1]:g}] px; "
            "using the nearest endpoint bias",
            RuntimeWarning,
            stacklevel=2,
        )
    return np.interp(
        values,
        table.columns,
        table.bias_mm,
        left=table.bias_mm[0],
        right=table.bias_mm[-1],
    )


def correct_frames(frames: Sequence[FramePoints], table: BiasTable) -> list[FramePoints]:
    corrected: list[FramePoints] = []
    for frame in frames:
        xyz = frame.xyz.copy()
        xyz[:, 2] -= interpolate_bias(frame.u, table)
        corrected.append(
            FramePoints(
                frame.path,
                frame.u.copy(),
                xyz,
                frame.compensation_axis,
            )
        )
    return corrected


def profile_residual(columns: np.ndarray, x_mean: np.ndarray, mean_z: np.ndarray, trend_mode: str) -> tuple[np.ndarray, np.ndarray]:
    coord = x_mean.copy()
    if np.count_nonzero(np.isfinite(coord)) < max(2, int(0.8 * coord.size)):
        coord = columns.copy()
    else:
        missing = ~np.isfinite(coord)
        if np.any(missing):
            coord[missing] = np.interp(columns[missing], columns[~missing], coord[~missing])
    trend = fit_trend(coord, mean_z, trend_mode)
    return mean_z - trend, trend


def evaluate_frames(frames: Sequence[FramePoints], table: BiasTable, args: argparse.Namespace) -> dict[str, Any]:
    residual_mode = str(getattr(args, "residual_mode", "aggregate-trend"))
    before_grid, _ = build_residual_profile_grid(frames, args)
    after_frames = correct_frames(frames, table)
    after_grid, _ = build_residual_profile_grid(after_frames, args)

    common = np.intersect1d(
        np.intersect1d(before_grid.columns, after_grid.columns), table.columns
    )
    if common.size < 10:
        raise ValueError("Evaluation frames overlap too few LUT columns")
    before_idx = np.searchsorted(before_grid.columns, common)
    after_idx = np.searchsorted(after_grid.columns, common)
    table_idx = np.searchsorted(table.columns, common)

    before_matrix = before_grid.z_by_frame[:, before_idx]
    after_matrix = after_grid.z_by_frame[:, after_idx]
    x_matrix = before_grid.x_by_frame[:, before_idx]
    before_mean = np.nanmean(before_matrix, axis=0)
    after_mean = np.nanmean(after_matrix, axis=0)
    x_mean = np.nanmedian(x_matrix, axis=0)

    if residual_mode == "per-frame-plane":
        before_residual = before_mean.copy()
        after_residual = after_mean.copy()
        before_trend = np.zeros_like(before_mean)
        after_trend = np.zeros_like(after_mean)
    else:
        before_residual, before_trend = profile_residual(
            common, x_mean, before_mean, args.trend
        )
        after_residual, after_trend = profile_residual(
            common, x_mean, after_mean, args.trend
        )

    frame_metrics: list[dict[str, float]] = []
    for i in range(before_matrix.shape[0]):
        b = before_matrix[i]
        a = after_matrix[i]
        valid = np.isfinite(b) & np.isfinite(a)
        if np.count_nonzero(valid) < 10:
            continue
        if residual_mode == "per-frame-plane":
            b_res = b[valid]
            a_res = a[valid]
        else:
            coord = x_mean[valid]
            if np.count_nonzero(np.isfinite(coord)) < 2:
                coord = common[valid]
            b_res = b[valid] - fit_trend(coord, b[valid], args.trend)
            a_res = a[valid] - fit_trend(coord, a[valid], args.trend)
        frame_metrics.append(
            {
                "before_pv_mm": float(np.ptp(b_res)),
                "after_pv_mm": float(np.ptp(a_res)),
                "before_rms_mm": float(np.sqrt(np.mean(b_res**2))),
                "after_rms_mm": float(np.sqrt(np.mean(a_res**2))),
            }
        )

    def safe_median(key: str) -> float:
        vals = [m[key] for m in frame_metrics]
        return float(np.median(vals)) if vals else float("nan")

    return {
        "compensation_axis": str(table.metadata.get("compensation_axis", "u")),
        "residual_mode": residual_mode,
        "columns": common,
        "xg_mm": x_mean,
        "before_mean_z_mm": before_mean,
        "after_mean_z_mm": after_mean,
        "before_trend_z_mm": before_trend,
        "after_trend_z_mm": after_trend,
        "before_residual_mm": before_residual,
        "after_residual_mm": after_residual,
        "lut_bias_mm": table.bias_mm[table_idx],
        "metrics": {
            "evaluation_frame_count": len(frames),
            "profile_before_pv_mm": float(np.ptp(before_residual)),
            "profile_after_pv_mm": float(np.ptp(after_residual)),
            "profile_before_rms_mm": float(np.sqrt(np.mean(before_residual**2))),
            "profile_after_rms_mm": float(np.sqrt(np.mean(after_residual**2))),
            "median_frame_before_pv_mm": safe_median("before_pv_mm"),
            "median_frame_after_pv_mm": safe_median("after_pv_mm"),
            "median_frame_before_rms_mm": safe_median("before_rms_mm"),
            "median_frame_after_rms_mm": safe_median("after_rms_mm"),
        },
    }


def concatenate_clouds(frames: Sequence[FramePoints]) -> tuple[np.ndarray, np.ndarray]:
    u = np.concatenate([f.u for f in frames])
    xyz = np.vstack([f.xyz for f in frames])
    return u, xyz


def save_bias_table(output_dir: Path, table: BiasTable) -> None:
    payload = {
        "compensation_axis": str(table.metadata.get("compensation_axis", "u")),
        "columns": table.columns,
        "xg_mm": table.xg_mm,
        "raw_mean_z_mm": table.raw_mean_z_mm,
        "trend_z_mm": table.trend_z_mm,
        "bias_mm": table.bias_mm,
        "sample_count": table.sample_count,
        "repeatability_sigma_mm": table.repeatability_sigma_mm,
        "metadata": table.metadata,
    }
    np.save(output_dir / "ground_bias_table.npy", payload, allow_pickle=True)
    with (output_dir / "ground_bias_table.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        axis = str(table.metadata.get("compensation_axis", "u"))
        coordinate_header = "column_u_px" if axis == "u" else "row_v_px"
        ground_header = "xg_mm" if axis == "u" else "yg_mm"
        writer.writerow(
            [
                coordinate_header,
                ground_header,
                "raw_mean_z_mm",
                "retained_trend_z_mm",
                "bias_mm",
                "sample_count",
                "repeatability_sigma_mm",
            ]
        )
        for row in zip(
            table.columns,
            table.xg_mm,
            table.raw_mean_z_mm,
            table.trend_z_mm,
            table.bias_mm,
            table.sample_count,
            table.repeatability_sigma_mm,
        ):
            writer.writerow(row)


def save_npz_cloud(path: Path, u: np.ndarray, xyz: np.ndarray) -> None:
    np.savez_compressed(path, u=u, x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], points=xyz)


def save_ascii_ply(path: Path, xyz: np.ndarray) -> None:
    with path.open("w", encoding="ascii", newline="\n") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {xyz.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        np.savetxt(f, xyz, fmt="%.8f %.8f %.8f")


def set_equal_3d(ax: Any, xyz: np.ndarray) -> None:
    mins = np.nanmin(xyz, axis=0)
    maxs = np.nanmax(xyz, axis=0)
    centers = (mins + maxs) / 2.0
    spans = np.maximum(maxs - mins, 1e-9)
    radius = max(spans[0], spans[1], spans[2]) / 2.0
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def save_profile_figure(output_dir: Path, evaluation: Mapping[str, Any], dpi: int) -> None:
    compensation_axis = str(evaluation.get("compensation_axis", "u"))
    residual_mode = str(evaluation.get("residual_mode", "aggregate-trend"))
    axis_label = "column u" if compensation_axis == "u" else "row v"
    columns = np.asarray(evaluation["columns"])
    before = np.asarray(evaluation["before_mean_z_mm"])
    after = np.asarray(evaluation["after_mean_z_mm"])
    before_res = np.asarray(evaluation["before_residual_mm"])
    after_res = np.asarray(evaluation["after_residual_mm"])

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(columns, before, label="Before compensation")
    axes[0].plot(columns, after, label="After compensation")
    axes[0].set_ylabel(
        "Mean per-frame plane residual (mm)"
        if residual_mode == "per-frame-plane"
        else "Mean Zg (mm)"
    )
    axes[0].set_title(
        f"Ground Z profile before and after {axis_label}-wise compensation"
    )
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(columns, before_res, label="Before: detrended residual")
    axes[1].plot(columns, after_res, label="After: detrended residual")
    axes[1].axhline(0.0, linewidth=1.0)
    axes[1].set_xlabel(f"Image {axis_label} (px)")
    axes[1].set_ylabel("Residual Zg (mm)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "z_profile_before_after.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_pointcloud_figure(
    output_dir: Path,
    before_xyz: np.ndarray,
    after_xyz: np.ndarray,
    max_points: int,
    dpi: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    n = before_xyz.shape[0]
    if n > max_points:
        idx = rng.choice(n, size=max_points, replace=False)
        before_plot = before_xyz[idx]
        after_plot = after_xyz[idx]
    else:
        before_plot = before_xyz
        after_plot = after_xyz

    combined = np.vstack((before_plot, after_plot))
    z_min, z_max = np.nanpercentile(combined[:, 2], [1, 99])

    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    ax1.scatter(before_plot[:, 0], before_plot[:, 1], before_plot[:, 2], s=1, c=before_plot[:, 2], cmap="viridis")
    ax2.scatter(after_plot[:, 0], after_plot[:, 1], after_plot[:, 2], s=1, c=after_plot[:, 2], cmap="viridis")

    for ax, title in ((ax1, "Before compensation"), (ax2, "After compensation")):
        ax.set_title(title)
        ax.set_xlabel("Xg (mm)")
        ax.set_ylabel("Yg (mm)")
        ax.set_zlabel("Zg (mm)")
        ax.set_zlim(z_min, z_max)
        ax.view_init(elev=25, azim=-65)
    set_equal_3d(ax1, before_plot)
    set_equal_3d(ax2, after_plot)
    ax1.set_zlim(z_min, z_max)
    ax2.set_zlim(z_min, z_max)

    fig.suptitle("Ground point cloud before and after axis-wise compensation")
    fig.tight_layout()
    fig.savefig(output_dir / "pointcloud_before_after.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def write_report(
    output_dir: Path,
    args: argparse.Namespace,
    table: BiasTable,
    evaluation: Mapping[str, Any],
    build_frames: Sequence[FramePoints],
    eval_frames: Sequence[FramePoints],
) -> None:
    metrics = evaluation["metrics"]
    compensation_axis = str(table.metadata.get("compensation_axis", "u"))
    axis_name_zh = "图像列 u" if compensation_axis == "u" else "图像行 v"
    validation_mode = "independent holdout" if args.validation_count > 0 else "self-evaluation using the LUT-building frames"
    residual_mode = str(table.metadata.get("residual_mode", "aggregate-trend"))
    if residual_mode == "per-frame-plane":
        compensation_definition = f"""每帧先用 iterative MAD 稳健拟合
`Zg = a*Xg + b*Yg + c`，取 signed vertical Z residual，再按
{compensation_axis} 聚合：

```text
residual_i = Zg_i - (a_i*Xg_i + b_i*Yg_i + c_i)
bias({compensation_axis}) = {args.aggregate}_i residual_i({compensation_axis})
Z_corrected = Z_raw - bias({compensation_axis})
```"""
    else:
        compensation_definition = f"""默认 `--trend linear` 时，先对跨帧聚合地面剖面拟合线性趋势，再将残差作为逐轴系统偏差：

```text
bias({compensation_axis}) = aggregate_Z({compensation_axis}) - linear_trend({compensation_axis})
Z_corrected = Z_raw - bias({compensation_axis})
```"""
    reduction = float("nan")
    if metrics["profile_before_pv_mm"] > 0:
        reduction = 100.0 * (1.0 - metrics["profile_after_pv_mm"] / metrics["profile_before_pv_mm"])

    report = f"""# 地面点云逐点轴向补偿结果报告

## 1. 运行摘要

- 输入目录：`{args.input_dir}`
- 建表帧数：{len(build_frames)}
- 评估帧数：{len(eval_frames)}
- 评估方式：{validation_mode}
- 有效补偿列数：{table.columns.size}
- 补偿轴：`{compensation_axis}`
- residual 模式：`{residual_mode}`
- {axis_name_zh} 范围：{table.columns.min():.3f}–{table.columns.max():.3f} px
- 趋势保留方式：`{args.trend}`
- 跨帧聚合方式：`{args.aggregate}`
- MAD 离群剔除阈值：{args.mad_threshold}
- 平滑窗口：{args.smooth_window}

## 2. 关键结果

| 指标 | 补偿前 | 补偿后 |
|---|---:|---:|
| 平均剖面去趋势峰谷值 P–V | {metrics['profile_before_pv_mm']:.6f} mm | {metrics['profile_after_pv_mm']:.6f} mm |
| 平均剖面去趋势 RMS | {metrics['profile_before_rms_mm']:.6f} mm | {metrics['profile_after_rms_mm']:.6f} mm |
| 单帧 P–V 中位数 | {metrics['median_frame_before_pv_mm']:.6f} mm | {metrics['median_frame_after_pv_mm']:.6f} mm |
| 单帧 RMS 中位数 | {metrics['median_frame_before_rms_mm']:.6f} mm | {metrics['median_frame_after_rms_mm']:.6f} mm |

平均剖面 P–V 降低比例：{reduction:.2f}%

补偿表各列跨帧重复性 σ 中位数：{np.nanmedian(table.repeatability_sigma_mm):.6f} mm。

## 3. 输出文件

- `ground_bias_table.npy`：Python 使用的完整补偿表与元数据。
- `ground_bias_table.csv`：可用 Excel 查看和绘图的逐轴补偿表。
- `z_profile_before_after.png`：补偿前后平均 Z 剖面及去趋势残差。
- `pointcloud_before_after.png`：补偿前后点云三维对比图。
- `ground_cloud_before.npz`、`ground_cloud_after.npz`：补偿前后合并点云。
- `ground_cloud_before.ply`、`ground_cloud_after.ply`：可在 CloudCompare 中打开的 ASCII PLY（若未使用 `--no-ply`）。
- `compensation_metrics.json`：结构化运行参数与评估指标。

## 4. 补偿定义

{compensation_definition}

这样会保留地面的整体高度和倾斜，只消除随图像列稳定重复的弯曲误差。

## 5. 结果解释注意事项

1. 当 `--validation-count 0` 时，补偿表与评估使用同一批帧，补偿后的平均剖面会非常平，这是建表数据上的自评估结果，不代表独立测量精度。
2. 建议额外运行一次 `--validation-count 6`，以前 25 帧建表、后 6 帧独立验证，观察补偿对未参与建表帧的效果。
3. 该补偿表绑定当前机械和光学状态。相机、激光器、基线、工作距离、焦距或光圈发生变化后必须重新采集标准平面并重建 LUT。
4. 补偿消除的是固定系统偏差的综合结果，不能单独证明误差来自激光 smile、承载板不平或外参残差。
5. 如果标准平面本身存在明显不平，补偿表会同时吸收这部分形貌。最终精度实验应使用平面度已知的基准板。
"""
    (output_dir / "ground_flatness_compensation_report.md").write_text(report, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.input_dir = args.input_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.compensation_axis == "v" and args.column_order == "u,x,y,z":
        args.column_order = "v,x,y,z"
    order = parse_column_order(args.column_order, args.compensation_axis)
    files = discover_inputs(args.input_dir, args.glob_pattern)
    adapter_fn = load_adapter(args.adapter.expanduser().resolve()) if args.adapter else None

    frames: list[FramePoints] = []
    errors: list[str] = []
    for path in files:
        try:
            frame = filter_frame(
                load_frame(path, order, adapter_fn, args.compensation_axis), args
            )
            frames.append(frame)
            print(f"[OK] {path.name}: {frame.u.size} points")
        except Exception as exc:
            errors.append(f"{path}: {exc}")
            print(f"[SKIP] {path.name}: {exc}", file=sys.stderr)

    if len(frames) < 2:
        detail = "\n".join(errors[:10])
        raise RuntimeError(f"Only {len(frames)} usable frames were loaded.\n{detail}")

    if args.validation_count < 0 or args.validation_count >= len(frames):
        raise ValueError("--validation-count must be >=0 and smaller than the number of loaded frames")

    if args.validation_count > 0:
        build_frames = frames[:-args.validation_count]
        eval_frames = frames[-args.validation_count:]
    else:
        build_frames = frames
        eval_frames = frames

    table, _, _ = build_bias_table(build_frames, args)
    corrected_all = correct_frames(frames, table)
    evaluation = evaluate_frames(eval_frames, table, args)

    save_bias_table(args.output_dir, table)

    before_u, before_xyz = concatenate_clouds(frames)
    after_u, after_xyz = concatenate_clouds(corrected_all)
    save_npz_cloud(args.output_dir / "ground_cloud_before.npz", before_u, before_xyz)
    save_npz_cloud(args.output_dir / "ground_cloud_after.npz", after_u, after_xyz)
    if not args.no_ply:
        save_ascii_ply(args.output_dir / "ground_cloud_before.ply", before_xyz)
        save_ascii_ply(args.output_dir / "ground_cloud_after.ply", after_xyz)

    save_profile_figure(args.output_dir, evaluation, args.dpi)
    save_pointcloud_figure(
        args.output_dir,
        before_xyz,
        after_xyz,
        args.max_plot_points,
        args.dpi,
        args.seed,
    )

    metrics_payload = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "loaded_frame_count": len(frames),
        "build_frame_count": len(build_frames),
        "evaluation_frame_count": len(eval_frames),
        "skipped_files": errors,
        "table_metadata": table.metadata,
        "metrics": evaluation["metrics"],
        "repeatability_sigma_median_mm": float(np.nanmedian(table.repeatability_sigma_mm)),
    }
    (args.output_dir / "compensation_metrics.json").write_text(
        json.dumps(json_safe(metrics_payload), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(args.output_dir, args, table, evaluation, build_frames, eval_frames)

    m = evaluation["metrics"]
    print("\n=== Completed ===")
    print(f"Loaded frames: {len(frames)}")
    print(f"LUT columns: {table.columns.size}")
    print(f"Profile P-V before: {m['profile_before_pv_mm']:.6f} mm")
    print(f"Profile P-V after : {m['profile_after_pv_mm']:.6f} mm")
    print(f"Outputs: {args.output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted by user", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
