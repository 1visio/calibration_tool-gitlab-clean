#!/usr/bin/env python3
"""Diagnose whether ground extrinsics are the source of flat-ground curvature."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import reconstruct_ground_pointcloud_cloudcompare_v3 as steger_reconstruct  # noqa: E402
import reconstruct_ground_pointcloud_interactive as reconstruct_core  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--intrinsics", type=Path, required=True)
    parser.add_argument("--laser-plane", type=Path, required=True)
    parser.add_argument("--extrinsics", type=Path, required=True)
    parser.add_argument("--bias-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--extractors",
        nargs="+",
        choices=("centroid", "steger"),
        default=("centroid", "steger"),
    )
    return parser


def fit_3d_line(points: np.ndarray) -> dict[str, Any]:
    pts = np.asarray(points, dtype=np.float64)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 2:
        return {"point_count": int(len(pts)), "rmse_mm": None}
    centre = pts.mean(axis=0)
    _, _, vh = np.linalg.svd(pts - centre, full_matrices=False)
    direction = vh[0]
    nonzero = np.flatnonzero(np.abs(direction) > 1e-12)
    if nonzero.size and direction[nonzero[0]] < 0.0:
        direction = -direction
    projected = centre + ((pts - centre) @ direction)[:, None] * direction
    dist = np.linalg.norm(pts - projected, axis=1)
    return {
        "point_count": int(len(pts)),
        "rmse_mm": float(np.sqrt(np.mean(dist**2))),
        "mean_abs_distance_mm": float(np.mean(dist)),
        "p95_abs_distance_mm": float(np.percentile(dist, 95)),
        "max_abs_distance_mm": float(np.max(dist)),
        "centre_mm": [float(x) for x in centre],
        "direction": [float(x) for x in direction],
    }


def fit_z_plane(xyz: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    pts = np.asarray(xyz, dtype=np.float64)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 3:
        return {"point_count": int(len(pts)), "rmse_mm": None}, np.zeros(len(pts))
    design = np.column_stack([pts[:, 0], pts[:, 1], np.ones(len(pts))])
    coeff, _residuals, rank, singular = np.linalg.lstsq(design, pts[:, 2], rcond=None)
    residual = pts[:, 2] - design @ coeff
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0.0 else math.inf
    return {
        "point_count": int(len(pts)),
        "model": "Z = a*X + b*Y + c",
        "a": float(coeff[0]),
        "b": float(coeff[1]),
        "c_mm": float(coeff[2]),
        "rank": int(rank),
        "condition_number": condition,
        "rmse_mm": float(np.sqrt(np.mean(residual**2))),
        "mean_abs_residual_mm": float(np.mean(np.abs(residual))),
        "p95_abs_residual_mm": float(np.percentile(np.abs(residual), 95)),
        "pv_residual_mm": float(np.max(residual) - np.min(residual)),
    }, residual


def fit_z_linear_x(xyz: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    pts = np.asarray(xyz, dtype=np.float64)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 2:
        return {"point_count": int(len(pts)), "rmse_mm": None}, np.zeros(len(pts))
    design = np.column_stack([pts[:, 0], np.ones(len(pts))])
    coeff, *_ = np.linalg.lstsq(design, pts[:, 2], rcond=None)
    residual = pts[:, 2] - design @ coeff
    return {
        "point_count": int(len(pts)),
        "model": "Z = a*X + c",
        "a": float(coeff[0]),
        "c_mm": float(coeff[1]),
        "rmse_mm": float(np.sqrt(np.mean(residual**2))),
        "p95_abs_residual_mm": float(np.percentile(np.abs(residual), 95)),
        "pv_residual_mm": float(np.max(residual) - np.min(residual)),
    }, residual


def fit_series_linear(coord: np.ndarray, z: np.ndarray, label: str) -> tuple[dict[str, Any], np.ndarray]:
    coord = np.asarray(coord, dtype=np.float64).reshape(-1)
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    valid = np.isfinite(coord) & np.isfinite(z)
    coord = coord[valid]
    z = z[valid]
    if coord.size < 2:
        return {"point_count": int(coord.size), "rmse_mm": None}, np.zeros(coord.size)
    design = np.column_stack([coord, np.ones(coord.size)])
    coeff, *_ = np.linalg.lstsq(design, z, rcond=None)
    residual = z - design @ coeff
    return {
        "point_count": int(coord.size),
        "model": f"Z = a*{label} + c",
        "a": float(coeff[0]),
        "c_mm": float(coeff[1]),
        "z_std_mm": float(np.std(z)),
        "z_pv_mm": float(np.max(z) - np.min(z)),
        "rmse_mm": float(np.sqrt(np.mean(residual**2))),
        "p95_abs_residual_mm": float(np.percentile(np.abs(residual), 95)),
        "pv_residual_mm": float(np.max(residual) - np.min(residual)),
    }, residual


def z_distribution(z: np.ndarray) -> dict[str, Any]:
    values = np.asarray(z, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {}
    centred = values - np.median(values)
    return {
        "point_count": int(values.size),
        "mean_mm": float(np.mean(values)),
        "median_mm": float(np.median(values)),
        "std_mm": float(np.std(values)),
        "pv_mm": float(np.max(values) - np.min(values)),
        "rms_about_median_mm": float(np.sqrt(np.mean(centred**2))),
        "p95_abs_about_median_mm": float(np.percentile(np.abs(centred), 95)),
    }


def profile_by_u(frame: pd.DataFrame, residual_column: str) -> pd.DataFrame:
    temp = frame.copy()
    temp["u_bin"] = np.rint(temp["u"].to_numpy(dtype=np.float64)).astype(int)
    return temp.groupby("u_bin", as_index=False).agg(
        u_mean=("u", "mean"),
        xg_mean=("Xg_raw", "mean"),
        residual_mean=(residual_column, "mean"),
        residual_std=(residual_column, "std"),
        count=(residual_column, "count"),
    )


def profile_metrics(profile: pd.DataFrame) -> dict[str, Any]:
    values = profile["residual_mean"].to_numpy(dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {}
    return {
        "column_count": int(values.size),
        "mean_mm": float(np.mean(values)),
        "rms_mm": float(np.sqrt(np.mean(values**2))),
        "pv_mm": float(np.max(values) - np.min(values)),
        "p95_abs_mm": float(np.percentile(np.abs(values), 95)),
    }


def plot_profile(profile: pd.DataFrame, output_path: Path, title: str) -> None:
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.plot(profile["u_bin"], profile["residual_mean"], linewidth=1.0)
    axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    axis.set_xlabel("u bin / px")
    axis.set_ylabel("mean residual / mm")
    axis.set_title(title)
    axis.grid(True, linestyle="--", alpha=0.35)
    figure.tight_layout()
    figure.savefig(output_path, dpi=170)
    plt.close(figure)


def plot_z_vs_x(frame: pd.DataFrame, output_path: Path, title: str, z_column: str) -> None:
    figure, axis = plt.subplots(figsize=(10, 6))
    sample = frame if len(frame) <= 6000 else frame.sample(6000, random_state=20260730)
    scatter = axis.scatter(sample["Xg_raw"], sample[z_column], c=sample["u"], s=4, cmap="turbo")
    figure.colorbar(scatter, ax=axis, label="u / px")
    axis.set_xlabel("Xg_raw / mm")
    axis.set_ylabel(f"{z_column} / mm")
    axis.set_title(title)
    axis.grid(True, linestyle="--", alpha=0.35)
    figure.tight_layout()
    figure.savefig(output_path, dpi=170)
    plt.close(figure)


def load_bias_table(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(path, allow_pickle=True).item()
    columns = np.asarray(payload["columns"], dtype=np.float64).reshape(-1)
    bias = np.asarray(payload["bias_mm"], dtype=np.float64).reshape(-1)
    order = np.argsort(columns)
    return columns[order], bias[order]


def add_bias(frame: pd.DataFrame, columns: np.ndarray, bias: np.ndarray) -> pd.DataFrame:
    point_bias = np.interp(
        frame["u"].to_numpy(dtype=np.float64),
        columns,
        bias,
        left=bias[0],
        right=bias[-1],
    )
    result = frame.copy()
    result["ground_bias_mm"] = point_bias
    result["Zg_bias_corrected"] = result["Zg_raw"] - point_bias
    return result


def reconstruct_one_dataset(
    extractor: str,
    args: argparse.Namespace,
    calibration: reconstruct_core.Calibration,
    extraction_params: reconstruct_core.ExtractionParams,
    reconstruction_params: reconstruct_core.ReconstructionParams,
    bias_columns: np.ndarray,
    bias_mm: np.ndarray,
    logger: logging.Logger,
) -> dict[str, Any]:
    output_dir = args.output_dir / extractor
    output_dir.mkdir(parents=True, exist_ok=True)
    images = reconstruct_core.discover_images(args.input_dir)
    steger_settings = steger_reconstruct.StegerSettings()
    frames: list[pd.DataFrame] = []
    per_frame_rows: list[dict[str, Any]] = []

    logger.info("Reconstructing %s frames with extractor=%s", len(images), extractor)
    for image_path in images:
        gray = reconstruct_core.read_mono8(image_path)
        if extractor == "centroid":
            pixels, _meta, _signal = reconstruct_core.extract_laser_centres(gray, extraction_params)
        elif extractor == "steger":
            pixels, _meta, _signal = steger_reconstruct.extract_laser_centres_steger(
                gray, extraction_params, steger_settings
            )
        else:
            raise ValueError(extractor)

        frame, filtered = reconstruct_core.reconstruct_points(
            pixels, calibration, reconstruction_params
        )
        frame = frame.rename(
            columns={
                "Xc_mm": "Xc",
                "Yc_mm": "Yc",
                "Zc_mm": "Zc",
                "Xg_mm": "Xg_raw",
                "Yg_mm": "Yg_raw",
                "Zg_mm": "Zg_raw",
            }
        )
        frame = frame[["u", "v", "Xc", "Yc", "Zc", "Xg_raw", "Yg_raw", "Zg_raw"]]
        frame.insert(0, "image", image_path.name)
        frame = add_bias(frame, bias_columns, bias_mm)
        frame_dir = output_dir / image_path.stem
        frame_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(frame_dir / "diagnostic_points.csv", index=False, float_format="%.9f")
        frames.append(frame)

        camera_line = fit_3d_line(frame[["Xc", "Yc", "Zc"]].to_numpy())
        ground_line = fit_3d_line(frame[["Xg_raw", "Yg_raw", "Zg_raw"]].to_numpy())
        raw_plane, _ = fit_z_plane(frame[["Xg_raw", "Yg_raw", "Zg_raw"]].to_numpy())
        corrected_plane, _ = fit_z_plane(
            frame[["Xg_raw", "Yg_raw", "Zg_bias_corrected"]].to_numpy()
        )
        per_frame_rows.append(
            {
                "image": image_path.name,
                "extractor": extractor,
                "extracted_point_count": int(len(pixels)),
                "valid_point_count": int(len(frame)),
                "filtered_point_count": int(len(pixels) - len(frame)),
                **{f"filter_{key}": int(value) for key, value in filtered.items()},
                "camera_3d_line_rmse_mm": camera_line["rmse_mm"],
                "ground_raw_3d_line_rmse_mm": ground_line["rmse_mm"],
                "raw_plane_rmse_mm": raw_plane["rmse_mm"],
                "raw_plane_pv_residual_mm": raw_plane["pv_residual_mm"],
                "corrected_plane_rmse_mm": corrected_plane["rmse_mm"],
                "corrected_plane_pv_residual_mm": corrected_plane["pv_residual_mm"],
            }
        )

    all_points = pd.concat(frames, ignore_index=True)
    raw_plane, raw_plane_residual = fit_z_plane(
        all_points[["Xg_raw", "Yg_raw", "Zg_raw"]].to_numpy()
    )
    raw_linear_x, raw_linear_x_residual = fit_z_linear_x(
        all_points[["Xg_raw", "Yg_raw", "Zg_raw"]].to_numpy()
    )
    corrected_plane, corrected_plane_residual = fit_z_plane(
        all_points[["Xg_raw", "Yg_raw", "Zg_bias_corrected"]].to_numpy()
    )
    corrected_linear_x, corrected_linear_x_residual = fit_z_linear_x(
        all_points[["Xg_raw", "Yg_raw", "Zg_bias_corrected"]].to_numpy()
    )
    raw_linear_u, raw_linear_u_residual = fit_series_linear(
        all_points["u"].to_numpy(dtype=np.float64),
        all_points["Zg_raw"].to_numpy(dtype=np.float64),
        "u",
    )
    corrected_linear_u, corrected_linear_u_residual = fit_series_linear(
        all_points["u"].to_numpy(dtype=np.float64),
        all_points["Zg_bias_corrected"].to_numpy(dtype=np.float64),
        "u",
    )
    all_points["raw_plane_residual"] = raw_plane_residual
    all_points["raw_linear_x_residual"] = raw_linear_x_residual
    all_points["raw_linear_u_residual"] = raw_linear_u_residual
    all_points["corrected_plane_residual"] = corrected_plane_residual
    all_points["corrected_linear_x_residual"] = corrected_linear_x_residual
    all_points["corrected_linear_u_residual"] = corrected_linear_u_residual
    all_points.to_csv(output_dir / "all_diagnostic_points.csv", index=False, float_format="%.9f")
    pd.DataFrame(per_frame_rows).to_csv(
        output_dir / "per_frame_metrics.csv", index=False, float_format="%.9f"
    )

    raw_profile = profile_by_u(all_points, "raw_plane_residual")
    corrected_profile = profile_by_u(all_points, "corrected_plane_residual")
    raw_u_linear_profile = profile_by_u(all_points, "raw_linear_u_residual")
    corrected_u_linear_profile = profile_by_u(all_points, "corrected_linear_u_residual")
    raw_profile.to_csv(output_dir / "u_profile_raw_plane_residual.csv", index=False, float_format="%.9f")
    corrected_profile.to_csv(
        output_dir / "u_profile_corrected_plane_residual.csv",
        index=False,
        float_format="%.9f",
    )
    raw_u_linear_profile.to_csv(
        output_dir / "u_profile_raw_linear_u_residual.csv",
        index=False,
        float_format="%.9f",
    )
    corrected_u_linear_profile.to_csv(
        output_dir / "u_profile_corrected_linear_u_residual.csv",
        index=False,
        float_format="%.9f",
    )
    plot_profile(raw_profile, output_dir / "u_profile_raw_plane_residual.png", f"{extractor}: raw plane residual by u")
    plot_profile(
        corrected_profile,
        output_dir / "u_profile_corrected_plane_residual.png",
        f"{extractor}: bias-corrected plane residual by u",
    )
    plot_profile(
        raw_u_linear_profile,
        output_dir / "u_profile_raw_linear_u_residual.png",
        f"{extractor}: raw residual after Z=a*u+c",
    )
    plot_profile(
        corrected_u_linear_profile,
        output_dir / "u_profile_corrected_linear_u_residual.png",
        f"{extractor}: corrected residual after Z=a*u+c",
    )
    plot_z_vs_x(all_points, output_dir / "zg_raw_vs_xg.png", f"{extractor}: raw Zg vs Xg", "Zg_raw")
    plot_z_vs_x(
        all_points,
        output_dir / "zg_bias_corrected_vs_xg.png",
        f"{extractor}: bias-corrected Zg vs Xg",
        "Zg_bias_corrected",
    )

    camera_line = fit_3d_line(all_points[["Xc", "Yc", "Zc"]].to_numpy())
    ground_line = fit_3d_line(all_points[["Xg_raw", "Yg_raw", "Zg_raw"]].to_numpy())
    corrected_line = fit_3d_line(
        all_points[["Xg_raw", "Yg_raw", "Zg_bias_corrected"]].to_numpy()
    )
    metrics = {
        "extractor": extractor,
        "frame_count": len(images),
        "total_valid_point_count": int(len(all_points)),
        "camera_3d_line_fit": camera_line,
        "ground_raw_3d_line_fit": ground_line,
        "ground_bias_corrected_3d_line_fit": corrected_line,
        "line_rmse_camera_minus_ground_raw_mm": float(camera_line["rmse_mm"] - ground_line["rmse_mm"]),
        "ground_raw_plane_fit": raw_plane,
        "ground_raw_linear_x_fit": raw_linear_x,
        "ground_bias_corrected_plane_fit": corrected_plane,
        "ground_bias_corrected_linear_x_fit": corrected_linear_x,
        "ground_raw_linear_u_fit": raw_linear_u,
        "ground_bias_corrected_linear_u_fit": corrected_linear_u,
        "ground_raw_z_distribution": z_distribution(all_points["Zg_raw"].to_numpy(dtype=np.float64)),
        "ground_bias_corrected_z_distribution": z_distribution(
            all_points["Zg_bias_corrected"].to_numpy(dtype=np.float64)
        ),
        "raw_u_profile_after_plane_detrend": profile_metrics(raw_profile),
        "corrected_u_profile_after_plane_detrend": profile_metrics(corrected_profile),
        "raw_u_profile_after_linear_u_detrend": profile_metrics(raw_u_linear_profile),
        "corrected_u_profile_after_linear_u_detrend": profile_metrics(corrected_u_linear_profile),
        "files": {
            "all_points_csv": str(output_dir / "all_diagnostic_points.csv"),
            "per_frame_metrics_csv": str(output_dir / "per_frame_metrics.csv"),
            "raw_profile_csv": str(output_dir / "u_profile_raw_plane_residual.csv"),
            "corrected_profile_csv": str(output_dir / "u_profile_corrected_plane_residual.csv"),
            "raw_linear_u_profile_csv": str(output_dir / "u_profile_raw_linear_u_residual.csv"),
            "corrected_linear_u_profile_csv": str(output_dir / "u_profile_corrected_linear_u_residual.csv"),
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metrics


def markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = list(rows[0])
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        values = []
        for header in headers:
            value = row[header]
            values.append(f"{value:.6f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "diagnostic.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger("ground_extrinsic_diagnostic")
    logger.info("Starting diagnostic")
    logger.info("Input dir: %s", args.input_dir)
    logger.info("Intrinsics: %s", args.intrinsics)
    logger.info("Laser plane: %s", args.laser_plane)
    logger.info("Extrinsics: %s", args.extrinsics)
    logger.info("Bias table: %s", args.bias_table)

    calibration = reconstruct_core.load_calibration(
        args.intrinsics,
        args.laser_plane,
        args.extrinsics,
    )
    extraction_params = reconstruct_core.ExtractionParams(
        background_kernel=51,
        min_local_contrast_dn=20.0,
        centroid_window_radius=5,
        segment_min_columns=42,
        continuity_max_column_gap=2,
        continuity_max_vertical_jump=14.0,
        correction_window=7,
        correction_max_shift=3.5,
    )
    reconstruction_params = reconstruct_core.ReconstructionParams(
        parallel_epsilon=1.0e-9,
        min_camera_depth_mm=100.0,
        max_camera_depth_mm=1500.0,
    )
    bias_columns, bias_mm = load_bias_table(args.bias_table)
    all_metrics: dict[str, Any] = {
        "inputs": {
            "input_dir": str(args.input_dir),
            "intrinsics": str(args.intrinsics),
            "laser_plane": str(args.laser_plane),
            "extrinsics": str(args.extrinsics),
            "bias_table": str(args.bias_table),
        },
        "output_dir": str(args.output_dir),
        "diagnostic_definition": {
            "3d_line_rmse": "RMS orthogonal distance to the best-fit 3D line by SVD.",
            "plane_detrend": "Least-squares Zg = a*Xg + b*Yg + c; inspect residuals by u.",
            "bias_corrected": "Zg_bias_corrected = Zg_raw - interp(bias_table, u).",
        },
        "extractors": {},
    }
    summary_rows: list[dict[str, Any]] = []
    for extractor in args.extractors:
        metrics = reconstruct_one_dataset(
            extractor,
            args,
            calibration,
            extraction_params,
            reconstruction_params,
            bias_columns,
            bias_mm,
            logger,
        )
        all_metrics["extractors"][extractor] = metrics
        summary_rows.append(
            {
                "extractor": extractor,
                "points": metrics["total_valid_point_count"],
                "camera_line_rmse_mm": metrics["camera_3d_line_fit"]["rmse_mm"],
                "ground_raw_line_rmse_mm": metrics["ground_raw_3d_line_fit"]["rmse_mm"],
                "line_rmse_diff_mm": metrics["line_rmse_camera_minus_ground_raw_mm"],
                "raw_plane_rmse_mm": metrics["ground_raw_plane_fit"]["rmse_mm"],
                "raw_plane_pv_mm": metrics["ground_raw_plane_fit"]["pv_residual_mm"],
                "raw_z_std_mm": metrics["ground_raw_z_distribution"]["std_mm"],
                "raw_z_pv_mm": metrics["ground_raw_z_distribution"]["pv_mm"],
                "raw_linear_u_rmse_mm": metrics["ground_raw_linear_u_fit"]["rmse_mm"],
                "raw_linear_u_pv_mm": metrics["ground_raw_linear_u_fit"]["pv_residual_mm"],
                "raw_u_profile_linear_rms_mm": metrics["raw_u_profile_after_linear_u_detrend"]["rms_mm"],
                "raw_u_profile_linear_pv_mm": metrics["raw_u_profile_after_linear_u_detrend"]["pv_mm"],
                "corrected_plane_rmse_mm": metrics["ground_bias_corrected_plane_fit"]["rmse_mm"],
                "corrected_z_std_mm": metrics["ground_bias_corrected_z_distribution"]["std_mm"],
                "corrected_z_pv_mm": metrics["ground_bias_corrected_z_distribution"]["pv_mm"],
                "corrected_linear_u_rmse_mm": metrics["ground_bias_corrected_linear_u_fit"]["rmse_mm"],
                "corrected_linear_u_pv_mm": metrics["ground_bias_corrected_linear_u_fit"]["pv_residual_mm"],
                "corrected_u_profile_linear_rms_mm": metrics["corrected_u_profile_after_linear_u_detrend"]["rms_mm"],
                "corrected_u_profile_linear_pv_mm": metrics["corrected_u_profile_after_linear_u_detrend"]["pv_mm"],
            }
        )

    (args.output_dir / "diagnostic_summary.json").write_text(
        json.dumps(all_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "summary_metrics.csv", index=False, float_format="%.9f")
    report = [
        "# Ground Extrinsic Curvature Diagnostic",
        "",
        "## Summary",
        "",
        markdown_table(summary_rows),
        "",
        "## Interpretation",
        "",
        "- If camera and ground raw 3D line RMSE are equal, the rigid extrinsic did not create curvature.",
        "- Plane detrending removes global offset and linear tilt. Remaining residual profile by u indicates fixed nonlinear/column-wise shape.",
        "- Note: free `Zg = a*Xg + b*Yg + c` plane fitting is degenerate for one laser stripe because all reconstructed points lie on the transformed laser plane. Use `Zg` distribution and `Zg = a*u + c` residuals for the practical curvature check.",
        "- Bias-corrected metrics use the supplied LUT for comparison; raw metrics are the main extrinsic diagnostic.",
        "",
        "## Files",
        "",
        "- `diagnostic.log`: execution log.",
        "- `summary_metrics.csv`: compact metric table.",
        "- `diagnostic_summary.json`: full structured metrics.",
        "- `<extractor>/all_diagnostic_points.csv`: reconstructed points and residuals.",
        "- `<extractor>/per_frame_metrics.csv`: per-frame metrics.",
        "- `<extractor>/u_profile_raw_plane_residual.csv/png`: raw plane-detrended residual by u.",
        "- `<extractor>/u_profile_corrected_plane_residual.csv/png`: LUT-corrected plane-detrended residual by u.",
        "- `<extractor>/u_profile_raw_linear_u_residual.csv/png`: raw residual by u after removing `Zg = a*u + c`.",
        "- `<extractor>/u_profile_corrected_linear_u_residual.csv/png`: LUT-corrected residual by u after removing `Zg = a*u + c`.",
    ]
    (args.output_dir / "README_diagnostic.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    logger.info("Diagnostic complete: %s", args.output_dir)
    return 0


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
