#!/usr/bin/env python3
"""Steger laser-plane calibration with three-frame subpixel motion checks."""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import yaml

import calibrate_laser_plane_core_v2 as v2


LOGGER = logging.getLogger("laser_plane_core_v3")


@dataclass(frozen=True)
class MotionSettings:
    pattern_cols: int
    pattern_rows: int
    policy: str = "report"
    median_threshold_px: float = 0.75
    p95_threshold_px: float = 1.25
    ransac_threshold_px: float = 0.75
    min_inlier_ratio: float = 0.70
    max_reprojection_error_px: float = 0.50
    max_forward_backward_error_px: float = 0.50
    min_ecc_correlation: float = 0.55
    min_points: int = 12


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("必须大于 0")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = v2.build_parser()
    parser.description = __doc__
    parser.add_argument(
        "--triplet-motion-policy",
        choices=("report", "exclude_motion", "exclude_motion_and_unresolved"),
        default="report",
        help="三图运动门控策略；默认只报告，不改变拟合姿态",
    )
    parser.add_argument(
        "--triplet-motion-median-threshold-px",
        type=v2.positive_float,
        default=MotionSettings.median_threshold_px,
    )
    parser.add_argument(
        "--triplet-motion-p95-threshold-px",
        type=v2.positive_float,
        default=MotionSettings.p95_threshold_px,
    )
    parser.add_argument(
        "--triplet-motion-ransac-threshold-px",
        type=v2.positive_float,
        default=MotionSettings.ransac_threshold_px,
    )
    parser.add_argument(
        "--triplet-motion-min-inlier-ratio",
        type=v2.unit_interval,
        default=MotionSettings.min_inlier_ratio,
    )
    parser.add_argument(
        "--triplet-motion-max-reprojection-error-px",
        type=v2.positive_float,
        default=MotionSettings.max_reprojection_error_px,
    )
    parser.add_argument(
        "--triplet-motion-max-forward-backward-error-px",
        type=v2.positive_float,
        default=MotionSettings.max_forward_backward_error_px,
    )
    parser.add_argument(
        "--triplet-motion-min-points",
        type=positive_int,
        default=MotionSettings.min_points,
    )
    parser.add_argument(
        "--triplet-motion-min-ecc-correlation",
        type=v2.unit_interval,
        default=MotionSettings.min_ecc_correlation,
    )
    return parser


def display_u8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def normalized_u8(image: np.ndarray) -> np.ndarray:
    gray = display_u8(image)
    if int(np.max(gray)) == int(np.min(gray)):
        return gray.copy()
    return cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def tracking_image(image: np.ndarray) -> np.ndarray:
    gray = normalized_u8(image)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    gx = cv2.Sobel(enhanced, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(enhanced, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def align_corner_order(
    target: np.ndarray,
    reference: np.ndarray,
    pattern_cols: int,
    pattern_rows: int,
) -> np.ndarray:
    grid = target.reshape(pattern_rows, pattern_cols, 2)
    candidates = (
        grid,
        grid[::-1, ::-1],
        grid[:, ::-1],
        grid[::-1, :],
    )
    return min(
        (item.reshape(-1, 2) for item in candidates),
        key=lambda item: float(np.median(np.linalg.norm(item - reference, axis=1))),
    ).copy()


def detect_short_exposure_corners(
    image: np.ndarray,
    reference: np.ndarray,
    settings: MotionSettings,
) -> np.ndarray | None:
    normalized = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(
        normalized_u8(image)
    )
    pattern = (settings.pattern_cols, settings.pattern_rows)
    base_flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_ACCURACY
    found, corners = cv2.findChessboardCornersSB(normalized, pattern, flags=base_flags)
    if not found:
        found, corners = cv2.findChessboardCornersSB(
            normalized, pattern, flags=base_flags | cv2.CALIB_CB_EXHAUSTIVE
        )
    if not found or corners is None:
        return None
    points = corners.reshape(-1, 2).astype(np.float32)
    if len(points) != len(reference):
        return None
    return align_corner_order(
        points,
        np.asarray(reference, dtype=np.float32),
        settings.pattern_cols,
        settings.pattern_rows,
    )


def estimate_correspondence_motion(
    source_points: np.ndarray,
    target_points: np.ndarray,
    settings: MotionSettings,
    method: str,
    forward_backward_error: np.ndarray | None = None,
) -> dict[str, Any]:
    source = np.asarray(source_points, dtype=np.float32).reshape(-1, 2)
    target = np.asarray(target_points, dtype=np.float32).reshape(-1, 2)
    result: dict[str, Any] = {
        "method": method,
        "tracking_ok": False,
        "point_count": int(len(source)),
        "inlier_count": 0,
        "inlier_ratio": 0.0,
        "median_displacement_px": math.nan,
        "p95_displacement_px": math.nan,
        "max_displacement_px": math.nan,
        "median_dx_px": math.nan,
        "median_dy_px": math.nan,
        "median_reprojection_error_px": math.nan,
        "median_forward_backward_error_px": math.nan,
    }
    if len(source) < 4 or len(source) != len(target):
        return result
    homography, mask = cv2.findHomography(
        source,
        target,
        cv2.RANSAC,
        settings.ransac_threshold_px,
    )
    if homography is None or mask is None:
        return result
    inliers = mask.ravel().astype(bool)
    inlier_count = int(np.count_nonzero(inliers))
    ratio = float(inlier_count / len(source))
    projected = cv2.perspectiveTransform(source.reshape(-1, 1, 2), homography).reshape(-1, 2)
    residual = np.linalg.norm(projected - target, axis=1)
    displacement_xy = projected - source
    displacement = np.linalg.norm(displacement_xy, axis=1)
    median_residual = float(np.median(residual[inliers])) if inlier_count else math.nan
    median_fb = (
        float(np.median(forward_backward_error[inliers]))
        if forward_backward_error is not None and inlier_count
        else math.nan
    )
    enough_points = inlier_count >= settings.min_points
    result.update(
        inlier_count=inlier_count,
        inlier_ratio=ratio,
        median_displacement_px=float(np.median(displacement)),
        p95_displacement_px=float(np.percentile(displacement, 95)),
        max_displacement_px=float(np.max(displacement)),
        median_dx_px=float(np.median(displacement_xy[:, 0])),
        median_dy_px=float(np.median(displacement_xy[:, 1])),
        median_reprojection_error_px=median_residual,
        median_forward_backward_error_px=median_fb,
        tracking_ok=bool(
            enough_points
            and ratio >= settings.min_inlier_ratio
            and median_residual <= settings.max_reprojection_error_px
            and (
                forward_backward_error is None
                or median_fb <= settings.max_forward_backward_error_px
            )
        ),
    )
    return result


def track_points_lk(
    source_image: np.ndarray,
    target_image: np.ndarray,
    seed_points: np.ndarray,
    settings: MotionSettings,
) -> dict[str, Any]:
    source = tracking_image(source_image)
    target = tracking_image(target_image)
    seeds = np.asarray(seed_points, dtype=np.float32).reshape(-1, 1, 2)
    lk = {
        "winSize": (31, 31),
        "maxLevel": 3,
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.001),
    }
    forward, forward_status, _ = cv2.calcOpticalFlowPyrLK(source, target, seeds, None, **lk)
    if forward is None or forward_status is None:
        return estimate_correspondence_motion([], [], settings, "lk_gradient")
    backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
        target, source, forward, None, **lk
    )
    if backward is None or backward_status is None:
        return estimate_correspondence_motion([], [], settings, "lk_gradient")
    forward_points = forward.reshape(-1, 2)
    backward_points = backward.reshape(-1, 2)
    source_points = seeds.reshape(-1, 2)
    fb_error = np.linalg.norm(backward_points - source_points, axis=1)
    valid = (
        forward_status.ravel().astype(bool)
        & backward_status.ravel().astype(bool)
        & np.isfinite(forward_points).all(axis=1)
        & (fb_error <= settings.max_forward_backward_error_px)
    )
    return estimate_correspondence_motion(
        source_points[valid],
        forward_points[valid],
        settings,
        "lk_gradient",
        fb_error[valid],
    )


def track_ecc(
    source_image: np.ndarray,
    target_image: np.ndarray,
    reference_points: np.ndarray,
    roi_polygon: np.ndarray,
    settings: MotionSettings,
) -> dict[str, Any]:
    source = tracking_image(source_image).astype(np.float32) / 255.0
    target = tracking_image(target_image).astype(np.float32) / 255.0
    polygon = np.asarray(roi_polygon, dtype=np.float32).reshape(-1, 2)
    x, y, width, height = cv2.boundingRect(np.rint(polygon).astype(np.int32))
    padding = 20
    left, top = max(0, x - padding), max(0, y - padding)
    right = min(source.shape[1], x + width + padding)
    bottom = min(source.shape[0], y + height + padding)
    source_crop = source[top:bottom, left:right]
    target_crop = target[top:bottom, left:right]
    mask = np.zeros(source_crop.shape, dtype=np.uint8)
    local_polygon = np.rint(polygon - np.asarray([left, top])).astype(np.int32)
    cv2.fillConvexPoly(mask, local_polygon, 255)
    warp = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1.0e-6)
    try:
        correlation, warp = cv2.findTransformECC(
            source_crop,
            target_crop,
            warp,
            cv2.MOTION_AFFINE,
            criteria,
            inputMask=mask,
            gaussFiltSize=5,
        )
    except cv2.error:
        return estimate_correspondence_motion([], [], settings, "ecc_gradient_affine")
    local_points = (
        np.asarray(reference_points, dtype=np.float32).reshape(-1, 2)
        - np.asarray([left, top], dtype=np.float32)
    )
    target_points = cv2.transform(local_points.reshape(-1, 1, 2), warp).reshape(-1, 2)
    target_points += np.asarray([left, top], dtype=np.float32)
    result = estimate_correspondence_motion(
        reference_points, target_points, settings, "ecc_gradient_affine"
    )
    result["ecc_correlation"] = float(correlation)
    result["tracking_ok"] = bool(
        result["tracking_ok"] and correlation >= settings.min_ecc_correlation
    )
    return result


def pair_motion(
    source_image: np.ndarray,
    target_image: np.ndarray,
    source_corners: np.ndarray | None,
    target_corners: np.ndarray | None,
    fallback_seeds: np.ndarray,
    roi_polygon: np.ndarray,
    settings: MotionSettings,
) -> dict[str, Any]:
    if source_corners is not None and target_corners is not None:
        direct = estimate_correspondence_motion(
            source_corners, target_corners, settings, "chessboard_corners_sb"
        )
        if direct["tracking_ok"]:
            return direct
    seeds = source_corners if source_corners is not None else fallback_seeds
    lk_result = track_points_lk(source_image, target_image, seeds, settings)
    if lk_result["tracking_ok"]:
        return lk_result
    return track_ecc(
        source_image, target_image, fallback_seeds, roi_polygon, settings
    )


def classify_triplet(
    pairs: Mapping[str, Mapping[str, Any]], settings: MotionSettings
) -> tuple[str, str | None]:
    unresolved = [name for name, item in pairs.items() if not item["tracking_ok"]]
    if unresolved:
        return "unresolved", unresolved[0]
    moved = [
        name
        for name, item in pairs.items()
        if float(item["median_displacement_px"]) > settings.median_threshold_px
        or float(item["p95_displacement_px"]) > settings.p95_threshold_px
    ]
    if moved:
        return "motion", moved[0]
    return "static", None


def flatten_pair(prefix: str, values: Mapping[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def measure_triplet_motion(
    prepared: Any,
    nolaser_path: Path,
    settings: MotionSettings,
    common: Any,
) -> dict[str, Any]:
    chess = common.to_gray(common.read_image(prepared.chess_path), prepared.chess_path)
    nolaser = common.to_gray(common.read_image(nolaser_path), nolaser_path)
    laser = common.to_gray(common.read_image(prepared.laser_path), prepared.laser_path)
    chess_corners = np.asarray(prepared.pose.corners, dtype=np.float32).reshape(-1, 2)
    roi_polygon = np.asarray(prepared.pose.roi_polygon, dtype=np.float32)
    nolaser_corners = detect_short_exposure_corners(
        nolaser, chess_corners, settings
    )
    laser_corners = detect_short_exposure_corners(laser, chess_corners, settings)
    pairs = {
        "chess_to_nolaser": pair_motion(
            chess,
            nolaser,
            chess_corners,
            nolaser_corners,
            chess_corners,
            roi_polygon,
            settings,
        ),
        "nolaser_to_laser": pair_motion(
            nolaser,
            laser,
            nolaser_corners,
            laser_corners,
            chess_corners,
            roi_polygon,
            settings,
        ),
        "chess_to_laser": pair_motion(
            chess,
            laser,
            chess_corners,
            laser_corners,
            chess_corners,
            roi_polygon,
            settings,
        ),
    }
    status, worst_pair = classify_triplet(pairs, settings)
    resolved = [item for item in pairs.values() if item["tracking_ok"]]
    maximum_median = (
        max(float(item["median_displacement_px"]) for item in resolved)
        if resolved
        else math.nan
    )
    maximum_p95 = (
        max(float(item["p95_displacement_px"]) for item in resolved)
        if resolved
        else math.nan
    )
    result: dict[str, Any] = {
        "tracking_ok": status != "unresolved",
        "movement_method": "triplet_subpixel_chessboard",
        "triplet_motion_status": status,
        "triplet_worst_pair": worst_pair or "none",
        "median_displacement_px": maximum_median,
        "p95_displacement_px": maximum_p95,
        "direct_nolaser_corners_found": nolaser_corners is not None,
        "direct_laser_corners_found": laser_corners is not None,
    }
    for name, values in pairs.items():
        result.update(flatten_pair(name, values))
    return result


def excluded_triplet_ids(
    frames: Sequence[Any], args: argparse.Namespace, settings: MotionSettings
) -> set[int]:
    excluded = set(args.manual_exclude_ids)
    if settings.policy == "report":
        return excluded
    for frame in frames:
        status = frame.pair_motion.get("triplet_motion_status", "unresolved")
        reject = status == "motion" or (
            status == "unresolved"
            and settings.policy == "exclude_motion_and_unresolved"
        )
        if reject:
            excluded.add(frame.result.image_id)
            LOGGER.warning(
                "ID %03d 三图运动状态=%s，按策略 %s 排除",
                frame.result.image_id,
                status,
                settings.policy,
            )
    return excluded


def annotate_outputs(output_dir: Path, settings: MotionSettings) -> None:
    metadata = asdict(settings)
    metadata["method"] = "three_pair_subpixel_chessboard_homography"
    for name in ("laser_plane.yaml", "laser_plane_core_config_used.yaml"):
        path = output_dir / name
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        data["triplet_motion"] = metadata
        path.write_text(
            yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
    with (output_dir / "calibration_report.md").open("a", encoding="utf-8") as stream:
        stream.write(
            "\n## 三图亚像素运动门控\n\n"
            "- 配对：chess→nolaser、nolaser→laser、chess→laser\n"
            "- 主方法：短曝光棋盘亚像素角点 + RANSAC 单应性\n"
            "- 回退方法：梯度图金字塔 LK + 正反向检查\n"
            f"- 策略：{settings.policy}\n"
            f"- 中位位移阈值：{settings.median_threshold_px:.3f} px\n"
            f"- P95 位移阈值：{settings.p95_threshold_px:.3f} px\n"
        )


def settings_from_args(args: argparse.Namespace) -> MotionSettings:
    return MotionSettings(
        pattern_cols=int(args.pattern_cols),
        pattern_rows=int(args.pattern_rows),
        policy=str(args.triplet_motion_policy),
        median_threshold_px=float(args.triplet_motion_median_threshold_px),
        p95_threshold_px=float(args.triplet_motion_p95_threshold_px),
        ransac_threshold_px=float(args.triplet_motion_ransac_threshold_px),
        min_inlier_ratio=float(args.triplet_motion_min_inlier_ratio),
        max_reprojection_error_px=float(
            args.triplet_motion_max_reprojection_error_px
        ),
        max_forward_backward_error_px=float(
            args.triplet_motion_max_forward_backward_error_px
        ),
        min_ecc_correlation=float(args.triplet_motion_min_ecc_correlation),
        min_points=int(args.triplet_motion_min_points),
    )


def run(args: argparse.Namespace) -> np.ndarray:
    settings = settings_from_args(args)
    core = v2._import_core()
    original_measure = core.measure_pair_motion
    original_exclude = core.auto_excluded_ids

    def runtime_measure(
        prepared: Any, nolaser_path: Path, config: Mapping[str, Any]
    ) -> dict[str, Any]:
        return measure_triplet_motion(
            prepared, nolaser_path, settings, core.common
        )

    def runtime_exclude(
        frames: Sequence[Any], runtime_args: argparse.Namespace, config: Mapping[str, Any]
    ) -> set[int]:
        return excluded_triplet_ids(frames, runtime_args, settings)

    core.measure_pair_motion = runtime_measure
    core.auto_excluded_ids = runtime_exclude
    try:
        plane = v2.run(args)
    finally:
        core.measure_pair_motion = original_measure
        core.auto_excluded_ids = original_exclude
    annotate_outputs(args.output_dir, settings)
    return plane


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        run(args)
        return 0
    except (
        FileNotFoundError,
        FileExistsError,
        ModuleNotFoundError,
        RuntimeError,
        ValueError,
        cv2.error,
    ) as exc:
        logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")
        LOGGER.exception("标定失败：%s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
