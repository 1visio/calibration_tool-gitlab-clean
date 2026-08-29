#!/usr/bin/env python3
"""Ground extrinsic calibration using the unified realtime Steger extractor.

This V2 entry point preserves the original ``calibrate_ground_extrinsics_steger.py``
for traceability and routes the Steger core through ``realtime_steger.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import calibrate_ground_extrinsics as core  # noqa: E402
import steger_laser_center as steger  # noqa: E402


LOGGER = logging.getLogger("ground_extrinsics_steger_v2")


def build_parser() -> argparse.ArgumentParser:
    parser = core.build_parser()
    parser.description = __doc__
    parser.add_argument("--steger-config", type=Path, default=None)
    parser.add_argument(
        "--steger-sigma-px",
        type=steger.positive_float,
        default=None,
        help="兼容旧参数；覆盖统一实时 Steger 配置中的 sigma（像素）。",
    )
    parser.add_argument(
        "--steger-max-offset-px",
        type=steger.positive_float,
        default=None,
        help="Maximum subpixel shift along the ridge normal; default 0.75.",
    )
    parser.add_argument(
        "--steger-min-normal-y",
        type=steger.unit_interval,
        default=None,
        help="Minimum absolute vertical component of the ridge normal; default 0.5.",
    )
    parser.add_argument(
        "--steger-min-response-ratio",
        type=steger.nonnegative_float,
        default=None,
        help="Minimum negative-curvature response as a fraction of full scale; default 0.0005.",
    )
    parser.add_argument(
        "--steger-background-window-px",
        type=steger.positive_odd_int,
        default=None,
        help="Vertical percentile-filter window for local background removal; default 31.",
    )
    parser.add_argument(
        "--steger-min-prominence-ratio",
        type=steger.nonnegative_float,
        default=None,
        help="Minimum column-peak prominence as a fraction of full scale; default 0.010.",
    )
    parser.add_argument(
        "--steger-profile-smoothing-sigma-px",
        type=steger.nonnegative_float,
        default=None,
        help="Vertical smoothing sigma before peak search; default 0.8.",
    )
    parser.add_argument(
        "--steger-geometric-filter",
        choices=("flat-ground", "continuity-only", "none"),
        default="flat-ground",
        help=(
            "Post-filter mode for ground laser images. "
            "'flat-ground' applies continuity and undistorted line RANSAC; "
            "'continuity-only' skips RANSAC; 'none' keeps Steger-valid columns."
        ),
    )
    parser.add_argument("--steger-threshold", type=steger.nonnegative_float, default=None)
    parser.add_argument("--steger-deriv-thresh", type=steger.positive_float, default=None)
    parser.add_argument("--steger-roi-margin", type=int, default=None)
    parser.add_argument("--steger-roi-max-height", type=steger.positive_int, default=None)
    parser.add_argument("--steger-scan-axis", choices=("column", "row"), default=None)
    return parser


def _settings_from_args(args: argparse.Namespace) -> steger.StegerSettings:
    return steger.settings_from_args(args)


def _post_filter_names(args: argparse.Namespace) -> list[str]:
    if args.steger_geometric_filter == "flat-ground":
        return [
            "continuity_filter",
            "undistorted_image_line_ransac",
            "laser_plane_intersection_depth_range",
        ]
    if args.steger_geometric_filter == "continuity-only":
        return ["continuity_filter", "laser_plane_intersection_depth_range"]
    return ["laser_plane_intersection_depth_range"]


def process_laser_image_steger_v2(
    path: Path,
    intrinsics: core.Intrinsics,
    laser_plane: np.ndarray,
    args: argparse.Namespace,
    overlay_path: Path,
) -> core.LaserObservation:
    gray = core.to_gray(core.read_image(path), path)
    expected_shape = (intrinsics.image_size[1], intrinsics.image_size[0])
    if gray.shape != expected_shape:
        raise core.CalibrationError(
            f"Image size {gray.shape[::-1]} does not match intrinsics {intrinsics.image_size}: {path}"
        )

    settings = _settings_from_args(args)
    extracted = steger.extract_steger_columns(gray, settings)
    quality_valid = extracted.valid.copy()
    if args.steger_geometric_filter in ("flat-ground", "continuity-only"):
        quality_valid = steger.continuity_filter_columns(
            extracted.u_px,
            extracted.v_px,
            quality_valid,
            args.continuity_window,
            args.continuity_max_deviation_px,
        )

    quality_indices = np.flatnonzero(quality_valid)
    candidate_pixels = np.column_stack(
        [extracted.u_px[quality_indices], extracted.v_px[quality_indices]]
    )
    if args.steger_geometric_filter == "flat-ground":
        line_inliers, _residuals = steger.line_ransac_filter_undistorted(
            candidate_pixels,
            intrinsics.camera_matrix,
            intrinsics.dist_coeffs,
            args.line_ransac_threshold_px,
        )
        minimum = max(30, int(np.ceil(0.5 * len(candidate_pixels))))
        if int(np.count_nonzero(line_inliers)) < minimum:
            raise core.CalibrationError(
                "Laser line RANSAC has too few inliers: "
                f"{np.count_nonzero(line_inliers)}/{len(candidate_pixels)}"
            )
        quality_valid[quality_indices[~line_inliers]] = False
        candidate_pixels = candidate_pixels[line_inliers]

    points, intersection_valid = core.intersect_laser_plane(
        candidate_pixels,
        intrinsics,
        laser_plane,
        args.min_depth_mm,
        args.max_depth_mm,
    )
    accepted_pixels = candidate_pixels[intersection_valid]
    if len(accepted_pixels) < 30:
        raise core.CalibrationError(f"Too few valid ground laser points: {len(accepted_pixels)}")

    overlay = core.display_bgr(gray)
    rejected = np.flatnonzero(~quality_valid & np.isfinite(extracted.v_px))
    for index in rejected[::4]:
        point = (
            int(round(float(extracted.u_px[index]))),
            int(round(float(extracted.v_px[index]))),
        )
        cv2.circle(overlay, point, 1, (0, 0, 255), -1)
    for point in accepted_pixels:
        cv2.circle(overlay, tuple(np.rint(point).astype(int)), 2, (0, 255, 0), -1)
    label = (
        f"Steger v2 accepted={len(accepted_pixels)}/{len(extracted.u_px)} "
        f"filter={args.steger_geometric_filter}"
    )
    cv2.putText(overlay, label, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    core.write_image(overlay_path, overlay)
    LOGGER.debug("%s Steger metadata: %s", path.name, extracted.metadata)
    return core.LaserObservation(
        path,
        accepted_pixels,
        points,
        len(extracted.u_px),
        int(np.count_nonzero(quality_valid)),
    )


def _metadata(settings: steger.StegerSettings, args: argparse.Namespace) -> dict[str, Any]:
    return steger.settings_metadata(settings, _post_filter_names(args))


def _rewrite_outputs_with_metadata(
    output_dir: Path,
    result: dict[str, Any],
    settings: steger.StegerSettings,
    args: argparse.Namespace,
) -> None:
    result.setdefault("ground_laser", {})
    metadata = _metadata(settings, args)
    result["ground_laser"]["centre_extraction"] = metadata
    result["algorithm_variant"] = {
        "base_module": "calibrate_ground_extrinsics.py",
        "changed_component": "ground laser centre extraction via shared Steger core",
        "centre_extraction": metadata,
    }
    serialised = core._serialisable(result)
    (output_dir / "camera_ground_extrinsics.yaml").write_text(
        yaml.safe_dump(serialised, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    (output_dir / "camera_ground_extrinsics.json").write_text(
        json.dumps(serialised, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    settings = _settings_from_args(args)
    original = core.process_laser_image
    core.process_laser_image = process_laser_image_steger_v2
    try:
        result = core.run(args)
    finally:
        core.process_laser_image = original
    _rewrite_outputs_with_metadata(args.output_dir, result, settings, args)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s: %(message)s",
    )
    try:
        run(args)
    except (
        core.CalibrationError,
        FileNotFoundError,
        NotADirectoryError,
        OSError,
        cv2.error,
        ValueError,
    ) as exc:
        LOGGER.error("Calibration failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
