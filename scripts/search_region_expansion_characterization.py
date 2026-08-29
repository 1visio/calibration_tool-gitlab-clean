#!/usr/bin/env python3
"""一次性 Stage 2B：真实数据 search-region expansion characterization。

只通过 Stage 1 ``LaserSearchRegion`` 扩展每帧正式 auto region；不修改共享
Steger、GUI、online 或任何生产配置。
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


LEVELS_PX = (0, 8, 16, 32, 48)
ADJACENT_LEVELS = ((8, 16), (16, 32), (32, 48))
DEFAULT_GIT_REF = "feature/phase-a-baseline-distance-laser-angle"
DATA_ROOT = "experiments/geometry_baseline_angle/data"
STABILITY_P95_MAX_PX = 0.01
STABILITY_VALID_FRACTION_DELTA_MAX = 0.01
STABILITY_SAME_FLOOR_FRACTION_MIN = 0.99


@dataclass(frozen=True, slots=True)
class CaseSpec:
    case_id: str
    dataset_id: str
    series: str
    roi_start_u: int | None
    roi_end_u: int | None
    role: str


@dataclass(frozen=True, slots=True)
class FrameLevel:
    center_px: np.ndarray
    valid: np.ndarray
    response: np.ndarray
    region_start_px: float
    region_end_px: float


CASES = (
    CaseSpec(
        "B05_A10_full_scanlines",
        "B05_A10",
        "multiheight",
        None,
        None,
        "Stage 2A boundary-sensitive positive",
    ),
    CaseSpec(
        "B05_A10_boundary_sensitive_h10",
        "B05_A10",
        "multiheight",
        1857,
        1899,
        "isolated boundary-sensitive H10 ROI",
    ),
    CaseSpec(
        "B12p5_A10_normal_reference",
        "B12p5_A10",
        "reference",
        None,
        None,
        "normal negative control",
    ),
    CaseSpec(
        "B05_A10_h1_truncation",
        "B05_A10",
        "multiheight",
        1800,
        1833,
        "known H1 truncation",
    ),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--git-ref", default=DEFAULT_GIT_REF)
    parser.add_argument("--repo", type=Path, default=root)
    parser.add_argument(
        "--calibration-src",
        type=Path,
        default=root.parent / "calibration" / "src",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "experiments" / "search_region_expansion_characterization",
    )
    parser.add_argument("--max-frames", type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames 必须为正数")
    repo = args.repo.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    realtime = load_realtime_steger(args.calibration_src)
    options = realtime.load_steger_options()
    if options["scan_axis"] != "column":
        raise ValueError("本实验真实数据为横向激光，冻结配置必须是 scan_axis=column")

    grouped: dict[tuple[str, str], list[CaseSpec]] = {}
    for case in CASES:
        grouped.setdefault((case.dataset_id, case.series), []).append(case)
    samples: dict[str, dict[int, list[FrameLevel]]] = {
        case.case_id: {level: [] for level in LEVELS_PX} for case in CASES
    }
    source_files: dict[str, list[str]] = {case.case_id: [] for case in CASES}

    for (dataset_id, series), cases in grouped.items():
        prefix = f"{DATA_ROOT}/{dataset_id}/images/{series}"
        paths = git_tiff_paths(repo, args.git_ref, prefix)
        if args.max_frames is not None:
            paths = paths[: args.max_frames]
        if not paths:
            raise FileNotFoundError(f"Git ref 中没有 TIFF：{args.git_ref}:{prefix}")
        for path in paths:
            image = read_git_image(repo, args.git_ref, path)
            formal = realtime.extract_steger(image, options)
            start = _required_int(formal.metadata, "final_search_region_start_px")
            end = _required_int(formal.metadata, "final_search_region_end_px")
            extracted = {0: formal}
            for level in LEVELS_PX[1:]:
                extracted[level] = realtime.extract_steger(
                    image,
                    options,
                    search_region=realtime.LaserSearchRegion(
                        start - level,
                        end + level,
                        source=f"stage2b_formal_plus_{level}px_each_side",
                    ),
                )
            for case in cases:
                source_files[case.case_id].append(path)
                for level, result in extracted.items():
                    samples[case.case_id][level].append(
                        select_frame_level(result, case)
                    )

    level_rows = [
        summarize_level(case, level, samples[case.case_id][level], samples[case.case_id][0])
        for case in CASES
        for level in LEVELS_PX
    ]
    adjacent_rows = [
        summarize_adjacent(
            case,
            before,
            after,
            samples[case.case_id][before],
            samples[case.case_id][after],
        )
        for case in CASES
        for before, after in ADJACENT_LEVELS
    ]
    recommendation = infer_recommendation(level_rows, adjacent_rows)
    write_csv(output / "per_level_metrics.csv", level_rows)
    write_csv(output / "adjacent_shift_metrics.csv", adjacent_rows)
    plot_metrics(output / "center_shift_p95_by_expansion.png", level_rows, adjacent_rows)
    summary = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_ref": args.git_ref,
        "repo": str(repo),
        "calibration_src": str(args.calibration_src.expanduser().resolve()),
        "frozen_steger_options": options,
        "expansion_semantics": "symmetric pixels added to each normal-axis boundary",
        "levels_px_each_side": list(LEVELS_PX),
        "adjacent_levels": [list(pair) for pair in ADJACENT_LEVELS],
        "same_candidate_definition": "floor(normal-axis subpixel center) unchanged",
        "stability_criteria": {
            "adjacent_center_shift_p95_max_px": STABILITY_P95_MAX_PX,
            "adjacent_valid_fraction_delta_max": STABILITY_VALID_FRACTION_DELTA_MAX,
            "same_floor_candidate_fraction_min": STABILITY_SAME_FLOOR_FRACTION_MIN,
            "later_transition_policy": "all available later transitions; at least one",
        },
        "cases": [asdict(case) for case in CASES],
        "source_files": source_files,
        "per_level_metrics": level_rows,
        "adjacent_shift_metrics": adjacent_rows,
        **recommendation,
        "production_integration": False,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(recommendation, ensure_ascii=False))
    return 0


def load_realtime_steger(calibration_src: str | Path) -> Any:
    source = Path(calibration_src).expanduser().resolve()
    expected = source / "realtime_steger.py"
    if not expected.is_file():
        raise FileNotFoundError(expected)
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    module = importlib.import_module("realtime_steger")
    if Path(module.__file__).resolve() != expected:
        raise RuntimeError(f"realtime_steger 来源错误：{module.__file__}")
    return module


def git_tiff_paths(repo: Path, git_ref: str, prefix: str) -> list[str]:
    result = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", git_ref, "--", prefix],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return sorted(
        line.strip().replace("\\", "/")
        for line in result.stdout.splitlines()
        if line.strip().lower().endswith((".tif", ".tiff"))
    )


def read_git_image(repo: Path, git_ref: str, path: str) -> np.ndarray:
    result = subprocess.run(
        ["git", "show", f"{git_ref}:{path}"],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    image = cv2.imdecode(np.frombuffer(result.stdout, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 2:
        raise ValueError(f"无法解码二维灰度图：{git_ref}:{path}")
    return image


def select_frame_level(extraction: Any, case: CaseSpec) -> FrameLevel:
    start = float(extraction.metadata["final_search_region_start_px"])
    end = float(extraction.metadata["final_search_region_end_px"])
    width = len(extraction.valid)
    left = 0 if case.roi_start_u is None else case.roi_start_u
    right = width - 1 if case.roi_end_u is None else case.roi_end_u
    if not 0 <= left <= right < width:
        raise ValueError(f"{case.case_id} ROI 超出图像：[{left}, {right}] vs {width}")
    selected = slice(left, right + 1)
    return FrameLevel(
        center_px=np.asarray(extraction.v_px[selected], dtype=np.float64).copy(),
        valid=np.asarray(extraction.valid[selected], dtype=bool).copy(),
        response=np.asarray(extraction.response[selected], dtype=np.float64).copy(),
        region_start_px=start,
        region_end_px=end,
    )


def summarize_level(
    case: CaseSpec,
    level: int,
    frames: list[FrameLevel],
    formal_frames: list[FrameLevel],
) -> dict[str, Any]:
    opportunity_count = sum(frame.valid.size for frame in frames)
    valid_count = sum(int(np.count_nonzero(frame.valid)) for frame in frames)
    centers = _concat(
        frame.center_px[frame.valid & np.isfinite(frame.center_px)] for frame in frames
    )
    responses = _concat(
        frame.response[frame.valid & np.isfinite(frame.response)] for frame in frames
    )
    clearances = _concat(
        np.minimum(
            frame.center_px[frame.valid & np.isfinite(frame.center_px)] - frame.region_start_px,
            frame.region_end_px - frame.center_px[frame.valid & np.isfinite(frame.center_px)],
        )
        for frame in frames
    )
    paired = pair_samples(formal_frames, frames)
    starts = np.asarray([frame.region_start_px for frame in frames], dtype=np.float64)
    ends = np.asarray([frame.region_end_px for frame in frames], dtype=np.float64)
    return {
        "case_id": case.case_id,
        "role": case.role,
        "dataset_id": case.dataset_id,
        "series": case.series,
        "roi_start_u": case.roi_start_u,
        "roi_end_u": case.roi_end_u,
        "frame_count": len(frames),
        "expansion_each_side_px": level,
        "opportunity_count": opportunity_count,
        "valid_count": valid_count,
        "valid_fraction": valid_count / opportunity_count if opportunity_count else 0.0,
        "region_start_p50_px": _percentile(starts, 50),
        "region_end_p50_px": _percentile(ends, 50),
        "region_size_p50_px": _percentile(ends - starts, 50),
        "boundary_clearance_min_px": _minimum(clearances),
        "boundary_clearance_p05_px": _percentile(clearances, 5),
        "boundary_clearance_p50_px": _percentile(clearances, 50),
        "response_mean": _mean(responses),
        "response_p50": _percentile(responses, 50),
        "response_p95": _percentile(responses, 95),
        "paired_to_formal_count": paired["paired_count"],
        "paired_to_formal_fraction": paired["paired_count"] / opportunity_count if opportunity_count else 0.0,
        "center_shift_vs_formal_p50_px": paired["shift_p50_px"],
        "center_shift_vs_formal_p95_px": paired["shift_p95_px"],
        "center_shift_vs_formal_max_px": paired["shift_max_px"],
        "same_floor_candidate_vs_formal_fraction": paired["same_floor_fraction"],
        "response_ratio_vs_formal_p50": paired["response_ratio_p50"],
        "response_ratio_vs_formal_p95": paired["response_ratio_p95"],
    }


def summarize_adjacent(
    case: CaseSpec,
    before_level: int,
    after_level: int,
    before_frames: list[FrameLevel],
    after_frames: list[FrameLevel],
) -> dict[str, Any]:
    opportunity_count = sum(frame.valid.size for frame in before_frames)
    paired = pair_samples(before_frames, after_frames)
    before_valid = sum(int(np.count_nonzero(frame.valid)) for frame in before_frames)
    after_valid = sum(int(np.count_nonzero(frame.valid)) for frame in after_frames)
    return {
        "case_id": case.case_id,
        "role": case.role,
        "from_expansion_px": before_level,
        "to_expansion_px": after_level,
        "opportunity_count": opportunity_count,
        "paired_count": paired["paired_count"],
        "paired_fraction": paired["paired_count"] / opportunity_count if opportunity_count else 0.0,
        "from_valid_fraction": before_valid / opportunity_count if opportunity_count else 0.0,
        "to_valid_fraction": after_valid / opportunity_count if opportunity_count else 0.0,
        "valid_fraction_delta": (
            (after_valid - before_valid) / opportunity_count if opportunity_count else 0.0
        ),
        "center_shift_p50_px": paired["shift_p50_px"],
        "center_shift_p95_px": paired["shift_p95_px"],
        "center_shift_max_px": paired["shift_max_px"],
        "same_floor_candidate_fraction": paired["same_floor_fraction"],
        "response_ratio_p50": paired["response_ratio_p50"],
        "response_ratio_p95": paired["response_ratio_p95"],
    }


def pair_samples(before_frames: list[FrameLevel], after_frames: list[FrameLevel]) -> dict[str, Any]:
    if len(before_frames) != len(after_frames):
        raise ValueError("paired frame 数量不一致")
    before_centers: list[np.ndarray] = []
    after_centers: list[np.ndarray] = []
    before_responses: list[np.ndarray] = []
    after_responses: list[np.ndarray] = []
    for before, after in zip(before_frames, after_frames, strict=True):
        paired = (
            before.valid
            & after.valid
            & np.isfinite(before.center_px)
            & np.isfinite(after.center_px)
            & np.isfinite(before.response)
            & np.isfinite(after.response)
        )
        before_centers.append(before.center_px[paired])
        after_centers.append(after.center_px[paired])
        before_responses.append(before.response[paired])
        after_responses.append(after.response[paired])
    first = _concat(before_centers)
    second = _concat(after_centers)
    first_response = _concat(before_responses)
    second_response = _concat(after_responses)
    if not first.size:
        return {
            "paired_count": 0,
            "shift_p50_px": None,
            "shift_p95_px": None,
            "shift_max_px": None,
            "same_floor_fraction": None,
            "response_ratio_p50": None,
            "response_ratio_p95": None,
        }
    shifts = np.abs(second - first)
    ratios = np.divide(
        second_response,
        first_response,
        out=np.full_like(second_response, np.nan),
        where=first_response > 0.0,
    )
    ratios = ratios[np.isfinite(ratios)]
    return {
        "paired_count": int(first.size),
        "shift_p50_px": _percentile(shifts, 50),
        "shift_p95_px": _percentile(shifts, 95),
        "shift_max_px": _maximum(shifts),
        "same_floor_fraction": float(np.mean(np.floor(first) == np.floor(second))),
        "response_ratio_p50": _percentile(ratios, 50),
        "response_ratio_p95": _percentile(ratios, 95),
    }


def infer_recommendation(
    level_rows: list[dict[str, Any]],
    adjacent_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    by_level = {(row["case_id"], row["expansion_each_side_px"]): row for row in level_rows}
    by_pair = {
        (row["case_id"], row["from_expansion_px"], row["to_expansion_px"]): row
        for row in adjacent_rows
    }
    evidence_cases = [case.case_id for case in CASES]
    for candidate in (8, 16, 32):
        later_pairs = [pair for pair in ADJACENT_LEVELS if pair[0] >= candidate]
        if not later_pairs:
            continue
        stable = True
        for case_id in evidence_cases:
            for before, after in later_pairs:
                row = by_pair[(case_id, before, after)]
                if (
                    row["center_shift_p95_px"] is None
                    or row["center_shift_p95_px"] > STABILITY_P95_MAX_PX
                    or abs(row["valid_fraction_delta"]) > STABILITY_VALID_FRACTION_DELTA_MAX
                    or row["same_floor_candidate_fraction"] is None
                    or row["same_floor_candidate_fraction"] < STABILITY_SAME_FLOOR_FRACTION_MIN
                ):
                    stable = False
        if stable:
            clearances = [
                by_level[(case_id, candidate)]["boundary_clearance_p05_px"]
                for case_id in evidence_cases
                if by_level[(case_id, candidate)]["boundary_clearance_p05_px"] is not None
            ]
            if len(clearances) == len(evidence_cases):
                return {
                    "characterization_complete": True,
                    "recommendation_status": "supported",
                    "recommended_minimum_safe_clearance_px": int(math.ceil(min(clearances))),
                    "minimum_stable_expansion_each_side_px": candidate,
                    "minimum_observed_boundary_clearance_p05_px": min(clearances),
                }
    return {
        "characterization_complete": False,
        "recommendation_status": "inconclusive",
        "recommended_minimum_safe_clearance_px": None,
        "minimum_stable_expansion_each_side_px": None,
        "minimum_observed_boundary_clearance_p05_px": None,
    }


def plot_metrics(
    path: Path,
    level_rows: list[dict[str, Any]],
    adjacent_rows: list[dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(len(CASES), 1, figsize=(9, 10), sharex=True)
    for axis, case in zip(axes, CASES, strict=True):
        levels = [row for row in level_rows if row["case_id"] == case.case_id]
        adjacent = [row for row in adjacent_rows if row["case_id"] == case.case_id]
        axis.plot(
            [row["expansion_each_side_px"] for row in levels],
            [_nan(row["center_shift_vs_formal_p95_px"]) for row in levels],
            marker="o",
            label="vs formal P95",
        )
        axis.plot(
            [row["to_expansion_px"] for row in adjacent],
            [_nan(row["center_shift_p95_px"]) for row in adjacent],
            marker="s",
            label="adjacent P95",
        )
        axis.axhline(STABILITY_P95_MAX_PX, color="tab:red", linestyle="--", linewidth=1)
        axis.set_title(case.case_id)
        axis.set_ylabel("abs shift P95 (px)")
        axis.grid(alpha=0.25)
        axis.legend()
    axes[-1].set_xlabel("symmetric expansion at each boundary (px)")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("CSV rows 不能为空")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _required_int(metadata: dict[str, Any], key: str) -> int:
    value = metadata.get(key)
    if value is None or not math.isfinite(float(value)):
        raise ValueError(f"正式 extraction 缺少 {key}")
    return int(round(float(value)))


def _concat(values: Iterable[np.ndarray]) -> np.ndarray:
    arrays = [np.asarray(value, dtype=np.float64) for value in values]
    arrays = [value for value in arrays if value.size]
    return np.concatenate(arrays) if arrays else np.empty(0, dtype=np.float64)


def _percentile(values: np.ndarray, percentile: float) -> float | None:
    return float(np.percentile(values, percentile)) if values.size else None


def _minimum(values: np.ndarray) -> float | None:
    return float(np.min(values)) if values.size else None


def _maximum(values: np.ndarray) -> float | None:
    return float(np.max(values)) if values.size else None


def _mean(values: np.ndarray) -> float | None:
    return float(np.mean(values)) if values.size else None


def _nan(value: Any) -> float:
    return float("nan") if value is None else float(value)


if __name__ == "__main__":
    raise SystemExit(main())
