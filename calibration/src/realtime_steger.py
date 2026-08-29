#!/usr/bin/env python3
"""统一的实时线激光 Steger/Hessian 中心提取器。

该模块是标定和在线测量共同使用的低层 extractor。上层可以继续叠加
棋盘 ROI、连续性、直线 RANSAC 和激光平面相交等几何门控，但不得再实现
另一套中心定位公式。
"""

from __future__ import annotations

import argparse
import math
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import yaml


_SCAN_AXES = ("column", "row")
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "realtime_steger.yaml"
DEFAULT_MINIMUM_SAFE_CLEARANCE_PX = 14


class _DerivativeWorkspace:
    """Thread-local reusable buffers for the five Gaussian derivatives."""

    __slots__ = ("shape", "dtype", "outputs")

    def __init__(self, shape: tuple[int, int], dtype: np.dtype) -> None:
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self.outputs = tuple(
            np.empty(shape, dtype=self.dtype) for _ in range(5)
        )


_WORKSPACE_LOCAL = threading.local()


def _derivative_workspace(
    shape: tuple[int, int], dtype: np.dtype
) -> tuple[np.ndarray, ...]:
    """Return buffers matching the current band without sharing threads."""

    resolved_dtype = np.dtype(dtype)
    cached = getattr(_WORKSPACE_LOCAL, "value", None)
    if (
        cached is None
        or cached.shape != shape
        or cached.dtype != resolved_dtype
    ):
        cached = _DerivativeWorkspace(shape, resolved_dtype)
        _WORKSPACE_LOCAL.value = cached
    return cached.outputs


@dataclass(frozen=True, slots=True)
class StegerParams:
    """实时 extractor 的参数；``sigma`` 是高斯导数尺度（像素）。"""

    sigma: float = 1.5
    threshold: float = 30.0
    deriv_thresh: float = 0.5
    roi_margin: int = 120
    roi_max_height: int = 512
    scan_axis: str = "column"

    def __post_init__(self) -> None:
        if not math.isfinite(self.sigma) or self.sigma <= 0.0:
            raise ValueError("sigma 必须为有限正数")
        if not math.isfinite(self.threshold) or self.threshold < 0.0:
            raise ValueError("threshold 必须为有限非负数")
        if not math.isfinite(self.deriv_thresh) or self.deriv_thresh <= 0.0:
            raise ValueError("deriv_thresh 必须为有限正数")
        if self.roi_margin < 0:
            raise ValueError("roi_margin 不能为负数")
        if self.roi_max_height < 3:
            raise ValueError("roi_max_height 必须 >= 3")
        if self.scan_axis not in _SCAN_AXES:
            raise ValueError(f"scan_axis 必须是 {_SCAN_AXES} 之一")


@dataclass(frozen=True, slots=True)
class LaserSearchRegion:
    """原图中沿激光条纹法向轴的半开搜索区间 ``[start_px, end_px)``。"""

    start_px: int
    end_px: int
    source: str

    def __post_init__(self) -> None:
        for name in ("start_px", "end_px"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"LaserSearchRegion.{name} 必须是整数")
            object.__setattr__(self, name, int(value))
        if self.end_px <= self.start_px:
            raise ValueError(
                "LaserSearchRegion 必须满足 start_px < end_px（半开区间不能为空）"
            )
        source = str(self.source).strip()
        if not source:
            raise ValueError("LaserSearchRegion.source 不能为空")
        object.__setattr__(self, "source", source)

    def clipped(self, extent_px: int) -> LaserSearchRegion:
        """裁到法向轴长度；裁后为空时明确拒绝。"""

        if extent_px <= 0:
            raise ValueError("normal-axis extent 必须为正数")
        start = max(0, min(int(extent_px), self.start_px))
        end = max(0, min(int(extent_px), self.end_px))
        if end <= start:
            raise ValueError(
                "LaserSearchRegion 裁剪到图像法向轴范围后为空："
                f"[{self.start_px}, {self.end_px}) vs extent={extent_px}"
            )
        return LaserSearchRegion(start, end, self.source)


@dataclass(frozen=True, slots=True)
class DetectorSummary:
    """Gaussian/Hessian 前的 normal-axis 强度 detector 摘要。"""

    normal_axis: str
    normal_axis_extent: int
    row_peak: np.ndarray
    row_sum: np.ndarray
    threshold: float
    seed: int | None
    adaptive_threshold: float | None
    active_mask: np.ndarray
    active_intervals: tuple[tuple[int, int], ...]
    seed_active_interval: tuple[int, int] | None
    roi_margin: int
    roi_max_height: int
    margin_before_clip: tuple[int, int] | None
    margin_after_clip: tuple[int, int] | None
    margin_clamped_start: bool
    margin_clamped_end: bool
    roi_max_height_applied: bool
    auto_search_region: LaserSearchRegion | None


@dataclass(frozen=True, slots=True)
class OutsideRegionEvidence:
    """不依赖 Hessian 的 final-region 外 normal-axis 强度证据。"""

    outside_active_intervals: tuple[tuple[int, int], ...]
    outside_peak_intervals: tuple[tuple[int, int], ...]
    outside_active_position_count: int
    outside_peak_position_count: int
    outside_peak_max_intensity_dn: float | None


@dataclass(frozen=True, slots=True)
class ProductionSearchRegionResolution:
    """Shadow resolver 的提议；本阶段不得替换正式 search region。"""

    proposed_search_region: LaserSearchRegion | None
    would_expand: bool
    reason: str


@dataclass(frozen=True, slots=True)
class StegerExtraction:
    """逐列结果；``valid`` 之外的列均不应进入几何拟合。"""

    u_px: np.ndarray
    v_px: np.ndarray
    valid: np.ndarray
    response: np.ndarray
    offset_px: np.ndarray
    normal_y_abs: np.ndarray
    corrected_signal: np.ndarray
    metadata: dict[str, Any]
    diagnostics: StegerColumnDiagnostics | None = None
    detector_summary: DetectorSummary | None = None

    @property
    def pixels(self) -> np.ndarray:
        indices = np.flatnonzero(self.valid & np.isfinite(self.v_px))
        if indices.size == 0:
            return np.empty((0, 2), dtype=np.float64)
        return np.column_stack([self.u_px[indices], self.v_px[indices]])


@dataclass(frozen=True, slots=True)
class StegerColumnDiagnostics:
    """Diagnostic-only per-column gate state; never used to alter formal output."""

    band_top_px: int | None
    band_bottom_px: int | None
    full_image_max_intensity_dn: np.ndarray
    max_intensity_dn: np.ndarray
    max_ridge_response: np.ndarray
    min_subpixel_offset_px: np.ndarray
    intensity_peak_present: np.ndarray
    intensity_peak_outside_detected_band: np.ndarray
    derivative_condition_passed: np.ndarray
    ridge_response_passed: np.ndarray
    subpixel_offset_passed: np.ndarray
    accepted: np.ndarray
    rejection_reason: np.ndarray


def _positive_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise argparse.ArgumentTypeError("必须是有限正数")
    return result


def _nonnegative_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise argparse.ArgumentTypeError("必须是有限非负数")
    return result


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("必须是正整数")
    return result


def params_from_options(options: Mapping[str, Any] | None = None) -> StegerParams:
    """校验配置映射；只接受实时 Steger 的参数名。"""

    resolved = dict(options or {})
    unknown = set(resolved) - {field for field in StegerParams.__dataclass_fields__}
    if unknown:
        raise ValueError(f"实时 steger 不认识的参数: {sorted(unknown)}")
    return StegerParams(**resolved)


def load_steger_options(path: str | Path | None = None) -> dict[str, Any]:
    """加载统一 YAML 配置中的 ``steger`` 段。"""

    config_path = Path(path).expanduser().resolve() if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise FileNotFoundError(f"实时 Steger 配置不存在: {config_path}")
    document = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(document, Mapping):
        raise ValueError(f"实时 Steger 配置根节点必须是映射: {config_path}")
    options = document.get("steger", document.get("options", document))
    if not isinstance(options, Mapping):
        raise ValueError(f"实时 Steger 配置缺少 steger 映射: {config_path}")
    return asdict(params_from_options(options))


def merge_options(
    base: Mapping[str, Any] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """合并配置；``None`` 覆盖值表示“使用 profile 默认值”。"""

    result = dict(base or load_steger_options())
    for key, value in dict(overrides or {}).items():
        if value is not None:
            result[key] = value
    return asdict(params_from_options(result))


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """向标定/V4 CLI 添加可选覆盖项。"""

    parser.add_argument(
        "--steger-config",
        type=Path,
        default=None,
        help=f"统一实时 Steger 配置；默认 {DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument("--steger-sigma", type=_positive_float, default=None)
    parser.add_argument("--steger-threshold", type=_nonnegative_float, default=None)
    parser.add_argument("--steger-deriv-thresh", type=_positive_float, default=None)
    parser.add_argument("--steger-roi-margin", type=int, default=None)
    parser.add_argument("--steger-roi-max-height", type=_positive_int, default=None)
    parser.add_argument("--steger-scan-axis", choices=_SCAN_AXES, default=None)


def options_from_args(args: argparse.Namespace) -> dict[str, Any]:
    profile = load_steger_options(getattr(args, "steger_config", None))
    return merge_options(
        profile,
        {
            "sigma": getattr(args, "steger_sigma", None),
            "threshold": getattr(args, "steger_threshold", None),
            "deriv_thresh": getattr(args, "steger_deriv_thresh", None),
            "roi_margin": getattr(args, "steger_roi_margin", None),
            "roi_max_height": getattr(args, "steger_roi_max_height", None),
            "scan_axis": getattr(args, "steger_scan_axis", None),
        },
    )


def _mask_intervals(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    """把一维 bool mask 转成半开连续区间。"""

    values = np.asarray(mask, dtype=bool).reshape(-1)
    if not values.size or not np.any(values):
        return ()
    transitions = np.diff(np.pad(values.astype(np.int8), (1, 1)))
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    return tuple((int(start), int(end)) for start, end in zip(starts, ends, strict=True))


def _build_detector_summary(gray: np.ndarray, params: StegerParams) -> DetectorSummary:
    """完整保存既有 auto-band detector 的 pre-Hessian 输入和决策。"""

    row_peak = np.max(gray, axis=1)
    row_sum = np.sum(gray, axis=1, dtype=np.float64)
    extent = int(gray.shape[0])
    normal_axis = "v" if params.scan_axis == "column" else "u"
    if float(np.max(row_peak)) < params.threshold:
        return DetectorSummary(
            normal_axis=normal_axis,
            normal_axis_extent=extent,
            row_peak=row_peak,
            row_sum=row_sum,
            threshold=float(params.threshold),
            seed=None,
            adaptive_threshold=None,
            active_mask=np.zeros(extent, dtype=bool),
            active_intervals=(),
            seed_active_interval=None,
            roi_margin=params.roi_margin,
            roi_max_height=params.roi_max_height,
            margin_before_clip=None,
            margin_after_clip=None,
            margin_clamped_start=False,
            margin_clamped_end=False,
            roi_max_height_applied=False,
            auto_search_region=None,
        )

    seed = int(row_sum.argmax())
    adaptive_threshold = max(params.threshold, 0.3 * float(row_peak[seed]))
    active = row_peak >= adaptive_threshold
    intervals = _mask_intervals(active)
    seed_interval = next(
        (interval for interval in intervals if interval[0] <= seed < interval[1]),
        None,
    )
    if seed_interval is None:
        raise RuntimeError("auto-band seed 未落在 active interval 内")
    raw_top, raw_bottom_exclusive = seed_interval
    margin_before_clip = (
        raw_top - params.roi_margin,
        raw_bottom_exclusive + params.roi_margin,
    )
    top = max(0, margin_before_clip[0])
    bottom = min(extent, margin_before_clip[1])
    margin_after_clip = (top, bottom)
    roi_max_height_applied = bottom - top > params.roi_max_height
    if roi_max_height_applied:
        top = max(0, seed - params.roi_max_height // 2)
        bottom = min(extent, top + params.roi_max_height)
        top = max(0, bottom - params.roi_max_height)
    return DetectorSummary(
        normal_axis=normal_axis,
        normal_axis_extent=extent,
        row_peak=row_peak,
        row_sum=row_sum,
        threshold=float(params.threshold),
        seed=seed,
        adaptive_threshold=float(adaptive_threshold),
        active_mask=active,
        active_intervals=intervals,
        seed_active_interval=seed_interval,
        roi_margin=params.roi_margin,
        roi_max_height=params.roi_max_height,
        margin_before_clip=margin_before_clip,
        margin_after_clip=margin_after_clip,
        margin_clamped_start=margin_before_clip[0] < 0,
        margin_clamped_end=margin_before_clip[1] > extent,
        roi_max_height_applied=roi_max_height_applied,
        auto_search_region=LaserSearchRegion(top, bottom, "auto"),
    )


def _detector_trace(summary: DetectorSummary) -> dict[str, Any]:
    seed_interval = summary.seed_active_interval
    auto_region = summary.auto_search_region
    return {
        "raw_candidate_top": seed_interval[0] if seed_interval is not None else None,
        "raw_candidate_bottom": seed_interval[1] if seed_interval is not None else None,
        "margin_before_clip": (
            list(summary.margin_before_clip)
            if summary.margin_before_clip is not None
            else None
        ),
        "margin_after_clip": (
            list(summary.margin_after_clip)
            if summary.margin_after_clip is not None
            else None
        ),
        "roi_max_height_applied": summary.roi_max_height_applied,
        "final_band_top": auto_region.start_px if auto_region is not None else None,
        "final_band_bottom": auto_region.end_px if auto_region is not None else None,
        "seed_row": summary.seed,
        "adaptive_row_threshold": summary.adaptive_threshold,
    }


def _detect_steger_band(
    gray: np.ndarray,
    params: StegerParams,
    diagnostic_trace: dict[str, Any] | None = None,
    detector_summary_out: list[DetectorSummary] | None = None,
) -> tuple[int, int] | None:
    """定位亮条纹带；结果公式保持不变，并可旁路返回 detector summary。"""

    summary = _build_detector_summary(gray, params)
    if diagnostic_trace is not None:
        diagnostic_trace.update(_detector_trace(summary))
    if detector_summary_out is not None:
        detector_summary_out.append(summary)
    region = summary.auto_search_region
    return (region.start_px, region.end_px) if region is not None else None


def _empty_extraction(
    image: np.ndarray,
    options: Mapping[str, Any],
    *,
    diagnostic: bool = False,
    band_metadata: Mapping[str, Any] | None = None,
    detector_summary: DetectorSummary | None = None,
) -> StegerExtraction:
    height, width = image.shape
    nan = np.full(width, np.nan, dtype=np.float64)
    diagnostics = None
    if diagnostic:
        diagnostics = StegerColumnDiagnostics(
            band_top_px=None,
            band_bottom_px=None,
            full_image_max_intensity_dn=np.max(image, axis=0).astype(np.float64),
            max_intensity_dn=np.max(image, axis=0).astype(np.float64),
            max_ridge_response=nan.copy(),
            min_subpixel_offset_px=nan.copy(),
            intensity_peak_present=np.zeros(width, dtype=bool),
            intensity_peak_outside_detected_band=np.zeros(width, dtype=bool),
            derivative_condition_passed=np.zeros(width, dtype=bool),
            ridge_response_passed=np.zeros(width, dtype=bool),
            subpixel_offset_passed=np.zeros(width, dtype=bool),
            accepted=np.zeros(width, dtype=bool),
            rejection_reason=np.full(width, "no_intensity_peak", dtype="U48"),
        )
    return StegerExtraction(
        u_px=np.arange(width, dtype=np.float64),
        v_px=nan.copy(),
        valid=np.zeros(width, dtype=bool),
        response=nan.copy(),
        offset_px=nan.copy(),
        normal_y_abs=nan.copy(),
        corrected_signal=np.zeros((height, width), dtype=np.float32),
        metadata={
            "method": "steger_realtime",
            **{key: float(value) if isinstance(value, (int, float)) else str(value) for key, value in options.items()},
            **dict(band_metadata or {}),
            "valid_column_count": 0.0,
        },
        diagnostics=diagnostics,
        detector_summary=detector_summary,
    )


def _region_from_legacy_bounds(
    bounds: tuple[int, int] | None,
) -> LaserSearchRegion | None:
    if bounds is None:
        return None
    if len(bounds) != 2:
        raise ValueError("additional_band_bounds 必须是 (top, bottom_exclusive)")
    start, end = map(int, bounds)
    if end <= start:
        raise ValueError("additional_band_bounds 不能为空")
    return LaserSearchRegion(start, end, "additional_band_bounds")


def _merge_search_regions(
    auto_region: LaserSearchRegion | None,
    additional_region: LaserSearchRegion | None,
) -> LaserSearchRegion | None:
    if auto_region is None:
        return additional_region
    if additional_region is None:
        return auto_region
    return LaserSearchRegion(
        min(auto_region.start_px, additional_region.start_px),
        max(auto_region.end_px, additional_region.end_px),
        f"{auto_region.source}+{additional_region.source}",
    )


def _outside_region_evidence(
    summary: DetectorSummary,
    region: LaserSearchRegion | None,
    threshold: float,
) -> OutsideRegionEvidence:
    """在 Hessian 前提取 final region 外的 active/threshold-peak 证据。"""

    outside = np.ones(summary.normal_axis_extent, dtype=bool)
    if region is not None:
        clipped = region.clipped(summary.normal_axis_extent)
        outside[clipped.start_px : clipped.end_px] = False
    outside_active = summary.active_mask & outside
    outside_peak = (summary.row_peak >= float(threshold)) & outside
    peak_values = summary.row_peak[outside_peak]
    return OutsideRegionEvidence(
        outside_active_intervals=_mask_intervals(outside_active),
        outside_peak_intervals=_mask_intervals(outside_peak),
        outside_active_position_count=int(np.count_nonzero(outside_active)),
        outside_peak_position_count=int(np.count_nonzero(outside_peak)),
        outside_peak_max_intensity_dn=(
            float(np.max(peak_values)) if peak_values.size else None
        ),
    )


def resolve_production_search_region(
    detector_summary: DetectorSummary,
    current_search_region: LaserSearchRegion | None = None,
    *,
    minimum_safe_clearance_px: int = DEFAULT_MINIMUM_SAFE_CLEARANCE_PX,
) -> ProductionSearchRegionResolution:
    """基于 pre-Steger 强度证据生成 shadow proposal，不修改正式 region。"""

    if not isinstance(detector_summary, DetectorSummary):
        raise TypeError("detector_summary 必须是 DetectorSummary")
    if (
        isinstance(minimum_safe_clearance_px, bool)
        or not isinstance(minimum_safe_clearance_px, (int, np.integer))
        or minimum_safe_clearance_px < 0
    ):
        raise ValueError("minimum_safe_clearance_px 必须是非负整数")
    clearance = int(minimum_safe_clearance_px)
    current = current_search_region or detector_summary.auto_search_region
    if current is None:
        return ProductionSearchRegionResolution(None, False, "no_current_search_region")
    if not isinstance(current, LaserSearchRegion):
        raise TypeError("current_search_region 必须是 LaserSearchRegion 或 None")
    current = current.clipped(detector_summary.normal_axis_extent)
    peak_mask = detector_summary.row_peak >= detector_summary.threshold
    peak_intervals = _mask_intervals(peak_mask)
    if not peak_intervals:
        return ProductionSearchRegionResolution(
            current,
            False,
            "no_significant_intensity_evidence",
        )

    target_start = min(
        current.start_px,
        min(start - clearance for start, _end in peak_intervals),
    )
    target_end = max(
        current.end_px,
        max(end + clearance for _start, end in peak_intervals),
    )
    target_start = max(0, target_start)
    target_end = min(detector_summary.normal_axis_extent, target_end)
    outside = any(
        start < current.start_px or end > current.end_px
        for start, end in peak_intervals
    )
    near_boundary = any(
        start < current.start_px + clearance
        or end > current.end_px - clearance
        for start, end in peak_intervals
    )
    would_expand = (
        target_start < current.start_px or target_end > current.end_px
    )
    if would_expand:
        reason = (
            "significant_intensity_outside_current_region"
            if outside
            else "significant_intensity_near_current_boundary"
        )
        proposed = LaserSearchRegion(
            target_start,
            target_end,
            f"shadow_production_resolver:{reason}",
        )
    elif outside or near_boundary:
        reason = "expansion_limited_by_image_boundary"
        proposed = current
    else:
        reason = "current_region_has_safe_intensity_clearance"
        proposed = current
    return ProductionSearchRegionResolution(proposed, would_expand, reason)


def _shadow_metadata(
    summary: DetectorSummary,
    final_region: LaserSearchRegion | None,
    params: StegerParams,
) -> dict[str, Any]:
    """把 compact detector/evidence/resolution 写入 metadata，不保存大数组副本。"""

    evidence = _outside_region_evidence(summary, final_region, params.threshold)
    resolution = resolve_production_search_region(
        summary,
        final_region,
    )
    proposed = resolution.proposed_search_region
    return {
        "detector_summary_available": True,
        "detector_normal_axis": summary.normal_axis,
        "detector_normal_axis_extent_px": summary.normal_axis_extent,
        "detector_seed_px": summary.seed,
        "detector_adaptive_threshold_dn": summary.adaptive_threshold,
        "detector_threshold_dn": summary.threshold,
        "detector_active_intervals_px": [list(value) for value in summary.active_intervals],
        "detector_seed_active_interval_px": (
            list(summary.seed_active_interval)
            if summary.seed_active_interval is not None
            else None
        ),
        "detector_roi_margin_px": summary.roi_margin,
        "detector_roi_max_height_px": summary.roi_max_height,
        "detector_margin_clamped_start": summary.margin_clamped_start,
        "detector_margin_clamped_end": summary.margin_clamped_end,
        "detector_roi_max_height_applied": summary.roi_max_height_applied,
        "outside_region_active_intervals_px": [
            list(value) for value in evidence.outside_active_intervals
        ],
        "outside_region_peak_intervals_px": [
            list(value) for value in evidence.outside_peak_intervals
        ],
        "outside_region_active_position_count": evidence.outside_active_position_count,
        "outside_region_peak_position_count": evidence.outside_peak_position_count,
        "outside_region_peak_max_intensity_dn": evidence.outside_peak_max_intensity_dn,
        "shadow_resolver_available": True,
        "shadow_minimum_safe_clearance_px": DEFAULT_MINIMUM_SAFE_CLEARANCE_PX,
        "shadow_proposed_search_region_start_px": (
            float(proposed.start_px) if proposed is not None else None
        ),
        "shadow_proposed_search_region_end_px": (
            float(proposed.end_px) if proposed is not None else None
        ),
        "shadow_proposed_search_region_source": (
            proposed.source if proposed is not None else None
        ),
        "shadow_would_expand": resolution.would_expand,
        "shadow_reason": resolution.reason,
    }


def _extract_columnwise(
    gray: np.ndarray,
    params: StegerParams,
    *,
    diagnostic: bool = False,
    additional_search_region: LaserSearchRegion | None = None,
    use_auto_band: bool = True,
) -> StegerExtraction:
    try:
        from scipy import ndimage
    except ImportError as error:
        raise RuntimeError("实时 Steger 需要 scipy；请安装项目 requirements") from error

    # The input is Mono8/Mono12 and the derivative pipeline is dominated by
    # memory bandwidth. Keep the working image and Gaussian/Hessian arrays in
    # float32; public centre coordinates remain float64 below. This halves the
    # working-set size while retaining sub-pixel precision for the output API.
    image = np.asarray(gray, dtype=np.float32)
    auto_band_trace: dict[str, Any] = {}
    detector_summaries: list[DetectorSummary] = []
    original_band_bounds = (
        _detect_steger_band(
            image,
            params,
            diagnostic_trace=auto_band_trace,
            detector_summary_out=detector_summaries,
        )
        if use_auto_band
        else None
    )
    detector_summary = detector_summaries[0] if detector_summaries else None
    auto_region = (
        LaserSearchRegion(*original_band_bounds, source="auto")
        if original_band_bounds is not None
        else None
    )
    reference_region = (
        additional_search_region.clipped(image.shape[0])
        if additional_search_region is not None
        else None
    )
    final_region = _merge_search_regions(auto_region, reference_region)
    reference_band_bounds = (
        (reference_region.start_px, reference_region.end_px)
        if reference_region is not None
        else None
    )
    band_bounds = (
        (final_region.start_px, final_region.end_px)
        if final_region is not None
        else None
    )
    band_metadata = {
        **auto_band_trace,
        "original_band_top_px": (
            float(original_band_bounds[0]) if original_band_bounds is not None else None
        ),
        "original_band_bottom_exclusive_px": (
            float(original_band_bounds[1]) if original_band_bounds is not None else None
        ),
        "reference_envelope_top_px": (
            float(reference_band_bounds[0]) if reference_band_bounds is not None else None
        ),
        "reference_envelope_bottom_exclusive_px": (
            float(reference_band_bounds[1]) if reference_band_bounds is not None else None
        ),
        "final_band_top_px": float(band_bounds[0]) if band_bounds is not None else None,
        "final_band_bottom_exclusive_px": (
            float(band_bounds[1]) if band_bounds is not None else None
        ),
    }
    if detector_summary is not None:
        # Shadow proposal is deliberately metadata-only.  ``band_bounds`` above
        # remains the sole region used by the formal Gaussian/Hessian pass.
        band_metadata.update(_shadow_metadata(detector_summary, final_region, params))
    if band_bounds is None:
        return _empty_extraction(
            image,
            asdict(params),
            diagnostic=diagnostic,
            band_metadata=band_metadata,
            detector_summary=detector_summary,
        )

    top, bottom = band_bounds
    band = np.ascontiguousarray(image[top:bottom])
    ry, rx, ryy, rxx, rxy = _derivative_workspace(band.shape, band.dtype)
    for output, order in zip(
        (ry, rx, ryy, rxx, rxy),
        ((1, 0), (0, 1), (2, 0), (0, 2), (1, 1)),
    ):
        ndimage.gaussian_filter(
            band, params.sigma, order=order, output=output
        )

    root = np.sqrt(((rxx - ryy) * 0.5) ** 2 + rxy**2)
    midpoint = (rxx + ryy) * 0.5
    eigenvalue_1 = midpoint + root
    eigenvalue_2 = midpoint - root
    main_eigenvalue = np.where(
        np.abs(eigenvalue_2) >= np.abs(eigenvalue_1), eigenvalue_2, eigenvalue_1
    )
    candidate_1_x = rxy
    candidate_1_y = main_eigenvalue - rxx
    candidate_2_x = main_eigenvalue - ryy
    candidate_2_y = rxy
    norm_1 = np.hypot(candidate_1_x, candidate_1_y)
    norm_2 = np.hypot(candidate_2_x, candidate_2_y)
    use_first = norm_1 >= norm_2
    normal_x = np.where(use_first, candidate_1_x, candidate_2_x)
    normal_y = np.where(use_first, candidate_1_y, candidate_2_y)
    normal_norm = np.hypot(normal_x, normal_y)
    safe_normal = normal_norm > np.finfo(np.float64).eps
    normal_x = np.divide(normal_x, normal_norm, out=np.zeros_like(normal_x), where=safe_normal)
    normal_y = np.divide(normal_y, normal_norm, out=np.zeros_like(normal_y), where=safe_normal)
    first_derivative = rx * normal_x + ry * normal_y
    second_derivative = (
        rxx * normal_x**2 + 2.0 * rxy * normal_x * normal_y + ryy * normal_y**2
    )
    offset = np.divide(
        -first_derivative,
        second_derivative,
        out=np.full_like(first_derivative, np.nan),
        where=np.abs(second_derivative) > np.finfo(np.float64).eps,
    )
    offset_x = offset * normal_x
    offset_y = offset * normal_y
    valid = (
        safe_normal
        & (second_derivative < -params.deriv_thresh)
        & (np.abs(offset_x) <= 0.6)
        & (np.abs(offset_y) <= 0.6)
        & (band >= params.threshold)
    )
    strength = np.where(valid, -second_derivative, -np.inf)
    best_row = np.argmax(strength, axis=0)
    columns = np.arange(band.shape[1])
    best_strength = strength[best_row, columns]
    has_candidate = np.isfinite(best_strength) & (best_strength > 0.0)
    centre_v = best_row.astype(np.float64) + offset_y[best_row, columns] + top

    height, width = image.shape
    v_px = np.full(width, np.nan, dtype=np.float64)
    response = np.full(width, np.nan, dtype=np.float64)
    offset_px = np.full(width, np.nan, dtype=np.float64)
    normal_y_abs = np.full(width, np.nan, dtype=np.float64)
    v_px[has_candidate] = centre_v[has_candidate]
    response[has_candidate] = -second_derivative[best_row[has_candidate], columns[has_candidate]]
    offset_px[has_candidate] = np.hypot(
        offset_x[best_row[has_candidate], columns[has_candidate]],
        offset_y[best_row[has_candidate], columns[has_candidate]],
    )
    normal_y_abs[has_candidate] = np.abs(normal_y[best_row[has_candidate], columns[has_candidate]])
    diagnostics = None
    if diagnostic:
        intensity_mask = band >= params.threshold
        full_image_max_intensity = np.max(image, axis=0).astype(np.float64)
        band_max_intensity = np.max(band, axis=0).astype(np.float64)
        intensity_peak_outside_detected_band = (
            (full_image_max_intensity >= params.threshold)
            & (band_max_intensity < params.threshold)
        )
        negative_ridge_mask = intensity_mask & safe_normal & (second_derivative < 0.0)
        response_mask = intensity_mask & safe_normal & (
            second_derivative < -params.deriv_thresh
        )
        offset_mask = response_mask & (np.abs(offset_x) <= 0.6) & (np.abs(offset_y) <= 0.6)
        intensity_peak_present = np.any(intensity_mask, axis=0)
        derivative_condition_passed = np.any(negative_ridge_mask, axis=0)
        ridge_response_passed = np.any(response_mask, axis=0)
        subpixel_offset_passed = np.any(offset_mask, axis=0)
        max_response = np.max(
            np.where(negative_ridge_mask, -second_derivative, -np.inf), axis=0
        ).astype(np.float64)
        max_response[~np.isfinite(max_response)] = np.nan
        candidate_offset = np.hypot(offset_x, offset_y)
        min_offset = np.min(
            np.where(response_mask, candidate_offset, np.inf), axis=0
        ).astype(np.float64)
        min_offset[~np.isfinite(min_offset)] = np.nan
        rejection_reason = np.full(width, "other", dtype="U48")
        rejection_reason[has_candidate] = "accepted"
        rejected = ~has_candidate
        rejection_reason[rejected & ~intensity_peak_present] = "no_intensity_peak"
        rejection_reason[
            rejected & intensity_peak_outside_detected_band
        ] = "intensity_peak_outside_detected_band"
        rejection_reason[
            rejected & intensity_peak_present & ~derivative_condition_passed
        ] = "derivative_condition_failed"
        rejection_reason[
            rejected & derivative_condition_passed & ~ridge_response_passed
        ] = "ridge_response_below_threshold"
        rejection_reason[
            rejected & ridge_response_passed & ~subpixel_offset_passed
        ] = "invalid_subpixel_offset"
        rejection_reason[
            rejected & subpixel_offset_passed
        ] = "no_valid_ridge_candidate"
        diagnostics = StegerColumnDiagnostics(
            band_top_px=top,
            band_bottom_px=bottom,
            full_image_max_intensity_dn=full_image_max_intensity,
            max_intensity_dn=band_max_intensity,
            max_ridge_response=max_response,
            min_subpixel_offset_px=min_offset,
            intensity_peak_present=intensity_peak_present,
            intensity_peak_outside_detected_band=intensity_peak_outside_detected_band,
            derivative_condition_passed=derivative_condition_passed,
            ridge_response_passed=ridge_response_passed,
            subpixel_offset_passed=subpixel_offset_passed,
            accepted=has_candidate.copy(),
            rejection_reason=rejection_reason,
        )
    return StegerExtraction(
        u_px=np.arange(width, dtype=np.float64),
        v_px=v_px,
        valid=has_candidate,
        response=response,
        offset_px=offset_px,
        normal_y_abs=normal_y_abs,
        corrected_signal=band.astype(np.float32, copy=False),
        metadata={
            "method": "steger_realtime",
            **{key: float(value) if isinstance(value, (int, float)) else str(value) for key, value in asdict(params).items()},
            **band_metadata,
            "band_top_px": float(top),
            "band_bottom_px": float(bottom),
            "valid_column_count": float(np.count_nonzero(has_candidate)),
        },
        diagnostics=diagnostics,
        detector_summary=detector_summary,
    )


def _params_and_image(
    gray: np.ndarray,
    options: Mapping[str, Any] | StegerParams | None,
) -> tuple[StegerParams, np.ndarray]:
    params = options if isinstance(options, StegerParams) else params_from_options(options)
    image = np.asarray(gray)
    if image.ndim != 2:
        raise ValueError("实时 Steger 只接受二维灰度图像")
    return params, image


def _with_normal_axis_metadata(
    extracted: StegerExtraction,
    params: StegerParams,
    search_region: LaserSearchRegion | None,
) -> StegerExtraction:
    metadata = dict(extracted.metadata)
    metadata["normal_axis"] = "v" if params.scan_axis == "column" else "u"
    metadata["final_search_region_start_px"] = metadata.get("final_band_top_px")
    metadata["final_search_region_end_px"] = metadata.get(
        "final_band_bottom_exclusive_px"
    )
    if search_region is not None:
        metadata.update(
            {
                "additional_search_region_start_px": float(search_region.start_px),
                "additional_search_region_end_px": float(search_region.end_px),
                "additional_search_region_source": search_region.source,
            }
        )
    metadata["valid_scanline_count"] = metadata.get("valid_column_count")
    return StegerExtraction(
        u_px=extracted.u_px,
        v_px=extracted.v_px,
        valid=extracted.valid,
        response=extracted.response,
        offset_px=extracted.offset_px,
        normal_y_abs=extracted.normal_y_abs,
        corrected_signal=extracted.corrected_signal,
        metadata=metadata,
        diagnostics=extracted.diagnostics,
        detector_summary=extracted.detector_summary,
    )


def _restore_row_axis(extracted: StegerExtraction) -> StegerExtraction:
    """把 columnwise 转置域结果恢复为原图 ``(u, v)`` 坐标。"""

    return StegerExtraction(
        u_px=np.ascontiguousarray(extracted.v_px),
        v_px=np.ascontiguousarray(extracted.u_px),
        valid=extracted.valid,
        response=extracted.response,
        offset_px=extracted.offset_px,
        normal_y_abs=extracted.normal_y_abs,
        corrected_signal=np.ascontiguousarray(extracted.corrected_signal.T),
        metadata=dict(extracted.metadata),
        diagnostics=extracted.diagnostics,
        detector_summary=extracted.detector_summary,
    )


def extract_steger(
    gray: np.ndarray,
    options: Mapping[str, Any] | StegerParams | None = None,
    *,
    search_region: LaserSearchRegion | None = None,
    diagnostic: bool = False,
    use_auto_band: bool = True,
) -> StegerExtraction:
    """按原图法向轴 search region 提取，统一处理 column/row 与转置恢复。"""

    params, image = _params_and_image(gray, options)
    if search_region is not None and not isinstance(search_region, LaserSearchRegion):
        raise TypeError("search_region 必须是 LaserSearchRegion 或 None")
    working_image = (
        image
        if params.scan_axis == "column"
        else np.ascontiguousarray(image.T)
    )
    extracted = _extract_columnwise(
        working_image,
        params,
        diagnostic=diagnostic,
        additional_search_region=search_region,
        use_auto_band=use_auto_band,
    )
    if params.scan_axis == "row":
        extracted = _restore_row_axis(extracted)
    return _with_normal_axis_metadata(extracted, params, search_region)


def extract_steger_columns(
    gray: np.ndarray,
    options: Mapping[str, Any] | StegerParams | None = None,
    *,
    diagnostic: bool = False,
    additional_search_region: LaserSearchRegion | None = None,
    additional_band_bounds: tuple[int, int] | None = None,
) -> StegerExtraction:
    """兼容逐列 API；旧 ``additional_band_bounds`` 暂保留为别名。"""

    params, image = _params_and_image(gray, options)
    if params.scan_axis != "column":
        raise ValueError("extract_steger_columns 只支持 scan_axis=column")
    if additional_search_region is not None and additional_band_bounds is not None:
        raise ValueError(
            "additional_search_region 与 additional_band_bounds 不能同时指定"
        )
    if additional_search_region is not None and not isinstance(
        additional_search_region, LaserSearchRegion
    ):
        raise TypeError("additional_search_region 必须是 LaserSearchRegion 或 None")
    resolved_region = additional_search_region or _region_from_legacy_bounds(
        additional_band_bounds
    )
    return _extract_columnwise(
        image,
        params,
        diagnostic=diagnostic,
        additional_search_region=resolved_region,
    )


def steger_backend(
    gray: np.ndarray,
    options: Mapping[str, Any] | StegerParams | None = None,
    *,
    search_region: LaserSearchRegion | None = None,
    use_auto_band: bool = True,
) -> np.ndarray:
    """兼容点数组 API；未传 search region 时结果与既有实现一致。"""

    if search_region is None:
        # 保留旧逐帧入口的轻量路径：不构造统一 API 增加的元数据和诊断包装。
        params, image = _params_and_image(gray, options)
        if params.scan_axis == "column":
            return _extract_columnwise(image, params).pixels
        transposed = _extract_columnwise(
            np.ascontiguousarray(image.T),
            params,
        )
        return transposed.pixels[:, ::-1]
    return extract_steger(
        gray,
        options,
        search_region=search_region,
        use_auto_band=use_auto_band,
    ).pixels


def continuity_filter_columns(
    u_px: np.ndarray,
    v_px: np.ndarray,
    valid: np.ndarray,
    window: int,
    max_deviation_px: float,
) -> np.ndarray:
    if window % 2 == 0:
        raise ValueError("continuity window must be odd")
    result = valid.copy()
    radius = window // 2
    for index in np.flatnonzero(valid):
        left = max(0, index - radius)
        right = min(len(valid), index + radius + 1)
        neighbours = v_px[left:right][valid[left:right]]
        if neighbours.size >= 3 and abs(v_px[index] - np.median(neighbours)) > max_deviation_px:
            result[index] = False
    return result


def points_from_valid_columns(
    u_px: np.ndarray,
    v_px: np.ndarray,
    valid: np.ndarray,
    max_column_gap: float,
    max_vertical_jump: float,
    min_columns: int,
) -> tuple[np.ndarray, dict[str, float]]:
    indices = np.flatnonzero(valid & np.isfinite(v_px))
    if indices.size == 0:
        return np.empty((0, 2), dtype=np.float64), {
            "candidate_point_count": 0.0,
            "raw_segment_count": 0.0,
            "accepted_segment_count": 0.0,
            "extracted_point_count": 0.0,
        }
    candidates = np.column_stack([u_px[indices], v_px[indices]]).astype(np.float64)
    order = np.argsort(candidates[:, 0], kind="mergesort")
    candidates = candidates[order]
    breaks = np.where(
        (np.diff(candidates[:, 0]) > max_column_gap)
        | (np.abs(np.diff(candidates[:, 1])) > max_vertical_jump)
    )[0] + 1
    raw_segments = np.split(np.arange(len(candidates)), breaks)
    accepted = [segment for segment in raw_segments if len(segment) >= min_columns]
    points = (
        np.concatenate([candidates[segment] for segment in accepted], axis=0)
        if accepted
        else np.empty((0, 2), dtype=np.float64)
    )
    return points, {
        "candidate_point_count": float(len(candidates)),
        "raw_segment_count": float(len(raw_segments)),
        "accepted_segment_count": float(len(accepted)),
        "extracted_point_count": float(len(points)),
    }


def line_ransac_filter_undistorted(
    pixels: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    threshold_px: float,
    iterations: int = 500,
    min_inlier_ratio: float = 0.5,
    seed: int = 20260722,
) -> tuple[np.ndarray, np.ndarray]:
    pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    if len(pixels) < 2:
        return np.zeros(len(pixels), dtype=bool), np.full(len(pixels), np.inf)
    ideal = cv2.undistortPoints(
        pixels.reshape(-1, 1, 2), camera_matrix, dist_coeffs, P=camera_matrix
    ).reshape(-1, 2)
    generator = np.random.default_rng(seed)
    best_mask = np.zeros(len(ideal), dtype=bool)
    best_median = math.inf
    for _ in range(iterations):
        first, second = generator.choice(len(ideal), size=2, replace=False)
        direction = ideal[second] - ideal[first]
        length = float(np.linalg.norm(direction))
        if length <= np.finfo(float).eps:
            continue
        direction /= length
        residual = np.abs(
            direction[0] * (ideal[:, 1] - ideal[first, 1])
            - direction[1] * (ideal[:, 0] - ideal[first, 0])
        )
        mask = residual <= threshold_px
        count = int(np.count_nonzero(mask))
        median = float(np.median(residual[mask])) if count else math.inf
        if count > np.count_nonzero(best_mask) or (
            count == np.count_nonzero(best_mask) and median < best_median
        ):
            best_mask, best_median = mask, median
    minimum = max(2, int(math.ceil(min_inlier_ratio * len(ideal))))
    if int(np.count_nonzero(best_mask)) < minimum:
        return np.zeros(len(pixels), dtype=bool), np.full(len(pixels), np.inf)
    centred = ideal[best_mask] - np.mean(ideal[best_mask], axis=0)
    _, _, right = np.linalg.svd(centred, full_matrices=False)
    direction = right[0]
    centre = np.mean(ideal[best_mask], axis=0)
    residual = np.abs(
        direction[0] * (ideal[:, 1] - centre[1])
        - direction[1] * (ideal[:, 0] - centre[0])
    )
    return residual <= threshold_px, residual


def signal_to_u8(signal: np.ndarray, full_scale: float | None = None) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float32)
    if full_scale is None:
        full_scale = float(np.nanmax(values)) if values.size else 0.0
    if full_scale <= 0.0 or not np.isfinite(full_scale):
        return np.zeros(values.shape, dtype=np.uint8)
    return np.clip(values * 255.0 / full_scale, 0.0, 255.0).astype(np.uint8)


def settings_metadata(options: Mapping[str, Any] | StegerParams, post_filters: list[str]) -> dict[str, Any]:
    params = options if isinstance(options, StegerParams) else params_from_options(options)
    return {"method": "steger_realtime", **asdict(params), "post_filters": post_filters}
