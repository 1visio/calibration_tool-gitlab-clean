"""GUI 实时 Steger search-region 健康诊断。

本模块只消费正式 ``StegerExtraction`` 的 metadata/diagnostics，不参与
centerline 选择，也不把辅助 warning 合并到 ``FrameQuality.warnings``。
"""

from __future__ import annotations

import importlib
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..laser import normalize_laser_orientation


_OUTSIDE_PEAK_WARNING_FRACTION = 0.01


@dataclass(frozen=True, slots=True)
class SearchRegionHealth:
    normal_axis: str
    search_region_start_px: float | None
    search_region_end_px: float | None
    search_region_size_px: float
    boundary_clearance_min_px: float | None
    boundary_clearance_p05_px: float | None
    boundary_clearance_median_px: float | None
    kernel_support_px: int
    boundary_inside_kernel_fraction: float
    outside_search_region_peak_fraction: float
    status: str
    warning_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["warning_reasons"] = list(self.warning_reasons)
        return payload


def analyze_search_region_health(
    extraction: Any,
    *,
    sigma: float,
) -> SearchRegionHealth:
    """从一次正式 extraction 计算 normal-axis 搜索域健康度。"""

    if not math.isfinite(float(sigma)) or sigma <= 0.0:
        raise ValueError("sigma 必须是有限正数")
    metadata = extraction.metadata
    normal_axis = str(metadata.get("normal_axis", "v"))
    if normal_axis not in {"u", "v"}:
        raise ValueError("StegerExtraction.metadata.normal_axis 必须是 u 或 v")

    start = _optional_finite_float(metadata.get("final_search_region_start_px"))
    end = _optional_finite_float(metadata.get("final_search_region_end_px"))
    if (start is None) != (end is None):
        raise ValueError("Steger search region 的 start/end 必须同时存在或同时为空")
    if start is not None and end is not None and end <= start:
        raise ValueError("Steger search region 必须满足 start < end")

    kernel_support = int(math.ceil(4.0 * float(sigma)))
    clearances = np.empty(0, dtype=np.float64)
    if start is not None and end is not None:
        coordinates = np.asarray(
            extraction.v_px if normal_axis == "v" else extraction.u_px,
            dtype=np.float64,
        )
        valid = np.asarray(extraction.valid, dtype=bool)
        if coordinates.shape != valid.shape:
            raise ValueError("Steger center coordinates 与 valid shape 不一致")
        accepted = coordinates[valid & np.isfinite(coordinates)]
        clearances = np.minimum(accepted - start, end - accepted)

    if clearances.size:
        boundary_min = float(np.min(clearances))
        boundary_p05 = float(np.percentile(clearances, 5))
        boundary_median = float(np.median(clearances))
        inside_fraction = float(np.mean(clearances < kernel_support))
    else:
        boundary_min = None
        boundary_p05 = None
        boundary_median = None
        inside_fraction = 0.0

    diagnostics = extraction.diagnostics
    outside_fraction = 0.0
    if diagnostics is not None:
        outside = np.asarray(
            diagnostics.intensity_peak_outside_detected_band,
            dtype=bool,
        )
        outside_fraction = float(np.mean(outside)) if outside.size else 0.0

    reasons: list[str] = []
    if boundary_p05 is not None and boundary_p05 < kernel_support:
        reasons.append("center_near_search_boundary")
    if outside_fraction > _OUTSIDE_PEAK_WARNING_FRACTION:
        reasons.append("possible_signal_outside_search_region")

    return SearchRegionHealth(
        normal_axis=normal_axis,
        search_region_start_px=start,
        search_region_end_px=end,
        search_region_size_px=(end - start) if start is not None and end is not None else 0.0,
        boundary_clearance_min_px=boundary_min,
        boundary_clearance_p05_px=boundary_p05,
        boundary_clearance_median_px=boundary_median,
        kernel_support_px=kernel_support,
        boundary_inside_kernel_fraction=inside_fraction,
        outside_search_region_peak_fraction=outside_fraction,
        status="WARNING" if reasons else "GOOD",
        warning_reasons=tuple(reasons),
    )


class RealtimeStegerQualityAnalyzer:
    """为 GUI 复用共享正式 extractor 的单次逐帧分析器。"""

    def __init__(
        self,
        calibration_src: str | Path,
        laser_orientation: str,
        *,
        options: Mapping[str, Any] | None = None,
    ) -> None:
        self.module = _load_shared_realtime_steger(calibration_src)
        resolved = dict(options or self.module.load_steger_options())
        orientation = normalize_laser_orientation(laser_orientation)
        resolved["scan_axis"] = "column" if orientation == "horizontal" else "row"
        self.options = self.module.merge_options(resolved)

    def analyze(self, image: np.ndarray) -> dict[str, Any]:
        started = time.perf_counter()
        extraction = self.module.extract_steger(
            image,
            self.options,
            diagnostic=True,
        )
        extracted = time.perf_counter()
        health = analyze_search_region_health(
            extraction,
            sigma=float(self.options["sigma"]),
        ).to_dict()
        completed = time.perf_counter()
        health.update(
            steger_processing_ms=(extracted - started) * 1000.0,
            health_metrics_processing_ms=(completed - extracted) * 1000.0,
            search_region_total_processing_ms=(completed - started) * 1000.0,
        )
        return health


def _load_shared_realtime_steger(calibration_src: str | Path) -> Any:
    source = Path(calibration_src).expanduser().resolve()
    module_path = source / "realtime_steger.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"共享 realtime_steger.py 不存在：{module_path}")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    module = importlib.import_module("realtime_steger")
    loaded_path = Path(module.__file__).resolve()
    if loaded_path != module_path:
        raise RuntimeError(
            "realtime_steger 已从其它目录加载："
            f"期望 {module_path}，实际 {loaded_path}"
        )
    return module


def _optional_finite_float(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Steger search region 坐标必须是有限数")
    return result


__all__ = [
    "RealtimeStegerQualityAnalyzer",
    "SearchRegionHealth",
    "analyze_search_region_health",
]
