"""激光面模型的统一命名、选择和 YAML 兼容层。

拟合脚本和验收/运行配置都通过本模块识别激光面模型，避免一处把旧的
``plane_abcd`` 当成平面、另一处又把它当成未知模型。模型参数本身保持
与测量端相同的相机坐标系和毫米单位。
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import yaml


SUPPORTED_LASER_MODEL_TYPES = (
    "global_plane",
    "quadratic_graph",
    "circular_cone",
)
DEFAULT_LASER_MODEL = "circular_cone"

_MODEL_ALIASES = {
    "global_plane": "global_plane",
    "plane": "global_plane",
    "plane_abcd": "global_plane",
    "laser_plane": "global_plane",
    "quadratic_graph": "quadratic_graph",
    "quadratic": "quadratic_graph",
    "quadratic_surface": "quadratic_graph",
    "circular_cone": "circular_cone",
    "cone": "circular_cone",
}


class LaserModelConfigError(ValueError):
    """激光面模型配置无效。"""


def normalize_model_type(
    value: Any,
    *,
    default: str = DEFAULT_LASER_MODEL,
) -> str:
    """把模型名称归一化为三个正式名称。

    ``plane_abcd``、``plane`` 和 ``laser_plane`` 是历史别名，统一映射到
    ``global_plane``。空值使用默认圆锥模型。
    """

    raw = default if value in (None, "") else value
    if not isinstance(raw, str):
        raise LaserModelConfigError(f"激光面模型名称必须是字符串：{raw!r}")
    key = raw.strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return _MODEL_ALIASES[key]
    except KeyError as exc:
        supported = " / ".join(SUPPORTED_LASER_MODEL_TYPES)
        raise LaserModelConfigError(
            f"不支持的激光面模型 {raw!r}；应为 {supported}"
        ) from exc


def select_default_model(config: Mapping[str, Any] | None) -> str:
    """从拟合配置中读取默认模型，缺省为圆锥模型。

    接受以下写法，按优先级从高到低读取：

    * ``default_model: circular_cone``；
    * ``laser_model: circular_cone``；
    * ``laser_model: {default: circular_cone}``；
    * ``models: {default: circular_cone}``。
    """

    if not isinstance(config, Mapping):
        return DEFAULT_LASER_MODEL

    value: Any = config.get("default_model")
    if value in (None, ""):
        value = config.get("laser_model")
    if isinstance(value, Mapping):
        value = value.get("default", value.get("model_type"))
    if value in (None, ""):
        models = config.get("models")
        if isinstance(models, Mapping):
            value = models.get("default")
    return normalize_model_type(value)


def load_laser_model(path: str | Path) -> dict[str, Any]:
    """读取并规范化激光面模型 YAML。

    没有 ``model_type`` 的旧文件必须包含 ``plane_abcd``、四元
    ``coefficients`` 或 ``plane: {a,b,c,d}``，并会被转换为
    ``global_plane``。
    """

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise LaserModelConfigError(f"激光面模型文件不存在：{source}")
    try:
        document = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise LaserModelConfigError(f"无法读取激光面模型 {source}：{exc}") from exc
    if not isinstance(document, Mapping):
        raise LaserModelConfigError(f"激光面模型顶层必须是映射：{source.name}")
    normalized = normalize_laser_model_document(document)
    normalized["source_path"] = str(source)
    return normalized


def normalize_laser_model_document(
    document: Mapping[str, Any],
    *,
    default: str = DEFAULT_LASER_MODEL,
) -> dict[str, Any]:
    """规范化一个模型映射，并校验三种模型的核心参数。"""

    if not isinstance(document, Mapping):
        raise LaserModelConfigError("激光面模型必须是映射")
    result = dict(document)
    raw_type = result.get("model_type")
    if raw_type in (None, ""):
        plane = _legacy_plane(result)
        if plane is None:
            raise LaserModelConfigError(
                "模型缺少 model_type；旧格式必须提供 plane_abcd / plane / coefficients"
            )
        raw_type = "global_plane"
        result["normal"] = plane[:3].tolist()
        result["d_mm"] = float(plane[3])

    model_type = normalize_model_type(raw_type, default=default)
    result["model_type"] = model_type
    if model_type == "global_plane":
        _normalize_global_plane(result)
    elif model_type == "quadratic_graph":
        _validate_quadratic_graph(result)
    elif model_type == "circular_cone":
        _validate_circular_cone(result)
    return result


def _legacy_plane(document: Mapping[str, Any]) -> np.ndarray | None:
    raw: Any = document.get("plane_abcd")
    if raw is None:
        plane = document.get("plane")
        coefficients = document.get("coefficients")
        if isinstance(plane, Mapping):
            raw = [plane.get(name) for name in ("a", "b", "c", "d")]
        elif coefficients is not None:
            raw = coefficients
    if raw is None:
        return None
    if isinstance(raw, Mapping):
        raw = [raw.get(name) for name in ("a", "b", "c", "d")]
    try:
        array = np.asarray(raw, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise LaserModelConfigError("旧平面配置的 plane_abcd 必须是四个数值") from exc
    if array.size != 4 or not np.isfinite(array).all():
        raise LaserModelConfigError("旧平面配置的 plane_abcd 必须是四个有限数值")
    norm = float(np.linalg.norm(array[:3]))
    if norm <= np.finfo(np.float64).eps:
        raise LaserModelConfigError("旧平面配置的法向量不能为零")
    return array / norm


def _normalize_global_plane(document: dict[str, Any]) -> None:
    if "normal" not in document or "d_mm" not in document:
        plane = _legacy_plane(document)
        if plane is None:
            raise LaserModelConfigError("global_plane 需要 normal/d_mm 或旧 plane_abcd")
        document["normal"] = plane[:3].tolist()
        document["d_mm"] = float(plane[3])
        return
    try:
        normal = np.asarray(document["normal"], dtype=np.float64).reshape(-1)
        d = float(document["d_mm"])
    except (TypeError, ValueError) as exc:
        raise LaserModelConfigError("global_plane 的 normal/d_mm 必须是数值") from exc
    if normal.size != 3 or not np.isfinite(normal).all() or not np.isfinite(d):
        raise LaserModelConfigError("global_plane 的 normal/d_mm 包含无效数值")
    norm = float(np.linalg.norm(normal))
    if norm <= np.finfo(np.float64).eps:
        raise LaserModelConfigError("global_plane.normal 不能为零向量")
    document["normal"] = (normal / norm).tolist()
    document["d_mm"] = float(d / norm)


def _validate_quadratic_graph(document: Mapping[str, Any]) -> None:
    required = ("dependent_axis", "independent_axes", "normalization", "coefficients")
    missing = [key for key in required if key not in document]
    if missing:
        raise LaserModelConfigError(
            f"quadratic_graph 缺少参数：{', '.join(missing)}"
        )
    dependent = str(document["dependent_axis"]).strip().upper()
    independent = document["independent_axes"]
    if not isinstance(independent, (list, tuple)) or len(independent) != 2:
        raise LaserModelConfigError("quadratic_graph.independent_axes 必须包含两个坐标轴")
    axes = [str(item).strip().upper() for item in independent]
    if {dependent, *axes} != {"X", "Y", "Z"}:
        raise LaserModelConfigError(
            "quadratic_graph 的 dependent_axis/independent_axes 必须覆盖 X/Y/Z"
        )
    normalization = document["normalization"]
    if not isinstance(normalization, Mapping):
        raise LaserModelConfigError("quadratic_graph.normalization 必须是映射")
    for name, length in (
        ("independent_center_mm", 2),
        ("independent_scale_mm", 2),
        ("coefficients", 6),
    ):
        value = normalization.get(name) if name != "coefficients" else document[name]
        array = _finite_vector(value, name, length)
        if name == "independent_scale_mm" and np.any(array <= 0.0):
            raise LaserModelConfigError("quadratic_graph.independent_scale_mm 必须为正数")


def _validate_circular_cone(document: Mapping[str, Any]) -> None:
    if document.get("fit_success") is False:
        raise LaserModelConfigError("circular_cone 标记 fit_success=false，不能用于正式标定")
    for name, length in (
        ("axis_unit_camera", 3),
        ("apex_camera_mm", 3),
    ):
        _finite_vector(document.get(name), name, length)
    try:
        angle = float(document["half_apex_angle_deg"])
    except (KeyError, TypeError, ValueError) as exc:
        raise LaserModelConfigError(
            "circular_cone.half_apex_angle_deg 必须是数值"
        ) from exc
    if not np.isfinite(angle) or not 0.0 < angle < 90.0:
        raise LaserModelConfigError(
            "circular_cone.half_apex_angle_deg 必须位于 (0, 90)"
        )
    axis = np.asarray(document["axis_unit_camera"], dtype=np.float64).reshape(-1)
    if np.linalg.norm(axis) <= np.finfo(np.float64).eps:
        raise LaserModelConfigError("circular_cone.axis_unit_camera 不能为零向量")


def _finite_vector(value: Any, name: str, length: int) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise LaserModelConfigError(f"{name} 必须是数值向量") from exc
    if array.size != length or not np.isfinite(array).all():
        raise LaserModelConfigError(f"{name} 必须是 {length} 个有限数值")
    return array
