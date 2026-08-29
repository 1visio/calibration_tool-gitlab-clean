from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ..errors import ConfigError
from .daheng import DahengCameraProvider
from .models import CameraProvider
from .mvs import MvsCameraProvider
from .synthetic import SyntheticCameraProvider


def build_camera_provider(
    backend: str,
    *,
    calibration_src: str | Path,
    backend_options: Mapping[str, Any] | None = None,
) -> CameraProvider:
    options = dict(backend_options or {})
    if backend == "mvs":
        if options:
            raise ConfigError(f"MVS 后端不支持这些 backend_options：{sorted(options)}")
        return MvsCameraProvider(calibration_src)
    if backend == "daheng":
        allowed = {"discovery_timeout_ms"}
        unknown = set(options) - allowed
        if unknown:
            raise ConfigError(f"daheng 后端未知参数：{sorted(unknown)}")
        try:
            return DahengCameraProvider(**options)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"daheng 后端参数无效：{exc}") from exc
    if backend == "synthetic":
        allowed = {"target_fps", "seed"}
        unknown = set(options) - allowed
        if unknown:
            raise ConfigError(f"synthetic 后端未知参数：{sorted(unknown)}")
        try:
            return SyntheticCameraProvider(**options)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"synthetic 后端参数无效：{exc}") from exc
    raise ConfigError(f"未知相机后端：{backend}")
