"""大恒 Galaxy USB3 相机适配器。

gxipy 随 Galaxy SDK 提供，并非本项目的必装依赖。这里延迟加载 SDK，
确保 synthetic/MVS 后端在未安装 Galaxy SDK 的机器上仍可正常使用。
"""

from __future__ import annotations

import ctypes
import importlib
import os
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from .models import CameraConfig, CameraDeviceInfo, CapturedFrame


_SDK_ROOT_ENV_NAMES = ("DAHENG_GALAXY_ROOT", "GALAXY_SDK_ROOT")
_PYTHON_PATH_ENV_NAMES = (
    "DAHENG_GALAXY_PYTHON_PATH",
    "GALAXY_PYTHON_PATH",
)
_DLL_HANDLES: list[object] = []
_DLL_PATHS: set[str] = set()


@dataclass(frozen=True, slots=True)
class _DeviceRecord:
    info: CameraDeviceInfo
    raw: dict[str, Any]


def _pointer_directory() -> str:
    return "Win64" if ctypes.sizeof(ctypes.c_void_p) == 8 else "Win32"


def _genicam_directory_name() -> str:
    return "Win64_x64" if ctypes.sizeof(ctypes.c_void_p) == 8 else "Win32_i86"


def _sdk_root_candidates() -> list[Path]:
    candidates: list[Path] = []
    for name in _SDK_ROOT_ENV_NAMES:
        value = os.environ.get(name)
        if value:
            candidates.append(Path(value))

    genicam_root = os.environ.get("GALAXY_GENICAM_ROOT")
    if genicam_root:
        candidates.append(Path(genicam_root).parent)
    for name in ("PROGRAMFILES", "PROGRAMFILES(X86)"):
        value = os.environ.get(name)
        if value:
            candidates.append(Path(value) / "Daheng Imaging" / "GalaxySDK")
    candidates.append(Path(r"C:\Program Files\Daheng Imaging\GalaxySDK"))

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.resolve(strict=False)).lower()
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def _append_environment_path(name: str, directory: Path) -> None:
    value = str(directory)
    parts = [item for item in os.environ.get(name, "").split(os.pathsep) if item]
    if value not in parts:
        os.environ[name] = os.pathsep.join([value, *parts])


def _configure_sdk_paths(root: Path) -> Path:
    if not root.is_dir():
        raise FileNotFoundError(f"Galaxy SDK 目录不存在: {root}")

    bitness = _pointer_directory()
    genicam_root = root / "GenICam"
    python_dir = root / "Development" / "Samples" / "Python"
    api_dir = root / "APIDll" / bitness
    genicam_bin = genicam_root / "bin" / _genicam_directory_name()
    gentl_dir = root / "GenTL" / bitness
    if not (python_dir / "gxipy").is_dir():
        raise FileNotFoundError(f"Galaxy SDK 缺少 Python gxipy: {python_dir}")
    if not genicam_root.is_dir():
        raise FileNotFoundError(f"Galaxy SDK 缺少 GenICam: {genicam_root}")
    if not api_dir.is_dir() or not genicam_bin.is_dir():
        raise FileNotFoundError(
            f"Galaxy SDK 缺少 {_pointer_directory()} 运行库: {api_dir} / {genicam_bin}"
        )

    os.environ.setdefault("GALAXY_GENICAM_ROOT", str(genicam_root))
    gentl_variable = (
        "GENICAM_GENTL64_PATH"
        if ctypes.sizeof(ctypes.c_void_p) == 8
        else "GENICAM_GENTL32_PATH"
    )
    if gentl_dir.is_dir():
        _append_environment_path(gentl_variable, gentl_dir)
    _append_environment_path("PATH", api_dir)
    _append_environment_path("PATH", genicam_bin)
    if hasattr(os, "add_dll_directory"):
        for directory in (api_dir, genicam_bin):
            key = str(directory.resolve(strict=False)).lower()
            if key not in _DLL_PATHS:
                _DLL_HANDLES.append(os.add_dll_directory(str(directory)))
                _DLL_PATHS.add(key)

    for name in _PYTHON_PATH_ENV_NAMES:
        configured = os.environ.get(name)
        if configured:
            python_path = Path(configured)
            if python_path.is_dir() and str(python_path) not in sys.path:
                sys.path.insert(0, str(python_path))
    if str(python_dir) not in sys.path:
        sys.path.insert(0, str(python_dir))
    return root


def _load_gxipy() -> ModuleType:
    errors: list[str] = []
    required = (
        "DeviceManager",
        "GxDeviceClassList",
        "GxFrameStatusList",
        "GxPixelFormatEntry",
    )
    for root in _sdk_root_candidates():
        try:
            _configure_sdk_paths(root)
            sdk = importlib.import_module("gxipy")
        except Exception as error:  # gxipy/底层 DLL 可能抛出多种异常
            errors.append(f"{root}: {error}")
            continue
        missing = [name for name in required if not hasattr(sdk, name)]
        if not missing:
            return sdk
        errors.append(f"{root}: gxipy 缺少 {', '.join(missing)}")

    try:
        sdk = importlib.import_module("gxipy")
    except Exception as error:
        errors.append(f"sys.path: {error}")
    else:
        missing = [name for name in required if not hasattr(sdk, name)]
        if not missing:
            return sdk
        errors.append(f"sys.path: gxipy 缺少 {', '.join(missing)}")

    detail = "; ".join(errors) or "没有找到 Galaxy SDK"
    raise RuntimeError(
        "无法加载大恒 Galaxy SDK 的 gxipy。请确认已安装完整 SDK，或设置 "
        "DAHENG_GALAXY_ROOT/DAHENG_GALAXY_PYTHON_PATH。"
        f"\n详情: {detail}"
    )


def _feature_available(feature: object, method_name: str) -> bool:
    method = getattr(feature, method_name, None)
    return True if method is None else bool(method())


def _get_feature_object(camera: object, name: str, *, required: bool = True) -> object | None:
    feature = getattr(camera, name, None)
    if feature is None or not _feature_available(feature, "is_implemented"):
        if required:
            raise RuntimeError(f"当前大恒相机不支持必需节点 {name}")
        return None
    return feature


def _set_feature(camera: object, name: str, value: object) -> None:
    feature = _get_feature_object(camera, name)
    assert feature is not None
    if not _feature_available(feature, "is_writable"):
        raise RuntimeError(f"当前大恒相机节点 {name} 不可写")
    try:
        feature.set(value)
    except Exception as error:
        raise RuntimeError(f"设置大恒相机节点 {name}={value!r} 失败: {error}") from error


def _set_optional_feature(camera: object, name: str, value: object) -> None:
    feature = _get_feature_object(camera, name, required=False)
    if feature is None or not _feature_available(feature, "is_writable"):
        return
    try:
        feature.set(value)
    except Exception:
        return


def _get_feature(camera: object, name: str, default: object) -> object:
    feature = _get_feature_object(camera, name, required=False)
    if feature is None or not _feature_available(feature, "is_readable"):
        return default
    try:
        return feature.get()
    except Exception:
        return default


def _enum_value(sdk: ModuleType, class_name: str, member: str, fallback: object) -> object:
    enum_class = getattr(sdk, class_name, None)
    return getattr(enum_class, member, fallback)


def _pixel_format_value(sdk: ModuleType, pixel_format: str) -> object:
    enum_class = getattr(sdk, "GxPixelFormatEntry", None)
    value = getattr(enum_class, pixel_format.upper(), None)
    if value is None:
        raise RuntimeError(f"当前 gxipy 不提供像素格式 {pixel_format}")
    return value


def _apply_config(sdk: ModuleType, camera: object, config: CameraConfig) -> CameraConfig:
    _set_optional_feature(camera, "ExposureAuto", _enum_value(sdk, "GxAutoEntry", "OFF", 0))
    _set_optional_feature(camera, "GainAuto", _enum_value(sdk, "GxAutoEntry", "OFF", 0))
    _set_feature(camera, "TriggerMode", _enum_value(sdk, "GxSwitchEntry", "OFF", "Off"))
    _set_feature(camera, "PixelFormat", _pixel_format_value(sdk, config.pixel_format))
    # 先清零偏移，避免旧 ROI 使新宽高越过传感器边界。
    _set_feature(camera, "OffsetX", 0)
    _set_feature(camera, "OffsetY", 0)
    _set_feature(camera, "Width", config.width)
    _set_feature(camera, "Height", config.height)
    _set_feature(camera, "OffsetX", config.offset_x)
    _set_feature(camera, "OffsetY", config.offset_y)
    _set_feature(camera, "ExposureTime", config.exposure_us)
    _set_feature(camera, "Gain", config.gain_db)
    return CameraConfig(
        exposure_us=float(_get_feature(camera, "ExposureTime", config.exposure_us)),
        gain_db=float(_get_feature(camera, "Gain", config.gain_db)),
        pixel_format=config.pixel_format,
        offset_x=int(_get_feature(camera, "OffsetX", config.offset_x)),
        offset_y=int(_get_feature(camera, "OffsetY", config.offset_y)),
        width=int(_get_feature(camera, "Width", config.width)),
        height=int(_get_feature(camera, "Height", config.height)),
        timeout_ms=config.timeout_ms,
    )


def _enumerate_u3v(sdk: ModuleType, manager: object, timeout_ms: int) -> list[_DeviceRecord]:
    try:
        count, device_info = manager.update_all_device_list(timeout_ms)
    except Exception as error:
        raise RuntimeError(f"枚举大恒 USB3 相机失败: {error}") from error
    if not device_info or int(count) <= 0:
        return []

    u3v_class = int(getattr(getattr(sdk, "GxDeviceClassList"), "U3V", 3))
    records: list[_DeviceRecord] = []
    for item in device_info:
        try:
            device_class = int(item.get("device_class", -1))
        except (TypeError, ValueError):
            continue
        if device_class != u3v_class:
            continue
        serial = str(item.get("sn", "")).strip()
        if not serial:
            continue
        records.append(
            _DeviceRecord(
                info=CameraDeviceInfo(
                    model=str(item.get("model_name", "UNKNOWN")).strip() or "UNKNOWN",
                    serial_number=serial,
                    transport="USB3",
                ),
                raw=dict(item),
            )
        )
    return records


def _select_device(records: list[_DeviceRecord], serial_number: str) -> _DeviceRecord:
    serial = serial_number.strip() if serial_number else ""
    matching = [record for record in records if not serial or record.info.serial_number == serial]
    if len(matching) == 1:
        return matching[0]
    discovered = ", ".join(record.info.display_name for record in records) or "无"
    if not matching:
        raise RuntimeError(f"未找到大恒 USB3 相机 {serial!r}。已发现: {discovered}")
    raise RuntimeError(f"发现多台大恒 USB3 相机，请选择序列号。已发现: {discovered}")


def _call_or_default(instance: object, method_name: str, default: object) -> object:
    method = getattr(instance, method_name, None)
    if method is None:
        return default
    try:
        return method()
    except Exception:
        return default


class DahengCameraProvider:
    def __init__(self, discovery_timeout_ms: int = 1000) -> None:
        if discovery_timeout_ms <= 0:
            raise ValueError("discovery_timeout_ms 必须为正数")
        self.discovery_timeout_ms = int(discovery_timeout_ms)

    def list_devices(self) -> list[CameraDeviceInfo]:
        sdk = _load_gxipy()
        manager = sdk.DeviceManager()
        return [
            record.info
            for record in _enumerate_u3v(sdk, manager, self.discovery_timeout_ms)
        ]

    def open(self, serial_number: str, config: CameraConfig) -> "DahengCameraSession":
        return DahengCameraSession.open(serial_number, config)


class DahengCameraSession:
    def __init__(
        self,
        sdk: ModuleType,
        manager: object,
        camera: object,
        device: CameraDeviceInfo,
        config: CameraConfig,
    ) -> None:
        self.sdk = sdk
        self._manager = manager
        self.camera = camera
        self.device = device
        self.config = config
        self.network_packet_size: int | None = None
        self._started = False
        self._closed = False

    @classmethod
    def open(cls, serial_number: str, config: CameraConfig) -> "DahengCameraSession":
        sdk = _load_gxipy()
        manager = sdk.DeviceManager()
        camera: object | None = None
        try:
            selected = _select_device(
                _enumerate_u3v(sdk, manager, config.timeout_ms), serial_number
            )
            camera = manager.open_device_by_sn(selected.info.serial_number)
            if camera is None:
                raise RuntimeError(f"无法打开大恒相机 {selected.info.serial_number}")
            applied = _apply_config(sdk, camera, config)
            return cls(sdk, manager, camera, selected.info, applied)
        except Exception:
            if camera is not None:
                try:
                    camera.close_device()
                except Exception:
                    pass
            raise

    def configure(self, config: CameraConfig) -> CameraConfig:
        if self._closed:
            raise RuntimeError("相机已经关闭")
        if self._started:
            raise RuntimeError("请先停止取流，再修改采集参数")
        self.config = _apply_config(self.sdk, self.camera, config)
        return self.config

    def update_exposure_gain(self, exposure_us: float, gain_db: float) -> CameraConfig:
        if self._closed:
            raise RuntimeError("相机已经关闭")
        _set_feature(self.camera, "ExposureTime", float(exposure_us))
        _set_feature(self.camera, "Gain", float(gain_db))
        self.config = replace(
            self.config,
            exposure_us=float(_get_feature(self.camera, "ExposureTime", exposure_us)),
            gain_db=float(_get_feature(self.camera, "Gain", gain_db)),
        )
        return self.config

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("相机已经关闭")
        if not self._started:
            self.camera.stream_on()
            self._started = True

    def get_frame(self, timeout_ms: int | None = None) -> CapturedFrame:
        if self._closed:
            raise RuntimeError("相机已经关闭")
        if not self._started:
            raise RuntimeError("相机尚未开始取流")
        timeout = timeout_ms if timeout_ms is not None else self.config.timeout_ms
        raw_image = self.camera.data_stream[0].get_image(timeout)
        if raw_image is None:
            raise TimeoutError(f"等待大恒相机图像超时：{timeout} ms")

        status = int(_call_or_default(raw_image, "get_status", 0))
        success = int(getattr(getattr(self.sdk, "GxFrameStatusList"), "SUCCESS", 0))
        if status != success:
            raise RuntimeError(f"Galaxy SDK 返回异常帧，status={status}")
        numpy_image = raw_image.get_numpy_array()
        if numpy_image is None:
            raise RuntimeError("Galaxy SDK 返回空的 NumPy 图像")
        image = np.array(numpy_image, copy=True)
        if image.ndim != 2:
            raise RuntimeError(f"期望 Mono 二维图像，Galaxy SDK 实际返回 shape={image.shape}")
        expected_shape = (self.config.height, self.config.width)
        if image.shape != expected_shape:
            raise RuntimeError(
                f"相机返回 {image.shape[1]}×{image.shape[0]}，期望 "
                f"{self.config.width}×{self.config.height}"
            )
        expected_dtype = np.dtype(
            np.uint8 if self.config.pixel_format == "Mono8" else np.uint16
        )
        if image.dtype != expected_dtype:
            raise RuntimeError(
                f"像素格式 {self.config.pixel_format} 返回 dtype={image.dtype}，"
                f"期望 {expected_dtype}"
            )
        if self.config.pixel_format == "Mono12" and image.size:
            maximum = int(image.max())
            if maximum > 4095:
                if maximum <= 65520 and np.all((image & 0x0F) == 0):
                    image = np.right_shift(image, 4).astype(np.uint16, copy=False)
                else:
                    raise RuntimeError("Mono12 数据格式无法识别，请确认未选择 Mono12Packed")

        timestamp_value = _call_or_default(raw_image, "get_timestamp", None)
        return CapturedFrame(
            image=image,
            camera_frame_number=int(_call_or_default(raw_image, "get_frame_id", 0)),
            camera_timestamp_ticks=None if timestamp_value is None else int(timestamp_value),
            host_timestamp_ns=time.time_ns(),
            host_monotonic_ns=time.perf_counter_ns(),
            offset_x=self.config.offset_x,
            offset_y=self.config.offset_y,
        )

    def stop(self) -> None:
        if self._started:
            self.camera.stream_off()
            self._started = False

    def close(self) -> None:
        if self._closed:
            return
        errors: list[str] = []
        if self._started:
            try:
                self.stop()
            except Exception as error:
                errors.append(f"停止大恒相机取流失败: {error}")
                self._started = False
        try:
            self.camera.close_device()
        except Exception as error:
            errors.append(f"关闭大恒相机失败: {error}")
        self._closed = True
        self._manager = None
        if errors:
            raise RuntimeError("；".join(errors))
