from __future__ import annotations

import importlib
import time
from typing import Any

import numpy as np

from ..models import Frame, FrameMetadata


class DahengUsb3FrameSource:
    """使用大恒 Galaxy SDK 同步采集一帧 USB3 黑白图像。"""

    def __init__(
        self,
        serial_number: str | None = None,
        device_index: int | None = None,
        width: int = 4096,
        height: int = 3000,
        offset_x_px: int = 0,
        offset_y_px: int = 0,
        full_width_px: int = 4096,
        full_height_px: int = 3000,
        pixel_format: str = "Mono8",
        exposure_us: float = 2000.0,
        gain_db: float = 0.0,
        discovery_timeout_ms: int = 1000,
        capture_timeout_ms: int = 3000,
    ) -> None:
        normalized_serial = serial_number.strip() if serial_number else None
        if normalized_serial is None and device_index is None:
            raise ValueError("serial_number 和 device_index 至少设置一个")
        if device_index is not None and device_index < 1:
            raise ValueError("大恒设备索引从 1 开始")
        if width <= 0 or height <= 0:
            raise ValueError("采集 Width/Height 必须为正数")
        if offset_x_px < 0 or offset_y_px < 0:
            raise ValueError("相机 OffsetX/OffsetY 不能为负数")
        if offset_x_px + width > full_width_px:
            raise ValueError("硬件 ROI 在 x 方向超出完整图像")
        if offset_y_px + height > full_height_px:
            raise ValueError("硬件 ROI 在 y 方向超出完整图像")
        if pixel_format not in {"Mono8", "Mono12"}:
            raise ValueError("当前大恒适配器仅支持 Mono8 或 Mono12")
        if exposure_us <= 0.0:
            raise ValueError("exposure_us 必须大于零")
        if gain_db < 0.0:
            raise ValueError("gain_db 不能为负数")
        if discovery_timeout_ms <= 0 or capture_timeout_ms <= 0:
            raise ValueError("设备发现和取帧超时必须大于零")

        self.serial_number = normalized_serial
        self.device_index = device_index
        self.width = int(width)
        self.height = int(height)
        self.offset_x_px = int(offset_x_px)
        self.offset_y_px = int(offset_y_px)
        self.full_width_px = int(full_width_px)
        self.full_height_px = int(full_height_px)
        self.pixel_format = pixel_format
        self.exposure_us = float(exposure_us)
        self.gain_db = float(gain_db)
        self.discovery_timeout_ms = int(discovery_timeout_ms)
        self.capture_timeout_ms = int(capture_timeout_ms)

    def capture(self) -> Frame:
        gx = _load_gxipy()
        manager = gx.DeviceManager()
        device_count, device_info = manager.update_device_list(
            self.discovery_timeout_ms
        )
        if device_count == 0:
            raise RuntimeError("Galaxy SDK 未发现大恒相机；请检查 USB3、驱动和相机供电")

        selected_info = self._select_device_info(device_info)
        camera = None
        stream_started = False
        try:
            camera = self._open_camera(manager)
            self._configure_camera(camera, gx)
            camera.stream_on()
            stream_started = True

            raw_image = camera.data_stream[0].get_image(self.capture_timeout_ms)
            if raw_image is None:
                raise TimeoutError(
                    f"等待大恒相机图像超时：{self.capture_timeout_ms} ms"
                )
            _raise_if_incomplete(raw_image, gx)
            numpy_image = raw_image.get_numpy_array()
            if numpy_image is None:
                raise RuntimeError("Galaxy SDK 返回空的 NumPy 图像")

            image = np.array(numpy_image, copy=True)
            if image.ndim != 2:
                raise RuntimeError(
                    f"期望 Mono 二维图像，Galaxy SDK 实际返回 shape={image.shape}"
                )

            actual_height, actual_width = image.shape
            actual_offset_x = int(
                _get_feature(camera, "OffsetX", self.offset_x_px)
            )
            actual_offset_y = int(
                _get_feature(camera, "OffsetY", self.offset_y_px)
            )
            frame_id = int(_call_or_default(raw_image, "get_frame_id", 0))

            return Frame(
                image=image,
                metadata=FrameMetadata(
                    frame_id=frame_id,
                    timestamp_ns=time.time_ns(),
                    width=actual_width,
                    height=actual_height,
                    exposure_us=float(
                        _get_feature(camera, "ExposureTime", self.exposure_us)
                    ),
                    gain_db=float(_get_feature(camera, "Gain", self.gain_db)),
                    pixel_format=self.pixel_format,
                    camera_model=str(selected_info.get("model_name", "UNKNOWN")),
                    serial_number=str(selected_info.get("sn", "UNKNOWN")),
                    sdk_version=str(getattr(gx, "__version__", "unknown")),
                    offset_x_px=actual_offset_x,
                    offset_y_px=actual_offset_y,
                    full_width_px=self.full_width_px,
                    full_height_px=self.full_height_px,
                ),
            )
        finally:
            if camera is not None and stream_started:
                try:
                    camera.stream_off()
                except Exception:
                    pass
            if camera is not None:
                try:
                    camera.close_device()
                except Exception:
                    pass

    def _select_device_info(self, device_info: list[dict[str, Any]]) -> dict[str, Any]:
        if self.serial_number is not None:
            for item in device_info:
                if str(item.get("sn", "")) == self.serial_number:
                    return item
            available = [str(item.get("sn", "UNKNOWN")) for item in device_info]
            raise RuntimeError(
                f"未发现序列号 {self.serial_number!r}；当前设备：{available}"
            )

        assert self.device_index is not None
        if self.device_index > len(device_info):
            raise RuntimeError(
                f"device_index={self.device_index} 超出已发现设备数 {len(device_info)}"
            )
        return device_info[self.device_index - 1]

    def _open_camera(self, manager: Any) -> Any:
        if self.serial_number is not None:
            return manager.open_device_by_sn(self.serial_number)
        assert self.device_index is not None
        return manager.open_device_by_index(self.device_index)

    def _configure_camera(self, camera: Any, gx: Any) -> None:
        _set_optional_feature(camera, "ExposureAuto", gx.GxAutoEntry.OFF)
        _set_optional_feature(camera, "GainAuto", gx.GxAutoEntry.OFF)
        _set_feature(camera, "TriggerMode", gx.GxSwitchEntry.OFF)

        pixel_entry_name = self.pixel_format.upper()
        pixel_entry = getattr(gx.GxPixelFormatEntry, pixel_entry_name, None)
        if pixel_entry is None:
            raise RuntimeError(f"当前 gxipy 不提供像素格式 {self.pixel_format}")
        _set_feature(camera, "PixelFormat", pixel_entry)

        _set_feature(camera, "OffsetX", 0)
        _set_feature(camera, "OffsetY", 0)
        _set_feature(camera, "Width", self.width)
        _set_feature(camera, "Height", self.height)
        _set_feature(camera, "OffsetX", self.offset_x_px)
        _set_feature(camera, "OffsetY", self.offset_y_px)
        _set_feature(camera, "ExposureTime", self.exposure_us)
        _set_feature(camera, "Gain", self.gain_db)


def _load_gxipy() -> Any:
    try:
        return importlib.import_module("gxipy")
    except (ModuleNotFoundError, ImportError, OSError) as exc:
        raise RuntimeError(
            "无法加载大恒 gxipy。请安装 Galaxy Windows SDK，并在 VS Code 中选择"
            "能够 import gxipy 的 Python 解释器。"
        ) from exc


def _set_feature(camera: Any, name: str, value: Any) -> None:
    feature = getattr(camera, name, None)
    if feature is None or not _is_feature_available(feature, "is_implemented"):
        raise RuntimeError(f"当前大恒相机不支持必需节点 {name}")
    if not _is_feature_available(feature, "is_writable"):
        raise RuntimeError(f"当前大恒相机节点 {name} 不可写")
    feature.set(value)


def _set_optional_feature(camera: Any, name: str, value: Any) -> None:
    feature = getattr(camera, name, None)
    if feature is None:
        return
    if not _is_feature_available(feature, "is_implemented"):
        return
    if not _is_feature_available(feature, "is_writable"):
        return
    feature.set(value)


def _get_feature(camera: Any, name: str, default: Any) -> Any:
    feature = getattr(camera, name, None)
    if feature is None or not _is_feature_available(feature, "is_implemented"):
        return default
    try:
        return feature.get()
    except Exception:
        return default


def _is_feature_available(feature: Any, method_name: str) -> bool:
    method = getattr(feature, method_name, None)
    return True if method is None else bool(method())


def _raise_if_incomplete(raw_image: Any, gx: Any) -> None:
    get_status = getattr(raw_image, "get_status", None)
    status_list = getattr(gx, "GxFrameStatusList", None)
    incomplete = getattr(status_list, "INCOMPLETE", None)
    if get_status is not None and incomplete is not None:
        if get_status() == incomplete:
            raise RuntimeError("Galaxy SDK 返回不完整帧")


def _call_or_default(instance: Any, method_name: str, default: Any) -> Any:
    method = getattr(instance, method_name, None)
    return default if method is None else method()
