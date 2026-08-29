from __future__ import annotations

import argparse
import importlib
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import ModuleType

from .models import CameraConfig, CameraDeviceInfo, CapturedFrame


class MvsCameraProvider:
    """复用 calibration/src 中已验证的 HIKROBOT MVS 实现。"""

    def __init__(self, calibration_src: str | Path) -> None:
        self.calibration_src = Path(calibration_src).expanduser().resolve()
        if not self.calibration_src.is_dir():
            raise RuntimeError(f"calibration/src 不存在：{self.calibration_src}")

    def _module(self) -> ModuleType:
        if str(self.calibration_src) not in sys.path:
            sys.path.insert(0, str(self.calibration_src))
        return importlib.import_module("capture_chessboard_exposure_series")

    def list_devices(self) -> list[CameraDeviceInfo]:
        module = self._module()
        sdk = module.load_mvs_sdk()
        return [
            CameraDeviceInfo(record.model, record.serial_number, record.ip_address)
            for record in module.enumerate_gige_devices(sdk)
        ]

    def open(self, serial_number: str, config: CameraConfig) -> "MvsSessionAdapter":
        return MvsSessionAdapter(self._module(), serial_number, config)


class MvsSessionAdapter:
    def __init__(self, module: ModuleType, serial_number: str, config: CameraConfig) -> None:
        self.module = module
        self.serial_number = serial_number
        self._inner = None
        self._started = False
        self._closed = False
        self.network_packet_size: int | None = None
        self.config = config
        self.device = CameraDeviceInfo("", serial_number)
        self._open_inner(config)

    def _namespace(self, config: CameraConfig) -> argparse.Namespace:
        return argparse.Namespace(
            exposure_us=config.exposure_us,
            gain=config.gain_db,
            pixel_format=config.pixel_format,
            offset_x=config.offset_x,
            offset_y=config.offset_y,
            width=config.width,
            height=config.height,
            timeout_ms=config.timeout_ms,
            serial_number=self.serial_number or None,
        )

    def _open_inner(self, config: CameraConfig) -> None:
        inner = self.module.MvsCameraSession.open(self._namespace(config))
        settings = inner.settings
        self._inner = inner
        self.config = CameraConfig(
            exposure_us=float(settings.exposure_us),
            gain_db=float(settings.gain),
            pixel_format=str(settings.pixel_format),
            offset_x=int(settings.offset_x),
            offset_y=int(settings.offset_y),
            width=int(settings.width),
            height=int(settings.height),
            timeout_ms=config.timeout_ms,
        )
        self.device = CameraDeviceInfo(
            model=str(settings.model),
            serial_number=str(settings.serial_number),
            ip_address=str(settings.ip_address),
        )
        self.network_packet_size = int(settings.packet_size) or None

    def configure(self, config: CameraConfig) -> CameraConfig:
        if self._closed:
            raise RuntimeError("相机已经关闭")
        if self._started:
            raise RuntimeError("请先停止取流，再修改采集参数")
        assert self._inner is not None
        self._inner.close()
        self._open_inner(config)
        return self.config

    def update_exposure_gain(self, exposure_us: float, gain_db: float) -> CameraConfig:
        """取流期间在线写入曝光/增益，并以 SDK 回读值更新当前配置。"""
        if self._closed:
            raise RuntimeError("相机已经关闭")
        assert self._inner is not None
        actual_exposure = self.module._set_float(
            self._inner.cam, self._inner.sdk, "ExposureTime", float(exposure_us)
        )
        actual_gain = self.module._set_float(
            self._inner.cam, self._inner.sdk, "Gain", float(gain_db)
        )
        self._inner.settings = replace(
            self._inner.settings,
            exposure_us=actual_exposure,
            gain=actual_gain,
        )
        self.config = replace(
            self.config,
            exposure_us=actual_exposure,
            gain_db=actual_gain,
        )
        return self.config

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("相机已经关闭")
        assert self._inner is not None
        self._inner.start()
        self._started = True

    def get_frame(self, timeout_ms: int | None = None) -> CapturedFrame:
        if not self._started:
            raise RuntimeError("相机尚未开始取流")
        assert self._inner is not None
        frame = self._inner.get_frame(timeout_ms or self.config.timeout_ms)
        return CapturedFrame(
            image=frame.image,
            camera_frame_number=int(frame.camera_frame_number),
            camera_timestamp_ticks=frame.camera_timestamp_ticks,
            host_timestamp_ns=time.time_ns(),
            host_monotonic_ns=time.perf_counter_ns(),
            offset_x=self.config.offset_x,
            offset_y=self.config.offset_y,
        )

    def stop(self) -> None:
        if self._started:
            assert self._inner is not None
            self._inner.stop()
            self._started = False

    def close(self) -> None:
        if self._closed:
            return
        if self._started:
            self.stop()
        if self._inner is not None:
            self._inner.close()
        self._closed = True
