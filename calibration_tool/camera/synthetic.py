from __future__ import annotations

import time
from dataclasses import replace

import numpy as np

from .models import CameraConfig, CameraDeviceInfo, CapturedFrame


class SyntheticCameraProvider:
    def __init__(self, *, target_fps: float = 30.0, seed: int = 20260803) -> None:
        if target_fps <= 0:
            raise ValueError("target_fps 必须为正数")
        self.target_fps = float(target_fps)
        self.seed = int(seed)
        self.device = CameraDeviceInfo(
            model="SIMULATED-MV-CS050-60GM",
            serial_number="SIM-001",
            ip_address="127.0.0.1",
            transport="Synthetic",
        )

    def list_devices(self) -> list[CameraDeviceInfo]:
        return [self.device]

    def open(self, serial_number: str, config: CameraConfig) -> "SyntheticCameraSession":
        if serial_number and serial_number != self.device.serial_number:
            raise RuntimeError(f"未找到模拟相机序列号：{serial_number}")
        return SyntheticCameraSession(self.device, config, self.target_fps, self.seed)


class SyntheticCameraSession:
    network_packet_size: int | None = None

    def __init__(
        self,
        device: CameraDeviceInfo,
        config: CameraConfig,
        target_fps: float,
        seed: int,
    ) -> None:
        self.device = device
        self.config = config
        self._period_s = 1.0 / target_fps
        self._seed = seed
        self._started = False
        self._closed = False
        self._frame_number = 0
        self._next_frame_at = 0.0
        self._prepare_geometry()

    def _prepare_geometry(self) -> None:
        self._rows = np.arange(self.config.height, dtype=np.float32)[:, None]
        self._columns = np.arange(self.config.width, dtype=np.float32)[None, :]

    def configure(self, config: CameraConfig) -> CameraConfig:
        if self._closed:
            raise RuntimeError("模拟相机已经关闭")
        if self._started:
            raise RuntimeError("请先停止取流，再修改采集参数")
        self.config = config
        self._prepare_geometry()
        return config

    def update_exposure_gain(self, exposure_us: float, gain_db: float) -> CameraConfig:
        if self._closed:
            raise RuntimeError("模拟相机已经关闭")
        self.config = replace(
            self.config,
            exposure_us=float(exposure_us),
            gain_db=float(gain_db),
        )
        return self.config

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("模拟相机已经关闭")
        self._started = True
        self._next_frame_at = time.monotonic()

    def get_frame(self, timeout_ms: int | None = None) -> CapturedFrame:
        if not self._started:
            raise RuntimeError("模拟相机尚未开始取流")
        delay = self._next_frame_at - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self._next_frame_at = max(self._next_frame_at + self._period_s, time.monotonic())
        self._frame_number += 1
        centre = self.config.height * 0.52 + 4.0 * np.sin(
            self._columns / 170.0 + self._frame_number / 14.0
        )
        stripe = np.exp(-0.5 * ((self._rows - centre) / 1.8) ** 2)
        exposure_scale = self.config.exposure_us / 1200.0
        rng = np.random.default_rng(self._seed + self._frame_number)
        if self.config.pixel_format == "Mono8":
            noise = rng.normal(0.0, 1.2, stripe.shape)
            image = np.clip(12.0 + 220.0 * exposure_scale * stripe + noise, 0, 255).astype(np.uint8)
        else:
            noise = rng.normal(0.0, 12.0, stripe.shape)
            image = np.clip(180.0 + 3500.0 * exposure_scale * stripe + noise, 0, 4095).astype(np.uint16)
        return CapturedFrame(
            image=np.ascontiguousarray(image),
            camera_frame_number=self._frame_number,
            camera_timestamp_ticks=self._frame_number * 1000,
            host_timestamp_ns=time.time_ns(),
            host_monotonic_ns=time.perf_counter_ns(),
            offset_x=self.config.offset_x,
            offset_y=self.config.offset_y,
        )

    def stop(self) -> None:
        self._started = False

    def close(self) -> None:
        self._started = False
        self._closed = True
