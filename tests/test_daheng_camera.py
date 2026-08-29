from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from calibration_tool.camera.config import load_camera_config
from calibration_tool.camera.daheng import DahengCameraProvider, DahengCameraSession
from calibration_tool.camera.factory import build_camera_provider
from calibration_tool.camera.models import CameraConfig


class FakeFeature:
    def __init__(self, value: object) -> None:
        self.value = value
        self.history: list[object] = []

    def is_implemented(self) -> bool:
        return True

    def is_writable(self) -> bool:
        return True

    def set(self, value: object) -> None:
        self.value = value
        self.history.append(value)

    def get(self) -> object:
        return self.value


class FakeRawImage:
    def __init__(
        self,
        image: np.ndarray | None,
        *,
        status: int = 0,
        frame_id: int = 42,
        timestamp: int = 1234,
    ) -> None:
        self.image = image
        self.status = status
        self.frame_id = frame_id
        self.timestamp = timestamp

    def get_status(self) -> int:
        return self.status

    def get_numpy_array(self) -> np.ndarray | None:
        return self.image

    def get_frame_id(self) -> int:
        return self.frame_id

    def get_timestamp(self) -> int:
        return self.timestamp


class FakeStream:
    def __init__(self, raw_image: FakeRawImage | None) -> None:
        self.raw_image = raw_image
        self.requested_timeout: int | None = None

    def get_image(self, timeout: int) -> FakeRawImage | None:
        self.requested_timeout = timeout
        return self.raw_image


class FakeCamera:
    def __init__(self, raw_image: FakeRawImage | None) -> None:
        self.Width = FakeFeature(8)
        self.Height = FakeFeature(4)
        self.OffsetX = FakeFeature(0)
        self.OffsetY = FakeFeature(0)
        self.PixelFormat = FakeFeature(0)
        self.ExposureAuto = FakeFeature(0)
        self.GainAuto = FakeFeature(0)
        self.TriggerMode = FakeFeature(0)
        self.ExposureTime = FakeFeature(1000.0)
        self.Gain = FakeFeature(0.0)
        self.data_stream = [FakeStream(raw_image)]
        self.stream_started = False
        self.stream_stopped = False
        self.closed = False

    def stream_on(self) -> None:
        self.stream_started = True

    def stream_off(self) -> None:
        self.stream_stopped = True

    def close_device(self) -> None:
        self.closed = True


class FakeManager:
    def __init__(self, camera: FakeCamera) -> None:
        self.camera = camera
        self.devices = [
            {
                "sn": "SN-USB3-001",
                "model_name": "ME2P-1230-23U3M",
                "device_class": 3,
            },
            {
                "sn": "SN-GIGE-001",
                "model_name": "OTHER-GIGE",
                "device_class": 2,
            },
        ]
        self.opened_by: str | None = None

    def update_all_device_list(self, timeout: int) -> tuple[int, list[dict[str, object]]]:
        return len(self.devices), self.devices

    def open_device_by_sn(self, serial_number: str) -> FakeCamera:
        self.opened_by = serial_number
        return self.camera


def make_fake_sdk(raw_image: FakeRawImage | None) -> tuple[object, FakeManager]:
    camera = FakeCamera(raw_image)
    manager = FakeManager(camera)
    sdk = SimpleNamespace(
        DeviceManager=lambda: manager,
        GxDeviceClassList=SimpleNamespace(U3V=3),
        GxFrameStatusList=SimpleNamespace(SUCCESS=0, INCOMPLETE=-1),
        GxPixelFormatEntry=SimpleNamespace(MONO8=8, MONO12=12),
        GxAutoEntry=SimpleNamespace(OFF=0),
        GxSwitchEntry=SimpleNamespace(OFF=0),
    )
    return sdk, manager


def make_config(pixel_format: str = "Mono8") -> CameraConfig:
    return CameraConfig(
        pixel_format=pixel_format,
        width=8,
        height=4,
        offset_x=2,
        offset_y=3,
        timeout_ms=1234,
    )


class DahengCameraSessionTests(unittest.TestCase):
    def test_provider_filters_to_usb3(self) -> None:
        sdk, _ = make_fake_sdk(None)
        with patch("calibration_tool.camera.daheng._load_gxipy", return_value=sdk):
            devices = DahengCameraProvider().list_devices()
        self.assertEqual(len(devices), 1)
        self.assertEqual(devices[0].serial_number, "SN-USB3-001")
        self.assertEqual(devices[0].transport, "USB3")

    def test_session_configures_streams_updates_and_copies_frame(self) -> None:
        sdk_image = np.arange(32, dtype=np.uint8).reshape(4, 8)
        sdk, manager = make_fake_sdk(FakeRawImage(sdk_image))
        with patch("calibration_tool.camera.daheng._load_gxipy", return_value=sdk):
            session = DahengCameraSession.open("SN-USB3-001", make_config())
            session.start()
            updated = session.update_exposure_gain(1500.0, 2.0)
            frame = session.get_frame()
            session.close()

        self.assertEqual(manager.opened_by, "SN-USB3-001")
        self.assertEqual((updated.exposure_us, updated.gain_db), (1500.0, 2.0))
        self.assertTrue(manager.camera.stream_started)
        self.assertTrue(manager.camera.stream_stopped)
        self.assertTrue(manager.camera.closed)
        self.assertEqual(manager.camera.data_stream[0].requested_timeout, 1234)
        self.assertEqual(frame.camera_frame_number, 42)
        self.assertEqual(frame.camera_timestamp_ticks, 1234)
        self.assertEqual((frame.offset_x, frame.offset_y), (2, 3))
        self.assertTrue(frame.image.flags.owndata)
        sdk_image.fill(0)
        self.assertNotEqual(int(frame.image.sum()), 0)
        session.close()

    def test_configure_is_rejected_while_streaming(self) -> None:
        sdk, _ = make_fake_sdk(FakeRawImage(np.ones((4, 8), dtype=np.uint8)))
        with patch("calibration_tool.camera.daheng._load_gxipy", return_value=sdk):
            session = DahengCameraSession.open("SN-USB3-001", make_config())
            session.start()
            with self.assertRaisesRegex(RuntimeError, "停止取流"):
                session.configure(make_config())
            session.close()

    def test_timeout_and_incomplete_frame_are_reported(self) -> None:
        for raw_image, expected in (
            (None, TimeoutError),
            (FakeRawImage(np.ones((4, 8), dtype=np.uint8), status=-1), RuntimeError),
        ):
            sdk, manager = make_fake_sdk(raw_image)
            with patch("calibration_tool.camera.daheng._load_gxipy", return_value=sdk):
                session = DahengCameraSession.open("SN-USB3-001", make_config())
                session.start()
                with self.assertRaises(expected):
                    session.get_frame()
                session.close()
            self.assertTrue(manager.camera.closed)

    def test_mono12_is_normalized_and_packed_data_is_rejected(self) -> None:
        for value, should_fail in ((4095, False), (4095 << 4, False), (0xF00F, True)):
            sdk, _ = make_fake_sdk(
                FakeRawImage(np.full((4, 8), value, dtype=np.uint16))
            )
            with patch("calibration_tool.camera.daheng._load_gxipy", return_value=sdk):
                session = DahengCameraSession.open("SN-USB3-001", make_config("Mono12"))
                session.start()
                if should_fail:
                    with self.assertRaisesRegex(RuntimeError, "Mono12 数据格式无法识别"):
                        session.get_frame()
                else:
                    self.assertEqual(int(session.get_frame().image.max()), 4095)
                session.close()

    def test_config_and_factory_accept_daheng(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "camera.yaml"
            path.write_text(
                "backend: daheng\nbackend_options:\n  discovery_timeout_ms: 250\n",
                encoding="utf-8",
            )
            runtime = load_camera_config(path)
        provider = build_camera_provider(
            runtime["backend"],
            calibration_src=runtime["calibration_src"],
            backend_options=runtime["backend_options"],
        )
        self.assertIsInstance(provider, DahengCameraProvider)
        self.assertEqual(provider.discovery_timeout_ms, 250)


if __name__ == "__main__":
    unittest.main()
