"""Capture line-laser datasets with a HIKROBOT MV-CS050-60GM GigE camera."""

from __future__ import annotations

import argparse
import csv
import ctypes
import importlib
import math
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Callable, Protocol, Sequence

import cv2
import numpy as np


EXPECTED_MODEL = "MV-CS050-60GM"
DEFAULT_DATASET = Path(__file__).resolve().parents[1]
PIXEL_FORMATS = {"mono8": "Mono8", "mono12": "Mono12"}
IMAGE_FORMATS = {"png": ".png", "tiff": ".tiff"}
TIMESTAMP_FIELDS = (
    "filename",
    "host_time_s",
    "host_time_ns",
    "host_monotonic_s",
    "camera_frame_number",
    "camera_timestamp_ticks",
)
METADATA_FIELDS = (
    "exp_id",
    "image_dir",
    "exposure_us",
    "gain",
    "material",
    "distance_mm",
    "baseline_mm",
    "laser_angle_deg",
    "frame_count",
    "pixel_format",
    "image_format",
    "roi_offset_x",
    "roi_offset_y",
    "roi_width",
    "roi_height",
    "camera_model",
    "camera_serial",
    "camera_ip",
    "gev_packet_size",
    "remark",
)

_DLL_DIRECTORY_HANDLES: list[object] = []


@dataclass(frozen=True)
class CameraSettings:
    model: str
    serial_number: str
    ip_address: str
    exposure_us: float
    gain: float
    pixel_format: str
    offset_x: int
    offset_y: int
    width: int
    height: int
    packet_size: int


@dataclass(frozen=True)
class CapturedFrame:
    image: np.ndarray
    camera_frame_number: int
    camera_timestamp_ticks: int | None


@dataclass(frozen=True)
class CaptureResult:
    saved_frames: int
    total_elapsed_s: float
    actual_fps: float
    arrival_fps: float
    output_dir: Path


@dataclass(frozen=True)
class DeviceRecord:
    model: str
    serial_number: str
    ip_address: str
    device_info: object


class CameraSession(Protocol):
    settings: CameraSettings

    def start(self) -> None: ...

    def get_frame(self, timeout_ms: int) -> CapturedFrame: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite value greater than zero")
    return parsed


def finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("must be finite")
    return parsed


def pixel_format_arg(value: str) -> str:
    try:
        return PIXEL_FORMATS[value.casefold()]
    except KeyError as exc:
        raise argparse.ArgumentTypeError("pixel format must be Mono8 or Mono12") from exc


def image_format_arg(value: str) -> str:
    normalized = value.casefold()
    if normalized not in IMAGE_FORMATS:
        raise argparse.ArgumentTypeError("image format must be png or tiff")
    return normalized


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Capture a repeatable monochrome line-laser dataset with a HIKROBOT "
            f"{EXPECTED_MODEL} GigE camera."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Dataset root (default: {DEFAULT_DATASET}).",
    )
    parser.add_argument("--exp-id", required=True, help="Unique experiment identifier.")
    parser.add_argument("--frames", type=positive_int, default=50)
    parser.add_argument("--exposure-us", type=positive_float, required=True)
    parser.add_argument("--gain", type=finite_float, default=0.0)
    parser.add_argument("--material", required=True)
    parser.add_argument("--distance-mm", type=positive_float, required=True)
    parser.add_argument("--baseline-mm", type=positive_float, required=True)
    parser.add_argument("--laser-angle-deg", type=finite_float, required=True)
    parser.add_argument("--pixel-format", type=pixel_format_arg, default="Mono12")
    parser.add_argument("--image-format", type=image_format_arg, default="png")
    parser.add_argument("--offset-x", type=nonnegative_int, default=0)
    parser.add_argument("--offset-y", type=nonnegative_int, default=0)
    parser.add_argument("--width", type=positive_int)
    parser.add_argument("--height", type=positive_int)
    parser.add_argument("--warmup-frames", type=nonnegative_int, default=10)
    parser.add_argument("--timeout-ms", type=positive_int, default=2000)
    parser.add_argument(
        "--serial-number",
        help="Required only when more than one matching MV-CS050-60GM is connected.",
    )
    parser.add_argument("--remark", default="")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not args.exp_id or args.exp_id != args.exp_id.strip():
        raise ValueError("exp_id must be non-empty and have no surrounding whitespace")
    if (
        args.exp_id in {".", ".."}
        or Path(args.exp_id).name != args.exp_id
        or "/" in args.exp_id
        or "\\" in args.exp_id
    ):
        raise ValueError("exp_id must be a single safe directory name")
    if not args.material or args.material != args.material.strip():
        raise ValueError("material must be non-empty and have no surrounding whitespace")
    if args.serial_number is not None:
        if not args.serial_number or args.serial_number != args.serial_number.strip():
            raise ValueError(
                "serial-number must be non-empty and have no surrounding whitespace"
            )

    width_given = args.width is not None
    height_given = args.height is not None
    if width_given != height_given:
        raise ValueError("width and height must be provided together")
    if not width_given and (args.offset_x != 0 or args.offset_y != 0):
        raise ValueError("offset-x and offset-y must be zero in full-frame mode")


def _mvs_python_candidates() -> list[Path]:
    candidates: list[Path] = []
    configured = os.environ.get("MVS_PYTHON_PATH")
    if configured:
        candidates.append(Path(configured))

    for env_name in ("PROGRAMFILES(X86)", "PROGRAMFILES"):
        root = os.environ.get(env_name)
        if root:
            candidates.append(
                Path(root) / "MVS" / "Development" / "Samples" / "Python" / "MvImport"
            )
    return candidates


def _configure_mvs_dll_search_path() -> None:
    runtime_name = "Win64_x64" if ctypes.sizeof(ctypes.c_void_p) == 8 else "Win32_i86"
    roots = [
        os.environ.get("PROGRAMFILES(X86)"),
        os.environ.get("PROGRAMFILES"),
    ]
    for root in roots:
        if not root:
            continue
        runtime = Path(root) / "Common Files" / "MVS" / "Runtime" / runtime_name
        if not runtime.is_dir():
            continue
        if hasattr(os, "add_dll_directory"):
            _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(runtime)))
        elif str(runtime) not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = f"{runtime}{os.pathsep}{os.environ.get('PATH', '')}"


def load_mvs_sdk() -> ModuleType:
    """Load the official MVS Python wrapper without requiring it for --help/tests."""
    _configure_mvs_dll_search_path()
    errors: list[str] = []
    search_dirs: list[Path | None] = [None, *_mvs_python_candidates()]
    for search_dir in search_dirs:
        if search_dir is not None:
            if not search_dir.is_dir():
                continue
            if str(search_dir) not in sys.path:
                sys.path.insert(0, str(search_dir))
        try:
            sdk = importlib.import_module("MvCameraControl_class")
        except Exception as exc:
            location = "sys.path" if search_dir is None else str(search_dir)
            errors.append(f"{location}: {exc}")
            continue

        required = (
            "MvCamera",
            "MV_CC_DEVICE_INFO_LIST",
            "MV_CC_DEVICE_INFO",
            "MV_FRAME_OUT",
            "MV_GIGE_DEVICE",
            "MV_ACCESS_Exclusive",
        )
        missing = [name for name in required if not hasattr(sdk, name)]
        if missing:
            errors.append(f"{search_dir or 'sys.path'}: missing {', '.join(missing)}")
            continue
        return sdk

    detail = "; ".join(errors) if errors else "no MVS Python directory was found"
    raise RuntimeError(
        "Cannot load the HIKROBOT MVS Python SDK. Install MVS with Development/Samples, "
        "or set MVS_PYTHON_PATH to its MvImport directory. "
        f"Details: {detail}"
    )


def _ret_hex(ret: int) -> str:
    return f"0x{int(ret) & 0xFFFFFFFF:08X}"


def _check_ret(operation: str, ret: int) -> None:
    if int(ret) != 0:
        raise RuntimeError(f"{operation} failed with MVS status {_ret_hex(ret)}")


def _decode_c_string(value: object) -> str:
    raw = bytes(value).split(b"\0", 1)[0]
    for encoding in ("utf-8", "gb18030", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


def _ipv4_from_mvs(value: int) -> str:
    return ".".join(str((int(value) >> shift) & 0xFF) for shift in (24, 16, 8, 0))


def enumerate_gige_devices(sdk: ModuleType) -> list[DeviceRecord]:
    device_list = sdk.MV_CC_DEVICE_INFO_LIST()
    _check_ret(
        "MV_CC_EnumDevices",
        sdk.MvCamera.MV_CC_EnumDevices(sdk.MV_GIGE_DEVICE, device_list),
    )
    records: list[DeviceRecord] = []
    for index in range(int(device_list.nDeviceNum)):
        pointer = device_list.pDeviceInfo[index]
        if not pointer:
            continue
        info = ctypes.cast(pointer, ctypes.POINTER(sdk.MV_CC_DEVICE_INFO)).contents
        if int(info.nTLayerType) != int(sdk.MV_GIGE_DEVICE):
            continue
        gige = info.SpecialInfo.stGigEInfo
        records.append(
            DeviceRecord(
                model=_decode_c_string(gige.chModelName),
                serial_number=_decode_c_string(gige.chSerialNumber),
                ip_address=_ipv4_from_mvs(gige.nCurrentIp),
                device_info=info,
            )
        )
    return records


def select_device(
    devices: Sequence[DeviceRecord], serial_number: str | None
) -> DeviceRecord:
    matching = [device for device in devices if device.model == EXPECTED_MODEL]
    if serial_number is not None:
        matching = [
            device for device in matching if device.serial_number == serial_number
        ]
    if len(matching) == 1:
        return matching[0]

    discovered = ", ".join(
        f"{device.model or 'unknown'} SN={device.serial_number or 'unknown'} "
        f"IP={device.ip_address}"
        for device in devices
    ) or "none"
    if not matching:
        suffix = f" with serial {serial_number!r}" if serial_number else ""
        raise RuntimeError(
            f"No {EXPECTED_MODEL}{suffix} was found. Discovered GigE devices: {discovered}"
        )
    raise RuntimeError(
        f"Multiple {EXPECTED_MODEL} cameras were found; pass --serial-number. "
        f"Discovered GigE devices: {discovered}"
    )


def _new_sdk_struct(sdk: ModuleType, *names: str) -> object:
    for name in names:
        struct_type = getattr(sdk, name, None)
        if struct_type is not None:
            return struct_type()
    raise RuntimeError(f"MVS Python SDK is missing structure: {' or '.join(names)}")


def _get_integer(cam: object, sdk: ModuleType, name: str) -> object:
    value = _new_sdk_struct(sdk, "MVCC_INTVALUE_EX", "MVCC_INTVALUE")
    getter = getattr(cam, "MV_CC_GetIntValueEx", None)
    if getter is None:
        getter = getattr(cam, "MV_CC_GetIntValue", None)
    if getter is None:
        raise RuntimeError("MVS Python SDK has no integer feature getter")
    _check_ret(f"get {name}", getter(name, value))
    return value


def _set_integer(cam: object, sdk: ModuleType, name: str, requested: int) -> int:
    value = _get_integer(cam, sdk, name)
    minimum = int(value.nMin)
    maximum = int(value.nMax)
    increment = max(1, int(value.nInc))
    if requested < minimum or requested > maximum:
        raise ValueError(
            f"{name}={requested} is outside the camera range [{minimum}, {maximum}]"
        )
    if (requested - minimum) % increment != 0:
        raise ValueError(
            f"{name}={requested} is not aligned to increment {increment} "
            f"from minimum {minimum}"
        )
    setter = getattr(cam, "MV_CC_SetIntValueEx", None)
    if setter is None:
        setter = getattr(cam, "MV_CC_SetIntValue", None)
    if setter is None:
        raise RuntimeError("MVS Python SDK has no integer feature setter")
    _check_ret(f"set {name}", setter(name, int(requested)))
    return int(_get_integer(cam, sdk, name).nCurValue)


def _integer_max(cam: object, sdk: ModuleType, name: str) -> int:
    return int(_get_integer(cam, sdk, name).nMax)


def _set_float(cam: object, sdk: ModuleType, name: str, requested: float) -> float:
    value = _new_sdk_struct(sdk, "MVCC_FLOATVALUE")
    _check_ret(f"get {name}", cam.MV_CC_GetFloatValue(name, value))
    minimum = float(value.fMin)
    maximum = float(value.fMax)
    if requested < minimum or requested > maximum:
        raise ValueError(
            f"{name}={requested:g} is outside the camera range "
            f"[{minimum:g}, {maximum:g}]"
        )
    _check_ret(f"set {name}", cam.MV_CC_SetFloatValue(name, float(requested)))
    _check_ret(f"read back {name}", cam.MV_CC_GetFloatValue(name, value))
    return float(value.fCurValue)


def _set_enum(cam: object, name: str, value: str) -> None:
    setter = getattr(cam, "MV_CC_SetEnumValueByString", None)
    if setter is None:
        raise RuntimeError("MVS Python SDK has no symbolic enum setter")
    _check_ret(f"set {name}={value}", setter(name, value))


def _camera_timestamp_ticks(frame_info: object) -> int | None:
    if hasattr(frame_info, "nDevTimeStampHigh") and hasattr(
        frame_info, "nDevTimeStampLow"
    ):
        return (int(frame_info.nDevTimeStampHigh) << 32) | int(
            frame_info.nDevTimeStampLow
        )
    if hasattr(frame_info, "nDevTimeStamp"):
        return int(frame_info.nDevTimeStamp)
    return None


class MvsCameraSession:
    def __init__(
        self,
        sdk: ModuleType,
        cam: object,
        settings: CameraSettings,
    ) -> None:
        self.sdk = sdk
        self.cam = cam
        self.settings = settings
        self._started = False
        self._closed = False

    @classmethod
    def open(cls, args: argparse.Namespace) -> "MvsCameraSession":
        sdk = load_mvs_sdk()
        selected = select_device(enumerate_gige_devices(sdk), args.serial_number)
        cam = sdk.MvCamera()
        handle_created = False
        device_open = False
        try:
            _check_ret("MV_CC_CreateHandle", cam.MV_CC_CreateHandle(selected.device_info))
            handle_created = True
            _check_ret(
                "MV_CC_OpenDevice",
                cam.MV_CC_OpenDevice(sdk.MV_ACCESS_Exclusive, 0),
            )
            device_open = True

            packet_size = int(cam.MV_CC_GetOptimalPacketSize())
            if packet_size > 0:
                _check_ret(
                    "set GevSCPSPacketSize",
                    cam.MV_CC_SetIntValue("GevSCPSPacketSize", packet_size),
                )
            else:
                packet_size = 0

            _set_enum(cam, "AcquisitionMode", "Continuous")
            _set_enum(cam, "TriggerMode", "Off")
            _set_enum(cam, "ExposureAuto", "Off")
            _set_enum(cam, "GainAuto", "Off")
            _set_enum(cam, "PixelFormat", args.pixel_format)
            exposure_us = _set_float(cam, sdk, "ExposureTime", args.exposure_us)
            gain = _set_float(cam, sdk, "Gain", args.gain)

            _set_integer(cam, sdk, "OffsetX", 0)
            _set_integer(cam, sdk, "OffsetY", 0)
            if args.width is None:
                width = _set_integer(cam, sdk, "Width", _integer_max(cam, sdk, "Width"))
                height = _set_integer(
                    cam, sdk, "Height", _integer_max(cam, sdk, "Height")
                )
                offset_x = 0
                offset_y = 0
            else:
                width = _set_integer(cam, sdk, "Width", args.width)
                height = _set_integer(cam, sdk, "Height", args.height)
                offset_x = _set_integer(cam, sdk, "OffsetX", args.offset_x)
                offset_y = _set_integer(cam, sdk, "OffsetY", args.offset_y)

            settings = CameraSettings(
                model=selected.model,
                serial_number=selected.serial_number,
                ip_address=selected.ip_address,
                exposure_us=exposure_us,
                gain=gain,
                pixel_format=args.pixel_format,
                offset_x=offset_x,
                offset_y=offset_y,
                width=width,
                height=height,
                packet_size=packet_size,
            )
            return cls(sdk=sdk, cam=cam, settings=settings)
        except Exception:
            if device_open:
                cam.MV_CC_CloseDevice()
            if handle_created:
                cam.MV_CC_DestroyHandle()
            raise

    def start(self) -> None:
        _check_ret("MV_CC_StartGrabbing", self.cam.MV_CC_StartGrabbing())
        self._started = True

    def get_frame(self, timeout_ms: int) -> CapturedFrame:
        frame_out = self.sdk.MV_FRAME_OUT()
        ctypes.memset(ctypes.byref(frame_out), 0, ctypes.sizeof(frame_out))
        ret = self.cam.MV_CC_GetImageBuffer(frame_out, timeout_ms)
        _check_ret("MV_CC_GetImageBuffer", ret)
        try:
            info = frame_out.stFrameInfo
            width = int(info.nWidth)
            height = int(info.nHeight)
            if (width, height) != (self.settings.width, self.settings.height):
                raise RuntimeError(
                    "Unexpected frame size: "
                    f"got {width}x{height}, expected "
                    f"{self.settings.width}x{self.settings.height}"
                )
            dtype = np.dtype(np.uint8 if self.settings.pixel_format == "Mono8" else "<u2")
            expected_bytes = width * height * dtype.itemsize
            frame_length = int(info.nFrameLen)
            if frame_length < expected_bytes:
                raise RuntimeError(
                    f"Incomplete frame payload: got {frame_length} bytes, "
                    f"expected at least {expected_bytes}"
                )
            payload = ctypes.string_at(frame_out.pBufAddr, expected_bytes)
            image = np.frombuffer(payload, dtype=dtype).reshape(height, width).copy()
            if self.settings.pixel_format == "Mono12" and int(image.max()) > 4095:
                raise RuntimeError(
                    "Mono12 frame contains DN above 4095; check that PixelFormat is "
                    "unpacked Mono12 rather than Mono12Packed"
                )
            return CapturedFrame(
                image=image,
                camera_frame_number=int(info.nFrameNum),
                camera_timestamp_ticks=_camera_timestamp_ticks(info),
            )
        finally:
            free_ret = self.cam.MV_CC_FreeImageBuffer(frame_out)
            if int(free_ret) != 0:
                print(
                    "WARNING: MV_CC_FreeImageBuffer failed with "
                    f"{_ret_hex(free_ret)}",
                    file=sys.stderr,
                )

    def stop(self) -> None:
        if self._started:
            _check_ret("MV_CC_StopGrabbing", self.cam.MV_CC_StopGrabbing())
            self._started = False

    def close(self) -> None:
        if self._closed:
            return
        if self._started:
            self.stop()
        close_ret = self.cam.MV_CC_CloseDevice()
        destroy_ret = self.cam.MV_CC_DestroyHandle()
        self._closed = True
        _check_ret("MV_CC_CloseDevice", close_ret)
        _check_ret("MV_CC_DestroyHandle", destroy_ret)


def ensure_metadata(metadata_path: Path) -> set[str]:
    if not metadata_path.exists():
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with metadata_path.open("x", encoding="utf-8-sig", newline="") as handle:
            csv.DictWriter(handle, fieldnames=METADATA_FIELDS).writeheader()

    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != METADATA_FIELDS:
            raise RuntimeError(
                f"metadata.csv header must exactly match: {','.join(METADATA_FIELDS)}"
            )
        exp_ids: set[str] = set()
        for line_number, row in enumerate(reader, start=2):
            exp_id = (row.get("exp_id") or "").strip()
            if not exp_id and not any((value or "").strip() for value in row.values()):
                continue
            if not exp_id:
                raise RuntimeError(f"metadata.csv line {line_number}: exp_id is empty")
            if exp_id in exp_ids:
                raise RuntimeError(
                    f"metadata.csv line {line_number}: duplicate exp_id {exp_id!r}"
                )
            exp_ids.add(exp_id)
        return exp_ids


def append_metadata_atomic(metadata_path: Path, row: dict[str, object]) -> None:
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != METADATA_FIELDS:
            raise RuntimeError("metadata.csv header changed during capture")
        rows = list(reader)

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8-sig",
            newline="",
            dir=metadata_path.parent,
            prefix=".metadata.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=METADATA_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
            writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, metadata_path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def write_image(path: Path, image: np.ndarray) -> None:
    if image.ndim != 2 or image.dtype not in (np.dtype(np.uint8), np.dtype(np.uint16)):
        raise RuntimeError(
            f"Expected a 2-D uint8 or uint16 image, got shape={image.shape}, "
            f"dtype={image.dtype}"
        )
    try:
        saved = cv2.imwrite(str(path), np.ascontiguousarray(image))
    except cv2.error as exc:
        raise OSError(f"Failed to save image: {path}") from exc
    if not saved:
        raise OSError(f"Failed to save image: {path}")


def write_timestamps(path: Path, rows: Sequence[dict[str, str]]) -> None:
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TIMESTAMP_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _seconds_from_ns(value: int) -> str:
    seconds, nanoseconds = divmod(value, 1_000_000_000)
    return f"{seconds}.{nanoseconds:09d}"


def _metadata_row(
    args: argparse.Namespace, settings: CameraSettings
) -> dict[str, object]:
    return {
        "exp_id": args.exp_id,
        "image_dir": f"raw/{args.exp_id}",
        "exposure_us": f"{settings.exposure_us:.12g}",
        "gain": f"{settings.gain:.12g}",
        "material": args.material,
        "distance_mm": f"{args.distance_mm:.12g}",
        "baseline_mm": f"{args.baseline_mm:.12g}",
        "laser_angle_deg": f"{args.laser_angle_deg:.12g}",
        "frame_count": args.frames,
        "pixel_format": settings.pixel_format,
        "image_format": args.image_format,
        "roi_offset_x": settings.offset_x,
        "roi_offset_y": settings.offset_y,
        "roi_width": settings.width,
        "roi_height": settings.height,
        "camera_model": settings.model,
        "camera_serial": settings.serial_number,
        "camera_ip": settings.ip_address,
        "gev_packet_size": settings.packet_size,
        "remark": args.remark,
    }


def print_camera_parameters(settings: CameraSettings, args: argparse.Namespace) -> None:
    print("Current camera parameters:")
    print(f"  model: {settings.model}")
    print(f"  serial_number: {settings.serial_number}")
    print(f"  ip_address: {settings.ip_address}")
    print(f"  exposure_us: {settings.exposure_us:g}")
    print(f"  gain: {settings.gain:g}")
    print(f"  pixel_format: {settings.pixel_format}")
    print(f"  image_format: {args.image_format}")
    print(f"  gev_packet_size: {settings.packet_size or 'SDK default'}")
    print(
        "  roi: "
        f"offset=({settings.offset_x}, {settings.offset_y}), "
        f"size=({settings.width}, {settings.height})"
    )
    print(f"  warmup_frames: {args.warmup_frames}")
    print(f"  frames_to_save: {args.frames}")
    print(f"  frame_timeout_ms: {args.timeout_ms}")


def _cleanup_directory(path: Path | None) -> None:
    if path is None or not path.exists():
        return
    try:
        shutil.rmtree(path)
    except OSError as exc:
        print(f"WARNING: could not clean temporary output {path}: {exc}", file=sys.stderr)


def acquire_dataset(
    args: argparse.Namespace,
    session_factory: Callable[[argparse.Namespace], CameraSession] = MvsCameraSession.open,
    image_writer: Callable[[Path, np.ndarray], None] = write_image,
) -> CaptureResult:
    validate_args(args)
    dataset = args.dataset.expanduser().resolve()
    raw_root = dataset / "raw"
    metadata_path = dataset / "metadata.csv"
    raw_root.mkdir(parents=True, exist_ok=True)

    existing_ids = ensure_metadata(metadata_path)
    final_dir = raw_root / args.exp_id
    if args.exp_id in existing_ids:
        raise RuntimeError(f"exp_id already exists in metadata.csv: {args.exp_id}")
    if final_dir.exists():
        raise RuntimeError(f"experiment output directory already exists: {final_dir}")

    temp_dir = Path(tempfile.mkdtemp(prefix=f".{args.exp_id}.", dir=raw_root))
    session: CameraSession | None = None
    stream_started = False
    published = False
    try:
        session = session_factory(args)
        settings = session.settings
        print_camera_parameters(settings, args)
        session.start()
        stream_started = True

        for warmup_index in range(args.warmup_frames):
            try:
                session.get_frame(args.timeout_ms)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"Failed to acquire warmup frame {warmup_index + 1}: {exc}"
                ) from exc

        timestamp_rows: list[dict[str, str]] = []
        received_monotonic_ns: list[int] = []
        image_suffix = IMAGE_FORMATS[args.image_format]
        capture_start_ns = time.perf_counter_ns()
        for frame_number in range(1, args.frames + 1):
            try:
                frame = session.get_frame(args.timeout_ms)
            except RuntimeError as exc:
                raise RuntimeError(f"Failed to acquire frame {frame_number}: {exc}") from exc

            host_time_ns = time.time_ns()
            host_monotonic_ns = time.perf_counter_ns()
            filename = f"frame_{frame_number:04d}{image_suffix}"
            image_writer(temp_dir / filename, frame.image)
            received_monotonic_ns.append(host_monotonic_ns)
            timestamp_rows.append(
                {
                    "filename": filename,
                    "host_time_s": _seconds_from_ns(host_time_ns),
                    "host_time_ns": str(host_time_ns),
                    "host_monotonic_s": _seconds_from_ns(host_monotonic_ns),
                    "camera_frame_number": str(frame.camera_frame_number),
                    "camera_timestamp_ticks": (
                        ""
                        if frame.camera_timestamp_ticks is None
                        else str(frame.camera_timestamp_ticks)
                    ),
                }
            )

        total_elapsed_s = (time.perf_counter_ns() - capture_start_ns) / 1e9
        saved_frames = sum(
            path.is_file() for path in temp_dir.glob(f"frame_*{image_suffix}")
        )
        if saved_frames != args.frames:
            raise RuntimeError(
                f"Saved frame count mismatch: expected {args.frames}, got {saved_frames}"
            )
        write_timestamps(temp_dir / "timestamps.csv", timestamp_rows)

        actual_fps = saved_frames / total_elapsed_s if total_elapsed_s > 0 else 0.0
        if len(received_monotonic_ns) > 1:
            arrival_elapsed_s = (
                received_monotonic_ns[-1] - received_monotonic_ns[0]
            ) / 1e9
            arrival_fps = (
                (len(received_monotonic_ns) - 1) / arrival_elapsed_s
                if arrival_elapsed_s > 0
                else 0.0
            )
        else:
            arrival_fps = 0.0

        session.stop()
        stream_started = False
        temp_dir.rename(final_dir)
        published = True
        append_metadata_atomic(metadata_path, _metadata_row(args, settings))
        return CaptureResult(
            saved_frames=saved_frames,
            total_elapsed_s=total_elapsed_s,
            actual_fps=actual_fps,
            arrival_fps=arrival_fps,
            output_dir=final_dir,
        )
    except Exception:
        _cleanup_directory(final_dir if published else temp_dir)
        raise
    finally:
        if stream_started and session is not None:
            try:
                session.stop()
            except Exception as exc:
                print(f"WARNING: failed to stop camera stream: {exc}", file=sys.stderr)
        if session is not None:
            try:
                session.close()
            except Exception as exc:
                print(f"WARNING: failed to close camera: {exc}", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = acquire_dataset(args)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Saved frames: {result.saved_frames}")
    print(f"Total elapsed: {result.total_elapsed_s:.6f} s")
    print(f"Actual FPS (saved pipeline): {result.actual_fps:.3f}")
    if result.saved_frames > 1:
        print(f"Arrival FPS (host receive intervals): {result.arrival_fps:.3f}")
    else:
        print("Arrival FPS (host receive intervals): N/A (requires at least 2 frames)")
    print(f"Output directory: {result.output_dir}")
    return 0



# ---------------------------------------------------------------------------
# Chessboard exposure-series capture mode
# ---------------------------------------------------------------------------

SERIES_LOG_FIELDS = (
    "pose_id",
    "capture_index",
    "requested_exposure_us",
    "actual_exposure_us",
    "relative_path",
    "host_time_ns",
    "camera_frame_number",
    "camera_timestamp_ticks",
    "gain",
    "pixel_format",
    "camera_model",
    "camera_serial",
)


def safe_name(value: str, field_name: str) -> str:
    """Validate a name that will be used as one filename component."""
    if not value or value != value.strip():
        raise argparse.ArgumentTypeError(
            f"{field_name} must be non-empty and have no surrounding whitespace"
        )
    if value in {".", ".."} or Path(value).name != value or "/" in value or "\\" in value:
        raise argparse.ArgumentTypeError(
            f"{field_name} must be a single safe filename component"
        )
    return value


def pose_id_arg(value: str) -> str:
    return safe_name(value, "pose-id")


def build_series_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Capture one fixed chessboard pose at several exposure times. "
            "Output layout: <output>/<exposure>us/<pose-id>.tif"
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output root, for example D:\\Docs\\...\\chessboard_exposure_test",
    )
    parser.add_argument(
        "--pose-id",
        type=pose_id_arg,
        required=True,
        help="Pose filename without extension, for example pose_01",
    )
    parser.add_argument(
        "--exposures-us",
        type=positive_float,
        nargs="+",
        required=True,
        help="Exposure sequence in microseconds, for example 500 1000 1500 2000",
    )
    parser.add_argument("--gain", type=finite_float, default=0.0)
    parser.add_argument("--pixel-format", type=pixel_format_arg, default="Mono12")
    parser.add_argument(
        "--image-format",
        choices=("tif", "png"),
        default="tif",
        help="Saved image format; tif is recommended for Mono12",
    )
    parser.add_argument("--offset-x", type=nonnegative_int, default=0)
    parser.add_argument("--offset-y", type=nonnegative_int, default=0)
    parser.add_argument("--width", type=positive_int)
    parser.add_argument("--height", type=positive_int)
    parser.add_argument(
        "--settle-frames",
        type=nonnegative_int,
        default=5,
        help="Frames discarded after every exposure change, default 5",
    )
    parser.add_argument(
        "--frames-per-exposure",
        type=positive_int,
        default=1,
        help="How many images to save for each exposure value, default 1",
    )
    parser.add_argument("--timeout-ms", type=positive_int, default=2000)
    parser.add_argument(
        "--serial-number",
        help="Required only when multiple matching cameras are connected",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of an existing image for the same exposure and pose",
    )
    return parser


def format_exposure_folder(exposure_us: float) -> str:
    """500.0 -> 500us; 500.5 -> 500.5us."""
    if float(exposure_us).is_integer():
        value = str(int(exposure_us))
    else:
        value = f"{exposure_us:.12g}".rstrip("0").rstrip(".")
    return f"{value}us"


def update_session_exposure(
    session: MvsCameraSession, requested_exposure_us: float
) -> float:
    """Set exposure and update the session's read-back settings."""
    actual = _set_float(
        session.cam,
        session.sdk,
        "ExposureTime",
        requested_exposure_us,
    )
    old = session.settings
    session.settings = CameraSettings(
        model=old.model,
        serial_number=old.serial_number,
        ip_address=old.ip_address,
        exposure_us=actual,
        gain=old.gain,
        pixel_format=old.pixel_format,
        offset_x=old.offset_x,
        offset_y=old.offset_y,
        width=old.width,
        height=old.height,
        packet_size=old.packet_size,
    )
    return actual


def append_series_log(log_path: Path, row: dict[str, object]) -> None:
    """Append one capture record; create the CSV header on first use."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    exists = log_path.exists()
    with log_path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SERIES_LOG_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def validate_series_args(args: argparse.Namespace) -> None:
    width_given = args.width is not None
    height_given = args.height is not None
    if width_given != height_given:
        raise ValueError("width and height must be provided together")
    if not width_given and (args.offset_x != 0 or args.offset_y != 0):
        raise ValueError("offset-x and offset-y must be zero in full-frame mode")

    if len(set(args.exposures_us)) != len(args.exposures_us):
        raise ValueError("exposures-us contains duplicate values")

    if args.pixel_format == "Mono12" and args.image_format != "tif":
        raise ValueError("Mono12 should be saved as tif to preserve 12-bit data")

    if args.frames_per_exposure < 1:
        raise ValueError("frames-per-exposure must be at least 1")


def capture_exposure_series(args: argparse.Namespace) -> int:
    validate_series_args(args)
    output_root = args.output.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    extension = ".tif" if args.image_format == "tif" else ".png"
    targets: list[tuple[float, list[Path]]] = []
    for exposure in args.exposures_us:
        folder = format_exposure_folder(exposure)
        exposure_targets: list[Path] = []
        for capture_index in range(1, args.frames_per_exposure + 1):
            if capture_index == 1:
                filename = f"{args.pose_id}{extension}"
            else:
                filename = f"{args.pose_id}_{capture_index:02d}{extension}"
            target = output_root / folder / filename
            if target.exists() and not args.overwrite:
                raise RuntimeError(
                    f"Output already exists: {target}. "
                    "Use another pose-id or pass --overwrite."
                )
            exposure_targets.append(target)
        targets.append((exposure, exposure_targets))

    # MvsCameraSession.open expects the initial exposure in args.exposure_us.
    args.exposure_us = float(args.exposures_us[0])
    session: MvsCameraSession | None = None
    started = False

    try:
        session = MvsCameraSession.open(args)
        print("Camera opened:")
        print(f"  model: {session.settings.model}")
        print(f"  serial: {session.settings.serial_number}")
        print(f"  pixel_format: {session.settings.pixel_format}")
        print(f"  gain: {session.settings.gain:g}")
        print(f"  image_size: {session.settings.width} x {session.settings.height}")
        print(f"  pose_id: {args.pose_id}")
        print("Keep the chessboard completely fixed during this command.\n")

        session.start()
        started = True

        for index, (requested_exposure, exposure_targets) in enumerate(targets, start=1):
            actual_exposure = update_session_exposure(session, requested_exposure)

            # Discard several frames so the newly set exposure has taken effect.
            for settle_index in range(args.settle_frames):
                try:
                    session.get_frame(args.timeout_ms)
                except RuntimeError as exc:
                    raise RuntimeError(
                        f"Failed to acquire settle frame {settle_index + 1} "
                        f"at {requested_exposure:g} us: {exc}"
                    ) from exc

            for capture_index, target in enumerate(exposure_targets, start=1):
                frame = session.get_frame(args.timeout_ms)
                target.parent.mkdir(parents=True, exist_ok=True)
                write_image(target, frame.image)

                host_time_ns = time.time_ns()
                relative_path = target.relative_to(output_root).as_posix()
                append_series_log(
                    output_root / "capture_log.csv",
                    {
                        "pose_id": args.pose_id,
                        "capture_index": str(capture_index),
                        "requested_exposure_us": f"{requested_exposure:.12g}",
                        "actual_exposure_us": f"{actual_exposure:.12g}",
                        "relative_path": relative_path,
                        "host_time_ns": str(host_time_ns),
                        "camera_frame_number": str(frame.camera_frame_number),
                        "camera_timestamp_ticks": (
                            ""
                            if frame.camera_timestamp_ticks is None
                            else str(frame.camera_timestamp_ticks)
                        ),
                        "gain": f"{session.settings.gain:.12g}",
                        "pixel_format": session.settings.pixel_format,
                        "camera_model": session.settings.model,
                        "camera_serial": session.settings.serial_number,
                    },
                )

                print(
                    f"[{index}/{len(targets)}] "
                    f"exposure={requested_exposure:g} us, "
                    f"capture={capture_index}/{len(exposure_targets)} -> {target}"
                )

        print("\nCapture completed.")
        print(f"Output root: {output_root}")
        print("Move the chessboard to the next pose, then rerun with the next pose-id.")
        return 0

    finally:
        if started and session is not None:
            try:
                session.stop()
            except Exception as exc:
                print(f"WARNING: failed to stop camera stream: {exc}", file=sys.stderr)
        if session is not None:
            try:
                session.close()
            except Exception as exc:
                print(f"WARNING: failed to close camera: {exc}", file=sys.stderr)


def series_main(argv: Sequence[str] | None = None) -> int:
    parser = build_series_parser()
    args = parser.parse_args(argv)
    try:
        return capture_exposure_series(args)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(series_main())
