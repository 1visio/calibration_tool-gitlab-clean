from __future__ import annotations

import csv
import io
import os
import shutil
import tempfile
import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cv2
import yaml

from ..errors import CaptureError
from ..io_utils import load_document, sha256_file
from .config import capture_plan_hash, capture_plan_payload
from .models import (
    CameraConfig,
    CameraProvider,
    CapturePlan,
    CaptureResult,
    CaptureTask,
    CapturedFrame,
    FrameQuality,
    ProgressCallback,
    QualityThresholds,
    requires_camera_reconfigure,
)
from .quality import analyze_frame, quality_to_dict, summarize_quality


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(text, encoding="utf-8")
        for attempt in range(8):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt == 7:
                    raise
                # Windows Defender、索引器或实时 CSV 读取可能短暂占用目标文件。
                time.sleep(min(0.02 * (2**attempt), 0.2))
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_yaml(path: Path, value: Any) -> None:
    _atomic_text(path, yaml.safe_dump(value, allow_unicode=True, sort_keys=False))


def _write_image(path: Path, frame: CapturedFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    ok, encoded = cv2.imencode(suffix, frame.image)
    if not ok:
        raise CaptureError(f"OpenCV 无法编码图像：{path}")
    temporary = path.with_name(f".{path.stem}.tmp{suffix}")
    temporary.write_bytes(encoded.tobytes())
    os.replace(temporary, path)


def _quality_summary_from_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    warning_counts: dict[str, int] = {}
    passed = 0
    for record in records:
        quality = record["quality"]
        if quality["passed"]:
            passed += 1
        for warning in quality["warnings"]:
            warning_counts[warning] = warning_counts.get(warning, 0) + 1
    total = len(records)
    return {
        "frames": total,
        "passed": passed,
        "warnings": total - passed,
        "warning_counts": warning_counts,
    }


def _write_frames_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "task_id", "pose_id", "role", "index", "filename", "sha256",
        "camera_frame_number", "camera_frame_gap", "camera_timestamp_ticks",
        "transport_warnings",
        "host_timestamp_ns", "host_monotonic_ns", "exposure_us", "gain_db",
        "pixel_format", "offset_x", "offset_y", "width", "height",
        "quality_passed", "quality_warnings", "mean_dn", "p01_dn", "p50_dn",
        "p99_dn", "dynamic_range_u8", "saturation_fraction", "dark_fraction",
        "focus_laplacian", "laser_coverage", "chessboard_detected",
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for record in records:
        applied = record["applied_camera"]
        quality = record["quality"]
        writer.writerow({
            **{key: (
                ";".join(record.get(key, [])) if key == "transport_warnings" else record.get(key)
            ) for key in fields},
            **{key: applied.get(key) for key in (
                "exposure_us", "gain_db", "pixel_format", "offset_x", "offset_y", "width", "height"
            )},
            "quality_passed": quality["passed"],
            "quality_warnings": ";".join(quality["warnings"]),
            **{key: quality.get(key) for key in (
                "mean_dn", "p01_dn", "p50_dn", "p99_dn", "dynamic_range_u8",
                "saturation_fraction", "dark_fraction", "focus_laplacian",
                "laser_coverage", "chessboard_detected",
            )},
        })
    _atomic_text(path, stream.getvalue())


def _new_manifest(plan: CapturePlan) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "dataset_id": plan.dataset_id,
        "status": "in_progress",
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "completed_at": None,
        "plan_sha256": capture_plan_hash(plan),
        "plan": capture_plan_payload(plan),
        "laser": {"orientation": plan.laser.orientation},
        "device": None,
        "network_packet_size": None,
        "tasks": {
            task.task_id: {
                "status": "pending",
                "frames_expected": task.frames,
                "frames_captured": 0,
                "error": None,
            }
            for task in plan.tasks
        },
        "frames": [],
        "quality_summary": {},
    }


def _save_state(
    work_dir: Path,
    manifest: dict[str, Any],
    *,
    write_frames_csv: bool = True,
) -> None:
    manifest["updated_at"] = _utc_now()
    manifest["quality_summary"] = _quality_summary_from_records(manifest["frames"])
    _atomic_yaml(work_dir / "dataset_manifest.yaml", manifest)
    if write_frames_csv:
        _write_frames_csv(work_dir / "frames.csv", manifest["frames"])


def run_capture_plan(
    plan: CapturePlan,
    provider: CameraProvider,
    *,
    resume: bool = False,
    interactive: bool = False,
    interactive_preview: bool = False,
    progress: ProgressCallback | None = None,
    prompt: Callable[[str], str] = input,
    before_task: Callable[[CaptureTask], bool | None] | None = None,
    cancel_event: threading.Event | None = None,
) -> CaptureResult:
    """执行采集计划；每个 task 完成后原子提交并更新可续采 manifest。

    ``before_task`` 在工作线程中调用，可用于 GUI 的人工确认 gate；调用前会先
    在已停止的 session 上应用当前 task 配置，返回 ``False`` 表示取消当前采集。
    ``cancel_event`` 是可选的线程安全取消信号，不传入时保持旧调用行为。
    """

    def cancelled() -> bool:
        return cancel_event is not None and cancel_event.is_set()

    def ensure_not_cancelled() -> None:
        if cancelled():
            raise CaptureError("采集已取消")
    output_dir = plan.output_dir.expanduser().resolve()
    work_dir = output_dir.parent / f".{output_dir.name}.inprogress"
    manifest_path = work_dir / "dataset_manifest.yaml"
    if output_dir.exists():
        raise CaptureError(f"输出数据集已经存在，不会覆盖：{output_dir}")
    if work_dir.exists() and not resume:
        raise CaptureError(f"发现未完成数据集；确认后使用 --resume：{work_dir}")

    if resume:
        if not manifest_path.is_file():
            raise CaptureError(f"没有可续采的 manifest：{manifest_path}")
        manifest = load_document(manifest_path)
        if manifest.get("plan_sha256") != capture_plan_hash(plan):
            raise CaptureError("当前采集计划与未完成数据集不一致，拒绝续采")
    else:
        work_dir.mkdir(parents=True, exist_ok=False)
        manifest = _new_manifest(plan)
        _save_state(work_dir, manifest)

    pending = [task for task in plan.tasks if manifest["tasks"][task.task_id]["status"] != "completed"]
    session = None
    active_task_id: str | None = None
    try:
        ensure_not_cancelled()
        if pending:
            session = provider.open(plan.serial_number, pending[0].config)
            manifest["device"] = asdict(session.device)
            manifest["network_packet_size"] = session.network_packet_size
            _save_state(work_dir, manifest)

        for task in pending:
            active_task_id = task.task_id
            task_state = manifest["tasks"][task.task_id]
            task_state.update(status="capturing", frames_captured=0, error=None, started_at=_utc_now())
            manifest["frames"] = [item for item in manifest["frames"] if item["task_id"] != task.task_id]
            _save_state(work_dir, manifest)
            if progress:
                progress({
                    "event": "task_started",
                    "task_id": task.task_id,
                    "pose_id": task.pose_id,
                    "role": task.role,
                    "exposure_us": task.config.exposure_us,
                    "laser_state": task.tags.get("laser_state", "unchanged"),
                    "instruction": task.instruction,
                    "relative_output_path": task.relative_path(1).as_posix(),
                })
            ensure_not_cancelled()
            # The task configuration must be applied while the session is idle,
            # before the GUI gate is released.  Otherwise the page can show the
            # next task while the camera is still using the previous task's
            # exposure/ROI until the user confirms it.  The session is stopped
            # after every task, so this remains a single worker-owned session.
            assert session is not None
            if session.config != task.config:
                if requires_camera_reconfigure(session.config, task.config):
                    session.configure(task.config)
                else:
                    # 常见三联图只改变曝光/增益。海康 MVS 的完整 configure
                    # 会关闭并重开 SDK session，因此优先使用两后端共有的在线接口。
                    session.update_exposure_gain(
                        task.config.exposure_us,
                        task.config.gain_db,
                    )
            if before_task is not None:
                try:
                    approved = before_task(task)
                except CaptureError:
                    raise
                except Exception as exc:
                    raise CaptureError(f"任务前确认失败：{exc}") from exc
                if approved is False:
                    raise CaptureError("采集已取消")
            ensure_not_cancelled()
            if interactive and task.instruction and not interactive_preview:
                prompt(f"{task.instruction}\n准备完成后按 Enter 开始采集 {task.task_id}：")
            if interactive_preview:
                _interactive_task_preview(session, task, plan)
            applied = session.config
            session.start()
            for _ in range(task.settle_frames):
                ensure_not_cancelled()
                session.get_frame(applied.timeout_ms)

            staging = work_dir / ".task_staging" / task.task_id
            if staging.exists():
                shutil.rmtree(staging)
            task_records: list[dict[str, Any]] = []
            previous_frame_number: int | None = None
            for index in range(1, task.frames + 1):
                ensure_not_cancelled()
                frame = session.get_frame(applied.timeout_ms)
                relative = task.relative_path(index)
                staged_path = staging / relative
                _write_image(staged_path, frame)
                quality = analyze_frame(
                    frame.image,
                    sensor_max_value=applied.sensor_max_value,
                    mode=task.quality_mode,
                    thresholds=plan.quality_thresholds,
                    board_pattern=plan.board_pattern,
                    laser_orientation=plan.laser.orientation,
                )
                gap = None if previous_frame_number is None else frame.camera_frame_number - previous_frame_number
                previous_frame_number = frame.camera_frame_number
                task_records.append({
                    "task_id": task.task_id,
                    "pose_id": task.pose_id,
                    "role": task.role,
                    "tags": task.tags,
                    "index": index,
                    "filename": relative.as_posix(),
                    "sha256": sha256_file(staged_path, normalize_newlines=False),
                    "camera_frame_number": frame.camera_frame_number,
                    "camera_frame_gap": gap,
                    "transport_warnings": [] if gap in (None, 1) else ["camera_frame_gap"],
                    "camera_timestamp_ticks": frame.camera_timestamp_ticks,
                    "host_timestamp_ns": frame.host_timestamp_ns,
                    "host_monotonic_ns": frame.host_monotonic_ns,
                    "requested_camera": asdict(task.config),
                    "applied_camera": asdict(applied),
                    "quality": quality_to_dict(quality),
                })
                if progress:
                    progress({
                        "event": "frame",
                        "task_id": task.task_id,
                        "pose_id": task.pose_id,
                        "role": task.role,
                        "index": index,
                        "frames": task.frames,
                        "relative_output_path": relative.as_posix(),
                        "exposure_us": task.config.exposure_us,
                        "laser_state": task.tags.get("laser_state", "unchanged"),
                        "quality": quality_to_dict(quality),
                    })
                    if task_records[-1]["quality"]["warnings"]:
                        progress({
                            "event": "quality_warning",
                            "task_id": task.task_id,
                            "index": index,
                            "relative_output_path": relative.as_posix(),
                            "warnings": task_records[-1]["quality"]["warnings"],
                        })
            session.stop()

            for record in task_records:
                relative = Path(record["filename"])
                destination = work_dir / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staging / relative, destination)
            shutil.rmtree(staging)
            manifest["frames"].extend(task_records)
            task_state.update(
                status="completed",
                frames_captured=len(task_records),
                completed_at=_utc_now(),
                quality_summary=_quality_summary_from_records(task_records),
            )
            _save_state(work_dir, manifest)
            if progress:
                progress({
                    "event": "task_completed",
                    "task_id": task.task_id,
                    "pose_id": task.pose_id,
                    "role": task.role,
                    "frames": task.frames,
                    "relative_output_path": task.relative_path(task.frames).as_posix(),
                })
            active_task_id = None

        manifest["status"] = "completed"
        manifest["completed_at"] = _utc_now()
        _save_state(work_dir, manifest)
        staging_root = work_dir / ".task_staging"
        if staging_root.is_dir():
            staging_root.rmdir()
        os.replace(work_dir, output_dir)
    except Exception as exc:
        if session is not None:
            try:
                session.stop()
            except Exception:
                pass
        if active_task_id and manifest_path.parent.exists():
            manifest["tasks"][active_task_id].update(status="failed", error=str(exc))
            manifest["status"] = "failed"
            _save_state(work_dir, manifest)
        if isinstance(exc, CaptureError):
            raise
        raise CaptureError(f"采集失败，现场已保留在 {work_dir}：{exc}") from exc
    finally:
        if session is not None:
            try:
                session.close()
            except Exception:
                pass

    summary = manifest["quality_summary"]
    return CaptureResult(
        output_dir=output_dir,
        task_count=len(plan.tasks),
        frame_count=int(summary["frames"]),
        quality_passed_frames=int(summary["passed"]),
        quality_warning_frames=int(summary["warnings"]),
    )


def _interactive_task_preview(
    session,
    task,
    plan: CapturePlan,
) -> None:
    """在批量采集前打开可确认的实时窗口，不写入预览帧。"""
    window = f"capture-plan · {task.task_id}"
    try:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        session.start()
        while True:
            frame = session.get_frame(session.config.timeout_ms)
            quality = analyze_frame(
                frame.image,
                sensor_max_value=session.config.sensor_max_value,
                mode=task.quality_mode,
                thresholds=plan.quality_thresholds,
                board_pattern=plan.board_pattern,
                laser_orientation=plan.laser.orientation,
            )
            display = cv2.convertScaleAbs(
                frame.image,
                alpha=255.0 / float(session.config.sensor_max_value),
            )
            if task.quality_mode == "chessboard" and plan.board_pattern is not None:
                _draw_chessboard_corners(display, plan.board_pattern)
            display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
            lines = [
                f"{task.task_id}  exposure={session.config.exposure_us:g} us  gain={session.config.gain_db:g} dB",
                f"settle_frames={task.settle_frames}  frames={task.frames}",
                f"quality={'PASS' if quality.passed else 'WARN: ' + ', '.join(quality.warnings)}",
                f"range={quality.dynamic_range_u8:.1f} DN8  focus={quality.focus_laplacian:.1f}",
                "Enter/Space: continue    Esc/Q: abort",
            ]
            if quality.laser_coverage is not None:
                lines.append(f"laser coverage={quality.laser_coverage:.1%}")
            if quality.chessboard_hint:
                lines.append(quality.chessboard_hint)
            for index, line in enumerate(lines):
                cv2.putText(
                    display,
                    line,
                    (16, 30 + index * 26),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0) if quality.passed else (0, 165, 255),
                    2,
                    cv2.LINE_AA,
                )
            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (13, 32):
                return
            if key in (27, ord("q"), ord("Q")):
                raise CaptureError(f"用户取消任务预览：{task.task_id}")
    except CaptureError:
        raise
    except Exception as exc:
        raise CaptureError(f"无法打开交互预览窗口：{exc}") from exc
    finally:
        try:
            session.stop()
        finally:
            try:
                cv2.destroyWindow(window)
                cv2.waitKey(1)
            except cv2.error:
                pass


def _draw_chessboard_corners(image, pattern: tuple[int, int]) -> bool:
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(image, pattern, flags)
    if not found and hasattr(cv2, "findChessboardCornersSB"):
        found, corners = cv2.findChessboardCornersSB(
            image,
            pattern,
            cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE,
        )
    if found:
        cv2.drawChessboardCorners(image, pattern, corners, True)
    return bool(found)


def preview_camera(
    provider: CameraProvider,
    serial_number: str,
    config: CameraConfig,
    *,
    frames: int = 20,
    warmup_frames: int = 5,
    quality_mode: str = "generic",
    quality_thresholds: QualityThresholds = QualityThresholds(),
    board_pattern: tuple[int, int] | None = None,
    laser_orientation: str = "horizontal",
    snapshot: Path | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    if frames <= 0 or warmup_frames < 0:
        raise CaptureError("frames 必须为正数，warmup_frames 不能为负数")
    session = provider.open(serial_number, config)
    qualities: list[FrameQuality] = []
    last_frame: CapturedFrame | None = None
    try:
        session.start()
        for _ in range(warmup_frames):
            session.get_frame(session.config.timeout_ms)
        for index in range(1, frames + 1):
            last_frame = session.get_frame(session.config.timeout_ms)
            quality = analyze_frame(
                last_frame.image,
                sensor_max_value=session.config.sensor_max_value,
                mode=quality_mode,
                thresholds=quality_thresholds,
                board_pattern=board_pattern,
                laser_orientation=laser_orientation,
            )
            qualities.append(quality)
            if progress:
                progress({"event": "preview_frame", "index": index, "quality": quality_to_dict(quality)})
        if snapshot is not None and last_frame is not None:
            _write_image(snapshot.expanduser().resolve(), last_frame)
    except Exception as exc:
        if isinstance(exc, CaptureError):
            raise
        raise CaptureError(f"相机预览失败：{exc}") from exc
    finally:
        try:
            session.stop()
        finally:
            session.close()
    return {
        "device": asdict(session.device),
        "applied_camera": asdict(session.config),
        "frames": frames,
        "quality_summary": summarize_quality(qualities),
        "last_frame_quality": quality_to_dict(qualities[-1]),
        "snapshot": str(snapshot.expanduser().resolve()) if snapshot else None,
    }
