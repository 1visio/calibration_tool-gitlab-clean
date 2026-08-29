from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import threading
import time
from dataclasses import replace
from typing import Any, Callable

from PySide6.QtCore import QObject, QRunnable, QThread, Signal, Slot

from ..camera.models import (
    CameraConfig,
    CameraProvider,
    QualityThresholds,
    requires_camera_reconfigure,
)
from ..camera.quality import analyze_frame, quality_to_dict


class WorkerSignals(QObject):
    started = Signal()
    progress = Signal(object)
    result = Signal(object)
    error = Signal(str)
    finished = Signal()


class FunctionWorker(QRunnable):
    """在线程池中运行 `fn(progress_callback)`。"""

    def __init__(self, fn: Callable[[Callable[[object], None]], Any]) -> None:
        super().__init__()
        self.fn = fn
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        self.signals.started.emit()
        try:
            result = self.fn(self.signals.progress.emit)
        except Exception as exc:  # Qt 边界统一转换为可展示消息
            self.signals.error.emit(str(exc))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()


class PreviewThread(QThread):
    frame_ready = Signal(object, object)
    opened = Signal(object, object)
    failed = Signal(str)
    settings_applied = Signal(object)
    parameter_update_failed = Signal(str)

    def __init__(
        self,
        provider: CameraProvider,
        serial_number: str,
        config: CameraConfig,
        quality_mode: str,
        thresholds: QualityThresholds,
        board_pattern: tuple[int, int] | None,
        laser_orientation: str = "horizontal",
        steger_quality_analyzer: Any | None = None,
        initial_discard_frames: int = 3,
        parent: QObject | None = None,
        quality_refresh_interval_s: float = 0.25,
    ) -> None:
        super().__init__(parent)
        self.provider = provider
        self.serial_number = serial_number
        self.config = config
        self.quality_mode = quality_mode
        self.thresholds = thresholds
        self.board_pattern = board_pattern
        self.laser_orientation = laser_orientation
        self.steger_quality_analyzer = steger_quality_analyzer
        self.initial_discard_frames = max(0, int(initial_discard_frames))
        self.quality_refresh_interval_s = max(0.0, float(quality_refresh_interval_s))
        self._parameter_lock = threading.Lock()
        self._pending_exposure_gain: tuple[float, float] | None = None
        self._pending_task: tuple[CameraConfig, str, int] | None = None
        self._fresh_quality_after_frames: int | None = None
        self._quality_executor: ThreadPoolExecutor | None = None
        self._quality_future: Future[tuple[int, int, dict[str, Any]]] | None = None
        self._quality_generation = 0
        self._quality_payload: dict[str, Any] | None = None
        self._next_quality_submission_at = 0.0

    def request_exposure_gain(self, exposure_us: float, gain_db: float) -> None:
        with self._parameter_lock:
            self._pending_exposure_gain = (float(exposure_us), float(gain_db))

    def request_task_config(
        self,
        config: CameraConfig,
        quality_mode: str,
        settle_frames: int = 0,
    ) -> bool:
        """在线请求下一个任务配置，不中断普通曝光/增益切换的取流。

        默认三联图只改变曝光和质量模式，工作线程会在同一个 session 上
        更新参数并继续发帧；ROI/像素格式变化则由线程内部执行必要的停流重配。
        """

        if not self.isRunning():
            return False
        with self._parameter_lock:
            self._pending_task = (config, str(quality_mode), max(0, int(settle_frames)))
        return True

    def request_fresh_quality_after_frames(self, discard_frames: int = 0) -> None:
        """要求丢弃指定帧后，对下一帧同步计算完整质量。

        GUI 普通预览可以复用最近一次质量结果；真正保存帧通过此请求取得与
        当前 camera_frame_number 对应的完整结果，不把限频结果写入数据集。
        """

        with self._parameter_lock:
            self._fresh_quality_after_frames = max(0, int(discard_frames))

    def cancel_fresh_quality_request(self) -> None:
        with self._parameter_lock:
            self._fresh_quality_after_frames = None

    def _take_pending_exposure_gain(self) -> tuple[float, float] | None:
        with self._parameter_lock:
            pending = self._pending_exposure_gain
            self._pending_exposure_gain = None
            return pending

    def _take_pending_task(self) -> tuple[CameraConfig, str, int] | None:
        with self._parameter_lock:
            pending = self._pending_task
            self._pending_task = None
            return pending

    def _take_fresh_quality_due(self) -> bool:
        with self._parameter_lock:
            remaining = self._fresh_quality_after_frames
            if remaining is None:
                return False
            if remaining > 0:
                self._fresh_quality_after_frames = remaining - 1
                return False
            self._fresh_quality_after_frames = None
            return True

    @staticmethod
    def _requires_restart(current: CameraConfig, target: CameraConfig) -> bool:
        return requires_camera_reconfigure(current, target)

    def _emit_frame(
        self,
        session: Any,
        *,
        settling: bool = False,
        settle_frames_remaining: int = 0,
    ) -> None:
        """读取并发送一帧预览。

        稳定帧仍然发送给 GUI，避免任务切换时画面停在上一帧；额外的
        ``settling`` 元数据只用于 GUI 禁用保存按钮，不改变质量分析结果。
        """

        frame = session.get_frame(session.config.timeout_ms)
        payload = self._quality_for_frame(
            frame,
            sensor_max_value=session.config.sensor_max_value,
            force_fresh=self._take_fresh_quality_due(),
        )
        payload["settling"] = bool(settling)
        payload["settle_frames_remaining"] = max(0, int(settle_frames_remaining))
        self.frame_ready.emit(frame, payload)

    def _compute_quality_payload(
        self,
        image: Any,
        *,
        sensor_max_value: float,
        quality_mode: str,
        generation: int,
        camera_frame_number: int,
    ) -> tuple[int, int, dict[str, Any]]:
        processing_started = time.perf_counter()
        quality = analyze_frame(
            image,
            sensor_max_value=sensor_max_value,
            mode=quality_mode,
            thresholds=self.thresholds,
            board_pattern=self.board_pattern,
            laser_orientation=self.laser_orientation,
        )
        payload = quality_to_dict(quality)
        if quality_mode == "laser" and self.steger_quality_analyzer is not None:
            try:
                payload["search_region_health"] = self.steger_quality_analyzer.analyze(
                    image
                )
            except Exception as exc:
                # search-region health 是旁路辅助显示，失败不得中断预览或改变
                # FrameQuality.warnings/passed 与保存门禁。
                payload["search_region_health"] = {
                    "status": "WARNING",
                    "warning_reasons": ["search_region_health_unavailable"],
                    "error": str(exc),
                }
        payload["preview_quality_processing_ms"] = (
            time.perf_counter() - processing_started
        ) * 1000.0
        payload["preview_quality_source_frame_number"] = int(camera_frame_number)
        return generation, int(camera_frame_number), payload

    def _collect_quality_future(self, *, wait: bool = False) -> None:
        future = self._quality_future
        if future is None or (not wait and not future.done()):
            return
        generation, _frame_number, payload = future.result()
        self._quality_future = None
        if generation == self._quality_generation:
            self._quality_payload = payload
            self._next_quality_submission_at = (
                time.perf_counter() + self.quality_refresh_interval_s
            )

    def _store_quality_result(
        self,
        result: tuple[int, int, dict[str, Any]],
    ) -> None:
        generation, _frame_number, payload = result
        if generation == self._quality_generation:
            self._quality_payload = payload
            self._next_quality_submission_at = (
                time.perf_counter() + self.quality_refresh_interval_s
            )

    def _quality_for_frame(
        self,
        frame: Any,
        *,
        sensor_max_value: float,
        force_fresh: bool,
    ) -> dict[str, Any]:
        self._collect_quality_future()
        generation = self._quality_generation
        frame_number = int(frame.camera_frame_number)
        mode = self.quality_mode

        if force_fresh:
            # 同一个 analyzer 不并发执行；保存帧宁可等待旧的 preview 分析完成，
            # 也必须对当前像素执行一次完整质量计算。
            self._collect_quality_future(wait=True)
            self._store_quality_result(self._compute_quality_payload(
                frame.image,
                sensor_max_value=sensor_max_value,
                quality_mode=mode,
                generation=generation,
                camera_frame_number=frame_number,
            ))
        elif self._quality_payload is None:
            # 首帧同步建立完整 payload，避免 GUI 在后台分析完成前显示伪造值。
            self._collect_quality_future(wait=True)
            if self._quality_payload is None:
                self._store_quality_result(self._compute_quality_payload(
                    frame.image,
                    sensor_max_value=sensor_max_value,
                    quality_mode=mode,
                    generation=generation,
                    camera_frame_number=frame_number,
                ))
        elif (
            self._quality_executor is not None
            and self._quality_future is None
            and time.perf_counter() >= self._next_quality_submission_at
        ):
            # 最多保留一个正在运行的分析；不排队旧帧。下一次提交总是使用
            # 当时最新的相机帧，并复制像素以隔离相机 SDK buffer 生命周期。
            image_snapshot = frame.image.copy()
            self._quality_future = self._quality_executor.submit(
                self._compute_quality_payload,
                image_snapshot,
                sensor_max_value=sensor_max_value,
                quality_mode=mode,
                generation=generation,
                camera_frame_number=frame_number,
            )

        assert self._quality_payload is not None
        payload = dict(self._quality_payload)
        source_frame = int(payload["preview_quality_source_frame_number"])
        payload["preview_quality_fresh"] = source_frame == frame_number
        payload["preview_quality_reused"] = source_frame != frame_number
        payload["preview_quality_age_frames"] = max(0, frame_number - source_frame)
        return payload

    def _invalidate_quality_cache(self) -> None:
        self._quality_generation += 1
        self._quality_payload = None
        self._next_quality_submission_at = 0.0

    def _shutdown_quality_executor(self) -> None:
        executor = self._quality_executor
        self._quality_executor = None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        self._quality_future = None

    def run(self) -> None:
        session = None
        try:
            self._quality_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="calibration-preview-quality",
            )
            session = self.provider.open(self.serial_number, self.config)
            self.opened.emit(session.device, session.config)
            session.start()
            for remaining in range(self.initial_discard_frames, 0, -1):
                # 保持实时画面；最后一帧稳定帧到达后即可允许保存。
                self._emit_frame(
                    session,
                    settling=remaining > 1,
                    settle_frames_remaining=remaining - 1,
                )
            while not self.isInterruptionRequested():
                pending_task = self._take_pending_task()
                if pending_task is not None:
                    target, quality_mode, settle_frames = pending_task
                    try:
                        if self._requires_restart(self.config, target):
                            # 仅结构性参数需要停流；曝光/增益和质量模式保持连续取流。
                            session.stop()
                            applied = session.configure(target)
                            session.start()
                        else:
                            applied = session.config
                            if (
                                applied.exposure_us != target.exposure_us
                                or applied.gain_db != target.gain_db
                            ):
                                applied = session.update_exposure_gain(
                                    target.exposure_us, target.gain_db
                                )
                            applied = replace(
                                applied,
                                pixel_format=target.pixel_format,
                                offset_x=target.offset_x,
                                offset_y=target.offset_y,
                                width=target.width,
                                height=target.height,
                                timeout_ms=target.timeout_ms,
                            )
                        self.config = applied
                        self.quality_mode = quality_mode
                        self._invalidate_quality_cache()
                        self.settings_applied.emit(self.config)
                        for remaining in range(settle_frames, 0, -1):
                            self._emit_frame(
                                session,
                                settling=remaining > 1,
                                settle_frames_remaining=remaining - 1,
                            )
                    except Exception as exc:
                        self.parameter_update_failed.emit(str(exc))

                pending = self._take_pending_exposure_gain()
                if pending is not None:
                    try:
                        self.config = session.update_exposure_gain(*pending)
                    except Exception as exc:
                        self.parameter_update_failed.emit(str(exc))
                    else:
                        self._invalidate_quality_cache()
                        self.settings_applied.emit(self.config)
                        # 旧曝光仍可能滞留在传输队列中，但这些帧也继续显示。
                        for remaining in range(2, 0, -1):
                            self._emit_frame(
                                session,
                                settling=remaining > 1,
                                settle_frames_remaining=remaining - 1,
                            )
                self._emit_frame(session)
        except Exception as exc:
            if not self.isInterruptionRequested():
                self.failed.emit(str(exc))
        finally:
            self._shutdown_quality_executor()
            if session is not None:
                try:
                    session.stop()
                finally:
                    session.close()

    def stop(self, timeout_ms: int = 5000) -> bool:
        self.requestInterruption()
        return self.wait(timeout_ms)
