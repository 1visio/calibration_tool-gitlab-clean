"""线程安全的批量采集确认 gate。

gate 本身不触碰任何 Qt 控件。采集线程只发出 ``task_requested`` 信号并在
``threading.Event`` 上等待；主线程的按钮槽通过 ``approve``/``cancel`` 唤醒它。
"""

from __future__ import annotations

import threading
from typing import Any

from PySide6.QtCore import QObject, Signal


class CaptureTaskGate(QObject):
    """把每个 CaptureTask 的人工准备确认桥接到工作线程。"""

    task_requested = Signal(object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._state_lock = threading.Lock()
        self._approved = threading.Event()
        self._cancelled = threading.Event()

    def wait_for_task(
        self,
        task: Any,
        cancel_event: threading.Event | None = None,
    ) -> bool:
        """在采集线程等待主线程确认；取消时总能解除等待。"""

        with self._state_lock:
            self._approved.clear()
            self._cancelled.clear()
            if cancel_event is not None and cancel_event.is_set():
                return False
        self.task_requested.emit(task)
        while not self._approved.wait(timeout=0.1):
            if cancel_event is not None and cancel_event.is_set():
                self.cancel()
                break
            if self._cancelled.is_set():
                break
        return self._approved.is_set() and not self._cancelled.is_set()

    def approve(self) -> None:
        self._approved.set()

    def cancel(self) -> None:
        self._cancelled.set()
        # 让 wait() 立刻返回，而不是等超时轮询。
        self._approved.set()


__all__ = ["CaptureTaskGate"]
