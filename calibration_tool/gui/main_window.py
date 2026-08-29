from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThreadPool, Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from .pages import CalibrationPage, CameraPage, CapturePage, ProjectPage, ResultsPage
from .project import WizardProject


class CalibrationWizardWindow(QMainWindow):
    STEP_NAMES = ("项目", "相机与曝光", "批量采集", "一键标定", "报告、补偿与验收")

    def __init__(
        self,
        *,
        project_path: Path | None = None,
        default_camera_config: Path,
        default_camera_channel: str | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.thread_pool = QThreadPool(self)
        self.thread_pool.setMaxThreadCount(3)
        self.project: WizardProject | None = None
        self.setWindowTitle("线激光扫描系统 · 标定向导 MVP")
        self.resize(1440, 900)
        self._build_ui(default_camera_config, default_camera_channel)
        self._connect_signals()
        self._apply_style()
        if project_path is not None:
            self.load_project(project_path)
        else:
            self.project_page.apply()

    def _build_ui(
        self,
        default_camera_config: Path,
        default_camera_channel: str | None,
    ) -> None:
        central = QWidget(self); root = QVBoxLayout(central); root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)
        header = QFrame(); header.setObjectName("header"); header_layout = QHBoxLayout(header)
        title = QLabel("线激光标定向导"); title.setObjectName("appTitle")
        self.project_label = QLabel("未选择项目")
        header_layout.addWidget(title); header_layout.addSpacing(20); header_layout.addWidget(self.project_label); header_layout.addStretch()
        root.addWidget(header)
        body = QWidget(); body_layout = QHBoxLayout(body); body_layout.setContentsMargins(0, 0, 0, 0); body_layout.setSpacing(0)
        self.steps = QListWidget(); self.steps.setObjectName("steps"); self.steps.setFixedWidth(190)
        self.steps.addItems([f"{index + 1}  {name}" for index, name in enumerate(self.STEP_NAMES)])
        self.stack = QStackedWidget()
        self.project_page = ProjectPage(default_camera_config, default_camera_channel)
        self.camera_page = CameraPage(self.thread_pool)
        self.capture_page = CapturePage(self.thread_pool, self.camera_page)
        self.calibration_page = CalibrationPage(self.thread_pool)
        self.results_page = ResultsPage(self.thread_pool)
        for page in (self.project_page, self.camera_page, self.capture_page, self.calibration_page, self.results_page):
            wrapper = QWidget(); layout = QVBoxLayout(wrapper); layout.setContentsMargins(24, 20, 24, 16); layout.addWidget(page)
            self.stack.addWidget(wrapper)
        body_layout.addWidget(self.steps); body_layout.addWidget(self.stack, 1); root.addWidget(body, 1)
        footer = QFrame(); footer.setObjectName("footer"); footer_layout = QHBoxLayout(footer)
        self.previous_button = QPushButton("上一步"); self.next_button = QPushButton("下一步")
        footer_layout.addStretch(); footer_layout.addWidget(self.previous_button); footer_layout.addWidget(self.next_button)
        root.addWidget(footer); self.setCentralWidget(central); self.setStatusBar(QStatusBar())
        self.steps.setCurrentRow(0)

    def _connect_signals(self) -> None:
        self.steps.currentRowChanged.connect(self._set_step)
        self.previous_button.clicked.connect(lambda: self.steps.setCurrentRow(max(0, self.steps.currentRow() - 1)))
        self.next_button.clicked.connect(lambda: self.steps.setCurrentRow(min(len(self.STEP_NAMES) - 1, self.steps.currentRow() + 1)))
        self.project_page.project_changed.connect(self.set_project)
        self.capture_page.capture_finished.connect(self._capture_finished)
        self.capture_page.request_camera_page.connect(lambda: self.steps.setCurrentRow(1))
        self.calibration_page.workflow_finished.connect(self._workflow_finished)

    def _set_step(self, index: int) -> None:
        if index < 0:
            return
        self.stack.setCurrentIndex(index)
        if index == 4:
            self.results_page.prepare_default_plan(announce=False)
        self.previous_button.setEnabled(index > 0); self.next_button.setEnabled(index < len(self.STEP_NAMES) - 1)

    def set_project(self, project: WizardProject) -> None:
        self.project = project
        self.project_label.setText(f"{project.project_id}  ·  {project.workspace}")
        self.camera_page.stop_preview()
        loaded = self.camera_page.load_config(
            project.camera_config,
            channel=project.camera_channel,
        )
        if loaded and self.camera_page.runtime is not None:
            self.camera_page.runtime["laser"] = project.laser
            # 项目/通道应用后后台枚举；失败只写状态，不在启动时弹出阻塞对话框。
            self.camera_page.enumerate_devices(silent=True)
        self.capture_page.set_project(project)
        self.calibration_page.set_project(project)
        self.results_page.set_project(project)
        # ResultsPage 会为未指定计划的项目准备默认验收计划；同步回项目页，
        # 用户随后保存项目时也能保留该路径。
        self.project_page.acceptance_plan.setText(str(project.acceptance_plan or ""))
        self.statusBar().showMessage("项目配置已应用", 5000)

    def load_project(self, path: Path) -> None:
        project = WizardProject.load(path)
        self.project_page.set_project(project)
        self.project_page.apply()

    def _workflow_finished(self, result: dict) -> None:
        self.results_page.update_acceptance_from_workflow(result)
        self.results_page.show_result(result)
        self.steps.setCurrentRow(4)
        self.statusBar().showMessage(f"标定 workflow：{result.get('status')}", 15000)

    def _capture_finished(self, result: object) -> None:
        artifacts = getattr(result, "capture_artifacts", None) or self.capture_page.last_capture_artifacts
        if isinstance(artifacts, dict):
            if self.project is not None:
                try:
                    backup = self.project.record_capture_artifacts(artifacts)
                except Exception as exc:
                    self.statusBar().showMessage(f"采集完成，但项目路径记录保存失败：{exc}", 15000)
                else:
                    suffix = f"；项目备份：{backup}" if backup else "；关闭前保存项目即可保留记录"
                    self.statusBar().showMessage(f"采集完成：{artifacts.get('dataset_root')}{suffix}", 15000)
            else:
                self.statusBar().showMessage(f"采集完成：{artifacts.get('dataset_root')}", 10000)
            self.calibration_page.set_capture_artifacts(artifacts)
            self.results_page.set_capture_artifacts(artifacts)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        self.capture_page.cancel_capture()
        self.camera_page.stop_preview()
        self.thread_pool.waitForDone(5000)
        event.accept()

    def _apply_style(self) -> None:
        self.setStyleSheet("""
            QMainWindow { background: #f3f5f7; }
            QFrame#header { background: #17212b; color: white; }
            QFrame#header QLabel { color: white; }
            QLabel#appTitle { font-size: 20px; font-weight: 700; }
            QLabel#pageTitle { font-size: 20px; font-weight: 650; color: #17212b; margin-bottom: 8px; }
            QListWidget#steps { background: #222f3d; color: #dbe3eb; border: 0; padding-top: 12px; font-size: 14px; }
            QListWidget#steps::item { height: 48px; padding-left: 12px; }
            QListWidget#steps::item:selected { background: #1769aa; color: white; }
            QFrame#footer { background: white; border-top: 1px solid #d7dce2; }
            QPushButton { min-height: 30px; padding: 2px 12px; }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { min-height: 28px; }
            QGroupBox { font-weight: 600; }
        """)
