from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
from PySide6.QtCore import QCoreApplication

from calibration_tool.camera.models import (
    CameraConfig,
    CapturedFrame,
    QualityThresholds,
)
from calibration_tool.camera.quality import analyze_frame, quality_to_dict
from calibration_tool.camera.steger_quality import analyze_search_region_health
from calibration_tool.gui.pages import _search_region_quality_text
from calibration_tool.gui.workers import PreviewThread


def _extraction(
    *,
    normal_axis: str,
    coordinates: list[float],
    valid: list[bool],
    outside: list[bool],
    start: float = 10.0,
    end: float = 30.0,
) -> SimpleNamespace:
    values = np.asarray(coordinates, dtype=np.float64)
    return SimpleNamespace(
        u_px=values if normal_axis == "u" else np.arange(len(values), dtype=float),
        v_px=values if normal_axis == "v" else np.arange(len(values), dtype=float),
        valid=np.asarray(valid, dtype=bool),
        metadata={
            "normal_axis": normal_axis,
            "final_search_region_start_px": start,
            "final_search_region_end_px": end,
        },
        diagnostics=SimpleNamespace(
            intensity_peak_outside_detected_band=np.asarray(outside, dtype=bool)
        ),
    )


class SearchRegionHealthTests(unittest.TestCase):
    def test_column_metrics_use_original_v_normal_axis(self) -> None:
        health = analyze_search_region_health(
            _extraction(
                normal_axis="v",
                coordinates=[15.0, 25.0, 12.0, np.nan],
                valid=[True, True, True, False],
                outside=[False, True, False, False],
            ),
            sigma=1.5,
        )

        self.assertEqual(health.normal_axis, "v")
        self.assertEqual(health.search_region_size_px, 20.0)
        self.assertEqual(health.boundary_clearance_min_px, 2.0)
        self.assertAlmostEqual(health.boundary_clearance_p05_px, 2.3)
        self.assertEqual(health.boundary_clearance_median_px, 5.0)
        self.assertEqual(health.kernel_support_px, 6)
        self.assertEqual(health.boundary_inside_kernel_fraction, 1.0)
        self.assertEqual(health.outside_search_region_peak_fraction, 0.25)
        self.assertEqual(
            health.warning_reasons,
            (
                "center_near_search_boundary",
                "possible_signal_outside_search_region",
            ),
        )

    def test_row_metrics_use_original_u_normal_axis(self) -> None:
        health = analyze_search_region_health(
            _extraction(
                normal_axis="u",
                coordinates=[18.0, 20.0, 22.0],
                valid=[True, True, True],
                outside=[False, False, False],
            ),
            sigma=1.5,
        )

        self.assertEqual(health.normal_axis, "u")
        self.assertEqual(health.boundary_clearance_p05_px, 8.0)
        self.assertEqual(health.boundary_inside_kernel_fraction, 0.0)
        self.assertEqual(health.status, "GOOD")
        self.assertEqual(health.warning_reasons, ())

    def test_gui_text_shows_metrics_and_machine_readable_reasons(self) -> None:
        text = _search_region_quality_text(
            {
                "search_region_health": {
                    "status": "WARNING",
                    "boundary_clearance_p05_px": 3.535,
                    "kernel_support_px": 6,
                    "outside_search_region_peak_fraction": 0.282,
                    "warning_reasons": [
                        "center_near_search_boundary",
                        "possible_signal_outside_search_region",
                    ],
                }
            }
        )

        self.assertIn("Search region: WARNING", text)
        self.assertIn("Boundary P05: 3.5 px", text)
        self.assertIn("Kernel support: 6 px", text)
        self.assertIn("Outside-band risk: 28.2%", text)
        self.assertIn("center_near_search_boundary", text)
        self.assertIn("possible_signal_outside_search_region", text)


class _FakeSession:
    def __init__(self, image: np.ndarray) -> None:
        self.config = CameraConfig(width=image.shape[1], height=image.shape[0])
        self.frame = CapturedFrame(
            image=image,
            camera_frame_number=1,
            camera_timestamp_ticks=None,
            host_timestamp_ns=1,
            host_monotonic_ns=1,
        )

    def get_frame(self, _timeout_ms: int) -> CapturedFrame:
        return self.frame


class _CountingAnalyzer:
    def __init__(self) -> None:
        self.calls = 0

    def analyze(self, _image: np.ndarray) -> dict[str, object]:
        self.calls += 1
        return {
            "status": "GOOD",
            "boundary_clearance_p05_px": 20.0,
            "kernel_support_px": 6,
            "outside_search_region_peak_fraction": 0.0,
            "warning_reasons": [],
        }


class PreviewThreadSearchRegionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QCoreApplication.instance() or QCoreApplication([])

    def test_laser_preview_runs_one_steger_analysis_without_changing_formal_quality(self) -> None:
        image = np.zeros((48, 80), dtype=np.uint8)
        image[23:26, :] = 120
        analyzer = _CountingAnalyzer()
        thread = PreviewThread(
            provider=None,
            serial_number="SIM",
            config=CameraConfig(width=80, height=48),
            quality_mode="laser",
            thresholds=QualityThresholds(),
            board_pattern=None,
            steger_quality_analyzer=analyzer,
        )
        emitted: list[dict[str, object]] = []
        thread.frame_ready.connect(lambda _frame, quality: emitted.append(quality))

        thread._emit_frame(_FakeSession(image))

        self.assertEqual(analyzer.calls, 1)
        self.assertEqual(len(emitted), 1)
        expected = quality_to_dict(
            analyze_frame(image, sensor_max_value=255, mode="laser")
        )
        self.assertEqual(emitted[0]["warnings"], expected["warnings"])
        self.assertEqual(emitted[0]["passed"], expected["passed"])
        self.assertEqual(emitted[0]["search_region_health"]["status"], "GOOD")

    def test_non_laser_preview_does_not_run_steger(self) -> None:
        image = np.zeros((48, 80), dtype=np.uint8)
        analyzer = _CountingAnalyzer()
        thread = PreviewThread(
            provider=None,
            serial_number="SIM",
            config=CameraConfig(width=80, height=48),
            quality_mode="generic",
            thresholds=QualityThresholds(),
            board_pattern=None,
            steger_quality_analyzer=analyzer,
        )

        thread._emit_frame(_FakeSession(image))

        self.assertEqual(analyzer.calls, 0)


if __name__ == "__main__":
    unittest.main()
