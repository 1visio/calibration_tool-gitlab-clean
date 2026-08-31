"""线激光统一标定工具的阶段 0/1 核心。"""

from .laser_models import DEFAULT_LASER_MODEL, SUPPORTED_LASER_MODEL_TYPES
from .calibration_run import CalibrationRun, CalibrationStage
from .calibration_run_io import (
    calibration_run_from_report,
    infer_calibration_run_root,
    load_calibration_run,
    save_calibration_run,
)
from .calibration_results import (
    CalibrationResultsSummary,
    GroundExtrinsicsDetails,
    GroundValidationRow,
    IntrinsicsDetails,
    IntrinsicsReprojectionRow,
    LaserModelComparison,
    LaserModelParameters,
    LaserSurfaceDetails,
    load_laser_surface_details,
    load_ground_extrinsics_details,
    load_intrinsics_details,
    summarize_calibration_run,
)

__version__ = "0.1.0"

__all__ = [
    "CalibrationRun",
    "CalibrationStage",
    "CalibrationResultsSummary",
    "GroundExtrinsicsDetails",
    "GroundValidationRow",
    "IntrinsicsDetails",
    "IntrinsicsReprojectionRow",
    "LaserModelComparison",
    "LaserModelParameters",
    "LaserSurfaceDetails",
    "DEFAULT_LASER_MODEL",
    "SUPPORTED_LASER_MODEL_TYPES",
    "calibration_run_from_report",
    "infer_calibration_run_root",
    "load_calibration_run",
    "save_calibration_run",
    "load_laser_surface_details",
    "load_ground_extrinsics_details",
    "load_intrinsics_details",
    "summarize_calibration_run",
    "__version__",
]
