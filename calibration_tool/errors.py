class CalibrationToolError(RuntimeError):
    """统一标定工具可向用户展示的错误。"""


class ConfigError(CalibrationToolError):
    """配置缺失、格式错误或路径不安全。"""


class GoldenBaselineError(CalibrationToolError):
    """Golden baseline 创建或校验失败。"""


class StageExecutionError(CalibrationToolError):
    """标定阶段执行失败。"""


class CameraError(CalibrationToolError):
    """相机连接、取流或参数配置失败。"""


class CaptureError(CalibrationToolError):
    """采集计划无效或采集过程失败。"""
