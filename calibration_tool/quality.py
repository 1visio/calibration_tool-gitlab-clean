from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .errors import ConfigError
from .io_utils import dump_yaml, load_document


def audit_baseline(
    baseline_path: str | Path,
    policy_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    baseline = load_document(baseline_path)
    policy = load_document(policy_path)
    gates: list[dict[str, Any]] = []

    for source_name, source in _mapping(baseline.get("sources"), "sources").items():
        for finding in source.get("findings", []):
            gates.append(
                {
                    "id": f"source.{source_name}.{finding['code']}",
                    "status": "fail" if finding.get("severity") == "fail" else "warn",
                    "actual": None,
                    "expected": None,
                    "message": finding.get("message", ""),
                }
            )
        if source.get("manifest") is None:
            gates.append(
                {
                    "id": f"source.{source_name}.manifest_missing",
                    "status": "warn",
                    "actual": None,
                    "expected": "self-contained manifest",
                    "message": "运行配置尚未绑定自包含 manifest",
                }
            )

    regressions = _mapping(baseline.get("regressions"), "regressions")
    for item in policy.get("metric_gates", []):
        if not isinstance(item, Mapping):
            raise ConfigError("metric_gates 每项必须是映射")
        metric_ref = str(item["metric"])
        group, _, metric = metric_ref.partition(".")
        if not group or not metric or group not in regressions:
            raise ConfigError(f"未知回归指标：{metric_ref}")
        values = regressions[group].get("values", {})
        if metric not in values:
            raise ConfigError(f"回归指标不存在：{metric_ref}")
        actual = values[metric]
        expected = item.get("expected")
        passed = _compare(actual, str(item.get("op", "eq")), expected)
        failure_status = str(item.get("failure_status", "fail"))
        gates.append(
            {
                "id": str(item.get("id", metric_ref)),
                "status": "pass" if passed else failure_status,
                "actual": actual,
                "expected": {"op": item.get("op", "eq"), "value": expected},
                "message": str(item.get("message", "")),
            }
        )

    counts = {status: sum(gate["status"] == status for gate in gates) for status in ("pass", "warn", "fail")}
    overall = "fail" if counts["fail"] else ("warn" if counts["warn"] else "pass")
    report = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "baseline": str(Path(baseline_path).resolve()),
        "policy": str(Path(policy_path).resolve()),
        "overall": overall,
        "counts": counts,
        "gates": gates,
    }
    if output_path is not None:
        dump_yaml(output_path, report)
    return report


def _compare(actual: Any, op: str, expected: Any) -> bool:
    if op == "eq":
        return actual == expected
    if op == "ne":
        return actual != expected
    if op == "le":
        return actual is not None and float(actual) <= float(expected)
    if op == "lt":
        return actual is not None and float(actual) < float(expected)
    if op == "ge":
        return actual is not None and float(actual) >= float(expected)
    if op == "gt":
        return actual is not None and float(actual) > float(expected)
    if op == "truthy":
        return bool(actual)
    raise ConfigError(f"未知门禁操作：{op}")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{name} 必须是映射")
    return value

