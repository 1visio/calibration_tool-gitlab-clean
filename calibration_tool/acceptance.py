from __future__ import annotations

import csv
import base64
import html
import io
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .bundle import build_calibration_bundle
from .errors import CalibrationToolError, ConfigError
from .golden import check_golden_baseline
from .io_utils import dump_yaml, load_document, resolve_relative, sha256_file
from .profiles import load_runtime_profile


REPORT_FILES = ("acceptance_report.yaml", "acceptance_report.html", "acceptance_metrics.csv")


def build_acceptance_report(
    plan_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    plan_file = Path(plan_path).expanduser().resolve()
    plan = load_document(plan_file)
    if plan.get("schema_version") != 1:
        raise ConfigError("acceptance plan schema_version 必须为 1")
    policy_value = plan.get("policy")
    if not isinstance(policy_value, str) or not policy_value:
        raise ConfigError("acceptance plan.policy 必须是路径")
    policy_file = resolve_relative(plan_file, policy_value)
    policy = load_document(policy_file)
    if policy.get("schema_version") != 1:
        raise ConfigError("acceptance policy schema_version 必须为 1")
    target_value = output_dir or plan.get("output_dir")
    if not target_value:
        raise ConfigError("acceptance plan 缺少 output_dir")
    target = (
        Path(target_value).expanduser().resolve()
        if output_dir is not None
        else resolve_relative(plan_file, str(target_value))
    )
    existing = [target / name for name in REPORT_FILES if (target / name).exists()]
    if existing and not overwrite:
        raise CalibrationToolError(f"验收报告已存在；确认后使用 --overwrite：{existing[0]}")
    target.mkdir(parents=True, exist_ok=True)

    gates: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    inputs = _mapping(plan.get("inputs", {}), "inputs")
    required = _mapping(policy.get("required", {}), "required")
    expected_extractor = inputs.get("expected_extractor")
    provenance: dict[str, Any] = {
        "plan": _artifact(plan_file, "acceptance_plan"),
        "policy": _artifact(policy_file, "acceptance_policy"),
    }
    artifacts.extend((provenance["plan"], provenance["policy"]))

    workflow = _inspect_workflow(
        plan_file,
        inputs.get("workflow_report"),
        required=bool(required.get("workflow_completed", True)),
        gates=gates,
        artifacts=artifacts,
    )
    quality_reports = _inspect_quality_reports(
        plan_file,
        inputs.get("quality_reports", []),
        required=bool(required.get("quality_report", True)),
        gates=gates,
        artifacts=artifacts,
    )
    runtime_profile = _inspect_runtime_profile(
        plan_file,
        inputs.get("runtime_config"),
        expected_extractor=str(expected_extractor) if expected_extractor else None,
        required=bool(required.get("runtime_profile_clean", True)),
        gates=gates,
        artifacts=artifacts,
    )
    golden = _inspect_golden(
        plan_file,
        inputs.get("golden_baseline"),
        required=bool(required.get("golden_match", True)),
        gates=gates,
        artifacts=artifacts,
    )
    compensation = _inspect_compensation(
        plan_file,
        inputs.get("compensation_metrics"),
        policy=_mapping(policy.get("compensation", {}), "compensation"),
        required=bool(required.get("compensation", True)),
        gates=gates,
        artifacts=artifacts,
    )
    _inspect_explicit_artifacts(
        plan_file,
        inputs.get("artifacts", []),
        required=bool(required.get("artifacts", True)),
        gates=gates,
        artifacts=artifacts,
    )

    artifacts = _deduplicate_artifacts(artifacts)
    counts = {status: sum(item["status"] == status for item in gates) for status in ("pass", "warn", "fail")}
    overall = "fail" if counts["fail"] else ("warn" if counts["warn"] else "pass")
    report = {
        "schema_version": 1,
        "report_id": str(plan.get("report_id", target.name)),
        "title": str(plan.get("title", "线激光标定验收报告")),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "plan": str(plan_file),
        "policy": str(policy_file),
        "overall": overall,
        "decision": "accepted" if overall != "fail" else "rejected",
        "counts": counts,
        "summary": {
            "workflow": None if workflow is None else workflow.get("status"),
            "quality_report_count": len(quality_reports),
            "runtime_extractor": None if runtime_profile is None else runtime_profile.get("selected_extractor", {}).get("method"),
            "runtime_laser_model": None if runtime_profile is None else (runtime_profile.get("laser_model") or {}).get("model_type"),
            "golden_matches": None if golden is None else golden.get("matches"),
            "compensation_independent_validation": None if compensation is None else compensation.get("independent_validation"),
        },
        "workflow": workflow,
        "quality_reports": quality_reports,
        "runtime_profile": runtime_profile,
        "golden": golden,
        "compensation": compensation,
        "gates": gates,
        "artifacts": artifacts,
        "release": {"status": "disabled"},
    }
    yaml_path = target / "acceptance_report.yaml"
    html_path = target / "acceptance_report.html"
    csv_path = target / "acceptance_metrics.csv"
    dump_yaml(yaml_path, report)
    _write_text(html_path, _render_html(report))
    _write_metrics_csv(csv_path, gates)

    release = _mapping(plan.get("release", {}), "release")
    if bool(release.get("enabled", False)):
        if report["decision"] != "accepted":
            report["release"] = {"status": "blocked", "reason": "验收未通过"}
        else:
            config_value = release.get("runtime_config") or inputs.get("runtime_config")
            output_value = release.get("output_dir")
            package_id = release.get("package_id")
            if not all(isinstance(value, str) and value for value in (config_value, output_value, package_id)):
                raise ConfigError("启用 release 时需要 runtime_config、output_dir 和 package_id")
            bundle_output = resolve_relative(plan_file, str(output_value))
            report["release"] = {
                "status": "published",
                "output_dir": str(bundle_output),
                "package_id": str(package_id),
            }
            # The bundle must embed the final report, including the release decision.
            # If publication fails, preserve an explicit failed state in the external report.
            dump_yaml(yaml_path, report)
            _write_text(html_path, _render_html(report))
            try:
                bundle = build_calibration_bundle(
                    resolve_relative(plan_file, str(config_value)),
                    bundle_output,
                    str(package_id),
                    expected_extractor=str(expected_extractor) if expected_extractor else None,
                    quality_report=yaml_path,
                )
            except Exception as exc:
                report["release"] = {
                    "status": "failed",
                    "output_dir": str(bundle_output),
                    "package_id": str(package_id),
                    "reason": str(exc),
                }
                dump_yaml(yaml_path, report)
                _write_text(html_path, _render_html(report))
                raise
            report["release"]["package_id"] = bundle["package_id"]

            # Keep the bundle copy and its manifest hashes synchronized with the
            # final report in case the bundle implementation normalizes package_id.
            for source, name in (
                (yaml_path, "acceptance_report.yaml"),
                (html_path, "acceptance_report.html"),
            ):
                destination = bundle_output / name
                if destination.is_file():
                    shutil.copy2(source, destination)
                    bundle["quality"]["reports"][source.suffix.lstrip(".")] = {
                        "path": name,
                        "sha256": sha256_file(destination),
                    }
            dump_yaml(bundle_output / "calibration_bundle.yaml", bundle)
        dump_yaml(yaml_path, report)
        _write_text(html_path, _render_html(report))
    report["report_files"] = {
        "yaml": str(yaml_path),
        "html": str(html_path),
        "metrics_csv": str(csv_path),
    }
    return report


def _inspect_workflow(
    plan_file: Path,
    value: Any,
    *,
    required: bool,
    gates: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not value:
        _add_gate(gates, "workflow.present", not required, None, "workflow report", "缺少 workflow 报告")
        return None
    path = resolve_relative(plan_file, str(value))
    if not path.is_file():
        _add_gate(gates, "workflow.present", False, str(path), "existing file", "workflow 报告不存在")
        return None
    document = load_document(path)
    artifacts.append(_artifact(path, "workflow_report"))
    _add_gate(
        gates,
        "workflow.completed",
        document.get("status") == "completed",
        document.get("status"),
        "completed",
        "workflow 必须完成",
    )
    for stage in document.get("stages", []):
        if not isinstance(stage, Mapping):
            continue
        name = str(stage.get("stage", "unknown"))
        _add_gate(
            gates,
            f"stage.{name}.status",
            stage.get("status") == "completed",
            stage.get("status"),
            "completed",
            "标定阶段必须完成并通过质量门禁",
        )
        for gate in stage.get("quality_gates", []):
            if isinstance(gate, Mapping):
                gates.append({**dict(gate), "id": f"stage.{name}.{gate.get('id', 'quality')}"})
    return document


def _inspect_quality_reports(
    plan_file: Path,
    value: Any,
    *,
    required: bool,
    gates: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    values = value if isinstance(value, list) else ([] if value in (None, "") else [value])
    if not values:
        _add_gate(gates, "quality_reports.present", not required, 0, ">= 1", "缺少质量报告")
        return []
    reports: list[dict[str, Any]] = []
    for index, item in enumerate(values, start=1):
        path = resolve_relative(plan_file, str(item))
        if not path.is_file():
            _add_gate(gates, f"quality_report.{index}.present", False, str(path), "existing file", "质量报告不存在")
            continue
        document = load_document(path)
        reports.append({"path": str(path), "overall": document.get("overall"), "counts": document.get("counts")})
        artifacts.append(_artifact(path, "quality_report"))
        status = str(document.get("overall", "fail"))
        _add_gate(
            gates,
            f"quality_report.{index}.overall",
            status != "fail",
            status,
            "pass or warn",
            "上游质量报告不能为 fail",
            failure_status="warn" if status == "warn" else "fail",
        )
        for raw in document.get("gates", []):
            if isinstance(raw, Mapping):
                gate = dict(raw)
                gate["id"] = f"quality_report.{index}.{gate.get('id', 'gate')}"
                gates.append(gate)
    return reports


def _inspect_runtime_profile(
    plan_file: Path,
    value: Any,
    *,
    expected_extractor: str | None,
    required: bool,
    gates: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not value:
        _add_gate(gates, "runtime.present", not required, None, "runtime config", "缺少运行配置")
        return None
    path = resolve_relative(plan_file, str(value))
    if not path.is_file():
        _add_gate(gates, "runtime.present", False, str(path), "existing file", "运行配置不存在")
        return None
    profile = load_runtime_profile(path, expected_extractor=expected_extractor)
    artifacts.append(_artifact(path, "runtime_config"))
    findings = profile.get("findings", [])
    _add_gate(
        gates,
        "runtime.profile_clean",
        not any(item.get("severity") == "fail" for item in findings),
        len([item for item in findings if item.get("severity") == "fail"]),
        0,
        "运行配置、manifest 和算法 profile 必须一致",
    )
    for index, finding in enumerate(findings, start=1):
        status = "fail" if finding.get("severity") == "fail" else "warn"
        gates.append({
            "id": f"runtime.finding.{index}.{finding.get('code', 'unknown')}",
            "status": status,
            "actual": None,
            "expected": None,
            "message": finding.get("message", ""),
        })
    for name, record in profile.get("calibration_files", {}).items():
        if record.get("exists"):
            artifacts.append(_artifact(Path(record["path"]), f"calibration_{name}"))
    manifest = profile.get("manifest")
    if isinstance(manifest, Mapping) and manifest.get("exists"):
        artifacts.append(_artifact(Path(str(manifest["path"])), "calibration_manifest"))
    return profile


def _inspect_golden(
    plan_file: Path,
    value: Any,
    *,
    required: bool,
    gates: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not value:
        _add_gate(gates, "golden.present", not required, None, "golden baseline", "缺少 golden baseline")
        return None
    path = resolve_relative(plan_file, str(value))
    if not path.is_file():
        _add_gate(gates, "golden.present", False, str(path), "existing file", "golden baseline 不存在")
        return None
    result = check_golden_baseline(path)
    artifacts.append(_artifact(path, "golden_baseline"))
    _add_gate(
        gates,
        "golden.matches",
        bool(result.get("matches")),
        result.get("change_count"),
        0,
        "当前配置和标定产物不能偏离 golden baseline",
    )
    return result


def _inspect_compensation(
    plan_file: Path,
    value: Any,
    *,
    policy: Mapping[str, Any],
    required: bool,
    gates: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not value:
        _add_gate(gates, "compensation.present", not required, None, "compensation metrics", "缺少补偿指标")
        return None
    path = resolve_relative(plan_file, str(value))
    if not path.is_file():
        _add_gate(gates, "compensation.present", False, str(path), "existing file", "补偿指标不存在")
        return None
    document = load_document(path)
    artifacts.append(_artifact(path, "compensation_metrics"))
    for name in (
        "ground_bias_table.csv",
        "ground_bias_table.npy",
        "z_profile_before_after.png",
        "pointcloud_before_after.png",
    ):
        sibling = path.parent / name
        if sibling.is_file():
            artifacts.append(_artifact(sibling, "compensation_artifact"))
    loaded = int(document.get("loaded_frame_count", 0))
    build = int(document.get("build_frame_count", 0))
    evaluation = int(document.get("evaluation_frame_count", 0))
    independent = evaluation > 0 and build + evaluation == loaded
    _add_gate(
        gates,
        "compensation.independent_validation",
        independent,
        {"loaded": loaded, "build": build, "evaluation": evaluation},
        "evaluation > 0 and build + evaluation == loaded",
        "补偿必须使用未参与建表的 holdout 帧验收",
    )
    metrics = _mapping(document.get("metrics", {}), "compensation.metrics")
    before_pv = _number(metrics.get("profile_before_pv_mm"))
    after_pv = _number(metrics.get("profile_after_pv_mm"))
    before_rms = _number(metrics.get("profile_before_rms_mm"))
    after_rms = _number(metrics.get("profile_after_rms_mm"))
    pv_reduction = _reduction(before_pv, after_pv)
    rms_reduction = _reduction(before_rms, after_rms)
    _metric_le(gates, "compensation.after_pv", after_pv, policy.get("max_profile_after_pv_mm"), "补偿后平均剖面 P–V")
    _metric_le(gates, "compensation.after_rms", after_rms, policy.get("max_profile_after_rms_mm"), "补偿后平均剖面 RMS")
    _metric_ge(gates, "compensation.pv_reduction", pv_reduction, policy.get("min_pv_reduction_fraction"), "P–V 降幅")
    _metric_ge(gates, "compensation.rms_reduction", rms_reduction, policy.get("min_rms_reduction_fraction"), "RMS 降幅")
    repeatability = _number(document.get("repeatability_sigma_median_mm"))
    _metric_le(gates, "compensation.repeatability", repeatability, policy.get("max_repeatability_sigma_mm"), "重复性 sigma 中位数")
    return {
        "path": str(path),
        "loaded_frame_count": loaded,
        "build_frame_count": build,
        "evaluation_frame_count": evaluation,
        "independent_validation": independent,
        "before": {"profile_pv_mm": before_pv, "profile_rms_mm": before_rms},
        "after": {"profile_pv_mm": after_pv, "profile_rms_mm": after_rms},
        "reduction": {"pv_fraction": pv_reduction, "rms_fraction": rms_reduction},
        "repeatability_sigma_median_mm": repeatability,
    }


def _inspect_explicit_artifacts(
    plan_file: Path,
    value: Any,
    *,
    required: bool,
    gates: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
) -> None:
    values = value if isinstance(value, list) else ([] if value in (None, "") else [value])
    if required and not values:
        _add_gate(gates, "artifacts.present", False, 0, ">= 1", "缺少显式验收产物")
    for index, item in enumerate(values, start=1):
        path = resolve_relative(plan_file, str(item))
        exists = path.is_file()
        _add_gate(gates, f"artifact.{index}.exists", exists, str(path), "existing file", "验收产物必须存在")
        if exists:
            artifacts.append(_artifact(path, "explicit"))


def _artifact(path: Path, role: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    return {
        "role": role,
        "path": str(resolved),
        "exists": resolved.is_file(),
        "size_bytes": resolved.stat().st_size if resolved.is_file() else None,
        "sha256": sha256_file(resolved, normalize_newlines=False) if resolved.is_file() else None,
    }


def _deduplicate_artifacts(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for item in items:
        unique.setdefault(str(item["path"]).casefold(), item)
    return sorted(unique.values(), key=lambda item: (str(item["role"]), str(item["path"])))


def _add_gate(
    gates: list[dict[str, Any]],
    gate_id: str,
    passed: bool,
    actual: Any,
    expected: Any,
    message: str,
    *,
    failure_status: str = "fail",
) -> None:
    gates.append({
        "id": gate_id,
        "status": "pass" if passed else failure_status,
        "actual": actual,
        "expected": expected,
        "message": message,
    })


def _metric_le(gates: list[dict[str, Any]], gate_id: str, actual: float | None, limit: Any, message: str) -> None:
    if limit is None:
        return
    _add_gate(gates, gate_id, actual is not None and actual <= float(limit), actual, f"<= {float(limit):g}", message)


def _metric_ge(gates: list[dict[str, Any]], gate_id: str, actual: float | None, limit: Any, message: str) -> None:
    if limit is None:
        return
    _add_gate(gates, gate_id, actual is not None and actual >= float(limit), actual, f">= {float(limit):g}", message)


def _number(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _reduction(before: float | None, after: float | None) -> float | None:
    if before is None or after is None or before <= 0:
        return None
    return 1.0 - after / before


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{name} 必须是映射")
    return value


def _write_metrics_csv(path: Path, gates: list[dict[str, Any]]) -> None:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=("id", "status", "actual", "expected", "message"))
    writer.writeheader()
    for gate in gates:
        writer.writerow({name: gate.get(name) for name in writer.fieldnames})
    _write_text(path, stream.getvalue())


def _write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _render_html(report: Mapping[str, Any]) -> str:
    status = str(report["overall"])
    compensation = report.get("compensation") or {}
    gates = report.get("gates", [])
    artifacts = report.get("artifacts", [])
    gate_rows = "".join(
        "<tr>"
        f"<td><code>{html.escape(str(gate.get('id', '')))}</code></td>"
        f"<td><span class='badge {html.escape(str(gate.get('status', '')))}'>{html.escape(str(gate.get('status', '')))}</span></td>"
        f"<td>{html.escape(str(gate.get('actual', '')))}</td>"
        f"<td>{html.escape(str(gate.get('expected', '')))}</td>"
        f"<td>{html.escape(str(gate.get('message', '')))}</td>"
        "</tr>"
        for gate in gates
    )
    artifact_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(item.get('role', '')))}</td>"
        f"<td class='path'>{html.escape(str(item.get('path', '')))}</td>"
        f"<td>{html.escape(str(item.get('size_bytes', '')))}</td>"
        f"<td><code>{html.escape(str(item.get('sha256', '')))}</code></td>"
        "</tr>"
        for item in artifacts
    )
    compensation_html = "<p>未提供补偿指标。</p>"
    if compensation:
        before = compensation.get("before", {})
        after = compensation.get("after", {})
        reduction = compensation.get("reduction", {})
        compensation_html = f"""
        <div class="metrics">
          {_metric_card('补偿前 P–V', before.get('profile_pv_mm'), 'mm')}
          {_metric_card('补偿后 P–V', after.get('profile_pv_mm'), 'mm')}
          {_metric_card('P–V 降幅', _percent(reduction.get('pv_fraction')), '')}
          {_metric_card('补偿前 RMS', before.get('profile_rms_mm'), 'mm')}
          {_metric_card('补偿后 RMS', after.get('profile_rms_mm'), 'mm')}
          {_metric_card('RMS 降幅', _percent(reduction.get('rms_fraction')), '')}
        </div>
        <p>独立 holdout：<strong>{'是' if compensation.get('independent_validation') else '否'}</strong>；
        建表 {compensation.get('build_frame_count')} 帧，验收 {compensation.get('evaluation_frame_count')} 帧。</p>
        {_comparison_bar('P–V', before.get('profile_pv_mm'), after.get('profile_pv_mm'))}
        {_comparison_bar('RMS', before.get('profile_rms_mm'), after.get('profile_rms_mm'))}
        {_embedded_compensation_images(artifacts)}
        """
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>{html.escape(str(report['title']))}</title>
<style>
body{{font-family:"Segoe UI","Microsoft YaHei",sans-serif;margin:0;background:#f3f5f7;color:#18212b}}
header{{background:#17212b;color:white;padding:28px 5vw}} main{{max-width:1280px;margin:24px auto;padding:0 24px}}
.summary{{display:flex;gap:18px;align-items:center;background:white;padding:20px;border-radius:8px;border:1px solid #dce1e6}}
.badge{{padding:4px 10px;border-radius:99px;font-weight:700;text-transform:uppercase}} .pass{{background:#d8f3df;color:#176b35}}
.warn{{background:#fff1c7;color:#825b00}} .fail{{background:#ffd9d9;color:#9c2020}}
section{{background:white;margin:18px 0;padding:20px;border:1px solid #dce1e6;border-radius:8px}}
.metrics{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}} .metric{{background:#f6f8fa;padding:14px;border-radius:6px}}
.metric b{{display:block;font-size:22px;margin-top:6px}} table{{width:100%;border-collapse:collapse;font-size:13px}}
.comparison{{display:grid;grid-template-columns:90px 1fr 90px;gap:10px;align-items:center;margin:10px 0}}
.track{{height:18px;background:#edf0f3;border-radius:4px;overflow:hidden}} .bar{{height:100%;background:#d65b5b}}
.bar.after{{background:#2c9b61}} .figures{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:18px}}
.figures figure{{margin:0;border:1px solid #e0e4e8;padding:8px}} .figures img{{max-width:100%;height:auto}}
th,td{{border-bottom:1px solid #e5e8eb;padding:8px;text-align:left;vertical-align:top}} th{{background:#f6f8fa}}
code,.path{{font-family:Consolas,monospace;word-break:break-all}} footer{{color:#667085;padding:24px 0}}
</style></head><body>
<header><h1>{html.escape(str(report['title']))}</h1><div>{html.escape(str(report['report_id']))}</div></header><main>
<div class="summary"><span class="badge {status}">{status}</span><div><b>验收结论：{html.escape(str(report['decision']))}</b><br>
PASS {report['counts']['pass']} · WARN {report['counts']['warn']} · FAIL {report['counts']['fail']}</div></div>
<section><h2>补偿前后对比</h2>{compensation_html}</section>
<section><h2>验收门禁</h2><table><thead><tr><th>ID</th><th>状态</th><th>实际</th><th>要求</th><th>说明</th></tr></thead><tbody>{gate_rows}</tbody></table></section>
<section><h2>产物与哈希</h2><table><thead><tr><th>角色</th><th>路径</th><th>字节</th><th>SHA-256</th></tr></thead><tbody>{artifact_rows}</tbody></table></section>
<section><h2>发布状态</h2><pre>{html.escape(str(report.get('release')))}</pre></section>
<footer>生成时间：{html.escape(str(report['generated_utc']))}</footer></main></body></html>"""


def _metric_card(label: str, value: Any, suffix: str) -> str:
    text = "--" if value is None else (value if isinstance(value, str) else f"{float(value):.6g}")
    return f"<div class='metric'>{html.escape(label)}<b>{html.escape(str(text))} {html.escape(suffix)}</b></div>"


def _percent(value: Any) -> str:
    return "--" if value is None else f"{float(value) * 100:.2f}%"


def _comparison_bar(label: str, before: Any, after: Any) -> str:
    if before is None or after is None or float(before) <= 0:
        return ""
    ratio = max(0.0, min(1.0, float(after) / float(before)))
    return f"""
    <div class='comparison'><b>{html.escape(label)} 前</b><div class='track'><div class='bar' style='width:100%'></div></div><span>{float(before):.6g} mm</span></div>
    <div class='comparison'><b>{html.escape(label)} 后</b><div class='track'><div class='bar after' style='width:{ratio * 100:.3f}%'></div></div><span>{float(after):.6g} mm</span></div>
    """


def _embedded_compensation_images(artifacts: Any) -> str:
    figures: list[str] = []
    captions = {
        "z_profile_before_after.png": "补偿前后 Z 剖面与残差",
        "pointcloud_before_after.png": "补偿前后点云",
    }
    for item in artifacts:
        path = Path(str(item.get("path", ""))) if isinstance(item, Mapping) else Path()
        if path.name not in captions or not path.is_file():
            continue
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        figures.append(
            f"<figure><img src='data:image/png;base64,{encoded}' alt='{html.escape(captions[path.name])}'>"
            f"<figcaption>{html.escape(captions[path.name])}</figcaption></figure>"
        )
    return "" if not figures else f"<div class='figures'>{''.join(figures)}</div>"
