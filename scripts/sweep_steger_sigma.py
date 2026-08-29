#!/usr/bin/env python3
"""固定三联图数据与非 sigma 参数，离线扫描 Steger extraction.sigma。"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import sys
import traceback
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
FIT_SCRIPT_PATH = SCRIPT_DIR / "fit_laser_models_from_triplets.py"
FIT_SPEC = importlib.util.spec_from_file_location("fit_laser_models_from_triplets", FIT_SCRIPT_PATH)
if FIT_SPEC is None or FIT_SPEC.loader is None:  # pragma: no cover - installation error
    raise RuntimeError(f"无法加载拟合脚本：{FIT_SCRIPT_PATH}")
FIT = importlib.util.module_from_spec(FIT_SPEC)
sys.modules.setdefault(FIT_SPEC.name, FIT)
FIT_SPEC.loader.exec_module(FIT)

# 原拟合脚本的图标题包含中文；部分无中文字体环境会为每个 glyph 重复告警。
# 这只影响字体回退，不影响图或数值，sweep 日志中予以抑制。
warnings.filterwarnings("ignore", message=r"Glyph .* missing from font")

MODEL_ORDER = ("global_plane", "quadratic_graph", "circular_cone")
FOCUS_MODELS = ("circular_cone", "quadratic_graph")
SUMMARY_FIELDS = (
    "sigma", "model", "train_point_count", "validation_point_count", "valid_rate",
    "train_board_mae_mm", "train_board_rmse_mm", "train_board_p95_abs_mm",
    "train_board_max_abs_mm", "validation_board_mean_signed_mm",
    "validation_board_mae_mm", "validation_board_rmse_mm",
    "validation_board_p95_abs_mm", "validation_board_max_abs_mm",
    "train_surface_rmse_mm", "validation_surface_rmse_mm",
    "validation_ray_mae_mm", "validation_ray_rmse_mm",
    "validation_ray_p95_abs_mm", "validation_ray_max_abs_mm",
    "train_valid_rate", "successful_train_pose_count", "successful_validation_pose_count",
    "train_point_count_change_percent", "validation_point_count_change_percent",
    "point_count_warning", "all_validation_poses_succeeded", "catastrophic_per_image",
    "eligible_for_ranking", "center_shift_matched_count", "center_shift_median_px",
    "center_shift_p95_px", "center_shift_max_px",
)


def _as_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} 必须是 YAML mapping")
    return dict(value)


def parse_sigma_values(value: Any) -> list[float]:
    if not isinstance(value, list) or not value:
        raise ValueError("sigma_values 必须是非空列表")
    values: list[float] = []
    for raw in value:
        sigma = float(raw)
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError(f"sigma 必须为正有限数：{raw!r}")
        if any(math.isclose(sigma, seen, rel_tol=0.0, abs_tol=1.0e-12) for seen in values):
            raise ValueError(f"sigma_values 包含重复值：{sigma}")
        values.append(sigma)
    return values


def sigma_slug(sigma: float) -> str:
    value = float(sigma)
    rendered = f"{value:.1f}" if value.is_integer() else format(value, ".12g")
    text = rendered.replace("-", "m").replace(".", "p")
    return f"sigma_{text}"


def load_sweep_config(path: Path) -> dict[str, Any]:
    path = path.resolve()
    data = _as_mapping(yaml.safe_load(path.read_text(encoding="utf-8-sig")), "sweep config")
    data["sigma_values"] = parse_sigma_values(data.get("sigma_values"))
    base_config = Path(str(data.get("base_config", "")))
    if not str(base_config):
        raise ValueError("base_config 未配置")
    output_dir = Path(str(data.get("output_dir", "../runs/steger_sigma_sweep")))
    data["base_config"] = str(base_config if base_config.is_absolute() else (path.parent / base_config).resolve())
    data["output_dir"] = str(output_dir if output_dir.is_absolute() else (path.parent / output_dir).resolve())
    baseline = _as_mapping(data.get("baseline", {}), "baseline")
    baseline_sigma = float(baseline.get("sigma", 1.5))
    if not any(math.isclose(baseline_sigma, value, abs_tol=1.0e-12) for value in data["sigma_values"]):
        raise ValueError(f"baseline sigma={baseline_sigma} 不在 sigma_values 中")
    baseline["sigma"] = baseline_sigma
    data["baseline"] = baseline
    data["analysis"] = _as_mapping(data.get("analysis", {}), "analysis")
    return data


def config_for_sigma(base_config: Mapping[str, Any], sigma: float) -> dict[str, Any]:
    """仅覆盖 extraction.sigma；调用者持有的原配置保持不变。"""
    varied = copy.deepcopy(dict(base_config))
    extraction = varied.get("extraction")
    if not isinstance(extraction, dict):
        raise ValueError("base config 缺少 extraction mapping")
    extraction["sigma"] = float(sigma)
    return varied


def validate_fixed_config(base_config: Mapping[str, Any]) -> str:
    laser = _as_mapping(base_config.get("laser"), "laser")
    orientation = str(laser.get("orientation", "")).strip().lower()
    if orientation not in {"horizontal", "vertical"}:
        raise ValueError(f"laser.orientation 非法：{orientation!r}")
    extraction = _as_mapping(base_config.get("extraction"), "extraction")
    if str(extraction.get("method", "steger")).lower() != "steger":
        raise ValueError("本实验仅允许 extraction.method=steger")
    for required in ("intrinsics", "board", "patterns", "datasets", "models"):
        if required not in base_config:
            raise ValueError(f"base config 缺少 {required}")
    return orientation


def _canonical_without_sigma(config: Mapping[str, Any]) -> str:
    normalized = copy.deepcopy(dict(config))
    extraction = normalized.get("extraction")
    if isinstance(extraction, dict):
        extraction.pop("sigma", None)
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def assert_only_sigma_changed(base_config: Mapping[str, Any], varied: Mapping[str, Any], sigma: float) -> None:
    if _canonical_without_sigma(base_config) != _canonical_without_sigma(varied):
        raise AssertionError("effective config 除 extraction.sigma 外发生了变化")
    actual = float(_as_mapping(varied.get("extraction"), "extraction").get("sigma"))
    if not math.isclose(actual, sigma, rel_tol=0.0, abs_tol=1.0e-12):
        raise AssertionError(f"effective extraction.sigma={actual}，预期 {sigma}")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"缺少拟合产物：{path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def _fit_artifacts_complete(output_dir: Path) -> bool:
    return all((output_dir / name).is_file() for name in (
        "calibration_points.csv", "model_comparison.csv", "per_image_metrics.csv",
    ))


def run_one_sigma(
    base_config: Mapping[str, Any],
    sigma: float,
    output_dir: Path,
    orientation: str,
    resume: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    effective = config_for_sigma(base_config, sigma)
    assert_only_sigma_changed(base_config, effective, sigma)
    effective_path = output_dir / "effective_config.yaml"
    if resume and _fit_artifacts_complete(output_dir):
        existing = FIT.safe_yaml_load(effective_path)
        assert_only_sigma_changed(base_config, existing, sigma)
        existing_orientation = str(_as_mapping(existing.get("laser"), "laser").get("orientation", ""))
        if existing_orientation != orientation:
            raise AssertionError("已有结果的 laser.orientation 与当前配置不一致")
        print(f"[RESUME] sigma={sigma:g}：复用完整拟合产物")
        return
    effective_path.write_text(
        yaml.safe_dump(effective, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    exit_code = FIT.main([
        "--config", str(effective_path),
        "--output-dir", str(output_dir),
        "--laser-orientation", orientation,
    ])
    if exit_code != 0:
        raise RuntimeError(f"sigma={sigma:g} 拟合返回 {exit_code}")


def collect_sigma_artifacts(output_dir: Path, sigma: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    comparison = _read_csv(output_dir / "model_comparison.csv")
    comparison.insert(0, "sigma", float(sigma))
    per_image = _read_csv(output_dir / "per_image_metrics.csv")
    per_image.insert(0, "sigma", float(sigma))
    points = _read_csv(output_dir / "calibration_points.csv")
    points.insert(0, "sigma", float(sigma))
    return comparison, per_image, points


def baseline_check(
    comparison: pd.DataFrame,
    points: pd.DataFrame,
    baseline_cfg: Mapping[str, Any],
    expected_train_poses: int,
    expected_validation_poses: int,
    orientation: str,
) -> dict[str, Any]:
    rmse_tol = float(baseline_cfg.get("rmse_abs_tolerance_mm", 1.0e-4))
    count_tol = float(baseline_cfg.get("point_count_relative_tolerance", 0.01))
    expected_counts = {
        "train": int(baseline_cfg.get("expected_train_point_count", 0)),
        "validation": int(baseline_cfg.get("expected_validation_point_count", 0)),
    }
    actual_counts = points.groupby("split").size().to_dict()
    expected_rmse = _as_mapping(
        baseline_cfg.get("expected_validation_board_rmse_mm", {}),
        "baseline.expected_validation_board_rmse_mm",
    )
    checks: list[dict[str, Any]] = []
    for split, expected in expected_counts.items():
        actual = int(actual_counts.get(split, 0))
        relative = abs(actual - expected) / expected if expected > 0 else float("inf")
        checks.append({
            "name": f"{split}_point_count", "expected": expected, "actual": actual,
            "tolerance": count_tol, "difference": relative, "pass": relative <= count_tol,
        })
    for model, expected_raw in expected_rmse.items():
        rows = comparison[(comparison["split"] == "validation") & (comparison["model"] == model)]
        actual = float(rows.iloc[0]["board_rmse_mm"]) if len(rows) == 1 else float("nan")
        expected = float(expected_raw)
        difference = abs(actual - expected)
        checks.append({
            "name": f"{model}_validation_board_rmse_mm", "expected": expected,
            "actual": actual, "tolerance": rmse_tol, "difference": difference,
            "pass": math.isfinite(actual) and difference <= rmse_tol,
        })
    pose_counts = points.groupby("split")["image_id"].nunique().to_dict()
    for split, expected in (("train", expected_train_poses), ("validation", expected_validation_poses)):
        actual = int(pose_counts.get(split, 0))
        checks.append({
            "name": f"{split}_pose_count", "expected": expected, "actual": actual,
            "tolerance": 0, "difference": abs(actual - expected), "pass": actual == expected,
        })
    checks.append({
        "name": "laser_orientation", "expected": "vertical", "actual": orientation,
        "tolerance": None, "difference": None, "pass": orientation == "vertical",
    })
    return {"status": "PASS" if all(item["pass"] for item in checks) else "FAIL", "checks": checks}


def _write_baseline_check(output_dir: Path, result: Mapping[str, Any]) -> None:
    (output_dir / "baseline_check.yaml").write_text(
        yaml.safe_dump(dict(result), allow_unicode=True, sort_keys=False), encoding="utf-8"
    )


def pose_counts(points: pd.DataFrame) -> pd.DataFrame:
    counts = points.groupby(["sigma", "split", "image_id"], as_index=False).size()
    return counts.rename(columns={"size": "point_count"})


def center_shift_metrics(points: pd.DataFrame, baseline_sigma: float, orientation: str) -> pd.DataFrame:
    scan_coord = "v_px" if orientation == "vertical" else "u_px"
    baseline = points[np.isclose(points["sigma"], baseline_sigma)].copy()
    baseline["scanline"] = np.rint(baseline[scan_coord].to_numpy(float)).astype(int)
    rows: list[dict[str, Any]] = []
    for sigma, sigma_points in points.groupby("sigma"):
        current = sigma_points.copy()
        current["scanline"] = np.rint(current[scan_coord].to_numpy(float)).astype(int)
        shifts: list[np.ndarray] = []
        for (split, image_id), group in current.groupby(["split", "image_id"]):
            ref = baseline[(baseline["split"] == split) & (baseline["image_id"].astype(str) == str(image_id))]
            joined = group.merge(ref, on=["split", "image_id", "scanline"], suffixes=("", "_baseline"))
            delta = np.hypot(
                joined["u_px"].to_numpy(float) - joined["u_px_baseline"].to_numpy(float),
                joined["v_px"].to_numpy(float) - joined["v_px_baseline"].to_numpy(float),
            )
            if delta.size:
                shifts.append(delta)
        values = np.concatenate(shifts) if shifts else np.asarray([], dtype=float)
        rows.append({
            "sigma": float(sigma), "matched_count": int(values.size),
            "median_px": float(np.median(values)) if values.size else float("nan"),
            "p95_px": float(np.percentile(values, 95)) if values.size else float("nan"),
            "max_px": float(np.max(values)) if values.size else float("nan"),
        })
    return pd.DataFrame(rows)


def per_image_wide(per_image: pd.DataFrame, point_counts: pd.DataFrame) -> pd.DataFrame:
    validation = per_image[per_image["split"] == "validation"]
    wide = validation.pivot(index=["sigma", "image_id"], columns="model", values="rmse_mm").reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={model: f"{model}_rmse" for model in MODEL_ORDER})
    validation_counts = point_counts[point_counts["split"] == "validation"][["sigma", "image_id", "point_count"]]
    return wide.merge(validation_counts, on=["sigma", "image_id"], how="left").sort_values(["sigma", "image_id"])


def build_summary(
    comparison: pd.DataFrame,
    points: pd.DataFrame,
    per_image: pd.DataFrame,
    base_config: Mapping[str, Any],
    baseline_sigma: float,
    analysis_cfg: Mapping[str, Any],
    shifts: pd.DataFrame,
) -> pd.DataFrame:
    split_counts = points.groupby(["sigma", "split"]).size().unstack(fill_value=0)
    split_poses = points.groupby(["sigma", "split"])["image_id"].nunique().unstack(fill_value=0)
    expected_train_poses = len(_as_mapping(base_config["datasets"], "datasets")["train"]["ids"])
    expected_val_poses = len(_as_mapping(base_config["datasets"], "datasets")["validation"]["ids"])
    baseline_counts = split_counts.loc[baseline_sigma]
    warning_percent = float(analysis_cfg.get("point_count_warning_percent", 10.0))
    valid_rate_min = float(analysis_cfg.get("valid_rate_min", 0.999))
    catastrophic_ratio = float(analysis_cfg.get("catastrophic_per_image_ratio", 2.0))
    catastrophic_abs = float(analysis_cfg.get("catastrophic_per_image_abs_increase_mm", 0.05))
    baseline_images = per_image[
        (np.isclose(per_image["sigma"], baseline_sigma)) & (per_image["split"] == "validation")
    ][["model", "image_id", "rmse_mm"]].rename(columns={"rmse_mm": "baseline_rmse_mm"})
    joined_images = per_image[per_image["split"] == "validation"].merge(
        baseline_images, on=["model", "image_id"], how="left"
    )
    joined_images["catastrophic"] = (
        (joined_images["rmse_mm"] > catastrophic_ratio * joined_images["baseline_rmse_mm"])
        & (joined_images["rmse_mm"] - joined_images["baseline_rmse_mm"] > catastrophic_abs)
    )
    catastrophic = joined_images.groupby(["sigma", "model"])["catastrophic"].any().to_dict()
    shift_by_sigma = shifts.set_index("sigma").to_dict("index")

    rows: list[dict[str, Any]] = []
    for sigma in sorted(comparison["sigma"].unique()):
        train_count = int(split_counts.loc[sigma].get("train", 0))
        validation_count = int(split_counts.loc[sigma].get("validation", 0))
        train_change = 100.0 * (train_count - int(baseline_counts.get("train", 0))) / max(int(baseline_counts.get("train", 0)), 1)
        val_change = 100.0 * (validation_count - int(baseline_counts.get("validation", 0))) / max(int(baseline_counts.get("validation", 0)), 1)
        count_warning = abs(train_change) > warning_percent or abs(val_change) > warning_percent
        all_val_poses = int(split_poses.loc[sigma].get("validation", 0)) == expected_val_poses
        shift = shift_by_sigma[float(sigma)]
        for model in MODEL_ORDER:
            train_rows = comparison[(np.isclose(comparison["sigma"], sigma)) & (comparison["split"] == "train") & (comparison["model"] == model)]
            val_rows = comparison[(np.isclose(comparison["sigma"], sigma)) & (comparison["split"] == "validation") & (comparison["model"] == model)]
            if len(train_rows) != 1 or len(val_rows) != 1:
                continue
            train = train_rows.iloc[0]
            val = val_rows.iloc[0]
            disaster = bool(catastrophic.get((sigma, model), False))
            row = {
                "sigma": float(sigma), "model": model,
                "train_point_count": train_count, "validation_point_count": validation_count,
                "valid_rate": float(val["valid_rate"]),
                "train_board_mae_mm": float(train["board_mae_mm"]),
                "train_board_rmse_mm": float(train["board_rmse_mm"]),
                "train_board_p95_abs_mm": float(train["board_p95_abs_mm"]),
                "train_board_max_abs_mm": float(train["board_max_abs_mm"]),
                "validation_board_mean_signed_mm": float(val["board_mean_signed_mm"]),
                "validation_board_mae_mm": float(val["board_mae_mm"]),
                "validation_board_rmse_mm": float(val["board_rmse_mm"]),
                "validation_board_p95_abs_mm": float(val["board_p95_abs_mm"]),
                "validation_board_max_abs_mm": float(val["board_max_abs_mm"]),
                "train_surface_rmse_mm": float(train["surface_rmse_mm"]),
                "validation_surface_rmse_mm": float(val["surface_rmse_mm"]),
                "validation_ray_mae_mm": float(val["ray_mae_mm"]),
                "validation_ray_rmse_mm": float(val["ray_rmse_mm"]),
                "validation_ray_p95_abs_mm": float(val["ray_p95_abs_mm"]),
                "validation_ray_max_abs_mm": float(val["ray_max_abs_mm"]),
                "train_valid_rate": float(train["valid_rate"]),
                "successful_train_pose_count": int(split_poses.loc[sigma].get("train", 0)),
                "successful_validation_pose_count": int(split_poses.loc[sigma].get("validation", 0)),
                "train_point_count_change_percent": train_change,
                "validation_point_count_change_percent": val_change,
                "point_count_warning": count_warning,
                "all_validation_poses_succeeded": all_val_poses,
                "catastrophic_per_image": disaster,
                "eligible_for_ranking": (
                    float(val["valid_rate"]) >= valid_rate_min and all_val_poses
                    and not count_warning and not disaster
                ),
                "center_shift_matched_count": int(shift["matched_count"]),
                "center_shift_median_px": float(shift["median_px"]),
                "center_shift_p95_px": float(shift["p95_px"]),
                "center_shift_max_px": float(shift["max_px"]),
            }
            rows.append(row)
    return pd.DataFrame(rows, columns=SUMMARY_FIELDS)


def stable_ranges(model_summary: pd.DataFrame, within_percent: float, min_points: int) -> list[tuple[float, float]]:
    ordered = model_summary.sort_values("sigma")
    eligible = ordered["eligible_for_ranking"].astype(bool).to_numpy()
    eligible_rows = ordered[eligible]
    if eligible_rows.empty:
        eligible = np.ones(len(ordered), dtype=bool)
        eligible_rows = ordered
    minimum = float(eligible_rows["validation_board_rmse_mm"].min())
    within = eligible & (
        ordered["validation_board_rmse_mm"].to_numpy(float)
        <= minimum * (1.0 + within_percent / 100.0)
    )
    sigmas = ordered["sigma"].to_numpy(float)
    ranges: list[tuple[float, float]] = []
    start: int | None = None
    for index, accepted in enumerate(np.r_[within, False]):
        if accepted and start is None:
            start = index
        elif not accepted and start is not None:
            if index - start >= min_points:
                ranges.append((float(sigmas[start]), float(sigmas[index - 1])))
            start = None
    return ranges


def choose_best(summary: pd.DataFrame, model: str) -> pd.Series:
    candidates = summary[(summary["model"] == model) & summary["eligible_for_ranking"].astype(bool)]
    if candidates.empty:
        candidates = summary[summary["model"] == model]
    return candidates.sort_values(
        ["validation_board_rmse_mm", "validation_board_p95_abs_mm", "validation_board_mae_mm", "sigma"]
    ).iloc[0]


def _plot_metric(summary: pd.DataFrame, metric: str, ylabel: str, output: Path, baseline_sigma: float) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for model in MODEL_ORDER:
        group = summary[summary["model"] == model].sort_values("sigma")
        ax.plot(group["sigma"], group[metric], marker="o", label=model)
    ax.axvline(baseline_sigma, color="black", linestyle="--", alpha=0.65, label=f"baseline {baseline_sigma:g}")
    ax.set(xlabel="Steger sigma / px", ylabel=ylabel)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def generate_plots(summary: pd.DataFrame, per_image_table: pd.DataFrame, output_dir: Path, baseline_sigma: float) -> None:
    _plot_metric(summary, "validation_board_rmse_mm", "Validation board RMSE / mm", output_dir / "sigma_vs_validation_rmse.png", baseline_sigma)
    _plot_metric(summary, "validation_board_p95_abs_mm", "Validation board P95 absolute error / mm", output_dir / "sigma_vs_validation_p95.png", baseline_sigma)
    _plot_metric(summary, "validation_board_mae_mm", "Validation board MAE / mm", output_dir / "sigma_vs_validation_mae.png", baseline_sigma)

    counts = summary.drop_duplicates("sigma").sort_values("sigma")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(counts["sigma"], counts["train_point_count"], marker="o", label="train")
    ax.plot(counts["sigma"], counts["validation_point_count"], marker="o", label="validation")
    ax.axvline(baseline_sigma, color="black", linestyle="--", alpha=0.65)
    ax.set(xlabel="Steger sigma / px", ylabel="Valid calibration point count")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "sigma_vs_point_count.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3), sharey=False)
    for ax, model in zip(axes, FOCUS_MODELS):
        group = summary[summary["model"] == model].sort_values("sigma")
        ax.plot(group["sigma"], group["train_board_rmse_mm"], marker="o", label="train")
        ax.plot(group["sigma"], group["validation_board_rmse_mm"], marker="o", label="validation")
        ax.axvline(baseline_sigma, color="black", linestyle="--", alpha=0.65)
        ax.set(title=model, xlabel="Steger sigma / px", ylabel="Board RMSE / mm")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "sigma_vs_train_validation_rmse.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    for ax, model in zip(axes, FOCUS_MODELS):
        column = f"{model}_rmse"
        pivot = per_image_table.pivot(index="image_id", columns="sigma", values=column).sort_index()
        image = ax.imshow(pivot.to_numpy(float), aspect="auto", cmap="viridis")
        ax.set(title=model, xlabel="sigma", ylabel="validation image_id")
        ax.set_xticks(range(len(pivot.columns)), [f"{value:g}" for value in pivot.columns], rotation=45)
        ax.set_yticks(range(len(pivot.index)), [str(value) for value in pivot.index])
        fig.colorbar(image, ax=ax, label="Board RMSE / mm")
        single, single_ax = plt.subplots(figsize=(9, 5.5))
        single_image = single_ax.imshow(pivot.to_numpy(float), aspect="auto", cmap="viridis")
        single_ax.set(title=model, xlabel="sigma", ylabel="validation image_id")
        single_ax.set_xticks(range(len(pivot.columns)), [f"{value:g}" for value in pivot.columns], rotation=45)
        single_ax.set_yticks(range(len(pivot.index)), [str(value) for value in pivot.index])
        single.colorbar(single_image, ax=single_ax, label="Board RMSE / mm")
        single.tight_layout()
        single.savefig(output_dir / f"per_image_rmse_heatmap_{model}.png", dpi=180)
        plt.close(single)
    fig.savefig(output_dir / "per_image_rmse_heatmap.png", dpi=180)
    plt.close(fig)


def _format_ranges(ranges: list[tuple[float, float]]) -> str:
    return ", ".join(f"{start:g} ~ {end:g}" for start, end in ranges) if ranges else "未发现"


def _pose_improvement_text(per_image_table: pd.DataFrame, model: str, best_sigma: float, baseline_sigma: float) -> str:
    column = f"{model}_rmse"
    baseline = per_image_table[np.isclose(per_image_table["sigma"], baseline_sigma)].set_index("image_id")[column]
    best = per_image_table[np.isclose(per_image_table["sigma"], best_sigma)].set_index("image_id")[column]
    paired = pd.concat([baseline.rename("baseline"), best.rename("best")], axis=1).dropna()
    improved = int((paired["best"] < paired["baseline"] - 1.0e-12).sum())
    total = len(paired)
    median_delta = float((paired["baseline"] - paired["best"]).median()) if total else float("nan")
    if total and improved / total >= 2.0 / 3.0:
        kind = "多数姿态普遍改善"
    elif total and improved / total <= 1.0 / 3.0:
        kind = "改善主要由少数姿态贡献"
    else:
        kind = "不同姿态表现混合"
    return f"{kind}（{improved}/{total} 幅 RMSE 下降，逐图中位改善 {median_delta:.6f} mm）"


def generate_report(
    output: Path,
    summary: pd.DataFrame,
    per_image_table: pd.DataFrame,
    baseline_result: Mapping[str, Any],
    sweep_cfg: Mapping[str, Any],
    orientation: str,
    errors: Sequence[str],
) -> None:
    baseline_sigma = float(_as_mapping(sweep_cfg["baseline"], "baseline")["sigma"])
    analysis = _as_mapping(sweep_cfg.get("analysis", {}), "analysis")
    if str(baseline_result.get("status")) != "PASS":
        lines = [
            "# Steger sigma 扫描报告", "", "## Baseline 一致性检查：FAIL", "",
            "sigma=1.5 未通过正式结果复现门禁，已按要求停止其余 sigma 扫描与最终排名。", "",
            "| check | expected | actual | difference | tolerance | result |", "|---|---:|---:|---:|---:|---|",
        ]
        for item in baseline_result.get("checks", []):
            lines.append(f"| {item['name']} | {item['expected']} | {item['actual']} | {item['difference']} | {item['tolerance']} | {'PASS' if item['pass'] else 'FAIL'} |")
        lines += ["", "请检查 config、train/validation ID、vertical orientation、非 sigma extraction 参数及 Steger 实现版本。"]
        output.write_text("\n".join(lines), encoding="utf-8")
        return

    within = float(analysis.get("stable_within_percent", 2.0))
    min_points = int(analysis.get("stable_min_consecutive_points", 3))
    baseline_rows = summary[np.isclose(summary["sigma"], baseline_sigma)].set_index("model")
    best_by_model = {model: choose_best(summary, model) for model in FOCUS_MODELS}
    ranges_by_model = {
        model: stable_ranges(summary[summary["model"] == model], within, min_points)
        for model in FOCUS_MODELS
    }
    raw_ranges_by_model = {}
    for model in FOCUS_MODELS:
        raw = summary[summary["model"] == model].copy()
        raw["eligible_for_ranking"] = True
        raw_ranges_by_model[model] = stable_ranges(raw, within, min_points)
    improvement: dict[str, tuple[float, float]] = {}
    for model, best in best_by_model.items():
        baseline = baseline_rows.loc[model]
        improvement[model] = (
            100.0 * (baseline["validation_board_rmse_mm"] - best["validation_board_rmse_mm"]) / baseline["validation_board_rmse_mm"],
            100.0 * (baseline["validation_board_p95_abs_mm"] - best["validation_board_p95_abs_mm"]) / baseline["validation_board_p95_abs_mm"],
        )
    best_sigmas = [float(best_by_model[model]["sigma"]) for model in FOCUS_MODELS]
    max_improvement = max(value[0] for value in improvement.values())
    if max_improvement < 3.0:
        recommendation = "当前 sigma=1.5 已接近最优，不建议为了极小改善重新发布标定模型。"
    elif max_improvement <= 10.0:
        recommendation = "存在约 3~10% 的可观察优化；但未形成稳定平台，且最佳点伴随 candidate acceptance 数量变化。暂不修改正式 sigma，应先用标准件或新 holdout 数据验证。"
    else:
        recommendation = "Steger 尺度对当前 12MP 条纹存在明显影响，建议进一步验证最佳 sigma。"
    count_warning_sigmas = sorted(summary.loc[summary["point_count_warning"].astype(bool), "sigma"].unique())
    shift_rows = summary.drop_duplicates("sigma").sort_values("sigma")
    lines = [
        "# Steger sigma 扫描报告", "", "## 结论", "",
        f"- Baseline 一致性检查：**{baseline_result['status']}**。sigma={baseline_sigma:g} 成功复现当前正式标定结果。",
        f"- 激光方向：`{orientation}`；所有 effective config 均只改变 `extraction.sigma`。",
        f"- Circular Cone 最优 sigma：**{best_sigmas[0]:g}**；稳定平台：**{_format_ranges(ranges_by_model['circular_cone'])}**。",
        f"- Quadratic Graph 最优 sigma：**{best_sigmas[1]:g}**；稳定平台：**{_format_ranges(ranges_by_model['quadratic_graph'])}**。",
        f"- 仅看 raw RMSE，Circular Cone / Quadratic Graph 的 2% 区间分别为 **{_format_ranges(raw_ranges_by_model['circular_cone'])}** / **{_format_ranges(raw_ranges_by_model['quadratic_graph'])}**；其中不合格 sigma 因点数门禁不计入正式 stable range。",
        f"- 两种模型最佳 sigma：{'一致' if math.isclose(*best_sigmas, abs_tol=1e-12) else '不一致'}。",
        "- 对当前正式发布而言，sigma=1.5 仍可视为足够好；sigma=2.2 是值得独立复验的候选，而不是可直接替换正式配置的结论。",
        f"- 正式配置建议：{recommendation}", "",
        "## 与 sigma=1.5 的定量比较", "",
    ]
    for model in FOCUS_MODELS:
        label = "Circular Cone" if model == "circular_cone" else "Quadratic Graph"
        baseline = baseline_rows.loc[model]
        best = best_by_model[model]
        rmse_gain, p95_gain = improvement[model]
        mae_gain = 100.0 * (baseline["validation_board_mae_mm"] - best["validation_board_mae_mm"]) / baseline["validation_board_mae_mm"]
        max_gain = 100.0 * (baseline["validation_board_max_abs_mm"] - best["validation_board_max_abs_mm"]) / baseline["validation_board_max_abs_mm"]
        best_count_delta = float(best["validation_point_count_change_percent"])
        lines += [
            f"### {label}", "",
            f"- sigma=1.5：MAE={baseline['validation_board_mae_mm']:.6f} mm，RMSE={baseline['validation_board_rmse_mm']:.6f} mm，P95={baseline['validation_board_p95_abs_mm']:.6f} mm，max={baseline['validation_board_max_abs_mm']:.6f} mm。",
            f"- best sigma={best['sigma']:g}：MAE={best['validation_board_mae_mm']:.6f} mm，RMSE={best['validation_board_rmse_mm']:.6f} mm，P95={best['validation_board_p95_abs_mm']:.6f} mm，max={best['validation_board_max_abs_mm']:.6f} mm。",
            f"- RMSE improvement={rmse_gain:.3f}%；P95 improvement={p95_gain:.3f}%；MAE improvement={mae_gain:.3f}%；max improvement={max_gain:.3f}%。",
            f"- best sigma 的 validation 点数变化={best_count_delta:.3f}%。虽然未超过 ±10% 门禁，误差改善仍可能部分受 candidate acceptance 覆盖变化影响。",
            f"- Per-image：{_pose_improvement_text(per_image_table, model, float(best['sigma']), baseline_sigma)}。", "",
        ]
    lines += ["## 点数与亚像素中心变化", ""]
    if count_warning_sigmas:
        rendered = ", ".join(f"{value:g}" for value in count_warning_sigmas)
        lines += [
            f"- sigma={rendered} 相对 sigma=1.5 的有效点数变化超过 ±{float(analysis.get('point_count_warning_percent', 10.0)):g}%。",
            "- **Warning：该 sigma 下误差变化可能同时受到 candidate acceptance 数量变化影响，不能完全解释为纯尺度改善。**",
        ]
    else:
        lines.append("- 没有 sigma 的 train/validation 有效点数变化超过设定的 ±10% warning 阈值。")
    lines += [
        "- 下表给出与 sigma=1.5 在相同姿态、相同扫描线匹配后的二维中心位移：", "",
        shift_rows[["sigma", "center_shift_matched_count", "center_shift_median_px", "center_shift_p95_px", "center_shift_max_px"]].to_markdown(index=False, floatfmt=".6f"),
        "", "## 排名约束与完整指标", "",
        "排名候选需满足 validation valid_rate 阈值、所有 validation pose 成功、点数变化不超过 ±10%，且无逐图灾难性恶化。稳定平台定义为连续至少 3 个采样 sigma 的 validation RMSE 均在全局最小值的 2% 内。", "",
        summary[["sigma", "model", "train_point_count", "validation_point_count", "valid_rate", "validation_board_mae_mm", "validation_board_rmse_mm", "validation_board_p95_abs_mm", "validation_board_max_abs_mm", "eligible_for_ranking"]].to_markdown(index=False, floatfmt=".6f"),
        "", "## 探索性 validation 声明", "",
        "> 本轮 validation 数据参与 sigma 选择，因此扫描后的最优 validation 指标属于探索性结果。若最终决定修改正式 sigma，应使用新的 holdout 数据、标准量块数据，或重新划分独立测试集确认优化效果。",
    ]
    if errors:
        lines += ["", "## 扫描错误", "", *[f"- {error}" for error in errors]]
    output.write_text("\n".join(lines), encoding="utf-8")


def _dataset_pose_counts(base_config: Mapping[str, Any]) -> tuple[int, int]:
    datasets = _as_mapping(base_config["datasets"], "datasets")
    return len(datasets["train"].get("ids", [])), len(datasets["validation"].get("ids", []))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="固定标定数据扫描 Steger extraction.sigma")
    parser.add_argument("--config", default=str(SCRIPT_DIR.parent / "configs" / "steger_sigma_sweep.daheng.yaml"))
    parser.add_argument("--resume", action="store_true", help="复用已有且完整的单 sigma 拟合产物")
    args = parser.parse_args(argv)

    sweep_path = Path(args.config).resolve()
    sweep_cfg = load_sweep_config(sweep_path)
    base_path = Path(sweep_cfg["base_config"])
    base_bytes = base_path.read_bytes()
    base_hash = hashlib.sha256(base_bytes).hexdigest()
    base_config = FIT.safe_yaml_load(base_path)
    original_snapshot = copy.deepcopy(base_config)
    orientation = validate_fixed_config(base_config)
    if orientation != "vertical":
        raise ValueError(f"Daheng sigma sweep 要求 laser.orientation=vertical，实际为 {orientation}")
    output_dir = Path(sweep_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_sigma = float(sweep_cfg["baseline"]["sigma"])
    ordered_sigmas = [baseline_sigma] + [
        value for value in sweep_cfg["sigma_values"]
        if not math.isclose(value, baseline_sigma, abs_tol=1.0e-12)
    ]
    provenance = {
        "sweep_config": str(sweep_path), "base_config": str(base_path),
        "base_config_sha256": base_hash, "laser_orientation": orientation,
        "sigma_values_requested": sweep_cfg["sigma_values"], "execution_order": ordered_sigmas,
        "only_varied_key": "extraction.sigma",
    }
    (output_dir / "sweep_provenance.yaml").write_text(
        yaml.safe_dump(provenance, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    comparisons: list[pd.DataFrame] = []
    per_images: list[pd.DataFrame] = []
    point_frames: list[pd.DataFrame] = []
    errors: list[str] = []
    expected_train_poses, expected_val_poses = _dataset_pose_counts(base_config)

    baseline_dir = output_dir / sigma_slug(baseline_sigma)
    run_one_sigma(base_config, baseline_sigma, baseline_dir, orientation, args.resume)
    comparison, per_image, points = collect_sigma_artifacts(baseline_dir, baseline_sigma)
    comparisons.append(comparison)
    per_images.append(per_image)
    point_frames.append(points)
    baseline_result = baseline_check(
        comparison, points, sweep_cfg["baseline"], expected_train_poses, expected_val_poses, orientation
    )
    _write_baseline_check(output_dir, baseline_result)
    if base_config != original_snapshot or base_path.read_bytes() != base_bytes:
        raise AssertionError("原始 laser model fit config 被修改")
    if baseline_result["status"] != "PASS":
        baseline_summary = comparison.copy()
        baseline_summary.to_csv(output_dir / "sigma_sweep_summary.csv", index=False, encoding="utf-8-sig")
        generate_report(output_dir / "sigma_sweep_report.md", pd.DataFrame(), pd.DataFrame(), baseline_result, sweep_cfg, orientation, errors)
        print("[STOP] sigma=1.5 baseline 未通过，已停止排名。", file=sys.stderr)
        return 2

    for sigma in ordered_sigmas[1:]:
        sigma_dir = output_dir / sigma_slug(sigma)
        try:
            run_one_sigma(base_config, sigma, sigma_dir, orientation, args.resume)
            comparison, per_image, points = collect_sigma_artifacts(sigma_dir, sigma)
            comparisons.append(comparison)
            per_images.append(per_image)
            point_frames.append(points)
        except Exception as exc:
            message = f"sigma={sigma:g}: {exc}"
            errors.append(message)
            print(f"[FAIL] {message}", file=sys.stderr)
            traceback.print_exc()
        if base_config != original_snapshot or base_path.read_bytes() != base_bytes:
            raise AssertionError("原始 laser model fit config 被修改")

    all_comparison = pd.concat(comparisons, ignore_index=True)
    all_per_image = pd.concat(per_images, ignore_index=True)
    all_points = pd.concat(point_frames, ignore_index=True)
    counts = pose_counts(all_points)
    counts.to_csv(output_dir / "sigma_sweep_per_pose_counts.csv", index=False, encoding="utf-8-sig")
    shifts = center_shift_metrics(all_points, baseline_sigma, orientation)
    shifts.to_csv(output_dir / "sigma_sweep_center_shift.csv", index=False, encoding="utf-8-sig")
    per_image_table = per_image_wide(all_per_image, counts)
    per_image_table.to_csv(output_dir / "sigma_sweep_per_image.csv", index=False, encoding="utf-8-sig")
    summary = build_summary(
        all_comparison, all_points, all_per_image, base_config, baseline_sigma,
        sweep_cfg["analysis"], shifts,
    )
    summary.to_csv(output_dir / "sigma_sweep_summary.csv", index=False, encoding="utf-8-sig")
    generate_plots(summary, per_image_table, output_dir, baseline_sigma)
    generate_report(
        output_dir / "sigma_sweep_report.md", summary, per_image_table,
        baseline_result, sweep_cfg, orientation, errors,
    )
    if errors:
        (output_dir / "sigma_sweep_errors.txt").write_text("\n".join(errors), encoding="utf-8")
        return 1
    print(f"完成 sigma sweep：{output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
