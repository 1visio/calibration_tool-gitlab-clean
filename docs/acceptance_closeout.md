# 阶段 4：报告、补偿和验收闭环

## 闭环顺序

```text
采集 manifest
    ↓
统一 workflow 与阶段质量门禁
    ↓
独立 holdout 补偿评估
    ↓
运行 config / manifest / extractor 一致性
    ↓
golden 漂移检查与产物 SHA-256
    ↓
acceptance_report.yaml + HTML + CSV
    ↓
accepted 才允许生成 calibration bundle
```

阶段 4 不重新实现标定或补偿数学，而是使用阶段 1 的 stage 结果和
`generate_ground_bias_compensation.py` 的输出完成最终判定。

## 补偿验收

逐列补偿表定义为：

```text
bias(u) = mean_Z(u) - retained_trend(u)
Z_corrected = Z_raw - bias(u)
```

正式验收必须满足：

- `evaluation_frame_count > 0`；
- `build_frame_count + evaluation_frame_count == loaded_frame_count`；
- 补偿后 P–V、RMS 不超过 policy 上限；
- P–V、RMS 相对补偿前达到最小降幅；
- 重复性 sigma 不超过上限。

如果建表和评估使用同一批帧，即使补偿后残差接近零，也会被
`compensation.independent_validation` 门禁拒绝。

默认阈值位于 `configs/acceptance_policy.yaml`：

| 指标 | 默认要求 |
|---|---:|
| 补偿后平均剖面 P–V | ≤ 0.25 mm |
| 补偿后平均剖面 RMS | ≤ 0.05 mm |
| P–V 降幅 | ≥ 80% |
| RMS 降幅 | ≥ 80% |
| 重复性 sigma 中位数 | ≤ 0.02 mm |

## 整体验收门禁

除补偿外，报告还会检查：

- workflow 是否完成、各 stage 是否为 `completed`；
- stage 自带的独立验证和质量门禁；
- 上游质量报告是否存在 fail；
- 运行配置、manifest、标定文件哈希和中心提取器是否一致；
- 当前文件是否偏离 golden baseline；
- 明确要求的诊断图、补偿表和其他产物是否存在；
- 全部输入产物的 SHA-256。

任一 fail 都会得到 `decision: rejected`。warn 会保留在报告中，但没有 fail 时仍可接受。

## 当前历史数据的结果

`configs/acceptance_plan.example.yaml` 使用现有 golden 审计与独立 holdout 补偿结果：

| 指标 | 补偿前 | 补偿后 | 降幅 |
|---|---:|---:|---:|
| 平均剖面 P–V | 2.581978 mm | 0.058733 mm | 97.73% |
| 平均剖面 RMS | 0.662812 mm | 0.004441 mm | 99.33% |

补偿使用 8 帧建表、2 帧独立验收，重复性 sigma 中位数为 0.003680 mm，补偿门禁通过。
但整体验收仍为 `rejected`，因为阶段 0 记录的历史链仍包含以下未关闭问题：

- 当前运行 extractor 与标定 manifest / shared Steger 链不一致；
- 内参没有独立测试图；
- 激光平面三联图存在运动和未解析项；
- 历史混合式地面平整度未通过；
- 历史 golden 补偿项没有独立验证。

这份 rejected 报告是现状审计，不应通过放宽阶段 4 policy 变成正式发布结果。正确做法是用阶段 2/3
重新采集独立数据、运行完整 workflow，然后用 `acceptance_policy.yaml` 重新验收。

## 发布闭环

验收计划的 `release.enabled` 默认关闭。启用后：

- rejected：只生成报告，release 标记为 `blocked`；
- accepted：调用 bundle 发布逻辑，复制不可拆分标定文件；
- 发布包额外携带 `acceptance_report.yaml/html` 及它们的 SHA-256；
- 不允许用旧报告绕过运行 profile、manifest 或标定文件哈希检查。
