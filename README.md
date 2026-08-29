# 线激光标定工具

本项目提供线激光 3D 扫描系统的统一标定入口，包含相机采集、相机内参、激光面/曲面、地面外参、地面偏差补偿、验收报告和标定包发布等功能。仓库已内置运行所需的 `calibration/src` 算法源码与通用配置，可以从一个干净的源码目录开始建立新标定项目。

## 环境要求

- Windows 10/11
- Python 3.10 或更高版本
- 使用真实相机时，需要安装对应厂商 SDK：海康 MVS 或大恒 Galaxy SDK

建议在项目根目录创建独立虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

需要运行测试或分析脚本时：

```powershell
python -m pip install -e ".[test]"
```

## 快速开始

先验证 CLI 和内置算法目录：

```powershell
python -m calibration_tool list-stages
python -m calibration_tool camera-list --backend synthetic
```

无相机演练 GUI：

```powershell
python -m calibration_tool gui --simulate
```

建立实际项目时，复制示例配置再修改，不要直接覆盖模板：

```powershell
Copy-Item configs\wizard_project.example.yaml configs\wizard_project.local.yaml
Copy-Item configs\camera_channels.example.yaml configs\camera_channels.local.yaml
python -m calibration_tool gui --project configs\wizard_project.local.yaml
```

完整操作说明见 [线激光标定工具用户手册](docs/线激光标定工具用户手册.md)。

## 主要命令

```text
list-stages                 列出可用标定阶段
run <stage>                 单独运行一个标定阶段
workflow <plan.yaml>        按 YAML 顺序运行多个阶段
camera-list                 枚举相机
camera-preview              预览并检查图像质量
capture-plan                执行批量采集计划
acceptance-report           生成验收报告
bundle-build                构建发布标定包
gui                         启动图形界面
```

用 `python -m calibration_tool <命令> --help` 查看详细参数。

## 目录说明

```text
calibration_tool/   CLI、GUI、采集、流程、验收与发布逻辑
calibration/src/    内置标定算法源码
calibration/config/ 算法默认配置
configs/            可复制的通用配置模板
docs/               用户手册与必要说明
scripts/            测试及诊断所需的少量分析脚本
tests/              自动化测试
data/               本地采集数据（不提交）
projects/           本地项目工作区（不提交）
runs/               标定运行结果（不提交）
reports/            本地验收报告（不提交）
releases/           本地发布包（不提交）
```

## 配置与数据约定

- 示例 YAML 中的相对路径以 YAML 文件自身所在目录为基准。
- `calibration_src` 默认使用仓库内的 `calibration/src`，无需依赖相邻目录。
- 相机序列号、项目路径和数据集路径只写入本地配置；不要把现场机器路径写回示例模板。
- 原始图像、点云、CSV、运行报告和发布包默认由 `.gitignore` 排除。
- 示例 workflow 的所有阶段默认关闭。确认输入、输出和参数后，再逐项设置 `enabled: true`。
- 本仓库不附带任何“已验收”的 golden baseline 或生产标定包。请基于本项目真实数据生成，并通过质量门禁与验收后再发布。

## 相机 SDK

- `synthetic`：无需硬件，用于安装和流程演练。
- `mvs`：安装海康 MVS SDK，按 `configs/camera.mvs.example.yaml` 配置。
- `daheng`：安装大恒 Galaxy SDK；非默认安装位置可设置 `DAHENG_GALAXY_ROOT`。

## 测试

```powershell
python -m pytest -q
```

测试会使用临时目录或模拟数据，不需要提交本地标定数据。

