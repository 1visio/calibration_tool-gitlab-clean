"""标定采集预检 —— 在拍摄现场判断这组图够不够，还缺什么姿态。

用法（边拍边跑，几十秒出结果）：

    python check_calibration_capture.py --dir <拍摄目录> --pattern "chess *.tif" \
        --focal-length-mm 16 --pixel-um 3.45

不需要事先标定。有已知内参时用 --reference-yaml 更准；都没有则内部做一次快速标定。
结果带缓存，补拍后再跑只处理新增图像。

检查五项，对应 calibrate_chessboard_opencv_v2.py 的验收门槛：
  1 板面倾角（大小 + 方位覆盖）—— 决定 f/深度、主点能否解耦
  2 工作距离跨度              —— 决定 f 与 Z 的耦合能否打开
  3 画面覆盖（网格 + 半径）    —— 决定畸变系数能否被约束
  4 清晰度（角点处实测模糊 σ）—— 验证收光圈/倾角的取舍是否成立
  5 数量（图像数、角点数）

模糊 σ 的估计原理：一个被 σ 高斯模糊的阶跃边缘，其最大梯度为 ΔI/(σ√(2π))，
故 σ ≈ ΔI / (max_grad · 2.5066)。在每个角点邻域内测，得到该点的实际模糊。
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SECTORS = ["→", "↘", "↓", "↙", "←", "↖", "↑", "↗"]
ZONE_NAMES = [["左上", "上中", "右上"], ["左中", "正中", "右中"], ["左下", "下中", "右下"]]
GAUSS_EDGE = math.sqrt(2.0 * math.pi)   # 2.5066
# 估计器自身的基底：Sobel 3x3 核的平滑 + 像素积分。用合成阶跃边缘标定得到 0.77 px。
# 真实 σ 按 sqrt(raw² − floor²) 反解；标定显示 σ_true ≈ 0.987·σ_corr + 0.07（σ>0.5px 时）。
# 低于约 0.4 px 分辨不出来，一律视为"锐利"。
BLUR_FLOOR_PX = 0.77


@dataclass
class Shot:
    name: str
    ok: bool
    reason: str = ""
    tilt_deg: float = 0.0
    tilt_azimuth_deg: float = 0.0
    depth_mm: float = 0.0
    blur_sigma_median: float = 0.0
    blur_sigma_max: float = 0.0
    clipped_fraction: float = 0.0
    board_area_fraction: float = 0.0
    centre_u: float = 0.0
    centre_v: float = 0.0
    radius_max: float = 0.0


# --------------------------------------------------------------------------- #
def natural_key(path: Path):
    return [(1, int(p)) if p.isdigit() else (0, p.casefold())
            for p in re.split(r"(\d+)", path.name)]


def read_image(path: Path):
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        return None if buf.size == 0 else cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except (OSError, cv2.error):
        return None


def detect(gray, pattern_size):
    if hasattr(cv2, "findChessboardCornersSB"):
        flags = (cv2.CALIB_CB_NORMALIZE_IMAGE
                 | getattr(cv2, "CALIB_CB_EXHAUSTIVE", 0)
                 | getattr(cv2, "CALIB_CB_ACCURACY", 0))
        try:
            found, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags=flags)
            if found and corners is not None:
                return corners.reshape(-1, 2).astype(np.float64)
        except cv2.error:
            pass
    found, corners = cv2.findChessboardCorners(
        gray, pattern_size, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not found or corners is None:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    return cv2.cornerSubPix(gray, corners.astype(np.float32), (11, 11), (-1, -1),
                            criteria).reshape(-1, 2).astype(np.float64)


def blur_sigma_at(gray32, corners, half=15, floor=BLUR_FLOOR_PX):
    """在每个角点邻域用「最大梯度法」估计等效高斯模糊 σ（像素）。"""
    h, w = gray32.shape
    sigmas, clipped = [], 0.0
    total = 0
    for u, v in corners:
        x, y = int(round(u)), int(round(v))
        if x - half < 0 or y - half < 0 or x + half >= w or y + half >= h:
            continue
        patch = gray32[y - half:y + half + 1, x - half:x + half + 1]
        lo, hi = np.percentile(patch, [2, 98])
        contrast = hi - lo
        if contrast < 12:                       # 对比度太低，测不准
            continue
        gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3) / 8.0
        gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3) / 8.0
        grad = np.hypot(gx, gy)
        peak = float(np.percentile(grad, 99.0))
        if peak <= 1e-6:
            continue
        raw = contrast / (peak * GAUSS_EDGE)
        sigmas.append(math.sqrt(max(raw * raw - floor * floor, 0.0)))
        clipped += float(np.mean((patch <= 1) | (patch >= 254)))
        total += 1
    if not sigmas:
        return float("nan"), float("nan"), 0.0
    return (float(np.median(sigmas)), float(np.percentile(sigmas, 95)),
            clipped / max(total, 1))


def board_pose(object_points, corners, K, D):
    ok, rvec, tvec = cv2.solvePnP(object_points, corners, K, D,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    normal = R @ np.array([0.0, 0.0, 1.0])
    if normal[2] > 0:                            # 统一朝向相机
        normal = -normal
    tilt = math.degrees(math.acos(min(1.0, abs(normal[2]))))
    azimuth = math.degrees(math.atan2(normal[1], normal[0])) % 360.0
    return tilt, azimuth, float(tvec.reshape(-1)[2])


# --------------------------------------------------------------------------- #
def bar(value, limit, width=18, reverse=False):
    ratio = value / limit if limit else 0.0
    ratio = min(1.0, ratio if not reverse else (limit / value if value else 2.0))
    filled = int(round(ratio * width))
    return "█" * filled + "░" * (width - filled)


def verdict_mark(status):
    return {"PASS": "✓", "WARN": "⚠", "FAIL": "✗"}[status]


def grade(value, limit, mode):
    if not np.isfinite(value):
        return "WARN"
    if mode == "min":
        return "PASS" if value >= limit else ("WARN" if value >= 0.7 * limit else "FAIL")
    return "PASS" if value <= limit else ("WARN" if value <= 1.5 * limit else "FAIL")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="标定采集预检：判断这组图够不够，还缺什么",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--dir", type=Path, required=True, help="拍摄目录（可含子目录 fit/test）")
    ap.add_argument("--pattern", default="chess *.tif", help="文件 glob")
    ap.add_argument("--recursive", action="store_true", help="递归子目录")
    ap.add_argument("--pattern-cols", type=int, default=6)
    ap.add_argument("--pattern-rows", type=int, default=5)
    ap.add_argument("--square-size-mm", type=float, default=30.0)
    ap.add_argument("--focal-length-mm", type=float, default=None,
                    help="镜头标称焦距，配合 --pixel-um 用作名义内参（推荐）")
    ap.add_argument("--pixel-um", type=float, default=3.45, help="像元尺寸 µm")
    ap.add_argument("--reference-yaml", type=Path, default=None,
                    help="已有 calibration_result.yaml，优先用它的内参")
    ap.add_argument("--report", type=Path, default=None, help="输出 CSV 逐图报告")
    ap.add_argument("--dashboard", type=Path, default=None, help="输出 PNG 看板")
    ap.add_argument("--cache", type=Path, default=None,
                    help="缓存文件（默认 <dir>/.capture_check_cache.json），补拍后只处理新图")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--blur-floor-px", type=float, default=BLUR_FLOOR_PX,
                    help="模糊估计器基底（合成阶跃标定值），一般无需改")

    g = ap.add_argument_group("门槛（与 v2 标定脚本一致）")
    g.add_argument("--min-tilt-median", type=float, default=20.0)
    g.add_argument("--min-tilt-max", type=float, default=30.0)
    g.add_argument("--min-strong-tilt-images", type=int, default=6,
                   help="倾角 ≥30° 的图像张数下限")
    g.add_argument("--min-tilt-sectors", type=int, default=6,
                   help="需要覆盖的倾斜方位扇区数（共 8 个）")
    g.add_argument("--min-depth-span", type=float, default=0.20)
    g.add_argument("--min-radius-coverage", type=float, default=0.90)
    g.add_argument("--max-empty-cells", type=int, default=4)
    g.add_argument("--max-blur-sigma", type=float, default=1.20,
                   help="角点处模糊 σ 上限（px）；超过说明失焦或抖动")
    g.add_argument("--max-clipped", type=float, default=0.02, help="过曝/欠曝像素占比上限")
    g.add_argument("--min-images", type=int, default=20)
    g.add_argument("--min-total-points", type=int, default=1500)
    args = ap.parse_args()

    pattern_size = (args.pattern_cols, args.pattern_rows)
    obj = np.zeros((args.pattern_cols * args.pattern_rows, 3), np.float64)
    obj[:, :2] = np.mgrid[0:args.pattern_cols, 0:args.pattern_rows].T.reshape(-1, 2) \
        * args.square_size_mm

    globber = args.dir.rglob if args.recursive else args.dir.glob
    paths = sorted((p for p in globber(args.pattern)
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
                   key=natural_key)
    if not paths:
        print(f"错误：{args.dir} 下没有匹配 {args.pattern!r} 的图像", file=sys.stderr)
        return 2

    cache_path = args.cache or (args.dir / ".capture_check_cache.json")
    cache = {}
    if not args.no_cache and cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cache = {}

    # ---- 逐图检测与测量 ----
    records, corner_sets, image_size = [], {}, None
    fresh = 0
    for path in paths:
        stat = path.stat()
        key = f"{path.name}|{int(stat.st_mtime)}|{stat.st_size}"
        image = read_image(path)
        if image is None:
            records.append(Shot(path.name, False, "读取失败"))
            continue
        if image_size is None:
            image_size = (image.shape[1], image.shape[0])
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        entry = cache.get(key)
        if entry is not None:
            corners = np.asarray(entry["corners"], dtype=np.float64)
            blur = entry["blur"]
        else:
            corners = detect(gray, pattern_size)
            if corners is None:
                records.append(Shot(path.name, False, "未检出完整棋盘"))
                cache[key] = {"corners": None, "blur": None}
                print(f"  ✗ {path.name:24s} 未检出完整棋盘")
                continue
            blur = blur_sigma_at(gray.astype(np.float32), corners,
                                 floor=args.blur_floor_px)
            cache[key] = {"corners": corners.tolist(), "blur": list(blur)}
            fresh += 1
        if corners is None:
            records.append(Shot(path.name, False, "未检出完整棋盘"))
            continue
        corner_sets[path.name] = corners
        records.append(Shot(path.name, True, "", blur_sigma_median=blur[0],
                            blur_sigma_max=blur[1], clipped_fraction=blur[2],
                            centre_u=float(corners[:, 0].mean()),
                            centre_v=float(corners[:, 1].mean())))
    if not args.no_cache:
        cache_path.write_text(json.dumps(cache), encoding="utf-8")

    good = [r for r in records if r.ok]
    if len(good) < 3:
        print(f"检出图像不足 3 张（{len(good)}/{len(records)}），无法评估")
        return 2

    # ---- 内参来源 ----
    width, height = image_size
    K = D = None
    source = ""
    if args.reference_yaml and args.reference_yaml.exists():
        import yaml
        data = yaml.safe_load(args.reference_yaml.read_text(encoding="utf-8"))
        K = np.array(data["camera_matrix"], float)
        D = np.array(data["dist_coeffs"], float)
        source = f"参考文件 {args.reference_yaml.name}"
    elif args.focal_length_mm:
        f = args.focal_length_mm / (args.pixel_um * 1e-3)
        K = np.array([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1]])
        D = np.zeros(5)
        source = f"名义值（{args.focal_length_mm}mm / {args.pixel_um}µm → f={f:.0f}px）"
    else:
        pts_o = [obj.astype(np.float32).reshape(-1, 1, 3) for _ in good]
        pts_i = [corner_sets[r.name].astype(np.float32).reshape(-1, 1, 2) for r in good]
        _, K, D, _, _ = cv2.calibrateCamera(pts_o, pts_i, (width, height), None, None,
                                            flags=cv2.CALIB_FIX_K3)
        source = "内部快速标定（数据退化时该值本身不可信，仅用于姿态诊断）"

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    r_corner = max(math.hypot((x - cx) / fx, (y - cy) / fy)
                   for x in (0.0, width) for y in (0.0, height))

    for record in good:
        corners = corner_sets[record.name]
        pose = board_pose(obj, corners, K, D)
        if pose is None:
            record.ok = False
            record.reason = "位姿求解失败"
            continue
        record.tilt_deg, record.tilt_azimuth_deg, record.depth_mm = pose
        hull = cv2.convexHull(corners.astype(np.float32))
        record.board_area_fraction = float(cv2.contourArea(hull) / (width * height))
        record.radius_max = float(np.max(np.hypot((corners[:, 0] - cx) / fx,
                                                  (corners[:, 1] - cy) / fy)))
    good = [r for r in good if r.ok]

    # ---- 汇总 ----
    tilts = np.array([r.tilt_deg for r in good])
    depths = np.array([r.depth_mm for r in good])
    blur_med = np.array([r.blur_sigma_median for r in good])
    blur_max = np.array([r.blur_sigma_max for r in good])
    clipped = np.array([r.clipped_fraction for r in good])
    all_corners = np.vstack([corner_sets[r.name] for r in good])
    radius_max = float(np.max(np.hypot((all_corners[:, 0] - cx) / fx,
                                       (all_corners[:, 1] - cy) / fy)))
    radius_cov = radius_max / r_corner
    occupancy = np.zeros((8, 8), int)
    for x, y in all_corners:
        if 0 <= x < width and 0 <= y < height:
            occupancy[int(y / height * 8), int(x / width * 8)] += 1
    empty_cells = int((occupancy == 0).sum())
    zones = np.zeros((3, 3), int)
    for r in good:
        zones[min(2, int(r.centre_v / height * 3)), min(2, int(r.centre_u / width * 3))] += 1
    sector_hit = np.zeros(8, int)
    for r in good:
        if r.tilt_deg >= 20.0:
            sector_hit[int(((r.tilt_azimuth_deg + 22.5) % 360) // 45)] += 1
    depth_span = float((depths.max() - depths.min()) / depths.mean())
    strong = int((tilts >= 30.0).sum())
    total_points = int(len(all_corners))

    checks = [
        ("板面倾角中位", float(np.median(tilts)), args.min_tilt_median, "min", "°"),
        ("板面倾角最大", float(tilts.max()), args.min_tilt_max, "min", "°"),
        ("≥30° 的图像数", float(strong), float(args.min_strong_tilt_images), "min", "张"),
        ("倾斜方位扇区数", float((sector_hit > 0).sum()), float(args.min_tilt_sectors),
         "min", "/8"),
        ("工作距离相对跨度", depth_span, args.min_depth_span, "min", ""),
        ("半径覆盖率", radius_cov, args.min_radius_coverage, "min", ""),
        ("8×8 空格子", float(empty_cells), float(args.max_empty_cells), "max", "格"),
        ("角点模糊 σ 中位", float(np.median(blur_med)), args.max_blur_sigma, "max", "px"),
        ("角点模糊 σ 最差", float(np.max(blur_max)), args.max_blur_sigma * 1.6, "max", "px"),
        ("过曝欠曝占比", float(np.max(clipped)), args.max_clipped, "max", ""),
        ("有效图像数", float(len(good)), float(args.min_images), "min", "张"),
        ("总角点数", float(total_points), float(args.min_total_points), "min", "点"),
    ]
    results = [(name, value, limit, mode, unit, grade(value, limit, mode))
               for name, value, limit, mode, unit in checks]

    # ---- 打印 ----
    print("\n" + "=" * 64)
    print(f"标定采集预检　{args.dir}")
    print("=" * 64)
    print(f"图像 {len(records)} 张｜检出 {len(good)} 张｜失败 "
          f"{len(records) - len(good)} 张｜本次新处理 {fresh} 张")
    print(f"内参来源：{source}")
    failures = [r for r in records if not r.ok]
    if failures:
        print("失败图：" + "，".join(f"{r.name}({r.reason})" for r in failures[:8])
              + ("…" if len(failures) > 8 else ""))

    print("\n【1】板面倾角")
    print(f"    中位 {np.median(tilts):5.1f}°　最大 {tilts.max():5.1f}°　"
          f"最小 {tilts.min():5.1f}°　≥30°: {strong} 张")
    hist, _ = np.histogram(tilts, bins=[0, 10, 20, 30, 40, 90])
    print("    分布  <10°:{0}  10-20°:{1}  20-30°:{2}  30-40°:{3}  >40°:{4}".format(*hist))
    print("    倾斜方位覆盖（≥20°）：" +
          " ".join(f"{SECTORS[i]}{sector_hit[i]}" if sector_hit[i] else f"{SECTORS[i]}·"
                   for i in range(8)))

    print("\n【2】工作距离")
    print(f"    {depths.min():.0f} – {depths.max():.0f} mm　相对跨度 {depth_span * 100:.1f}%"
          f"　中位 {np.median(depths):.0f} mm")

    print("\n【3】画面覆盖")
    print(f"    归一化半径 {radius_max:.3f} / {r_corner:.3f}（图像四角）= {radius_cov * 100:.1f}%")
    print(f"    8×8 网格空格子 {empty_cells}/64")
    for row in occupancy:
        print("      " + " ".join(f"{v:4d}" if v else "   ." for v in row))
    print("    板心 3×3 分区分布：")
    for row in zones:
        print("      " + " ".join(f"{v:3d}" for v in row))

    print("\n【4】清晰度（角点处实测模糊 σ）")
    print(f"    逐图中位 σ：{np.median(blur_med):.2f} px（最差图 "
          f"{good[int(np.argmax(blur_med))].name} = {blur_med.max():.2f} px）")
    print(f"    单点最差 σ：{blur_max.max():.2f} px（{good[int(np.argmax(blur_max))].name}）")
    ratio = blur_max / np.maximum(blur_med, 1e-6)
    worst = good[int(np.argmax(ratio))]
    print(f"    图内不均匀度 max/median 最大 {ratio.max():.2f}（{worst.name}）"
          f"　{'← 倾角导致边缘失焦，正常' if ratio.max() > 1.5 else ''}")
    if np.max(clipped) > 0.005:
        print(f"    ⚠ 过曝/欠曝最高 {np.max(clipped) * 100:.1f}%"
              f"（{good[int(np.argmax(clipped))].name}）")

    print("\n【5】验收")
    print(f"    {'项目':<18}{'实测':>10}  {'判据':<14}{'状态'}")
    for name, value, limit, mode, unit, status in results:
        op = "≥" if mode == "min" else "≤"
        print(f"    {name:<18}{value:10.3f}{unit:<3}{op} {limit:<10.3f} "
              f"{verdict_mark(status)} {status}")

    fails = [r for r in results if r[5] == "FAIL"]
    warns = [r for r in results if r[5] == "WARN"]
    overall = "不合格" if fails else ("基本可用，建议补拍" if warns else "合格")
    print(f"\n    判定：{verdict_mark('FAIL' if fails else ('WARN' if warns else 'PASS'))} "
          f"{overall}（FAIL {len(fails)}｜WARN {len(warns)}）")

    # ---- 待补拍建议 ----
    todo = []
    if np.median(tilts) < args.min_tilt_median or strong < args.min_strong_tilt_images:
        todo.append(f"倾角 ≥30° 的图还缺约 {max(0, args.min_strong_tilt_images - strong)} 张")
    missing = [SECTORS[i] for i in range(8) if sector_hit[i] == 0]
    if len(missing) > 8 - args.min_tilt_sectors:
        todo.append(f"缺这些方向的大倾角图：{' '.join(missing)}"
                    "（方位定义：法线在图像坐标下偏向的方向，→ 为右、↓ 为下）")
    if depth_span < args.min_depth_span:
        need = np.median(depths)
        todo.append(f"工作距离太集中，补拍 ~{need * 0.85:.0f} mm 和 ~{need * 1.15:.0f} mm 各 3 张")
    empty_zone = [ZONE_NAMES[i][j] for i in range(3) for j in range(3) if zones[i, j] == 0]
    if empty_zone:
        todo.append(f"板心从未出现在这些区域：{'、'.join(empty_zone)}，各补 2 张")
    if radius_cov < args.min_radius_coverage or empty_cells > args.max_empty_cells:
        todo.append("边角覆盖不足：把板子推到画面四角与四边；普通棋盘不许出画，"
                    "如需贴边请改用 ChArUco")
    if np.median(blur_med) > args.max_blur_sigma:
        todo.append("整体偏糊：收小一档光圈并延长曝光，或检查对焦与机械抖动")
    if float(np.max(clipped)) > args.max_clipped:
        todo.append("存在过曝/欠曝：降低曝光或改善打光均匀性")
    if len(good) < args.min_images:
        todo.append(f"有效图像还差 {args.min_images - len(good)} 张")
    if total_points < args.min_total_points:
        per = args.pattern_cols * args.pattern_rows
        todo.append(f"总角点数偏少（每张仅 {per} 点）：建议换 ≥9×7 的棋盘或 ChArUco，"
                    f"当前需 {math.ceil(args.min_total_points / per)} 张才够")
    if len(records) - len(good) > 0:
        todo.append(f"{len(records) - len(good)} 张未检出，建议按上面列出的文件名补拍替换")

    print("\n【6】待补拍")
    if todo:
        for item in todo:
            print(f"    • {item}")
    else:
        print("    无 —— 可以直接进 calibrate_chessboard_opencv_v2.py 正式标定")

    # ---- 附件输出 ----
    if args.report:
        import csv
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(asdict(records[0])))
            writer.writeheader()
            for record in records:
                writer.writerow(asdict(record))
        print(f"\n逐图报告：{args.report}")

    if args.dashboard:
        draw_dashboard(args.dashboard, all_corners, occupancy, tilts, sector_hit,
                       depths, blur_med, (width, height), overall)
        print(f"看板：{args.dashboard}")

    return 1 if fails else 0


def draw_dashboard(path: Path, corners, occupancy, tilts, sector_hit, depths,
                   blur, image_size, overall):
    width, height = image_size
    scale = 520.0 / width
    ch, cw = int(height * scale), 520
    right_w, pad = 300, 20
    body_h = max(ch + 40, 660)
    canvas = np.full((body_h + 60, cw + right_w + pad, 3), 255, np.uint8)

    # 左：角点覆盖图
    top = 34
    for i in range(1, 8):
        cv2.line(canvas, (int(cw * i / 8), top), (int(cw * i / 8), top + ch),
                 (228, 228, 228), 1)
        cv2.line(canvas, (0, top + int(ch * i / 8)), (cw, top + int(ch * i / 8)),
                 (228, 228, 228), 1)
    for x, y in corners:
        cv2.circle(canvas, (int(x * scale), top + int(y * scale)), 2, (190, 70, 40), -1)
    for r in range(8):
        for c in range(8):
            if occupancy[r, c] == 0:
                cv2.rectangle(canvas, (int(cw * c / 8) + 1, top + int(ch * r / 8) + 1),
                              (int(cw * (c + 1) / 8) - 1, top + int(ch * (r + 1) / 8) - 1),
                              (0, 0, 230), 1)
    cv2.putText(canvas, f"corner coverage   {(occupancy == 0).sum()}/64 cells empty (red)",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (20, 20, 20), 1, cv2.LINE_AA)

    # 右：三个直方图
    def hist_panel(x0, y0, w, h, values, bins, title, limit=None, limit_label=""):
        cv2.putText(canvas, title, (x0, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                    (20, 20, 20), 1, cv2.LINE_AA)
        counts, edges = np.histogram(values, bins=bins)
        top_n = max(1, counts.max())
        for i, count in enumerate(counts):
            bx = x0 + int(i * w / len(counts))
            bw = max(4, int(w / len(counts)) - 4)
            bh = int(count / top_n * h)
            if bh:
                cv2.rectangle(canvas, (bx, y0 + h - bh), (bx + bw, y0 + h),
                              (200, 120, 60), -1)
            cv2.putText(canvas, str(count), (bx + 1, y0 + h - bh - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, (70, 70, 70), 1, cv2.LINE_AA)
        cv2.line(canvas, (x0, y0 + h), (x0 + w, y0 + h), (120, 120, 120), 1)
        cv2.putText(canvas, f"{edges[0]:.0f}", (x0, y0 + h + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (110, 110, 110), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{edges[-1]:.0f}", (x0 + w - 26, y0 + h + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (110, 110, 110), 1, cv2.LINE_AA)
        if limit is not None and edges[-1] > edges[0]:
            lx = x0 + int((limit - edges[0]) / (edges[-1] - edges[0]) * w)
            if x0 <= lx <= x0 + w:
                cv2.line(canvas, (lx, y0 - 2), (lx, y0 + h), (0, 0, 220), 1)
                cv2.putText(canvas, limit_label, (max(x0, lx - 24), y0 + h + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 0, 200), 1, cv2.LINE_AA)

    rx = cw + pad
    hist_panel(rx, 48, right_w - pad, 92, tilts, [0, 10, 20, 30, 40, 90],
               "board tilt (deg)", 30, "min")
    hist_panel(rx, 196, right_w - pad, 92, depths, 6, "working distance (mm)")
    hist_panel(rx, 344, right_w - pad, 92, blur, 6, "corner blur sigma (px)", 1.2, "max")

    # 右下：倾斜方位覆盖
    cv2.putText(canvas, "tilt azimuth coverage (>=20 deg)", (rx, 470),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (20, 20, 20), 1, cv2.LINE_AA)
    cx0, cy0, rad = rx + (right_w - pad) // 2, 560, 62
    cv2.circle(canvas, (cx0, cy0), rad, (235, 235, 235), 1)
    for i in range(8):
        angle = math.radians(i * 45)
        hit = sector_hit[i] > 0
        px = int(cx0 + rad * math.cos(angle))
        py = int(cy0 + rad * math.sin(angle))
        colour = (60, 165, 60) if hit else (205, 205, 205)
        cv2.line(canvas, (cx0, cy0), (px, py), colour, 2)
        cv2.circle(canvas, (px, py), 11, colour, -1)
        cv2.putText(canvas, str(sector_hit[i]), (px - 4, py + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{int((sector_hit > 0).sum())}/8 sectors",
                (cx0 - 34, cy0 + rad + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (110, 110, 110), 1, cv2.LINE_AA)

    label = {"合格": ("PASS - ready to calibrate", (60, 160, 60)),
             "基本可用，建议补拍": ("USABLE - shoot more", (0, 150, 220)),
             "不合格": ("FAIL - do not calibrate yet", (40, 40, 220))}[overall]
    cv2.rectangle(canvas, (0, body_h + 6), (canvas.shape[1], body_h + 54), label[1], -1)
    cv2.putText(canvas, label[0], (16, body_h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.78,
                (255, 255, 255), 2, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", canvas)
    if ok:
        buf.tofile(path)


if __name__ == "__main__":
    raise SystemExit(main())
