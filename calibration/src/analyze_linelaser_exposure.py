#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""评估不同曝光下棋盘格表面激光线的提取质量。"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='评估棋盘格表面激光线的最佳曝光。')
    p.add_argument('--input', required=True, type=Path, help='激光开启图像根目录')
    p.add_argument('--output', type=Path, default=Path('linelaser_exposure_result'))
    p.add_argument('--background-root', type=Path,
                   help='可选：同姿态同曝光的激光关闭图像根目录')
    p.add_argument('--cols', type=int, default=6, help='横向内角点数，默认6')
    p.add_argument('--rows', type=int, default=5, help='纵向内角点数，默认5')
    p.add_argument('--max-value', type=float, default=0,
                   help='Mono8填255，Mono12填4095；0为自动')
    p.add_argument('--scan-axis', choices=['auto', 'x', 'y'], default='auto',
                   help='x=激光近似水平逐列提取；y=近似竖直逐行提取')
    p.add_argument('--centroid-half-width', type=int, default=5,
                   help='灰度重心窗口半宽，默认5像素')
    p.add_argument('--min-snr', type=float, default=3.0,
                   help='最低信噪比，默认3')
    p.add_argument('--min-peak-ratio', type=float, default=0.015,
                   help='最低峰值占满量程比例，默认0.015')
    p.add_argument('--sat-ratio', type=float, default=0.98,
                   help='饱和判定比例，默认0.98')
    p.add_argument('--board-erode', type=int, default=4,
                   help='棋盘格掩膜向内收缩像素，默认4')
    p.add_argument('--local-bg-sigma', type=float, default=7.0,
                   help='无背景图时局部背景高斯尺度，默认7')
    p.add_argument('--ransac-residual', type=float, default=2.0,
                   help='直线RANSAC内点阈值，默认2像素')
    p.add_argument('--save-signal', action='store_true',
                   help='保存差分或局部背景扣除图')
    return p


def read_img(path: Path):
    try:
        return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_UNCHANGED)
    except Exception:
        return None


def write_img(path: Path, img: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or '.png', img)
    if not ok:
        raise OSError(f'无法编码图像：{path}')
    buf.tofile(str(path))


def gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)


def max_dn(g: np.ndarray, specified: float) -> float:
    if specified > 0:
        return specified
    if g.dtype == np.uint8:
        return 255.0
    if g.dtype == np.uint16:
        return 4095.0
    return max(float(np.max(g)), 1.0)


def u8(g: np.ndarray, maximum: float) -> np.ndarray:
    return np.clip(g.astype(np.float32) * 255.0 / maximum, 0, 255).astype(np.uint8)


def images(root: Path):
    return sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in EXTS)


def group_name(root: Path, path: Path) -> str:
    rel = path.relative_to(root)
    return rel.parts[0] if len(rel.parts) > 1 else 'root'


def exposure_us(group: str):
    m = re.search(r'(\d+(?:\.\d+)?)', group)
    return float(m.group(1)) if m else None


def paired(root: Path, rel: Path):
    exact = root / rel
    if exact.is_file():
        return exact
    folder = root / rel.parent
    if not folder.is_dir():
        return None
    found = [p for p in folder.iterdir()
             if p.is_file() and p.suffix.lower() in EXTS and p.stem.lower() == rel.stem.lower()]
    return sorted(found)[0] if found else None


def chessboard(g8: np.ndarray, pattern):
    found, corners = False, None
    if hasattr(cv2, 'findChessboardCornersSB'):
        try:
            found, corners = cv2.findChessboardCornersSB(
                g8, pattern, cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE)
        except cv2.error:
            pass
    if not found:
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(g8, pattern, flags)
        if found:
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3)
            corners = cv2.cornerSubPix(g8, corners, (11, 11), (-1, -1), criteria)
    return bool(found), corners


def board_mask(shape, corners, cols, rows, erode_px):
    pts = corners.reshape(-1, 2).astype(np.float32)
    grid = np.array([(x, y) for y in range(rows) for x in range(cols)], np.float32)
    H, _ = cv2.findHomography(grid, pts)
    if H is not None:
        outer = np.array([[[-1, -1], [cols, -1], [cols, rows], [-1, rows]]], np.float32)
        poly = cv2.perspectiveTransform(outer, H)[0]
    else:
        poly = cv2.convexHull(pts).reshape(-1, 2)
    h, w = shape
    poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
    poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)
    mask = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(mask, np.round(poly).astype(np.int32), 255)
    if erode_px > 0:
        k = 2 * erode_px + 1
        mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return mask, poly


def signal_image(on: np.ndarray, off: np.ndarray | None, sigma: float):
    onf = on.astype(np.float32)
    if off is not None:
        return np.maximum(onf - off.astype(np.float32), 0)
    smooth = cv2.GaussianBlur(onf, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return np.maximum(onf - smooth, 0)


def choose_axis(sig: np.ndarray, mask: np.ndarray):
    vals = sig[mask > 0]
    if vals.size == 0:
        return 'x'
    th = np.percentile(vals, 99)
    ys, xs = np.nonzero((sig >= th) & (mask > 0))
    if len(xs) < 20:
        return 'x'
    return 'x' if np.ptp(xs) >= np.ptp(ys) else 'y'


def fwhm(profile, peak, baseline):
    half = baseline + 0.5 * (float(profile[peak]) - baseline)
    left = peak
    while left > 0 and profile[left] >= half:
        left -= 1
    right = peak
    while right < len(profile) - 1 and profile[right] >= half:
        right += 1
    return float(right - left)


def centers(raw, sig, mask, axis, maximum, args):
    result = []
    count = sig.shape[1] if axis == 'x' else sig.shape[0]
    for scan in range(count):
        if axis == 'x':
            valid = np.flatnonzero(mask[:, scan] > 0)
            prof, raw_prof = sig[:, scan], raw[:, scan]
        else:
            valid = np.flatnonzero(mask[scan, :] > 0)
            prof, raw_prof = sig[scan, :], raw[scan, :]
        if len(valid) < 5:
            continue
        vals = prof[valid].astype(float)
        peak = int(valid[np.argmax(vals)])
        bg = float(np.median(vals))
        mad = float(np.median(np.abs(vals - bg)))
        noise = max(1.4826 * mad, 1e-6)
        peak_dn = float(prof[peak])
        snr = (peak_dn - bg) / noise
        if peak_dn < args.min_peak_ratio * maximum or snr < args.min_snr:
            continue
        lo = max(peak - args.centroid_half_width, int(valid[0]))
        hi = min(peak + args.centroid_half_width, int(valid[-1]))
        c = np.arange(lo, hi + 1, dtype=float)
        weights = np.maximum(prof[lo:hi + 1].astype(float) - bg, 0)
        if weights.sum() <= 0:
            continue
        center = float(np.dot(c, weights) / weights.sum())
        x, y = (float(scan), center) if axis == 'x' else (center, float(scan))
        result.append(dict(scan=scan, x=x, y=y, snr=snr,
                           peak_dn=peak_dn, fwhm=fwhm(prof, peak, bg),
                           saturated=float(raw_prof[peak] >= args.sat_ratio * maximum)))
    return result


def ransac(points, axis, threshold, iterations=300):
    if len(points) < 2:
        return np.zeros(len(points), bool), np.nan, np.nan
    xy = np.array([[p['x'], p['y']] for p in points], float)
    independent = xy[:, 0] if axis == 'x' else xy[:, 1]
    dependent = xy[:, 1] if axis == 'x' else xy[:, 0]
    rng = np.random.default_rng(12345)
    best = np.zeros(len(points), bool)
    for _ in range(iterations):
        i, j = rng.choice(len(points), 2, replace=False)
        dx = independent[j] - independent[i]
        if abs(dx) < 1e-9:
            continue
        a = (dependent[j] - dependent[i]) / dx
        b = dependent[i] - a * independent[i]
        mask = np.abs(dependent - (a * independent + b)) <= threshold
        if mask.sum() > best.sum():
            best = mask
    if best.sum() < 2:
        best[:] = True
    a, b = np.polyfit(independent[best], dependent[best], 1)
    residual = np.abs(dependent - (a * independent + b))
    best = residual <= threshold
    if best.sum() >= 2:
        a, b = np.polyfit(independent[best], dependent[best], 1)
    return best, float(a), float(b)


def longest_run(scans, possible):
    if not scans or possible <= 0:
        return 0.0
    s = sorted(set(int(v) for v in scans))
    longest = current = 1
    for a, b in zip(s[:-1], s[1:]):
        current = current + 1 if b == a + 1 else 1
        longest = max(longest, current)
    return longest / possible


def surface_classes(reference, mask, maximum):
    ref8 = u8(reference, maximum)
    pixels = ref8[mask > 0]
    classes = np.zeros(mask.shape, np.uint8)
    if len(pixels) == 0:
        return classes
    th, _ = cv2.threshold(pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    classes[(ref8 <= th) & (mask > 0)] = 1
    classes[(ref8 > th) & (mask > 0)] = 2
    return classes


def finite(values, operation='median'):
    a = np.asarray(values, float)
    a = a[np.isfinite(a)]
    if not len(a):
        return np.nan
    if operation == 'mean':
        return float(np.mean(a))
    if operation == 'p10':
        return float(np.percentile(a, 10))
    return float(np.median(a))


def score(coverage, continuity, snr, saturation, width, residual, black, white):
    cov = np.clip(coverage, 0, 1)
    cont = np.clip(continuity, 0, 1)
    sn = np.clip(snr / 12, 0, 1) if np.isfinite(snr) else 0
    sat = 1 - np.clip(saturation / 0.05, 0, 1)
    wid = math.exp(-max(0, width - 3) / 5) if np.isfinite(width) else 0
    res = math.exp(-residual / 1.5) if np.isfinite(residual) else 0
    bal = min(black, white) if np.isfinite(black) and np.isfinite(white) else cov
    return 100 * (0.30*cov + 0.18*cont + 0.15*sn + 0.15*sat +
                  0.08*wid + 0.09*res + 0.05*np.clip(bal, 0, 1))


def save_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def main():
    args = parser().parse_args()
    root = args.input.resolve()
    out = args.output.resolve()
    bg_root = args.background_root.resolve() if args.background_root else None
    if not root.is_dir():
        print(f'[错误] 输入目录不存在：{root}', file=sys.stderr); return 1
    if bg_root and not bg_root.is_dir():
        print(f'[错误] 背景目录不存在：{bg_root}', file=sys.stderr); return 1
    files = images(root)
    if not files:
        print('[错误] 未找到图像', file=sys.stderr); return 1

    rows = []
    for idx, path in enumerate(files, 1):
        rel = path.relative_to(root)
        group = group_name(root, path)
        img = read_img(path)
        if img is None:
            continue
        on = gray(img)
        maximum = max_dn(on, args.max_value)

        off = None
        bg_path = paired(bg_root, rel) if bg_root else None
        if bg_path:
            bg_img = read_img(bg_path)
            if bg_img is not None:
                off = gray(bg_img)
                if off.shape != on.shape:
                    off = None

        reference = off if off is not None else on
        found, corners = chessboard(u8(reference, maximum), (args.cols, args.rows))
        base = dict(group=group, exposure_us=exposure_us(group), filename=str(rel),
                    background_found=int(off is not None), board_detected=int(found))
        if not found:
            base.update(scan_axis='', possible_scanlines=0, raw_center_count=0,
                        valid_center_count=0, coverage=0.0, longest_run_ratio=0.0,
                        median_snr=np.nan, p10_snr=np.nan, median_fwhm_px=np.nan,
                        saturated_point_ratio=np.nan, line_residual_rms_px=np.nan,
                        black_coverage=np.nan, white_coverage=np.nan, quality_score=0.0)
            rows.append(base)
            print(f'[{idx}/{len(files)}] {group} 棋盘格检测失败：{rel.name}')
            continue

        mask, poly = board_mask(on.shape, corners, args.cols, args.rows, args.board_erode)
        sig = signal_image(on, off, args.local_bg_sigma) * (mask > 0)
        axis = choose_axis(sig, mask) if args.scan_axis == 'auto' else args.scan_axis
        raw_pts = centers(on, sig, mask, axis, maximum, args)
        inliers, a, b = ransac(raw_pts, axis, args.ransac_residual)
        valid = [p for p, keep in zip(raw_pts, inliers) if keep]
        possible = int(np.sum(np.any(mask > 0, axis=0 if axis == 'x' else 1)))
        coverage = len(valid) / possible if possible else 0.0
        continuity = longest_run([p['scan'] for p in valid], possible)

        classes = surface_classes(reference, mask, maximum)
        # 用拟合激光线在每个扫描位置的预测交点判断其落在黑格还是白格，
        # 这样即使某一列/行提取失败，也能正确统计该表面类别的理论可提取数量。
        black_possible = white_possible = 0
        if np.isfinite(a) and np.isfinite(b):
            if axis == 'x':
                for s in range(mask.shape[1]):
                    y_pred = int(round(a * s + b))
                    if 0 <= y_pred < mask.shape[0] and mask[y_pred, s] > 0:
                        white_possible += classes[y_pred, s] == 2
                        black_possible += classes[y_pred, s] == 1
            else:
                for s in range(mask.shape[0]):
                    x_pred = int(round(a * s + b))
                    if 0 <= x_pred < mask.shape[1] and mask[s, x_pred] > 0:
                        white_possible += classes[s, x_pred] == 2
                        black_possible += classes[s, x_pred] == 1
        black_valid = white_valid = 0
        for p in valid:
            x = int(np.clip(round(p['x']), 0, mask.shape[1]-1))
            y = int(np.clip(round(p['y']), 0, mask.shape[0]-1))
            black_valid += classes[y, x] == 1; white_valid += classes[y, x] == 2
        black_cov = black_valid / black_possible if black_possible else np.nan
        white_cov = white_valid / white_possible if white_possible else np.nan

        med_snr = finite([p['snr'] for p in valid])
        p10_snr = finite([p['snr'] for p in valid], 'p10')
        med_width = finite([p['fwhm'] for p in valid])
        sat = finite([p['saturated'] for p in valid], 'mean')
        if valid:
            xy = np.array([[p['x'], p['y']] for p in valid])
            residuals = xy[:, 1] - (a*xy[:, 0]+b) if axis == 'x' else xy[:, 0] - (a*xy[:, 1]+b)
            rms = float(np.sqrt(np.mean(residuals**2)))
        else:
            rms = np.nan
        q = score(coverage, continuity, med_snr, sat, med_width, rms, black_cov, white_cov)
        base.update(scan_axis=axis, possible_scanlines=possible,
                    raw_center_count=len(raw_pts), valid_center_count=len(valid),
                    coverage=coverage, longest_run_ratio=continuity,
                    median_snr=med_snr, p10_snr=p10_snr,
                    median_fwhm_px=med_width, saturated_point_ratio=sat,
                    line_residual_rms_px=rms, black_coverage=black_cov,
                    white_coverage=white_cov, quality_score=q)
        rows.append(base)

        vis = cv2.cvtColor(u8(on, maximum), cv2.COLOR_GRAY2BGR)
        cv2.polylines(vis, [np.round(poly).astype(np.int32)], True, (0,255,255), 2)
        for p, keep in zip(raw_pts, inliers):
            cv2.circle(vis, (round(p['x']), round(p['y'])), 1,
                       (0,255,0) if keep else (0,0,255), -1)
        cv2.putText(vis, f'{group} cov={coverage*100:.1f}% sat={sat*100:.2f}% score={q:.1f}',
                    (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        write_img(out/'annotated'/rel.with_suffix('.png'), vis)
        if args.save_signal:
            vals = sig[mask > 0]
            high = max(float(np.percentile(vals, 99.7)), 1.0) if len(vals) else 1.0
            write_img(out/'signal'/rel.with_suffix('.png'), np.clip(sig*255/high,0,255).astype(np.uint8))
        print(f'[{idx}/{len(files)}] {group}: 覆盖={coverage*100:.1f}% 黑={black_cov*100:.1f}% 白={white_cov*100:.1f}% 饱和={sat*100:.2f}% 得分={q:.1f}')

    fields = ['group','exposure_us','filename','background_found','board_detected',
              'scan_axis','possible_scanlines','raw_center_count','valid_center_count',
              'coverage','longest_run_ratio','median_snr','p10_snr','median_fwhm_px',
              'saturated_point_ratio','line_residual_rms_px','black_coverage',
              'white_coverage','quality_score']
    save_csv(out/'image_metrics.csv', rows, fields)

    grouped = defaultdict(list)
    for r in rows:
        grouped[r['group']].append(r)
    summaries = []
    for group, rs in grouped.items():
        valid = [r for r in rs if r['board_detected']] or rs
        s = dict(group=group, exposure_us=exposure_us(group), image_count=len(rs),
                 board_detection_rate=finite([r['board_detected'] for r in rs], 'mean'),
                 background_pair_rate=finite([r['background_found'] for r in rs], 'mean'),
                 median_coverage=finite([r['coverage'] for r in valid]),
                 worst_coverage=finite([r['coverage'] for r in valid], 'p10'),
                 median_longest_run_ratio=finite([r['longest_run_ratio'] for r in valid]),
                 median_snr=finite([r['median_snr'] for r in valid]),
                 median_p10_snr=finite([r['p10_snr'] for r in valid]),
                 median_fwhm_px=finite([r['median_fwhm_px'] for r in valid]),
                 mean_saturated_point_ratio=finite([r['saturated_point_ratio'] for r in valid], 'mean'),
                 median_line_residual_rms_px=finite([r['line_residual_rms_px'] for r in valid]),
                 median_black_coverage=finite([r['black_coverage'] for r in valid]),
                 median_white_coverage=finite([r['white_coverage'] for r in valid]),
                 median_quality_score=finite([r['quality_score'] for r in valid]),
                 worst_quality_score=finite([r['quality_score'] for r in valid], 'p10'))
        summaries.append(s)
    summaries.sort(key=lambda r: float('inf') if r['exposure_us'] is None else r['exposure_us'])
    sfields = list(summaries[0].keys())
    save_csv(out/'exposure_summary.csv', summaries, sfields)
    ranked = sorted(summaries, key=lambda r: -r['median_quality_score'])
    with (out/'recommendation.txt').open('w', encoding='utf-8') as f:
        f.write('曝光候选排序（综合分仅用于同一数据集内比较）\n\n')
        for i, r in enumerate(ranked, 1):
            f.write(f"{i}. {r['group']}: score={r['median_quality_score']:.2f}, "
                    f"coverage={r['median_coverage']*100:.1f}%, "
                    f"black={r['median_black_coverage']*100:.1f}%, "
                    f"white={r['median_white_coverage']*100:.1f}%, "
                    f"sat={r['mean_saturated_point_ratio']*100:.2f}%, "
                    f"FWHM={r['median_fwhm_px']:.2f}px\n")
    print(f'\n结果已保存到：{out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
