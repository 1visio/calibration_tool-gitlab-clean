"""生成合成棋盘格图像（内参、畸变、位姿全部已知），用于验证标定脚本的正确性。

渲染方式：对输出图像的每个像素做 undistortPoints 得到归一化坐标，再用平面单应
把它映射回棋盘纹理坐标，最后 remap。因此渲染过程严格遵循 OpenCV 的成像模型，
标定脚本应当能把真值参数还原出来。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

SQ = 30.0                     # mm
COLS, ROWS = 6, 5             # 内角点数
PX_PER_MM = 8.0
TEX_X0, TEX_X1 = -2 * SQ, (COLS + 1) * SQ
TEX_Y0, TEX_Y1 = -2 * SQ, (ROWS + 1) * SQ


def build_texture() -> np.ndarray:
    w = int((TEX_X1 - TEX_X0) * PX_PER_MM)
    h = int((TEX_Y1 - TEX_Y0) * PX_PER_MM)
    tex = np.full((h, w), 235, np.uint8)
    for i in range(0, COLS + 1):
        for j in range(0, ROWS + 1):
            if (i + j) % 2:
                continue
            x0 = int(((i - 1) * SQ - TEX_X0) * PX_PER_MM)
            x1 = int((i * SQ - TEX_X0) * PX_PER_MM)
            y0 = int(((j - 1) * SQ - TEX_Y0) * PX_PER_MM)
            y1 = int((j * SQ - TEX_Y0) * PX_PER_MM)
            tex[y0:y1, x0:x1] = 25
    return tex


def euler(rx: float, ry: float, rz: float) -> np.ndarray:
    rx, ry, rz = np.radians([rx, ry, rz])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def render(K, D, size, R, t, tex, rng) -> np.ndarray:
    width, height = size
    grid = np.stack(np.meshgrid(np.arange(width, dtype=np.float32),
                                np.arange(height, dtype=np.float32)), -1)
    normalized = cv2.undistortPoints(grid.reshape(-1, 1, 2), K, D).reshape(-1, 2)
    M = np.column_stack([R[:, 0], R[:, 1], t.reshape(3)])
    inv = np.linalg.inv(M)
    homo = np.column_stack([normalized, np.ones(len(normalized))])
    board = (inv @ homo.T).T
    with np.errstate(divide="ignore", invalid="ignore"):
        bx = board[:, 0] / board[:, 2]
        by = board[:, 1] / board[:, 2]
    behind = board[:, 2] <= 0
    map_x = ((bx - TEX_X0) * PX_PER_MM).astype(np.float32)
    map_y = ((by - TEX_Y0) * PX_PER_MM).astype(np.float32)
    map_x[behind] = -1
    map_y[behind] = -1
    image = cv2.remap(tex, map_x.reshape(height, width), map_y.reshape(height, width),
                      cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=180)
    image = cv2.GaussianBlur(image, (3, 3), 0.6)
    image = np.clip(image.astype(np.float32)
                    + rng.normal(0, 1.2, image.shape), 0, 255).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def object_points() -> np.ndarray:
    pts = np.zeros((COLS * ROWS, 3), np.float64)
    pts[:, :2] = np.mgrid[0:COLS, 0:ROWS].T.reshape(-1, 2) * SQ
    return pts


def pose_raw(K, rx, ry, rz, z, u, v) -> tuple[np.ndarray, np.ndarray]:
    R = euler(rx, ry, rz)
    centre = np.array([(COLS - 1) * SQ / 2.0, (ROWS - 1) * SQ / 2.0, 0.0])
    xn = (u - K[0, 2]) / K[0, 0]
    yn = (v - K[1, 2]) / K[1, 1]
    t = np.array([xn * z, yn * z, z]) - R @ centre
    return R, t.reshape(3, 1)


def pose_for(K, D, size, rx, ry, rz, z, fu, fv, margin=26.0):
    """把板子放在期望位置；若超出画面，先平移贴边，仍不行就退远，直到全部角点入画。"""
    width, height = size
    obj = object_points()
    for _ in range(40):
        u, v = fu * width, fv * height
        R, t = pose_raw(K, rx, ry, rz, z, u, v)
        rvec, _ = cv2.Rodrigues(R)
        pts = cv2.projectPoints(obj, rvec, t, K, D)[0].reshape(-1, 2)
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        if (x1 - x0) > width - 2 * margin or (y1 - y0) > height - 2 * margin:
            z *= 1.08
            continue
        dx = max(0.0, margin - x0) - max(0.0, x1 - (width - margin))
        dy = max(0.0, margin - y0) - max(0.0, y1 - (height - margin))
        if abs(dx) < 0.5 and abs(dy) < 0.5:
            return R, t
        fu += dx / width
        fv += dy / height
    return pose_raw(K, rx, ry, rz, z, fu * width, fv * height)


GOOD_POSES = [
    (35, 5, 10, 520, 0.22, 0.22), (-33, -8, -15, 540, 0.78, 0.22),
    (8, 34, 20, 530, 0.22, 0.78), (-10, -36, -25, 545, 0.78, 0.78),
    (40, 0, 0, 700, 0.50, 0.18), (-38, 0, 90, 690, 0.50, 0.82),
    (0, 38, 45, 710, 0.18, 0.50), (0, -40, -60, 700, 0.82, 0.50),
    (25, 25, 30, 460, 0.35, 0.35), (-27, 24, -35, 470, 0.65, 0.35),
    (26, -26, 70, 480, 0.35, 0.65), (-24, -25, -80, 465, 0.65, 0.65),
    (5, 5, 0, 830, 0.50, 0.50), (30, -15, 15, 820, 0.30, 0.50),
    (-30, 15, -15, 815, 0.70, 0.50), (15, 30, 55, 610, 0.50, 0.30),
    (-15, -30, -55, 600, 0.50, 0.70), (42, 12, 25, 560, 0.20, 0.50),
    (-42, -12, -25, 570, 0.80, 0.50), (12, 42, 35, 640, 0.50, 0.85),
    (20, -20, 5, 750, 0.25, 0.25), (-20, 20, -5, 760, 0.75, 0.75),
]

# 模仿 calib03 的退化采集：倾角小、深度几乎不变、集中在中央
POOR_POSES = [
    (4, 2, 180, 705, 0.48, 0.52), (3, -1, -90, 700, 0.40, 0.50),
    (2, 3, -95, 710, 0.55, 0.50), (-11, 3, 150, 700, 0.52, 0.46),
    (1, 11, 0, 695, 0.49, 0.55), (2, -2, -88, 665, 0.44, 0.47),
    (5, 1, -130, 705, 0.50, 0.43), (6, 1, -150, 706, 0.51, 0.41),
    (-1, -2, 97, 700, 0.47, 0.57), (-1, 3, 150, 700, 0.50, 0.50),
    (7, -8, 100, 670, 0.53, 0.56), (7, -9, 69, 660, 0.46, 0.57),
    (14, -4, -18, 662, 0.41, 0.56), (3, 1, -91, 715, 0.45, 0.49),
    (-6, 0, 145, 720, 0.53, 0.51), (6, -9, 91, 668, 0.49, 0.58),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--quality", choices=["good", "poor"], default="good")
    parser.add_argument("--width", type=int, default=1224)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--fx", type=float, default=2400.0)
    parser.add_argument("--fy", type=float, default=2394.0)
    parser.add_argument("--cx", type=float, default=605.0)
    parser.add_argument("--cy", type=float, default=518.0)
    parser.add_argument("--k1", type=float, default=-0.12)
    parser.add_argument("--k2", type=float, default=0.08)
    parser.add_argument("--p1", type=float, default=0.00045)
    parser.add_argument("--p2", type=float, default=-0.00032)
    parser.add_argument("--test-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260726)
    args = parser.parse_args()

    K = np.array([[args.fx, 0, args.cx], [0, args.fy, args.cy], [0, 0, 1]], float)
    D = np.array([args.k1, args.k2, args.p1, args.p2, 0.0])
    size = (args.width, args.height)
    tex = build_texture()
    rng = np.random.default_rng(args.seed)
    poses = GOOD_POSES if args.quality == "good" else POOR_POSES

    fit_dir = args.out / "fit"
    test_dir = args.out / "test"
    fit_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    truth = {"camera_matrix": K.tolist(), "dist_coeffs": D.tolist(),
             "image_size": list(size), "square_size_mm": SQ,
             "pattern": [COLS, ROWS], "poses": []}
    for index, (rx, ry, rz, z, fu, fv) in enumerate(poses, start=1):
        R, t = pose_for(K, D, size, rx, ry, rz, z, fu, fv)
        image = render(K, D, size, R, t, tex, rng)
        target = test_dir if index > len(poses) - args.test_count else fit_dir
        path = target / f"chess {index:03d}.png"
        cv2.imwrite(str(path), image)
        rvec, _ = cv2.Rodrigues(R)
        truth["poses"].append({"image": path.name, "set": target.name,
                               "rvec": rvec.reshape(3).tolist(),
                               "tvec": t.reshape(3).tolist()})
        print(f"渲染 {path}")
    (args.out / "ground_truth.yaml").write_text(
        __import__("yaml").safe_dump(truth, sort_keys=False, allow_unicode=True),
        encoding="utf-8")
    print(f"真值写入 {args.out / 'ground_truth.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
