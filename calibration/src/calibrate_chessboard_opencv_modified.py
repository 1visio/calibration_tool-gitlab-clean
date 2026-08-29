from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("必须为正整数")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("必须大于 0")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 OpenCV 棋盘格图像进行单目相机标定")
    parser.add_argument("--image-dir", type=Path, required=True, help="标定图像文件夹")
    parser.add_argument(
        "--pattern-cols",
        type=positive_int,
        default=6,
        help="横向内角点数量（默认：6）",
    )
    parser.add_argument(
        "--pattern-rows",
        type=positive_int,
        default=5,
        help="纵向内角点数量（默认：5）",
    )
    parser.add_argument(
        "--square-size-mm",
        type=positive_float,
        default=30.0,
        help="棋盘格单格边长，单位 mm（默认：30）",
    )
    parser.add_argument("--output", type=Path, required=True, help="标定结果输出目录")
    parser.add_argument(
        "--image-pattern",
        default=None,
        help="Optional filename glob, for example 'chess *.tif'",
    )
    parser.add_argument(
        "--max-images",
        type=positive_int,
        default=None,
        help="Use at most the first N sorted matching images",
    )
    parser.add_argument(
        "--reject-outliers",
        action="store_true",
        help="Reject robust reprojection-error outliers and recalibrate",
    )
    parser.add_argument(
        "--outlier-sigma",
        type=positive_float,
        default=3.0,
        help="Robust-sigma multiplier for reprojection outliers (default: 3.0)",
    )
    parser.add_argument(
        "--compare-distortion-models",
        action="store_true",
        help="Compare the standard 5-parameter model with CALIB_FIX_K3",
    )
    parser.add_argument(
        "--validation-images",
        type=positive_int,
        default=6,
        help="Use the next N matching images only for frozen-intrinsics validation",
    )
    return parser.parse_args()


def list_images(
    image_dir: Path,
    image_pattern: str | None = None,
    max_images: int | None = None,
) -> list[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"图像目录不存在：{image_dir}")
    if not image_dir.is_dir():
        raise NotADirectoryError(f"--image-dir 不是目录：{image_dir}")

    candidates = image_dir.glob(image_pattern) if image_pattern else image_dir.iterdir()
    images = sorted(
        (
            path
            for path in candidates
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: path.name.lower(),
    )
    if max_images is not None:
        images = images[:max_images]
    if not images:
        raise ValueError(f"图像目录中未找到支持的图像：{image_dir}")
    return images


def read_image(path: Path) -> np.ndarray | None:
    """Read an image while supporting non-ASCII paths on Windows."""
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
        if encoded.size == 0:
            return None
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    except (OSError, cv2.error):
        return None


def write_image(path: Path, image: np.ndarray) -> None:
    """Write an image while supporting non-ASCII paths on Windows."""
    success, encoded = cv2.imencode(path.suffix, image)
    if not success:
        raise RuntimeError(f"OpenCV 无法编码图像：{path}")
    try:
        encoded.tofile(path)
    except OSError as exc:
        raise OSError(f"无法写入图像：{path}") from exc


def detect_corners(
    gray: np.ndarray,
    pattern_size: tuple[int, int],
) -> tuple[bool, np.ndarray | None]:
    """Detect chessboard corners with an SB-first fallback strategy.

    ``findChessboardCornersSB`` already returns sub-pixel corner positions, so
    its result is returned directly.  Only the legacy
    ``findChessboardCorners`` result is refined with ``cornerSubPix``.
    """

    # Prefer the sector-based detector when it is available. Its returned
    # coordinates are already sub-pixel accurate, so do not refine them again.
    if hasattr(cv2, "findChessboardCornersSB"):
        try:
            found_sb, corners_sb = cv2.findChessboardCornersSB(
                gray,
                pattern_size,
                flags=cv2.CALIB_CB_NORMALIZE_IMAGE,
            )
            if found_sb and corners_sb is not None:
                return True, corners_sb.astype(np.float32)
        except cv2.error:
            # Fall back to the legacy detector below.
            pass

    # Legacy detector: obtain an initial integer/sub-pixel estimate first.
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if not found or corners is None:
        return False, None

    # The legacy detector benefits from a separate sub-pixel refinement step.
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )
    refined = cv2.cornerSubPix(
        gray,
        corners.astype(np.float32),
        (11, 11),
        (-1, -1),
        criteria,
    )
    return True, refined


def create_object_points(
    pattern_cols: int,
    pattern_rows: int,
    square_size_mm: float,
) -> np.ndarray:
    points = np.zeros((pattern_cols * pattern_rows, 3), dtype=np.float32)
    points[:, :2] = (
        np.mgrid[0:pattern_cols, 0:pattern_rows].T.reshape(-1, 2)
        * square_size_mm
    )
    return points


def calculate_reprojection_errors(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    rotation_vectors: tuple[np.ndarray, ...],
    translation_vectors: tuple[np.ndarray, ...],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> list[float]:
    errors: list[float] = []
    for object_point, observed, rvec, tvec in zip(
        object_points,
        image_points,
        rotation_vectors,
        translation_vectors,
        strict=True,
    ):
        projected, _ = cv2.projectPoints(
            object_point,
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs,
        )
        residual = observed.reshape(-1, 2) - projected.reshape(-1, 2)
        rmse = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
        errors.append(rmse)
    return errors


def find_reprojection_outliers(
    errors: list[float],
    sigma_multiplier: float,
) -> tuple[list[int], float | None]:
    if len(errors) < 4:
        return [], None
    values = np.asarray(errors, dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    robust_sigma = 1.4826 * mad
    if robust_sigma <= np.finfo(np.float64).eps:
        return [], None
    threshold = median + sigma_multiplier * robust_sigma
    return [index for index, error in enumerate(errors) if error > threshold], threshold


def write_failed_images(output_dir: Path, failed_paths: list[Path]) -> None:
    content = "".join(f"{path.name}\n" for path in failed_paths)
    (output_dir / "failed_images.txt").write_text(content, encoding="utf-8")


def write_error_csv(
    output_dir: Path,
    successful_paths: list[Path],
    errors: list[float],
) -> None:
    with (output_dir / "per_image_error.csv").open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["image", "reprojection_error_pixels"])
        for path, error in zip(successful_paths, errors, strict=True):
            writer.writerow([path.name, f"{error:.9f}"])


def write_extrinsics_csv(
    output_dir: Path,
    successful_paths: list[Path],
    rotation_vectors: tuple[np.ndarray, ...],
    translation_vectors: tuple[np.ndarray, ...],
) -> None:
    with (output_dir / "extrinsics.csv").open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "image",
                "rvec_x",
                "rvec_y",
                "rvec_z",
                "tvec_x_mm",
                "tvec_y_mm",
                "tvec_z_mm",
            ]
        )
        for path, rvec, tvec in zip(
            successful_paths,
            rotation_vectors,
            translation_vectors,
            strict=True,
        ):
            values = (*rvec.reshape(-1), *tvec.reshape(-1))
            writer.writerow([path.name, *(f"{value:.17g}" for value in values)])


def write_yaml(
    output_dir: Path,
    image_size: tuple[int, int],
    pattern_cols: int,
    pattern_rows: int,
    square_size_mm: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    mean_error: float,
) -> None:
    matrix_rows = [
        "  - [" + ", ".join(f"{value:.17g}" for value in row) + "]"
        for row in camera_matrix
    ]
    distortion_values = ", ".join(
        f"{value:.17g}" for value in dist_coeffs.reshape(-1)
    )
    content = "\n".join(
        [
            f"image_width: {image_size[0]}",
            f"image_height: {image_size[1]}",
            f"pattern_cols: {pattern_cols}",
            f"pattern_rows: {pattern_rows}",
            f"square_size_mm: {square_size_mm:.17g}",
            "camera_matrix:",
            *matrix_rows,
            f"dist_coeffs: [{distortion_values}]",
            f"mean_reprojection_error: {mean_error:.17g}",
            "",
        ]
    )
    (output_dir / "calibration_result.yaml").write_text(content, encoding="utf-8")


def write_camera_intrinsics_yaml(
    output_dir: Path,
    image_size: tuple[int, int],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    rms: float,
    image_paths: list[Path],
    detected_paths: set[Path],
    used_paths: set[Path],
    image_errors: dict[Path, float],
    outlier_threshold: float | None,
) -> None:
    matrix_rows = [
        "  - [" + ", ".join(f"{value:.17g}" for value in row) + "]"
        for row in camera_matrix
    ]
    distortion_values = ", ".join(
        f"{value:.17g}" for value in dist_coeffs.reshape(-1)
    )
    lines = [
        f"image_width: {image_size[0]}",
        f"image_height: {image_size[1]}",
        "camera_matrix:",
        *matrix_rows,
        f"dist_coeffs: [{distortion_values}]",
        f"RMS: {rms:.17g}",
        "outlier_threshold_pixels: "
        + ("null" if outlier_threshold is None else f"{outlier_threshold:.17g}"),
        "per_image_reprojection_errors:",
    ]
    for path in image_paths:
        escaped_name = path.name.replace('"', '\\"')
        detected = path in detected_paths
        used = path in used_paths
        error = image_errors.get(path)
        lines.extend(
            [
                f'  - image: "{escaped_name}"',
                f"    detection_success: {str(detected).lower()}",
                f"    used_for_calibration: {str(used).lower()}",
                "    reprojection_error_pixels: "
                + ("null" if error is None else f"{error:.17g}"),
            ]
        )
    lines.append("")
    (output_dir / "camera_intrinsics.yaml").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def write_calibration_report_csv(
    output_dir: Path,
    image_paths: list[Path],
    detected_paths: set[Path],
    used_paths: set[Path],
    image_errors: dict[Path, float],
) -> None:
    with (output_dir / "calibration_report.csv").open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "image",
                "detection_success",
                "used_for_calibration",
                "status",
                "reprojection_error_pixels",
            ]
        )
        for path in image_paths:
            detected = path in detected_paths
            used = path in used_paths
            if not detected:
                status = "detection_failed"
            elif not used:
                status = "reprojection_outlier"
            else:
                status = "used"
            error = image_errors.get(path)
            writer.writerow(
                [
                    path.name,
                    str(detected).lower(),
                    str(used).lower(),
                    status,
                    "" if error is None else f"{error:.17g}",
                ]
            )


def write_report(
    output_dir: Path,
    total_count: int,
    successful_paths: list[Path],
    failed_paths: list[Path],
    errors: list[float],
) -> None:
    mean_error = float(np.mean(errors))
    max_index = int(np.argmax(errors))
    report = "\n".join(
        [
            "OpenCV 单目棋盘格相机标定报告",
            "=" * 34,
            f"输入图像数量: {total_count}",
            f"使用图像数量: {len(successful_paths)}",
            f"失败图像数量: {len(failed_paths)}",
            f"平均重投影误差 (pixels): {mean_error:.9f}",
            f"最大单张误差 (pixels): {errors[max_index]:.9f}",
            f"最大误差图像: {successful_paths[max_index].name}",
            "",
        ]
    )
    report += (
        "extrinsics.csv \u5df2\u4fdd\u5b58\u6bcf\u5f20\u56fe\u50cf\u7684\u68cb\u76d8\u683c"
        "\u76f8\u5bf9\u4e8e\u76f8\u673a\u7684\u4f4d\u59ff\u3002\n"
    )
    (output_dir / "calibration_report.txt").write_text(report, encoding="utf-8-sig")


def save_undistorted_images(
    successful_paths: list[Path],
    output_dir: Path,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> None:
    for path in successful_paths:
        image = read_image(path)
        if image is None:
            raise RuntimeError(f"生成去畸变图时无法重新读取：{path}")
        undistorted = cv2.undistort(image, camera_matrix, dist_coeffs)
        write_image(output_dir / path.name, undistorted)


def save_reprojection_overlays(
    successful_paths: list[Path],
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    rotation_vectors: tuple[np.ndarray, ...],
    translation_vectors: tuple[np.ndarray, ...],
    output_dir: Path,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> None:
    for path, object_point, observed, rvec, tvec in zip(
        successful_paths,
        object_points,
        image_points,
        rotation_vectors,
        translation_vectors,
        strict=True,
    ):
        image = read_image(path)
        if image is None:
            raise RuntimeError(f"Unable to reread image for reprojection: {path}")
        projected, _ = cv2.projectPoints(
            object_point,
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs,
        )
        overlay = image.copy()
        for observed_point, projected_point in zip(
            observed.reshape(-1, 2),
            projected.reshape(-1, 2),
            strict=True,
        ):
            observed_xy = tuple(np.rint(observed_point).astype(int))
            projected_xy = tuple(np.rint(projected_point).astype(int))
            cv2.drawMarker(
                overlay,
                observed_xy,
                (0, 255, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=12,
                thickness=1,
            )
            cv2.circle(overlay, projected_xy, 3, (0, 0, 255), -1)
        cv2.putText(
            overlay,
            "Detected: green cross | Reprojected: red dot",
            (24, 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            "Detected: green cross | Reprojected: red dot",
            (24, 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        write_image(output_dir / f"{path.stem}_reprojection.png", overlay)


def calibrate(args: argparse.Namespace) -> None:
    image_paths = list_images(args.image_dir, args.image_pattern, args.max_images)
    output_dir = args.output
    detected_dir = output_dir / "detected"
    undistort_dir = output_dir / "undistort_check"
    reprojection_dir = output_dir / "reprojection"
    detected_dir.mkdir(parents=True, exist_ok=True)
    undistort_dir.mkdir(parents=True, exist_ok=True)
    reprojection_dir.mkdir(parents=True, exist_ok=True)

    pattern_size = (args.pattern_cols, args.pattern_rows)
    object_point_template = create_object_points(
        args.pattern_cols,
        args.pattern_rows,
        args.square_size_mm,
    )
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    successful_paths: list[Path] = []
    failed_paths: list[Path] = []
    expected_size: tuple[int, int] | None = None
    expected_size_path: Path | None = None

    for path in image_paths:
        image = read_image(path)
        if image is None:
            print(f"[失败] 无法读取：{path.name}", file=sys.stderr)
            failed_paths.append(path)
            continue

        image_size = (image.shape[1], image.shape[0])
        if expected_size is None:
            expected_size = image_size
            expected_size_path = path
        elif image_size != expected_size:
            raise ValueError(
                "图像尺寸不一致："
                f"{expected_size_path.name} 为 {expected_size[0]}x{expected_size[1]}，"
                f"{path.name} 为 {image_size[0]}x{image_size[1]}"
            )

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        try:
            found, corners = detect_corners(gray, pattern_size)
        except cv2.error as exc:
            print(f"[失败] 角点检测异常：{path.name} ({exc})", file=sys.stderr)
            failed_paths.append(path)
            continue

        if not found or corners is None:
            print(f"[失败] 未检测到完整棋盘格：{path.name}")
            failed_paths.append(path)
            continue

        object_points.append(object_point_template.copy())
        image_points.append(corners)
        successful_paths.append(path)

        overlay = image.copy()
        cv2.drawChessboardCorners(overlay, pattern_size, corners, True)
        write_image(detected_dir / path.name, overlay)
        print(f"[成功] {path.name}")

    write_failed_images(output_dir, failed_paths)
    if expected_size is None:
        raise RuntimeError("所有输入图像均无法读取")
    if not successful_paths:
        raise RuntimeError("没有图像成功检测到棋盘格内角点，无法进行标定")

    if args.compare_distortion_models:
        all_matching_paths = list_images(args.image_dir, args.image_pattern)
        fit_path_set = set(image_paths)
        validation_paths = [
            path for path in all_matching_paths if path not in fit_path_set
        ][: args.validation_images]
        if len(validation_paths) != args.validation_images:
            raise RuntimeError(
                f"Expected {args.validation_images} validation images, "
                f"found {len(validation_paths)}"
            )
        from distortion_model_comparison import run_distortion_model_comparison

        run_distortion_model_comparison(
            output_dir=output_dir,
            image_size=expected_size,
            pattern_size=pattern_size,
            object_point_template=object_point_template,
            selected_fit_paths=image_paths,
            successful_fit_paths=successful_paths,
            fit_object_points=object_points,
            fit_image_points=image_points,
            validation_paths=validation_paths,
            reject_outliers=args.reject_outliers,
            outlier_sigma=args.outlier_sigma,
            detect_corners=detect_corners,
            find_outliers=find_reprojection_outliers,
            read_image=read_image,
            write_image=write_image,
        )
        print(f"Distortion model comparison saved to: {output_dir.resolve()}")
        return

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        expected_size,
        None,
        None,
    )
    errors = calculate_reprojection_errors(
        object_points,
        image_points,
        tuple(rvecs),
        tuple(tvecs),
        camera_matrix,
        dist_coeffs,
    )
    detected_paths = list(successful_paths)
    image_errors = dict(zip(detected_paths, errors, strict=True))
    outlier_indices: list[int] = []
    outlier_threshold: float | None = None
    if args.reject_outliers:
        outlier_indices, outlier_threshold = find_reprojection_outliers(
            errors,
            args.outlier_sigma,
        )

    if outlier_indices:
        outlier_index_set = set(outlier_indices)
        keep_indices = [
            index for index in range(len(successful_paths)) if index not in outlier_index_set
        ]
        if len(keep_indices) < 3:
            raise RuntimeError("Outlier rejection left fewer than three calibration images")
        for index in outlier_indices:
            print(
                f"[OUTLIER] {successful_paths[index].name}: {errors[index]:.9f} pixels"
            )
        successful_paths = [successful_paths[index] for index in keep_indices]
        object_points = [object_points[index] for index in keep_indices]
        image_points = [image_points[index] for index in keep_indices]
        rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            object_points,
            image_points,
            expected_size,
            None,
            None,
        )
        errors = calculate_reprojection_errors(
            object_points,
            image_points,
            tuple(rvecs),
            tuple(tvecs),
            camera_matrix,
            dist_coeffs,
        )
        image_errors.update(zip(successful_paths, errors, strict=True))

    mean_error = float(np.mean(errors))
    detected_path_set = set(detected_paths)
    used_path_set = set(successful_paths)

    write_error_csv(output_dir, successful_paths, errors)
    write_extrinsics_csv(
        output_dir,
        successful_paths,
        tuple(rvecs),
        tuple(tvecs),
    )
    write_yaml(
        output_dir,
        expected_size,
        args.pattern_cols,
        args.pattern_rows,
        args.square_size_mm,
        camera_matrix,
        dist_coeffs,
        mean_error,
    )
    write_camera_intrinsics_yaml(
        output_dir,
        expected_size,
        camera_matrix,
        dist_coeffs,
        rms,
        image_paths,
        detected_path_set,
        used_path_set,
        image_errors,
        outlier_threshold,
    )
    write_calibration_report_csv(
        output_dir,
        image_paths,
        detected_path_set,
        used_path_set,
        image_errors,
    )
    np.save(output_dir / "camera_matrix.npy", camera_matrix)
    np.save(output_dir / "dist_coeffs.npy", dist_coeffs)
    save_undistorted_images(
        successful_paths,
        undistort_dir,
        camera_matrix,
        dist_coeffs,
    )
    save_reprojection_overlays(
        successful_paths,
        object_points,
        image_points,
        tuple(rvecs),
        tuple(tvecs),
        reprojection_dir,
        camera_matrix,
        dist_coeffs,
    )
    write_report(
        output_dir,
        len(image_paths),
        successful_paths,
        failed_paths,
        errors,
    )

    print(f"标定完成，结果已保存到：{output_dir.resolve()}")
    print(f"OpenCV RMS：{rms:.9f} pixels")
    print(f"平均重投影误差：{mean_error:.9f} pixels")


def main() -> int:
    args = parse_args()
    try:
        calibrate(args)
    except (OSError, ValueError, RuntimeError, cv2.error) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
