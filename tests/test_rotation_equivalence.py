import numpy as np

from scripts.rotation_equivalence_test import (
    compare_points,
    restore_rotated_points,
    rotate_image,
)


def test_restore_clockwise_points_to_original_coordinates() -> None:
    image = np.arange(3 * 5, dtype=np.uint8).reshape(3, 5)
    original = np.array([[1.25, 0.0], [3.75, 2.0]], dtype=np.float64)
    rotated = np.column_stack(
        (image.shape[0] - 1.0 - original[:, 1], original[:, 0])
    )

    restored = restore_rotated_points(rotated, image.shape, "clockwise")

    np.testing.assert_allclose(restored, original, atol=0.0)
    np.testing.assert_array_equal(
        rotate_image(image, "clockwise"), np.rot90(image, k=-1)
    )


def test_restore_counterclockwise_points_to_original_coordinates() -> None:
    image = np.arange(3 * 5, dtype=np.uint8).reshape(3, 5)
    original = np.array([[1.25, 0.0], [3.75, 2.0]], dtype=np.float64)
    rotated = np.column_stack(
        (original[:, 1], image.shape[1] - 1.0 - original[:, 0])
    )

    restored = restore_rotated_points(rotated, image.shape, "counterclockwise")

    np.testing.assert_allclose(restored, original, atol=0.0)
    np.testing.assert_array_equal(
        rotate_image(image, "counterclockwise"), np.rot90(image, k=1)
    )


def test_compare_points_pairs_by_scanline_instead_of_array_order() -> None:
    native = np.array([[10.2, 2.0], [10.1, 1.0]], dtype=np.float64)
    restored = np.array([[10.11, 1.0], [10.18, 2.0]], dtype=np.float64)

    comparison = compare_points(native, restored, "clockwise")

    assert [row["v"] for row in comparison.rows] == [1, 2]
    np.testing.assert_allclose(
        [row["delta_u_px"] for row in comparison.rows],
        [0.01, -0.02],
        atol=1.0e-12,
    )
    assert comparison.only_native_v == []
    assert comparison.only_rotated_v == []
