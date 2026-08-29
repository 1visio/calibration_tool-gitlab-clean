import math
import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
LOCAL_COPY_DIR = Path(__file__).resolve().parent
for import_dir in (SCRIPT_DIR, LOCAL_COPY_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from circular_cone_local_parameterization import (  # noqa: E402
    angles_to_axis,
    legacy_to_local,
    local_to_legacy,
    transverse_basis,
)


class CircularConeLocalParameterizationTest(unittest.TestCase):
    def test_roundtrip_preserves_legacy_vector(self) -> None:
        p_ref = np.array([-80.0, 12.0, 690.0])
        cases = (
            (1.2, -0.7, np.array([-115.6, 1.7, 327.0]), math.radians(89.07)),
            (1.85, 0.01, np.array([-134.3, 14.6, 267.4]), math.radians(88.89)),
            (2.2, 2.4, np.array([80.0, -30.0, 100.0]), math.radians(75.0)),
        )
        for theta, phi, apex, alpha in cases:
            legacy = np.array([theta, phi, *apex, alpha], dtype=float)
            local = legacy_to_local(legacy, p_ref)
            recovered = local_to_legacy(local, p_ref)
            np.testing.assert_allclose(recovered, legacy, rtol=0.0, atol=2.0e-12)

    def test_basis_is_orthonormal_and_right_handed(self) -> None:
        for theta, phi in ((1.85, 0.01), (0.2, 0.7), (2.8, -2.0)):
            axis = angles_to_axis(theta, phi)
            e1, e2 = transverse_basis(axis)
            np.testing.assert_allclose(np.linalg.norm(e1), 1.0, atol=1.0e-14)
            np.testing.assert_allclose(np.linalg.norm(e2), 1.0, atol=1.0e-14)
            np.testing.assert_allclose(e1 @ axis, 0.0, atol=1.0e-14)
            np.testing.assert_allclose(e2 @ axis, 0.0, atol=1.0e-14)
            np.testing.assert_allclose(np.cross(e1, e2), axis, atol=1.0e-14)

    def test_axis_orientation_is_not_flipped(self) -> None:
        p_ref = np.array([0.0, 0.0, 700.0])
        legacy = np.array([1.8557, 0.0100, -115.6, 1.7, 327.0, math.radians(89.07)])
        recovered = local_to_legacy(legacy_to_local(legacy, p_ref), p_ref)
        d0 = angles_to_axis(*legacy[:2])
        d1 = angles_to_axis(*recovered[:2])
        self.assertGreater(float(d0 @ d1), 1.0 - 1.0e-14)


if __name__ == "__main__":
    unittest.main()
