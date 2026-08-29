"""等价的局部 Circular Cone 参数化。

The production cone remains represented by ``axis + apex + alpha``.  This
module only supplies a bijective coordinate change around a fixed, data-derived
reference point ``P_ref``; it never changes the cone equation or intersection
rule.

Local coordinates are

``[theta_axis, phi_axis, c1, c2, rho_ref, q]``

where ``q = cot(alpha)`` and ``rho_ref`` is the signed cone radius at the
plane perpendicular to the axis and passing through the axis-line projection
associated with ``P_ref``.  ``c1,c2`` are that projection's transverse
coordinates relative to ``P_ref``.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np


EPS = 1.0e-14


def normalize_axis(axis: Sequence[float]) -> np.ndarray:
    value = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(value).all() or norm <= EPS:
        raise ValueError("axis must be a finite non-zero 3-vector")
    return value / norm


def angles_to_axis(theta_axis: float, phi_axis: float) -> np.ndarray:
    theta = float(theta_axis)
    phi = float(phi_axis)
    axis = np.array(
        [math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta)],
        dtype=np.float64,
    )
    return normalize_axis(axis)


def axis_to_angles(axis: Sequence[float]) -> tuple[float, float]:
    value = normalize_axis(axis)
    theta = math.acos(float(np.clip(value[2], -1.0, 1.0)))
    phi = math.atan2(float(value[1]), float(value[0]))
    return theta, phi


def transverse_basis(axis: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic right-handed (e1,e2,axis) basis.

    The camera Z direction is preferred as the reference vector except close
    to parallel, where camera Y is used.  This branch is deterministic and is
    evaluated identically by both conversion directions.
    """
    d = normalize_axis(axis)
    reference = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(reference @ d)) > 0.95:
        reference = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    e1 = reference - float(reference @ d) * d
    e1 /= max(float(np.linalg.norm(e1)), EPS)
    e2 = np.cross(d, e1)
    e2 /= max(float(np.linalg.norm(e2)), EPS)
    return e1, e2


def _reference_point(reference: Sequence[float]) -> np.ndarray:
    point = np.asarray(reference, dtype=np.float64).reshape(3)
    if not np.isfinite(point).all():
        raise ValueError("P_ref must be finite")
    return point


def legacy_to_local(
    legacy_theta: Sequence[float],
    p_ref: Sequence[float],
) -> np.ndarray:
    """Convert ``[theta,phi,A_x,A_y,A_z,alpha]`` to local coordinates.

    ``alpha`` is in radians and all spatial values are millimetres.  The axis
    orientation is retained exactly; no sign/nappe canonicalization is done.
    """
    theta = np.asarray(legacy_theta, dtype=np.float64).reshape(6)
    if not np.isfinite(theta).all():
        raise ValueError("legacy parameters must be finite")
    axis = angles_to_axis(float(theta[0]), float(theta[1]))
    apex = theta[2:5]
    alpha = float(theta[5])
    if not 0.0 < alpha < math.pi / 2.0:
        raise ValueError("alpha must be in (0, pi/2)")
    e1, e2 = transverse_basis(axis)
    reference = _reference_point(p_ref)

    # C_ref is the axis-line intersection with the plane normal to d through
    # P_ref.  Its transverse offset from P_ref has exactly two coordinates.
    s_ref = float((reference - apex) @ axis)
    c_ref = apex + s_ref * axis
    offset = c_ref - reference
    q = 1.0 / math.tan(alpha)
    rho_ref = s_ref * math.tan(alpha)
    return np.array(
        [theta[0], theta[1], float(offset @ e1), float(offset @ e2), rho_ref, q],
        dtype=np.float64,
    )


def local_to_legacy(
    local_theta: Sequence[float],
    p_ref: Sequence[float],
) -> np.ndarray:
    """Convert local coordinates back to the formal legacy vector."""
    local = np.asarray(local_theta, dtype=np.float64).reshape(6)
    if not np.isfinite(local).all():
        raise ValueError("local parameters must be finite")
    theta_axis, phi_axis, c1, c2, rho_ref, q = local
    if q <= 0.0:
        raise ValueError("q=cot(alpha) must be positive for the physical cone")
    axis = angles_to_axis(float(theta_axis), float(phi_axis))
    e1, e2 = transverse_basis(axis)
    reference = _reference_point(p_ref)
    c_ref = reference + c1 * e1 + c2 * e2
    s_ref = rho_ref * q
    apex = c_ref - s_ref * axis
    alpha = math.atan2(1.0, float(q))
    return np.array(
        [theta_axis, phi_axis, apex[0], apex[1], apex[2], alpha],
        dtype=np.float64,
    )


def legacy_model_to_theta(model: Mapping[str, object]) -> np.ndarray:
    """Map a production model mapping to the formal six-vector."""
    axis = normalize_axis(model["axis_unit_camera"])  # type: ignore[index]
    theta, phi = axis_to_angles(axis)
    apex = np.asarray(model["apex_camera_mm"], dtype=np.float64).reshape(3)  # type: ignore[index]
    alpha = math.radians(float(model["half_apex_angle_deg"]))  # type: ignore[index]
    return np.array([theta, phi, *apex, alpha], dtype=np.float64)


def theta_to_model(theta: Sequence[float], z_range_mm: Sequence[float] | None = None) -> dict[str, object]:
    """Map a formal six-vector to a runtime Circular Cone mapping."""
    value = np.asarray(theta, dtype=np.float64).reshape(6)
    axis = angles_to_axis(float(value[0]), float(value[1]))
    alpha = float(value[5])
    if not 0.0 < alpha < math.pi / 2.0:
        raise ValueError("alpha must be in (0, pi/2)")
    result: dict[str, object] = {
        "model_type": "circular_cone",
        "description": "equivalent local-parameterization diagnostic mapping",
        "axis_unit_camera": axis.tolist(),
        "apex_camera_mm": value[2:5].tolist(),
        "half_apex_angle_deg": math.degrees(alpha),
    }
    if z_range_mm is not None:
        result["z_valid_range_mm"] = [float(z_range_mm[0]), float(z_range_mm[1])]
    return result

