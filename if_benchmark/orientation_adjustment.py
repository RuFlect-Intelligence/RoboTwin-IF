"""Shared contract for instruction-parameterized rigid orientation changes.

The published action is relative to the frozen initial object orientation.
M4 only proves that the asset can be acquired for
``adjust_orientation_subject``; the requested delta is carried by the goal and
is executed/checkered in the exact M12/M15 scene.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


ORIENTATION_ADJUSTMENT_VERSION = 1
ORIENTATION_ADJUSTMENT_USE = "adjust_orientation_subject"
ORIENTATION_ADJUSTMENT_VERB = "adjust_orientation"
ORIENTATION_ADJUSTMENT_STATE = "orientation_adjusted"
ORIENTATION_REFERENCE = "initial_orientation"
SUPPORT_NORMAL_AXIS = "support_normal"
PUBLISHED_ANGLE_DEGREES = (-90, -60, -45, -30, 30, 45, 60, 90)


def orientation_delta_id(delta_deg: int) -> str:
    value = int(delta_deg)
    return f"delta_{'p' if value > 0 else 'm'}{abs(value)}_deg"


def normalize_orientation_adjustment_params(
    params: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Validate and canonicalize the public orientation action parameters."""

    row = dict(params or {})
    axis = str(row.get("orientation_axis") or "")
    reference = str(row.get("orientation_reference") or "")
    raw_delta = row.get("rotation_delta_deg")
    if axis != SUPPORT_NORMAL_AXIS:
        raise ValueError("orientation_axis must be support_normal")
    if reference != ORIENTATION_REFERENCE:
        raise ValueError("orientation_reference must be initial_orientation")
    if isinstance(raw_delta, bool):
        raise ValueError("rotation_delta_deg must be numeric")
    try:
        delta = int(raw_delta)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("rotation_delta_deg must be an integer") from exc
    if float(raw_delta) != float(delta):
        raise ValueError("rotation_delta_deg must be an integer")
    if delta not in PUBLISHED_ANGLE_DEGREES:
        raise ValueError(
            "rotation_delta_deg must be one of "
            f"{list(PUBLISHED_ANGLE_DEGREES)}"
        )
    unknown = sorted(
        set(row)
        - {
            "orientation_axis",
            "orientation_reference",
            "rotation_delta_deg",
        }
    )
    if unknown:
        raise ValueError(f"unsupported orientation parameters: {unknown}")
    return {
        "orientation_axis": SUPPORT_NORMAL_AXIS,
        "orientation_reference": ORIENTATION_REFERENCE,
        "rotation_delta_deg": delta,
    }


def normalize_quaternion_wxyz(value: Sequence[float]) -> list[float]:
    if len(value) != 4:
        raise ValueError("quaternion must contain four values")
    quat = [float(item) for item in value]
    norm = math.sqrt(sum(item * item for item in quat))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("quaternion must have finite non-zero norm")
    normalized = [item / norm for item in quat]
    # Canonical sign makes hashes and diagnostics deterministic.
    if normalized[0] < 0.0:
        normalized = [-item for item in normalized]
    return normalized


def quaternion_multiply_wxyz(
    left: Sequence[float], right: Sequence[float]
) -> list[float]:
    lw, lx, ly, lz = normalize_quaternion_wxyz(left)
    rw, rx, ry, rz = normalize_quaternion_wxyz(right)
    return normalize_quaternion_wxyz(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ]
    )


def target_quaternion_wxyz(
    initial_quaternion_wxyz: Sequence[float],
    params: Mapping[str, Any],
) -> list[float]:
    """Return ``R_world_support_normal(delta) * q_initial``."""

    normalized = normalize_orientation_adjustment_params(params)
    half = math.radians(float(normalized["rotation_delta_deg"])) * 0.5
    world_delta = [math.cos(half), 0.0, 0.0, math.sin(half)]
    return quaternion_multiply_wxyz(world_delta, initial_quaternion_wxyz)


__all__ = [
    "ORIENTATION_ADJUSTMENT_STATE",
    "ORIENTATION_ADJUSTMENT_USE",
    "ORIENTATION_ADJUSTMENT_VERB",
    "ORIENTATION_ADJUSTMENT_VERSION",
    "ORIENTATION_REFERENCE",
    "PUBLISHED_ANGLE_DEGREES",
    "SUPPORT_NORMAL_AXIS",
    "normalize_orientation_adjustment_params",
    "orientation_delta_id",
    "normalize_quaternion_wxyz",
    "quaternion_multiply_wxyz",
    "target_quaternion_wxyz",
]
