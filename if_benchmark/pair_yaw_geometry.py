"""Canonical-pose geometry and relative-yaw domains for M5 pair validation.

The solver deliberately proves only a geometry proxy.  Terminal-settle reports
remain pointwise simulation evidence at the selected maximum-margin witness.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Callable


YAW_SOLVER_POLICY_VERSION = "m5_relative_yaw_v2_connected_components"
DEFAULT_ANGLE_STEP_DEG = 2.0
DEFAULT_BOUNDARY_TOLERANCE_DEG = 0.5
DEFAULT_BAND_EROSION_DEG = 2.0
DEFAULT_MIN_CONTINUOUS_BAND_DEG = 4.0


def yaw_solver_policy_hash() -> str:
    payload = {
        "version": YAW_SOLVER_POLICY_VERSION,
        "angle_step_deg": DEFAULT_ANGLE_STEP_DEG,
        "boundary_tolerance_deg": DEFAULT_BOUNDARY_TOLERANCE_DEG,
        "band_erosion_deg": DEFAULT_BAND_EROSION_DEG,
        "min_continuous_band_deg": DEFAULT_MIN_CONTINUOUS_BAND_DEG,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def pose_aware_box_geometry(endpoint: dict[str, Any], canonical_pose: dict[str, Any]) -> dict[str, Any] | None:
    """Project the metadata box after applying the canonical q0 orientation."""

    extents = ((endpoint.get("metadata") or {}).get("scaled_extents") or [])
    if len(extents) != 3:
        return None
    try:
        ex, ey, ez = (float(value) for value in extents)
        q0 = [float(value) for value in canonical_pose["q0_wxyz"]]
    except (KeyError, TypeError, ValueError):
        return None
    if min(ex, ey, ez) <= 0:
        return None

    rotation = quaternion_matrix_wxyz(q0)
    points: list[list[float]] = []
    for sx in (-0.5, 0.5):
        for sy in (-0.5, 0.5):
            for sz in (-0.5, 0.5):
                local = [sx * ex, sy * ey, sz * ez]
                points.append(
                    [
                        sum(rotation[row][column] * local[column] for column in range(3))
                        for row in range(3)
                    ]
                )
    min_xyz = [min(point[axis] for point in points) for axis in range(3)]
    max_xyz = [max(point[axis] for point in points) for axis in range(3)]
    footprint_points = [[point[0], point[1]] for point in points]
    x = max_xyz[0] - min_xyz[0]
    y = max_xyz[1] - min_xyz[1]
    return {
        "x": clean_float(x),
        "y": clean_float(y),
        "area": clean_float(x * y),
        "height": clean_float(max_xyz[2] - min_xyz[2]),
        "points_xy": [[clean_float(value) for value in point] for point in footprint_points],
        "source": "canonical_q0_transformed_scaled_extents",
    }


def solve_relative_yaw_domain(
    relation: str,
    subject_geometry: dict[str, Any],
    receiver_geometry: dict[str, Any],
    *,
    usable_factor: float,
    edge_clearance_m: float,
    receiver_support_proxy: dict[str, Any] | None = None,
    period_deg: float = 360.0,
    angle_step_deg: float = DEFAULT_ANGLE_STEP_DEG,
    boundary_tolerance_deg: float = DEFAULT_BOUNDARY_TOLERANCE_DEG,
    erosion_deg: float = DEFAULT_BAND_EROSION_DEG,
    min_continuous_band_deg: float = DEFAULT_MIN_CONTINUOUS_BAND_DEG,
) -> dict[str, Any]:
    """Return conservative continuous relative-yaw bands and point witnesses."""

    period = float(period_deg)
    if period <= 0:
        raise ValueError("relative yaw period must be positive")
    subject_points = _center_points(subject_geometry.get("points_xy") or [])
    if not subject_points:
        raise ValueError("subject geometry has no footprint points")

    receiver_outer_x = float(receiver_geometry["x"])
    receiver_outer_y = float(receiver_geometry["y"])
    support_proxy = receiver_support_proxy or {}
    min_half = _positive_float(support_proxy.get("min_half_extent_xy"))
    if relation == "stack" and min_half is not None:
        usable_x = usable_y = 2.0 * min_half
        geometry_source = "grounded_support_proxy"
        sampling_eligible = True
    else:
        usable_x = receiver_outer_x * float(usable_factor)
        usable_y = receiver_outer_y * float(usable_factor)
        geometry_source = "canonical_aabb_fraction_proxy"
        sampling_eligible = False

    margin_at = _margin_function(
        subject_points,
        usable_x,
        usable_y,
        float(edge_clearance_m),
    )
    outer_margin_at = _margin_function(subject_points, receiver_outer_x, receiver_outer_y, 0.0)
    raw_bands, sampled = _solve_nonnegative_bands(
        margin_at,
        period_deg=period,
        angle_step_deg=float(angle_step_deg),
        boundary_tolerance_deg=float(boundary_tolerance_deg),
    )
    best_delta, best_margin = max(sampled, key=lambda item: (item[1], -item[0]))
    outer_sampled = _sample_margin(outer_margin_at, period, float(angle_step_deg))
    outer_best_delta, outer_best_margin = max(outer_sampled, key=lambda item: (item[1], -item[0]))

    bands: list[dict[str, Any]] = []
    for component_index, (raw_start, raw_end, full_period) in enumerate(raw_bands):
        component_id = f"component_{component_index}"
        start = raw_start
        end = raw_end
        if not full_period:
            start += float(erosion_deg)
            end -= float(erosion_deg)
        width = max(0.0, end - start)
        witness_delta, witness_margin = _max_margin_in_unwrapped_interval(
            margin_at,
            start if width > 0 else raw_start,
            end if width > 0 else raw_end,
            period,
            float(angle_step_deg),
        )
        continuous = full_period or width >= float(min_continuous_band_deg)
        if not continuous:
            start = witness_delta
            end = witness_delta
        segments = _split_unwrapped_interval(start, end, period) if continuous else [(witness_delta % period, witness_delta % period)]
        for segment_start, segment_end in segments:
            bands.append(
                {
                    "band_id": f"band_{len(bands)}",
                    "connected_component_id": component_id,
                    "start_deg": clean_float(segment_start),
                    "end_deg": clean_float(segment_end),
                    "interval": "half_open" if continuous else "point",
                    "continuous": bool(continuous),
                    "geometry_source": geometry_source,
                    "max_margin_m": clean_float(witness_margin),
                    "witness_delta_deg": clean_float(witness_delta % period),
                    "witness_status": "pending_terminal",
                    "sampling_eligible": bool(sampling_eligible and continuous),
                }
            )

    fallback_witness = None
    if not bands:
        fallback_witness = {
            "band_id": "fallback_witness",
            "connected_component_id": "fallback_witness",
            "start_deg": clean_float(best_delta),
            "end_deg": clean_float(best_delta),
            "interval": "point",
            "continuous": False,
            "geometry_source": geometry_source,
            "max_margin_m": clean_float(best_margin),
            "witness_delta_deg": clean_float(best_delta),
            "witness_status": "pending_terminal",
            "sampling_eligible": False,
        }

    return {
        "convention": "subject_yaw_minus_receiver_yaw",
        "period_deg": clean_float(period),
        "solver_policy_hash": yaw_solver_policy_hash(),
        "geometry_source": geometry_source,
        "usable_region": {
            "x": clean_float(usable_x),
            "y": clean_float(usable_y),
            "factor": clean_float(float(usable_factor)),
        },
        "edge_clearance_m": clean_float(float(edge_clearance_m)),
        "angle_step_deg": clean_float(float(angle_step_deg)),
        "boundary_tolerance_deg": clean_float(float(boundary_tolerance_deg)),
        "band_erosion_deg": clean_float(float(erosion_deg)),
        "geometric_bands": bands,
        "fallback_witness": fallback_witness,
        "best_delta_deg": clean_float(best_delta),
        "best_margin_m": clean_float(best_margin),
        "outer_best_delta_deg": clean_float(outer_best_delta),
        "outer_best_margin_m": clean_float(outer_best_margin),
        "outer_impossible": bool(outer_best_margin < 0.0),
    }


def quaternion_matrix_wxyz(q: list[float]) -> list[list[float]]:
    if len(q) != 4:
        raise ValueError("quaternion must contain four values")
    w, x, y, z = (float(value) for value in q)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1e-12:
        raise ValueError("quaternion norm must be positive")
    w, x, y, z = (value / norm for value in (w, x, y, z))
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def _margin_function(
    subject_points: list[list[float]],
    receiver_x: float,
    receiver_y: float,
    edge_clearance_m: float,
) -> Callable[[float], float]:
    half_x = float(receiver_x) / 2.0
    half_y = float(receiver_y) / 2.0
    subject_half_x = max(abs(point[0]) for point in subject_points)
    subject_half_y = max(abs(point[1]) for point in subject_points)

    def margin(delta_deg: float) -> float:
        radians = math.radians(float(delta_deg))
        cosine = abs(math.cos(radians))
        sine = abs(math.sin(radians))
        # The canonical footprint is deliberately represented by its
        # conservative oriented rectangle.  This keeps the continuous-domain
        # solver cheap enough for the full pair matrix.
        max_x = cosine * subject_half_x + sine * subject_half_y
        max_y = sine * subject_half_x + cosine * subject_half_y
        return min(half_x - max_x, half_y - max_y) - float(edge_clearance_m)

    return margin


def _solve_nonnegative_bands(
    margin_at: Callable[[float], float],
    *,
    period_deg: float,
    angle_step_deg: float,
    boundary_tolerance_deg: float,
) -> tuple[list[tuple[float, float, bool]], list[tuple[float, float]]]:
    sampled = _sample_margin(margin_at, period_deg, angle_step_deg)
    valid = [margin >= 0.0 for _, margin in sampled]
    if all(valid):
        return [(0.0, period_deg, True)], sampled
    if not any(valid):
        return [], sampled

    count = len(sampled)
    first_invalid = next(index for index, item in enumerate(valid) if not item)
    ordered_indices = [(first_invalid + 1 + offset) % count for offset in range(count)]
    runs: list[list[int]] = []
    current: list[int] = []
    for index in ordered_indices:
        if valid[index]:
            current.append(index)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)

    step = period_deg / count
    bands: list[tuple[float, float, bool]] = []
    for run in runs:
        unwrapped = [float(index) * step for index in run]
        for index in range(1, len(unwrapped)):
            if unwrapped[index] < unwrapped[index - 1]:
                unwrapped[index:] = [value + period_deg for value in unwrapped[index:]]
        first_valid = unwrapped[0]
        last_valid = unwrapped[-1]
        start = _bisect_zero(
            margin_at,
            first_valid - step,
            first_valid,
            period_deg,
            boundary_tolerance_deg,
            seek_first_valid=True,
        )
        end = _bisect_zero(
            margin_at,
            last_valid,
            last_valid + step,
            period_deg,
            boundary_tolerance_deg,
            seek_first_valid=False,
        )
        bands.append((start, end, False))
    return bands, sampled


def _bisect_zero(
    margin_at: Callable[[float], float],
    low: float,
    high: float,
    period_deg: float,
    tolerance_deg: float,
    *,
    seek_first_valid: bool,
) -> float:
    while high - low > tolerance_deg:
        midpoint = (low + high) / 2.0
        valid = margin_at(midpoint % period_deg) >= 0.0
        if seek_first_valid:
            if valid:
                high = midpoint
            else:
                low = midpoint
        else:
            if valid:
                low = midpoint
            else:
                high = midpoint
    return high if seek_first_valid else low


def _sample_margin(
    margin_at: Callable[[float], float],
    period_deg: float,
    angle_step_deg: float,
) -> list[tuple[float, float]]:
    count = max(1, int(math.ceil(period_deg / angle_step_deg)))
    step = period_deg / count
    return [(clean_float(index * step), float(margin_at(index * step))) for index in range(count)]


def _max_margin_in_unwrapped_interval(
    margin_at: Callable[[float], float],
    start: float,
    end: float,
    period_deg: float,
    angle_step_deg: float,
) -> tuple[float, float]:
    if end < start:
        start, end = end, start
    width = max(0.0, end - start)
    count = max(1, int(math.ceil(width / angle_step_deg)))
    candidates = [start + width * index / count for index in range(count + 1)]
    return max(
        ((candidate, float(margin_at(candidate % period_deg))) for candidate in candidates),
        key=lambda item: (item[1], -item[0]),
    )


def _split_unwrapped_interval(start: float, end: float, period_deg: float) -> list[tuple[float, float]]:
    if end - start >= period_deg - 1e-9:
        return [(0.0, period_deg)]
    start_mod = start % period_deg
    end_mod = end % period_deg
    if int(math.floor(start / period_deg)) == int(math.floor(end / period_deg)) and start_mod <= end_mod:
        return [(start_mod, end_mod)]
    return [(0.0, end_mod), (start_mod, period_deg)]


def _center_points(points: list[Any]) -> list[list[float]]:
    parsed: list[list[float]] = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        parsed.append([float(point[0]), float(point[1])])
    if not parsed:
        return []
    center_x = (min(point[0] for point in parsed) + max(point[0] for point in parsed)) / 2.0
    center_y = (min(point[1] for point in parsed) + max(point[1] for point in parsed)) / 2.0
    return [[point[0] - center_x, point[1] - center_y] for point in parsed]


def _positive_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def clean_float(value: float) -> float:
    return round(float(value), 12)


def canonical_hash(data: Any) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
