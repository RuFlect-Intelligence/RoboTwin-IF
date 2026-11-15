"""Pure terminal predicates for M5 articulated receiver probes."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def articulation_state_metrics(
    *,
    final_qpos: list[float],
    resolved_joint_targets: list[dict[str, Any]],
) -> dict[str, Any]:
    joints: list[dict[str, Any]] = []
    for target in resolved_joint_targets:
        index = int(target["joint_index"])
        if index < 0 or index >= len(final_qpos):
            joints.append(
                {
                    "joint_index": index,
                    "held": False,
                    "failure_reason": "joint_index_out_of_range",
                }
            )
            continue
        final = float(final_qpos[index])
        expected = float(target["target_qpos"])
        tolerance = float(target["tolerance_qpos"])
        error = abs(final - expected)
        joints.append(
            {
                "joint_index": index,
                "joint_name": target.get("joint_name"),
                "target_qpos": expected,
                "final_qpos": final,
                "absolute_error": error,
                "tolerance_qpos": tolerance,
                "held": error <= tolerance,
            }
        )
    return {
        "held": bool(joints) and all(joint["held"] for joint in joints),
        "joints": joints,
    }


def articulated_place_in_success_metrics(
    *,
    subject_state: dict[str, Any],
    cavity_world_matrix: Any,
    cavity: dict[str, Any],
    contact_by_link: dict[str, Any],
    subject_table_contact: bool,
    subject_world_contact: bool,
    terminal_stability: dict[str, Any],
    articulation_state: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate place_in in the reviewed cavity-link frame.

    The center may be inside while part of a tall subject remains above the
    opening.  Physical support must come from an explicitly allowed cavity
    link, never from the door, table, or world.
    """

    matrix = np.asarray(cavity_world_matrix, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError("cavity_world_matrix must be 4x4")
    inverse = np.linalg.inv(matrix)
    center_world = np.asarray(subject_state["aabb"]["center"], dtype=float)
    center_local = _transform_point(inverse, center_world)
    corners_world = _aabb_corners(subject_state["aabb"])
    corners_local = np.asarray([_transform_point(inverse, corner) for corner in corners_world])
    local_min = corners_local.min(axis=0)

    # ``usable_region`` is the exposed, collision-free region used to choose
    # a static Terminal witness. A released object may settle deeper into the
    # same physical cavity, so skill/settle containment uses the separately
    # reviewed full interior when present.
    region = cavity.get("containment_region") or cavity["usable_region"]
    region_source = (
        "containment_region"
        if cavity.get("containment_region") is not None
        else "usable_region"
    )
    region_center = np.asarray(region["center_xyz"], dtype=float)
    half_extents = np.asarray(region["half_extents_xyz"], dtype=float)
    lower = region_center - half_extents
    upper = region_center + half_extents
    xy_inside = bool(np.all(center_local[:2] >= lower[:2]) and np.all(center_local[:2] <= upper[:2]))
    center_z_inside = bool(lower[2] <= center_local[2] <= upper[2])
    support_plane_z = float(cavity["support_plane_z"])
    bottom_above_floor = bool(local_min[2] >= support_plane_z - float(cavity.get("floor_tolerance_m") or 0.012))

    opening = cavity["opening_plane"]
    opening_point = np.asarray(opening["point_xyz"], dtype=float)
    opening_normal = np.asarray(opening["outward_normal_xyz"], dtype=float)
    opening_normal /= max(float(np.linalg.norm(opening_normal)), 1e-12)
    signed_opening_distance = float(np.dot(center_local - opening_point, opening_normal))
    behind_opening = signed_opening_distance <= float(cavity.get("opening_tolerance_m") or 0.005)

    allowed_links = [str(link) for link in cavity["allowed_support_links"]]
    forbidden_links = [str(link) for link in cavity["forbidden_support_links"]]
    active_links = sorted(link for link, metrics in contact_by_link.items() if _contact_active(metrics))
    allowed_contact = any(link in active_links for link in allowed_links)
    forbidden_contact = any(link in active_links for link in forbidden_links)
    checks = {
        "articulation_state_held": bool(articulation_state.get("held")),
        "terminal_stable": bool(terminal_stability.get("stable")),
        "subject_center_inside_cavity_xy": xy_inside,
        "subject_center_inside_cavity_z": center_z_inside,
        "subject_behind_opening_plane": behind_opening,
        "subject_not_below_cavity_floor": bottom_above_floor,
        "allowed_cavity_support_contact": allowed_contact,
        "no_forbidden_link_support": not forbidden_contact,
        "no_table_support": not bool(subject_table_contact),
        "no_world_support": not bool(subject_world_contact),
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    return {
        "success": all(checks.values()),
        "failure_reason": "ok" if all(checks.values()) else _failure_reason(failed_checks),
        "failed_checks": failed_checks,
        "checks": checks,
        "subject_center_cavity_local": _clean_vector(center_local),
        "subject_local_aabb_min": _clean_vector(local_min),
        "containment_region_source": region_source,
        "containment_region_lower": _clean_vector(lower),
        "containment_region_upper": _clean_vector(upper),
        # Compatibility aliases retained for old report/render consumers.
        "usable_region_lower": _clean_vector(lower),
        "usable_region_upper": _clean_vector(upper),
        "opening_signed_distance_m": _clean_float(signed_opening_distance),
        "active_contact_links": active_links,
        "allowed_support_links": allowed_links,
        "forbidden_support_links": forbidden_links,
        "articulation_state": articulation_state,
        "terminal_stability": terminal_stability,
    }


def cavity_witness_world_pose(
    *,
    cavity_world_matrix: Any,
    cavity: dict[str, Any],
    subject_q0_wxyz: list[float],
    witness_delta_deg: float,
    subject_local_bottom_z: float,
    clearance_m: float,
) -> dict[str, list[float]]:
    """Construct an upright root witness from a cavity-local support point."""

    matrix = np.asarray(cavity_world_matrix, dtype=float)
    center = np.asarray(cavity["usable_region"]["center_xyz"], dtype=float).copy()
    center[2] = float(cavity["support_plane_z"]) + float(clearance_m) - float(subject_local_bottom_z)
    world_position = _transform_point(matrix, center)
    frame_yaw = math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))
    witness_yaw = frame_yaw + math.radians(float(witness_delta_deg))
    yaw_q = [math.cos(witness_yaw / 2.0), 0.0, 0.0, math.sin(witness_yaw / 2.0)]
    world_q = _quat_multiply_wxyz(yaw_q, subject_q0_wxyz)
    return {
        "p": _clean_vector(world_position),
        "q_wxyz": _clean_vector(np.asarray(world_q, dtype=float)),
    }


def _failure_reason(failed_checks: list[str]) -> str:
    precedence = [
        ("articulation_state_held", "receiver_state_drift"),
        ("no_table_support", "subject_resting_on_table"),
        ("no_world_support", "subject_resting_on_world"),
        ("no_forbidden_link_support", "subject_supported_by_occluding_link"),
        ("allowed_cavity_support_contact", "missing_cavity_support_contact"),
        ("subject_center_inside_cavity_xy", "outside_receiver_cavity"),
        ("subject_center_inside_cavity_z", "outside_receiver_cavity"),
        ("subject_behind_opening_plane", "outside_receiver_opening"),
        ("subject_not_below_cavity_floor", "penetrating_cavity_floor"),
        ("terminal_stable", "terminal_unstable"),
    ]
    for check, reason in precedence:
        if check in failed_checks:
            return reason
    return "checker_false"


def _contact_active(metrics: Any) -> bool:
    if isinstance(metrics, dict):
        return bool(metrics.get("active"))
    return bool(metrics)


def _aabb_corners(aabb: dict[str, Any]) -> np.ndarray:
    lower = np.asarray(aabb["min"], dtype=float)
    upper = np.asarray(aabb["max"], dtype=float)
    return np.asarray(
        [
            [x, y, z]
            for x in (lower[0], upper[0])
            for y in (lower[1], upper[1])
            for z in (lower[2], upper[2])
        ],
        dtype=float,
    )


def _transform_point(matrix: np.ndarray, point: np.ndarray) -> np.ndarray:
    homogeneous = np.ones(4, dtype=float)
    homogeneous[:3] = point[:3]
    return (matrix @ homogeneous)[:3]


def _quat_multiply_wxyz(left: list[float], right: list[float]) -> list[float]:
    w1, x1, y1, z1 = (float(value) for value in left)
    w2, x2, y2, z2 = (float(value) for value in right)
    return [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]


def _clean_vector(values: np.ndarray) -> list[float]:
    return [_clean_float(value) for value in values.tolist()]


def _clean_float(value: float) -> float:
    rounded = round(float(value), 12)
    return 0.0 if rounded == -0.0 else rounded
