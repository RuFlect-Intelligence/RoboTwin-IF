"""Simulator-independent R02 Contact-constrained grasp contract.

RoboTwin Contact Points are authored actor-local SE(3) frames.  R02 treats
each frame as one complete orientation and contact-region hypothesis.  A table
acquisition may translate that frame by the minimum world-Z distance required
to keep the reviewed gripper collision envelope above the live support plane,
but only while the authored Contact remains inside the inner-pad length.  No
wrist-roll, symmetry, centre-offset, or random pose is generated here.
"""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import transforms3d as t3d

from if_benchmark.grasp_validation.contact_contract import (
    CONTACT_TO_GRASP_FRAME,
    EXACT_GRASP_POLICY,
    EXACT_GRASP_POLICY_VERSION as EXACT_GRASP_POLICY_VERSION,
    GraspContractError,
    _clean_float,
    allowed_contact_ids as allowed_contact_ids,
    build_contact_contract as build_contact_contract,
    clean_matrix,
    clean_vector,
    contact_groups_for_ids as contact_groups_for_ids,
    validate_contact_matrix,
)
from if_benchmark.schema import artifact_hash, canonical_json, deterministic_id

APPROACH_CLASS_ORDER: dict[str, int] = {
    "top_down": 0,
    "oblique_down": 1,
    "horizontal": 2,
    "upward": 3,
}

def scaled_contact_world_matrix(
    local_matrix: Sequence[Sequence[float]],
    scale_xyz: Sequence[float],
    root_matrix: Sequence[Sequence[float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the exact translation-only scale and actor-root transform."""

    local = validate_contact_matrix(local_matrix)
    scale = np.asarray(scale_xyz, dtype=float)
    root = np.asarray(root_matrix, dtype=float)
    if scale.shape != (3,) or not np.isfinite(scale).all():
        raise GraspContractError("Contact scale must contain three finite values")
    if np.any(scale <= 0.0):
        raise GraspContractError("Contact scale values must be positive")
    if root.shape != (4, 4) or not np.isfinite(root).all():
        raise GraspContractError("actor root matrix must be finite 4x4")
    scaled = local.copy()
    scaled[:3, 3] *= scale
    return scaled, root @ scaled


def grasp_frame_from_contact(
    world_contact_matrix: Sequence[Sequence[float]],
) -> np.ndarray:
    contact = validate_contact_matrix(world_contact_matrix)
    return contact @ CONTACT_TO_GRASP_FRAME


def approach_direction(
    world_contact_matrix: Sequence[Sequence[float]],
) -> np.ndarray:
    """Return authored pre-grasp -> grasp direction without changing rotation."""

    direction = grasp_frame_from_contact(world_contact_matrix)[:3, 0]
    norm = float(np.linalg.norm(direction))
    if not math.isclose(norm, 1.0, abs_tol=5e-3):
        raise GraspContractError(
            f"Contact approach axis must be unit length, got {norm}"
        )
    return direction.copy()


def classify_approach(
    direction: Sequence[float],
    *,
    table_acquisition: bool = True,
) -> str:
    """Classify the exact world approach used by the table-pick policy."""

    vector = np.asarray(direction, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise GraspContractError("approach direction must contain finite xyz")
    norm = float(np.linalg.norm(vector))
    if not math.isclose(norm, 1.0, abs_tol=5e-3):
        raise GraspContractError(
            f"approach direction must be unit length, got {norm}"
        )
    if not table_acquisition:
        return "unbiased"
    z = float(vector[2])
    if z <= -0.85:
        return "top_down"
    if z <= -0.30:
        return "oblique_down"
    if z < 0.30:
        return "horizontal"
    return "upward"


def exact_grasp_poses(
    world_contact_matrix: Sequence[Sequence[float]],
    retreat_distance_m: float,
    *,
    gripper_tcp_offset_m: float = 0.12,
) -> dict[str, Any]:
    """Project one Contact into exact Grasp and Pre-Grasp 6D poses."""

    retreat = float(retreat_distance_m)
    tcp_offset = float(gripper_tcp_offset_m)
    if retreat < 0.0 or tcp_offset < 0.0:
        raise GraspContractError("grasp offsets must be non-negative")
    grasp_frame = grasp_frame_from_contact(world_contact_matrix)
    rotation = grasp_frame[:3, :3]
    contact_position = grasp_frame[:3, 3]
    grasp_position = contact_position + rotation @ np.asarray(
        [-tcp_offset, 0.0, 0.0], dtype=float
    )
    pre_position = contact_position + rotation @ np.asarray(
        [-tcp_offset - retreat, 0.0, 0.0], dtype=float
    )
    quaternion = t3d.quaternions.mat2quat(rotation)
    grasp_pose = clean_vector([*grasp_position, *quaternion])
    pre_grasp_pose = clean_vector([*pre_position, *quaternion])
    return {
        "contact_position_world": clean_vector(contact_position),
        "approach_direction_world": clean_vector(rotation[:, 0]),
        "grasp_pose": grasp_pose,
        "pre_grasp_pose": pre_grasp_pose,
        "gripper_tcp_offset_m": _clean_float(tcp_offset),
        "retreat_distance_m": _clean_float(retreat),
        "cartesian_approach_length_m": _clean_float(retreat),
    }


def support_safe_grasp_poses(
    world_contact_matrix: Sequence[Sequence[float]],
    retreat_distance_m: float,
    *,
    support_plane_z_m: float,
    geometry: Mapping[str, Any] | None = None,
    gripper_tcp_offset_m: float = 0.12,
) -> dict[str, Any]:
    """Project one Contact to the lowest table-safe gripper pose.

    The authored orientation is immutable.  The only permitted freedom is a
    world-+Z translation that removes support-plane penetration by the
    reviewed gripper collision envelope.  The original Contact must remain
    within the inner pad's longitudinal ``x/z`` extent after translation.
    """

    support_z = float(support_plane_z_m)
    if not math.isfinite(support_z):
        raise GraspContractError("support plane must be finite")
    profile = dict(
        geometry
        if geometry is not None
        else EXACT_GRASP_POLICY["table_gripper_geometry"]
    )
    envelope_min = np.asarray(
        profile.get("collision_envelope_min_xyz_m"), dtype=float
    )
    envelope_max = np.asarray(
        profile.get("collision_envelope_max_xyz_m"), dtype=float
    )
    pad_x = np.asarray(profile.get("inner_pad_x_interval_m"), dtype=float)
    pad_z = np.asarray(profile.get("inner_pad_z_interval_m"), dtype=float)
    if (
        envelope_min.shape != (3,)
        or envelope_max.shape != (3,)
        or pad_x.shape != (2,)
        or pad_z.shape != (2,)
        or not np.isfinite(
            np.concatenate([envelope_min, envelope_max, pad_x, pad_z])
        ).all()
        or np.any(envelope_max <= envelope_min)
        or pad_x[1] <= pad_x[0]
        or pad_z[1] <= pad_z[0]
    ):
        raise GraspContractError("invalid table-gripper geometry profile")
    support_buffer = float(profile.get("support_buffer_m", 0.0))
    if not math.isfinite(support_buffer) or support_buffer < 0.0:
        raise GraspContractError("support buffer must be finite and non-negative")

    authored = validate_contact_matrix(world_contact_matrix)
    original = exact_grasp_poses(
        authored,
        retreat_distance_m,
        gripper_tcp_offset_m=gripper_tcp_offset_m,
    )
    grasp_pose = np.asarray(original["grasp_pose"], dtype=float)
    rotation = t3d.quaternions.quat2mat(grasp_pose[3:])
    corners = np.asarray(
        [
            [x, y, z]
            for x in (envelope_min[0], envelope_max[0])
            for y in (envelope_min[1], envelope_max[1])
            for z in (envelope_min[2], envelope_max[2])
        ],
        dtype=float,
    )
    minimum_before = float(
        np.min(grasp_pose[2] + (rotation @ corners.T).T[:, 2])
    )
    adjustment = max(0.0, support_z + support_buffer - minimum_before)
    effective = authored.copy()
    effective[2, 3] += adjustment
    result = exact_grasp_poses(
        effective,
        retreat_distance_m,
        gripper_tcp_offset_m=gripper_tcp_offset_m,
    )
    adjusted_grasp = np.asarray(result["grasp_pose"], dtype=float)
    authored_contact_in_gripper = rotation.T @ (
        authored[:3, 3] - adjusted_grasp[:3]
    )
    tolerance = 1e-9
    pad_contains_contact = bool(
        pad_x[0] - tolerance
        <= authored_contact_in_gripper[0]
        <= pad_x[1] + tolerance
        and pad_z[0] - tolerance
        <= authored_contact_in_gripper[2]
        <= pad_z[1] + tolerance
    )
    minimum_after = minimum_before + adjustment
    return {
        **result,
        "authored_contact_position_world": clean_vector(authored[:3, 3]),
        "effective_contact_position_world": clean_vector(effective[:3, 3]),
        "support_clearance_adjustment_m": _clean_float(adjustment),
        "support_plane_z_m": _clean_float(support_z),
        "gripper_support_min_z_before_m": _clean_float(minimum_before),
        "gripper_support_min_z_after_m": _clean_float(minimum_after),
        "authored_contact_in_gripper_xyz_m": clean_vector(
            authored_contact_in_gripper
        ),
        "inner_pad_contains_authored_contact": pad_contains_contact,
    }


def joint_path_metrics(
    trajectories: Sequence[Sequence[Sequence[float]]],
    joint_limits: Sequence[Sequence[float]],
) -> dict[str, float | int]:
    """Measure continuous safety/travel over chained pre and grasp paths."""

    paths = [np.asarray(path, dtype=float) for path in trajectories]
    if not paths or any(path.ndim != 2 or len(path) == 0 for path in paths):
        raise GraspContractError("joint trajectories must be non-empty 2D arrays")
    limits = np.asarray(joint_limits, dtype=float)
    if limits.ndim != 2 or limits.shape[1] != 2:
        raise GraspContractError("joint limits must contain lower/upper pairs")
    joint_count = min(limits.shape[0], *(path.shape[1] for path in paths))
    if joint_count <= 0:
        raise GraspContractError("joint trajectories have no bounded joints")
    limits = limits[:joint_count]
    spans = limits[:, 1] - limits[:, 0]
    if not np.isfinite(spans).all() or np.any(spans <= 1e-9):
        raise GraspContractError("joint limit spans must be finite and positive")
    minimum_margin = math.inf
    maximum_step = 0.0
    total_travel = 0.0
    waypoint_count = 0
    previous: np.ndarray | None = None
    for path in paths:
        values = path[:, :joint_count]
        if not np.isfinite(values).all():
            raise GraspContractError("joint trajectory contains non-finite values")
        margins = np.minimum(values - limits[:, 0], limits[:, 1] - values) / spans
        minimum_margin = min(minimum_margin, float(np.min(margins)))
        for row in values:
            if previous is not None:
                delta = np.abs(row - previous) / spans
                maximum_step = max(maximum_step, float(np.max(delta)))
                total_travel += float(np.sum(delta))
            previous = row
        waypoint_count += len(values)
    return {
        "minimum_joint_limit_margin": _clean_float(minimum_margin),
        "maximum_normalized_joint_step": _clean_float(maximum_step),
        "normalized_total_travel": _clean_float(total_travel),
        "path_segment_count": len(paths),
        "path_waypoint_count": waypoint_count,
    }


def trajectory_summary(
    trajectory: Sequence[Sequence[float]],
) -> dict[str, Any]:
    path = np.asarray(trajectory, dtype=float)
    if path.ndim != 2 or len(path) == 0 or not np.isfinite(path).all():
        raise GraspContractError("trajectory must be a finite non-empty 2D array")
    payload = clean_matrix(path)
    return {
        "steps": int(path.shape[0]),
        "joint_count": int(path.shape[1]),
        "start_qpos": clean_vector(path[0]),
        "terminal_qpos": clean_vector(path[-1]),
        "trajectory_sha256": artifact_hash(payload),
    }


def grasp_candidate_sort_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    approach_class = str(candidate.get("approach_class") or "upward")
    tier = APPROACH_CLASS_ORDER.get(approach_class, 0)
    metrics = candidate.get("path_metrics") or {}
    groups = tuple(int(value) for value in candidate.get("contact_group_indices") or [])
    return (
        tier,
        round(float(metrics.get("normalized_total_travel", math.inf)), 12),
        round(float(metrics.get("maximum_normalized_joint_step", math.inf)), 12),
        -round(float(metrics.get("minimum_joint_limit_margin", -math.inf)), 12),
        round(float(candidate.get("cartesian_path_length_m", math.inf)), 12),
        int(candidate.get("contact_id", 1 << 30)),
        groups,
        round(float(candidate.get("retreat_distance_m", math.inf)), 12),
    )


def select_best_retreat_per_contact(
    candidates: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Collapse variants using the largest reachable alignment clearance.

    The retreat domain is an ordered path-recovery ladder, not a travel-cost
    optimization domain.  Choosing the shortest reachable retreat lets the
    wrist rotate beside the object and defeats the separation between high
    alignment and fixed-orientation approach.  Path cost ranks Contacts only
    after one clearance-safe retreat has been selected for each Contact.
    """

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        row = dict(candidate)
        if str(row.get("status")) != "reachable":
            continue
        grouped[int(row["contact_id"])].append(row)
    selected = []
    for _, rows in sorted(grouped.items()):
        if not rows:
            continue
        maximum_retreat = max(float(row["retreat_distance_m"]) for row in rows)
        clearance_rows = [
            row
            for row in rows
            if math.isclose(
                float(row["retreat_distance_m"]),
                maximum_retreat,
                abs_tol=1e-12,
            )
        ]
        selected.append(min(clearance_rows, key=grasp_candidate_sort_key))
    return sorted(selected, key=grasp_candidate_sort_key)


def candidate_id(payload: Mapping[str, Any]) -> str:
    identity = {
        key: payload.get(key)
        for key in (
            "contact_contract_hash",
            "arm",
            "world_yaw_deg",
            "reach_zone",
            "contact_id",
            "retreat_distance_m",
            "support_clearance_adjustment_m",
        )
    }
    return deterministic_id("gc", identity, length=20)


def trajectory_fingerprint(trajectories: Sequence[Sequence[Sequence[float]]]) -> str:
    return artifact_hash(
        [clean_matrix(np.asarray(path, dtype=float)) for path in trajectories]
    )


def canonical_contract_json(contract: Mapping[str, Any]) -> str:
    """Convenience boundary used by deterministic regression tests."""

    return canonical_json(dict(contract))
