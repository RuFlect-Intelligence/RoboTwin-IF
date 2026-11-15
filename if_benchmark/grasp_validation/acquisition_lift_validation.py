"""Geometry sub-gate for the planning-only acquisition-to-carry transition.

The caller must separately validate the *same complete path* with strict
CuRobo self/world constraints and the shared all-waypoint robot-mesh FCL
validator. This module never certifies a grasp or physical execution.
"""
from __future__ import annotations

import hashlib
from itertools import product
import math
from typing import Sequence

import numpy as np

from if_benchmark.schema import artifact_hash


ACQUISITION_LIFT_VALIDATOR_VERSION = "acquisition_support_exit_geometry_v2_source_support_floor"
_TABLE_MARGIN_M = 1e-6
_BOX_ENVELOPE_M = 0.005
_HELD_PADDING_M = 0.005
_MAX_EVALUATION_ROUNDOFF_M = 1e-9
_U64 = np.finfo(np.float64).eps / 2.0
_GAMMA128 = 128.0 * _U64 / (1.0 - 128.0 * _U64)


def _array(value, name):
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _array_hash(array):
    values = np.ascontiguousarray(array, dtype="<f8")
    return {"shape": list(values.shape), "dtype": "float64_le",
            "sha256": hashlib.sha256(values.tobytes()).hexdigest()}


def _roundoff(scale):
    """Conservative arithmetic guard for the supplied binary64 geometry only.

    gamma_128 bounds a deliberately over-counted evaluation chain of unit
    quaternion normalization/rotation, support projection, and subtraction.
    It is NOT an allowance for uncertainty in an upstream FK or physical pose.
    Reject poorly scaled inputs instead of granting a larger support allowance.
    """
    result = _GAMMA128 * np.maximum(1.0, scale)
    if np.max(result) > _MAX_EVALUATION_ROUNDOFF_M:
        raise ValueError("geometry scale exceeds the support roundoff bound")
    return result


def _rotation(quaternion):
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError("table quaternion must be nonzero and finite")
    w, x, y, z = quaternion / norm
    return np.asarray([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])


def _validate_expansion(raw, expanded, world_to_parent):
    """Require the canonical eight-corner 5 mm world-box expansion per vertex.

    Use the original attachment transform, including its representable FK
    rounding, rather than inventing a fresh exactly orthogonal approximation.
    Requiring the complete corner set proves this envelope was not reduced.
    Raw geometry and transform provenance remain bound by the live-piece receipt.
    """
    if expanded.shape != (8 * len(raw), 3):
        raise ValueError("expanded held vertices must have eight corners per raw vertex")
    corners = expanded.reshape(len(raw), 8, 3)
    guard = float(_roundoff(np.max(np.abs(raw)) + np.max(np.abs(expanded)) + 1.0))
    offsets = (np.asarray(list(product((-_BOX_ENVELOPE_M, _BOX_ENVELOPE_M), repeat=3)))
               @ world_to_parent[:3, :3].T)
    if not np.allclose(corners, raw[:, None, :] + offsets[None, :, :],
                       atol=guard, rtol=0):
        raise ValueError("held expansion has missing, shifted, or inconsistent corners")


def validate_acquisition_lift_path(
    *, robot_spheres_world, sphere_link_names: Sequence[str], ee_world_matrices,
    raw_held_vertices_parent, expanded_held_vertices_parent, held_expansion_world_to_parent,
    source_raw_vertices_world, padding_m,
    table_obstacles, required_pick_clearance_m, arm: str,
    trajectory_sha256: str, geometry_sha256: str,
):
    """Validate all supplied waypoints with a bounded acquisition support prefix.

    Native robot spheres must already contain the unchanged planner's 4 mm
    buffer. Only the selected arm's existing link7/link8 grasping fingers may
    rely on the separate exact robot-mesh FCL gate while leaving the support.
    That exception ends at the first waypoint with all native robot spheres
    and the complete expanded/padded held geometry clear of the table. It
    cannot restart later, and never applies to the final carry boundary.

    Existing settled support contact is bounded by the independently captured
    live source vertices, never by the path's first waypoint. This gate prevents
    deeper contact; the caller must independently establish that the initial
    source state is admissible. Scene stability alone does not prove that.
    """
    if arm not in {"left", "right"}:
        raise ValueError("arm must be left or right")
    prefix = "fl" if arm == "left" else "fr"
    touch_links = (f"{prefix}_link7", f"{prefix}_link8")
    spheres = _array(robot_spheres_world, "robot spheres")
    matrices = _array(ee_world_matrices, "EE matrices")
    raw = _array(raw_held_vertices_parent, "raw held vertices")
    expanded = _array(expanded_held_vertices_parent, "expanded held vertices")
    source = _array(source_raw_vertices_world, "independent source world vertices")
    expansion_transform = _array(held_expansion_world_to_parent, "attachment expansion transform")
    padding = float(padding_m)
    required = float(required_pick_clearance_m)
    if (not math.isfinite(padding) or padding != _HELD_PADDING_M
            or not math.isfinite(required) or required < 0.04):
        raise ValueError("original 5 mm held padding and at least 40 mm clearance are required")
    if (spheres.ndim != 3 or spheres.shape[-1] != 4 or not len(spheres)
            or not spheres.shape[1] or matrices.shape != (len(spheres), 4, 4)
            or raw.ndim != 2 or raw.shape[1:] != (3,) or not len(raw)
            or expanded.ndim != 2 or expanded.shape[1:] != (3,) or not len(expanded)
            or source.shape != raw.shape
            or expansion_transform.shape != (4, 4)):
        raise ValueError("complete path spheres, rigid matrices, and held vertices are required")
    names = list(sphere_link_names)
    if (len(names) != spheres.shape[1] or any(not isinstance(name, str) or not name for name in names)
            or "attached_object" in names or not set(touch_links).issubset(names)):
        raise ValueError("native sphere links must include both fixed arm-specific grasping fingers")
    active = spheres[:, :, 3] > 0.0
    if not active.any(axis=1).all():
        raise ValueError("each waypoint must contain active native robot spheres")
    if not np.array_equal(spheres[:, :, 3], np.broadcast_to(spheres[0, :, 3], active.shape)):
        raise ValueError("native sphere radii and buffer must remain unchanged along the path")
    all_matrices = np.concatenate((matrices, expansion_transform[None]))
    rotations = all_matrices[:, :3, :3]
    if (not np.allclose(all_matrices[:, 3], [0, 0, 0, 1], atol=1e-7, rtol=0)
            or not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-6, rtol=0)
            or not np.allclose(np.linalg.det(rotations), 1, atol=1e-6, rtol=0)):
        raise ValueError("EE matrices must be rigid FK poses")
    for label, value in (("trajectory", trajectory_sha256), ("geometry", geometry_sha256)):
        if (not isinstance(value, str) or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)):
            raise ValueError(f"{label} SHA256 must be a lowercase 64-character digest")
    _validate_expansion(raw, expanded, expansion_transform)
    expected_parent = source @ expansion_transform[:3, :3].T + expansion_transform[:3, 3]
    binding_scale = float((np.abs(source) @ np.abs(expansion_transform[:3, :3]).T
                           + np.abs(expansion_transform[:3, 3])).max() + np.abs(raw).max())
    binding_roundoff = float(_roundoff(binding_scale))
    if not np.allclose(raw, expected_parent, atol=binding_roundoff, rtol=0):
        raise ValueError("raw held vertices do not match the independent source and original attachment transform")
    if not table_obstacles:
        raise ValueError("all actual table collision boxes are required")
    distances = np.full(spheres.shape[:2], np.inf)
    table_top = -np.inf
    table_scale = 1.0
    canonical_tables = []
    for index, box in enumerate(table_obstacles):
        dims = _array(box["dims"], "table dimensions")
        pose = _array(box["pose"], "table pose")
        if dims.shape != (3,) or pose.shape != (7,) or np.any(dims <= 0):
            raise ValueError("actual table collision boxes need positive dimensions and pose7")
        rotation = _rotation(pose[3:])
        table_top = max(table_top, float(pose[2] + np.abs(rotation[2]) @ (dims / 2)))
        table_scale = max(table_scale, float(np.abs(pose[:3]).sum() + dims.sum()))
        local = (spheres[:, :, :3] - pose[:3]) @ rotation
        displacement = np.abs(local) - dims / 2
        gap = (np.linalg.norm(np.maximum(displacement, 0), axis=-1)
               + np.minimum(displacement.max(axis=-1), 0) - spheres[:, :, 3])
        distances = np.minimum(distances, np.where(active, gap, np.inf))
        canonical_tables.append({"name": str(box.get("name", f"table_{index}")),
                                 "dims": dims.tolist(), "pose": pose.tolist()})
    source_minimum_z = float(source[:, 2].min())
    source_clearance = source_minimum_z - table_top
    # Retain only existing source support contact. A positive source clearance
    # does not impose a no-downward-motion rule above the actual table plane.
    # This baseline is independent of every trajectory waypoint.
    support_floor = min(source_clearance, 0.0)
    raw_clearances, expanded_clearances, roundoff_bounds = [], [], []
    for matrix in matrices:
        raw_clearances.append(float((raw @ matrix[2, :3]).min() + matrix[2, 3] - table_top))
        expanded_clearances.append(float((expanded @ matrix[2, :3]).min()
                                         + matrix[2, 3] - table_top - padding))
        scale = float((np.abs(raw) @ np.abs(matrix[2, :3])).max()
                      + abs(matrix[2, 3]) + abs(table_top) + table_scale
                      + abs(source_minimum_z) + binding_scale)
        roundoff_bounds.append(float(_roundoff(scale)))
    raw_clearances = np.asarray(raw_clearances)
    expanded_clearances = np.asarray(expanded_clearances)
    roundoff_bounds = np.asarray(roundoff_bounds)
    robot_clearances = distances.min(axis=1)
    finger = np.asarray([name in touch_links for name in names])
    strict_nonfinger = np.any(distances[:, ~finger] < _TABLE_MARGIN_M, axis=1)
    finger_bad = np.any(distances[:, finger] < _TABLE_MARGIN_M, axis=1)
    fully_clear = ((robot_clearances >= _TABLE_MARGIN_M)
                   & (expanded_clearances >= _TABLE_MARGIN_M))
    exits = np.flatnonzero(fully_clear)
    exit_index = int(exits[0]) if len(exits) else None
    waived_finger = np.arange(len(spheres)) < (exit_index if exit_index is not None else len(spheres) - 1)
    waived_finger[-1] = False
    required_finger_failure = finger_bad & ~waived_finger
    raw_failure = raw_clearances < support_floor - roundoff_bounds
    held_reentry_failure = np.zeros(len(spheres), dtype=bool)
    if exit_index is not None:
        held_reentry_failure[exit_index:] = expanded_clearances[exit_index:] < _TABLE_MARGIN_M
    boundary_failure = []
    if robot_clearances[-1] < _TABLE_MARGIN_M:
        boundary_failure.append("native_buffered_robot_table_clearance")
    if expanded_clearances[-1] < _TABLE_MARGIN_M:
        boundary_failure.append("expanded_padded_held_table_clearance")
    if raw_clearances[-1] < required:
        boundary_failure.append("required_pick_support_clearance")
    failure = raw_failure | strict_nonfinger | required_finger_failure | held_reentry_failure
    if boundary_failure:
        failure[-1] = True
    needed_fcl = np.flatnonzero(waived_finger & finger_bad).tolist()
    payload = {
        "validator": ACQUISITION_LIFT_VALIDATOR_VERSION,
        "field_version": 2,
        "planning_only": True, "physical_certification": False,
        "validation_scope": "acquisition_support_and_native_robot_table_geometry_only",
        "geometry_checks_passed": not bool(np.any(failure)),
        "external_gates_required": ["same_path_strict_curobo_self_and_world_constraints",
                                    "same_path_robot_mesh_fcl_all_waypoints_subject_only_excluded",
                                    "complete_held_geometry_non_table_obstacles_and_non_touch_robot_links"],
        "caller_evidence_requirements": ["complete_native_sphere_link_set_and_original_4mm_buffer",
                                         "complete_live_raw_piece_vertices_and_original_attachment_transform",
                                         "independent_initial_source_support_admissibility_not_just_scene_stability",
                                         "complete_actual_table_collision_boxes",
                                         "trajectory_and_geometry_hashes_match_the_supplied_arrays"],
        "native_robot_buffer_source": "unchanged_planner_buffer_already_in_input_sphere_radii",
        "arm": arm, "acquisition_finger_touch_links": list(touch_links),
        "trajectory_sha256": trajectory_sha256, "geometry_sha256": geometry_sha256,
        "input_array_hashes": {name: _array_hash(value) for name, value in
                               (("robot_spheres_world", spheres), ("ee_world_matrices", matrices),
                                ("raw_held_vertices_parent", raw), ("expanded_held_vertices_parent", expanded),
                                ("held_expansion_world_to_parent", expansion_transform),
                                ("source_raw_vertices_world", source))},
        "sphere_link_names_hash": artifact_hash(names),
        "table_geometry_hash": artifact_hash(canonical_tables),
        "checked_waypoint_count": len(spheres), "table_top_plane_z": table_top,
        "dimension_envelope_m": _BOX_ENVELOPE_M, "padding_m": padding,
        "required_pick_clearance_m": required, "table_boundary_margin_m": _TABLE_MARGIN_M,
        "source_raw_minimum_world_z": source_minimum_z,
        "source_raw_table_clearance_m": source_clearance,
        "source_support_floor_m": support_floor,
        "source_parent_binding_roundoff_bound_m": binding_roundoff,
        "surface_failure_policy": "below_independent_source_support_floor_without_deeper_contact",
        "support_roundoff": {"model": "gamma128_binary64_supplied_geometry_evaluation_only",
                              "unit_roundoff": _U64, "operation_count_bound": 128,
                              "maximum_accepted_bound_m": _MAX_EVALUATION_ROUNDOFF_M,
                              "bounds_by_waypoint_m": roundoff_bounds.tolist(),
                              "does_not_cover_upstream_fk_or_physical_uncertainty": True},
        "support_exit_waypoint_index": exit_index,
        "finger_table_proxy_overlap_requiring_external_fcl_waypoints": needed_fcl,
        "surface_penetration_failed_waypoints": np.flatnonzero(raw_failure).tolist(),
        "nonfinger_buffered_robot_table_failed_waypoints": np.flatnonzero(strict_nonfinger).tolist(),
        "finger_buffered_robot_table_failed_waypoints": np.flatnonzero(required_finger_failure).tolist(),
        "held_envelope_reentry_failed_waypoints": np.flatnonzero(held_reentry_failure).tolist(),
        "carry_boundary_failure_reasons": boundary_failure,
        "carry_boundary_failed_waypoints": [len(spheres) - 1] if boundary_failure else [],
        "failed_waypoint_indices": np.flatnonzero(failure).tolist(),
        "raw_held_clearance_by_waypoint_m": raw_clearances.tolist(),
        "expanded_padded_held_clearance_by_waypoint_m": expanded_clearances.tolist(),
        "native_robot_minimum_table_clearance_by_waypoint_m": robot_clearances.tolist(),
    }
    return {**payload, "validation_hash": artifact_hash(payload)}
