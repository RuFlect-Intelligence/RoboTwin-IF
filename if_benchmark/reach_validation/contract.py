"""Shared exact-SceneSpec route witness and carry collision contracts.

The route witness is task-specific evidence produced after M13.  It freezes
which authored Contact and which reservation-local terminal pose were proved
by the exact-scene planner.  M15 consumes those semantic decisions but always
replans from the live robot state; no M4 canonical trajectory is replayed.
"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Mapping, Sequence

import numpy as np

from if_benchmark.relation_contract import (
    ANCHOR_PAIR_DISTANCE_BANDS_M,
    BETWEEN_MAXIMUM_PERPENDICULAR_M,
    DIRECTIONAL_ORTHOGONAL_GAP_MAX_M,
    RELATION_SURFACE_GAP_BANDS_M,
    RELATION_SURFACE_GAP_CONTRACT_VERSION,
    relation_anchor_pair_acceptance,
    relation_surface_gap_acceptance,
)
from if_benchmark.schema import artifact_hash
from if_benchmark.tasks.core import TASK_CONFIG
from if_benchmark.terminal_placement import (
    articulated_terminal_actor_pose,
    terminal_frame_binding,
    terminal_recipe_binding,
    validate_terminal_frame_binding,
)

from .core import content_hash


ROUTE_VALIDATION_VERSION = (
    "exact_scene_direct_route_witness_v5_shared_relation_geometry"
)
ROUTE_EXECUTION_CONTRACT_VERSION = (
    "exact_scene_route_execution_v6_operation_space_targets"
)
VERTICAL_CONTACT_RELATIONS = frozenset({"in", "on", "stack"})
SURFACE_GAP_RELATIONS = frozenset(RELATION_SURFACE_GAP_BANDS_M)
ANCHOR_PAIR_RELATIONS = frozenset(
    {"between", *ANCHOR_PAIR_DISTANCE_BANDS_M}
)
ROUTE_ACTOR_PHASES = ("lift", "transit", "approach", "descent")
ROUTE_CARRY_PHASES = ("transit", "approach", "descent")
ROUTE_PATH_PHASES = (
    "pre_grasp",
    "grasp",
    "lift",
    "transit",
    "approach",
    "descent",
    "release_retreat",
)

from if_benchmark.grasp_validation.vertical_lift import VERTICAL_LIFT_VERSION
from if_benchmark.grasp_validation.carry_motion import CARRY_MOTION_POLICY
from if_benchmark.manipulation_policy import POLICY as OPERATION_SPACE_POLICY


ROUTE_EXECUTION_CONTRACT = {
    "version": ROUTE_EXECUTION_CONTRACT_VERSION,
    "trajectory_consumption": "live_replan_from_selected_semantic_witness",
    "acquisition_lift_policy": VERTICAL_LIFT_VERSION,
    "carry_motion_policy": dict(CARRY_MOTION_POLICY),
    "operation_space_policy": dict(OPERATION_SPACE_POLICY),
    "canonical_probe_replay": False,
    "pick_witness_source": (
        "exact_m12_planning_request_and_m12_selected_contact"
    ),
    "phase_offsets_m": {
        # These values are projected from the one M13 SkillConfig policy.  The
        # route gate is a certificate for that skill, not a second 100 mm lift
        # policy hidden in M14/M15.
        "lift": float(TASK_CONFIG["skill_parameters"]["lift_m"]),
        "stack_lift": float(TASK_CONFIG["skill_parameters"]["stack_lift_m"]),
        "approach": float(TASK_CONFIG["skill_parameters"]["pre_grasp_m"]),
    },
    "carry_phase_sequence": list(ROUTE_CARRY_PHASES),
    "terminal_search": {
        "planar_offset_radii_m": [0.0, 0.02, 0.04, 0.05],
        "planar_yaw_deltas_deg": OPERATION_SPACE_POLICY["planar_yaws_deg"],
        "interior_offset_pairs_m": [[.02, .04], [.03, .04]],
        "vertical_xy_offsets_m": [-0.015, 0.0, 0.015],
        "vertical_yaw_deltas_deg": [0.0],
    },
    "held_collision": {
        "geometry": "live_world_axis_tight_aabb",
        "dimension_envelope_m": 0.005,
        "planner_padding_m": 0.005,
        "fixed_camera_pair_policy": "if_held_fixed_camera_exact_pair_v1",
    },
    "receiver_collision": {
        "transit": "included",
        "approach": "included",
        "vertical_descent": "excluded_after_reservation_precheck",
        "planar_descent": "included",
        "release_retreat": "included_after_release",
    },
    "release_retreat": {
        "frame": "arm",
        "axis": "negative_ee_local_x",
        "delta_m": 0.05,
        "fallback": None,
    },
    "terminal_tracking": {
        "translation_tolerance_m": 0.012,
        "orientation_tolerance_deg": 5.0,
        "reservation_margin_m": 0.0,
    },
    "terminal_phase_rebuild": {
        "source": "scene_spec_initial_plus_live_terminal_candidate",
        "approach_axis": "world_positive_z",
        "transit_z": "max(lift_z,approach_z)",
        "articulated_frame_delta_replay": False,
    },
    "terminal_relation_geometry": {
        "contract_version": RELATION_SURFACE_GAP_CONTRACT_VERSION,
        "surface_gap_bands_m": {
            relation: list(band)
            for relation, band in sorted(
                RELATION_SURFACE_GAP_BANDS_M.items()
            )
        },
        "directional_orthogonal_gap_maximum_m": (
            DIRECTIONAL_ORTHOGONAL_GAP_MAX_M
        ),
        "anchor_pair_distance_bands_m": {
            relation: list(band)
            for relation, band in sorted(
                ANCHOR_PAIR_DISTANCE_BANDS_M.items()
            )
        },
        "between_maximum_perpendicular_m": (
            BETWEEN_MAXIMUM_PERPENDICULAR_M
        ),
        "evidence": "shared_success_contract_relation_geometry_acceptance",
    },
    "articulated_terminal_frame_tracking": {
        "translation_tolerance_m": 0.003,
        "orientation_tolerance_deg": 1.0,
    },
}
ROUTE_EXECUTION_CONTRACT_HASH = artifact_hash(ROUTE_EXECUTION_CONTRACT)


class TerminalCandidateRejected(ValueError):
    """A deterministic search candidate is legal input but semantically bad."""


def route_execution_contract_identity() -> dict[str, Any]:
    return {
        "version": ROUTE_EXECUTION_CONTRACT_VERSION,
        "hash": ROUTE_EXECUTION_CONTRACT_HASH,
    }


def route_release_retreat_policy() -> dict[str, Any]:
    return deepcopy(ROUTE_EXECUTION_CONTRACT["release_retreat"])


def route_phase_policy(relation: str) -> dict[str, Any]:
    vertical = str(relation) in VERTICAL_CONTACT_RELATIONS
    return {
        "carry_phase_sequence": list(ROUTE_CARRY_PHASES),
        "carry_motion_policy": dict(CARRY_MOTION_POLICY),
        "receiver_collision": {
            "transit": "included",
            "approach": "included",
            "descent": (
                "excluded_after_reservation_precheck"
                if vertical
                else "included"
            ),
        },
        "release_retreat": route_release_retreat_policy(),
    }


def route_lift_delta_m(relation: str) -> float:
    key = "stack_lift" if str(relation) == "stack" else "lift"
    return float(ROUTE_EXECUTION_CONTRACT["phase_offsets_m"][key])


def terminal_search_offsets(relation: str) -> list[tuple[float, float]]:
    if str(relation) in VERTICAL_CONTACT_RELATIONS:
        return [
            (x, y)
            for x in (-0.015, 0.0, 0.015)
            for y in (-0.015, 0.0, 0.015)
        ]
    offsets = [(0.0, 0.0)]
    for radius in (0.02, 0.04, 0.05):
        offsets.extend(
            [
                (-radius, 0.0),
                (radius, 0.0),
                (0.0, -radius),
                (0.0, radius),
                (-radius, -radius),
                (-radius, radius),
                (radius, -radius),
                (radius, radius),
            ]
        )
    # Bounded interior refinements complement the coarse ring. They retain the
    # original domain and avoid placing every diagonal exactly on its boundary.
    for a, b in ROUTE_EXECUTION_CONTRACT["terminal_search"]["interior_offset_pairs_m"]:
        for x, y in ((a, b), (b, a)):
            offsets.extend((sx * x, sy * y) for sx in (-1., 1.) for sy in (-1., 1.))
    return offsets


def terminal_search_yaws(relation: str) -> tuple[float, ...]:
    return (
        (0.0,)
        if str(relation) in VERTICAL_CONTACT_RELATIONS
        else tuple(OPERATION_SPACE_POLICY["planar_yaws_deg"])
    )


def _quat_matrix(quaternion_wxyz: Sequence[float]) -> np.ndarray:
    q = np.asarray(quaternion_wxyz, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError("quaternion must contain four finite values")
    norm = float(np.linalg.norm(q))
    if norm <= 1e-12:
        raise ValueError("quaternion norm is zero")
    w, x, y, z = q / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _quat_multiply(
    left_wxyz: Sequence[float], right_wxyz: Sequence[float]
) -> list[float]:
    left = np.asarray(left_wxyz, dtype=float)
    right = np.asarray(right_wxyz, dtype=float)
    if left.shape != (4,) or right.shape != (4,):
        raise ValueError("quaternion multiplication requires two quaternions")
    w1, x1, y1, z1 = left
    w2, x2, y2, z2 = right
    value = np.asarray(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )
    value /= np.linalg.norm(value)
    return value.tolist()


def _actor_pose_matrix(pose: Sequence[float]) -> np.ndarray:
    value = np.asarray(pose, dtype=float)
    if value.shape != (7,) or not np.isfinite(value).all():
        raise ValueError("actor pose must contain finite xyz + quat_wxyz")
    result = np.eye(4, dtype=float)
    result[:3, :3] = _quat_matrix(value[3:])
    result[:3, 3] = value[:3]
    return result


def _z_quaternion(yaw_deg: float) -> list[float]:
    half = math.radians(float(yaw_deg)) * 0.5
    return [math.cos(half), 0.0, 0.0, math.sin(half)]


def _entity_world_aabb(
    entity: Mapping[str, Any],
    actor_pose: Sequence[float] | None = None,
) -> dict[str, list[float]]:
    """Project frozen q0-local tight bounds at one exact actor pose."""

    initial = entity.get("initial_pose") or {}
    bounds = initial.get("origin_relative_bounds") or {}
    minimum = np.asarray(bounds.get("min_xyz") or [], dtype=float)
    maximum = np.asarray(bounds.get("max_xyz") or [], dtype=float)
    q0 = np.asarray(initial.get("q0_wxyz") or [], dtype=float)
    if actor_pose is None:
        pose = entity.get("pose") or {}
        position = np.asarray(pose.get("position") or [], dtype=float)
        quaternion = np.asarray(
            pose.get("quat_wxyz") or [], dtype=float
        )
    else:
        value = np.asarray(actor_pose, dtype=float)
        position = value[:3] if value.shape == (7,) else np.asarray([])
        quaternion = value[3:] if value.shape == (7,) else np.asarray([])
    expected = (
        (minimum, (3,)),
        (maximum, (3,)),
        (q0, (4,)),
        (position, (3,)),
        (quaternion, (4,)),
    )
    if any(value.shape != shape for value, shape in expected):
        raise ValueError("SceneSpec entity lacks frozen terminal geometry")
    delta = _quat_matrix(quaternion) @ _quat_matrix(q0).T
    corners = np.asarray(
        [
            [x, y, z]
            for x in (minimum[0], maximum[0])
            for y in (minimum[1], maximum[1])
            for z in (minimum[2], maximum[2])
        ],
        dtype=float,
    )
    world = position + (delta @ corners.T).T
    world_minimum = np.min(world, axis=0)
    world_maximum = np.max(world, axis=0)
    return {
        "min": world_minimum.tolist(),
        "max": world_maximum.tolist(),
        "center": ((world_minimum + world_maximum) * 0.5).tolist(),
    }


def _entity_world_top_z(entity: Mapping[str, Any]) -> float:
    return float(_entity_world_aabb(entity)["max"][2])


def exact_carry_receiver_keys(
    carry_step: Mapping[str, Any],
) -> list[str]:
    """Return the non-empty, duplicate-free receiver list authored by M13."""

    receivers = [
        str(carry_step[key])
        for key in ("target", "target2")
        if carry_step.get(key) is not None
    ]
    if not receivers or any(not value for value in receivers):
        raise ValueError("carry receiver_keys must be non-empty")
    if len(set(receivers)) != len(receivers):
        raise ValueError("carry receiver_keys contain duplicates")
    return sorted(receivers)


def exact_reservation_receiver_keys(
    region: Mapping[str, Any],
) -> list[str]:
    """Return one reservation's exact non-empty, duplicate-free receivers."""

    raw = region.get("receiver_keys")
    if (
        not isinstance(raw, Sequence)
        or isinstance(raw, (str, bytes))
        or not raw
    ):
        raise ValueError("reservation receiver_keys must be a non-empty list")
    receivers = [str(value) for value in raw]
    if any(not value for value in receivers):
        raise ValueError("reservation receiver_keys contain an empty value")
    if len(set(receivers)) != len(receivers):
        raise ValueError("reservation receiver_keys contain duplicates")
    return sorted(receivers)


def _reservation_source(
    unit_artifact: Mapping[str, Any],
    scene_spec: Mapping[str, Any],
    carry_step: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    subject = str(carry_step.get("subject") or "")
    relation = str(carry_step.get("relation") or "")
    receivers = exact_carry_receiver_keys(carry_step)
    matches: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for unit in unit_artifact.get("units", []):
        for region in (unit.get("params") or {}).get(
            "reserved_regions", []
        ):
            if (
                str(region.get("subject_key") or region.get("subject") or "")
                != subject
                or str(region.get("relation") or "") != relation
            ):
                continue
            region_receivers = exact_reservation_receiver_keys(region)
            if receivers == region_receivers:
                matches.append((dict(unit), dict(region)))
    if len(matches) != 1:
        raise ValueError(
            f"terminal reservation cardinality is {len(matches)}, expected one"
        )
    unit, region = matches[0]
    unit_pose = next(
        (
            dict(value)
            for value in scene_spec.get("unit_poses", [])
            if str(value.get("unit_id") or "") == str(unit.get("id") or "")
        ),
        None,
    )
    if unit_pose is None:
        raise ValueError("terminal reservation Unit pose is missing")
    return unit, region, unit_pose


def reservation_terminal_base_pose(
    unit_artifact: Mapping[str, Any],
    scene_spec: Mapping[str, Any],
    carry_step: Mapping[str, Any],
    *,
    cavity_world_matrix: Sequence[Sequence[float]] | None = None,
    receiver_profiles: Mapping[str, Any] | None = None,
) -> tuple[list[float], dict[str, Any]]:
    """Derive the exact M11 reservation center in the frozen SceneSpec."""

    relation = str(carry_step["relation"])
    subject = str(carry_step["subject"])
    receivers = exact_carry_receiver_keys(carry_step)
    unit, region, unit_pose = _reservation_source(
        unit_artifact, scene_spec, carry_step
    )
    recipe_binding = terminal_recipe_binding(
        region,
        relation,
        receiver_profiles=receiver_profiles,
    )
    footprint = region.get("footprint") or {}
    center = np.asarray(footprint.get("center_xy") or [], dtype=float)
    position = np.asarray(unit_pose.get("position") or [], dtype=float)
    if center.shape != (2,) or position.shape != (3,):
        raise ValueError("terminal reservation pose is invalid")
    unit_yaw_deg = float(unit_pose.get("yaw_deg", 0.0))
    unit_yaw = math.radians(unit_yaw_deg)
    world_xy = position[:2] + np.asarray(
        [
            math.cos(unit_yaw) * center[0]
            - math.sin(unit_yaw) * center[1],
            math.sin(unit_yaw) * center[0]
            + math.cos(unit_yaw) * center[1],
        ],
        dtype=float,
    )
    entities = scene_spec.get("entities") or {}
    if subject not in entities or any(key not in entities for key in receivers):
        raise ValueError("terminal reservation entities are missing from SceneSpec")
    receiver_top = max(_entity_world_top_z(entities[key]) for key in receivers)
    table_top = min(
        (
            float((value.get("position") or [0.0, 0.0, 0.74])[2])
            for value in scene_spec.get("unit_poses", [])
        ),
        default=0.74,
    )
    clearance = float(
        recipe_binding["placement_recipe"].get(
            "clearance_m",
            region.get(
                "terminal_clearance_m",
                region.get("support_clearance_m", 0.015),
            ),
        )
    )
    subject_bounds = (
        ((unit.get("params") or {}).get("subject_z_bounds") or {}).get(subject)
        or [0.0, 0.0]
    )
    support_top = (
        receiver_top if relation in VERTICAL_CONTACT_RELATIONS else table_top
    )
    root_z = support_top + clearance - float(subject_bounds[0])

    subject_unit = next(
        (
            value
            for value in unit_artifact.get("units", [])
            if subject in value.get("entity_keys", [])
        ),
        None,
    )
    if subject_unit is None:
        raise ValueError("terminal subject Unit is missing")
    subject_unit_pose = next(
        (
            value
            for value in scene_spec.get("unit_poses", [])
            if str(value.get("unit_id") or "")
            == str(subject_unit.get("id") or "")
        ),
        None,
    )
    if subject_unit_pose is None:
        raise ValueError("terminal subject Unit pose is missing")
    subject_q = [
        float(value) for value in entities[subject]["pose"]["quat_wxyz"]
    ]
    articulated_recipe = (
        recipe_binding["placement_recipe"].get("constructor")
        == "articulated_place_in_cavity_link"
    )
    terminal_frame: dict[str, Any] | None = None
    reservation_frame_yaw_deg = unit_yaw_deg
    if articulated_recipe:
        if len(receivers) != 1 or cavity_world_matrix is None:
            raise ValueError(
                "articulated terminal requires one live cavity-link frame"
            )
        profile = recipe_binding.get("receiver_state_profile")
        if not isinstance(profile, Mapping):
            raise ValueError("articulated terminal reviewed profile is missing")
        initial = entities[subject].get("initial_pose") or {}
        q0 = initial.get("q0_wxyz")
        origin_bounds = initial.get("origin_relative_bounds") or {}
        lower = np.asarray(origin_bounds.get("min_xyz") or [], dtype=float)
        if not isinstance(q0, Sequence) or lower.shape != (3,):
            raise ValueError("articulated terminal subject geometry is missing")
        base = articulated_terminal_actor_pose(
            cavity_world_matrix=cavity_world_matrix,
            reviewed_receiver_profile=profile,
            placement_recipe=recipe_binding["placement_recipe"],
            subject_q0_wxyz=q0,
            subject_local_bottom_z=float(lower[2]),
            witness_delta_deg=float(region.get("relative_yaw_deg", 0.0)),
        )
        matrix = np.asarray(cavity_world_matrix, dtype=float)
        world_xy = np.asarray(base[:2], dtype=float)
        root_z = float(base[2])
        target_q = [float(value) for value in base[3:]]
        reservation_frame_yaw_deg = math.degrees(
            math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))
        )
        terminal_frame = terminal_frame_binding(matrix)
    elif relation in VERTICAL_CONTACT_RELATIONS:
        canonical_q = _quat_multiply(
            _z_quaternion(-float(subject_unit_pose.get("yaw_deg", 0.0))),
            subject_q,
        )
        target_q = _quat_multiply(
            _z_quaternion(
                unit_yaw_deg + float(region.get("relative_yaw_deg", 0.0))
            ),
            canonical_q,
        )
    else:
        target_q = subject_q
    usable = region.get("usable_region") or region.get("footprint") or {}
    if articulated_recipe:
        profile = recipe_binding["receiver_state_profile"]
        half_extents = np.asarray(
            profile["cavity"]["usable_region"]["half_extents_xyz"][:2],
            dtype=float,
        )
    else:
        half_extents = np.asarray(usable.get("half_extents_xy") or [], dtype=float)
    if half_extents.shape != (2,) and all(
        key in usable for key in ("x", "y")
    ):
        half_extents = 0.5 * np.asarray(
            [usable["x"], usable["y"]], dtype=float
        )
    if half_extents.shape != (2,) or np.any(half_extents < 0.0):
        raise ValueError("terminal reservation usable region is invalid")
    return (
        [*world_xy.tolist(), float(root_z), *target_q],
        {
            "reservation_region_id": str(region.get("id") or ""),
            "reservation_unit_id": str(unit.get("id") or ""),
            "reservation_unit_yaw_deg": unit_yaw_deg,
            "reservation_frame_yaw_deg": reservation_frame_yaw_deg,
            "usable_half_extents_xy_m": half_extents.tolist(),
            "placement_recipe_binding": {
                key: deepcopy(recipe_binding[key])
                for key in (
                    "placement_recipe",
                    "placement_recipe_source",
                    "placement_recipe_hash",
                    "receiver_state_profile_id",
                    "receiver_state_profile_hash",
                )
            },
            "terminal_frame_binding": terminal_frame,
        },
    )


def _terminal_relation_geometry(
    scene_spec: Mapping[str, Any],
    carry_step: Mapping[str, Any],
    actor_pose: Sequence[float],
) -> dict[str, Any] | None:
    """Evaluate a planar terminal with the final Checker's exact predicate."""

    relation = str(carry_step["relation"])
    if relation not in SURFACE_GAP_RELATIONS | ANCHOR_PAIR_RELATIONS:
        return None
    subject = str(carry_step["subject"])
    receivers = exact_carry_receiver_keys(carry_step)
    expected_receiver_count = 2 if relation in ANCHOR_PAIR_RELATIONS else 1
    if len(receivers) != expected_receiver_count:
        raise ValueError(
            f"{relation} terminal requires exactly "
            f"{expected_receiver_count} receiver(s)"
        )
    entities = scene_spec.get("entities") or {}
    if subject not in entities or any(key not in entities for key in receivers):
        raise ValueError("terminal relation geometry entities are missing")
    subject_aabb = _entity_world_aabb(entities[subject], actor_pose)
    receiver_aabbs = {
        key: _entity_world_aabb(entities[key]) for key in receivers
    }
    acceptance = _relation_geometry_acceptance(
        subject_aabb, receiver_aabbs, relation, receivers
    )
    return {
        "contract_version": RELATION_SURFACE_GAP_CONTRACT_VERSION,
        "relation": relation,
        "subject": subject,
        "receiver_keys": receivers,
        "subject_aabb": subject_aabb,
        "receiver_aabbs": receiver_aabbs,
        **acceptance,
    }


def _relation_geometry_acceptance(
    subject_aabb: Mapping[str, Any],
    receiver_aabbs: Mapping[str, Mapping[str, Any]],
    relation: str,
    receiver_keys: Sequence[str],
) -> dict[str, Any]:
    """Dispatch one exact terminal to the shared SuccessContract kernel."""

    normalized = str(relation)
    keys = [str(value) for value in receiver_keys]
    if normalized in SURFACE_GAP_RELATIONS:
        if len(keys) != 1 or keys[0] not in receiver_aabbs:
            raise ValueError("surface-gap relation receiver observation is missing")
        return relation_surface_gap_acceptance(
            subject_aabb,
            receiver_aabbs[keys[0]],
            normalized,
        )
    if normalized in ANCHOR_PAIR_RELATIONS:
        if len(keys) != 2 or any(key not in receiver_aabbs for key in keys):
            raise ValueError("anchor-pair relation receiver observations are missing")
        return relation_anchor_pair_acceptance(
            subject_aabb,
            receiver_aabbs[keys[0]],
            receiver_aabbs[keys[1]],
            normalized,
        )
    raise ValueError(f"terminal relation geometry is unsupported: {normalized}")


def terminal_candidate_from_sources(
    unit_artifact: Mapping[str, Any],
    scene_spec: Mapping[str, Any],
    carry_step: Mapping[str, Any],
    *,
    offset_xy_m: Sequence[float],
    yaw_delta_deg: float,
    cavity_world_matrix: Sequence[Sequence[float]] | None = None,
    receiver_profiles: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Reconstruct one terminal from its exact M11/SceneSpec lineage."""

    base, lineage = reservation_terminal_base_pose(
        unit_artifact,
        scene_spec,
        carry_step,
        cavity_world_matrix=cavity_world_matrix,
        receiver_profiles=receiver_profiles,
    )
    offset = np.asarray(offset_xy_m, dtype=float)
    relation = str(carry_step["relation"])
    if offset.shape != (2,) or not np.isfinite(offset).all():
        raise ValueError("terminal offset must contain two finite values")
    if not any(
        np.allclose(offset, candidate, atol=1e-12, rtol=0.0)
        for candidate in terminal_search_offsets(relation)
    ):
        raise ValueError("terminal offset is outside the frozen search domain")
    yaw_delta = float(yaw_delta_deg)
    if not any(
        math.isclose(yaw_delta, candidate, abs_tol=1e-12)
        for candidate in terminal_search_yaws(relation)
    ):
        raise ValueError("terminal yaw is outside the frozen search domain")
    half_extents = np.asarray(lineage["usable_half_extents_xy_m"], dtype=float)
    if np.any(np.abs(offset) > half_extents + 1e-9):
        raise TerminalCandidateRejected(
            "terminal offset lies outside the M11 usable region"
        )
    frame_yaw = math.radians(float(lineage["reservation_frame_yaw_deg"]))
    world_offset = np.asarray(
        [
            math.cos(frame_yaw) * offset[0]
            - math.sin(frame_yaw) * offset[1],
            math.sin(frame_yaw) * offset[0]
            + math.cos(frame_yaw) * offset[1],
        ],
        dtype=float,
    )
    actor_pose = [
        *(np.asarray(base[:3], dtype=float) + [*world_offset, 0.0]).tolist(),
        *_quat_multiply(_z_quaternion(yaw_delta), base[3:]),
    ]
    relation_geometry = _terminal_relation_geometry(
        scene_spec, carry_step, actor_pose
    )
    if (
        relation_geometry is not None
        and relation_geometry["passed"] is not True
    ):
        raise TerminalCandidateRejected(
            "terminal relation geometry rejected by shared SuccessContract"
        )
    terminal = {
        "actor_pose": actor_pose,
        "offset_xy_m": offset.tolist(),
        "world_offset_xy_m": world_offset.tolist(),
        "yaw_delta_deg": yaw_delta,
        "semantic_precheck": (
            "inside_reservation_and_shared_relation_geometry"
            if relation_geometry is not None
            else "inside_reservation_usable_region"
        ),
        "relation_geometry": relation_geometry,
        "reservation_region_id": lineage["reservation_region_id"],
        "reservation_unit_id": lineage["reservation_unit_id"],
        "reservation_unit_yaw_deg": lineage["reservation_unit_yaw_deg"],
        "reservation_frame_yaw_deg": lineage["reservation_frame_yaw_deg"],
        "reservation_usable_half_extents_xy_m": lineage[
            "usable_half_extents_xy_m"
        ],
        "placement_recipe_binding": lineage["placement_recipe_binding"],
        "terminal_frame_binding": lineage["terminal_frame_binding"],
    }
    terminal["reservation_lineage_hash"] = artifact_hash(
        {"base_actor_pose": base, "terminal_candidate": terminal}
    )
    return terminal


def route_terminal_admissibility(
    actual_actor_pose: Sequence[float],
    terminal_candidate: Mapping[str, Any],
    *,
    actual_subject_aabb: Mapping[str, Any] | None = None,
    actual_receiver_aabbs: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Evaluate live terminal tracking and the frozen reservation boundary."""

    actual = np.asarray(actual_actor_pose, dtype=float)
    target = np.asarray(terminal_candidate.get("actor_pose") or [], dtype=float)
    if actual.shape != (7,) or target.shape != (7,):
        raise ValueError("terminal tracking requires two xyz+quat poses")
    if not np.isfinite(actual).all() or not np.isfinite(target).all():
        raise ValueError("terminal tracking poses must be finite")
    translation = float(np.linalg.norm(actual[:3] - target[:3]))
    actual_norm = float(np.linalg.norm(actual[3:]))
    target_norm = float(np.linalg.norm(target[3:]))
    if actual_norm <= 1e-12 or target_norm <= 1e-12:
        raise ValueError("terminal tracking quaternion norm is zero")
    actual_q = actual[3:] / actual_norm
    target_q = target[3:] / target_norm
    cosine = min(1.0, max(-1.0, abs(float(np.dot(actual_q, target_q)))))
    orientation = math.degrees(2.0 * math.acos(cosine))
    world_offset = np.asarray(
        terminal_candidate.get("world_offset_xy_m") or [], dtype=float
    )
    half_extents = np.asarray(
        terminal_candidate.get("reservation_usable_half_extents_xy_m") or [],
        dtype=float,
    )
    if world_offset.shape != (2,) or half_extents.shape != (2,):
        raise ValueError("terminal reservation admissibility lineage is missing")
    base_xy = target[:2] - world_offset
    delta_world = actual[:2] - base_xy
    yaw = math.radians(
        float(terminal_candidate.get("reservation_frame_yaw_deg"))
    )
    local_offset = np.asarray(
        [
            math.cos(yaw) * delta_world[0] + math.sin(yaw) * delta_world[1],
            -math.sin(yaw) * delta_world[0] + math.cos(yaw) * delta_world[1],
        ],
        dtype=float,
    )
    policy = ROUTE_EXECUTION_CONTRACT["terminal_tracking"]
    margin = float(policy["reservation_margin_m"])
    inside = bool(np.all(np.abs(local_offset) <= half_extents + margin + 1e-9))
    translation_ok = translation <= float(policy["translation_tolerance_m"])
    orientation_ok = orientation <= float(policy["orientation_tolerance_deg"])
    relation_geometry = terminal_candidate.get("relation_geometry")
    live_relation_geometry: dict[str, Any] | None = None
    relation_geometry_ok = True
    if relation_geometry is not None:
        if not isinstance(relation_geometry, Mapping) or (
            relation_geometry.get("contract_version")
            != RELATION_SURFACE_GAP_CONTRACT_VERSION
        ):
            raise ValueError("terminal relation geometry lineage is stale")
        receiver_keys = [
            str(value)
            for value in relation_geometry.get("receiver_keys") or []
        ]
        if actual_subject_aabb is None or not isinstance(
            actual_receiver_aabbs, Mapping
        ):
            raise ValueError(
                "live terminal relation geometry observation is missing"
            )
        live_relation_geometry = _relation_geometry_acceptance(
            actual_subject_aabb,
            actual_receiver_aabbs,
            str(relation_geometry.get("relation") or ""),
            receiver_keys,
        )
        relation_geometry_ok = bool(live_relation_geometry["passed"])
    return {
        "passed": bool(
            inside
            and translation_ok
            and orientation_ok
            and relation_geometry_ok
        ),
        "inside_reservation": inside,
        "translation_ok": bool(translation_ok),
        "orientation_ok": bool(orientation_ok),
        "relation_geometry_ok": relation_geometry_ok,
        "live_relation_geometry": live_relation_geometry,
        "translation_error_m": translation,
        "orientation_error_deg": orientation,
        "actual_reservation_offset_xy_m": local_offset.tolist(),
        "translation_tolerance_m": float(policy["translation_tolerance_m"]),
        "orientation_tolerance_deg": float(policy["orientation_tolerance_deg"]),
        "reservation_half_extents_xy_m": half_extents.tolist(),
    }


def held_collision_descriptor_from_aabb(
    bounds: Mapping[str, Sequence[float]],
    *,
    name: str,
) -> dict[str, Any]:
    """Build the one conservative held-object proxy used by route and M15."""

    minimum = np.asarray(bounds.get("min") or [], dtype=float)
    maximum = np.asarray(bounds.get("max") or [], dtype=float)
    center = np.asarray(bounds.get("center") or [], dtype=float)
    if any(value.shape != (3,) for value in (minimum, maximum, center)):
        raise ValueError("held collision AABB must contain min/max/center xyz")
    if (
        not np.isfinite(minimum).all()
        or not np.isfinite(maximum).all()
        or not np.isfinite(center).all()
        or np.any(maximum <= minimum)
    ):
        raise ValueError("held collision AABB is invalid")
    envelope = float(
        ROUTE_EXECUTION_CONTRACT["held_collision"]["dimension_envelope_m"]
    )
    padding = float(
        ROUTE_EXECUTION_CONTRACT["held_collision"]["planner_padding_m"]
    )
    return {
        "name": str(name),
        "dims": (maximum - minimum + 2.0 * envelope).tolist(),
        "pose": [*center.tolist(), 1.0, 0.0, 0.0, 0.0],
        "padding_m": padding,
        "source": "live_world_axis_tight_aabb_shared_route_carry",
        "route_execution_contract": route_execution_contract_identity(),
    }


def held_collision_descriptor(actor: Any, *, name: str) -> dict[str, Any]:
    from if_benchmark.collision_geometry import collision_aabb_tight

    return held_collision_descriptor_from_aabb(
        collision_aabb_tight(actor), name=name
    )


def route_scene_collision_descriptor(
    entity_spec: Mapping[str, Any], actor: Any, *, name: str
) -> dict[str, Any]:
    """Build the identical unpadded live OBB used by route proof and M15."""

    initial = entity_spec.get("initial_pose") or {}
    bounds = initial.get("origin_relative_bounds") or {}
    minimum = np.asarray(bounds.get("min_xyz") or [], dtype=float)
    maximum = np.asarray(bounds.get("max_xyz") or [], dtype=float)
    q0 = np.asarray(initial.get("q0_wxyz") or [], dtype=float)
    if minimum.shape != (3,) or maximum.shape != (3,) or q0.shape != (4,):
        raise ValueError(f"route collision bounds are missing for {name}")
    dimensions = maximum - minimum
    if np.any(dimensions <= 0.0) or not np.isfinite(dimensions).all():
        raise ValueError(f"route collision bounds are invalid for {name}")
    pose = actor.get_pose()
    live_position = np.asarray(pose.p, dtype=float)
    live_quaternion = np.asarray(pose.q, dtype=float)
    if live_position.shape != (3,) or live_quaternion.shape != (4,):
        raise ValueError(f"route collision live pose is invalid for {name}")
    delta_rotation = _quat_matrix(live_quaternion) @ _quat_matrix(q0).T
    relative_center = 0.5 * (minimum + maximum)
    world_center = live_position + delta_rotation @ relative_center
    q0_norm = q0 / np.linalg.norm(q0)
    delta_quaternion = _quat_multiply(
        live_quaternion,
        [q0_norm[0], -q0_norm[1], -q0_norm[2], -q0_norm[3]],
    )
    return {
        "name": str(name),
        "dims": dimensions.tolist(),
        "pose": [*world_center.tolist(), *delta_quaternion],
        "padding_m": 0.0,
        "geometry_hash": initial.get("geometry_hash"),
        "source": "shared_route_m15_r01_live_obb_unpadded",
        "route_execution_contract": route_execution_contract_identity(),
    }


def route_scene_collision_obstacles(
    scene_spec: Mapping[str, Any],
    entities: Mapping[str, Any],
    *,
    excluded_entity_keys: Sequence[str],
    complete_geometry: bool = False,
    max_obstacles: int = 59,
) -> list[dict[str, Any]]:
    excluded = {str(value) for value in excluded_entity_keys}
    specs = scene_spec.get("entities") or {}
    if complete_geometry:
        from if_benchmark.grasp_validation.world_geometry import (
            live_world_geometry, world_piece_obstacles, verify_world_piece_obstacles,
        )
        retained = [(key, actor) for key, actor in sorted(entities.items()) if key not in excluded]
        if any(key not in specs for key, _ in retained):
            raise ValueError("live collision entity is missing its frozen SceneSpec identity")
        geometries = [live_world_geometry(actor, name=key) for key, actor in retained]
        rows = world_piece_obstacles(geometries, max_obstacles=max_obstacles)
        verify_world_piece_obstacles(rows, geometries, max_obstacles=max_obstacles)
        return rows
    return [
        route_scene_collision_descriptor(specs[key], actor, name=key)
        for key, actor in sorted(entities.items())
        if key not in excluded
    ]


def route_collision_excluded_entity_keys(
    *,
    subject: str,
    receiver_keys: Sequence[str],
    relation: str,
    phase: str,
) -> frozenset[str]:
    excluded = {str(subject)}
    if (
        str(relation) in VERTICAL_CONTACT_RELATIONS
        and str(phase) == "descent"
    ):
        excluded.update(str(value) for value in receiver_keys if str(value))
    return frozenset(excluded)


def route_phase_actor_poses(
    initial_actor_matrix: Sequence[Sequence[float]],
    terminal_actor_matrix: Sequence[Sequence[float]],
    *,
    relation: str,
) -> dict[str, np.ndarray]:
    initial = np.asarray(initial_actor_matrix, dtype=float)
    terminal = np.asarray(terminal_actor_matrix, dtype=float)
    if initial.shape != (4, 4) or terminal.shape != (4, 4):
        raise ValueError("route actor poses must be 4x4 matrices")
    lift_delta = route_lift_delta_m(relation)
    approach_delta = float(
        ROUTE_EXECUTION_CONTRACT["phase_offsets_m"]["approach"]
    )
    transit_z = max(
        float(initial[2, 3]) + lift_delta,
        float(terminal[2, 3]) + approach_delta,
    )
    result = {
        "lift": initial.copy(),
        "transit": terminal.copy(),
        "approach": terminal.copy(),
        "descent": terminal.copy(),
    }
    result["lift"][2, 3] += lift_delta
    result["transit"][2, 3] = transit_z
    result["approach"][2, 3] += approach_delta
    return result


def release_retreat_ee_matrix(
    descent_ee_matrix: Sequence[Sequence[float]],
) -> np.ndarray:
    """Apply the authored arm-frame +50 mm retreat to an EE matrix."""

    descent = np.asarray(descent_ee_matrix, dtype=float)
    if descent.shape != (4, 4) or not np.isfinite(descent).all():
        raise ValueError("descent EE pose must be a finite 4x4 matrix")
    result = descent.copy()
    delta = float(ROUTE_EXECUTION_CONTRACT["release_retreat"]["delta_m"])
    result[:3, 3] -= delta * descent[:3, 0]
    return result


def route_witness_hash(witness: Mapping[str, Any]) -> str:
    payload = deepcopy(dict(witness))
    payload.pop("witness_hash", None)
    return artifact_hash(payload)


def _finite_vector(value: Any, length: int, field: str) -> list[float]:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (length,) or not np.isfinite(vector).all():
        raise ValueError(f"route witness {field} must contain {length} finite values")
    return vector.tolist()


def _validate_se3(value: Any, field: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{field} must be a finite 4x4 matrix")
    if not np.allclose(
        matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-10, rtol=0.0
    ):
        raise ValueError(f"{field} homogeneous row is invalid")
    rotation = matrix[:3, :3]
    if not np.allclose(
        rotation.T @ rotation, np.eye(3), atol=1e-8, rtol=0.0
    ) or not math.isclose(
        float(np.linalg.det(rotation)), 1.0, abs_tol=1e-8
    ):
        raise ValueError(f"{field} rotation is not SO(3)")
    return matrix


def validate_route_witness_report(
    report: Mapping[str, Any],
    *,
    scene_spec_id: str,
    task_record_id: str,
    source_hashes: Mapping[str, str],
    skill_config: Mapping[str, Any],
    unit_artifact: Mapping[str, Any],
    scene_spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one normative route report against its exact M13 task join."""

    value = deepcopy(dict(report))
    observed_report_hash = str(value.pop("report_hash", ""))
    if not observed_report_hash or content_hash(value) != observed_report_hash:
        raise ValueError("route report self-hash mismatch")
    value["report_hash"] = observed_report_hash
    if value.get("artifact_kind") != "exact_scene_route_witness":
        raise ValueError("route report artifact kind is invalid")
    if value.get("version") != ROUTE_VALIDATION_VERSION:
        raise ValueError("route report version is stale")
    if value.get("status") != "witness_found":
        raise ValueError("route report does not contain a selected witness")
    if str(value.get("scene_spec_id")) != str(scene_spec_id):
        raise ValueError("route report SceneSpec id mismatch")
    if str(value.get("task_record_id")) != str(task_record_id):
        raise ValueError("route report TaskRecord id mismatch")
    if dict(value.get("source_hashes") or {}) != {
        str(key): str(item) for key, item in source_hashes.items()
    }:
        raise ValueError("route report source hashes are stale")
    if value.get("route_execution_contract") != route_execution_contract_identity():
        raise ValueError("route execution contract is stale")
    if not str(value.get("m12_pick_planning_request_hash") or ""):
        raise ValueError("route M12 Pick planning-request hash is missing")
    if not str(value.get("scene_validation_config_hash") or ""):
        raise ValueError("route report scene-validation config hash is missing")

    steps = list(skill_config.get("steps") or [])
    carry_steps = [step for step in steps if step.get("op") == "carry_to_relation"]
    grasp_steps = [step for step in steps if step.get("op") == "grasp"]
    release_steps = [step for step in steps if step.get("op") == "release"]
    retreat_steps = [
        step for step in steps if step.get("op") == "retreat_relative"
    ]
    if str(skill_config.get("route")) != "direct" or len(carry_steps) != 1:
        raise ValueError("route witness requires one direct carry step")
    carry = carry_steps[0]
    subject = str(carry.get("subject") or "")
    arm = str(carry.get("arm") or "")
    relation = str(carry.get("relation") or "")
    receivers = exact_carry_receiver_keys(carry)
    if len(grasp_steps) != 1 or any(
        str(grasp_steps[0].get(key) or "") != expected
        for key, expected in (("subject", subject), ("arm", arm))
    ) or not math.isclose(
        float(grasp_steps[0].get("total_lift_delta_m", -1.0)),
        route_lift_delta_m(relation),
        abs_tol=1e-12,
    ):
        raise ValueError("route witness grasp/carry identity mismatch")
    retreat_policy = route_release_retreat_policy()
    if (
        len(release_steps) != 1
        or len(retreat_steps) != 1
        or str(release_steps[0].get("arm") or "") != arm
        or str(retreat_steps[0].get("arm") or "") != arm
        or str(retreat_steps[0].get("frame") or "")
        != retreat_policy["frame"]
        or str(retreat_steps[0].get("axis") or "")
        != retreat_policy["axis"]
        or not math.isclose(
            float(retreat_steps[0].get("delta_m", -1.0)),
            float(retreat_policy["delta_m"]),
            abs_tol=1e-12,
        )
    ):
        raise ValueError("route release-retreat Skill IR contract mismatch")
    expected_identity = {
        "arm": arm,
        "route": "direct",
        "relation": relation,
        "subject": subject,
        "receivers": receivers,
    }
    if any(value.get(key) != expected for key, expected in expected_identity.items()):
        raise ValueError("route report task identity mismatch")

    selected = value.get("selected_witness")
    if not isinstance(selected, Mapping):
        raise ValueError("route report selected witness is missing")
    selected = deepcopy(dict(selected))
    if route_witness_hash(selected) != selected.get("witness_hash"):
        raise ValueError("route selected witness self-hash mismatch")
    if int(selected.get("contact_id", -1)) < 0:
        raise ValueError("route selected witness Contact is invalid")
    terminal = selected.get("terminal_candidate")
    if not isinstance(terminal, Mapping):
        raise ValueError("route selected terminal candidate is missing")
    terminal = deepcopy(dict(terminal))
    _finite_vector(terminal.get("actor_pose"), 7, "terminal actor pose")
    _finite_vector(terminal.get("offset_xy_m"), 2, "terminal offset")
    if not math.isfinite(float(terminal.get("yaw_delta_deg"))):
        raise ValueError("route selected terminal yaw is invalid")
    terminal_frame = terminal.get("terminal_frame_binding")
    cavity_world_matrix = (
        None
        if terminal_frame is None
        else validate_terminal_frame_binding(terminal_frame)
    )
    expected_terminal = terminal_candidate_from_sources(
        unit_artifact,
        scene_spec,
        carry,
        offset_xy_m=terminal["offset_xy_m"],
        yaw_delta_deg=float(terminal["yaw_delta_deg"]),
        cavity_world_matrix=cavity_world_matrix,
    )
    if terminal != expected_terminal:
        raise ValueError(
            "route selected terminal does not derive from M11 reservation lineage"
        )
    _validate_se3(
        selected.get("actor_to_ee_transform"),
        "route selected actor-to-EE transform",
    )
    phase_actor_poses = selected.get("phase_actor_poses")
    if not isinstance(phase_actor_poses, Mapping) or set(
        phase_actor_poses
    ) != set(ROUTE_ACTOR_PHASES):
        raise ValueError("route selected actor phase targets are incomplete")
    for phase in ROUTE_ACTOR_PHASES:
        _finite_vector(
            phase_actor_poses[phase], 7, f"route {phase} actor target"
        )
    subject_spec = (scene_spec.get("entities") or {}).get(subject) or {}
    initial_spec_pose = subject_spec.get("pose") or {}
    initial_pose = np.asarray(
        [
            *_finite_vector(
                initial_spec_pose.get("position"),
                3,
                "SceneSpec subject initial position",
            ),
            *_finite_vector(
                initial_spec_pose.get("quat_wxyz"),
                4,
                "SceneSpec subject initial quaternion",
            ),
        ],
        dtype=float,
    )
    terminal_pose = np.asarray(terminal["actor_pose"], dtype=float)
    expected_phase_matrices = route_phase_actor_poses(
        _actor_pose_matrix(initial_pose),
        _actor_pose_matrix(terminal_pose),
        relation=relation,
    )
    for phase, expected in expected_phase_matrices.items():
        observed = _actor_pose_matrix(phase_actor_poses[phase])
        if not np.allclose(observed, expected, atol=1e-10, rtol=0.0):
            raise ValueError(
                f"route {phase} actor target does not derive from its "
                "SceneSpec/terminal phase policy"
            )
    if selected.get("phase_policy") != route_phase_policy(relation):
        raise ValueError("route selected phase policy is stale")
    paths = selected.get("paths")
    if not isinstance(paths, Mapping) or set(paths) != set(ROUTE_PATH_PHASES):
        raise ValueError("route selected witness phase chain is incomplete")
    planner_contracts: dict[str, dict[str, Any]] = {}
    for phase in ROUTE_PATH_PHASES:
        path = paths[phase]
        if (
            not isinstance(path, Mapping)
            or int(path.get("steps", 0)) < 1
            or not str(path.get("trajectory_hash") or "")
        ):
            raise ValueError(f"route selected {phase} path is invalid")
        planner = path.get("planner_contract")
        if not isinstance(planner, Mapping) or set(planner) != {
            "expected_world_hash",
            "observed_world_hash",
            "expected_attachment_hash",
            "observed_attachment_hash",
        }:
            raise ValueError(f"route selected {phase} planner contract is missing")
        planner = dict(planner)
        if (
            not str(planner["expected_world_hash"] or "")
            or planner["expected_world_hash"]
            != planner["observed_world_hash"]
            or planner["expected_attachment_hash"]
            != planner["observed_attachment_hash"]
        ):
            raise ValueError(f"route selected {phase} planner contract is stale")
        planner_contracts[phase] = planner
    for phase in ("pre_grasp", "grasp", "release_retreat"):
        if planner_contracts[phase]["expected_attachment_hash"] is not None:
            raise ValueError(f"route selected {phase} must be unattached")
    attached_hashes = {
        planner_contracts[phase]["expected_attachment_hash"]
        for phase in ("lift", "transit", "approach", "descent")
    }
    if len(attached_hashes) != 1 or None in attached_hashes:
        raise ValueError("route attached carry phase lineage is inconsistent")
    strict_world = planner_contracts["transit"]["expected_world_hash"]
    if any(
        planner_contracts[phase]["expected_world_hash"] != strict_world
        for phase in ("lift", "approach", "release_retreat")
    ):
        raise ValueError("route strict receiver world was not restored")
    if (
        relation not in VERTICAL_CONTACT_RELATIONS
        and planner_contracts["descent"]["expected_world_hash"] != strict_world
    ):
        raise ValueError("route planar descent changed the collision world")
    if (
        relation in VERTICAL_CONTACT_RELATIONS
        and planner_contracts["descent"]["expected_world_hash"] == strict_world
    ):
        raise ValueError("route vertical descent did not exclude its receiver")
    return value


__all__ = [
    "ROUTE_ACTOR_PHASES",
    "ROUTE_CARRY_PHASES",
    "ROUTE_EXECUTION_CONTRACT",
    "ROUTE_EXECUTION_CONTRACT_HASH",
    "ROUTE_EXECUTION_CONTRACT_VERSION",
    "ROUTE_PATH_PHASES",
    "ROUTE_VALIDATION_VERSION",
    "SURFACE_GAP_RELATIONS",
    "TerminalCandidateRejected",
    "VERTICAL_CONTACT_RELATIONS",
    "exact_carry_receiver_keys",
    "exact_reservation_receiver_keys",
    "held_collision_descriptor",
    "held_collision_descriptor_from_aabb",
    "release_retreat_ee_matrix",
    "reservation_terminal_base_pose",
    "route_collision_excluded_entity_keys",
    "route_execution_contract_identity",
    "route_lift_delta_m",
    "route_phase_actor_poses",
    "route_phase_policy",
    "route_release_retreat_policy",
    "route_scene_collision_descriptor",
    "route_scene_collision_obstacles",
    "route_terminal_admissibility",
    "route_witness_hash",
    "terminal_candidate_from_sources",
    "terminal_search_offsets",
    "terminal_search_yaws",
    "validate_route_witness_report",
]
