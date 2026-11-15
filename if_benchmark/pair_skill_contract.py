"""Pure contracts shared by canonical pair-skill planning, workers and tests."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any


HANG_ENTRY_POLICY_ID = "m5_hang_stepped_entry_v1"
HANG_RELATION_POLICY_VERSION = (
    "m5_hang_on_v21_stepped_functional_axis_entry_and_release_gate"
)


INFRASTRUCTURE_FAILURES = {
    "worker_import_failed",
    "worker_exception",
    "worker_process_failed",
    "worker_timeout",
    "worker_no_json_result",
    "setup_failed",
    "actor_load_failed",
    "missing_profile",
    "stale_profile",
}

NON_SEMANTIC_FAILURES = INFRASTRUCTURE_FAILURES | {
    "source_initial_pose_mismatch",
}

VALID_RELATION_FAILURES = {
    "motion_planning_failed",
    "grasp_failed",
    "entry_collision",
    "drop",
    "checker_false",
    "release_failed",
    "settle_failed",
    "joint_state_drift",
    "forbidden_support_contact",
    "hang_pre_release_alignment_failed",
}


def hang_skill_phase_contract(probe_spec: dict[str, Any]) -> dict[str, Any]:
    """Return the fixed Hang trajectory phases and functional-axis offsets."""

    return {
        "entry_policy_id": HANG_ENTRY_POLICY_ID,
        "legacy_approach_distance_m": float(
            probe_spec.get("insertion_depth_m", 0.05)
        ),
        "legacy_insert_distance_m": -0.05,
        "legacy_phases": [
            "hang_insert",
            "hang_retract",
        ],
        "approach_distance_m": 0.10,
        "insert_distances_m": [
            0.05,
            0.0,
            -0.025,
            -0.05,
        ],
        "terminal_insert_distance_m": -0.05,
        "translation_tolerance_m": 0.02,
        "rotation_tolerance_deg": 10.0,
        "max_corrections": 1,
        "require_readiness_before_release": True,
        "phases": [
            "hang_approach",
            "hang_insert_before_release",
            "hang_release",
            "hang_retract",
        ],
    }


def canonical_scene_fixture(job: dict[str, Any], seed: int) -> dict[str, Any]:
    """Return the only scene layout allowed by the M5 canonical skill gate."""

    relation = str(job["relation"])
    arm = str(job.get("selected_arm") or "left")
    # Deterministic, bounded subject-only jitter. Receiver pose never changes.
    dx = (((int(seed) * 17 + 3) % 11) - 5) * 0.001
    dy = (((int(seed) * 29 + 7) % 11) - 5) * 0.001
    receiver_profile = job.get("receiver_state_profile") or {}
    if receiver_profile:
        fixture = receiver_profile.get("fixture") or {}
        receiver_xy = [float(value) for value in fixture["table_xy"]]
        receiver_z = float(fixture["root_z"])
        receiver_q = [float(value) for value in fixture["root_q0_wxyz"]]
        source = "receiver_state_profile.fixture"
    else:
        # RoboTwin's reviewed placement tasks use the robot-reachable front
        # table strip around y=-0.10.  Positive y=+0.10 pushes a side-grasped
        # object's EE goal beyond the canonical arm workspace.
        if relation == "hang_on":
            receiver_xy = [0.18, 0.15]
        elif relation == "place_in":
            # Keep the pick pose in M4's validated arm-side x region. The
            # receiver sits just across table center so large receivers cannot
            # overlap an elongated subject at setup.
            receiver_xy = [0.05 if arm == "left" else -0.05, -0.10]
        else:
            receiver_xy = [-0.05 if arm == "left" else 0.05, -0.10]
        receiver_z = 0.741
        receiver_q = None
        source = "canonical_clean_table"
    # RoboTwin's left arm operates on negative table x and the right arm on
    # positive x (the same fixed regions used by the M4 pick probes).
    if receiver_profile:
        subject_abs_x, subject_y = 0.20, -0.10
        fixture_class = "articulated_default"
        classified_pick_footprint = None
    else:
        if relation == "place_in":
            classified_pick_footprint = subject_pick_footprint(job)
            if classified_pick_footprint is None:
                footprint = (job.get("geometry_metrics") or {}).get("subject_local_footprint") or {}
                try:
                    classified_pick_footprint = [
                        float(footprint.get("x") or 0.0),
                        float(footprint.get("y") or 0.0),
                    ]
                except (TypeError, ValueError):
                    classified_pick_footprint = None
            (
                subject_abs_x,
                subject_y,
                fixture_class,
            ) = _rigid_place_in_pickup_lane(
                job,
                receiver_xy=receiver_xy,
                pick_footprint_xy=classified_pick_footprint,
            )
        else:
            subject_abs_x, subject_y = 0.20, -0.10
            fixture_class = "rigid_default"
            classified_pick_footprint = None
    subject_x = -subject_abs_x if arm == "left" else subject_abs_x
    return {
        "receiver": {
            "xy": receiver_xy,
            "z": receiver_z,
            "q_wxyz": receiver_q,
            "random_yaw": False,
            "source": source,
        },
        "subject": {
            "xy": [subject_x + dx, subject_y + dy],
            "jitter_xy": [dx, dy],
            "selected_arm": arm,
            "fixture_class": fixture_class,
            "classified_pick_footprint_xy_m": classified_pick_footprint,
        },
    }


def subject_pick_footprint(job: dict[str, Any]) -> list[float] | None:
    """Predict the table-plane AABB in the exact M4-validated pick pose.

    Terminal geometry stores an AABB in the relation's canonical quaternion.
    For the axis-aligned quarter-turn poses used by the catalog, that AABB can
    be un-permuted into local box dimensions and then projected through the
    M4-validated quaternion. This distinguishes an upright bottle from the
    same model validated on its side without loading the asset at runtime.
    """

    geometry = job.get("geometry_metrics") or {}
    footprint = geometry.get("subject_footprint") or {}
    try:
        terminal_world_dimensions = [
            float(footprint["x"]),
            float(footprint["y"]),
            float(geometry["subject_height"]),
        ]
    except (KeyError, TypeError, ValueError):
        return None
    terminal_pose = (
        ((job.get("probe_spec") or {}).get("canonical_poses") or {}).get("subject")
        or {}
    )
    terminal_q = terminal_pose.get("q0_wxyz")
    pick_q = _selected_m4_pick_quaternion(job)
    if terminal_q is None or pick_q is None:
        return None
    local_dimensions = _unpermute_axis_aligned_aabb(
        terminal_world_dimensions,
        terminal_q,
    )
    if local_dimensions is None:
        return None
    rotation = _absolute_rotation_matrix(pick_q)
    projected = [
        sum(rotation[row][column] * local_dimensions[column] for column in range(3))
        for row in range(3)
    ]
    if not all(math.isfinite(value) and value > 0.0 for value in projected):
        return None
    return [round(projected[0], 12), round(projected[1], 12)]


def _rigid_place_in_pickup_lane(
    job: dict[str, Any],
    *,
    receiver_xy: list[float],
    pick_footprint_xy: list[float] | None,
) -> tuple[float, float, str]:
    """Choose one of two reviewed lanes using frozen AABB clearance.

    The receiver remains fixed. The subject may use either the selected arm's
    outer side or the table-front corridor. This is a deterministic fixture
    choice from recorded geometry, not a runtime pose search.
    """

    try:
        subject_x = float((pick_footprint_xy or [])[0])
        subject_y = float((pick_footprint_xy or [])[1])
    except (IndexError, TypeError, ValueError):
        subject_x = subject_y = 0.0
    longest_pick_axis = max(subject_x, subject_y)
    shape_class = "elongated" if longest_pick_axis >= 0.18 else "compact"
    receiver_footprint = (job.get("geometry_metrics") or {}).get("receiver_outer_footprint") or {}
    try:
        receiver_x = float(receiver_footprint["x"])
        receiver_y = float(receiver_footprint["y"])
    except (KeyError, TypeError, ValueError):
        if shape_class == "elongated":
            return 0.16, -0.24, "rigid_place_in_elongated_front"
        return 0.20, -0.10, "rigid_place_in_compact_side"
    if min(subject_x, subject_y, receiver_x, receiver_y) <= 0.0:
        if shape_class == "elongated":
            return 0.16, -0.24, "rigid_place_in_elongated_front"
        return 0.20, -0.10, "rigid_place_in_compact_side"

    clearance = 0.025
    # Side lane: place the subject beyond the receiver's arm-side edge.
    arm_sign = -1.0 if str(job.get("selected_arm") or "left") == "left" else 1.0
    receiver_outer_edge = arm_sign * float(receiver_xy[0]) + receiver_x / 2.0
    side_abs_x = max(
        0.20,
        receiver_outer_edge + subject_x / 2.0 + clearance,
    )
    side_y = float(receiver_xy[1])
    side_table_margin = 0.46 - (side_abs_x + subject_x / 2.0)

    # Front lane: retain the reviewed x corridor, but derive y from the two
    # AABBs so large receivers cannot overlap the initial pick pose.
    front_abs_x = 0.16
    front_y = float(receiver_xy[1]) - receiver_y / 2.0 - subject_y / 2.0 - clearance
    front_table_margin = (front_y - subject_y / 2.0) - (-0.36)

    if front_table_margin > side_table_margin:
        return front_abs_x, front_y, f"rigid_place_in_{shape_class}_front"
    return side_abs_x, side_y, f"rigid_place_in_{shape_class}_side"


def table_aligned_root_position(
    *,
    root_xyz: list[float],
    aabb_center_xyz: list[float],
    aabb_min_z: float,
    target_xy: list[float],
    table_top_z: float,
    clearance_m: float = 0.002,
) -> list[float]:
    """Center an actor's AABB on the fixture and lift its bottom off the table."""

    root = [float(value) for value in root_xyz]
    center = [float(value) for value in aabb_center_xyz]
    root[0] += float(target_xy[0]) - center[0]
    root[1] += float(target_xy[1]) - center[1]
    root[2] += float(table_top_z) + float(clearance_m) - float(aabb_min_z)
    return root


def aabb_centered_root_position_xy(
    *,
    root_xyz: list[float],
    aabb_center_xyz: list[float],
    target_xy: list[float],
) -> list[float]:
    """Translate a root so its current AABB center lands on fixture xy."""

    root = [float(value) for value in root_xyz]
    center = [float(value) for value in aabb_center_xyz]
    root[0] += float(target_xy[0]) - center[0]
    root[1] += float(target_xy[1]) - center[1]
    return root


def setup_stability_decision(
    *,
    unstable_actor_names: list[str],
    subject_actor_name: str,
    has_validated_m4_pick_pose: bool,
) -> dict[str, Any]:
    """Delegate only the validated subject pose to matching M4 evidence."""

    unstable = [str(name) for name in unstable_actor_names]
    delegated = bool(
        has_validated_m4_pick_pose and str(subject_actor_name) in unstable
    )
    remaining = [
        name for name in unstable if not (delegated and name == str(subject_actor_name))
    ]
    return {
        "setup_stable": not remaining,
        "delegated_m4_subject_stability": delegated,
        "remaining_unstable_actor_names": remaining,
    }


def articulated_entry_waypoints(
    profile: dict[str, Any],
    *,
    cavity_world_xyz: list[float],
    final_world_xyz: list[float],
    cavity_world_q_wxyz: list[float],
) -> dict[str, list[float]]:
    """Build approach/final/retreat points from the reviewed cavity frame."""

    skill = profile.get("skill") or {}
    entry_mode = str(skill.get("entry_mode"))
    approach_distance = float(skill["approach_distance_m"])
    retreat_distance = float(skill["retreat_distance_m"])
    if entry_mode == "top_down":
        entry_direction = [0.0, 0.0, -1.0]
        approach = _add(final_world_xyz, [0.0, 0.0, approach_distance])
        retreat = _add(final_world_xyz, [0.0, 0.0, retreat_distance])
    elif entry_mode == "front_insertion":
        opening = (profile.get("cavity") or {}).get("opening_plane") or {}
        normal_local = _normalize(opening["outward_normal_xyz"])
        normal_world = rotate_vector(cavity_world_q_wxyz, normal_local)
        opening_point_world = transform_point(
            cavity_world_xyz,
            cavity_world_q_wxyz,
            opening["point_xyz"],
        )
        approach = _add(opening_point_world, _scale(normal_world, approach_distance))
        retreat = _add(opening_point_world, _scale(normal_world, retreat_distance))
        entry_direction = _scale(normal_world, -1.0)
    else:
        raise ValueError(f"unsupported articulated entry mode: {entry_mode}")
    return {
        "approach_world_xyz": approach,
        "final_world_xyz": [float(value) for value in final_world_xyz],
        "retreat_world_xyz": retreat,
        "entry_direction_world": entry_direction,
    }


def transform_point(
    origin_xyz: list[float],
    q_wxyz: list[float],
    point_xyz: list[float],
) -> list[float]:
    return _add(origin_xyz, rotate_vector(q_wxyz, point_xyz))


def rotate_vector(q_wxyz: list[float], vector_xyz: list[float]) -> list[float]:
    w, x, y, z = _normalize_quaternion(q_wxyz)
    vx, vy, vz = [float(value) for value in vector_xyz]
    # q * v * q^-1, expanded.
    return [
        (1 - 2 * (y * y + z * z)) * vx + 2 * (x * y - z * w) * vy + 2 * (x * z + y * w) * vz,
        2 * (x * y + z * w) * vx + (1 - 2 * (x * x + z * z)) * vy + 2 * (y * z - x * w) * vz,
        2 * (x * z - y * w) * vx + 2 * (y * z + x * w) * vy + (1 - 2 * (x * x + y * y)) * vz,
    ]


def _selected_m4_pick_quaternion(job: dict[str, Any]) -> list[float] | None:
    expected_use = f"pick_{str(job.get('selected_arm') or 'left')}"
    for evidence in job.get("m4_evidence_refs") or []:
        if str(evidence.get("use")) != expected_use:
            continue
        pose = evidence.get("validated_pose") or {}
        q_wxyz = pose.get("q_wxyz")
        if isinstance(q_wxyz, list) and len(q_wxyz) == 4:
            try:
                return [float(value) for value in q_wxyz]
            except (TypeError, ValueError):
                return None
    return None


def _absolute_rotation_matrix(q_wxyz: list[float]) -> list[list[float]]:
    w, x, y, z = _normalize_quaternion(q_wxyz)
    return [
        [
            abs(1 - 2 * (y * y + z * z)),
            abs(2 * (x * y - z * w)),
            abs(2 * (x * z + y * w)),
        ],
        [
            abs(2 * (x * y + z * w)),
            abs(1 - 2 * (x * x + z * z)),
            abs(2 * (y * z - x * w)),
        ],
        [
            abs(2 * (x * z - y * w)),
            abs(2 * (y * z + x * w)),
            abs(1 - 2 * (x * x + y * y)),
        ],
    ]


def _unpermute_axis_aligned_aabb(
    world_dimensions: list[float],
    q_wxyz: list[float],
) -> list[float] | None:
    """Invert an AABB projection when the quaternion is an axis permutation."""

    rotation = _absolute_rotation_matrix(q_wxyz)
    local_dimensions = [0.0, 0.0, 0.0]
    used_columns: set[int] = set()
    for world_axis, row in enumerate(rotation):
        local_axis = max(range(3), key=lambda index: row[index])
        coefficient = row[local_axis]
        if coefficient < 1.0 - 1e-5 or local_axis in used_columns:
            return None
        if any(value > 1e-5 for index, value in enumerate(row) if index != local_axis):
            return None
        local_dimensions[local_axis] = float(world_dimensions[world_axis]) / coefficient
        used_columns.add(local_axis)
    return local_dimensions if len(used_columns) == 3 else None


def world_direction_to_target_local(
    target_q_wxyz: list[float],
    world_direction_xyz: list[float],
) -> list[float]:
    """Express a desired world-space approach direction in target-local axes.

    ``Base_Task.get_place_pose`` treats a numeric ``pre_dis_axis`` as a vector
    in the target-pose frame.  Skill contracts are easier to review in world
    coordinates, so this conversion must happen exactly once at the adapter.
    """

    w, x, y, z = _normalize_quaternion(target_q_wxyz)
    return _normalize(rotate_vector([w, -x, -y, -z], world_direction_xyz))


def terminal_root_xyz(
    *,
    current_root_xyz: list[float],
    current_aabb: dict[str, list[float]],
    anchor_xyz: list[float],
    support_top_z: float,
    clearance_m: float,
) -> list[float]:
    """Reconstruct the root translation used by Terminal settle placement."""

    root = [float(value) for value in current_root_xyz]
    center = current_aabb.get("center")
    if center is None:
        center = [
            (float(current_aabb["min"][index]) + float(current_aabb["max"][index])) / 2.0
            for index in range(3)
        ]
    return [
        float(anchor_xyz[0]) - (float(center[0]) - root[0]),
        float(anchor_xyz[1]) - (float(center[1]) - root[1]),
        root[2] - float(current_aabb["min"][2]) + float(support_top_z) + float(clearance_m),
    ]


def aggregate_skill_attempts(
    job: dict[str, Any],
    attempts: list[dict[str, Any]],
    *,
    seeds: list[int],
    threshold: float = 0.6,
) -> dict[str, Any]:
    """Aggregate the fixed three-seed canonical skill gate."""

    by_seed = {int(row["seed"]): row for row in attempts if row.get("seed") is not None}
    ordered = [by_seed[seed] for seed in seeds if seed in by_seed]
    reasons = Counter(str(row.get("failure_reason") or "checker_false") for row in ordered if not row.get("success"))
    blocked = [
        row
        for row in ordered
        if str(row.get("failure_reason") or "") in NON_SEMANTIC_FAILURES
        or row.get("outcome") == "blocked"
    ]
    complete = len(ordered) == len(seeds)
    valid = [row for row in ordered if row not in blocked]
    success = sum(
        bool(row.get("plan_success"))
        and bool(row.get("released"))
        and bool(row.get("settled"))
        and bool(row.get("checker_success", row.get("success")))
        for row in valid
    )
    rate = success / len(valid) if valid else 0.0
    if blocked or not complete or len(valid) != len(seeds):
        outcome = "blocked"
        passed: bool | None = None
        verdict = str((blocked[0] if blocked else {}).get("failure_reason") or "incomplete_skill_attempts")
    elif rate >= float(threshold):
        outcome, passed, verdict = "pass", True, "ok"
    else:
        outcome, passed = "fail", False
        verdict = reasons.most_common(1)[0][0] if reasons else "checker_false"
    band_id = str(job.get("band_id") or (job.get("yaw_witness") or {}).get("band_id") or "fixed")
    row = {
        "scope": "pair",
        "stage": "canonical_skill",
        "relation": str(job["relation"]),
        "subject": _endpoint(job.get("subject") or {}),
        "receiver": _endpoint(job.get("receiver") or {}),
        "pair_key": str(job["pair_key"]),
        "job_id": str(job["job_id"]),
        "outcome": outcome,
        "pass": passed,
        "verdict": verdict,
        "threshold": float(threshold),
        "seeds": [int(seed) for seed in seeds],
        "attempts": len(ordered),
        "valid_attempts": len(valid),
        "plan_success": sum(bool(row.get("plan_success")) for row in valid),
        "success": success,
        "min_rate": rate,
        "fail_reasons": dict(sorted(reasons.items())),
        "validated_skill_recipe_id": job.get("recipe_id"),
        "validated_skill_placement_recipe": job.get("placement_recipe") or {},
        "validated_skill_band_ids": [band_id] if passed else [],
        "skill_preconditions": job.get("skill_preconditions") or {},
        "m4_evidence_refs": job.get("m4_evidence_refs") or [],
        "skill_probe_policy_hash": job.get("skill_probe_policy_hash"),
        "skill_relation_policy_hash": job.get(
            "skill_relation_policy_hash"
        ),
        "metrics": {
            "rate": rate,
            "per_seed": [
                {
                    "seed": row.get("seed"),
                    "plan_success": bool(row.get("plan_success")),
                    "released": bool(row.get("released")),
                    "settled": bool(row.get("settled")),
                    "checker_success": bool(row.get("checker_success", row.get("success"))),
                    "failure_reason": row.get("failure_reason"),
                }
                for row in ordered
            ],
        },
    }
    if job.get("initial_state_selection_hash"):
        row["initial_state_selection_version"] = job.get(
            "initial_state_selection_version"
        )
        row["initial_state_selection_hash"] = job[
            "initial_state_selection_hash"
        ]
    for key in ("receiver_state_profile_id", "receiver_state_profile_hash"):
        if job.get(key) is not None:
            row[key] = job[key]
    return row


def _endpoint(value: dict[str, Any]) -> dict[str, Any]:
    endpoint = {
        "asset": str(value.get("asset")),
        "model_id": str(value.get("model_id")),
    }
    # Pair probe plans use category/inventory_id to disambiguate endpoint
    # identities.  The aggregated report must preserve those optional fields;
    # otherwise a physically valid row can no longer be bound back to the
    # exact source job that produced it.
    for key in ("category", "inventory_id"):
        if value.get(key) is not None:
            endpoint[key] = value[key]
    return endpoint


def _normalize_quaternion(value: list[float]) -> list[float]:
    values = [float(item) for item in value]
    norm = math.sqrt(sum(item * item for item in values))
    if norm <= 1e-12:
        raise ValueError("zero quaternion")
    return [item / norm for item in values]


def _normalize(value: list[float]) -> list[float]:
    values = [float(item) for item in value]
    norm = math.sqrt(sum(item * item for item in values))
    if norm <= 1e-12:
        raise ValueError("zero vector")
    return [item / norm for item in values]


def _add(left: list[float], right: list[float]) -> list[float]:
    return [float(a) + float(b) for a, b in zip(left, right)]


def _subtract(left: list[float], right: list[float]) -> list[float]:
    return [float(a) - float(b) for a, b in zip(left, right)]


def _scale(value: list[float], factor: float) -> list[float]:
    return [float(item) * float(factor) for item in value]
