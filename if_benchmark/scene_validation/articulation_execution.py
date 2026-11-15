"""Simulator-independent policy for R09 closed-loop articulation execution."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import yaml

from if_benchmark.schema.serialization import artifact_hash, canonical_data
from if_benchmark.grasp_validation.core import (
    grasp_candidate_sort_key,
    joint_path_metrics,
)
from if_benchmark.grasp_validation.contact_contract import CONTACT_TO_GRASP_FRAME
from if_benchmark.execution_policy import (
    UNIFIED_ARTICULATION_EXECUTION_VERSION,
)
from if_benchmark.success_contract import SuccessContract, resolve_articulation_goal


ARTICULATION_EXECUTION_PROFILE_VERSION = "articulation_execution_profiles_v1"
ARTICULATION_MOTION_POLICY_VERSION = UNIFIED_ARTICULATION_EXECUTION_VERSION
CONTINUOUS_ARTICULATION_WAYPOINT_COUNT = 12
CONTINUOUS_ARTICULATION_PATH_SAMPLES_PER_SUBPATH = 24
CONTINUOUS_ARTICULATION_MAXIMUM_BOUNDARY_JUMP_RAD = 0.05
ARTICULATION_BILATERAL_FRAME_VERSION = (
    "moving_link_contact_component_bilateral_surface_frame_v3"
)
DEFAULT_ARTICULATION_EXECUTION_PROFILE = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "scene_validation"
    / "articulation_execution_profiles_v1.yml"
)


class ArticulationExecutionError(ValueError):
    """The reviewed articulation execution policy cannot be consumed safely."""


def articulation_motion_policy_binding() -> dict[str, Any]:
    """Return the exact continuous-motion contract shared by M4 and M15."""

    payload = canonical_data(
        {
            "version": ARTICULATION_MOTION_POLICY_VERSION,
            "planning": "hypothetical_predecessor_pose_chain",
            "retiming": "single_global_topp_250hz",
            "execution": "single_continuous_joint_replay",
            "servo_waypoint_count": CONTINUOUS_ARTICULATION_WAYPOINT_COUNT,
            "path_samples_per_subpath": (
                CONTINUOUS_ARTICULATION_PATH_SAMPLES_PER_SUBPATH
            ),
            "maximum_boundary_jump_rad": (
                CONTINUOUS_ARTICULATION_MAXIMUM_BOUNDARY_JUMP_RAD
            ),
            "simulator_state_projection": False,
            "terminal_candidate_selection": (
                "preferred_then_toward_near_interval_boundary"
            ),
            "acquisition_candidate_selection": (
                "all_authored_contacts_with_contact_component_surface_frame"
            ),
            "grasp_orientation": ARTICULATION_BILATERAL_FRAME_VERSION,
        }
    )
    return canonical_data({**payload, "hash": artifact_hash(payload)})


def load_articulation_execution_catalog(
    path: Path | str = DEFAULT_ARTICULATION_EXECUTION_PROFILE,
) -> dict[str, Any]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, Mapping):
        raise ArticulationExecutionError("articulation_profile_missing")
    data = canonical_data(dict(value))
    if data.get("profile_version") != ARTICULATION_EXECUTION_PROFILE_VERSION:
        raise ArticulationExecutionError("articulation_profile_version_mismatch")
    if data.get("review_status") != "reviewed":
        raise ArticulationExecutionError("articulation_profile_not_reviewed")
    source_row = data.get("source") or {}
    source_path = Path(str(source_row.get("asset_catalog_path") or ""))
    if not source_path.is_absolute():
        source_path = Path.cwd() / source_path
    if not source_path.is_file():
        raise ArticulationExecutionError("articulation_profile_source_missing")
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if digest != str(source_row.get("asset_catalog_sha256") or ""):
        raise ArticulationExecutionError("articulation_profile_hash_mismatch")
    data["profile_hash"] = artifact_hash(data)
    data["path"] = str(source)
    return data


def resolve_articulation_execution_profile(
    asset: str,
    model_id: str | int,
    use: str,
    *,
    contract: SuccessContract,
    catalog: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    catalog_value = dict(catalog or load_articulation_execution_catalog())
    key = f"{asset}|{model_id}"
    model = (catalog_value.get("models") or {}).get(key)
    if not isinstance(model, Mapping):
        raise ArticulationExecutionError("articulation_profile_missing")
    use_row = (model.get("uses") or {}).get(str(use))
    if not isinstance(use_row, Mapping):
        raise ArticulationExecutionError("articulation_profile_missing")
    expected_contract = str(catalog_value.get("success_contract_hash") or "")
    if expected_contract != contract.hash:
        raise ArticulationExecutionError("articulation_profile_hash_mismatch")
    profile = canonical_data(
        {
            **dict(use_row),
            "asset": str(asset),
            "model_id": str(model_id),
            "use": str(use),
            "joint_index": int(model["joint_index"]),
            "moving_link": str(model["moving_link"]),
            "profile_version": catalog_value["profile_version"],
            "source_asset_catalog_sha256": catalog_value["source"][
                "asset_catalog_sha256"
            ],
            "source_success_contract_hash": expected_contract,
            "motion_policy_version": ARTICULATION_MOTION_POLICY_VERSION,
            "motion_policy_hash": articulation_motion_policy_binding()["hash"],
        }
    )
    profile["profile_hash"] = artifact_hash(profile)
    _validate_execution_profile(profile)
    return profile


def _validate_execution_profile(profile: Mapping[str, Any]) -> None:
    motion_policy = articulation_motion_policy_binding()
    if (
        profile.get("motion_policy_version") != motion_policy["version"]
        or profile.get("motion_policy_hash") != motion_policy["hash"]
    ):
        raise ArticulationExecutionError("articulation_motion_policy_mismatch")
    if profile.get("engagement_mode") not in {
        "grasp_follow",
        "push_follow",
        "switch_press",
    }:
        raise ArticulationExecutionError("articulation_profile_invalid")
    if int(profile.get("maximum_acquisition_attempts", 0)) <= 0:
        raise ArticulationExecutionError("articulation_profile_invalid")
    if profile.get("engagement_mode") == "grasp_follow":
        if profile.get("grasp_orientation_mode") != ARTICULATION_BILATERAL_FRAME_VERSION:
            raise ArticulationExecutionError("articulation_profile_invalid")
        for field in (
            "minimum_closing_width_m",
            "maximum_closing_width_m",
            "maximum_contact_shape_distance_m",
            "local_surface_patch_radius_m",
            "minimum_opposed_normal_cosine",
        ):
            value = float(profile.get(field, 0.0))
            if not np.isfinite(value) or value <= 0.0:
                raise ArticulationExecutionError("articulation_profile_invalid")
        if float(profile["minimum_closing_width_m"]) >= float(
            profile["maximum_closing_width_m"]
        ):
            raise ArticulationExecutionError("articulation_profile_invalid")
        if not 0.0 < float(profile["minimum_opposed_normal_cosine"]) <= 1.0:
            raise ArticulationExecutionError("articulation_profile_invalid")
        if any(
            abs(float(value)) > 1e-12
            for value in profile.get("grasp_roll_candidates_rad", [0.0])
        ):
            raise ArticulationExecutionError("articulation_profile_invalid")
        if set(profile.get("bilateral_frame_variants") or []) != {
            "authored_closing_sign",
            "opposite_closing_sign",
        }:
            raise ArticulationExecutionError("articulation_profile_invalid")
        margin = float(profile.get("minimum_normalized_joint_margin", -1.0))
        if not np.isfinite(margin) or margin < 0.0:
            raise ArticulationExecutionError("articulation_profile_invalid")
    if profile.get("post_acquisition_attachment") == "fixed_constraint":
        if profile.get("engagement_mode") != "grasp_follow":
            raise ArticulationExecutionError("articulation_profile_invalid")
        if (
            profile.get("attachment_execution_projection")
            != "continuous_d6_follow_no_state_projection"
        ):
            raise ArticulationExecutionError("articulation_profile_invalid")
        positive_fields = (
            "guarded_approach_steps",
            "attachment_creation_settle_steps",
            "attachment_observation_steps",
            "attachment_creation_qpos_jump_ratio",
            "attached_arm_stiffness",
            "attached_arm_damping",
        )
        if any(float(profile.get(field, 0.0)) <= 0.0 for field in positive_fields):
            raise ArticulationExecutionError("articulation_profile_invalid")
        if any(
            float(profile.get(field, -1.0)) < 0.0
            for field in ("attachment_drive_stiffness", "attachment_drive_damping")
        ):
            raise ArticulationExecutionError("articulation_profile_invalid")
    segments = profile.get("segments")
    if not isinstance(segments, Sequence) or not segments:
        raise ArticulationExecutionError("articulation_profile_invalid")
    for segment in segments:
        if not isinstance(segment, Mapping):
            raise ArticulationExecutionError("articulation_profile_invalid")
        ids = segment.get("contact_ids")
        if not isinstance(ids, Sequence) or not ids:
            raise ArticulationExecutionError("articulation_profile_invalid")
        if "target_ratio" not in segment and not segment.get("final_target"):
            raise ArticulationExecutionError("articulation_profile_invalid")


def articulation_acquisition_candidates(
    profile: Mapping[str, Any],
    contact_ids: Sequence[Any],
    *,
    validated_candidates: Sequence[Mapping[str, Any]] | None = None,
) -> list[tuple[Any, float, float, str]]:
    """Return complete authored bases, or exact frozen M4 candidates for M15.

    Grasp-follow discovery starts from every reviewed Contact-distance pair.
    Its full gripper orientation is derived later from live moving-link
    collision geometry; ``roll_rad=0`` remains only a compatibility field in
    the frozen candidate schema.  A validated M15 run receives only the exact
    successful M4 Contact-distance candidates frozen in its evidence.
    """

    ids = list(contact_ids)
    distances = list(
        profile.get("pre_grasp_candidates_m", [profile.get("pre_grasp_m")])
    )
    rolls = list(profile.get("grasp_roll_candidates_rad", [0.0]))
    maximum = int(profile.get("maximum_acquisition_attempts", 0))
    if (
        not ids
        or not distances
        or any(value is None for value in distances)
        or not rolls
        or maximum <= 0
    ):
        raise ArticulationExecutionError("articulation_profile_invalid")
    grasp_follow = str(profile.get("engagement_mode")) == "grasp_follow"
    if grasp_follow:
        variants = list(profile.get("bilateral_frame_variants") or [])
        if not variants or set(variants) != {
            "authored_closing_sign",
            "opposite_closing_sign",
        }:
            raise ArticulationExecutionError("articulation_profile_invalid")
        schedule = [
            (contact_id, float(distance), 0.0, str(variant))
            for contact_id in ids
            for distance in distances
            for variant in variants
        ]
        allowed_rolls = {0.0, *(float(value) for value in rolls)}
        allowed_variants = set(variants)
    else:
        schedule = [
            (contact_id, float(distance), float(roll), "authored_contact_frame")
            for contact_id in ids
            for distance in distances
            for roll in rolls
        ]
        allowed_rolls = {float(value) for value in rolls}
        allowed_variants = {"authored_contact_frame"}
    if len(schedule) > maximum:
        raise ArticulationExecutionError("articulation_profile_invalid")
    if validated_candidates is None:
        return schedule
    exact: list[tuple[int, float, float, str]] = []
    seen: set[tuple[int, float, float, str]] = set()
    for row in validated_candidates:
        if (
            not isinstance(row, Mapping)
            or not isinstance(row.get("candidate"), Mapping)
            or str(row.get("moving_link") or "")
            != str(profile.get("moving_link") or "")
            or str(row.get("engagement_mode") or "")
            != str(profile.get("engagement_mode") or "")
        ):
            raise ArticulationExecutionError(
                "articulation_candidate_execution_profile_stale"
            )
        candidate = row["candidate"]
        try:
            value = (
                int(candidate["contact_id"]),
                float(candidate["pre_grasp_m"]),
                float(candidate["roll_rad"]),
                str(candidate["orientation_variant"]),
            )
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ArticulationExecutionError(
                "articulation_candidate_execution_profile_stale"
            ) from exc
        if (
            value[0] not in {int(contact_id) for contact_id in ids}
            or value[1] not in {float(distance) for distance in distances}
            or value[2] not in allowed_rolls
            or value[3] not in allowed_variants
        ):
            raise ArticulationExecutionError(
                "articulation_candidate_execution_profile_stale"
            )
        if value not in seen:
            seen.add(value)
            exact.append(value)
    return exact


def _articulation_acquisition_rank(
    candidate: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    joint_limits: Sequence[Sequence[float]] | None,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    path = np.asarray(plan.get("position"), dtype=float)
    if path.ndim != 2 or not len(path) or not np.isfinite(path).all():
        raise ArticulationExecutionError("articulation_continuous_path_invalid")
    if joint_limits is None:
        deltas = np.abs(np.diff(path, axis=0))
        metrics = {
            "minimum_joint_limit_margin": 0.0,
            "maximum_normalized_joint_step": (
                0.0 if not len(deltas) else float(np.max(deltas))
            ),
            "normalized_total_travel": float(np.sum(deltas)),
            "path_segment_count": 1,
            "path_waypoint_count": int(len(path)),
        }
    else:
        metrics = canonical_data(joint_path_metrics([path], joint_limits))
    ranked = {
        "approach_class": "unbiased",
        "path_metrics": metrics,
        "cartesian_path_length_m": float(plan.get("duration_sec") or 0.0),
        "contact_id": int(candidate["contact_id"]),
        "contact_group_indices": [],
        "retreat_distance_m": float(candidate["pre_grasp_m"]),
    }
    key = (
        *grasp_candidate_sort_key(ranked),
        abs(float(candidate["roll_rad"])),
        float(candidate["roll_rad"]),
        str(candidate.get("orientation_variant") or ""),
    )
    return key, metrics


def articulation_arm_joint_limits(
    robot: Any,
    arm: str,
) -> list[list[float]]:
    """Read the same bounded arm-joint domain used by rigid Pick scoring."""

    arm_name = str(arm)
    if arm_name not in {"left", "right"}:
        raise ArticulationExecutionError("articulation_arm_invalid")
    joints = (
        robot.left_arm_joints
        if arm_name == "left"
        else robot.right_arm_joints
    )
    result: list[list[float]] = []
    for joint in joints:
        bounds = np.asarray(joint.get_limits(), dtype=float).reshape(-1, 2)
        if not len(bounds) or not np.isfinite(bounds[0]).all():
            raise ArticulationExecutionError(
                "articulation_arm_joint_limits_invalid"
            )
        result.append([float(bounds[0, 0]), float(bounds[0, 1])])
    if not result:
        raise ArticulationExecutionError(
            "articulation_arm_joint_limits_invalid"
        )
    return result


def select_articulation_acquisition_plan(
    candidates: Sequence[Mapping[str, Any]],
    *,
    compile_candidate: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    joint_limits: Sequence[Sequence[float]] | None = None,
    minimum_normalized_joint_margin: float = 0.0,
    maximum_plans: int | None = None,
) -> dict[str, Any]:
    """Plan every geometry-resolved authored candidate, then replay once.

    Planning misses are not physical contact misses.  Keeping the two phases
    separate prevents a failed Pre-Grasp IK/path query from being published as
    ``articulation_contact_not_acquired`` and prevents retry/retreat motions
    from mutating the robot before another candidate is evaluated.  Every
    Contact orientation is already resolved from moving-link geometry before
    this function; there is no blind wrist-roll search phase.
    """

    attempts: list[dict[str, Any]] = []
    passed: list[dict[str, Any]] = []

    def evaluate(candidate: Mapping[str, Any], phase: str) -> None:
        candidate_index = len(attempts)
        plan = dict(compile_candidate(candidate))
        status = str(plan.get("status") or "failed")
        failure_code = plan.get("failure_code")
        rank_key = None
        metrics = None
        if status == "passed":
            try:
                rank_key, metrics = _articulation_acquisition_rank(
                    candidate, plan, joint_limits=joint_limits
                )
                if (
                    joint_limits is not None
                    and float(metrics["minimum_joint_limit_margin"])
                    < float(minimum_normalized_joint_margin)
                ):
                    status = "failed"
                    failure_code = "articulation_unsafe_joint_margin"
            except (ArticulationExecutionError, ValueError) as exc:
                status = "failed"
                failure_code = str(exc)
        row = canonical_data(
            {
                "candidate_index": candidate_index,
                "search_phase": phase,
                "contact_id": int(candidate["contact_id"]),
                "pre_grasp_m": float(candidate["pre_grasp_m"]),
                "roll_rad": float(candidate["roll_rad"]),
                "orientation_variant": str(
                    candidate.get("orientation_variant") or ""
                ),
                "grasp_orientation_mode": candidate.get(
                    "grasp_orientation_mode"
                ),
                "bilateral_geometry": candidate.get("bilateral_geometry"),
                "status": status,
                "failure_code": failure_code,
                "stage": plan.get("stage"),
                "planned_subpaths": plan.get("planned_subpaths"),
                "geometric_waypoints": plan.get("geometric_waypoints"),
                "retimed_waypoints": plan.get("retimed_waypoints"),
                "duration_sec": plan.get("duration_sec"),
                "collision": plan.get("collision"),
                "path_metrics": metrics,
            }
        )
        attempts.append(row)
        if status == "passed":
            passed.append(
                {
                    "candidate": candidate,
                    "plan": plan,
                    "rank_key": rank_key,
                    "attempt_index": candidate_index,
                }
            )

    base_candidates = list(candidates)
    plan_limit = None if maximum_plans is None else int(maximum_plans)
    if plan_limit is not None and len(base_candidates) > plan_limit:
        raise ArticulationExecutionError("articulation_profile_invalid")
    for candidate in base_candidates:
        evaluate(candidate, "bilateral_geometry")
    if passed:
        selected = min(passed, key=lambda row: row["rank_key"])
        attempts[selected["attempt_index"]]["selected"] = True
        return {
            "status": "passed",
            "failure_code": None,
            "candidate": selected["candidate"],
            "plan": selected["plan"],
            "planning_attempts": canonical_data(attempts),
        }
    return {
        "status": "failed",
        "failure_code": "articulation_no_pre_grasp_path",
        "planning_attempts": attempts,
    }


def segment_targets(
    profile: Mapping[str, Any], goal: Mapping[str, Any]
) -> list[dict[str, Any]]:
    direction = int(goal["direction_sign"])
    initial_ratio = float(goal["initial_ratio"])
    final_ratio = float(goal["target_ratio"])
    out: list[dict[str, Any]] = []
    previous = initial_ratio
    for index, raw in enumerate(profile["segments"]):
        row = canonical_data(dict(raw))
        ratio = final_ratio if row.get("final_target") else float(row["target_ratio"])
        if direction * (ratio - previous) <= 0.0:
            raise ArticulationExecutionError("articulation_segment_not_monotonic")
        if direction * (final_ratio - ratio) < -1e-9:
            raise ArticulationExecutionError("articulation_segment_beyond_target")
        row["segment_index"] = index
        row["resolved_target_ratio"] = ratio
        row["resolved_target_qpos"] = float(goal["lower"]) + float(goal["span"]) * ratio
        out.append(row)
        previous = ratio
    if abs(previous - final_ratio) > 1e-9:
        raise ArticulationExecutionError("articulation_final_segment_missing")
    return out


def preengagement_arm_fallback_allowed(result: Mapping[str, Any]) -> bool:
    """Permit an arm fallback only after a physically inert acquisition miss."""

    if result.get("failure_code") not in {
        "articulation_contact_not_acquired",
        "articulation_no_pre_grasp_path",
    }:
        return False
    goal = result.get("goal") or {}
    span = float(goal.get("span") or 0.0)
    tolerance = span * 0.001
    acquire_rows = [
        row for row in (result.get("trace") or []) if row.get("phase") == "acquire"
    ]
    return bool(acquire_rows) and all(
        abs(float(row.get("after_qpos", 0.0)) - float(row.get("before_qpos", 0.0)))
        <= tolerance
        for row in acquire_rows
    )


def future_contact_matrix(
    current_contact_matrix: Sequence[Sequence[float]],
    *,
    joint_type: str,
    joint_axis_world: Sequence[float],
    joint_pivot_world: Sequence[float],
    q_delta: float,
) -> np.ndarray:
    """Predict a moving-link Contact pose without mutating simulator qpos."""

    current = np.asarray(current_contact_matrix, dtype=float)
    axis = np.asarray(joint_axis_world, dtype=float)
    pivot = np.asarray(joint_pivot_world, dtype=float)
    if current.shape != (4, 4) or axis.shape != (3,) or pivot.shape != (3,):
        raise ArticulationExecutionError("articulation_joint_frame_invalid")
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12:
        raise ArticulationExecutionError("articulation_joint_axis_invalid")
    axis = axis / norm
    kind = str(joint_type).lower()
    if "prismatic" in kind:
        transform = np.eye(4, dtype=float)
        transform[:3, 3] = axis * float(q_delta)
        return transform @ current
    if "revolute" not in kind and "continuous" not in kind:
        raise ArticulationExecutionError("articulation_joint_type_unsupported")
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    angle = float(q_delta)
    rotation = (
        np.eye(3, dtype=float)
        + np.sin(angle) * skew
        + (1.0 - np.cos(angle)) * (skew @ skew)
    )
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation
    transform[:3, 3] = pivot - rotation @ pivot
    return transform @ current


def fixed_attachment_frames(
    parent_world_matrix: Sequence[Sequence[float]],
    child_world_matrix: Sequence[Sequence[float]],
    *,
    anchor_world_matrix: Sequence[Sequence[float]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build coincident local frames for a zero-snap fixed constraint."""

    parent_world = np.asarray(parent_world_matrix, dtype=float)
    child_world = np.asarray(child_world_matrix, dtype=float)
    anchor_world = np.asarray(
        child_world if anchor_world_matrix is None else anchor_world_matrix,
        dtype=float,
    )
    if any(matrix.shape != (4, 4) for matrix in (parent_world, child_world, anchor_world)):
        raise ArticulationExecutionError("articulation_attachment_frame_invalid")
    return (
        np.linalg.inv(parent_world) @ anchor_world,
        np.linalg.inv(child_world) @ anchor_world,
    )


def fixed_attachment_residual(
    parent_world_matrix: Sequence[Sequence[float]],
    child_world_matrix: Sequence[Sequence[float]],
    parent_local_frame: Sequence[Sequence[float]],
    child_local_frame: Sequence[Sequence[float]],
) -> dict[str, float]:
    """Measure the live disagreement between the two fixed-joint anchors."""

    parent_anchor = np.asarray(parent_world_matrix, dtype=float) @ np.asarray(
        parent_local_frame, dtype=float
    )
    child_anchor = np.asarray(child_world_matrix, dtype=float) @ np.asarray(
        child_local_frame, dtype=float
    )
    if parent_anchor.shape != (4, 4) or child_anchor.shape != (4, 4):
        raise ArticulationExecutionError("articulation_attachment_frame_invalid")
    relative_rotation = parent_anchor[:3, :3] @ child_anchor[:3, :3].T
    cosine = max(
        -1.0,
        min(1.0, (float(np.trace(relative_rotation)) - 1.0) / 2.0),
    )
    return {
        "translation_m": float(
            np.linalg.norm(parent_anchor[:3, 3] - child_anchor[:3, 3])
        ),
        "rotation_deg": float(np.degrees(np.arccos(cosine))),
    }


def guarded_approach_matrices(
    start_matrix: Sequence[Sequence[float]],
    end_matrix: Sequence[Sequence[float]],
    *,
    steps: int,
) -> list[np.ndarray]:
    """Split the final Contact approach into bounded, deterministic SE(3) steps."""

    start = np.asarray(start_matrix, dtype=float)
    end = np.asarray(end_matrix, dtype=float)
    count = int(steps)
    if start.shape != (4, 4) or end.shape != (4, 4) or count <= 0:
        raise ArticulationExecutionError("articulation_guarded_approach_invalid")
    start_rotation = start[:3, :3]
    end_rotation = end[:3, :3]
    relative = start_rotation.T @ end_rotation
    cosine = max(-1.0, min(1.0, (float(np.trace(relative)) - 1.0) / 2.0))
    angle = float(np.arccos(cosine))
    if np.allclose(start_rotation, end_rotation, rtol=0.0, atol=1e-8):
        angle = 0.0
        axis = np.asarray([1.0, 0.0, 0.0], dtype=float)
    elif abs(np.pi - angle) <= 1e-6:
        eigenvalues, eigenvectors = np.linalg.eigh(relative)
        axis = np.asarray(
            eigenvectors[:, int(np.argmin(np.abs(eigenvalues - 1.0)))],
            dtype=float,
        )
    else:
        axis = np.asarray(
            [
                relative[2, 1] - relative[1, 2],
                relative[0, 2] - relative[2, 0],
                relative[1, 0] - relative[0, 1],
            ],
            dtype=float,
        ) / (2.0 * np.sin(angle))
    axis_norm = float(np.linalg.norm(axis))
    if not np.isfinite(axis_norm) or axis_norm <= 1e-12:
        raise ArticulationExecutionError("articulation_guarded_approach_invalid")
    axis = axis / axis_norm
    out: list[np.ndarray] = []
    for index in range(1, count + 1):
        ratio = index / count
        matrix = np.eye(4, dtype=float)
        matrix[:3, :3] = start_rotation @ _axis_angle_matrix(axis, angle * ratio)
        matrix[:3, 3] = start[:3, 3] + ratio * (end[:3, 3] - start[:3, 3])
        out.append(matrix)
    return out


def continuous_articulation_qpos_targets(
    current_qpos: float,
    target_qpos: float,
    *,
    waypoint_count: int,
) -> list[float]:
    """Return deterministic interior-to-terminal samples for one motion.

    These values are geometric checkpoints, not execution segments.  The
    environment adapter plans them from hypothetical predecessor states,
    stitches the resulting joint paths, globally retimes the path, and only
    then performs one physical replay.
    """

    current = float(current_qpos)
    target = float(target_qpos)
    count = int(waypoint_count)
    if (
        count <= 0
        or not np.isfinite(current)
        or not np.isfinite(target)
        or abs(target - current) <= 1e-12
    ):
        raise ArticulationExecutionError(
            "articulation_continuous_path_invalid"
        )
    return [
        current + (target - current) * ((index + 1) / count)
        for index in range(count)
    ]


def continuous_articulation_target_candidates(
    segment: Mapping[str, Any],
    goal: Mapping[str, Any],
) -> list[float]:
    """Return collision-screened terminal alternatives before any replay.

    A final articulation predicate usually admits an interval rather than one
    exact qpos.  The profile's preferred target is tried first.  If that pose
    is unreachable or colliding, later candidates move toward the admissible
    boundary nearest the initial state while preserving the minimum signed
    progress.  Non-final segment waypoints remain exact.
    """

    preferred = float(segment["resolved_target_qpos"])
    if not segment.get("final_target"):
        return [preferred]
    lower = float(goal["lower"])
    span = float(goal["span"])
    initial_ratio = float(goal["initial_ratio"])
    direction = int(goal["direction_sign"])
    interval = [float(value) for value in goal["target_interval"]]
    minimum_progress = float(goal["minimum_signed_normalized_progress"])
    width = interval[1] - interval[0]
    margin = min(width * 0.01, 0.0025)
    near_ratio = (
        interval[0] + margin
        if direction > 0
        else interval[1] - margin
    )
    preferred_ratio = (preferred - lower) / span
    ratios = [
        preferred_ratio,
        preferred_ratio + 0.5 * (near_ratio - preferred_ratio),
        preferred_ratio + 0.75 * (near_ratio - preferred_ratio),
        near_ratio,
    ]
    candidates: list[float] = []
    for ratio in ratios:
        if not interval[0] <= ratio <= interval[1]:
            continue
        if direction * (ratio - initial_ratio) + 1e-12 < minimum_progress:
            continue
        qpos = lower + span * ratio
        if not any(abs(qpos - value) <= 1e-12 for value in candidates):
            candidates.append(qpos)
    if not candidates:
        raise ArticulationExecutionError(
            "articulation_continuous_target_interval_empty"
        )
    return candidates


def stitch_articulation_joint_paths(
    plans: Sequence[Mapping[str, Any]],
    *,
    samples_per_plan: int = CONTINUOUS_ARTICULATION_PATH_SAMPLES_PER_SUBPATH,
    maximum_boundary_jump_rad: float = (
        CONTINUOUS_ARTICULATION_MAXIMUM_BOUNDARY_JUMP_RAD
    ),
) -> np.ndarray:
    """Build one geometric joint path from collision-planned subpaths.

    Sampling limits the input size presented to the global time
    parameterizer while retaining every subpath endpoint.  No physical state
    is changed here and no per-subpath velocity is retained, because a single
    global retiming owns velocity continuity across former boundaries.
    """

    sample_count = int(samples_per_plan)
    maximum_jump = float(maximum_boundary_jump_rad)
    if sample_count < 2 or not np.isfinite(maximum_jump) or maximum_jump <= 0:
        raise ArticulationExecutionError(
            "articulation_continuous_path_invalid"
        )
    chunks: list[np.ndarray] = []
    previous_terminal: np.ndarray | None = None
    width: int | None = None
    for plan in plans:
        if str(plan.get("status")) != "Success":
            raise ArticulationExecutionError(
                "articulation_continuous_path_unreachable"
            )
        positions = np.asarray(plan.get("position"), dtype=float)
        if (
            positions.ndim != 2
            or not len(positions)
            or not np.isfinite(positions).all()
        ):
            raise ArticulationExecutionError(
                "articulation_continuous_path_invalid"
            )
        if width is None:
            width = int(positions.shape[1])
        if int(positions.shape[1]) != width:
            raise ArticulationExecutionError(
                "articulation_continuous_path_invalid"
            )
        if previous_terminal is not None:
            boundary_jump = float(
                np.max(np.abs(positions[0] - previous_terminal))
            )
            if boundary_jump > maximum_jump:
                raise ArticulationExecutionError(
                    "articulation_continuous_path_discontinuous"
                )
        indices = np.unique(
            np.linspace(
                0,
                len(positions) - 1,
                min(sample_count, len(positions)),
                dtype=int,
            )
        )
        sampled = positions[indices]
        if chunks and np.max(np.abs(sampled[0] - chunks[-1][-1])) <= 1e-9:
            sampled = sampled[1:]
        if len(sampled):
            chunks.append(sampled)
        previous_terminal = positions[-1]
    if not chunks:
        raise ArticulationExecutionError(
            "articulation_continuous_path_invalid"
        )
    path = np.concatenate(chunks, axis=0)
    if len(path) < 2:
        raise ArticulationExecutionError(
            "articulation_continuous_path_invalid"
        )
    keep = np.ones(len(path), dtype=bool)
    keep[1:] = np.max(np.abs(np.diff(path, axis=0)), axis=1) > 1e-9
    path = path[keep]
    if len(path) < 2:
        raise ArticulationExecutionError(
            "articulation_continuous_path_invalid"
        )
    return path


@dataclass(frozen=True)
class ContinuousArticulationPlanCallbacks:
    """Environment adapter for simulator-independent continuous compilation."""

    plan_pose: Callable[
        [Sequence[float], np.ndarray | None], Mapping[str, Any]
    ]
    retime_path: Callable[
        [np.ndarray], tuple[np.ndarray, np.ndarray, float]
    ]
    validate_path: Callable[[np.ndarray], Mapping[str, Any]]


@dataclass(frozen=True)
class ContinuousArticulationReplayCallbacks:
    """Environment adapter for one uninterrupted physical joint replay."""

    command_joint: Callable[[np.ndarray, np.ndarray, int], None]
    read_arm_qpos: Callable[[], Sequence[float]]
    observe_checkpoint: Callable[[int, int], Mapping[str, Any] | None]


def compile_continuous_articulation_path(
    target_poses: Sequence[Sequence[float]],
    *,
    runtime: ContinuousArticulationPlanCallbacks,
    stage: str,
) -> dict[str, Any]:
    """Plan all Cartesian checkpoints before globally retiming one path."""

    poses = [list(map(float, pose)) for pose in target_poses]
    if not poses or any(len(pose) != 7 for pose in poses):
        return {
            "status": "failed",
            "failure_code": "articulation_continuous_path_invalid",
            "stage": str(stage),
        }
    plans: list[Mapping[str, Any]] = []
    last_qpos: np.ndarray | None = None
    for pose in poses:
        plan = runtime.plan_pose(pose, last_qpos)
        plans.append(plan)
        if str(plan.get("status")) != "Success":
            return {
                "status": "failed",
                "failure_code": "articulation_continuous_path_unreachable",
                "stage": str(stage),
                "planned_subpaths": len(plans),
            }
        positions = np.asarray(plan.get("position"), dtype=float)
        if positions.ndim != 2 or not len(positions):
            return {
                "status": "failed",
                "failure_code": "articulation_continuous_path_invalid",
                "stage": str(stage),
                "planned_subpaths": len(plans),
            }
        last_qpos = positions[-1]
    try:
        geometry = stitch_articulation_joint_paths(plans)
        position, velocity, duration = runtime.retime_path(geometry)
        position = np.asarray(position, dtype=float)
        velocity = np.asarray(velocity, dtype=float)
        if (
            position.ndim != 2
            or not len(position)
            or velocity.shape != position.shape
            or not np.isfinite(position).all()
            or not np.isfinite(velocity).all()
        ):
            raise ArticulationExecutionError(
                "articulation_continuous_path_invalid"
            )
        collision = canonical_data(dict(runtime.validate_path(position)))
        if not bool(collision.get("collision_free")):
            return {
                "status": "failed",
                "failure_code": "articulation_continuous_path_collision",
                "stage": str(stage),
                "collision": collision,
            }
    except ArticulationExecutionError as exc:
        return {
            "status": "failed",
            "failure_code": str(exc),
            "stage": str(stage),
        }
    except Exception as exc:
        return {
            "status": "failed",
            "failure_code": "articulation_continuous_path_invalid",
            "stage": str(stage),
            "exception": f"{type(exc).__name__}:{exc}",
        }
    policy = articulation_motion_policy_binding()
    return {
        "status": "passed",
        "failure_code": None,
        "stage": str(stage),
        "motion_policy_version": policy["version"],
        "motion_policy_hash": policy["hash"],
        "position": position,
        "velocity": velocity,
        "duration_sec": float(duration),
        "planned_subpaths": len(plans),
        "geometric_waypoints": int(len(geometry)),
        "retimed_waypoints": int(len(position)),
        "collision": collision,
    }


def execute_continuous_articulation_path(
    plan: Mapping[str, Any],
    *,
    runtime: ContinuousArticulationReplayCallbacks,
    observation_count: int = CONTINUOUS_ARTICULATION_WAYPOINT_COUNT,
) -> dict[str, Any]:
    """Replay a compiled path once without replanning or state projection."""

    if str(plan.get("status")) != "passed":
        return dict(plan)
    positions = np.asarray(plan.get("position"), dtype=float)
    velocities = np.asarray(plan.get("velocity"), dtype=float)
    if (
        positions.ndim != 2
        or not len(positions)
        or velocities.shape != positions.shape
    ):
        return {
            "status": "failed",
            "failure_code": "articulation_continuous_path_invalid",
        }
    stride = max(1, len(positions) // max(1, int(observation_count)))
    maximum_joint_error = 0.0
    observation_rows: list[dict[str, Any]] = []
    for index, (position, velocity) in enumerate(zip(positions, velocities)):
        runtime.command_joint(position, velocity, index)
        measured = np.asarray(runtime.read_arm_qpos(), dtype=float)[: len(position)]
        if measured.shape != position.shape or not np.isfinite(measured).all():
            return {
                "status": "failed",
                "failure_code": "articulation_continuous_tracking_invalid",
                "executed_waypoints": index + 1,
            }
        maximum_joint_error = max(
            maximum_joint_error,
            float(np.max(np.abs(measured - position))),
        )
        if index % stride == 0 or index == len(positions) - 1:
            observed = runtime.observe_checkpoint(index, len(positions))
            if observed is not None:
                row = canonical_data(dict(observed))
                observation_rows.append(row)
                if row.get("passed") is False:
                    return {
                        "status": "failed",
                        "failure_code": row.get("failure_code")
                        or "articulation_contact_lost",
                        "executed_waypoints": index + 1,
                        "maximum_joint_error_rad": maximum_joint_error,
                        "observations": observation_rows,
                    }
    return {
        "status": "passed",
        "failure_code": None,
        "executed_waypoints": int(len(positions)),
        "maximum_joint_error_rad": maximum_joint_error,
        "observations": observation_rows,
    }


def _normalized_axis(value: np.ndarray, failure_code: str) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= 1e-9:
        raise ArticulationExecutionError(failure_code)
    return np.asarray(value, dtype=float) / norm


def _closest_point_on_triangle(
    point: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> np.ndarray:
    """Return the exact closest point on one triangle (Ericson regions)."""

    ab = b - a
    ac = c - a
    ap = point - a
    d1 = float(np.dot(ab, ap))
    d2 = float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return a
    bp = point - b
    d3 = float(np.dot(ab, bp))
    d4 = float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        return b
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        value = d1 / (d1 - d3)
        return a + value * ab
    cp = point - c
    d5 = float(np.dot(ab, cp))
    d6 = float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        return c
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        value = d2 / (d2 - d6)
        return a + value * ac
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        value = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + value * (c - b)
    denominator = 1.0 / (va + vb + vc)
    value = vb * denominator
    weight = vc * denominator
    return a + ab * value + ac * weight


def _local_surface_rows(
    contact_link: np.ndarray,
    shape_meshes_link: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for shape_index, raw_mesh in enumerate(shape_meshes_link):
        if not isinstance(raw_mesh, Mapping):
            raise ArticulationExecutionError(
                "articulation_collision_geometry_invalid"
            )
        vertices = np.asarray(raw_mesh.get("vertices"), dtype=float)
        triangles = np.asarray(raw_mesh.get("triangles"), dtype=int)
        if (
            vertices.ndim != 2
            or vertices.shape[1:] != (3,)
            or triangles.ndim != 2
            or triangles.shape[1:] != (3,)
            or not len(vertices)
            or not len(triangles)
            or not np.isfinite(vertices).all()
            or np.min(triangles) < 0
            or np.max(triangles) >= len(vertices)
        ):
            raise ArticulationExecutionError(
                "articulation_collision_geometry_invalid"
            )
        component_indices = _triangle_component_indices(vertices, triangles)
        for triangle_index, indices in enumerate(triangles):
            a, b, c = vertices[indices]
            normal = np.cross(b - a, c - a)
            norm = float(np.linalg.norm(normal))
            if not np.isfinite(norm) or norm <= 1e-12:
                continue
            normal /= norm
            nearest = _closest_point_on_triangle(contact_link, a, b, c)
            distance = float(np.linalg.norm(nearest - contact_link))
            rows.append(
                {
                    "shape_index": int(shape_index),
                    "component_index": int(component_indices[triangle_index]),
                    "triangle_index": int(triangle_index),
                    "nearest": nearest,
                    "normal": normal,
                    "distance": distance,
                }
            )
    rows.sort(
        key=lambda row: (
            row["distance"], row["shape_index"], row["triangle_index"]
        )
    )
    return rows


def _triangle_component_indices(
    vertices: np.ndarray,
    triangles: np.ndarray,
) -> list[int]:
    """Return deterministic welded-vertex connected components per triangle."""

    parent = list(range(len(triangles)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root == second_root:
            return
        if first_root < second_root:
            parent[second_root] = first_root
        else:
            parent[first_root] = second_root

    vertex_owner: dict[tuple[float, float, float], int] = {}
    welded = np.round(np.asarray(vertices, dtype=float), decimals=9)
    for triangle_index, indices in enumerate(triangles):
        for vertex_index in indices:
            key = tuple(float(value) for value in welded[int(vertex_index)])
            previous = vertex_owner.setdefault(key, triangle_index)
            union(triangle_index, previous)

    root_to_component: dict[int, int] = {}
    result: list[int] = []
    for triangle_index in range(len(triangles)):
        root = find(triangle_index)
        if root not in root_to_component:
            root_to_component[root] = len(root_to_component)
        result.append(root_to_component[root])
    return result


def derive_articulation_bilateral_grasp_frame(
    contact_matrix_world: Sequence[Sequence[float]],
    link_matrix_world: Sequence[Sequence[float]],
    shape_meshes_link: Sequence[Mapping[str, Any]],
    *,
    minimum_closing_width_m: float,
    maximum_closing_width_m: float,
    maximum_contact_shape_distance_m: float,
    local_surface_patch_radius_m: float,
    minimum_opposed_normal_cosine: float,
    orientation_variant: str = "authored_closing_sign",
) -> dict[str, Any]:
    """Construct an articulated grasp frame from local opposed surfaces.

    The authored Contact first binds to its nearest welded-vertex collision
    component.  Only triangles in that component and close to the Contact
    participate.  A candidate must expose two locally aligned, approximately
    opposed surfaces separated by a gripper-feasible thickness.  Geometry
    determines the closing axis; the authored Contact keeps the approach
    direction and resolves its sign.  Ordinary rigid-object grasp planning
    never calls this function.
    """

    contact = np.asarray(contact_matrix_world, dtype=float)
    link = np.asarray(link_matrix_world, dtype=float)
    if (
        contact.shape != (4, 4)
        or link.shape != (4, 4)
        or not np.isfinite(contact).all()
        or not np.isfinite(link).all()
    ):
        raise ArticulationExecutionError("articulation_contact_frame_invalid")
    try:
        link_inverse = np.linalg.inv(link)
    except np.linalg.LinAlgError as exc:
        raise ArticulationExecutionError(
            "articulation_moving_link_frame_invalid"
        ) from exc
    contact_link = (link_inverse @ contact)[:3, 3]
    base = contact @ CONTACT_TO_GRASP_FRAME
    authored_approach_link = link[:3, :3].T @ base[:3, 0]
    authored_closing_link = link[:3, :3].T @ base[:3, 1]

    surfaces = _local_surface_rows(contact_link, shape_meshes_link)
    if not surfaces:
        raise ArticulationExecutionError(
            "articulation_collision_geometry_missing"
        )
    maximum_distance = float(maximum_contact_shape_distance_m)
    if not np.isfinite(maximum_distance) or maximum_distance <= 0.0:
        raise ArticulationExecutionError("articulation_profile_invalid")
    nearest_distance = float(surfaces[0]["distance"])
    if not np.isfinite(nearest_distance) or nearest_distance > maximum_distance:
        raise ArticulationExecutionError(
            "articulation_contact_not_on_moving_link_geometry"
        )
    minimum_width = float(minimum_closing_width_m)
    maximum_width = float(maximum_closing_width_m)
    patch_radius = float(local_surface_patch_radius_m)
    normal_cosine = float(minimum_opposed_normal_cosine)
    if (
        not np.isfinite(minimum_width)
        or not np.isfinite(maximum_width)
        or not np.isfinite(patch_radius)
        or not np.isfinite(normal_cosine)
        or minimum_width <= 0.0
        or maximum_width <= minimum_width
        or patch_radius <= 0.0
        or not 0.0 < normal_cosine <= 1.0
    ):
        raise ArticulationExecutionError("articulation_profile_invalid")
    if orientation_variant not in {
        "authored_closing_sign",
        "opposite_closing_sign",
    }:
        raise ArticulationExecutionError(
            "articulation_bilateral_orientation_variant_invalid"
        )
    contact_component = (
        int(surfaces[0]["shape_index"]),
        int(surfaces[0]["component_index"]),
    )
    local_surfaces = [
        row
        for row in surfaces
        if (
            int(row["shape_index"]),
            int(row["component_index"]),
        )
        == contact_component
        and float(row["distance"]) <= maximum_distance + maximum_width
    ][:384]
    pairs: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    for first_index, first in enumerate(local_surfaces):
        for second in local_surfaces[first_index + 1 :]:
            first_normal = np.asarray(first["normal"], dtype=float)
            second_normal = np.asarray(second["normal"], dtype=float)
            opposed_cosine = abs(float(np.dot(first_normal, second_normal)))
            if opposed_cosine < normal_cosine:
                continue
            if float(np.dot(first_normal, second_normal)) > 0.0:
                second_normal = -second_normal
            closing = _normalized_axis(
                first_normal - second_normal,
                "articulation_no_local_bilateral_surface_pair",
            )
            first_point = np.asarray(first["nearest"], dtype=float)
            second_point = np.asarray(second["nearest"], dtype=float)
            separation = second_point - first_point
            separation_norm = float(np.linalg.norm(separation))
            if separation_norm <= 1e-9:
                continue
            thickness = abs(float(np.dot(separation, closing)))
            axis_cosine = thickness / separation_norm
            lateral = float(
                np.linalg.norm(
                    separation - closing * float(np.dot(separation, closing))
                )
            )
            if (
                thickness < minimum_width
                or thickness > maximum_width
                or axis_cosine < normal_cosine
                or lateral > patch_radius
            ):
                continue
            tangential_distances = []
            for surface_point in (first_point, second_point):
                delta = surface_point - contact_link
                tangent = delta - closing * float(np.dot(delta, closing))
                tangential_distances.append(float(np.linalg.norm(tangent)))
            maximum_tangential_distance = max(tangential_distances)
            if maximum_tangential_distance > patch_radius:
                continue
            if float(np.dot(closing, authored_closing_link)) < 0.0:
                closing = -closing
            score = (
                min(float(first["distance"]), float(second["distance"])),
                maximum_tangential_distance,
                lateral,
                1.0 - opposed_cosine,
                1.0 - axis_cosine,
                -abs(float(np.dot(closing, authored_closing_link))),
                thickness,
                int(first["shape_index"]),
                int(first["triangle_index"]),
                int(second["shape_index"]),
                int(second["triangle_index"]),
            )
            pairs.append(
                (
                    score,
                    {
                        "first": first,
                        "second": second,
                        "closing": closing,
                        "thickness": thickness,
                        "opposed_cosine": opposed_cosine,
                        "axis_cosine": axis_cosine,
                        "lateral": lateral,
                        "maximum_tangential_distance": (
                            maximum_tangential_distance
                        ),
                    },
                )
            )
    if not pairs:
        raise ArticulationExecutionError(
            "articulation_no_local_bilateral_surface_pair"
        )
    _score, selected = min(pairs, key=lambda row: row[0])
    closing_link = np.asarray(selected["closing"], dtype=float)
    if orientation_variant == "opposite_closing_sign":
        closing_link = -closing_link

    inward = authored_approach_link - closing_link * float(
        np.dot(authored_approach_link, closing_link)
    )
    orientation_source = (
        "authored_approach_projected_to_contact_component_surface_pair"
    )
    approach_link = _normalized_axis(
        inward, "articulation_bilateral_approach_axis_invalid"
    )
    orthogonal_link = _normalized_axis(
        np.cross(approach_link, closing_link),
        "articulation_bilateral_frame_degenerate",
    )
    closing_link = _normalized_axis(
        np.cross(orthogonal_link, approach_link),
        "articulation_bilateral_frame_degenerate",
    )
    expected_sign = 1.0 if orientation_variant == "authored_closing_sign" else -1.0
    if expected_sign * float(np.dot(closing_link, authored_closing_link)) < 0.0:
        closing_link = -closing_link
        orthogonal_link = -orthogonal_link

    rotation_link = np.column_stack(
        [approach_link, closing_link, orthogonal_link]
    )
    rotation_world = link[:3, :3] @ rotation_link
    return canonical_data(
        {
            "version": ARTICULATION_BILATERAL_FRAME_VERSION,
            "orientation_source": orientation_source,
            "orientation_variant": orientation_variant,
            "contact_component": {
                "shape_index": contact_component[0],
                "component_index": contact_component[1],
            },
            "shape_index": int(selected["first"]["shape_index"]),
            "surface_pair": [
                {
                    "shape_index": int(selected["first"]["shape_index"]),
                    "component_index": int(
                        selected["first"]["component_index"]
                    ),
                    "triangle_index": int(
                        selected["first"]["triangle_index"]
                    ),
                },
                {
                    "shape_index": int(selected["second"]["shape_index"]),
                    "component_index": int(
                        selected["second"]["component_index"]
                    ),
                    "triangle_index": int(
                        selected["second"]["triangle_index"]
                    ),
                },
            ],
            "contact_shape_distance_m": nearest_distance,
            "closing_extent_m": float(selected["thickness"]),
            "surface_opposition_cosine": float(selected["opposed_cosine"]),
            "surface_pair_axis_cosine": float(selected["axis_cosine"]),
            "surface_pair_lateral_offset_m": float(selected["lateral"]),
            "surface_pair_contact_tangent_distance_m": float(
                selected["maximum_tangential_distance"]
            ),
            "local_surface_patch_radius_m": patch_radius,
            "approach_axis_link": approach_link.tolist(),
            "closing_axis_link": closing_link.tolist(),
            "rotation_world": rotation_world.tolist(),
            "authored_closing_alignment": abs(
                float(np.dot(closing_link, authored_closing_link))
            ),
        }
    )


def articulation_grasp_matrix(
    contact_matrix: Sequence[Sequence[float]],
    *,
    pre_grasp_m: float,
    roll_magnitude_rad: float = 0.0,
    towards_world: Sequence[float] = (0.0, -1.0, 0.0),
    grasp_rotation_world: Sequence[Sequence[float]] | None = None,
) -> np.ndarray:
    """Lower a reviewed Contact and resolved orientation to an EE pose."""

    contact = np.asarray(contact_matrix, dtype=float)
    towards = np.asarray(towards_world, dtype=float)
    if contact.shape != (4, 4) or towards.shape != (3,):
        raise ArticulationExecutionError("articulation_contact_frame_invalid")
    base = contact @ CONTACT_TO_GRASP_FRAME
    distance = 0.12 + float(pre_grasp_m)

    if grasp_rotation_world is not None:
        rotation = np.asarray(grasp_rotation_world, dtype=float)
        if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
            raise ArticulationExecutionError("articulation_grasp_frame_invalid")
        result = np.eye(4, dtype=float)
        result[:3, :3] = rotation
        result[:3, 3] = base[:3, 3] + rotation @ np.asarray(
            [-distance, 0.0, 0.0], dtype=float
        )
        return result

    def resolved(angle: float) -> tuple[np.ndarray, np.ndarray]:
        roll = _axis_angle_matrix(np.asarray([0.0, 1.0, 0.0]), angle)
        rotation = base[:3, :3] @ roll
        offset = rotation @ np.asarray([-distance, 0.0, 0.0], dtype=float)
        return rotation, offset

    magnitude = abs(float(roll_magnitude_rad))
    rotation, offset = resolved(magnitude)
    if float(np.dot(offset, towards)) < 0.0:
        rotation, offset = resolved(-magnitude)
    result = np.eye(4, dtype=float)
    result[:3, :3] = rotation
    result[:3, 3] = base[:3, 3] + offset
    return result


def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / float(np.linalg.norm(axis))
    x, y, z = axis
    skew = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return (
        np.eye(3, dtype=float)
        + np.sin(angle) * skew
        + (1.0 - np.cos(angle)) * (skew @ skew)
    )


def classify_articulation_engagement(
    *,
    mode: str,
    subject: str,
    arm: str,
    expected_link: str,
    gripper: Mapping[str, Any],
    contacts: Sequence[Mapping[str, Any]],
    continuous_steps: int,
    required_steps: int,
    translation_drift_m: float = 0.0,
    rotation_drift_deg: float = 0.0,
    translation_limit_m: float = 0.003,
    rotation_limit_deg: float = 3.0,
    allowed_contact_regions: Sequence[str] = ("inner_pad", "fingertip"),
) -> dict[str, Any]:
    relevant: list[Mapping[str, Any]] = []
    subject_contacts: list[Mapping[str, Any]] = []
    for row in contacts:
        endpoints = {str(row.get("entity_a")), str(row.get("entity_b"))}
        if endpoints != {str(subject), f"gripper:{arm}"}:
            continue
        subject_contacts.append(row)
        subject_link = (
            row.get("link_a")
            if str(row.get("entity_a")) == str(subject)
            else row.get("link_b")
        )
        if str(subject_link or "") == str(expected_link):
            relevant.append(row)
    regions = {str(row.get("region") or "") for row in relevant}
    observed_subject_links = sorted(
        {
            str(
                row.get("link_a")
                if str(row.get("entity_a")) == str(subject)
                else row.get("link_b")
            )
            for row in subject_contacts
        }
    )
    fingers = {
        str(row.get("finger") or "")
        for row in relevant
        if row.get("region") == "inner_pad"
    }
    if mode == "grasp_follow":
        checks = {
            "actual_gripper_engaged": bool(gripper.get("engaged")),
            "bilateral_inner_contact": len(fingers) >= 2,
            "continuous_contact": int(continuous_steps) >= int(required_steps),
            "translation_consistent": float(translation_drift_m) <= float(translation_limit_m),
            "rotation_consistent": float(rotation_drift_deg) <= float(rotation_limit_deg),
        }
    else:
        allowed = {str(value) for value in allowed_contact_regions}
        checks = {
            "reviewed_pad_contact": bool(regions & allowed),
            "continuous_contact": int(continuous_steps) >= int(required_steps),
        }
    failed = [name for name, passed in checks.items() if not passed]
    failure_code = None
    if failed:
        if "actual_gripper_engaged" in failed:
            failure_code = "articulation_gripper_disengaged"
        elif "translation_consistent" in failed:
            failure_code = "articulation_translation_slip"
        elif "rotation_consistent" in failed:
            failure_code = "articulation_rotation_slip"
        elif relevant:
            failure_code = "articulation_closure_lost" if mode == "grasp_follow" else "articulation_contact_lost"
        elif subject_contacts:
            failure_code = "articulation_contact_wrong_link"
        else:
            failure_code = "articulation_contact_lost"
    return canonical_data(
        {
            "passed": not failed,
            "failure_code": failure_code,
            "failed_checks": failed,
            "checks": checks,
            "metrics": {
                "expected_link": expected_link,
                "contact_count": len(relevant),
                "subject_gripper_contact_count": len(subject_contacts),
                "observed_subject_links": observed_subject_links,
                "regions": sorted(regions),
                "fingers": sorted(fingers),
                "continuous_steps": int(continuous_steps),
                "translation_drift_m": float(translation_drift_m),
                "rotation_drift_deg": float(rotation_drift_deg),
                "actual_width_m": gripper.get("actual_width_m"),
            },
        }
    )


@dataclass(frozen=True)
class ArticulationRuntimeCallbacks:
    read_qpos: Callable[[], float]
    acquire: Callable[[Mapping[str, Any]], Mapping[str, Any]]
    command_step: Callable[[Mapping[str, Any], float], bool] | None
    observe: Callable[[Mapping[str, Any]], Mapping[str, Any]]
    release_retreat: Callable[[Mapping[str, Any]], None]
    settle_steps: Callable[[int], None]
    command_segment: Callable[
        [Mapping[str, Any], float], Mapping[str, Any]
    ] | None = None


def run_articulation_closed_loop(
    *,
    state: str,
    initial_qpos: float,
    lower: float,
    upper: float,
    state_profile: Mapping[str, Any],
    execution_profile: Mapping[str, Any],
    contract: SuccessContract,
    runtime: ArticulationRuntimeCallbacks,
) -> dict[str, Any]:
    """Run one bounded fail-fast articulation attempt through runtime callbacks."""

    goal = resolve_articulation_goal(
        state=state,
        initial_qpos=initial_qpos,
        lower=lower,
        upper=upper,
        state_profile=state_profile,
        contract=contract,
    )
    if goal["status"] != "resolved":
        return {"success": False, "failure_code": goal["failure_code"], "goal": goal, "trace": []}
    try:
        segments = segment_targets(execution_profile, goal)
    except ArticulationExecutionError as exc:
        return {"success": False, "failure_code": str(exc), "goal": goal, "trace": []}
    direction = int(goal["direction_sign"])
    span = float(goal["span"])
    step = span * float(execution_profile.get("step_ratio", 0.025))
    progress_epsilon = span * float(execution_profile.get("minimum_step_progress_ratio", 0.001))
    negative_tolerance = span * float(execution_profile.get("negative_progress_tolerance_ratio", 0.001))
    maximum_steps = int(execution_profile.get("maximum_steps", 48))
    maximum_consecutive_stalls = int(
        execution_profile.get("maximum_consecutive_stalled_steps", 1)
    )
    maximum_consecutive_negative = int(
        execution_profile.get("maximum_consecutive_negative_steps", 1)
    )
    target_tolerance = span * float(
        execution_profile.get("target_reached_tolerance_ratio", 0.0)
    )
    trace: list[dict[str, Any]] = []
    for segment in segments:
        before_acquisition = float(runtime.read_qpos())
        runtime_segment = {
            **segment,
            "direction_sign": direction,
            "negative_progress_tolerance_qpos": negative_tolerance,
            "goal": goal,
        }
        acquisition = canonical_data(dict(runtime.acquire(runtime_segment)))
        after_acquisition = float(runtime.read_qpos())
        acquisition_progress = direction * (
            after_acquisition - before_acquisition
        )
        trace.append(
            {
                "phase": "acquire",
                "segment_index": segment["segment_index"],
                "before_qpos": before_acquisition,
                "after_qpos": after_acquisition,
                "signed_progress": acquisition_progress,
                **acquisition,
            }
        )
        fixed_attachment_acquired = bool(
            execution_profile.get("post_acquisition_attachment")
            == "fixed_constraint"
            and acquisition.get("passed")
            and acquisition.get("attachment_active")
        )
        if (
            acquisition_progress < -negative_tolerance
            and not fixed_attachment_acquired
        ):
            return {
                "success": False,
                "failure_code": "articulation_negative_progress",
                "goal": goal,
                "trace": trace,
            }
        if not acquisition.get("passed"):
            return {
                "success": False,
                "failure_code": acquisition.get("failure_code") or "articulation_contact_not_acquired",
                "goal": goal,
                "trace": trace,
            }
        segment_steps = 0
        consecutive_stalls = 0
        consecutive_negative = 0
        target_qpos = float(segment["resolved_target_qpos"])
        if runtime.command_segment is not None:
            before = float(runtime.read_qpos())
            command = canonical_data(
                dict(runtime.command_segment(runtime_segment, target_qpos))
            )
            after = float(runtime.read_qpos())
            signed_progress = direction * (after - before)
            observation = canonical_data(dict(runtime.observe(runtime_segment)))
            row = {
                "phase": "continuous_servo",
                "segment_index": segment["segment_index"],
                "before_qpos": before,
                "requested_qpos": target_qpos,
                "accepted_target_qpos": command.get(
                    "accepted_target_qpos", target_qpos
                ),
                "after_qpos": after,
                "signed_progress": signed_progress,
                "planner_succeeded": command.get("status") == "passed",
                "command": command,
                "observation": observation,
            }
            trace.append(row)
            if command.get("status") != "passed":
                return {
                    "success": False,
                    "failure_code": command.get("failure_code")
                    or "articulation_step_unreachable",
                    "goal": goal,
                    "trace": trace,
                }
            if not observation.get("passed"):
                return {
                    "success": False,
                    "failure_code": observation.get("failure_code")
                    or "articulation_contact_lost",
                    "goal": goal,
                    "trace": trace,
                }
            if signed_progress < -negative_tolerance:
                return {
                    "success": False,
                    "failure_code": "articulation_negative_progress",
                    "goal": goal,
                    "trace": trace,
                }
            accepted_target = float(
                command.get("accepted_target_qpos", target_qpos)
            )
            if direction * (accepted_target - after) > target_tolerance:
                return {
                    "success": False,
                    "failure_code": "articulation_target_interval_not_reached",
                    "goal": goal,
                    "trace": trace,
                }
            runtime.release_retreat(runtime_segment)
            continue
        if runtime.command_step is None:
            return {
                "success": False,
                "failure_code": "articulation_motion_interface_missing",
                "goal": goal,
                "trace": trace,
            }
        while direction * (target_qpos - runtime.read_qpos()) > target_tolerance:
            if segment_steps >= maximum_steps:
                return {"success": False, "failure_code": "articulation_target_interval_not_reached", "goal": goal, "trace": trace}
            before = float(runtime.read_qpos())
            requested = before + direction * min(step, abs(target_qpos - before))
            planned = bool(runtime.command_step(segment, requested))
            after = float(runtime.read_qpos())
            signed_progress = direction * (after - before)
            observation = canonical_data(dict(runtime.observe(segment)))
            row = {
                "phase": "servo",
                "segment_index": segment["segment_index"],
                "step_index": segment_steps,
                "before_qpos": before,
                "requested_qpos": requested,
                "after_qpos": after,
                "signed_progress": signed_progress,
                "planner_succeeded": planned,
                "observation": observation,
            }
            if execution_profile.get("post_acquisition_attachment") == "fixed_constraint":
                row["attachment_projection"] = execution_profile.get(
                    "attachment_execution_projection"
                )
            trace.append(row)
            reached = direction * (target_qpos - after) <= target_tolerance
            if not observation.get("passed"):
                return {"success": False, "failure_code": observation.get("failure_code") or "articulation_contact_lost", "goal": goal, "trace": trace}
            if signed_progress < -negative_tolerance:
                consecutive_negative += 1
                row["consecutive_negative_steps"] = consecutive_negative
                if consecutive_negative >= maximum_consecutive_negative:
                    return {"success": False, "failure_code": "articulation_negative_progress", "goal": goal, "trace": trace}
            else:
                consecutive_negative = 0
            if not reached and signed_progress < progress_epsilon:
                consecutive_stalls += 1
                row["consecutive_stalled_steps"] = consecutive_stalls
                if consecutive_stalls >= maximum_consecutive_stalls:
                    return {"success": False, "failure_code": "articulation_progress_stalled" if planned else "articulation_step_unreachable", "goal": goal, "trace": trace}
            else:
                consecutive_stalls = 0
            segment_steps += 1
        runtime.release_retreat(segment)
    runtime.settle_steps(int(execution_profile.get("post_release_observation_steps", 20)))
    final_qpos = float(runtime.read_qpos())
    final_ratio = (final_qpos - float(goal["lower"])) / span
    interval = goal["target_interval"]
    progress = direction * (final_qpos - float(initial_qpos)) / span
    if not (float(interval[0]) <= final_ratio <= float(interval[1])):
        failure = "articulation_springback_after_release"
        success = False
    elif progress < float(goal["minimum_signed_normalized_progress"]):
        failure = "articulation_target_interval_not_reached"
        success = False
    else:
        failure = None
        success = True
    return canonical_data(
        {
            "success": success,
            "failure_code": failure,
            "goal": goal,
            "final_qpos": final_qpos,
            "final_ratio": final_ratio,
            "signed_normalized_progress": progress,
            "trace": trace,
        }
    )


__all__ = [
    "ARTICULATION_BILATERAL_FRAME_VERSION",
    "ARTICULATION_EXECUTION_PROFILE_VERSION",
    "ARTICULATION_MOTION_POLICY_VERSION",
    "CONTINUOUS_ARTICULATION_MAXIMUM_BOUNDARY_JUMP_RAD",
    "CONTINUOUS_ARTICULATION_PATH_SAMPLES_PER_SUBPATH",
    "CONTINUOUS_ARTICULATION_WAYPOINT_COUNT",
    "DEFAULT_ARTICULATION_EXECUTION_PROFILE",
    "ArticulationExecutionError",
    "ArticulationRuntimeCallbacks",
    "ContinuousArticulationPlanCallbacks",
    "ContinuousArticulationReplayCallbacks",
    "articulation_acquisition_candidates",
    "articulation_arm_joint_limits",
    "articulation_motion_policy_binding",
    "articulation_grasp_matrix",
    "derive_articulation_bilateral_grasp_frame",
    "classify_articulation_engagement",
    "fixed_attachment_frames",
    "fixed_attachment_residual",
    "future_contact_matrix",
    "guarded_approach_matrices",
    "continuous_articulation_qpos_targets",
    "continuous_articulation_target_candidates",
    "compile_continuous_articulation_path",
    "execute_continuous_articulation_path",
    "stitch_articulation_joint_paths",
    "load_articulation_execution_catalog",
    "preengagement_arm_fallback_allowed",
    "resolve_articulation_execution_profile",
    "run_articulation_closed_loop",
    "select_articulation_acquisition_plan",
    "segment_targets",
]
