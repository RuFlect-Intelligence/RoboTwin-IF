"""Simulator-facing planner adapter for the R02 exact-Contact contract."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Mapping, Sequence

import numpy as np
import transforms3d as t3d

from if_benchmark.runtime_errors import PhysicalCollisionValidationError
from if_benchmark.pair_terminal_settle_worker import actor_aabb
from if_benchmark.planner_batch import (
    CUROBO_FIXED_BATCH_SIZE,
    normalize_candidate_search_budget,
)
from if_benchmark.success_contract import continuous_acquisition_lift_delta

from .core import (
    EXACT_GRASP_POLICY,
    GraspContractError,
    allowed_contact_ids,
    approach_direction,
    build_contact_contract,
    candidate_id,
    classify_approach,
    clean_matrix,
    clean_vector,
    contact_groups_for_ids,
    exact_grasp_poses,
    joint_path_metrics,
    scaled_contact_world_matrix,
    select_best_retreat_per_contact,
    support_safe_grasp_poses,
    trajectory_fingerprint,
    trajectory_summary,
)
from .held_collision import temporary_held_collision, plan_held_path
from .planning_contract import PICK_PLANNING_ADAPTER_VERSION
from .collision_revalidation import revalidate_live_approach


# Compatibility export for existing diagnostics/tests.  The pure contract is
# the sole owner so M12 can validate this identity without simulator imports.
PLANNING_ADAPTER_VERSION = PICK_PLANNING_ADAPTER_VERSION

# CuRobo is compiled and warmed with this fixed pose width. Keep the value
# beside the padding/chunking adapter so higher-level code can fill a batch
# without importing the optional GPU planner module.
PLANNER_BATCH_POSE_CAPACITY = CUROBO_FIXED_BATCH_SIZE

# Execution dynamics are deliberately outside ``EXACT_GRASP_POLICY``.  That
# policy is part of the R02 Contact identity, while these values govern how a
# previously compiled joint path is replayed by the current embodiment.
COMPILED_TRAJECTORY_EXECUTION_POLICY = {
    "version": "continuous_compiled_replay_v2",
    # In-flight error is only a catastrophic runaway guard.  Pausing at an
    # intermediate waypoint changes the time parameterization and can prevent
    # an overshooting articulation from rejoining the compiled trajectory.
    # The collision certificate applies to the compiled joint path, not to an
    # articulation that trails it by a large angle.  Keep this as a bounded
    # tracking corridor so a lagging arm is stopped before it can continue far
    # away from the exact-mesh-validated path.
    "maximum_inflight_joint_error_rad": 0.02,
    "terminal_joint_error_rad": 0.05,
    "compiled_start_tolerance_rad": 0.02,
    "chained_start_tolerance_rad": 0.05,
    "maximum_terminal_hold_steps": 125,
}


class CollisionWorldUpdateError(RuntimeError):
    """The live collision scene could not be installed in both planners."""


def _contact_matrices(actor: Any) -> list[list[list[float]]]:
    config = getattr(actor, "config", None) or {}
    key = getattr(actor, "POINTS", {}).get("contact", "contact_points_pose")
    raw = config.get(key)
    if not isinstance(raw, list) or not raw:
        raise GraspContractError("actor has no authored Contact Points")
    matrices = []
    for index, item in enumerate(raw):
        matrix = item.get("matrix") if isinstance(item, Mapping) else item
        value = np.asarray(matrix, dtype=float)
        if value.shape != (4, 4):
            raise GraspContractError(
                f"Contact {index} must contain one 4x4 matrix"
            )
        matrices.append(value.tolist())
    return matrices


def _actor_scale(actor: Any) -> list[float]:
    raw = (getattr(actor, "config", None) or {}).get("scale", [1.0, 1.0, 1.0])
    if isinstance(raw, (int, float)):
        return [float(raw)] * 3
    values = [float(value) for value in raw]
    if len(values) == 1:
        values *= 3
    if len(values) != 3:
        raise GraspContractError("actor Contact scale must contain three values")
    return values


def _joint_limits(task: Any, arm: str) -> list[list[float]]:
    joints = (
        task.robot.left_arm_joints
        if str(arm) == "left"
        else task.robot.right_arm_joints
    )
    result = []
    for joint in joints:
        bounds = np.asarray(joint.get_limits(), dtype=float).reshape(-1, 2)
        if len(bounds) == 0:
            raise GraspContractError("arm joint has no limits")
        result.append([float(bounds[0, 0]), float(bounds[0, 1])])
    return result


def _planner(task: Any, arm: str):
    return (
        task.robot.left_plan_path
        if str(arm) == "left"
        else task.robot.right_plan_path
    )


def _batch_plan_paths(
    task: Any,
    arm: str,
    target_poses: Sequence[Sequence[float]],
    *,
    start_qposes: Sequence[Sequence[float]] | None = None,
    expected_world_hash: str | None = None,
    expected_attachment_hash: str | None = None,
    batch_size: int = PLANNER_BATCH_POSE_CAPACITY,
    candidate_search_budget: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Plan exact poses in fixed-size CuRobo batches.

    CuRobo is warmed with the repository's shared fixed-width batch shape. Padding is
    deliberately confined to this adapter and never becomes an R02 candidate.
    For the second segment, every row receives the terminal qpos of its own
    Pre-Grasp trajectory.
    """

    budget = normalize_candidate_search_budget(candidate_search_budget)
    if budget is not None and expected_attachment_hash is not None:
        raise ValueError("candidate search budget cannot limit attached mandatory paths")
    poses = [list(pose) for pose in target_poses]
    if not poses:
        return []
    starts = None
    if start_qposes is not None:
        starts = [np.asarray(row, dtype=float).reshape(-1) for row in start_qposes]
        if len(starts) != len(poses):
            raise GraspContractError(
                "batched start qpos must match the number of target poses"
            )
    method = getattr(
        task.robot,
        "left_plan_multi_path" if str(arm) == "left" else "right_plan_multi_path",
        None,
    )

    def single_exception_fallback(
        pose: Sequence[float],
        *,
        start_qpos: np.ndarray | None,
        batch_exception: str,
    ) -> dict[str, Any]:
        """Retry one exact query after a batch infrastructure exception.

        This is deliberately not an IK or pose fallback: the arm, target 7D
        pose, full-orientation constraint, and (for Grasp) chained start qpos
        are identical to the failed batch row. Padding rows never call this
        helper.
        """

        planner = _planner(task, arm)
        kwargs: dict[str, Any] = {"constraint_pose": None}
        if start_qpos is not None:
            kwargs["last_qpos"] = start_qpos
        if expected_world_hash is not None:
            kwargs["expected_world_hash"] = str(expected_world_hash)
        if expected_attachment_hash is not None:
            kwargs["expected_attachment_hash"] = str(
                expected_attachment_hash
            )
        provenance = {
            "planner_mode": "single_pose_exception_fallback",
            "planning_adapter_version": PLANNING_ADAPTER_VERSION,
            "batch_exception": str(batch_exception),
        }
        try:
            result = planner(list(pose), **kwargs)
        except TimeoutError:
            raise
        except Exception as exc:
            return {
                "status": "Exception",
                "exception": f"{type(exc).__name__}:{exc}",
                **provenance,
            }
        if not isinstance(result, Mapping):
            return {
                "status": "Exception",
                "exception": "single-pose fallback returned a non-mapping result",
                **provenance,
            }
        return {**dict(result), **provenance}

    def fallback_chunk(
        chunk_poses: Sequence[Sequence[float]],
        *,
        chunk_starts: Sequence[np.ndarray] | None,
        batch_exception: str,
    ) -> list[dict[str, Any]]:
        return [
            single_exception_fallback(
                pose,
                start_qpos=(
                    None if chunk_starts is None else chunk_starts[index]
                ),
                batch_exception=batch_exception,
            )
            for index, pose in enumerate(chunk_poses)
        ]

    if not callable(method):
        # Small fixtures and older non-Benchmark robots may only expose the
        # single-pose API.  This is an execution fallback, not a pose fallback.
        planner = _planner(task, arm)
        rows = []
        for index, pose in enumerate(poses):
            kwargs = {"constraint_pose": None}
            if starts is not None:
                kwargs["last_qpos"] = starts[index]
            if expected_world_hash is not None:
                kwargs["expected_world_hash"] = str(expected_world_hash)
            if expected_attachment_hash is not None:
                kwargs["expected_attachment_hash"] = str(
                    expected_attachment_hash
                )
            try:
                rows.append(dict(planner(pose, **kwargs)))
            except TimeoutError:
                raise
            except Exception as exc:
                rows.append(
                    {
                        "status": "Exception",
                        "exception": f"{type(exc).__name__}:{exc}",
                    }
                )
        return rows

    if batch_size <= 0:
        raise GraspContractError("planner batch size must be positive")
    rows: list[dict[str, Any]] = []
    for begin in range(0, len(poses), batch_size):
        chunk_poses = poses[begin : begin + batch_size]
        actual_count = len(chunk_poses)
        padded_poses = chunk_poses + [chunk_poses[-1]] * (batch_size - actual_count)
        kwargs: dict[str, Any] = {"constraint_pose": None}
        chunk_starts = None
        if starts is not None:
            chunk_starts = starts[begin : begin + batch_size]
            padded_starts = chunk_starts + [chunk_starts[-1]] * (
                batch_size - actual_count
            )
            kwargs["last_qpos"] = np.stack(padded_starts)
        if expected_world_hash is not None:
            kwargs["expected_world_hash"] = str(expected_world_hash)
        if expected_attachment_hash is not None:
            kwargs["expected_attachment_hash"] = str(
                expected_attachment_hash
            )
        if budget is not None:
            kwargs["candidate_search_budget"] = budget
        try:
            result = method(padded_poses, **kwargs)
        except TimeoutError:
            raise
        except Exception as exc:
            rows.extend(
                fallback_chunk(
                    chunk_poses,
                    chunk_starts=chunk_starts,
                    batch_exception=f"{type(exc).__name__}:{exc}",
                )
            )
            continue
        if not isinstance(result, Mapping):
            rows.extend(
                fallback_chunk(
                    chunk_poses,
                    chunk_starts=chunk_starts,
                    batch_exception="planner batch returned a non-mapping result",
                )
            )
            continue
        statuses = np.asarray(result.get("status", []), dtype=object).reshape(-1)
        batch_exception = result.get("exception")
        positions = result.get("position")
        position_array = None if positions is None else np.asarray(positions, dtype=float)
        velocities = result.get("velocity")
        velocity_array = (
            None if velocities is None else np.asarray(velocities, dtype=float)
        )
        last_steps = result.get("path_buffer_last_tstep")
        last_step_array = (
            None
            if last_steps is None
            else np.asarray(last_steps, dtype=int).reshape(-1)
        )
        for local_index in range(actual_count):
            if local_index >= len(statuses):
                rows.append(
                    single_exception_fallback(
                        chunk_poses[local_index],
                        start_qpos=(
                            None
                            if chunk_starts is None
                            else chunk_starts[local_index]
                        ),
                        batch_exception="planner batch result is missing row status",
                    )
                )
                continue
            status = str(statuses[local_index])
            if status == "Exception":
                rows.append(
                    single_exception_fallback(
                        chunk_poses[local_index],
                        start_qpos=(
                            None
                            if chunk_starts is None
                            else chunk_starts[local_index]
                        ),
                        batch_exception=str(
                            batch_exception or "planner batch row exception"
                        ),
                    )
                )
                continue
            if status != "Success":
                rows.append(
                    {
                        "status": status,
                        "planner_mode": "fixed_batch",
                        "planning_adapter_version": PLANNING_ADAPTER_VERSION,
                        **({
                            "candidate_search_budget": budget,
                            "candidate_search_budget_applied": True,
                        } if budget is not None and result.get(
                            "candidate_search_budget_applied"
                        ) is True else {}),
                    }
                )
                continue
            if (
                position_array is None
                or position_array.ndim != 3
                or local_index >= len(position_array)
            ):
                rows.append(
                    single_exception_fallback(
                        chunk_poses[local_index],
                        start_qpos=(
                            None
                            if chunk_starts is None
                            else chunk_starts[local_index]
                        ),
                        batch_exception="planner batch success row has no trajectory",
                    )
                )
                continue
            path = position_array[local_index]
            velocity_path = None
            if (
                velocity_array is not None
                and velocity_array.ndim == 3
                and local_index < len(velocity_array)
                and velocity_array[local_index].shape == path.shape
            ):
                velocity_path = velocity_array[local_index]
            if last_step_array is not None and local_index < len(last_step_array):
                stop = int(last_step_array[local_index])
                if stop > 0:
                    path = path[:stop]
                    if velocity_path is not None:
                        velocity_path = velocity_path[:stop]
            if path.ndim != 2 or len(path) == 0:
                rows.append(
                    single_exception_fallback(
                        chunk_poses[local_index],
                        start_qpos=(
                            None
                            if chunk_starts is None
                            else chunk_starts[local_index]
                        ),
                        batch_exception="planner batch success row has an empty trajectory",
                    )
                )
                continue
            row = {
                "status": "Success",
                "position": path,
                "planner_mode": "fixed_batch",
                "planning_adapter_version": PLANNING_ADAPTER_VERSION,
                **_planner_result_provenance(result),
            }
            if velocity_path is not None:
                row["velocity"] = velocity_path
            rows.append(row)
    if len(rows) != len(poses):
        raise GraspContractError(
            f"planner batch cardinality mismatch: {len(rows)} != {len(poses)}"
        )
    return rows


def _planner_result_provenance(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: result[key]
        for key in (
            "planner_mode",
            "planning_adapter_version",
            "batch_exception",
            "world_revision",
            "world_hash",
            "attachment_revision",
            "attachment_hash",
            "planner_status",
            "held_collision_policy",
            "fixed_camera_pair",
            "fixed_camera_path",
            "table_validation",
            "candidate_search_budget",
            "candidate_search_budget_applied",
        )
        if key in result
    }


def _current_ee_pose(task: Any, arm: str) -> np.ndarray:
    if hasattr(task, "get_arm_pose"):
        raw = task.get_arm_pose(arm)
    elif str(arm) == "left":
        raw = task.robot.get_left_ee_pose()
    else:
        raw = task.robot.get_right_ee_pose()
    if hasattr(raw, "p") and hasattr(raw, "q"):
        values = [*raw.p, *raw.q]
    else:
        values = raw
    pose = np.asarray(values, dtype=float)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise GraspContractError("current end-effector pose must be xyz+quat_wxyz")
    return pose


def _path(result: Mapping[str, Any] | None) -> np.ndarray | None:
    if not isinstance(result, Mapping) or str(result.get("status")) != "Success":
        return None
    raw = result.get("position")
    if raw is None:
        return None
    path = np.asarray(raw, dtype=float)
    if path.ndim != 2 or len(path) == 0 or not np.isfinite(path).all():
        return None
    return path


def _root_matrix(actor: Any) -> np.ndarray:
    pose = actor.get_pose()
    if hasattr(pose, "to_transformation_matrix"):
        return np.asarray(pose.to_transformation_matrix(), dtype=float)
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = t3d.quaternions.quat2mat(np.asarray(pose.q, dtype=float))
    matrix[:3, 3] = np.asarray(pose.p, dtype=float)
    return matrix


def _pose_vector_matrix(pose: Sequence[float]) -> np.ndarray:
    value = np.asarray(pose, dtype=float)
    if value.shape != (7,) or not np.isfinite(value).all():
        raise GraspContractError("pose must contain finite xyz+quat_wxyz")
    quaternion = value[3:]
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise GraspContractError("pose quaternion has zero norm")
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = t3d.quaternions.quat2mat(quaternion / norm)
    matrix[:3, 3] = value[:3]
    return matrix


def _matrix_pose_vector(matrix: Sequence[Sequence[float]]) -> list[float]:
    value = np.asarray(matrix, dtype=float)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise GraspContractError("pose matrix must be finite 4x4")
    return clean_vector(
        [*value[:3, 3], *t3d.quaternions.mat2quat(value[:3, :3])]
    )


def _obstacle_from_entity(spec: Mapping[str, Any], actor: Any, name: str) -> dict[str, Any] | None:
    initial = spec.get("initial_pose") or {}
    bounds = initial.get("origin_relative_bounds") or {}
    minimum = np.asarray(bounds.get("min_xyz") or [], dtype=float)
    maximum = np.asarray(bounds.get("max_xyz") or [], dtype=float)
    if minimum.shape != (3,) or maximum.shape != (3,):
        return None
    dimensions = maximum - minimum
    if not np.isfinite(dimensions).all() or np.any(dimensions <= 0.0):
        return None
    q0 = np.asarray(initial.get("q0_wxyz") or [], dtype=float)
    if q0.shape != (4,):
        return None
    pose = actor.get_pose()
    live_rotation = t3d.quaternions.quat2mat(np.asarray(pose.q, dtype=float))
    q0_rotation = t3d.quaternions.quat2mat(q0)
    # R01 bounds are expressed around the actor origin after applying q0.
    # The delta maps those frozen axes into the actor's current live rotation.
    delta_rotation = live_rotation @ q0_rotation.T
    relative_center = (minimum + maximum) * 0.5
    world_center = np.asarray(pose.p, dtype=float) + delta_rotation @ relative_center
    world_quaternion = t3d.quaternions.mat2quat(delta_rotation)
    return {
        "name": str(name),
        "dims": clean_vector(dimensions),
        "pose": clean_vector([*world_center, *world_quaternion]),
        "source": "r01_origin_relative_bounds_at_live_root_pose",
    }


def scene_collision_obstacles(
    task: Any,
    *,
    target_actor: Any,
) -> list[dict[str, Any]]:
    """Build conservative live OBBs for every non-target scene entity."""

    scene_spec = getattr(task, "scene_spec", None)
    entities = getattr(task, "entities", None)
    if not isinstance(scene_spec, Mapping) or not isinstance(entities, Mapping):
        return []
    specs = scene_spec.get("entities") or {}
    rows = []
    for entity_key, actor in sorted(entities.items()):
        if actor is target_actor:
            continue
        spec = specs.get(entity_key)
        if not isinstance(spec, Mapping):
            raise CollisionWorldUpdateError(
                f"collision entity has no SceneSpec record: {entity_key}"
            )
        obstacle = _obstacle_from_entity(spec, actor, entity_key)
        if obstacle is None:
            raise CollisionWorldUpdateError(
                f"collision entity lacks valid R01 bounds: {entity_key}"
            )
        rows.append(obstacle)
    return rows


def refresh_collision_world(
    task: Any,
    *,
    target_actor: Any,
    extra_obstacles: Sequence[Mapping[str, Any]] = (),
    arm: str | None = None,
) -> dict[str, Any]:
    obstacles = [
        *scene_collision_obstacles(task, target_actor=target_actor),
        *[deepcopy(dict(row)) for row in extra_obstacles],
    ]
    selected_updater = getattr(
        task.robot, "update_arm_world_obstacles", None
    )
    selected_only = arm in {"left", "right"} and callable(selected_updater)
    updater = (
        selected_updater
        if selected_only
        else getattr(task.robot, "update_world_obstacles", None)
    )
    if not callable(updater):
        raise CollisionWorldUpdateError("robot has no strict collision-world updater")
    try:
        result = updater(str(arm), obstacles) if selected_only else updater(obstacles)
    except TimeoutError:
        raise
    except Exception as exc:
        raise CollisionWorldUpdateError(
            f"collision world update failed: {type(exc).__name__}:{exc}"
        ) from exc
    if not isinstance(result, Mapping) or not bool(result.get("updated")):
        raise CollisionWorldUpdateError(
            f"collision world update did not confirm success: {result!r}"
        )
    return {**dict(result), "obstacles": deepcopy(obstacles)}


def _rejected_variant(
    base: Mapping[str, Any],
    *,
    reason: str,
    retreat_distance_m: float,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        **dict(base),
        "status": "rejected",
        "rejection_reason": str(reason),
        "retreat_distance_m": float(retreat_distance_m),
    }
    if details:
        row["rejection_details"] = deepcopy(dict(details))
    row["candidate_id"] = candidate_id(row)
    return row


def _plan_exact_contact_candidates_single_retreat(
    task: Any,
    actor: Any,
    *,
    arm: str,
    asset: str,
    model_id: str,
    use: str = "pick_subject",
    use_profile: str = "single_arm_pick",
    contact_groups: Sequence[Sequence[int]] = (),
    group_index: int | None = None,
    contact_ids: Sequence[int] | None = None,
    table_acquisition: bool = True,
    retreat_distances_m: Sequence[float] | None = None,
    minimum_normalized_joint_margin: float | None = None,
    world_yaw_deg: float | None = None,
    reach_zone: str | None = None,
    include_trajectories: bool = False,
    actor_root_matrix: Sequence[Sequence[float]] | None = None,
    support_plane_z_m: float | None = None,
    pre_grasp_obstacles: Sequence[Mapping[str, Any]] = (),
    grasp_obstacles: Sequence[Mapping[str, Any]] = (),
    require_attached_lift_witness: bool = False,
    required_pick_clearance_m: float | None = None,
    lift_tracking_guard_m: float = 0.0,
    post_lift_target_quats_wxyz: Sequence[Sequence[float]] = (),
    candidate_admission=None,
    stop_after_first_reachable_contact: bool = False,
    candidate_search_budget: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Plan both 6D segments for every permitted authored Contact.

    ``actor_root_matrix`` projects authored Contacts at a hypothetical actor
    pose without mutating the live simulator.  M15 handover uses this to prove
    that the receiving arm can reach the object *after* the source transfer,
    before either arm is physically moved.
    """

    budget = normalize_candidate_search_budget(candidate_search_budget)
    candidate_search_kwargs = (
        {"candidate_search_budget": budget} if budget is not None else {}
    )
    arm_name = str(arm)
    if arm_name not in {"left", "right"}:
        raise GraspContractError(f"unsupported arm: {arm_name}")
    orientation_targets: list[list[float]] = []
    for raw_quaternion in post_lift_target_quats_wxyz:
        quaternion = np.asarray(raw_quaternion, dtype=float)
        if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
            raise GraspContractError(
                "post-lift target orientation must be finite quat_wxyz"
            )
        norm = float(np.linalg.norm(quaternion))
        if norm <= 1e-12:
            raise GraspContractError(
                "post-lift target orientation has zero norm"
            )
        quaternion = quaternion / norm
        if quaternion[0] < 0.0:
            quaternion = -quaternion
        orientation_targets.append(clean_vector(quaternion))
    matrices = _contact_matrices(actor)
    contract = build_contact_contract(
        asset=str(asset),
        model_id=str(model_id),
        use=str(use),
        contact_matrices=matrices,
        contact_groups=contact_groups,
        use_profile=use_profile,
    )
    allowed = allowed_contact_ids(
        len(matrices),
        use_profile=use_profile,
        groups=contact_groups,
        group_index=group_index,
    )
    if contact_ids is not None:
        requested = list(dict.fromkeys(int(value) for value in contact_ids))
        invalid = sorted(set(requested) - set(allowed))
        if invalid:
            raise GraspContractError(
                f"requested Contacts are outside {use_profile} scope: {invalid}"
            )
        allowed = requested
    retreats = [
        float(value)
        for value in (
            retreat_distances_m
            if retreat_distances_m is not None
            else EXACT_GRASP_POLICY["retreat_distances_m"]
        )
    ]
    if not retreats or len(set(retreats)) != len(retreats) or any(
        value < 0.0 for value in retreats
    ):
        raise GraspContractError("retreat distances must be unique and non-negative")
    margin_floor = float(
        minimum_normalized_joint_margin
        if minimum_normalized_joint_margin is not None
        else EXACT_GRASP_POLICY["minimum_normalized_joint_margin"]
    )
    if margin_floor < 0.0:
        raise GraspContractError("joint-margin floor must be non-negative")
    memberships = contact_groups_for_ids(
        len(matrices),
        contact_groups,
        strict=use_profile != "single_arm_pick",
    )
    if actor_root_matrix is None:
        root = _root_matrix(actor)
    else:
        root = np.asarray(actor_root_matrix, dtype=float)
        if root.shape != (4, 4) or not np.isfinite(root).all():
            raise GraspContractError(
                "actor root override must contain one finite 4x4 matrix"
            )
    scale = _actor_scale(actor)
    attached_lift_delta_m = 0.0
    if require_attached_lift_witness:
        if actor_root_matrix is not None:
            raise GraspContractError(
                "attached held-lift witness requires a live actor pose"
            )
        if support_plane_z_m is None or required_pick_clearance_m is None:
            raise GraspContractError(
                "attached held-lift witness requires support plane and clearance"
            )
        bounds = actor_aabb(actor)
        minimum = np.asarray(bounds["min"], dtype=float)
        maximum = np.asarray(bounds["max"], dtype=float)
        center = np.asarray(bounds["center"], dtype=float)
        if any(value.shape != (3,) for value in (minimum, maximum, center)):
            raise GraspContractError("live actor AABB is malformed")
        attached_lift_delta_m = continuous_acquisition_lift_delta(
            clearance_before_m=float(minimum[2] - float(support_plane_z_m)),
            required_clearance_m=float(required_pick_clearance_m),
            tracking_guard_m=float(lift_tracking_guard_m),
        )
    joint_limits = _joint_limits(task, arm_name)
    current_ee = _current_ee_pose(task, arm_name)
    contact_rows: list[dict[str, Any]] = []
    variants: list[dict[str, Any]] = []
    try:
        collision_world = refresh_collision_world(
            task, target_actor=actor, arm=arm_name
        )
        collision_failure = None
    except CollisionWorldUpdateError as exc:
        collision_world = {
            "updated": False,
            "error": str(exc),
            "obstacles": [],
        }
        collision_failure = str(exc)

    planning_requests: list[dict[str, Any]] = []
    for contact_id in allowed:
        base = {
            "contact_contract_hash": contract["contact_contract_hash"],
            "asset": str(asset),
            "model_id": str(model_id),
            "use": str(use),
            "use_profile": str(use_profile),
            "arm": arm_name,
            "world_yaw_deg": (
                None if world_yaw_deg is None else float(world_yaw_deg)
            ),
            "reach_zone": reach_zone,
            "contact_id": int(contact_id),
            "contact_group_indices": memberships[int(contact_id)],
        }
        try:
            scaled_local, world = scaled_contact_world_matrix(
                matrices[contact_id], scale, root
            )
            if actor_root_matrix is None:
                actor_world = np.asarray(
                    actor.get_contact_point(contact_id, "matrix"), dtype=float
                )
                if actor_world.shape != (4, 4) or not np.allclose(
                    actor_world, world, atol=1e-8, rtol=0.0
                ):
                    raise GraspContractError(
                        "projected Contact differs from Actor.get_contact_point"
                    )
            direction = approach_direction(world)
            approach_class = classify_approach(
                direction, table_acquisition=table_acquisition
            )
            contact_row = {
                **base,
                "local_matrix": clean_matrix(matrices[contact_id]),
                "scaled_local_matrix": clean_matrix(scaled_local),
                "world_matrix": clean_matrix(world),
                "world_position": clean_vector(world[:3, 3]),
                "approach_direction_world": clean_vector(direction),
                "approach_class": approach_class,
                "variant_count": len(retreats),
            }
        except TimeoutError:
            raise
        except Exception as exc:
            contact_rows.append(
                {
                    **base,
                    "status": "rejected",
                    "rejection_reason": "invalid_contact_matrix",
                    "rejection_details": {
                        "error": f"{type(exc).__name__}:{exc}"
                    },
                }
            )
            for retreat in retreats:
                variants.append(
                    _rejected_variant(
                        base,
                        reason="invalid_contact_matrix",
                        retreat_distance_m=retreat,
                        details={"error": f"{type(exc).__name__}:{exc}"},
                    )
                )
            continue
        if collision_failure is not None:
            contact_rows.append(
                {
                    **contact_row,
                    "status": "rejected",
                    "rejection_reason": "collision_world_update_failed",
                }
            )
            for retreat in retreats:
                variants.append(
                    _rejected_variant(
                        contact_row,
                        reason="collision_world_update_failed",
                        retreat_distance_m=retreat,
                        details={"error": collision_failure},
                    )
                )
            continue
        if table_acquisition and approach_class == "upward":
            contact_rows.append(
                {
                    **contact_row,
                    "status": "rejected",
                    "rejection_reason": "upward_from_below_table_approach",
                }
            )
            for retreat in retreats:
                variants.append(
                    _rejected_variant(
                        contact_row,
                        reason="upward_from_below_table_approach",
                        retreat_distance_m=retreat,
                    )
                )
            continue
        contact_rows.append({**contact_row, "status": "evaluated"})
        for retreat in retreats:
            poses = (
                support_safe_grasp_poses(
                    world,
                    retreat,
                    support_plane_z_m=float(support_plane_z_m),
                )
                if table_acquisition and support_plane_z_m is not None
                else {
                    **exact_grasp_poses(world, retreat),
                    "authored_contact_position_world": clean_vector(
                        world[:3, 3]
                    ),
                    "effective_contact_position_world": clean_vector(
                        world[:3, 3]
                    ),
                    "support_clearance_adjustment_m": 0.0,
                    "support_plane_z_m": support_plane_z_m,
                    "inner_pad_contains_authored_contact": True,
                }
            )
            candidate_base = {
                **contact_row,
                **poses,
                "cartesian_path_length_m": round(
                    float(
                        np.linalg.norm(
                            np.asarray(poses["pre_grasp_pose"][:3])
                            - current_ee[:3]
                        )
                        + np.linalg.norm(
                            np.asarray(poses["grasp_pose"][:3])
                            - np.asarray(poses["pre_grasp_pose"][:3])
                        )
                    ),
                    12,
                ),
            }
            if not bool(poses["inner_pad_contains_authored_contact"]):
                variants.append(
                    _rejected_variant(
                        candidate_base,
                        reason="support_projection_outside_inner_pad",
                        retreat_distance_m=retreat,
                        details={
                            "authored_contact_in_gripper_xyz_m": poses.get(
                                "authored_contact_in_gripper_xyz_m"
                            ),
                            "support_clearance_adjustment_m": poses.get(
                                "support_clearance_adjustment_m"
                            ),
                        },
                    )
                )
                continue
            planning_requests.append(
                {
                    "candidate_base": candidate_base,
                    "retreat_distance_m": retreat,
                    "pre_grasp_pose": poses["pre_grasp_pose"],
                    "grasp_pose": poses["grasp_pose"],
                }
            )

    pre_grasp_collision_world = collision_world
    pre_grasp_collision_failure = None
    if pre_grasp_obstacles and planning_requests:
        try:
            pre_grasp_collision_world = refresh_collision_world(
                task,
                target_actor=actor,
                extra_obstacles=pre_grasp_obstacles,
                arm=arm_name,
            )
        except CollisionWorldUpdateError as exc:
            pre_grasp_collision_failure = str(exc)
    if pre_grasp_collision_failure is not None:
        for request in planning_requests:
            variants.append(
                _rejected_variant(
                    request["candidate_base"],
                    reason="pre_grasp_collision_world_update_failed",
                    retreat_distance_m=float(request["retreat_distance_m"]),
                    details={"error": pre_grasp_collision_failure},
                )
            )
        planning_requests = []

    pre_results = _batch_plan_paths(
        task,
        arm_name,
        [row["pre_grasp_pose"] for row in planning_requests],
        **candidate_search_kwargs,
    )
    pre_paths = [_path(result) for result in pre_results]
    physical_preflight_by_request: dict[int, dict[str, Any]] = {}
    physical_validator = getattr(
        task.robot, "validate_arm_path_physical_collisions", None
    )
    if actor_root_matrix is None and callable(physical_validator):
        for index, path in enumerate(pre_paths):
            if path is None:
                continue
            try:
                preflight = dict(
                    physical_validator(
                        arm_name,
                        path,
                        sample_stride=1,
                        # High alignment precedes any permitted target
                        # contact. Excluding the whole subject also hides
                        # forearm collisions that move it before Close.
                        excluded_entities=(
                            () if getattr(task, "_m15_strict_pregrasp_collision", False)
                            else (actor,)
                        ),
                    )
                )
            except TimeoutError:
                raise
            except Exception as exc:
                raise PhysicalCollisionValidationError(
                    "exact-mesh preflight failed for "
                    f"{arm_name} request {index}: {type(exc).__name__}: {exc}"
                ) from exc
            collision_free = preflight.get("collision_free")
            if not isinstance(collision_free, bool):
                raise PhysicalCollisionValidationError(
                    "exact-mesh preflight returned no boolean collision result "
                    f"for {arm_name} request {index}: {preflight!r}"
                )
            if preflight.get("error"):
                raise PhysicalCollisionValidationError(
                    "exact-mesh preflight reported a validator error for "
                    f"{arm_name} request {index}: {preflight['error']}"
                )
            if not collision_free and not preflight.get("collisions"):
                raise PhysicalCollisionValidationError(
                    "exact-mesh preflight rejected a path without collision "
                    f"evidence for {arm_name} request {index}"
                )
            physical_preflight_by_request[index] = preflight
    grasp_request_indices = [
        index
        for index, path in enumerate(pre_paths)
        if path is not None
        and bool(
            physical_preflight_by_request.get(
                index, {"collision_free": True}
            ).get("collision_free")
        )
    ]
    grasp_world_failed_indices: set[int] = set()
    if pre_grasp_obstacles and planning_requests:
        try:
            collision_world = refresh_collision_world(
                task,
                target_actor=actor,
                extra_obstacles=grasp_obstacles,
                arm=arm_name,
            )
        except CollisionWorldUpdateError as exc:
            grasp_world_failed_indices = set(grasp_request_indices)
            for index in grasp_request_indices:
                request = planning_requests[index]
                variants.append(
                    _rejected_variant(
                        request["candidate_base"],
                        reason="grasp_collision_world_update_failed",
                        retreat_distance_m=float(
                            request["retreat_distance_m"]
                        ),
                        details={"error": str(exc)},
                    )
                )
            grasp_request_indices = []
    grasp_results = _batch_plan_paths(
        task,
        arm_name,
        [planning_requests[index]["grasp_pose"] for index in grasp_request_indices],
        start_qposes=[pre_paths[index][-1] for index in grasp_request_indices],
        **candidate_search_kwargs,
    )
    grasp_by_request = dict(zip(grasp_request_indices, grasp_results))
    largest_reachable_retreat_by_contact: dict[int, float] = {}

    # Pre-Grasp and Approach remain one deterministic batch in authored order.
    # Evaluate attached phases by decreasing retreat within each Contact.
    contact_priority = {
        int(contact_id): index for index, contact_id in enumerate(allowed)
    }
    evaluation_order = list(range(len(planning_requests)))
    if require_attached_lift_witness:
        evaluation_order.sort(
            key=lambda index: (
                contact_priority[
                    int(
                        planning_requests[index]["candidate_base"][
                            "contact_id"
                        ]
                    )
                ],
                -float(planning_requests[index]["retreat_distance_m"]),
                index,
            )
        )
    for request_index in evaluation_order:
        request = planning_requests[request_index]
        candidate_base = request["candidate_base"]
        retreat = float(request["retreat_distance_m"])
        pre_result = pre_results[request_index]
        pre_path = pre_paths[request_index]
        if pre_path is None:
            exception = pre_result.get("exception")
            variants.append(
                _rejected_variant(
                    candidate_base,
                    reason=(
                        "pre_grasp_planner_exception"
                        if exception
                        else "no_pre_grasp_path"
                    ),
                    retreat_distance_m=retreat,
                    details=(
                        {
                            "error": exception,
                            **_planner_result_provenance(pre_result),
                        }
                        if exception
                        else {
                            "planner_status": pre_result.get("status"),
                            **_planner_result_provenance(pre_result),
                        }
                    ),
                )
            )
            continue
        physical_preflight = physical_preflight_by_request.get(request_index)
        if physical_preflight is not None and not bool(
            physical_preflight.get("collision_free")
        ):
            variants.append(
                _rejected_variant(
                    candidate_base,
                    reason="physical_mesh_pre_grasp_collision",
                    retreat_distance_m=retreat,
                    details={
                        **physical_preflight,
                        "pre_grasp_path": trajectory_summary(pre_path),
                    },
                )
            )
            continue
        if request_index in grasp_world_failed_indices:
            continue
        grasp_result = grasp_by_request[request_index]
        grasp_path = _path(grasp_result)
        if grasp_path is None:
            exception = grasp_result.get("exception")
            variants.append(
                _rejected_variant(
                    candidate_base,
                    reason=(
                        "grasp_planner_exception"
                        if exception
                        else "no_grasp_path"
                    ),
                    retreat_distance_m=retreat,
                    details={
                        **(
                            {
                                "error": exception,
                                **_planner_result_provenance(grasp_result),
                            }
                            if exception
                            else {
                                "planner_status": grasp_result.get("status"),
                                **_planner_result_provenance(grasp_result),
                            }
                        ),
                        "pre_grasp_path": trajectory_summary(pre_path),
                        "pre_grasp_planner": _planner_result_provenance(
                            pre_result
                        ),
                    },
                )
            )
            continue
        approach_physical_preflight = None
        if actor_root_matrix is None and callable(physical_validator):
            try:
                approach_physical_preflight = dict(
                    physical_validator(
                        arm_name,
                        grasp_path,
                        sample_stride=1,
                        excluded_entities=(actor,),
                    )
                )
            except TimeoutError:
                raise
            except Exception as exc:
                raise PhysicalCollisionValidationError(
                    "exact-mesh fixed-orientation Approach preflight failed "
                    f"for {arm_name} request {request_index}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            collision_free = approach_physical_preflight.get(
                "collision_free"
            )
            if not isinstance(collision_free, bool):
                raise PhysicalCollisionValidationError(
                    "exact-mesh fixed-orientation Approach preflight returned "
                    "no boolean collision result for "
                    f"{arm_name} request {request_index}: "
                    f"{approach_physical_preflight!r}"
                )
            if approach_physical_preflight.get("error"):
                raise PhysicalCollisionValidationError(
                    "exact-mesh fixed-orientation Approach preflight reported "
                    f"a validator error for {arm_name} request "
                    f"{request_index}: {approach_physical_preflight['error']}"
                )
            if not collision_free and not approach_physical_preflight.get(
                "collisions"
            ):
                raise PhysicalCollisionValidationError(
                    "exact-mesh fixed-orientation Approach preflight rejected "
                    "a path without collision evidence for "
                    f"{arm_name} request {request_index}"
                )
            if not collision_free:
                variants.append(
                    _rejected_variant(
                        candidate_base,
                        reason=(
                            "physical_mesh_fixed_orientation_approach_collision"
                        ),
                        retreat_distance_m=retreat,
                        details={
                            **approach_physical_preflight,
                            "grasp_path": trajectory_summary(grasp_path),
                        },
                    )
                )
                continue
        preliminary_metrics = joint_path_metrics(
            [pre_path, grasp_path], joint_limits
        )
        if float(
            preliminary_metrics["minimum_joint_limit_margin"]
        ) < margin_floor:
            variants.append(
                _rejected_variant(
                    {
                        **candidate_base,
                        "path_metrics": preliminary_metrics,
                    },
                    reason="unsafe_joint_margin",
                    retreat_distance_m=retreat,
                    details={"required_minimum": margin_floor},
                )
            )
            continue
        contact_id = int(candidate_base["contact_id"])
        dominating_retreat = largest_reachable_retreat_by_contact.get(contact_id)
        if (
            require_attached_lift_witness
            and dominating_retreat is not None
            and retreat < dominating_retreat - 1e-12
        ):
            variants.append(
                _rejected_variant(
                    {
                        **candidate_base,
                        "path_metrics": preliminary_metrics,
                    },
                    reason="dominated_by_larger_reachable_retreat",
                    retreat_distance_m=retreat,
                    details={
                        "dominating_retreat_distance_m": float(
                            dominating_retreat
                        ),
                        "selection_contract": (
                            "largest_reachable_retreat_per_contact"
                        ),
                    },
                )
            )
            continue
        admission_receipt = None
        held_lift_result = None
        held_lift_path = None
        held_lift_attachment = None
        held_lift_target_pose = None
        orientation_results: list[dict[str, Any]] = []
        orientation_paths: list[np.ndarray] = []
        orientation_target_ee_poses: list[list[float]] = []
        if require_attached_lift_witness:
            attach = getattr(task.robot, "attach_collision_object", None)
            detach = getattr(task.robot, "detach_collision_object", None)
            if not callable(attach) or not callable(detach):
                raise RuntimeError(
                    "attached_held_lift_planner_interface_missing"
                )
            held_lift_target_pose = list(request["grasp_pose"])
            held_lift_target_pose[2] = float(
                held_lift_target_pose[2] + attached_lift_delta_m
            )
            with temporary_held_collision(
                task, actor, arm_name, name=f"held:{asset}:{model_id}",
                last_qpos=grasp_path[-1], approach_world=collision_world,
            ) as binding:
                held_lift_attachment = binding["attachment"]
                held_world_hash = binding["world"][arm_name]["world_hash"]
                held_lift_result = plan_held_path(
                    task, actor, arm_name, grasp_path[-1], held_lift_target_pose,
                    world_hash=held_world_hash,
                    attachment_hash=held_lift_attachment["attachment_hash"],
                    support_plane_z_m=support_plane_z_m,
                    required_clearance_m=required_pick_clearance_m,
                )
                held_lift_path = _path(held_lift_result)
                if held_lift_path is not None and orientation_targets:
                    grasp_ee_matrix = _pose_vector_matrix(request["grasp_pose"])
                    actor_to_ee = np.linalg.inv(root) @ grasp_ee_matrix
                    lifted_actor_position = root[:3, 3].copy()
                    lifted_actor_position[2] += attached_lift_delta_m
                    for target_quaternion in orientation_targets:
                        target_actor_matrix = np.eye(4, dtype=float)
                        target_actor_matrix[:3, :3] = t3d.quaternions.quat2mat(
                            np.asarray(target_quaternion, dtype=float))
                        target_actor_matrix[:3, 3] = lifted_actor_position
                        target = _matrix_pose_vector(target_actor_matrix @ actor_to_ee)
                        orientation_target_ee_poses.append(target)
                        result = plan_held_path(task, actor, arm_name, held_lift_path[-1], target,
                            world_hash=held_world_hash,
                            attachment_hash=held_lift_attachment["attachment_hash"])
                        orientation_results.append(result)
                        path = _path(result)
                        if path is not None:
                            orientation_paths.append(path)
                if held_lift_path is not None and candidate_admission is not None:
                    admission_receipt = candidate_admission(
                        contact_id=int(contact_id), grasp_pose=list(request["grasp_pose"]),
                        lifted_qpos=held_lift_path[-1], binding=binding)
                    if not isinstance(admission_receipt, Mapping) or not isinstance(admission_receipt.get("passed"), bool):
                        raise RuntimeError("invalid goal followthrough admission receipt")
            if admission_receipt is not None and not admission_receipt["passed"]:
                variants.append(_rejected_variant(candidate_base,
                    reason="no_goal_followthrough_in_budget", retreat_distance_m=retreat,
                    details={"goal_followthrough": admission_receipt}))
                continue
            if held_lift_path is None:
                exception = (held_lift_result or {}).get("exception")
                variants.append(
                    _rejected_variant(
                        candidate_base,
                        reason=(
                            "attached_held_lift_planner_exception"
                            if exception
                            else "no_attached_held_lift_path"
                        ),
                        retreat_distance_m=retreat,
                        details={
                            **(
                                {"error": exception}
                                if exception
                                else {
                                    "planner_status": (
                                        held_lift_result or {}
                                    ).get("status")
                                }
                            ),
                            **_planner_result_provenance(
                                held_lift_result or {}
                            ),
                            "held_lift_delta_m": float(
                                attached_lift_delta_m
                            ),
                            "held_lift_target_pose": held_lift_target_pose,
                        },
                    )
                )
                continue
            if orientation_targets and len(orientation_paths) != len(
                orientation_targets
            ):
                failed_orientation_rows = [
                    {
                        "target_quat_wxyz": orientation_targets[index],
                        "target_ee_pose": (
                            orientation_target_ee_poses[index]
                            if index < len(orientation_target_ee_poses)
                            else None
                        ),
                        "planner_status": result.get("status"),
                        "exception": result.get("exception"),
                        **_planner_result_provenance(result),
                    }
                    for index, result in enumerate(orientation_results)
                    if _path(result) is None
                ]
                variants.append(
                    _rejected_variant(
                        candidate_base,
                        reason=(
                            "attached_post_lift_orientation_planner_exception"
                            if any(
                                row.get("exception")
                                for row in failed_orientation_rows
                            )
                            else "no_attached_post_lift_orientation_path"
                        ),
                        retreat_distance_m=retreat,
                        details={
                            "failed_orientation_targets": (
                                failed_orientation_rows
                            ),
                            "required_orientation_target_count": len(
                                orientation_targets
                            ),
                        },
                    )
                )
                continue
        planned_paths = [pre_path, grasp_path]
        if held_lift_path is not None:
            planned_paths.append(held_lift_path)
        planned_paths.extend(orientation_paths)
        metrics = joint_path_metrics(planned_paths, joint_limits)
        if float(metrics["minimum_joint_limit_margin"]) < margin_floor:
            variants.append(
                _rejected_variant(
                    {**candidate_base, "path_metrics": metrics},
                    reason="unsafe_joint_margin",
                    retreat_distance_m=retreat,
                    details={"required_minimum": margin_floor},
                )
            )
            continue
        path_record = {
            "pre_grasp": trajectory_summary(pre_path),
            "grasp": trajectory_summary(grasp_path),
            "pre_grasp_planner": _planner_result_provenance(pre_result),
            "grasp_planner": _planner_result_provenance(grasp_result),
            "chained_trajectory_sha256": trajectory_fingerprint(
                planned_paths
            ),
            "second_segment_start_qpos": clean_vector(pre_path[-1]),
        }
        if physical_preflight is not None:
            path_record["pre_grasp_physical_collision_preflight"] = (
                physical_preflight
            )
        if approach_physical_preflight is not None:
            path_record[
                "fixed_orientation_approach_physical_collision_preflight"
            ] = approach_physical_preflight
        if held_lift_path is not None:
            path_record.update(
                {
                    "attached_held_lift": trajectory_summary(
                        held_lift_path
                    ),
                    "attached_held_lift_planner": (
                        _planner_result_provenance(held_lift_result or {})
                    ),
                    "attached_held_lift_delta_m": float(
                        attached_lift_delta_m
                    ),
                    "attached_held_lift_target_pose": list(
                        held_lift_target_pose or request["grasp_pose"]
                    ),
                    "attached_held_lift_attachment": deepcopy(
                        held_lift_attachment
                    ),
                    "third_segment_start_qpos": clean_vector(
                        grasp_path[-1]
                    ),
                }
            )
        if orientation_targets:
            path_record.update(
                {
                    "attached_post_lift_orientation_targets_wxyz": deepcopy(
                        orientation_targets
                    ),
                    "attached_post_lift_orientation_target_ee_poses": deepcopy(
                        orientation_target_ee_poses
                    ),
                    "attached_post_lift_orientation_paths": [
                        trajectory_summary(path) for path in orientation_paths
                    ],
                    "attached_post_lift_orientation_planners": [
                        _planner_result_provenance(result)
                        for result in orientation_results
                    ],
                }
            )
        if admission_receipt is not None:
            path_record["goal_followthrough"] = admission_receipt
        candidate = {
            **candidate_base,
            "status": "reachable",
            "rejection_reason": None,
            "path_metrics": metrics,
            "paths": path_record,
            "minimum_required_joint_limit_margin": margin_floor,
        }
        candidate["candidate_id"] = candidate_id(candidate)
        if include_trajectories:
            pre_grasp_velocity = np.asarray(
                pre_result.get("velocity", []), dtype=float
            )
            if pre_grasp_velocity.shape != pre_path.shape:
                pre_grasp_velocity = np.zeros_like(pre_path)
            grasp_velocity = np.asarray(
                grasp_result.get("velocity", []), dtype=float
            )
            if grasp_velocity.shape != grasp_path.shape:
                grasp_velocity = np.zeros_like(grasp_path)
            candidate["_trajectories"] = {
                "pre_grasp": pre_path.copy(),
                "pre_grasp_velocity": pre_grasp_velocity.copy(),
                "grasp": grasp_path.copy(),
                "grasp_velocity": grasp_velocity.copy(),
            }
            if held_lift_path is not None:
                held_lift_velocity = np.asarray(
                    (held_lift_result or {}).get("velocity", []),
                    dtype=float,
                )
                if held_lift_velocity.shape != held_lift_path.shape:
                    held_lift_velocity = np.zeros_like(held_lift_path)
                candidate["_trajectories"].update(
                    {
                        "attached_held_lift": held_lift_path.copy(),
                        "attached_held_lift_velocity": (
                            held_lift_velocity.copy()
                        ),
                    }
                )
        if pre_grasp_obstacles:
            candidate["_pre_grasp_obstacles"] = [
                deepcopy(dict(row)) for row in pre_grasp_obstacles
            ]
        if grasp_obstacles:
            candidate["_grasp_obstacles"] = [
                deepcopy(dict(row)) for row in grasp_obstacles
            ]
        variants.append(candidate)
        largest_reachable_retreat_by_contact[contact_id] = max(
            retreat,
            largest_reachable_retreat_by_contact.get(contact_id, retreat),
        )
        if stop_after_first_reachable_contact:
            break

    selected = select_best_retreat_per_contact(variants)
    selected_ids = {int(row["contact_id"]) for row in selected}
    rejection_counts: dict[str, int] = {}
    for row in variants:
        reason = row.get("rejection_reason")
        if reason:
            rejection_counts[str(reason)] = rejection_counts.get(str(reason), 0) + 1
    result = {
        "policy_version": EXACT_GRASP_POLICY["version"],
        "planning_adapter_version": PLANNING_ADAPTER_VERSION,
        "contact_contract_id": contract["contact_contract_id"],
        "contact_contract_hash": contract["contact_contract_hash"],
        "asset": str(asset),
        "model_id": str(model_id),
        "use": str(use),
        "use_profile": str(use_profile),
        "arm": arm_name,
        "world_yaw_deg": world_yaw_deg,
        "reach_zone": reach_zone,
        "hypothetical_actor_pose": actor_root_matrix is not None,
        "table_acquisition": bool(table_acquisition),
        "support_plane_z_m": support_plane_z_m,
        "require_attached_lift_witness": bool(
            require_attached_lift_witness
        ),
        "required_pick_clearance_m": required_pick_clearance_m,
        "lift_tracking_guard_m": float(lift_tracking_guard_m),
        "attached_lift_delta_m": float(attached_lift_delta_m),
        "post_lift_target_quats_wxyz": deepcopy(orientation_targets),
        "allowed_contact_ids": allowed,
        "evaluated_contact_count": len(allowed),
        "retreat_distances_m": retreats,
        "candidate_variant_count": len(variants),
        "reachable_variant_count": sum(
            str(row.get("status")) == "reachable" for row in variants
        ),
        "reachable_contact_count": len(selected),
        "selected_contact_ids": sorted(selected_ids),
        "contacts": contact_rows,
        "variants": variants,
        "ranked_contact_candidates": selected,
        "best_candidate": selected[0] if selected else None,
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "collision_world": collision_world,
        "pre_grasp_collision_world": pre_grasp_collision_world,
        "status": "witness_found" if selected else "no_reachable_contact",
    }
    if budget is not None:
        result.update({
            "candidate_search_budget": budget,
            "candidate_search_budget_applied": any(
                row.get("candidate_search_budget_applied") is True
                for row in [*pre_results, *grasp_results]
            ),
            "candidate_search_budget_stages": ["pre_grasp", "approach"],
            "candidate_search_batch_pose_capacity": PLANNER_BATCH_POSE_CAPACITY,
            "candidate_search_success_ratio_semantics": "padded_pose_rows_not_unique_contacts",
        })
    return result


def _pruned_retreat_variant(
    contact: Mapping[str, Any],
    retreat: float,
    *,
    current_ee: np.ndarray,
    table_acquisition: bool,
    support_plane_z_m: float | None,
    reason: str,
    details: Mapping[str, Any],
) -> dict[str, Any]:
    """Build explicit evidence for a planner call dominated by a witness."""

    world = np.asarray(contact["world_matrix"], dtype=float)
    poses = (
        support_safe_grasp_poses(
            world,
            retreat,
            support_plane_z_m=float(support_plane_z_m),
        )
        if table_acquisition and support_plane_z_m is not None
        else {
            **exact_grasp_poses(world, retreat),
            "authored_contact_position_world": clean_vector(world[:3, 3]),
            "effective_contact_position_world": clean_vector(world[:3, 3]),
            "support_clearance_adjustment_m": 0.0,
            "support_plane_z_m": support_plane_z_m,
            "inner_pad_contains_authored_contact": True,
        }
    )
    base = {**deepcopy(dict(contact)), **poses}
    for key in ("status", "rejection_reason", "rejection_details"):
        base.pop(key, None)
    base["cartesian_path_length_m"] = round(
        float(
            np.linalg.norm(
                np.asarray(poses["pre_grasp_pose"][:3]) - current_ee[:3]
            )
            + np.linalg.norm(
                np.asarray(poses["grasp_pose"][:3])
                - np.asarray(poses["pre_grasp_pose"][:3])
            )
        ),
        12,
    )
    return _rejected_variant(
        base,
        reason=reason,
        retreat_distance_m=retreat,
        details=details,
    )


def plan_exact_contact_candidates(
    task: Any,
    actor: Any,
    *,
    arm: str,
    asset: str,
    model_id: str,
    use: str = "pick_subject",
    use_profile: str = "single_arm_pick",
    contact_groups: Sequence[Sequence[int]] = (),
    group_index: int | None = None,
    contact_ids: Sequence[int] | None = None,
    table_acquisition: bool = True,
    retreat_distances_m: Sequence[float] | None = None,
    minimum_normalized_joint_margin: float | None = None,
    world_yaw_deg: float | None = None,
    reach_zone: str | None = None,
    include_trajectories: bool = False,
    actor_root_matrix: Sequence[Sequence[float]] | None = None,
    support_plane_z_m: float | None = None,
    pre_grasp_obstacles: Sequence[Mapping[str, Any]] = (),
    grasp_obstacles: Sequence[Mapping[str, Any]] = (),
    require_attached_lift_witness: bool = False,
    required_pick_clearance_m: float | None = None,
    lift_tracking_guard_m: float = 0.0,
    post_lift_target_quats_wxyz: Sequence[Sequence[float]] = (),
    candidate_admission=None,
    stop_after_first_contact: bool = False,
    progressive_retreat_pruning: bool = True,
    candidate_search_budget: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Plan Contacts with a descending, proof-preserving retreat ladder."""

    budget = normalize_candidate_search_budget(candidate_search_budget)
    retreats = [
        float(value)
        for value in (
            retreat_distances_m
            if retreat_distances_m is not None
            else EXACT_GRASP_POLICY["retreat_distances_m"]
        )
    ]
    if not retreats or len(set(retreats)) != len(retreats) or any(
        value < 0.0 for value in retreats
    ):
        raise GraspContractError(
            "retreat distances must be unique and non-negative"
        )
    common = {
        "arm": arm,
        "asset": asset,
        "model_id": model_id,
        "use": use,
        "use_profile": use_profile,
        "contact_groups": contact_groups,
        "group_index": group_index,
        "table_acquisition": table_acquisition,
        "minimum_normalized_joint_margin": minimum_normalized_joint_margin,
        "world_yaw_deg": world_yaw_deg,
        "reach_zone": reach_zone,
        "include_trajectories": include_trajectories,
        "actor_root_matrix": actor_root_matrix,
        "support_plane_z_m": support_plane_z_m,
        "pre_grasp_obstacles": pre_grasp_obstacles,
        "grasp_obstacles": grasp_obstacles,
        "require_attached_lift_witness": require_attached_lift_witness,
        "required_pick_clearance_m": required_pick_clearance_m,
        "lift_tracking_guard_m": lift_tracking_guard_m,
        "post_lift_target_quats_wxyz": post_lift_target_quats_wxyz,
        **({"candidate_admission": candidate_admission} if candidate_admission is not None else {}),
    }
    if budget is not None:
        common["candidate_search_budget"] = budget
    if not require_attached_lift_witness or len(retreats) == 1:
        return _plan_exact_contact_candidates_single_retreat(
            task,
            actor,
            contact_ids=contact_ids,
            retreat_distances_m=retreats,
            stop_after_first_reachable_contact=stop_after_first_contact,
            **common,
        )
    if not progressive_retreat_pruning:
        # The fallback batches the complete Contact x retreat Pre-Grasp and
        # Approach product, which is much cheaper than issuing one failed
        # planner call per Contact. Once one candidate also passes the
        # attached phases, the existential Pick requirement is witnessed and
        # later attached-lift/orientation calls are proof-dominated.
        return _plan_exact_contact_candidates_single_retreat(
            task,
            actor,
            contact_ids=contact_ids,
            retreat_distances_m=retreats,
            stop_after_first_reachable_contact=stop_after_first_contact,
            **common,
        )

    matrices = _contact_matrices(actor)
    allowed = allowed_contact_ids(
        len(matrices),
        use_profile=use_profile,
        groups=contact_groups,
        group_index=group_index,
    )
    if contact_ids is not None:
        requested = list(dict.fromkeys(int(value) for value in contact_ids))
        invalid = sorted(set(requested) - set(allowed))
        if invalid:
            raise GraspContractError(
                f"requested Contacts are outside {use_profile} scope: {invalid}"
            )
        allowed = requested

    pending = list(allowed)
    tiers: list[dict[str, Any]] = []
    dominating_retreats: dict[int, float] = {}
    reachable: dict[int, dict[str, Any]] = {}
    for retreat in sorted(retreats, reverse=True):
        if not pending:
            break
        tier = _plan_exact_contact_candidates_single_retreat(
            task,
            actor,
            contact_ids=pending,
            retreat_distances_m=[retreat],
            stop_after_first_reachable_contact=stop_after_first_contact,
            **common,
        )
        tiers.append(tier)
        resolved = {
            int(row["contact_id"]): deepcopy(dict(row))
            for row in tier.get("ranked_contact_candidates") or []
        }
        for contact_id, candidate in resolved.items():
            dominating_retreats[contact_id] = float(retreat)
            reachable[contact_id] = candidate
        if resolved:
            pending = [value for value in pending if value not in resolved]
            if stop_after_first_contact:
                break

    result = deepcopy(tiers[0])
    variants = [
        deepcopy(dict(row))
        for tier in tiers
        for row in tier.get("variants") or []
    ]
    evaluated = {
        (int(row["contact_id"]), float(row["retreat_distance_m"]))
        for row in variants
    }
    contacts = {
        int(row["contact_id"]): deepcopy(dict(row))
        for row in result.get("contacts") or []
    }
    current_ee = _current_ee_pose(task, str(arm))
    witness_ids = sorted(reachable)
    for contact_id in allowed:
        for retreat in retreats:
            if (int(contact_id), float(retreat)) in evaluated:
                continue
            dominating = dominating_retreats.get(int(contact_id))
            if dominating is not None and retreat < dominating - 1e-12:
                reason = "dominated_by_larger_reachable_retreat"
                details = {
                    "dominating_retreat_distance_m": float(dominating),
                    "selection_contract": "largest_reachable_retreat_per_contact",
                }
            elif stop_after_first_contact and reachable:
                reason = "dominated_by_existing_contact_witness"
                details = {
                    "witness_contact_ids": witness_ids,
                    "selection_contract": "existential_contact_witness",
                }
            else:
                continue
            details["pruned_stages"] = [
                "pre_grasp",
                "fixed_orientation_approach",
                "attached_held_lift",
            ]
            variants.append(
                _pruned_retreat_variant(
                    contacts[int(contact_id)],
                    retreat,
                    current_ee=current_ee,
                    table_acquisition=table_acquisition,
                    support_plane_z_m=support_plane_z_m,
                    reason=reason,
                    details=details,
                )
            )

    selected = select_best_retreat_per_contact(variants)
    rejection_counts: dict[str, int] = {}
    for row in variants:
        reason = row.get("rejection_reason")
        if reason:
            rejection_counts[str(reason)] = (
                rejection_counts.get(str(reason), 0) + 1
            )
    result.update(
        {
            "retreat_distances_m": retreats,
            "candidate_variant_count": len(variants),
            "reachable_variant_count": sum(
                str(row.get("status")) == "reachable" for row in variants
            ),
            "reachable_contact_count": len(selected),
            "selected_contact_ids": sorted(
                int(row["contact_id"]) for row in selected
            ),
            "variants": variants,
            "ranked_contact_candidates": selected,
            "best_candidate": selected[0] if selected else None,
            "rejection_counts": dict(sorted(rejection_counts.items())),
            "status": "witness_found" if selected else "no_reachable_contact",
            "retreat_ladder_execution": "descending_progressive_v1",
        }
    )
    if budget is not None:
        result["candidate_search_budget_applied"] = any(
            tier.get("candidate_search_budget_applied") is True for tier in tiers
        )
    return result


def exact_grasp_actions(
    candidate: Mapping[str, Any],
    *,
    arm: str,
    gripper_pos: float = 0.0,
):
    """Lower one preplanned exact candidate without invoking legacy roll logic."""

    from envs.utils import Action, ArmTag

    arm_tag = ArmTag(str(arm))
    pre_pose = candidate.get("pre_grasp_pose")
    grasp_pose = candidate.get("grasp_pose")
    if not isinstance(pre_pose, Sequence) or len(pre_pose) != 7:
        raise GraspContractError("candidate lacks an exact Pre-Grasp pose")
    if not isinstance(grasp_pose, Sequence) or len(grasp_pose) != 7:
        raise GraspContractError("candidate lacks an exact Grasp pose")
    return arm_tag, [
        Action(arm_tag, "move", target_pose=list(pre_pose)),
        # No partial-pose constraint: the authored Contact owns the full 6D
        # terminal orientation for the straight Pre-Grasp -> Grasp segment.
        Action(arm_tag, "move", target_pose=list(grasp_pose)),
        Action(arm_tag, "close", target_gripper_pos=float(gripper_pos)),
    ]


def target_tcp_pose_from_gripper_pose(
    gripper_pose: Sequence[float],
    *,
    gripper_tcp_offset_m: float,
) -> list[float]:
    """Convert the planner's wrist/gripper target into the configured TCP pose."""

    value = np.asarray(gripper_pose, dtype=float)
    if value.shape != (7,) or not np.isfinite(value).all():
        raise GraspContractError("gripper target pose must contain finite xyz + quat")
    quaternion = value[3:]
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-9:
        raise GraspContractError("gripper target quaternion has zero norm")
    quaternion = quaternion / norm
    rotation = t3d.quaternions.quat2mat(quaternion)
    tcp_position = value[:3] + rotation @ np.asarray(
        [float(gripper_tcp_offset_m), 0.0, 0.0], dtype=float
    )
    return [*clean_vector(tcp_position), *clean_vector(quaternion)]


def pose_residual(
    actual_pose: Sequence[float], expected_pose: Sequence[float]
) -> dict[str, float]:
    """Measure translation and shortest quaternion-angle residual."""

    actual = np.asarray(actual_pose, dtype=float)
    expected = np.asarray(expected_pose, dtype=float)
    if actual.shape != (7,) or expected.shape != (7,):
        raise GraspContractError("pose residual expects xyz + quat_wxyz")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise GraspContractError("pose residual inputs must be finite")
    actual_q = actual[3:] / max(float(np.linalg.norm(actual[3:])), 1e-12)
    expected_q = expected[3:] / max(float(np.linalg.norm(expected[3:])), 1e-12)
    dot = float(np.clip(abs(np.dot(actual_q, expected_q)), -1.0, 1.0))
    return {
        "position_m": round(float(np.linalg.norm(actual[:3] - expected[:3])), 12),
        "orientation_deg": round(float(np.degrees(2.0 * np.arccos(dot))), 12),
    }


def _actor_pose_vector(actor: Any) -> np.ndarray:
    pose = actor.get_pose()
    return np.asarray([*pose.p, *pose.q], dtype=float)


def _actual_tcp_pose(task: Any, arm: str) -> np.ndarray:
    method = getattr(
        task.robot,
        "get_left_tcp_pose" if str(arm) == "left" else "get_right_tcp_pose",
        None,
    )
    if not callable(method):
        raise GraspContractError(f"{arm} arm does not expose a configured TCP pose")
    value = np.asarray(method(), dtype=float)
    if value.shape != (7,) or not np.isfinite(value).all():
        raise GraspContractError(f"{arm} TCP pose must contain finite xyz + quat")
    return value


def _active_scene_contacts(task: Any) -> list[dict[str, Any]]:
    getter = getattr(task.scene, "get_contacts", None)
    if not callable(getter):
        return []
    rows = []
    for contact in getter():
        bodies = list(getattr(contact, "bodies", []) or [])
        if len(bodies) != 2:
            continue
        names = [
            str(getattr(getattr(body, "entity", None), "name", "unknown"))
            for body in bodies
        ]
        impulse_norms = [
            float(
                np.linalg.norm(
                    np.asarray(
                        getattr(point, "impulse", [0.0, 0.0, 0.0]),
                        dtype=float,
                    )
                )
            )
            for point in list(getattr(contact, "points", []) or [])
        ]
        maximum_impulse = max(impulse_norms, default=0.0)
        if maximum_impulse <= 1e-8:
            continue
        rows.append(
            {
                "entity_a": names[0],
                "entity_b": names[1],
                "point_count": len(impulse_norms),
                "maximum_impulse_norm": round(maximum_impulse, 12),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            str(row["entity_a"]),
            str(row["entity_b"]),
            -float(row["maximum_impulse_norm"]),
        ),
    )


def _terminal_stage_check(
    task: Any,
    actor: Any,
    *,
    arm: str,
    stage: str,
    expected_tcp_pose: Sequence[float],
    actor_pose_before: Sequence[float],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    position_tolerance = float(policy["position_tolerance_m"])
    orientation_tolerance = float(policy["orientation_tolerance_deg"])
    actor_translation_limit = float(
        policy["maximum_actor_translation_before_close_m"]
    )
    actor_rotation_limit = float(policy["maximum_actor_rotation_before_close_deg"])
    settle_steps = max(0, int(policy["settle_steps"]))
    actor_initial = np.asarray(actor_pose_before, dtype=float)
    last: dict[str, Any] = {}
    for step_index in range(settle_steps + 1):
        actual_tcp = _actual_tcp_pose(task, arm)
        actor_live = _actor_pose_vector(actor)
        terminal = pose_residual(actual_tcp, expected_tcp_pose)
        actor_drift = pose_residual(actor_live, actor_initial)
        last = {
            "stage": str(stage),
            "settle_step_count": int(step_index),
            "expected_tcp_pose": clean_vector(expected_tcp_pose),
            "actual_tcp_pose": clean_vector(actual_tcp),
            "terminal_residual": terminal,
            "actor_pose_before": clean_vector(actor_initial),
            "actor_pose_live": clean_vector(actor_live),
            "actor_drift": actor_drift,
            "active_scene_contacts": _active_scene_contacts(task),
        }
        if (
            actor_drift["position_m"] > actor_translation_limit
            or actor_drift["orientation_deg"] > actor_rotation_limit
        ):
            return {
                **last,
                "status": "failed",
                "failure_reason": "actor_moved_before_close",
            }
        if (
            terminal["position_m"] <= position_tolerance
            and terminal["orientation_deg"] <= orientation_tolerance
        ):
            return {**last, "status": "passed", "failure_reason": None}
        if step_index < settle_steps:
            task.scene.step()
    return {
        **last,
        "status": "failed",
        "failure_reason": "terminal_tcp_residual_exceeded",
    }


def exact_grasp_stage_check(
    task: Any,
    actor: Any,
    candidate: Mapping[str, Any],
    *,
    arm: str,
    stage: str,
    actor_pose_before: Sequence[float],
) -> dict[str, Any]:
    """Apply the ordinary exact-grasp terminal gate to one bimanual arm."""

    pose_key = "pre_grasp_pose" if stage == "high_alignment" else "grasp_pose"
    target = candidate.get(pose_key)
    if not isinstance(target, Sequence) or len(target) != 7:
        raise GraspContractError(f"candidate lacks {pose_key}")
    tcp_offset = float(
        candidate.get(
            "gripper_tcp_offset_m",
            EXACT_GRASP_POLICY["gripper_tcp_offset_m"],
        )
    )
    return _terminal_stage_check(
        task,
        actor,
        arm=str(arm),
        stage=str(stage),
        expected_tcp_pose=target_tcp_pose_from_gripper_pose(
            target, gripper_tcp_offset_m=tcp_offset
        ),
        actor_pose_before=actor_pose_before,
        policy=EXACT_GRASP_POLICY["terminal_execution"],
    )


def _real_arm_joint_positions(
    task: Any, arm: str, joint_count: int
) -> np.ndarray:
    getter = getattr(
        task.robot,
        (
            "get_left_arm_real_jointState"
            if str(arm) == "left"
            else "get_right_arm_real_jointState"
        ),
        None,
    )
    if not callable(getter):
        raise GraspContractError(
            f"{arm} arm does not expose measured joint positions"
        )
    values = np.asarray(getter(), dtype=float).reshape(-1)
    if len(values) < joint_count or not np.isfinite(values[:joint_count]).all():
        raise GraspContractError(
            f"{arm} measured joint state is shorter than the compiled path"
        )
    return values[:joint_count]


def _collision_world_hash(world: Mapping[str, Any], arm: str) -> str | None:
    arm_world = world.get(str(arm))
    if isinstance(arm_world, Mapping) and arm_world.get("world_hash") is not None:
        return str(arm_world["world_hash"])
    if world.get("world_hash") is not None:
        return str(world["world_hash"])
    return None


def _execute_compiled_arm_path(
    task: Any,
    *,
    arm: str,
    path: Sequence[Sequence[float]],
    velocity: Sequence[Sequence[float]] | None,
    stage: str,
    expected_trajectory_sha256: str | None,
    start_tolerance_rad: float | None = None,
) -> dict[str, Any]:
    """Execute one compiled path without invoking the planner again.

    CuRobo and the simulator share a 250 Hz timestep.  The fast path therefore
    advances one compiled waypoint per physics step without inserting pauses
    that change the planned time parameterization.  In-flight error is logged
    and guarded only against catastrophic runaway; every terminal joint must
    still converge to the strict terminal band before TCP review begins.
    """

    positions = np.asarray(path, dtype=float)
    if (
        positions.ndim != 2
        or not len(positions)
        or not np.isfinite(positions).all()
    ):
        raise GraspContractError("compiled arm path must be a finite non-empty matrix")
    velocities = np.asarray(
        velocity if velocity is not None else [], dtype=float
    )
    if velocities.shape != positions.shape or not np.isfinite(velocities).all():
        velocities = np.zeros_like(positions)

    trajectory_sha256 = trajectory_summary(positions)["trajectory_sha256"]
    if (
        expected_trajectory_sha256
        and trajectory_sha256 != str(expected_trajectory_sha256)
    ):
        return {
            "status": "failed",
            "failure_reason": "compiled_trajectory_hash_mismatch",
            "stage": str(stage),
            "trajectory_sha256": trajectory_sha256,
            "expected_trajectory_sha256": str(expected_trajectory_sha256),
            "planned_waypoint_count": int(len(positions)),
            "executed_waypoint_count": 0,
            "physics_step_count": 0,
            "tracking_hold_step_count": 0,
        }

    policy = dict(COMPILED_TRAJECTORY_EXECUTION_POLICY)
    start_tolerance = float(
        policy["compiled_start_tolerance_rad"]
        if start_tolerance_rad is None
        else start_tolerance_rad
    )
    maximum_inflight_error = float(
        policy["maximum_inflight_joint_error_rad"]
    )
    terminal_tolerance = float(policy["terminal_joint_error_rad"])
    maximum_terminal_holds = max(
        0, int(policy["maximum_terminal_hold_steps"])
    )
    measured = _real_arm_joint_positions(task, str(arm), positions.shape[1])
    start_error = float(np.max(np.abs(measured - positions[0])))
    if start_error > start_tolerance:
        return {
            "status": "failed",
            "failure_reason": "compiled_start_state_mismatch",
            "stage": str(stage),
            "trajectory_sha256": trajectory_sha256,
            "planned_waypoint_count": int(len(positions)),
            "executed_waypoint_count": 0,
            "physics_step_count": 0,
            "tracking_hold_step_count": 0,
            "start_joint_error_rad": start_error,
            "start_joint_tolerance_rad": start_tolerance,
        }

    physics_steps = 0
    hold_steps = 0
    maximum_error = start_error
    latest_measured = measured.copy()

    def step_target(target: np.ndarray, target_velocity: np.ndarray) -> float:
        nonlocal physics_steps, maximum_error, latest_measured
        task.robot.set_arm_joints(target, target_velocity, str(arm))
        task.scene.step()
        physics_steps += 1
        save_freq = getattr(task, "save_freq", None)
        if (
            isinstance(save_freq, int)
            and save_freq > 0
            and physics_steps % save_freq == 0
        ):
            updater = getattr(task, "_update_render", None)
            recorder = getattr(task, "_take_picture", None)
            if callable(updater):
                updater()
            if callable(recorder):
                recorder()
        live = _real_arm_joint_positions(task, str(arm), positions.shape[1])
        latest_measured = live.copy()
        error = float(np.max(np.abs(live - target)))
        maximum_error = max(maximum_error, error)
        return error

    setattr(task, "_rollout_video_stage", str(stage))
    for waypoint_index, (target, target_velocity) in enumerate(
        zip(positions, velocities)
    ):
        error = step_target(target, target_velocity)
        is_terminal = waypoint_index == len(positions) - 1
        if not is_terminal and error <= maximum_inflight_error:
            continue
        if is_terminal:
            local_holds = 0
            while error > terminal_tolerance and local_holds < maximum_terminal_holds:
                error = step_target(target, np.zeros_like(target_velocity))
                local_holds += 1
                hold_steps += 1
        if not is_terminal and error > maximum_inflight_error:
            joint_error = latest_measured - target
            maximum_error_joint_index = int(np.argmax(np.abs(joint_error)))
            return {
                "status": "failed",
                "failure_reason": "trajectory_tracking_failed",
                "tracking_policy_version": str(policy["version"]),
                "stage": str(stage),
                "trajectory_sha256": trajectory_sha256,
                "planned_waypoint_count": int(len(positions)),
                "executed_waypoint_count": int(waypoint_index + 1),
                "physics_step_count": int(physics_steps),
                "tracking_hold_step_count": int(hold_steps),
                "failed_waypoint_index": int(waypoint_index),
                "terminal_waypoint": bool(is_terminal),
                "joint_error_rad": error,
                "joint_error_by_index_rad": clean_vector(joint_error),
                "maximum_error_joint_index": maximum_error_joint_index,
                "target_joint_positions_rad": clean_vector(target),
                "measured_joint_positions_rad": clean_vector(latest_measured),
                "target_joint_velocity_rad_s": clean_vector(target_velocity),
                "previous_target_step_rad": clean_vector(
                    np.zeros_like(target)
                    if waypoint_index == 0
                    else target - positions[waypoint_index - 1]
                ),
                "joint_tolerance_rad": (
                    maximum_inflight_error
                ),
                "maximum_joint_error_rad": maximum_error,
            }
    terminal_converged = bool(error <= terminal_tolerance)
    return {
        "status": "passed" if terminal_converged else "failed",
        "failure_reason": (
            None
            if terminal_converged
            else "trajectory_terminal_tracking_failed"
        ),
        "tracking_policy_version": str(policy["version"]),
        "stage": str(stage),
        "trajectory_sha256": trajectory_sha256,
        "planned_waypoint_count": int(len(positions)),
        "executed_waypoint_count": int(len(positions)),
        "physics_step_count": int(physics_steps),
        "tracking_hold_step_count": int(hold_steps),
        "start_joint_error_rad": start_error,
        "terminal_joint_error_rad": error,
        "terminal_joint_tolerance_rad": terminal_tolerance,
        "terminal_joint_converged": terminal_converged,
        "maximum_joint_error_rad": maximum_error,
    }


def execute_compiled_bimanual_paths(
    task: Any,
    *,
    left_path: Sequence[Sequence[float]],
    right_path: Sequence[Sequence[float]],
    left_velocity: Sequence[Sequence[float]] | None,
    right_velocity: Sequence[Sequence[float]] | None,
    stage: str,
) -> dict[str, Any]:
    """Replay two certified arm paths on the same physics clock.

    Each arm advances according to normalized path progress, so a shorter
    path cannot finish and wait while the other arm is still approaching the
    object.  This is the shared bimanual primitive used by both M4 and M15;
    it never invokes either planner during execution.
    """

    paths = {
        "left": np.asarray(left_path, dtype=float),
        "right": np.asarray(right_path, dtype=float),
    }
    velocities = {
        "left": np.asarray(
            [] if left_velocity is None else left_velocity, dtype=float
        ),
        "right": np.asarray(
            [] if right_velocity is None else right_velocity, dtype=float
        ),
    }
    for arm in ("left", "right"):
        path = paths[arm]
        if path.ndim != 2 or not len(path) or not np.isfinite(path).all():
            raise GraspContractError(
                f"compiled {arm} bimanual path must be finite and non-empty"
            )
        if velocities[arm].shape != path.shape or not np.isfinite(
            velocities[arm]
        ).all():
            velocities[arm] = np.zeros_like(path)

    policy = dict(COMPILED_TRAJECTORY_EXECUTION_POLICY)
    start_tolerance = float(policy["compiled_start_tolerance_rad"])
    terminal_tolerance = float(policy["terminal_joint_error_rad"])
    maximum_inflight_error = float(
        policy["maximum_inflight_joint_error_rad"]
    )
    maximum_terminal_holds = max(
        0, int(policy["maximum_terminal_hold_steps"])
    )
    start_errors = {
        arm: float(
            np.max(
                np.abs(
                    _real_arm_joint_positions(
                        task, arm, paths[arm].shape[1]
                    )
                    - paths[arm][0]
                )
            )
        )
        for arm in ("left", "right")
    }
    if any(value > start_tolerance for value in start_errors.values()):
        return {
            "status": "failed",
            "failure_reason": "compiled_bimanual_start_state_mismatch",
            "stage": str(stage),
            "start_joint_error_rad": start_errors,
            "start_joint_tolerance_rad": start_tolerance,
            "physics_step_count": 0,
        }

    indices = {"left": 0, "right": 0}
    maximum_errors = dict(start_errors)
    physics_steps = 0
    setattr(task, "_rollout_video_stage", str(stage))
    while any(indices[arm] < len(paths[arm]) for arm in ("left", "right")):
        for arm in ("left", "right"):
            index = min(indices[arm], len(paths[arm]) - 1)
            task.robot.set_arm_joints(
                paths[arm][index], velocities[arm][index], arm
            )
        task.scene.step()
        physics_steps += 1
        for arm in ("left", "right"):
            index = min(indices[arm], len(paths[arm]) - 1)
            measured = _real_arm_joint_positions(
                task, arm, paths[arm].shape[1]
            )
            error = float(np.max(np.abs(measured - paths[arm][index])))
            maximum_errors[arm] = max(maximum_errors[arm], error)
            if (
                indices[arm] < len(paths[arm])
                and (
                    indices[arm] == 0
                    or indices[arm] / len(paths[arm])
                    <= min(
                        indices[other] / len(paths[other])
                        for other in ("left", "right")
                        if other != arm
                    )
                )
            ):
                indices[arm] += 1
            if indices[arm] < len(paths[arm]) and error > maximum_inflight_error:
                return {
                    "status": "failed",
                    "failure_reason": "bimanual_trajectory_tracking_failed",
                    "failed_arm": arm,
                    "stage": str(stage),
                    "physics_step_count": physics_steps,
                    "joint_error_rad": error,
                    "joint_tolerance_rad": maximum_inflight_error,
                    "maximum_joint_error_rad": maximum_errors,
                }

        save_freq = getattr(task, "save_freq", None)
        if isinstance(save_freq, int) and save_freq > 0 and physics_steps % save_freq == 0:
            updater = getattr(task, "_update_render", None)
            recorder = getattr(task, "_take_picture", None)
            if callable(updater):
                updater()
            if callable(recorder):
                recorder()

    terminal_errors: dict[str, float] = {}
    hold_steps = 0
    while True:
        terminal_errors = {}
        for arm in ("left", "right"):
            target = paths[arm][-1]
            task.robot.set_arm_joints(
                target, np.zeros_like(target), arm
            )
            terminal_errors[arm] = float(
                np.max(
                    np.abs(
                        _real_arm_joint_positions(
                            task, arm, len(target)
                        )
                        - target
                    )
                )
            )
        if all(
            value <= terminal_tolerance
            for value in terminal_errors.values()
        ) or hold_steps >= maximum_terminal_holds:
            break
        task.scene.step()
        physics_steps += 1
        hold_steps += 1

    passed = all(
        value <= terminal_tolerance for value in terminal_errors.values()
    )
    return {
        "status": "passed" if passed else "failed",
        "failure_reason": (
            None if passed else "bimanual_trajectory_terminal_tracking_failed"
        ),
        "tracking_policy_version": str(policy["version"]),
        "stage": str(stage),
        "physics_step_count": physics_steps,
        "tracking_hold_step_count": hold_steps,
        "terminal_joint_error_rad": terminal_errors,
        "terminal_joint_tolerance_rad": terminal_tolerance,
        "maximum_joint_error_rad": maximum_errors,
        "planned_waypoint_count": {
            arm: int(len(paths[arm])) for arm in ("left", "right")
        },
    }


def _measured_terminal_collision_preflight(
    task: Any,
    actor: Any,
    *,
    arm: str,
    joint_count: int,
    allow_target_contact: bool = True,
) -> dict[str, Any] | None:
    """Exact-mesh-check the measured endpoint, not only its commanded qpos."""

    validator = getattr(
        task.robot, "validate_arm_path_physical_collisions", None
    )
    if not callable(validator):
        return None
    measured = _real_arm_joint_positions(task, arm, joint_count)
    try:
        result = dict(
            validator(
                arm,
                measured.reshape(1, -1),
                sample_stride=1,
                excluded_entities=(actor,) if allow_target_contact else (),
            )
        )
    except TimeoutError:
        raise
    except Exception as exc:
        raise PhysicalCollisionValidationError(
            "exact-mesh measured replay endpoint check failed for "
            f"{arm}: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(result.get("collision_free"), bool):
        raise PhysicalCollisionValidationError(
            "exact-mesh measured replay endpoint check returned no boolean "
            f"collision result for {arm}: {result!r}"
        )
    return result


def execute_exact_grasp_candidate(
    task: Any,
    actor: Any,
    candidate: Mapping[str, Any],
    *,
    arm: str,
    gripper_pos: float = 0.0,
    terminal_policy: Mapping[str, Any] | None = None,
    execute_alignment_motion: bool = True,
) -> dict[str, Any]:
    """Execute high alignment, verify it, then approach and close.

    Planning reachability is not execution success.  This adapter prevents a
    low or lagging wrist from entering the target while it is still rotating:
    the high alignment pose must physically converge, without moving the
    target actor, before the fixed-orientation approach is allowed to start.
    """

    policy = {
        **dict(EXACT_GRASP_POLICY["terminal_execution"]),
        **dict(terminal_policy or {}),
    }
    tcp_offset = float(
        candidate.get(
            "gripper_tcp_offset_m", EXACT_GRASP_POLICY["gripper_tcp_offset_m"]
        )
    )
    pre_pose = candidate.get("pre_grasp_pose")
    grasp_pose = candidate.get("grasp_pose")
    if not isinstance(pre_pose, Sequence) or not isinstance(grasp_pose, Sequence):
        raise GraspContractError("candidate lacks alignment/grasp poses")
    orientation_delta = pose_residual(
        [0.0, 0.0, 0.0, *pre_pose[3:]],
        [0.0, 0.0, 0.0, *grasp_pose[3:]],
    )["orientation_deg"]
    if orientation_delta > 1e-5:
        raise GraspContractError(
            "alignment and grasp must preserve the authored orientation"
        )
    expected_alignment_tcp = target_tcp_pose_from_gripper_pose(
        pre_pose, gripper_tcp_offset_m=tcp_offset
    )
    expected_grasp_tcp = target_tcp_pose_from_gripper_pose(
        grasp_pose, gripper_tcp_offset_m=tcp_offset
    )
    actor_pose_before = _actor_pose_vector(actor)
    arm_tag, sequence = exact_grasp_actions(
        candidate, arm=str(arm), gripper_pos=gripper_pos
    )
    stages: list[dict[str, Any]] = []
    pre_grasp_obstacles = candidate.get("_pre_grasp_obstacles") or []
    grasp_obstacles = candidate.get("_grasp_obstacles") or []
    trajectories = candidate.get("_trajectories") or {}
    compiled_execution = bool(
        isinstance(trajectories, Mapping)
        and np.asarray(trajectories.get("pre_grasp", [])).ndim == 2
        and len(np.asarray(trajectories.get("pre_grasp", [])))
        and np.asarray(trajectories.get("grasp", [])).ndim == 2
        and len(np.asarray(trajectories.get("grasp", [])))
    )
    trajectory_tracking: dict[str, Any] = {}
    path_records = candidate.get("paths") or {}

    def expected_world_hash(segment: str) -> str | None:
        record = path_records.get(f"{segment}_planner") or {}
        value = record.get("world_hash")
        return None if value is None else str(value)

    def verify_world(
        world: Mapping[str, Any], segment: str
    ) -> dict[str, Any] | None:
        expected = expected_world_hash(segment)
        actual = _collision_world_hash(world, str(arm))
        if expected is not None and actual is not None and actual != expected:
            return {
                "status": "failed",
                "failure_reason": "compiled_collision_world_mismatch",
                "segment": segment,
                "expected_world_hash": expected,
                "actual_world_hash": actual,
            }
        return None

    if (compiled_execution or pre_grasp_obstacles) and execute_alignment_motion:
        try:
            pre_grasp_world = refresh_collision_world(
                task,
                target_actor=actor,
                extra_obstacles=pre_grasp_obstacles,
            )
        except CollisionWorldUpdateError as exc:
            return {
                "status": "failed",
                "failure_reason": "pre_grasp_collision_world_update_failed",
                "collision_error": str(exc),
                "executed_alignment": False,
                "executed_approach": False,
                "closed": False,
                "stages": stages,
                "compiled_trajectory_execution": compiled_execution,
                "trajectory_tracking": trajectory_tracking,
            }
        world_failure = verify_world(pre_grasp_world, "pre_grasp")
        if world_failure is not None:
            return {
                **world_failure,
                "executed_alignment": False,
                "executed_approach": False,
                "closed": False,
                "stages": stages,
                "compiled_trajectory_execution": compiled_execution,
                "trajectory_tracking": trajectory_tracking,
            }

    if execute_alignment_motion:
        if compiled_execution:
            alignment_tracking = _execute_compiled_arm_path(
                task,
                arm=str(arm),
                path=trajectories["pre_grasp"],
                velocity=trajectories.get("pre_grasp_velocity"),
                stage="grasp_high_alignment",
                expected_trajectory_sha256=(
                    (path_records.get("pre_grasp") or {}).get(
                        "trajectory_sha256"
                    )
                ),
            )
            trajectory_tracking["high_alignment"] = alignment_tracking
            alignment_motion_succeeded = (
                alignment_tracking.get("status") == "passed"
            )
            if alignment_motion_succeeded:
                endpoint_collision = _measured_terminal_collision_preflight(
                    task,
                    actor,
                    arm=str(arm),
                    joint_count=len(
                        np.asarray(trajectories["pre_grasp"])[-1]
                    ),
                    allow_target_contact=not getattr(
                        task, "_m15_strict_pregrasp_collision", False
                    ),
                )
                alignment_tracking[
                    "measured_terminal_physical_collision_preflight"
                ] = endpoint_collision
                if (
                    endpoint_collision is not None
                    and not endpoint_collision["collision_free"]
                ):
                    alignment_tracking["status"] = "failed"
                    alignment_tracking["failure_reason"] = (
                        "physical_mesh_collision_during_replay"
                    )
                    alignment_motion_succeeded = False
        else:
            setattr(task, "_rollout_video_stage", "grasp_high_alignment")
            task.move((arm_tag, [sequence[0]]))
            alignment_motion_succeeded = bool(
                getattr(task, "plan_success", False)
            )
        if not alignment_motion_succeeded:
            return {
                "status": "failed",
                "failure_reason": (
                    str(alignment_tracking.get("failure_reason"))
                    if compiled_execution
                    else "alignment_execution_failed"
                ),
                "executed_alignment": False,
                "alignment_motion_executed": True,
                "executed_approach": False,
                "closed": False,
                "stages": stages,
                "compiled_trajectory_execution": compiled_execution,
                "trajectory_tracking": trajectory_tracking,
            }
    alignment = _terminal_stage_check(
        task,
        actor,
        arm=str(arm),
        stage="high_alignment",
        expected_tcp_pose=expected_alignment_tcp,
        actor_pose_before=actor_pose_before,
        policy=policy,
    )
    stages.append(alignment)
    if alignment["status"] != "passed":
        return {
            "status": "failed",
            "failure_reason": alignment["failure_reason"],
            "executed_alignment": True,
            "alignment_motion_executed": bool(execute_alignment_motion),
            "executed_approach": False,
            "closed": False,
            "stages": stages,
            "compiled_trajectory_execution": compiled_execution,
            "trajectory_tracking": trajectory_tracking,
        }

    if compiled_execution or pre_grasp_obstacles:
        try:
            grasp_world = refresh_collision_world(
                task,
                target_actor=actor,
                extra_obstacles=grasp_obstacles,
            )
        except CollisionWorldUpdateError as exc:
            return {
                "status": "failed",
                "failure_reason": "grasp_collision_world_update_failed",
                "collision_error": str(exc),
                "executed_alignment": True,
                "alignment_motion_executed": bool(execute_alignment_motion),
                "executed_approach": False,
                "closed": False,
                "stages": stages,
                "compiled_trajectory_execution": compiled_execution,
                "trajectory_tracking": trajectory_tracking,
            }
        world_failure = verify_world(grasp_world, "grasp")
        if (world_failure is not None and compiled_execution
                and getattr(task, "_m15_strict_pregrasp_collision", False)):
            try:
                def refreshed_hash():
                    world = refresh_collision_world(
                        task, target_actor=actor, extra_obstacles=grasp_obstacles,
                    )
                    return _collision_world_hash(world, str(arm))

                revalidation = revalidate_live_approach(
                    task, actor, candidate, arm=str(arm),
                    original_world_hash=str(world_failure["expected_world_hash"]),
                    live_world_hash=str(world_failure["actual_world_hash"]),
                    refresh_world=refreshed_hash,
                )
            except TimeoutError:
                raise
            except Exception as exc:
                raise PhysicalCollisionValidationError(
                    "live Approach collision revalidation failed: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            trajectory_tracking["grasp_world_revalidation"] = revalidation
            if revalidation["passed"]:
                world_failure = None
        if world_failure is not None:
            return {
                **world_failure,
                "executed_alignment": True,
                "alignment_motion_executed": bool(execute_alignment_motion),
                "executed_approach": False,
                "closed": False,
                "stages": stages,
                "compiled_trajectory_execution": compiled_execution,
                "trajectory_tracking": trajectory_tracking,
            }

    if compiled_execution:
        approach_tracking = _execute_compiled_arm_path(
            task,
            arm=str(arm),
            path=trajectories["grasp"],
            velocity=trajectories.get("grasp_velocity"),
            stage="grasp_fixed_orientation_approach",
            expected_trajectory_sha256=(
                (path_records.get("grasp") or {}).get("trajectory_sha256")
            ),
            start_tolerance_rad=float(
                COMPILED_TRAJECTORY_EXECUTION_POLICY[
                    "chained_start_tolerance_rad"
                ]
            ),
        )
        trajectory_tracking["fixed_orientation_approach"] = approach_tracking
        approach_motion_succeeded = approach_tracking.get("status") == "passed"
        if approach_motion_succeeded:
            endpoint_collision = _measured_terminal_collision_preflight(
                task,
                actor,
                arm=str(arm),
                joint_count=len(np.asarray(trajectories["grasp"])[-1]),
            )
            approach_tracking[
                "measured_terminal_physical_collision_preflight"
            ] = endpoint_collision
            if (
                endpoint_collision is not None
                and not endpoint_collision["collision_free"]
            ):
                approach_tracking["status"] = "failed"
                approach_tracking["failure_reason"] = (
                    "physical_mesh_collision_during_replay"
                )
                approach_motion_succeeded = False
    else:
        setattr(task, "_rollout_video_stage", "grasp_fixed_orientation_approach")
        task.move((arm_tag, [sequence[1]]))
        approach_motion_succeeded = bool(getattr(task, "plan_success", False))
    if not approach_motion_succeeded:
        return {
            "status": "failed",
            "failure_reason": (
                str(approach_tracking.get("failure_reason"))
                if compiled_execution
                else "approach_execution_failed"
            ),
            "executed_alignment": True,
            "alignment_motion_executed": bool(execute_alignment_motion),
            "executed_approach": False,
            "closed": False,
            "stages": stages,
            "compiled_trajectory_execution": compiled_execution,
            "trajectory_tracking": trajectory_tracking,
        }
    approach = _terminal_stage_check(
        task,
        actor,
        arm=str(arm),
        stage="fixed_orientation_approach",
        expected_tcp_pose=expected_grasp_tcp,
        actor_pose_before=actor_pose_before,
        policy=policy,
    )
    stages.append(approach)
    if approach["status"] != "passed":
        return {
            "status": "failed",
            "failure_reason": approach["failure_reason"],
            "executed_alignment": True,
            "alignment_motion_executed": bool(execute_alignment_motion),
            "executed_approach": True,
            "closed": False,
            "stages": stages,
            "compiled_trajectory_execution": compiled_execution,
            "trajectory_tracking": trajectory_tracking,
        }

    setattr(task, "_rollout_video_stage", "grasp_close")
    task.move((arm_tag, [sequence[2]]))
    closed = bool(getattr(task, "plan_success", False))
    return {
        "status": "closed" if closed else "failed",
        "failure_reason": None if closed else "close_execution_failed",
        "executed_alignment": True,
        "alignment_motion_executed": bool(execute_alignment_motion),
        "executed_approach": True,
        "closed": closed,
        "stages": stages,
        "compiled_trajectory_execution": compiled_execution,
        "trajectory_tracking": trajectory_tracking,
    }


def _raw_arm_gripper_actor_contacts(
    task: Any, actor: Any, arm: str
) -> list[dict[str, Any]]:
    """Observe RoboTwin-style contact with any link of one arm's gripper."""

    prefix = "fl_" if str(arm) == "left" else "fr_"
    gripper_names = {
        str(name)
        for name in getattr(task.robot, "gripper_name", [])
        if str(name).startswith(prefix)
    }
    actor_entity = getattr(actor, "actor", actor)
    rows: list[dict[str, Any]] = []
    for contact in task.scene.get_contacts():
        bodies = list(getattr(contact, "bodies", []) or [])
        if len(bodies) != 2:
            continue
        entities = [getattr(body, "entity", None) for body in bodies]
        if actor_entity not in entities:
            continue
        other_index = 1 if entities[0] is actor_entity else 0
        other = entities[other_index]
        other_name = str(getattr(other, "name", ""))
        if other_name not in gripper_names:
            continue
        rows.append(
            {
                "link": other_name,
                "point_count": len(getattr(contact, "points", []) or []),
            }
        )
    return rows


def execute_contact_limited_handover_candidate(
    task: Any,
    actor: Any,
    candidate: Mapping[str, Any],
    *,
    subject: str,
    arm: str,
    gripper_pos: float = 0.0,
    terminal_policy: Mapping[str, Any] | None = None,
    waypoint_stride: int = 1,
    stop_on_contact: bool = True,
) -> dict[str, Any]:
    """Validate a staged target, execute its one approach, and close once.

    The complete joint path is planned before execution.  During execution the
    With ``stop_on_contact`` the first live gripper/subject contact is its
    physical terminal event.  RoboTwin-style handover instead executes the
    complete Contact path before the single close action.
    """

    policy = {
        **dict(EXACT_GRASP_POLICY["terminal_execution"]),
        **dict(terminal_policy or {}),
    }
    tcp_offset = float(
        candidate.get(
            "gripper_tcp_offset_m", EXACT_GRASP_POLICY["gripper_tcp_offset_m"]
        )
    )
    pre_pose = candidate.get("pre_grasp_pose")
    if not isinstance(pre_pose, Sequence) or len(pre_pose) != 7:
        raise GraspContractError("candidate lacks an exact Pre-Grasp pose")
    expected_alignment_tcp = target_tcp_pose_from_gripper_pose(
        pre_pose, gripper_tcp_offset_m=tcp_offset
    )
    actor_pose_before = _actor_pose_vector(actor)
    alignment = _terminal_stage_check(
        task,
        actor,
        arm=str(arm),
        stage="high_alignment",
        expected_tcp_pose=expected_alignment_tcp,
        actor_pose_before=actor_pose_before,
        policy=policy,
    )
    stages = [alignment]
    if alignment["status"] != "passed":
        return {
            "status": "failed",
            "failure_reason": alignment["failure_reason"],
            "executed_alignment": True,
            "alignment_motion_executed": False,
            "executed_approach": False,
            "closed": False,
            "stages": stages,
        }

    try:
        refresh_collision_world(task, target_actor=actor)
    except CollisionWorldUpdateError as exc:
        return {
            "status": "failed",
            "failure_reason": "grasp_collision_world_update_failed",
            "collision_error": str(exc),
            "executed_alignment": True,
            "alignment_motion_executed": False,
            "executed_approach": False,
            "closed": False,
            "stages": stages,
        }

    trajectories = candidate.get("_trajectories") or {}
    path = np.asarray(trajectories.get("grasp", []), dtype=float)
    velocity = np.asarray(
        trajectories.get("grasp_velocity", []), dtype=float
    )
    if path.ndim != 2 or not len(path):
        raise GraspContractError("candidate lacks a planned Grasp trajectory")
    if velocity.shape != path.shape:
        velocity = np.zeros_like(path)
    stride = max(1, int(waypoint_stride))
    execution_indices = list(range(0, len(path), stride))
    if execution_indices[-1] != len(path) - 1:
        execution_indices.append(len(path) - 1)

    actor_translation_limit = float(
        policy["maximum_actor_translation_before_close_m"]
    )
    actor_rotation_limit = float(
        policy["maximum_actor_rotation_before_close_deg"]
    )
    gripper_endpoint = f"gripper:{arm}"
    contact_index = None
    matched_contact_count = 0
    matched_contact_regions: list[str] = []
    raw_contact_links: list[str] = []
    actor_drift = pose_residual(actor_pose_before, actor_pose_before)
    executed_waypoint_count = 0
    setattr(task, "_rollout_video_stage", "handover_contact_limited_approach")
    for path_index in execution_indices:
        qpos = path[path_index]
        qvel = velocity[path_index]
        task.robot.set_arm_joints(qpos, qvel, str(arm))
        task.scene.step()
        executed_waypoint_count += 1
        save_freq = getattr(task, "save_freq", None)
        if (
            isinstance(save_freq, int)
            and save_freq > 0
            and executed_waypoint_count % save_freq == 0
        ):
            task._update_render()
            task._take_picture()
        actor_live = _actor_pose_vector(actor)
        actor_drift = pose_residual(actor_live, actor_pose_before)
        contacts = task._unified_handover_active_contacts()
        matched = [
            row
            for row in contacts
            if str(row.get("arm")) == str(arm)
            and {
                str(row.get("entity_a")),
                str(row.get("entity_b")),
            }
            == {gripper_endpoint, str(subject)}
        ]
        raw_contacts = _raw_arm_gripper_actor_contacts(
            task, actor, str(arm)
        )
        if matched or raw_contacts:
            matched_contact_count = len(matched) + len(raw_contacts)
            matched_contact_regions = sorted(
                {
                    *{
                        str(row.get("region") or "unknown")
                        for row in matched
                    },
                    *({"raw_gripper"} if raw_contacts else set()),
                }
            )
            raw_contact_links = sorted(
                {str(row["link"]) for row in raw_contacts}
            )
            if stop_on_contact:
                contact_index = path_index
                break
        if (
            actor_drift["position_m"] > actor_translation_limit
            or actor_drift["orientation_deg"] > actor_rotation_limit
        ):
            break

    approach = {
        "stage": "contact_limited_approach",
        "planned_waypoint_count": int(len(path)),
        "execution_waypoint_stride": stride,
        "executed_waypoint_count": int(executed_waypoint_count),
        "executed_fraction": float(
            1.0
            if contact_index is None
            else (contact_index + 1) / len(path)
        ),
        "contact_detected": contact_index is not None,
        "contact_observed_during_approach": bool(
            matched_contact_count or raw_contact_links
        ),
        "matched_subject_contact_count": int(matched_contact_count),
        "matched_subject_contact_regions": matched_contact_regions,
        "raw_gripper_contact_links": raw_contact_links,
        "actor_drift": actor_drift,
    }
    stages.append(approach)
    if contact_index is None and (
        actor_drift["position_m"] > actor_translation_limit
        or actor_drift["orientation_deg"] > actor_rotation_limit
    ):
        failure = "actor_moved_before_close"
        approach.update({"status": "failed", "failure_reason": failure})
        return {
            "status": "failed",
            "failure_reason": failure,
            "executed_alignment": True,
            "alignment_motion_executed": False,
            "executed_approach": True,
            "closed": False,
            "stages": stages,
        }
    approach.update(
        {
            "status": "passed",
            "failure_reason": None,
            "terminal_event": (
                "live_subject_contact"
                if contact_index is not None
                else "planned_endpoint_without_preclose_contact"
            ),
        }
    )

    arm_tag, sequence = exact_grasp_actions(
        candidate, arm=str(arm), gripper_pos=gripper_pos
    )
    setattr(task, "_rollout_video_stage", "grasp_close_on_contact")
    task.move((arm_tag, [sequence[2]]))
    closed = bool(getattr(task, "plan_success", False))
    return {
        "status": "closed" if closed else "failed",
        "failure_reason": None if closed else "close_execution_failed",
        "executed_alignment": True,
        "alignment_motion_executed": False,
        "executed_approach": True,
        "closed": closed,
        "stages": stages,
    }


def diagnostic_candidate(candidate: Mapping[str, Any] | None) -> dict[str, Any]:
    """Strip runtime-only trajectory arrays before M4/M15 JSON diagnostics."""

    if candidate is None:
        return {}
    return {
        str(key): deepcopy(value)
        for key, value in candidate.items()
        if not str(key).startswith("_")
    }


def strip_runtime_arrays(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): strip_runtime_arrays(item)
            for key, item in value.items()
            if not str(key).startswith("_")
        }
    if isinstance(value, list):
        return [strip_runtime_arrays(item) for item in value]
    if isinstance(value, tuple):
        return [strip_runtime_arrays(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return deepcopy(value)
