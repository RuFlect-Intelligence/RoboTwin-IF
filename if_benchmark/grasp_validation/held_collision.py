"""One held geometry and support-exit gate for IF Pick and Carry.

Approach still owns its unladen robot checks. From hypothetical closure onward,
M4/M12/M15 use the same complete pieces, frame, table and native sphere cover.
No simulator operation is permitted by the planning helpers in this module.
"""
from __future__ import annotations

from .carry_motion import verify_carry_motion_receipt

from contextlib import contextmanager
from copy import deepcopy
from typing import Mapping
import sys

import numpy as np
import transforms3d as t3d

from if_benchmark.schema import artifact_hash
from .convex_cover import live_collision_pieces
from .core import trajectory_summary
from .held_camera import verify_camera_path_proof
from .vertical_lift import verify_vertical_lift_receipt

HELD_COLLISION_POLICY = "if_pick_carry_complete_geometry_v3_vertical_exact_fixed_camera"
# Retain the existing native-mode identifier; its geometry now also serves Pick.
HELD_GEOMETRY_MODE = "m15_carry_conservative_v1"


def live_table_obstacles(task):
    table = task.table
    world = np.asarray(table.get_pose().to_transformation_matrix(), dtype=float)
    if world.shape != (4, 4) or not np.isfinite(world).all():
        raise RuntimeError("held_table_pose_invalid")
    boxes = []
    for component in table.get_components():
        get_shapes = getattr(component, "get_collision_shapes", None)
        if not callable(get_shapes):
            continue
        for shape in get_shapes():
            if not hasattr(shape, "half_size"):
                raise RuntimeError("held_table_shape_not_box")
            half = np.asarray(shape.half_size, dtype=float)
            local = np.asarray(shape.local_pose.to_transformation_matrix(), dtype=float)
            if (half.shape != (3,) or not np.isfinite(half).all() or np.any(half <= 0)
                    or local.shape != (4, 4) or not np.isfinite(local).all()):
                raise RuntimeError("held_table_shape_invalid")
            pose = world @ local
            boxes.append({"name": f"m15_table_shape_{len(boxes)}",
                          "dims": (2 * half).tolist(),
                          "pose": [*pose[:3, 3], *t3d.quaternions.mat2quat(pose[:3, :3])],
                          "source": "live_sapien_table_box_shape_v1"})
    if not boxes:
        raise RuntimeError("held_table_collision_geometry_missing")
    return boxes


def complete_held_descriptor(actor, *, name):
    from if_benchmark.reach_validation.contract import (
        held_collision_descriptor, ROUTE_EXECUTION_CONTRACT,
    )
    return {**held_collision_descriptor(actor, name=name),
            "collision_pieces": live_collision_pieces(actor),
            "dimension_envelope_m": float(ROUTE_EXECUTION_CONTRACT["held_collision"]["dimension_envelope_m"])}


def refresh_held_world(task, actor, arm, *, require_existing_binding=False):
    from if_benchmark.reach_validation.contract import route_scene_collision_obstacles
    tables = live_table_obstacles(task)
    mode = task.robot.set_collision_geometry_mode(arm, HELD_GEOMETRY_MODE, table_obstacles=tables)
    if require_existing_binding and mode.get("changed") is not False:
        raise RuntimeError("held_collision_binding_lost_at_handoff")
    entities = getattr(task, "entities", None)
    scene = getattr(task, "scene_spec", None)
    if isinstance(entities, Mapping) and isinstance(scene, Mapping):
        excluded = [key for key, value in entities.items() if value is actor]
        if len(excluded) != 1:
            raise RuntimeError("held_subject_identity_missing_or_ambiguous")
        obstacles = route_scene_collision_obstacles(
            scene, entities, excluded_entity_keys=excluded,
            complete_geometry=True, max_obstacles=64-len(tables),
        )
    elif getattr(task, "subject", None) is actor:
        # The M4 adapter explicitly owns a single-subject, clean-table scene.
        obstacles = []
    else:
        raise RuntimeError("held_scene_geometry_unavailable")
    world = task.robot.update_arm_world_obstacles(arm, obstacles)
    if not isinstance(world, Mapping) or world.get("updated") is not True or not (world.get(arm) or {}).get("world_hash"):
        raise RuntimeError("held_collision_world_hash_missing")
    return {**world, "obstacles": deepcopy(obstacles), "tables": tables}


def bind_held_collision(task, actor, arm, *, name, last_qpos=None):
    world = refresh_held_world(task, actor, arm)
    descriptor = complete_held_descriptor(actor, name=name)
    options = {} if last_qpos is None else {"last_qpos": last_qpos}
    try:
        attached = task.robot.attach_collision_object(arm, descriptor, **options)
        if (attached.get("attached") is not True or not attached.get("attachment_hash")
                or attached.get("collision_geometry_mode") != HELD_GEOMETRY_MODE
                or not attached.get("cover_version") or not attached.get("sphere_tensor_sha256")
                or not isinstance(attached.get("coverage_certificate"), Mapping)
                or not isinstance(attached.get("fixed_camera_pair"), Mapping)):
            raise RuntimeError("held_collision_coverage_evidence_missing")
    except BaseException:
        task.robot.detach_collision_object(arm)
        raise
    return {"policy": HELD_COLLISION_POLICY, "world": world, "attachment": attached}


@contextmanager
def temporary_held_collision(task, actor, arm, *, name, last_qpos, approach_world):
    """Restore the unladen approach world even if binding or admission raises."""
    try:
        yield bind_held_collision(task, actor, arm, name=name, last_qpos=last_qpos)
    finally:
        pending = sys.exc_info()[1]
        try:
            try:
                task.robot.detach_collision_object(arm)
            finally:
                task.robot.set_collision_geometry_mode(arm, "acquisition")
                restored = task.robot.update_arm_world_obstacles(arm, approach_world["obstacles"])
                expected = (approach_world.get(arm) or {}).get("world_hash")
                if not expected or (restored.get(arm) or {}).get("world_hash") != expected:
                    raise RuntimeError("held_planning_approach_world_restore_mismatch")
        except BaseException as cleanup_error:
            if isinstance(pending, TimeoutError):
                raise pending from cleanup_error
            raise


def plan_held_path(task, actor, arm, start_qpos, target_pose, *, world_hash,
                   attachment_hash, support_plane_z_m=None, required_clearance_m=None, carry_motion=None, excluded_entities=()):
    """Plan and admit the exact arrays before anyone may select or execute them."""
    acquisition = required_clearance_m is not None
    if acquisition and (carry_motion is not None or excluded_entities):
        raise ValueError("acquisition cannot use Carry motion or receiver exclusions")
    result = dict(getattr(task.robot, f"{arm}_plan_path")(
        list(target_pose), last_qpos=np.asarray(start_qpos, dtype=float),
        expected_world_hash=world_hash, expected_attachment_hash=attachment_hash,
        m15_exact_table_validation=True,
        **({"m15_route_acquisition_clearance_m": float(required_clearance_m)} if acquisition else {}),
        **({"m15_carry_cartesian_motion": carry_motion} if carry_motion is not None else {}),
    ))
    if result.get("status") != "Success":
        return result
    path = np.asarray(result.get("position"), dtype=float)
    velocity = np.asarray(result.get("velocity"), dtype=float)
    if (path.ndim != 2 or not len(path) or not np.isfinite(path).all()
            or velocity.shape != path.shape or not np.isfinite(velocity).all()
            or result.get("world_hash") != world_hash
            or result.get("attachment_hash") != attachment_hash
            or (result.get("table_validation") or {}).get("attachment_hash") != attachment_hash
            or np.max(np.abs(path[0] - np.asarray(start_qpos))) > 1e-5):
        raise RuntimeError("held_path_binding_mismatch")
    verify_camera_path_proof(result.get("fixed_camera_path"), path, attachment_hash)
    if carry_motion is not None:
        verify_carry_motion_receipt(result.get("carry_motion"), path, mode=carry_motion)
    if acquisition:
        verify_vertical_lift_receipt(result.get("vertical_lift"), path)
        result = validate_acquisition_lift_fcl(task, arm, actor, path, result, {
            "support_plane_z_m": support_plane_z_m,
            "required_pick_clearance_m": required_clearance_m,
        })
    else:
        proof = result.get("table_validation") or {}
        digest = trajectory_summary(path)["trajectory_sha256"]
        if (proof.get("collision_free") is not True or proof.get("trajectory_sha256") != digest
                or proof.get("checked_waypoint_count") != len(path)
                or "planning_lock_joint_positions" not in proof):
            raise RuntimeError("held_carry_table_evidence_missing")
        locks = proof["planning_lock_joint_positions"]
        mesh = task.robot.validate_arm_path_physical_collisions(arm, path,
            sample_stride=1, excluded_entities=(actor, *excluded_entities), planning_lock_joint_positions=locks)
        if (not isinstance(mesh.get("collision_free"), bool) or mesh.get("sample_stride") != 1
                or mesh.get("excluded_entity_count") != 1 + len(excluded_entities) or mesh.get("skipped_scaled_entities")
                or mesh.get("planning_lock_joint_positions") != locks
                or (mesh["collision_free"] and mesh.get("checked_waypoint_count") != len(path))):
            raise RuntimeError("held_carry_mesh_evidence_missing")
        combined = {**proof, "full_robot_mesh_validation": dict(mesh),
                    "collision_free": mesh["collision_free"]}
        combined["validation_hash"] = artifact_hash(combined, omit_keys=("validation_hash",))
        result["table_validation"] = combined
        if not mesh["collision_free"]:
            result.update(status="Fail", planner_status="HELD_CARRY_MESH_COLLISION")
    result["held_collision_policy"] = HELD_COLLISION_POLICY
    return result


def execute_held_lift(task, actor, arm, delta_m, reference, *, support_plane_z_m,
                      required_clearance_m, terminal_settle_max_steps,
                      terminal_joint_tolerance_rad):
    """Both physical adapters execute one freshly admitted, unchanged lift."""
    binding = reference.get("carry_collision_binding") or {}
    if (reference.get("held_collision_policy") != HELD_COLLISION_POLICY
            or binding.get("attachment_hash") != reference.get("attachment_hash")):
        raise RuntimeError("held_lift_binding_missing")
    world = refresh_held_world(task, actor, arm, require_existing_binding=True)
    world_hash = world[arm]["world_hash"]
    reference["collision_world_hash"] = world_hash
    q = np.asarray(getattr(task.robot, f"get_{arm}_arm_real_jointState")(), dtype=float)
    q = q[:len(getattr(task.robot, f"{arm}_arm_joints"))]
    target = list(getattr(task.robot, f"get_{arm}_ee_pose")())
    target[2] += float(delta_m)
    result = plan_held_path(task, actor, arm, q, target,
        world_hash=world_hash, attachment_hash=reference["attachment_hash"],
        support_plane_z_m=support_plane_z_m, required_clearance_m=required_clearance_m)
    task.last_held_collision_plan = {k: deepcopy(v) for k, v in result.items() if k not in {"position", "velocity"}}
    if result.get("status") != "Success":
        task.plan_success = False
        return False
    control = {"left_arm": None, "left_gripper": None, "right_arm": None, "right_gripper": None}
    control[f"{arm}_arm"] = result
    task.take_dense_action(control)
    settled = task._settle_arm_terminal_target(arm, result,
        maximum_steps=terminal_settle_max_steps, tolerance_rad=terminal_joint_tolerance_rad)
    if settled:
        handoff = validate_held_handoff(task, actor, arm, reference)
        reference["held_handoff_validation"] = handoff
        task.last_held_collision_plan["held_handoff_validation"] = deepcopy(handoff)
        settled = handoff["collision_free"]
    task.plan_success = bool(settled)
    return bool(settled)


def validate_held_handoff(task, actor, arm, reference):
    """Check measured joints with strict Carry rules; no support-exit exception."""
    q = np.asarray(getattr(task.robot, f"get_{arm}_arm_real_jointState")(), dtype=float)
    path = q[:len(getattr(task.robot, f"{arm}_arm_joints"))][None]
    digest = trajectory_summary(path)["trajectory_sha256"]
    result = task.robot.validate_arm_path_collision_world(arm, path,
        expected_world_hash=reference["collision_world_hash"], held_table_validation=True)
    if (result.get("validator") != "if_held_handoff_native_and_exact_table_v1"
            or not isinstance(result.get("collision_free"), bool)
            or result.get("trajectory_sha256") != digest
            or result.get("checked_waypoint_count") != 1
            or result.get("world_hash") != reference["collision_world_hash"]
            or result.get("attachment_hash") != reference["attachment_hash"]
            or "planning_lock_joint_positions" not in result):
        raise RuntimeError("held_handoff_evidence_mismatch")
    pair = observe_live_held_camera(task, actor, arm, reference)
    locks = result["planning_lock_joint_positions"]
    mesh = task.robot.validate_arm_path_physical_collisions(arm, path, sample_stride=1,
        excluded_entities=(actor,), planning_lock_joint_positions=locks)
    if (not isinstance(mesh.get("collision_free"), bool) or mesh.get("checked_waypoint_count") != 1
            or mesh.get("planning_lock_joint_positions") != locks
            or mesh.get("excluded_entity_count") != 1 or mesh.get("sample_stride") != 1
            or mesh.get("skipped_scaled_entities")):
        raise RuntimeError("held_handoff_mesh_evidence_mismatch")
    proof = {**result, "full_robot_mesh_validation": mesh, "actual_fixed_camera_pair": pair,
             "collision_free": result["collision_free"] and mesh["collision_free"] and pair["passed"],
             "held_collision_policy": HELD_COLLISION_POLICY,
             "stage_boundary": "measured_lift_end_before_strict_carry"}
    proof["validation_hash"] = artifact_hash(proof)
    return proof


def validate_acquisition_lift_fcl(env, arm, subject, path, result, request):
    """Close the virtual acquisition gate with the same complete mesh path.

    This remains a planning witness; the shared physical Pick implementation
    still owns attachment, continuous lift, and physical certification.
    """
    from if_benchmark.grasp_validation.core import trajectory_summary
    proof = result.get("table_validation") or {}
    native = proof.get("native_non_table_validation") or {}
    path_hash = trajectory_summary(path)["trajectory_sha256"]
    if (proof.get("planner_checks_passed") is not True
            or proof.get("requires_full_robot_fcl") is not True
            or proof.get("trajectory_sha256") != path_hash
            or native.get("trajectory_sha256") != path_hash
            or native.get("collision_free") is not True
            or proof.get("checked_waypoint_count") != len(path)
            or native.get("checked_waypoint_count") != len(path)):
        raise RuntimeError("route_acquisition_planner_evidence_mismatch")
    verify_vertical_lift_receipt(proof.get("vertical_lift"), path)
    locks = proof["planning_lock_joint_positions"]
    fcl = env.robot.validate_arm_path_physical_collisions(
        arm, path, sample_stride=1, excluded_entities=(subject,),
        planning_lock_joint_positions=locks,
    )
    if (not isinstance(fcl, Mapping) or not isinstance(fcl.get("collision_free"), bool)
            or fcl.get("sample_stride") != 1 or fcl.get("excluded_entity_count") != 1
            or fcl.get("skipped_scaled_entities")
            or fcl.get("planning_lock_joint_positions") != locks):
        raise RuntimeError("route_acquisition_mesh_evidence_missing")
    if fcl["collision_free"] and fcl.get("checked_waypoint_count") != len(path):
        raise RuntimeError("route_acquisition_mesh_path_incomplete")
    final_bottom = (proof["raw_held_clearance_by_waypoint_m"][-1]
                    + proof["table_top_plane_z"])
    frozen_clearance = final_bottom - float(request["support_plane_z_m"])
    passed = fcl["collision_free"] and frozen_clearance >= float(request["required_pick_clearance_m"])
    combined = {
        **proof, "full_robot_mesh_validation": dict(fcl),
        "frozen_support_plane_z_m": float(request["support_plane_z_m"]),
        "terminal_frozen_support_clearance_m": frozen_clearance,
        "collision_free": bool(passed), "requires_full_robot_fcl": False,
        "physical_certification": False,
        "stage_boundary": "virtual_acquisition_lift_end_before_strict_carry",
    }
    combined["validation_hash"] = artifact_hash(combined, omit_keys=("validation_hash",))
    return {**result, "table_validation": combined,
            **({"status": "Fail", "planner_status": "ACQUISITION_LIFT_VALIDATION_FAILED"} if not passed else {})}


def observe_live_held_camera(task, actor, arm, reference):
    """Recheck observed physical geometry without rebinding the planned hold."""
    q = np.asarray(getattr(task.robot, f"get_{arm}_arm_real_jointState")(), dtype=float)
    q = q[:len(getattr(task.robot, f"{arm}_arm_joints"))]
    pair = task.robot.observe_held_camera_pair(arm, q,
        complete_held_descriptor(actor, name="held_actual_pose"),
        expected_attachment_hash=reference["attachment_hash"])
    if (not isinstance(pair.get("passed"), bool) or pair.get("attachment_hash") != reference["attachment_hash"]
            or pair.get("validation_hash") != artifact_hash(pair, omit_keys=("validation_hash",))):
        raise RuntimeError("held_handoff_actual_camera_evidence_mismatch")
    return pair
