"""Fresh collision evidence for an unchanged compiled trajectory.

The frozen planning record is never edited. A certificate can only be issued
by running both live validators, and is bound to the exact original path and
both world hashes; user-provided certificate fields are never consumed.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
import re
from typing import Any, Mapping

import numpy as np

from if_benchmark.schema import artifact_hash
from .core import trajectory_summary


@contextmanager
def preserve_collision_constraint_states(generator):
    constraints = [
        getattr(generator.rollout_fn, name)
        for name in ("primitive_collision_constraint", "robot_self_collision_constraint")
    ]
    states = [bool(value.enabled) for value in constraints]
    try:
        yield
    finally:
        for value, enabled in zip(constraints, states):
            if bool(value.enabled) != enabled:
                (value.enable_cost if enabled else value.disable_cost)()


def guard_start_state_constraints(generator):
    """Repair CuRobo's early-return constraint-state leak at its actual call."""
    original = generator.check_start_state
    if getattr(original, "_m15_constraint_guard_v1", False):
        return

    @wraps(original)
    def guarded(*args, **kwargs):
        with preserve_collision_constraint_states(generator):
            return original(*args, **kwargs)

    guarded._m15_constraint_guard_v1 = True
    generator.check_start_state = guarded


def revalidate_live_approach(
    task: Any,
    actor: Any,
    candidate: Mapping[str, Any],
    *,
    arm: str,
    original_world_hash: str,
    live_world_hash: str,
    refresh_world,
) -> dict[str, Any]:
    """Recheck every original waypoint against current CuRobo and FCL worlds."""
    paths = candidate.get("paths") or {}
    trajectories = candidate.get("_trajectories") or {}
    path = np.asarray(trajectories.get("grasp"), dtype=float)
    expected = (paths.get("grasp") or {}).get("trajectory_sha256")
    if path.ndim != 2 or not len(path) or not np.isfinite(path).all():
        raise ValueError("live revalidation requires the original finite compiled path")
    actual = trajectory_summary(path)["trajectory_sha256"]
    if not expected or str(expected) != actual:
        raise ValueError("live revalidation compiled trajectory fingerprint mismatch")
    if (not all(re.fullmatch(r"[0-9a-f]{64}", str(value)) for value in (original_world_hash, live_world_hash))
            or original_world_hash == live_world_hash
            or (paths.get("grasp_planner") or {}).get("world_hash") != original_world_hash):
        raise ValueError("live revalidation requires distinct complete world hashes")
    velocity = np.asarray(trajectories.get("grasp_velocity"), dtype=float)
    if velocity.shape != path.shape or not np.isfinite(velocity).all():
        raise ValueError("live revalidation compiled velocity is missing or invalid")
    planner_validator = getattr(task.robot, "validate_arm_path_collision_world", None)
    exact_validator = getattr(task.robot, "validate_arm_path_physical_collisions", None)
    if not callable(planner_validator) or not callable(exact_validator):
        raise RuntimeError("live revalidation requires both CuRobo and exact-mesh validators")
    planner = dict(planner_validator(arm, path, expected_world_hash=live_world_hash))
    if (planner.get("world_hash") != live_world_hash
            or planner.get("trajectory_sha256") != actual
            or planner.get("checked_waypoint_count") != len(path)
            or planner.get("constraints") != ["joint_bounds", "world_collision", "self_collision"]
            or planner.get("validator") != "curobo_native_path_constraints_v1"
            or not isinstance(planner.get("collision_free"), bool)):
        raise RuntimeError("live CuRobo validation returned incomplete or mismatched evidence")
    exact = dict(exact_validator(arm, path, sample_stride=1, excluded_entities=(actor,)))
    if (not isinstance(exact.get("collision_free"), bool)
            or exact.get("sample_stride") != 1
            or exact.get("skipped_scaled_entities")
            or exact.get("error")
            or (exact["collision_free"] and exact.get("checked_waypoint_count") != len(path))):
        raise RuntimeError("live exact-mesh validation returned incomplete collision evidence")
    if not exact["collision_free"] and not exact.get("collisions"):
        raise RuntimeError("live exact-mesh validation rejected without collision evidence")
    final_world_hash = str(refresh_world())
    payload = {
        "version": "unchanged_approach_live_collision_certificate_v1",
        "candidate_id": str(candidate.get("candidate_id") or ""),
        "contact_id": candidate.get("contact_id"),
        "contact_contract_hash": str(candidate.get("contact_contract_hash") or ""),
        "arm": str(arm),
        "original_world_hash": str(original_world_hash),
        "live_world_hash": str(live_world_hash),
        "world_hash_after_validation": final_world_hash,
        "trajectory_sha256": actual,
        "velocity_sha256": artifact_hash(velocity.tolist()),
        "waypoint_count": len(path),
        "planner_validation": deepcopy(planner),
        "physical_validation": deepcopy(exact),
        "passed": bool(planner["collision_free"] and exact["collision_free"]
                       and final_world_hash == live_world_hash),
    }
    return {**payload, "certificate_hash": artifact_hash(payload)}
