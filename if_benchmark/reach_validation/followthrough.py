"""Bounded, motion-only grasp admission before M12 freezes a Contact.

This reuses the attached object and the route planner's full collision gates.
It does not move/step the scene and is not a substitute for the post-M13 route
witness or M15 physical execution. Search exhaustion only rejects this branch.
"""
from __future__ import annotations

import numpy as np

from if_benchmark.schema import artifact_hash
from if_benchmark.manipulation_policy import POLICY


def validate_admission_receipt(receipt, domain, *, arm, contact_id):
    if (not isinstance(receipt, dict) or receipt.get("passed") is not True
            or receipt.get("receipt_hash") != artifact_hash(receipt, omit_keys=("receipt_hash",))
            or receipt.get("policy") != POLICY["version"]
            or receipt.get("operation_constraint") != domain
            or receipt.get("arm") != arm or receipt.get("contact_id") != contact_id
            or contact_id not in domain["allowed_contact_ids"]):
        raise ValueError("invalid goal-followthrough identity")
    goals = receipt.get("goals") or []
    if sorted(r.get("goal_variant_id", "") for r in goals) != sorted(domain["goal_variant_ids"]):
        raise ValueError("incomplete goal-followthrough coverage")
    for goal in goals:
        witness = goal.get("witness") or {}
        if goal.get("passed") is not True or not witness.get("terminal"):
            raise ValueError("missing goal-followthrough terminal")
        if set(witness.get("paths", {})) != {"transit", "approach", "descent"}:
            raise ValueError("incomplete goal-followthrough phases")
        for path in witness["paths"].values():
            binding = path.get("planner_contract") or {}
            if (not path.get("trajectory_hash") or not path.get("steps")
                    or not binding.get("expected_world_hash")
                    or binding["expected_world_hash"] != binding.get("observed_world_hash")
                    or not binding.get("expected_attachment_hash")
                    or binding["expected_attachment_hash"] != binding.get("observed_attachment_hash")):
                raise ValueError("stale goal-followthrough planner evidence")
    return receipt


def goal_followthrough_admission(env, target, realized):
    from .route import (_terminal_candidates, _route_phase_obstacles,
                        _route_actor_frame_split, _arm_world_hash, _pose,
                        _plan_phase, _planning_rejection, _compact_path,
                        _terminal_robot_collision_check)
    from if_benchmark.grasp_validation.carry_motion import carry_motion_mode

    domain = target["operation_constraint"]
    scene_op = env.scene_spec["provenance"]["metadata"]["operation_space"]
    if domain["constraint_hash"] != scene_op["constraint_hash"]:
        raise ValueError("followthrough operation constraint source mismatch")
    variants = {v["id"]: v for v in realized["goal_variants"]}
    goals = [(gid, variants[gid]["goal"]) for gid in domain["goal_variant_ids"]]
    subject, arm = target["entity_key"], target["arm"]
    if any(g["subject"] != subject or g["action_family"] != "O2O_relation" for _, g in goals):
        raise ValueError("followthrough target/goal mismatch")
    terminals, worlds = {}, {}

    def check(*, contact_id, grasp_pose, lifted_qpos, binding):
        if contact_id not in domain["allowed_contact_ids"]:
            raise ValueError("followthrough Contact outside frozen local domain")
        actor = env.entities[subject]
        live = [*actor.get_pose().p, *actor.get_pose().q]
        frozen = env.scene_spec["entities"][subject]["pose"]
        source = [*frozen["position"], *frozen["quat_wxyz"]]
        attachment_hash = binding["attachment"]["attachment_hash"]
        outcomes = []
        for gid, goal in goals:
            key = gid, contact_id
            step = {**goal, "arm": arm}
            if key not in terminals:
                terminals[key] = _terminal_candidates(env, step, arm=arm, contact_id=contact_id)
            if gid not in worlds:
                worlds[gid] = _route_phase_obstacles(env, subject=subject,
                    receiver_keys=[goal[k] for k in ("target", "target2") if goal.get(k)],
                    relation=goal["relation"])
            candidates = terminals[key][:POLICY["maximum_terminal_plans"]]
            failures, success = [], None
            for terminal in candidates:
                actor_to_ee, phases = _route_actor_frame_split(
                    scene_spec_actor_pose=source, live_actor_pose=live,
                    planner_grasp_pose=grasp_pose, terminal_actor_pose=terminal["actor_pose"],
                    relation=goal["relation"])
                current, paths = np.asarray(lifted_qpos).copy(), {}
                for phase in ("transit", "approach", "descent"):
                    obstacles = worlds[gid]["descent" if phase == "descent" else "transit"]
                    world = env.robot.update_world_obstacles(obstacles)
                    world_hash = _arm_world_hash(world, arm)
                    result, path = _plan_phase(env, arm, current,
                        _pose(phases[phase] @ actor_to_ee), world_hash=world_hash,
                        attachment_hash=attachment_hash, exact_table_validation=True,
                        carry_motion="descent" if phase == "descent" else carry_motion_mode(phase, phases))
                    if result.get("exception"):
                        raise RuntimeError(f"followthrough planner exception: {result['exception']}")
                    if path is None:
                        failures.append({"offset_xy_m": terminal["offset_xy_m"],
                                         "yaw_delta_deg": terminal["yaw_delta_deg"],
                                         **_planning_rejection(phase, result)})
                        break
                    paths[phase] = _compact_path(path, result,
                        expected_world_hash=world_hash, expected_attachment_hash=attachment_hash)
                    current = path[-1]
                else:
                    collision = _terminal_robot_collision_check(env, arm, current, worlds[gid]["descent"])
                    if not collision["collision_free"]:
                        failures.append({"rejection_reason": "descent_terminal_robot_collision",
                                         "collision": collision})
                        continue
                    success = {"terminal": terminal, "paths": paths}
                    break
            outcomes.append({"goal_variant_id": gid, "passed": success is not None,
                             "candidate_count": len(terminals[key]), "planner_budget": len(candidates),
                             "witness": success, "rejections": failures})
            if success is None:
                break
        result = {"policy": POLICY["version"], "operation_constraint": domain,
                  "arm": arm, "contact_id": int(contact_id), "goals": outcomes,
                  "passed": len(outcomes) == len(goals) and all(r["passed"] for r in outcomes),
                  "scope": "motion_only_transit_approach_descent_before_contact_freeze"}
        return {**result, "receipt_hash": artifact_hash(result)}

    return check
