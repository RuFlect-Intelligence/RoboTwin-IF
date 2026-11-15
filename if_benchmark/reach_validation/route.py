"""Exact-SceneSpec direct-route witness builder.

Failure remains bounded-search evidence and is never promoted to layout
unsat.  A passing, self-hash-valid report is a normative M15 route input.
"""

from __future__ import annotations

from if_benchmark.grasp_validation.carry_motion import carry_motion_mode, verify_carry_motion_receipt

from copy import deepcopy
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import transforms3d as t3d
import yaml

from if_benchmark.schema import artifact_hash
from if_benchmark.grasp_validation.held_collision import (
    validate_acquisition_lift_fcl as _validate_acquisition_lift_fcl,
    complete_held_descriptor,
)
from if_benchmark.grasp_validation.planning_contract import classify_pick_planning_failure
from if_benchmark.terminal_placement import (
    articulated_cavity_world_matrix,
    load_reviewed_receiver_profiles,
    terminal_recipe_binding,
)
from if_benchmark.scene_validation.core import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_REALIZATION_DIR,
    DEFAULT_SCENE_SPEC_DIR,
    DEFAULT_TASK_DIR,
    DEFAULT_UNIT_DIR,
    SceneValidationError,
    ValidationConfig,
    strict_source_join,
)

from .core import (
    DEFAULT_OUTPUT_ROOT,
    HIGH_CONFIDENCE_TASK_IDS,
    PILOT_TASK_IDS,
    rejection_counts,
    write_temporary_report,
)
from .contract import (
    ROUTE_ACTOR_PHASES,
    ROUTE_EXECUTION_CONTRACT,
    ROUTE_VALIDATION_VERSION,
    TerminalCandidateRejected,
    VERTICAL_CONTACT_RELATIONS,
    exact_carry_receiver_keys,
    exact_reservation_receiver_keys,
    held_collision_descriptor,
    release_retreat_ee_matrix,
    route_collision_excluded_entity_keys,
    route_execution_contract_identity,
    route_phase_actor_poses,
    route_phase_policy,
    route_scene_collision_obstacles,
    route_witness_hash,
    terminal_candidate_from_sources,
    terminal_search_offsets,
    terminal_search_yaws,
)


DEFAULT_PLAN = DEFAULT_OUTPUT_ROOT / "r04_pilot_route_plan.json"
DEFAULT_REPORTS = DEFAULT_OUTPUT_ROOT / "route_reports"


def _grasp_planning_diagnostics(grasp: Mapping[str, Any]) -> dict[str, Any]:
    """Retain failed Contact evidence without runtime-only trajectory arrays.

    Counts alone cannot identify which physical collision pair or planner
    request rejected the frozen Contact.  Reachable candidates stay in the
    existing witness; only rejected variants are copied into diagnostics.
    """
    result = {
        "status": grasp.get("status"),
        "evaluated_contact_count": grasp.get("evaluated_contact_count"),
        "reachable_contact_count": grasp.get("reachable_contact_count"),
        "rejection_counts": deepcopy(grasp.get("rejection_counts", {})),
    }
    for key in ("failure_reason", "contacts"):
        if key in grasp:
            result[key] = deepcopy(grasp[key])
    if "variants" in grasp:
        result["rejected_variants"] = [
            {key: deepcopy(value) for key, value in row.items()
             if not str(key).startswith("_")}
            for row in grasp["variants"]
            if row.get("rejection_reason")
        ]
    return result


class _RoutePickPlanningFailure(RuntimeError):
    def __init__(self, classification, grasp):
        self.classification = deepcopy(classification)
        self.grasp = _grasp_planning_diagnostics(grasp)
        super().__init__(f"route_pick_{classification['disposition']}")


def _check_failed_route_pick(grasp):
    counts = grasp.get("rejection_counts") or {}
    if not counts:
        return
    classification = classify_pick_planning_failure({"results": [{
        "status": "failed", "rejection_counts": counts,
    }]})
    if classification["disposition"] not in {"layout_unsat", "operation_search_exhausted"}:
        raise _RoutePickPlanningFailure(classification, grasp)


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return value


def _scene_index(
    scene_dir: Path = DEFAULT_SCENE_SPEC_DIR,
    *,
    wanted_ids: set[str] | None = None,
) -> dict[str, Path]:
    index = {}
    for path in sorted(scene_dir.glob("rsf_*/*.yml")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "scene_spec:" not in text or (
            wanted_ids is not None
            and not any(scene_id in text for scene_id in wanted_ids)
        ):
            continue
        value = _load_yaml(path)
        scene_id = str((value.get("scene_spec") or {}).get("id") or "")
        if scene_id:
            index[scene_id] = path
        if wanted_ids is not None and wanted_ids <= set(index):
            break
    return index


def build_route_plan(
    output: Path = DEFAULT_PLAN,
    *,
    preset: str = "r04-pilot",
    scene_dir: Path = DEFAULT_SCENE_SPEC_DIR,
    realization_dir: Path = DEFAULT_REALIZATION_DIR,
    unit_dir: Path = DEFAULT_UNIT_DIR,
    task_dir: Path = DEFAULT_TASK_DIR,
) -> dict[str, Any]:
    if preset not in {"r04-pilot", "all-direct"}:
        raise ValueError(f"unsupported route preset: {preset}")
    roots = {
        "realization_dir": str(realization_dir),
        "unit_dir": str(unit_dir),
        "task_dir": str(task_dir),
    }
    jobs: list[dict[str, Any]] = []
    if preset == "r04-pilot":
        wanted_ids = {scene_id for scene_id, _ in PILOT_TASK_IDS}
        scenes = _scene_index(scene_dir, wanted_ids=wanted_ids)
        for index, (scene_id, task_id) in enumerate(PILOT_TASK_IDS):
            path = scenes.get(scene_id)
            jobs.append(
                {
                    "job_index": index,
                    "scene_spec_id": scene_id,
                    "task_record_id": task_id,
                    "scene_path": None if path is None else str(path),
                    "source_available": path is not None,
                    "high_confidence": (scene_id, task_id)
                    in HIGH_CONFIDENCE_TASK_IDS,
                    "source_roots": roots,
                }
            )
    else:
        for path in sorted(scene_dir.glob("rsf_*/*.yml")):
            try:
                joined = strict_source_join(
                    path,
                    realization_dir=realization_dir,
                    unit_dir=unit_dir,
                    task_dir=task_dir,
                )
            except SceneValidationError:
                continue
            scene_id = str(joined["scene"]["scene_spec"]["id"])
            for task in joined["tasks"]:
                record = task["task_record"]
                skill = record.get("skill_config") or {}
                carries = [
                    step
                    for step in skill.get("steps", [])
                    if str(step.get("op") or "") == "carry_to_relation"
                ]
                if str(skill.get("route") or "") != "direct" or len(carries) != 1:
                    continue
                jobs.append(
                    {
                        "job_index": len(jobs),
                        "scene_spec_id": scene_id,
                        "task_record_id": str(record["id"]),
                        "scene_path": str(path),
                        "source_available": True,
                        "high_confidence": False,
                        "source_roots": roots,
                    }
                )
    plan = {
        "artifact_kind": "exact_scene_route_validation_plan",
        "version": ROUTE_VALIDATION_VERSION,
        "preset": preset,
        "job_count": len(jobs),
        "source_available_count": sum(job["source_available"] for job in jobs),
        "source_roots": roots,
        "jobs": jobs,
    }
    write_temporary_report(output, plan)
    return plan


def _matrix(pose: Sequence[float]) -> np.ndarray:
    value = np.asarray(pose, dtype=float)
    result = np.eye(4, dtype=float)
    result[:3, :3] = t3d.quaternions.quat2mat(value[3:])
    result[:3, 3] = value[:3]
    return result


def _pose(matrix: np.ndarray) -> list[float]:
    return [
        *matrix[:3, 3].tolist(),
        *t3d.quaternions.mat2quat(matrix[:3, :3]).tolist(),
    ]


def _route_actor_frame_split(
    *,
    scene_spec_actor_pose: Sequence[float],
    live_actor_pose: Sequence[float],
    planner_grasp_pose: Sequence[float],
    terminal_actor_pose: Sequence[float],
    relation: str,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Keep the frozen phase-policy frame separate from the live grasp frame."""

    live_actor_matrix = _matrix(live_actor_pose)
    actor_to_ee = np.linalg.inv(live_actor_matrix) @ _matrix(
        planner_grasp_pose
    )
    phase_actor = route_phase_actor_poses(
        _matrix(scene_spec_actor_pose),
        _matrix(terminal_actor_pose),
        relation=str(relation),
    )
    return actor_to_ee, phase_actor


def _scene_obstacles(env: Any, excluded: set[str]) -> list[dict[str, Any]]:
    return route_scene_collision_obstacles(
        env.scene_spec,
        env.entities,
        excluded_entity_keys=sorted(excluded),
        complete_geometry=True,
        max_obstacles=64 - len(env._m15_live_table_collision_obstacles()),
    )


def _route_phase_obstacles(
    env: Any,
    *,
    subject: str,
    receiver_keys: Sequence[str],
    relation: str,
) -> dict[str, list[dict[str, Any]]]:
    """Build each collision world from its declared phase contract."""

    return {
        phase: _scene_obstacles(
            env,
            set(
                route_collision_excluded_entity_keys(
                    subject=subject,
                    receiver_keys=receiver_keys,
                    relation=relation,
                    phase=phase,
                )
            ),
        )
        for phase in ("transit", "descent", "release_retreat")
    }


def _task(joined: Mapping[str, Any], task_id: str) -> dict[str, Any]:
    matches = [
        row["task_record"]
        for row in joined["tasks"]
        if str(row["task_record"].get("id")) == task_id
    ]
    if len(matches) != 1:
        raise RuntimeError(f"task_record_join_cardinality:{task_id}:{len(matches)}")
    return deepcopy(matches[0])


def _carry_step(task: Mapping[str, Any]) -> dict[str, Any]:
    steps = (task.get("skill_config") or {}).get("steps") or []
    rows = [step for step in steps if step.get("op") == "carry_to_relation"]
    if len(rows) != 1:
        raise RuntimeError(f"direct_route_carry_step_cardinality:{len(rows)}")
    return deepcopy(rows[0])


def _planner_method(env: Any, arm: str):
    return getattr(env.robot, f"{arm}_plan_path")


def _arm_world_hash(world: Mapping[str, Any], arm: str) -> str:
    return str((world.get(arm) or {}).get("world_hash") or "")


def _plan_phase(
    env: Any,
    arm: str,
    start_qpos: Sequence[float],
    target_pose: Sequence[float],
    *,
    world_hash: str,
    attachment_hash: str | None,
    exact_table_validation: bool = False,
    route_lift_support: bool = False,
    acquisition_clearance_m: float | None = None,
    carry_motion: str | None = None,
) -> tuple[dict[str, Any], np.ndarray | None]:
    method = _planner_method(env, arm)
    result = method(
        list(target_pose),
        last_qpos=np.asarray(start_qpos, dtype=float),
        expected_world_hash=world_hash,
        expected_attachment_hash=attachment_hash,
        **({"m15_exact_table_validation": True} if exact_table_validation else {}),
        **({"m15_route_lift_support": True} if route_lift_support else {}),
        **({"m15_route_acquisition_clearance_m": acquisition_clearance_m}
           if acquisition_clearance_m is not None else {}),
        **({"m15_carry_cartesian_motion": carry_motion} if carry_motion is not None else {}),
    )
    if str(result.get("status")) != "Success" or result.get("position") is None:
        return dict(result), None
    path = np.asarray(result["position"], dtype=float)
    if path.ndim != 2 or not len(path):
        return dict(result), None
    if carry_motion is not None:
        verify_carry_motion_receipt(result.get("carry_motion"), path, mode=carry_motion)
    return dict(result), path


def _enable_corrected_route_geometry(env, arm, subject, descriptor):
    """Bind virtual Carry to the same physical pieces/frame as real M15.

    The frozen Contact and acquisition oracle have already been selected.
    This changes only planning geometry, never the pristine SAPIEN scene.
    """
    env.robot.set_collision_geometry_mode(
        arm, "m15_carry_conservative_v1", table_obstacles=env._m15_live_table_collision_obstacles()
    )
    return complete_held_descriptor(subject, name=descriptor["name"])


def _planning_rejection(phase: str, result: Mapping[str, Any]) -> dict[str, Any]:
    planner_status = str(result.get("planner_status") or result.get("status") or "")
    normalized = planner_status.lower()
    if "ik" in normalized:
        category = "ik"
    elif any(token in normalized for token in ("collision", "invalidquery", "invalid_query")):
        category = "collision"
    else:
        category = "path"
    return {
        "rejection_reason": f"no_{phase}_path",
        "rejection_category": category,
        "planner_status": planner_status or None,
        "world_hash": result.get("world_hash"),
        "attachment_hash": result.get("attachment_hash"),
        **({"trajectory_interpolation": deepcopy(result["trajectory_interpolation"])}
           if result.get("trajectory_interpolation") is not None else {}),
        # The exact-table gate runs after CuRobo succeeds.  Preserve its
        # original collision/path evidence on rejection as on success; the
        # status string alone cannot distinguish support deepening from an
        # actual robot/table intersection.
        **({"table_validation": deepcopy(result["table_validation"])}
           if "table_validation" in result else {}),
    }


def _terminal_robot_collision_check(env, arm, qpos, obstacles):
    """Check the candidate endpoint in the actual motion-target world frame.

    The legacy route planner's world transform is not a valid certificate of
    this endpoint. FK uses the same calibrated frame as M15 motion targets;
    a rejected endpoint continues the existing finite terminal search.
    """
    rows = env.robot.get_collision_spheres(arm, qpos)
    spheres = np.asarray([[*row["center"], row["radius"]] for row in rows], dtype=float)
    if spheres.ndim != 2 or spheres.shape[1:] != (4,) or not len(spheres) or not np.isfinite(spheres).all():
        raise RuntimeError("route terminal robot collision FK is incomplete")
    active = spheres[:, 3] > 0
    if not active.any():
        raise RuntimeError("route terminal robot collision FK has no active spheres")
    collisions, minimum = [], None
    for box in obstacles:
        pose, dims = np.asarray(box["pose"], dtype=float), np.asarray(box["dims"], dtype=float)
        if (pose.shape != (7,) or dims.shape != (3,) or not np.isfinite(pose).all()
                or not np.isfinite(dims).all() or np.any(dims <= 0) or np.linalg.norm(pose[3:]) < 1e-12):
            raise RuntimeError("route terminal world obstacle is invalid")
        local = (spheres[:, :3] - pose[:3]) @ t3d.quaternions.quat2mat(pose[3:])
        delta = np.abs(local) - dims / 2
        distances = (np.linalg.norm(np.maximum(delta, 0), axis=1)
                     + np.minimum(delta.max(axis=1), 0) - spheres[:, 3])
        gap = float(distances[active].min())
        minimum = gap if minimum is None else min(minimum, gap)
        collisions.extend({"sphere": rows[int(i)]["name"], "obstacle": box["name"],
                           "clearance_m": float(distances[i])}
                          for i in np.flatnonzero(active & (distances < 1e-6)))
    payload = {"validator": "route_terminal_robot_spheres_calibrated_world_v1", "arm": arm,
               "collision_free": not collisions, "qpos": np.asarray(qpos, dtype=float).tolist(),
               "robot_spheres_hash": artifact_hash(rows), "obstacles_hash": artifact_hash(obstacles),
               "minimum_clearance_m": minimum, "collisions": collisions}
    return {**payload, "validation_hash": artifact_hash(payload)}


def _terminal_candidates(env: Any, step: Mapping[str, Any], *, arm=None, contact_id=None) -> list[dict[str, Any]]:
    relation = str(step["relation"])
    offsets = terminal_search_offsets(relation)
    yaw_deltas = terminal_search_yaws(relation)
    receivers = exact_carry_receiver_keys(step)
    regions: list[Mapping[str, Any]] = []
    for unit in env.unit_artifact.get("units", []):
        for region in (unit.get("params") or {}).get(
            "reserved_regions", []
        ):
            if (
                str(region.get("subject_key") or region.get("subject") or "")
                != str(step["subject"])
                or str(region.get("relation") or "") != relation
            ):
                continue
            if exact_reservation_receiver_keys(region) == receivers:
                regions.append(region)
    if len(regions) != 1:
        raise ValueError(
            f"terminal reservation cardinality is {len(regions)}, expected one"
        )
    profiles = load_reviewed_receiver_profiles()
    recipe_binding = terminal_recipe_binding(
        regions[0], relation, receiver_profiles=profiles
    )
    cavity_matrix = None
    if recipe_binding["receiver_state_profile"] is not None:
        if len(receivers) != 1:
            raise ValueError("articulated terminal requires exactly one receiver")
        receiver_key = receivers[0]
        cavity_matrix = articulated_cavity_world_matrix(
            env.entities[receiver_key],
            recipe_binding["receiver_state_profile"],
        )
    from .candidates import terminal_candidates, ranked_terminals, operation_filtered_terminals
    rows = list(terminal_candidates(env.unit_artifact, env.scene_spec, step,
                                    cavity_world_matrix=cavity_matrix))
    if arm is not None and contact_id is not None:
        rows = operation_filtered_terminals(env.unit_artifact, env.scene_spec, step, rows,
                                           arm=arm, contact_id=contact_id)
    else:
        rows = ranked_terminals(rows)
    return rows



def _m12_frozen_pick_selection(
    joined: Mapping[str, Any], *, subject: str, arm: str
) -> tuple[dict[str, Any], int]:
    """Return the exact M12 oracle request and selected Contact."""

    contract = joined.get("pick_planning_contract") or {}
    witness = contract.get("witness") or {}
    matches = [
        row
        for row in witness.get("results") or []
        if isinstance(row, Mapping)
        and str(row.get("entity_key") or "") == str(subject)
        and str(row.get("arm") or "") == str(arm)
        and row.get("status") == "passed"
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"m12_pick_witness_cardinality:{subject}:{arm}:{len(matches)}"
        )
    selected = matches[0].get("selected_candidate")
    if not isinstance(selected, Mapping) or selected.get("contact_id") is None:
        raise RuntimeError(f"m12_pick_contact_missing:{subject}:{arm}")
    contact_id = int(selected["contact_id"])
    request_targets = {
        (str(row.get("entity_key") or ""), str(row.get("arm") or "")): row
        for row in ((contract.get("request") or {}).get("targets") or [])
        if isinstance(row, Mapping)
    }
    target = request_targets.get((str(subject), str(arm)), {})
    request = target.get("planning_request") or {}
    if not isinstance(request, Mapping) or not request:
        raise RuntimeError(f"m12_pick_request_missing:{subject}:{arm}")
    if contact_id not in {
        int(value) for value in request.get("contact_ids", [])
    }:
        raise RuntimeError(f"m12_pick_contact_outside_request:{subject}:{arm}")
    binding = target.get("contact_contract_binding") or {}
    if not str(binding.get("contact_contract_hash") or ""):
        raise RuntimeError(f"m12_pick_contact_binding_missing:{subject}:{arm}")
    return deepcopy(dict(request)), contact_id


def _chain_metrics(paths: Mapping[str, np.ndarray], limits: Sequence[Sequence[float]]):
    total_steps = sum(len(path) for path in paths.values())
    total_travel = 0.0
    minimum_margin = 1.0
    for path in paths.values():
        if len(path) > 1:
            total_travel += float(np.abs(np.diff(path, axis=0)).sum())
        for row in path:
            margins = []
            for value, pair in zip(row, limits):
                lower, upper = float(pair[0]), float(pair[1])
                span = upper - lower
                if span > 0.0:
                    margins.append(min(value - lower, upper - value) / span)
            minimum_margin = min(minimum_margin, min(margins, default=0.0))
    return {
        "trajectory_steps": int(total_steps),
        "joint_l1_travel": round(total_travel, 10),
        "minimum_normalized_joint_margin": round(float(minimum_margin), 10),
    }


def _compact_path(
    path: np.ndarray,
    planner_result: Mapping[str, Any],
    *,
    expected_world_hash: str,
    expected_attachment_hash: str | None,
) -> dict[str, Any]:
    observed_world_hash = str(planner_result.get("world_hash") or "")
    observed_attachment_hash = planner_result.get("attachment_hash")
    observed_attachment_hash = (
        None
        if observed_attachment_hash in (None, "")
        else str(observed_attachment_hash)
    )
    if not expected_world_hash or not observed_world_hash:
        raise RuntimeError("route phase planner world provenance missing")
    return {
        "steps": int(len(path)),
        **({"table_validation": deepcopy(planner_result["table_validation"])}
           if planner_result.get("table_validation") is not None else {}),
        "start_qpos": path[0].round(8).tolist(),
        "terminal_qpos": path[-1].round(8).tolist(),
        "trajectory_hash": artifact_hash(path.round(10).tolist()),
        "planner_contract": {
            "expected_world_hash": str(expected_world_hash),
            "observed_world_hash": observed_world_hash,
            "expected_attachment_hash": expected_attachment_hash,
            "observed_attachment_hash": observed_attachment_hash,
            **({"trajectory_interpolation": deepcopy(planner_result["trajectory_interpolation"])}
               if planner_result.get("trajectory_interpolation") is not None else {}),
        },
    }


def validate_route_job(
    job: Mapping[str, Any],
    *,
    reports_dir: Path = DEFAULT_REPORTS,
    config_path: Path = DEFAULT_CONFIG_PATH,
    maximum_grasp_candidates: int = 3,
    maximum_terminal_candidates: int | None = None,
    prepared_runtime: Any | None = None,
) -> dict[str, Any]:
    """Prove one exact route, optionally borrowing a pristine scene runtime.

    The caller retains ownership of a prepared runtime.  Its source/config
    identity and unchanged setup state must pass the runtime's lifecycle guard
    both before and after planning.  This skips only simulator reconstruction;
    source joins, candidate search and route-witness semantics remain identical.
    """

    config = ValidationConfig.load(config_path)
    scene_id = str(job["scene_spec_id"])
    task_id = str(job["task_record_id"])
    report_path = reports_dir / scene_id / f"{task_id}.json"
    base = {
        "artifact_kind": "exact_scene_route_witness",
        "version": ROUTE_VALIDATION_VERSION,
        "route_execution_contract": route_execution_contract_identity(),
        "scene_validation_config_hash": config.hash,
        "scene_spec_id": scene_id,
        "task_record_id": task_id,
        "high_confidence": bool(job.get("high_confidence")),
        "layout_feedback": "delayed",
    }
    if not job.get("scene_path"):
        report = {
            **base,
            "status": "source_blocked",
            "failure_code": "scene_spec_source_missing",
        }
        write_temporary_report(report_path, report)
        return json.loads(report_path.read_text(encoding="utf-8"))

    runtime = None
    corrected_route_arm = None
    rejected: list[dict[str, Any]] = []
    try:
        roots = job.get("source_roots") or {}
        joined = strict_source_join(
            Path(str(job["scene_path"])),
            realization_dir=Path(
                str(roots.get("realization_dir") or DEFAULT_REALIZATION_DIR)
            ),
            unit_dir=Path(str(roots.get("unit_dir") or DEFAULT_UNIT_DIR)),
            task_dir=Path(str(roots.get("task_dir") or DEFAULT_TASK_DIR)),
        )
        task = _task(joined, task_id)
        skill = task.get("skill_config") or {}
        if str(skill.get("route")) != "direct":
            raise RuntimeError(f"route_validator_v1_direct_only:{skill.get('route')}")
        arm = str(skill.get("execution_arm") or (task.get("goal") or {}).get("arm"))
        if arm not in {"left", "right"}:
            raise RuntimeError(f"invalid_execution_arm:{arm}")
        step = _carry_step(task)
        if str(step.get("arm")) != arm:
            raise RuntimeError("d4_arm_switch_detected_in_skill_ir")
        subject_key = str(step["subject"])
        receiver_keys = exact_carry_receiver_keys(step)

        if prepared_runtime is None:
            from envs.if_scene_validation import IFSceneValidationRuntime
            runtime = IFSceneValidationRuntime(
                joined,
                evidence_dir=reports_dir / "runtime" / scene_id,
                config=config.data,
            )
            runtime.setup()
        else:
            runtime = prepared_runtime
            guard = getattr(runtime, "assert_pristine_for_planning", None)
            if not callable(guard):
                raise RuntimeError("prepared_route_runtime_guard_missing")
            guard(joined=joined, config=config.data)
        env = runtime.env
        assert env is not None
        subject = env.entities[subject_key]
        frozen_pick_request, frozen_contact_id = _m12_frozen_pick_selection(
            joined, subject=subject_key, arm=arm
        )
        # Constrain the search before its first-success early exit.  Filtering
        # afterwards alone can discard a different early winner without ever
        # planning the one Contact required by the M12 witness.  Keep the
        # canonical request/hash intact; this is a separate execution choice.
        pick_oracle = env.plan_pick_witness_record(
            frozen_pick_request,
            **({"required_contact_id": frozen_contact_id}
               if frozen_contact_id is not None else {}),
        )
        grasp = pick_oracle.get("_runtime_planning")
        if not isinstance(grasp, Mapping):
            grasp = pick_oracle.get("last_planning") or {}
        grasp_candidates = [
            row
            for row in grasp.get("ranked_contact_candidates", [])
            if int(row.get("contact_id", -1)) == frozen_contact_id
        ][:maximum_grasp_candidates]
        if not grasp_candidates:
            _check_failed_route_pick(grasp)
        if len(grasp_candidates) > 1:
            raise RuntimeError(
                f"m12_pick_contact_runtime_duplicate:{subject_key}:{arm}"
            )
        expected_contact_hash = str(
            next(
                row
                for row in (
                    (joined["pick_planning_contract"].get("request") or {}).get(
                        "targets", []
                    )
                )
                if str(row.get("entity_key") or "") == subject_key
                and str(row.get("arm") or "") == arm
            )["contact_contract_binding"]["contact_contract_hash"]
        )
        if grasp_candidates and str(
            grasp_candidates[0].get("contact_contract_hash")
            or grasp.get("contact_contract_hash")
            or ""
        ) != expected_contact_hash:
            raise RuntimeError(
                f"m12_pick_contact_contract_runtime_mismatch:{subject_key}:{arm}"
            )
        terminals = _terminal_candidates(env, step, arm=arm, contact_id=frozen_contact_id)
        from if_benchmark.manipulation_policy import POLICY
        terminal_budget = (POLICY["maximum_terminal_plans"] if maximum_terminal_candidates is None
                           else int(maximum_terminal_candidates))
        terminals = terminals[:terminal_budget]
        subject_descriptor = held_collision_descriptor(
            subject, name=f"held:{subject_key}"
        )
        phase_obstacles = _route_phase_obstacles(
            env,
            subject=subject_key,
            receiver_keys=sorted(receiver_keys),
            relation=str(step["relation"]),
        )
        vertical_contact = str(step["relation"]) in VERTICAL_CONTACT_RELATIONS
        joint_limits = [
            np.asarray(joint.get_limits(), dtype=float).reshape(-1, 2)[0].tolist()
            for joint in (
                env.robot.left_arm_joints
                if arm == "left"
                else env.robot.right_arm_joints
            )
        ]
        witnesses = []
        live_subject_pose = [*subject.get_pose().p, *subject.get_pose().q]
        scene_subject = env.scene_spec["entities"][subject_key]["pose"]
        scene_spec_subject_pose = [
            *scene_subject["position"],
            *scene_subject["quat_wxyz"],
        ]
        for grasp_candidate in grasp_candidates:
            trajectories = grasp_candidate.get("_trajectories") or {}
            grasp_path = np.asarray(trajectories.get("grasp"), dtype=float)
            pre_path = np.asarray(trajectories.get("pre_grasp"), dtype=float)
            if grasp_path.ndim != 2 or pre_path.ndim != 2:
                rejected.append(
                    {
                        "rejection_reason": "grasp_trajectory_missing",
                        "rejection_category": "path",
                    }
                )
                continue
            grasp_path_record = grasp_candidate.get("paths") or {}
            grasp_planner_results = {
                "pre_grasp": dict(
                    grasp_path_record.get("pre_grasp_planner") or {}
                ),
                "grasp": dict(grasp_path_record.get("grasp_planner") or {}),
            }
            grasp_expected = {
                phase: (
                    str(result.get("world_hash") or ""),
                    None,
                )
                for phase, result in grasp_planner_results.items()
            }
            start_qpos = grasp_path[-1]
            env.robot.detach_collision_object(arm)
            corrected_route_arm = arm
            subject_descriptor = _enable_corrected_route_geometry(env, arm, subject, subject_descriptor)
            world = env.robot.update_world_obstacles(
                phase_obstacles["transit"]
            )
            attached = env.robot.attach_collision_object(
                arm, subject_descriptor, last_qpos=start_qpos
            )
            world_hash = _arm_world_hash(world, arm)
            attachment_hash = str(attached["attachment_hash"])
            for terminal in terminals:
                # Every candidate starts from the same grasp witness and the
                # strict transit world, even if the previous vertical descent
                # temporarily excluded an intended receiver.
                world = env.robot.update_world_obstacles(
                    phase_obstacles["transit"]
                )
                world_hash = _arm_world_hash(world, arm)
                actor_to_ee, phase_actor = _route_actor_frame_split(
                    scene_spec_actor_pose=scene_spec_subject_pose,
                    live_actor_pose=live_subject_pose,
                    planner_grasp_pose=grasp_candidate["grasp_pose"],
                    terminal_actor_pose=terminal["actor_pose"],
                    relation=str(step["relation"]),
                )
                phase_paths: dict[str, np.ndarray] = {}
                phase_planner_results = dict(grasp_planner_results)
                phase_expected = dict(grasp_expected)
                current = start_qpos
                failed = None
                for phase in ("lift", "transit", "approach"):
                    ee_target = _pose(phase_actor[phase] @ actor_to_ee)
                    if phase == "lift":
                        # Acquisition starts at the live grasp; mapping the
                        # frozen actor pose back into this frame introduces a
                        # lateral/orientation correction before leaving support.
                        ee_target = list(grasp_candidate["grasp_pose"])
                        ee_target[2] += float(ROUTE_EXECUTION_CONTRACT["phase_offsets_m"][
                            "stack_lift" if str(step["relation"]) == "stack" else "lift"])
                    result, path = _plan_phase(
                        env,
                        arm,
                        current,
                        ee_target,
                        world_hash=world_hash,
                        attachment_hash=attachment_hash,
                        exact_table_validation=True,
                        carry_motion=(carry_motion_mode(phase, phase_actor) if phase != "lift" else None),
                        acquisition_clearance_m=(float(frozen_pick_request["required_pick_clearance_m"])
                                                 if phase == "lift" else None),
                    )
                    if path is None:
                        failed = _planning_rejection(phase, result)
                        break
                    if phase == "lift":
                        result = _validate_acquisition_lift_fcl(
                            env, arm, subject, path, result, frozen_pick_request,
                        )
                        if result["status"] != "Success":
                            failed = _planning_rejection(phase, result)
                            break
                    phase_paths[phase] = path
                    phase_planner_results[phase] = result
                    phase_expected[phase] = (world_hash, attachment_hash)
                    current = path[-1]
                if failed is not None:
                    rejected.append(
                        {
                            **terminal,
                            "contact_id": grasp_candidate["contact_id"],
                            **failed,
                        }
                    )
                    continue
                if vertical_contact:
                    descent_world = env.robot.update_world_obstacles(
                        phase_obstacles["descent"]
                    )
                    world_hash = _arm_world_hash(descent_world, arm)
                descent_ee_matrix = phase_actor["descent"] @ actor_to_ee
                ee_target = _pose(descent_ee_matrix)
                result, path = _plan_phase(
                    env,
                    arm,
                    current,
                    ee_target,
                    world_hash=world_hash,
                    attachment_hash=attachment_hash,
                    exact_table_validation=True,
                    carry_motion="descent",
                )
                if path is None:
                    rejected.append(
                        {
                            **terminal,
                            "contact_id": grasp_candidate["contact_id"],
                            **_planning_rejection("descent", result),
                        }
                    )
                    continue
                phase_paths["descent"] = path
                terminal_robot_validation = _terminal_robot_collision_check(
                    env, arm, path[-1], phase_obstacles["descent"]
                )
                if not terminal_robot_validation["collision_free"]:
                    rejected.append({
                        **terminal, "contact_id": grasp_candidate["contact_id"],
                        "rejection_reason": "descent_terminal_robot_collision",
                        "rejection_category": "collision",
                        "terminal_robot_collision_validation": terminal_robot_validation,
                    })
                    continue
                phase_planner_results["descent"] = result
                phase_expected["descent"] = (world_hash, attachment_hash)
                current = path[-1]
                env.robot.detach_collision_object(arm)
                # Descent is the only phase allowed to omit an intended
                # vertical receiver.  Restore it before proving the released
                # arm's retreat, otherwise the witness certifies a path
                # through collision geometry that M15 must treat as present.
                retreat_world = env.robot.update_world_obstacles(
                    phase_obstacles["release_retreat"]
                )
                world_hash = _arm_world_hash(retreat_world, arm)
                retreat_target = _pose(
                    release_retreat_ee_matrix(descent_ee_matrix)
                )
                result, retreat = _plan_phase(
                    env,
                    arm,
                    current,
                    retreat_target,
                    world_hash=world_hash,
                    attachment_hash=None,
                )
                if retreat is None:
                    rejected.append(
                        {
                            **terminal,
                            "contact_id": grasp_candidate["contact_id"],
                            **_planning_rejection("release_retreat", result),
                        }
                    )
                    # Reattach for the next terminal candidate.
                    attached = env.robot.attach_collision_object(
                        arm, subject_descriptor, last_qpos=start_qpos
                    )
                    attachment_hash = str(attached["attachment_hash"])
                    continue
                phase_paths["release_retreat"] = retreat
                phase_planner_results["release_retreat"] = result
                phase_expected["release_retreat"] = (world_hash, None)
                all_paths = {
                    "pre_grasp": pre_path,
                    "grasp": grasp_path,
                    **phase_paths,
                }
                metrics = _chain_metrics(all_paths, joint_limits)
                witness = {
                    "carry_collision_geometry_mode": "m15_carry_conservative_v1",
                    "terminal_robot_collision_validation": terminal_robot_validation,
                    "contact_id": int(grasp_candidate["contact_id"]),
                    "candidate_id": str(
                        grasp_candidate.get("candidate_id") or ""
                    ),
                    "contact_contract_hash": str(
                        grasp_candidate.get("contact_contract_hash")
                        or grasp.get("contact_contract_hash")
                        or ""
                    ),
                    "terminal_candidate": terminal,
                    "actor_to_ee_transform": actor_to_ee.round(10).tolist(),
                    "phase_actor_poses": {
                        phase: _pose(phase_actor[phase])
                        for phase in ROUTE_ACTOR_PHASES
                    },
                    "phase_policy": route_phase_policy(str(step["relation"])),
                    "paths": {
                        name: _compact_path(
                            path,
                            phase_planner_results[name],
                            expected_world_hash=phase_expected[name][0],
                            expected_attachment_hash=phase_expected[name][1],
                        )
                        for name, path in all_paths.items()
                    },
                    "qpos_chain": [
                        {
                            "phase": name,
                            "terminal_qpos": path[-1].round(8).tolist(),
                        }
                        for name, path in all_paths.items()
                    ],
                    "metrics": metrics,
                }
                # The route contract has four planner identity fields. Keep
                # interpolation diagnostics beside that binding, not inside it.
                # No motion/collision receipt or threshold is removed.
                for path_record in witness["paths"].values():
                    interpolation = path_record["planner_contract"].pop(
                        "trajectory_interpolation", None)
                    if interpolation is not None:
                        path_record["trajectory_interpolation"] = interpolation
                witness["witness_hash"] = route_witness_hash(witness)
                witnesses.append(witness)
                if len(witnesses) >= 3:
                    break
                attached = env.robot.attach_collision_object(
                    arm, subject_descriptor, last_qpos=start_qpos
                )
                attachment_hash = str(attached["attachment_hash"])
            env.robot.detach_collision_object(arm)
            if len(witnesses) >= 3:
                break
        selected = min(
            witnesses,
            key=lambda row: (
                -row["metrics"]["minimum_normalized_joint_margin"],
                row["metrics"]["joint_l1_travel"],
                row["contact_id"],
                row["terminal_candidate"]["offset_xy_m"],
                row["terminal_candidate"]["yaw_delta_deg"],
            ),
            default=None,
        )
        report = {
            **base,
            "status": (
                "witness_found" if selected else "no_witness_in_search_domain"
            ),
            "arm": arm,
            "route": "direct",
            "relation": str(step["relation"]),
            "subject": subject_key,
            "receivers": sorted(receiver_keys),
            "source_hashes": {
                "scene": joined["scene"]["artifact_hash"],
                "realized": joined["realized"]["artifact_hash"],
                "unit": joined["unit"]["artifact_hash"],
                "task": next(
                    row["artifact_hash"]
                    for row in joined["tasks"]
                    if row["task_record"]["id"] == task_id
                ),
            },
            "m12_pick_planning_request_hash": artifact_hash(
                frozen_pick_request
            ),
            "search_domain": {
                "m12_frozen_contact_ids": [frozen_contact_id],
                "grasp_candidate_count": len(grasp_candidates),
                "terminal_candidate_count": len(terminals),
                "phase_offsets_m": dict(
                    ROUTE_EXECUTION_CONTRACT["phase_offsets_m"]
                ),
                "descent_offsets_m": [0.0],
                "candidate_pair_count": len(grasp_candidates) * len(terminals),
            },
            "grasp_planning": _grasp_planning_diagnostics(grasp),
            "rejection_counts": rejection_counts(rejected),
            "rejection_category_counts": {
                category: sum(
                    row.get("rejection_category") == category for row in rejected
                )
                for category in ("semantic", "collision", "ik", "path")
            },
            "rejections": rejected,
            "witness_count": len(witnesses),
            "selected_witness": selected,
            "d4_arm_switch": False,
            "classification_limit": (
                "failure_does_not_imply_layout_unsat"
                if not selected
                else None
            ),
        }
        if prepared_runtime is not None:
            # Route planning may change the planner's collision world, but
            # must not advance or otherwise mutate this shared physical scene.
            runtime.assert_pristine_for_planning(joined=joined, config=config.data)
    except TimeoutError:
        raise
    except Exception as exc:
        source_failure = isinstance(exc, SceneValidationError) or str(exc).startswith(
            "source_"
        )
        report = {
            **base,
            "status": "source_blocked" if source_failure else "infrastructure_blocked",
            "failure_stage": "source" if source_failure else "infrastructure",
            "failure_code": f"{type(exc).__name__}:{exc}",
        }
        if isinstance(exc, _RoutePickPlanningFailure):
            disposition = exc.classification["disposition"]
            stage = {"source_failure": "source", "contract_failure": "internal"}.get(disposition, "infrastructure")
            report.update(
                status="source_blocked" if stage == "source" else "infrastructure_blocked",
                failure_stage=stage, failure_code=str(exc),
                grasp_planning=exc.grasp, failure_classification=exc.classification,
            )
    finally:
        pending = sys.exc_info()[1]
        cleanup_error = None
        try:
            if runtime is not None and corrected_route_arm is not None:
                runtime.env.robot.set_collision_geometry_mode(corrected_route_arm, "acquisition")
        except BaseException as cleanup:
            cleanup_error = cleanup
        try:
            if runtime is not None and prepared_runtime is None:
                runtime.close()
        except BaseException as cleanup:
            if cleanup_error is None:
                cleanup_error = cleanup
        if cleanup_error is not None:
            if pending is not None:
                raise pending from cleanup_error
            raise cleanup_error
    write_temporary_report(report_path, report)
    # The writer seals a copy. Return that same persisted report so callers
    # can bind its report_hash without a second, different source join.
    return json.loads(report_path.read_text(encoding="utf-8"))


def run_route_plan(
    plan_path: Path = DEFAULT_PLAN,
    *,
    reports_dir: Path = DEFAULT_REPORTS,
    config_path: Path = DEFAULT_CONFIG_PATH,
    job_index: int | None = None,
    high_confidence_only: bool = False,
    maximum_terminal_candidates: int | None = None,
) -> dict[str, int]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    jobs = list(plan.get("jobs") or [])
    if job_index is not None:
        jobs = [job for job in jobs if int(job["job_index"]) == int(job_index)]
    if high_confidence_only:
        jobs = [job for job in jobs if job.get("high_confidence")]
    counts: dict[str, int] = {}
    for job in jobs:
        report = validate_route_job(
            job,
            reports_dir=reports_dir,
            config_path=config_path,
            maximum_terminal_candidates=maximum_terminal_candidates,
        )
        status = str(report["status"])
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))
