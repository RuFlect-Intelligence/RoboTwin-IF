"""Standalone SAPIEN worker for M5 articulated receiver terminal probes.

The module deliberately has no registration in ``if_benchmark.__main__`` yet.
That keeps an in-flight rigid sweep on one immutable import graph.  It can be
invoked directly after reviewed receiver profiles are available:

    python -m if_benchmark.articulated_terminal_worker --probe-job-file JOB
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from .articulated_pair_validation import (
    articulated_policy_hash,
    resolve_joint_targets,
    validate_receiver_profile,
)
from .articulated_terminal_contract import (
    articulation_state_metrics,
    articulated_place_in_success_metrics,
)
from .terminal_placement import (
    articulated_cavity_world_matrix,
    articulated_terminal_actor_pose,
)
from .pair_terminal_settle_worker import (
    CONTACT_IMPULSE_EPS,
    SETTLE_MAX_STEPS,
    SETTLE_MIN_STEPS,
    STABILITY_WINDOW_STEPS,
    TABLE_TOP_Z,
    _entity_pose_state,
    actor_aabb,
    actor_state,
    contact_metrics,
    stability_from_snapshots,
)


WORKER_SCHEMA_VERSION = 1


def run_articulated_attempt(probe_job: dict[str, Any], *, seed: int) -> dict[str, Any]:
    started = time.monotonic()
    profile_id = str(probe_job.get("receiver_state_profile_id") or "")
    profile = probe_job.get("receiver_state_profile") or {}
    blockers = validate_receiver_profile(profile_id, profile)
    if blockers:
        return _result(
            probe_job,
            seed=seed,
            outcome="blocked",
            success=False,
            failure_reason="probe_spec_blocked",
            metrics={"blocked_reasons": blockers},
            duration_sec=time.monotonic() - started,
        )
    runtime: ArticulatedTerminalRuntime | None = None
    try:
        runtime = ArticulatedTerminalRuntime(probe_job, seed=seed)
        outcome = runtime.run()
        return _result(
            probe_job,
            seed=seed,
            outcome="pass" if outcome["success"] else "fail",
            success=bool(outcome["success"]),
            failure_reason=str(outcome["failure_reason"]),
            metrics=outcome["metrics"],
            duration_sec=time.monotonic() - started,
        )
    except Exception as exc:
        return _result(
            probe_job,
            seed=seed,
            outcome="blocked",
            success=False,
            failure_reason="worker_exception",
            metrics={
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "traceback_tail": traceback.format_exc()[-4000:],
            },
            duration_sec=time.monotonic() - started,
        )
    finally:
        if runtime is not None:
            runtime.close()


def run_articulated_batch(batch_file: Path, output_path: Path) -> list[dict[str, Any]]:
    payload = _load_json(batch_file)
    attempts = payload.get("attempts")
    if not isinstance(attempts, list):
        raise ValueError("articulated batch must contain an attempts list")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")
    results: list[dict[str, Any]] = []
    for index, attempt in enumerate(attempts):
        if not isinstance(attempt, dict) or not isinstance(attempt.get("job"), dict):
            raise ValueError(f"invalid articulated attempt at batch index {index}")
        result = run_articulated_attempt(
            attempt["job"],
            seed=int(attempt["seed"]),
        )
        result.update(
            {
                "attempt_id": str(attempt["attempt_id"]),
                "job_id": str(attempt["job_id"]),
                "job_index": int(attempt["job_index"]),
                "pair_key": str(attempt["pair_key"]),
                "seed": int(attempt["seed"]),
            }
        )
        results.append(result)
        with output_path.open("a", encoding="utf-8") as stream:
            json.dump(result, stream, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
        # Every attempt still owns a fresh Engine/Renderer/Scene.  Force
        # collection before the next attempt so batching only reuses Python
        # imports, never SAPIEN scene state.
        gc.collect()
    return results


class ArticulatedTerminalRuntime:
    def __init__(self, probe_job: dict[str, Any], *, seed: int):
        self.probe_job = probe_job
        self.seed = int(seed)
        self.profile = probe_job["receiver_state_profile"]
        self.profile_id = str(probe_job["receiver_state_profile_id"])
        self.scene: Any = None
        self.engine: Any = None
        self.renderer: Any = None
        self.table: Any = None
        self.ground: Any = None
        self.receiver: Any = None
        self.subject: Any = None
        self.resolved_joint_targets: list[dict[str, Any]] = []

    def run(self) -> dict[str, Any]:
        self._setup_scene()
        self._load_and_set_receiver()
        receiver_state_before = self._settle_receiver()
        if not receiver_state_before["held"]:
            return {
                "success": False,
                "failure_reason": "receiver_state_drift",
                "metrics": {"receiver_state": receiver_state_before, "phase": "receiver_only_settle"},
            }
        self._load_and_place_subject()
        terminal_stability = self._settle_subject()
        metrics = self._terminal_metrics(terminal_stability)
        return {
            "success": bool(metrics["success"]),
            "failure_reason": str(metrics["failure_reason"]),
            "metrics": metrics,
        }

    def close(self) -> None:
        self.subject = None
        self.receiver = None
        self.table = None
        self.ground = None
        self.scene = None
        self.renderer = None
        self.engine = None

    def _setup_scene(self) -> None:
        import sapien
        from sapien.render import set_global_config

        try:
            set_global_config(max_num_materials=50000, max_num_textures=50000)
        except Exception:
            pass
        self.engine = sapien.Engine()
        self.renderer = sapien.SapienRenderer()
        self.engine.set_renderer(self.renderer)
        self.scene = self.engine.create_scene(sapien.SceneConfig())
        self.scene.set_timestep(1 / 250)
        self.ground = self.scene.add_ground(0)
        self.scene.default_physical_material = self.scene.create_physical_material(0.5, 0.5, 0)
        from envs.utils import create_table

        self.table = create_table(
            self.scene,
            sapien.Pose(p=[0, 0, 0.74]),
            length=1.2,
            width=0.7,
            height=0.74,
            thickness=0.05,
            is_static=True,
            name="table",
        )

    def _load_and_set_receiver(self) -> None:
        import sapien
        from envs.utils import create_sapien_urdf_obj

        receiver = self.profile["receiver"]
        fixture = self.profile["fixture"]
        root_z = float(fixture.get("root_z") or TABLE_TOP_Z)
        pose = sapien.Pose(
            [
                float(fixture["table_xy"][0]),
                float(fixture["table_xy"][1]),
                root_z,
            ],
            [float(value) for value in fixture["root_q0_wxyz"]],
        )
        self.receiver = create_sapien_urdf_obj(
            scene=self.scene,
            pose=pose,
            modelname=str(receiver["asset"]),
            modelid=_model_id(receiver["model_id"]),
            fix_root_link=True,
            mass=None,
        )
        if self.receiver is None:
            raise RuntimeError("actor_load_failed:receiver")
        self.receiver.set_name(f"articulated_receiver:{receiver['asset']}:{receiver['model_id']}")
        qpos, targets = resolve_joint_targets(
            self.profile,
            self.receiver.get_qlimits(),
            self.receiver.get_qpos(),
        )
        self.resolved_joint_targets = targets
        self.receiver.set_qpos(np.asarray(qpos, dtype=float))
        self.receiver.set_qvel(np.zeros_like(np.asarray(qpos, dtype=float)))
        drive = self.profile.get("passive_joint_properties") or {}
        if drive:
            _hold_reviewed_joint_targets(self.receiver, targets, drive)

    def _settle_receiver(self) -> dict[str, Any]:
        for _ in range(SETTLE_MIN_STEPS):
            self.scene.step()
        return articulation_state_metrics(
            final_qpos=np.asarray(self.receiver.get_qpos(), dtype=float).tolist(),
            resolved_joint_targets=self.resolved_joint_targets,
        )

    def _load_and_place_subject(self) -> None:
        import sapien
        from envs.utils import create_actor

        subject = self.probe_job["subject"]
        canonical = self.probe_job["canonical_poses"]["subject"]
        q0 = [float(value) for value in canonical["q0_wxyz"]]
        self.subject = create_actor(
            self.scene,
            pose=sapien.Pose([0.0, -0.1, TABLE_TOP_Z + 0.25], q0),
            modelname=str(subject["asset"]),
            model_id=_model_id(subject["model_id"]),
            convex=True,
            is_static=False,
            mass=None,
        )
        if self.subject is None:
            raise RuntimeError("actor_load_failed:subject")
        self.subject.set_name(f"articulated_subject:{subject['asset']}:{subject['model_id']}")
        self.subject.set_mass(0.01)
        initial_pose = self.subject.get_pose()
        initial_aabb = actor_aabb(self.subject)
        local_bottom_z = float(initial_aabb["min"][2] - initial_pose.p[2])
        cavity_matrix = self._cavity_world_matrix()
        actor_pose = articulated_terminal_actor_pose(
            cavity_world_matrix=cavity_matrix,
            reviewed_receiver_profile=self.profile,
            placement_recipe=self.probe_job["placement_recipe"],
            subject_q0_wxyz=q0,
            witness_delta_deg=float((self.probe_job.get("yaw_witness") or {}).get("witness_delta_deg") or 0.0),
            subject_local_bottom_z=local_bottom_z,
        )
        self.subject.actor.set_pose(
            sapien.Pose(actor_pose[:3], actor_pose[3:])
        )

    def _settle_subject(self) -> dict[str, Any]:
        for _ in range(SETTLE_MIN_STEPS):
            self.scene.step()
        steps = SETTLE_MIN_STEPS
        last = {"stable": False, "failure_reason": "empty_stability_window"}
        while steps + STABILITY_WINDOW_STEPS <= SETTLE_MAX_STEPS:
            snapshots: list[list[dict[str, Any]]] = []
            for _ in range(STABILITY_WINDOW_STEPS):
                self.scene.step()
                snapshots.append([_entity_pose_state(self.subject.actor)])
            steps += STABILITY_WINDOW_STEPS
            last = stability_from_snapshots(snapshots)
            last["settle_steps"] = steps
            if last.get("stable"):
                return last
        return last

    def _terminal_metrics(self, terminal_stability: dict[str, Any]) -> dict[str, Any]:
        cavity = self.profile["cavity"]
        contact_by_link: dict[str, Any] = {}
        for link_name in sorted(set(cavity["allowed_support_links"]) | set(cavity["forbidden_support_links"])):
            link = self.receiver.link_dict.get(link_name)
            if link is None:
                contact_by_link[link_name] = {
                    "active": False,
                    "missing_link": True,
                    "impulse_epsilon": CONTACT_IMPULSE_EPS,
                }
                continue
            contact_by_link[link_name] = contact_metrics(
                self.scene,
                self.subject.actor,
                _link_entity(link),
            )
        table_contact = contact_metrics(self.scene, self.subject.actor, self.table)
        world_contact = (
            contact_metrics(self.scene, self.subject.actor, self.ground)
            if self.ground is not None
            else {"active": False}
        )
        receiver_state = articulation_state_metrics(
            final_qpos=np.asarray(self.receiver.get_qpos(), dtype=float).tolist(),
            resolved_joint_targets=self.resolved_joint_targets,
        )
        metrics = articulated_place_in_success_metrics(
            subject_state=actor_state(self.subject),
            cavity_world_matrix=self._cavity_world_matrix(),
            cavity=cavity,
            contact_by_link=contact_by_link,
            subject_table_contact=bool(table_contact["active"]),
            subject_world_contact=bool(world_contact["active"]),
            terminal_stability=terminal_stability,
            articulation_state=receiver_state,
        )
        metrics.update(
            {
                "contact_by_link": contact_by_link,
                "subject_table_contact": table_contact,
                "subject_world_contact": world_contact,
                "receiver_state_profile_id": self.profile_id,
                "receiver_state_profile_hash": self.probe_job.get("receiver_state_profile_hash"),
                "recipe_hash": self.probe_job.get("recipe_hash"),
                "articulated_policy_hash": articulated_policy_hash(),
            }
        )
        return metrics

    def _cavity_world_matrix(self) -> np.ndarray:
        return articulated_cavity_world_matrix(self.receiver, self.profile)


def _link_entity(link: Any) -> Any:
    entity = getattr(link, "entity", None)
    if entity is not None:
        return entity
    getter = getattr(link, "get_entity", None)
    if callable(getter):
        return getter()
    return link


def _model_id(value: Any) -> int | str:
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def _hold_reviewed_joint_targets(
    receiver: Any,
    resolved_targets: list[dict[str, Any]],
    drive: dict[str, Any],
) -> None:
    joints = list(receiver.actor.get_active_joints())
    for target in resolved_targets:
        index = int(target["joint_index"])
        if index >= len(joints):
            raise ValueError(f"joint_index_out_of_range:{index}")
        joint = joints[index]
        joint.set_drive_properties(
            stiffness=float(drive.get("stiffness") or 200.0),
            damping=float(drive.get("damping") or 20.0),
            force_limit=float(drive.get("force_limit") or 100.0),
        )
        if drive.get("friction") is not None:
            joint.set_friction(float(drive["friction"]))
        joint.set_drive_target(float(target["target_qpos"]))


def _result(
    probe_job: dict[str, Any],
    *,
    seed: int,
    outcome: str,
    success: bool,
    failure_reason: str,
    metrics: dict[str, Any],
    duration_sec: float,
) -> dict[str, Any]:
    return {
        "schema_version": WORKER_SCHEMA_VERSION,
        "attempt_id": f"{probe_job.get('job_id')}|seed{seed}",
        "job_id": probe_job.get("job_id"),
        "pair_key": probe_job.get("pair_key"),
        "relation": probe_job.get("relation"),
        "receiver_state_profile_id": probe_job.get("receiver_state_profile_id"),
        "band_id": probe_job.get("band_id"),
        "seed": int(seed),
        "outcome": str(outcome),
        "plan_success": outcome != "blocked",
        "success": bool(success),
        "failure_reason": "ok" if success else str(failure_reason),
        "duration_sec": round(float(duration_sec), 6),
        "metrics": _jsonable(metrics),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON mapping in {path}")
    return data


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--probe-job-file", type=Path)
    inputs.add_argument("--batch-file", type=Path)
    parser.add_argument("--batch-output", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.batch_file is not None:
        if args.batch_output is None:
            raise ValueError("--batch-output is required with --batch-file")
        results = run_articulated_batch(args.batch_file, args.batch_output)
        print(
            json.dumps(
                {
                    "batch_output": str(args.batch_output),
                    "batch_result_count": len(results),
                },
                sort_keys=True,
                ensure_ascii=False,
            )
        )
        return 0
    result = run_articulated_attempt(_load_json(args.probe_job_file), seed=args.seed)
    json.dump(result, sys.stdout, sort_keys=True, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0 if result["outcome"] != "blocked" else 2


if __name__ == "__main__":
    raise SystemExit(main())
