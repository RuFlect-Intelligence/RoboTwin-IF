"""RoboTwin worker for Milestone 5 canonical pair-skill probes."""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from .initial_pose.core import InitialPoseContractError
from .asset_probe_worker import (
    _build_runtime_args,
    _close_env_timed,
    _merge_env_diagnostics,
    _merge_env_timings,
    _record_timing,
    _record_total,
    _reset_reused_planner_state,
    _ensure_probe_symlink,
)


def run_attempt(
    probe_job: dict[str, Any],
    *,
    seed: int,
    task_config: str,
    run_dir: Path,
    runtime_args_template: dict[str, Any] | None = None,
    reusable_env: Any | None = None,
    attempt_id: str | None = None,
    job_index: int | None = None,
    warmup: bool = False,
) -> dict[str, Any]:
    timings: dict[str, Any] = {}
    total_started = time.monotonic()
    args = argparse.Namespace(
        attempt_id=attempt_id,
        job_id=probe_job.get("job_id"),
        job_index=job_index,
        pair_key=probe_job.get("pair_key"),
        relation=probe_job.get("relation"),
        seed=int(seed),
        warmup=bool(warmup),
    )
    try:
        runtime_args = (
            copy.deepcopy(runtime_args_template)
            if runtime_args_template is not None
            else _build_runtime_args(task_config, str(run_dir))
        )
        runtime_args["task_name"] = "if_pair_skill_probe"
        runtime_args["need_topp"] = False
        selected_arm = str(probe_job.get("selected_arm") or "")
        if selected_arm not in {"left", "right"}:
            raise ValueError(
                f"canonical skill job has invalid selected_arm: {selected_arm!r}"
            )
        runtime_args["curobo_planner_arms"] = (selected_arm,)
        if not runtime_args.get("skill_render_dir"):
            runtime_args = _prepare_physics_only_robot_runtime(
                runtime_args,
                run_dir,
            )
    except Exception as exc:
        return _failure(args, "setup_failed", exc, timings, total_started)
    try:
        from envs.if_pair_skill_probe import if_pair_skill_probe
    except Exception as exc:
        return _failure(args, "worker_import_failed", exc, timings, total_started)

    env = reusable_env or if_pair_skill_probe()
    timings["task_env_reused"] = reusable_env is not None
    try:
        started = time.monotonic()
        env.setup_demo(
            now_ep_num=0,
            seed=int(seed),
            task_variant=str(probe_job["relation"]),
            probe_job=probe_job,
            **runtime_args,
        )
        _record_timing(timings, "setup_demo_sec", started)
        _merge_env_timings(timings, env)
        _reset_reused_planner_state(env, timings, enabled=reusable_env is not None)
    except Exception as exc:
        _merge_env_timings(timings, env)
        _close_env_timed(env, timings)
        return _failure(args, _setup_failure_reason(exc), exc, timings, total_started)

    try:
        started = time.monotonic()
        info = env.play_once()
        _record_timing(timings, "play_once_sec", started)
    except Exception as exc:
        return _runtime_failure(
            args,
            probe_job,
            env,
            "worker_exception",
            exc,
            timings,
            total_started,
            failure_phase="play_once",
        )

    # Snapshot the physical action state before calling the checker.  A
    # checker exception is retryable infrastructure failure, but it must not
    # rewrite an already completed action as plan/release/settle failure.
    plan_success = bool(getattr(env, "plan_success", False))
    released = bool(getattr(env, "released", False))
    settled = bool(getattr(env, "settled", False))
    metrics = dict(getattr(env, "probe_metrics", {}) or {})
    checker_success = False
    if plan_success:
        try:
            started = time.monotonic()
            checker_success = bool(env.check_success())
            _record_timing(timings, "check_success_sec", started)
        except Exception as exc:
            return _runtime_failure(
                args,
                probe_job,
                env,
                "worker_exception",
                exc,
                timings,
                total_started,
                failure_phase="check_success",
                info=info,
                observed_state={
                    "plan_success": plan_success,
                    "released": released,
                    "settled": settled,
                    # Keep any evidence the checker emitted before raising.
                    "metrics": dict(
                        getattr(env, "probe_metrics", {}) or metrics
                    ),
                },
            )

    try:
        success = bool(plan_success and released and settled and checker_success)
        result = {
            "attempt_id": attempt_id,
            "job_id": probe_job.get("job_id"),
            "job_index": job_index,
            "pair_key": probe_job.get("pair_key"),
            "relation": probe_job.get("relation"),
            "subject": probe_job.get("subject"),
            "receiver": probe_job.get("receiver"),
            "receiver_state_profile_id": probe_job.get("receiver_state_profile_id"),
            "receiver_state_profile_hash": probe_job.get("receiver_state_profile_hash"),
            "seed": int(seed),
            "plan_success": plan_success,
            "released": released,
            "settled": settled,
            "checker_success": checker_success,
            "success": success,
            "failure_reason": _failure_reason(
                plan_success=plan_success,
                released=released,
                settled=settled,
                checker_success=checker_success,
                metrics=metrics,
            ),
            "info": info or {},
            "metrics": metrics,
            "warmup": bool(warmup),
        }
        predicate_result = metrics.get("predicate_result") or {}
        for key in (
            "success_contract_version",
            "success_contract_hash",
            "relation_contract_hash",
            "success_snapshot_hash",
        ):
            value = metrics.get(key, predicate_result.get(key))
            if value is not None:
                result[key] = value
        if predicate_result:
            result["predicate_result"] = predicate_result
        _merge_env_diagnostics(result, env)
        _close_env_timed(env, timings)
        _record_total(timings, total_started)
        result["timings"] = timings
        result["duration_sec"] = timings.get("total_worker_sec")
        return _jsonable(result)
    except Exception as exc:
        return _runtime_failure(
            args,
            probe_job,
            env,
            "worker_exception",
            exc,
            timings,
            total_started,
            failure_phase="result_serialization",
            info=info,
            observed_state={
                "plan_success": plan_success,
                "released": released,
                "settled": settled,
                "checker_success": checker_success,
                "metrics": metrics,
            },
        )


def run_batch(args: argparse.Namespace) -> list[dict[str, Any]]:
    payload = _load_json(Path(args.batch_file))
    attempts = payload.get("attempts") if isinstance(payload, dict) else payload
    if not isinstance(attempts, list):
        raise ValueError("skill batch must contain an attempts list")
    runtime_args = _build_runtime_args(args.task_config, args.run_dir)
    runtime_args["task_name"] = "if_pair_skill_probe"
    runtime_args["need_topp"] = False
    selected_arms = {
        str((attempt.get("probe_job") or {}).get("selected_arm") or "")
        for attempt in attempts
        if isinstance(attempt, dict)
    }
    if len(selected_arms) != 1 or not selected_arms <= {"left", "right"}:
        raise ValueError(
            "canonical skill batch must contain exactly one selected arm: "
            f"{sorted(selected_arms)!r}"
        )
    runtime_args["curobo_planner_arms"] = (next(iter(selected_arms)),)
    runtime_args = _prepare_physics_only_robot_runtime(
        runtime_args,
        Path(args.run_dir),
    )
    from envs.if_pair_skill_probe import if_pair_skill_probe

    reusable_env = if_pair_skill_probe() if args.reuse_task_env else None
    output = Path(args.batch_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    # Keep one line-buffered stream open so the parent watchdog observes every
    # completed attempt without one metadata/open round-trip per row.
    with output.open("w", encoding="utf-8", buffering=1) as stream:
        for attempt in attempts:
            result = run_attempt(
                attempt["probe_job"],
                seed=int(attempt["seed"]),
                task_config=args.task_config,
                run_dir=Path(args.run_dir),
                runtime_args_template=runtime_args,
                reusable_env=reusable_env,
                attempt_id=str(attempt["attempt_id"]),
                job_index=int(attempt["job_index"]),
                warmup=bool(attempt.get("warmup", False)),
            )
            results.append(result)
            json.dump(_jsonable(result), stream, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
    return results


def _prepare_physics_only_robot_runtime(
    runtime_args: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    """Bind SAPIEN to a visual-free URDF without changing CuRobo inputs.

    The generated CuRobo YAML remains a symlink to the normal run-local file,
    whose ``urdf_path`` still points at the canonical source URDF.  Only
    Robot's SAPIEN loader sees this copy with ``<visual>`` elements removed;
    joints, inertials, collisions, limits and mesh-relative paths are kept.
    """

    if runtime_args.get("physics_only_robot_runtime"):
        return runtime_args
    left_root = Path(str(runtime_args["left_robot_file"]))
    right_root = Path(str(runtime_args["right_robot_file"]))
    if left_root.resolve() != right_root.resolve():
        raise ValueError(
            "canonical skill physics-only runtime requires one dual-arm embodiment"
        )
    embodiment_config = runtime_args.get("left_embodiment_config") or {}
    relative_urdf = Path(str(embodiment_config.get("urdf_path") or ""))
    if not relative_urdf.parts or str(relative_urdf) in {"", "."}:
        raise ValueError("canonical skill embodiment has no URDF path")
    source_urdf = (left_root / relative_urdf).resolve()
    if not source_urdf.is_file():
        raise ValueError(f"canonical skill URDF does not exist: {source_urdf}")

    runtime_root = (
        Path(run_dir)
        / "runtime_embodiments_physics_only"
        / left_root.name
    )
    runtime_root.mkdir(parents=True, exist_ok=True)
    urdf_top_level = relative_urdf.parts[0]
    for child in left_root.iterdir():
        if child.name == urdf_top_level:
            continue
        _ensure_probe_symlink(runtime_root / child.name, child)

    target_urdf = runtime_root / relative_urdf
    target_urdf.parent.mkdir(parents=True, exist_ok=True)
    for child in source_urdf.parent.iterdir():
        if child.resolve() == source_urdf:
            continue
        _ensure_probe_symlink(target_urdf.parent / child.name, child)

    tree = ET.parse(source_urdf)
    visual_count = 0
    for link in tree.getroot().iter("link"):
        for visual in list(link.findall("visual")):
            link.remove(visual)
            visual_count += 1
    temporary = target_urdf.parent / f".{target_urdf.name}.{os.getpid()}.tmp"
    tree.write(temporary, encoding="utf-8", xml_declaration=True)
    os.replace(temporary, target_urdf)

    result = dict(runtime_args)
    result["left_robot_file"] = str(runtime_root)
    result["right_robot_file"] = str(runtime_root)
    result["physics_only_robot_runtime"] = True
    return result


def _failure_reason(
    *,
    plan_success: bool,
    released: bool,
    settled: bool,
    checker_success: bool,
    metrics: dict[str, Any],
) -> str:
    if not plan_success:
        return str(metrics.get("failure_reason") or "motion_planning_failed")
    if not released:
        return "release_failed"
    if not settled:
        return "settle_failed"
    if not checker_success:
        checks = metrics.get("checks") or {}
        if checks.get("joint_state_unchanged") is False:
            return "joint_state_drift"
        if checks.get("no_forbidden_link_support") is False:
            return "forbidden_support_contact"
        return str(metrics.get("failure_reason") or "checker_false")
    return "ok"


def _setup_failure_reason(exc: Exception) -> str:
    text = str(exc)
    if isinstance(exc, InitialPoseContractError) or "source_initial_pose" in text:
        return "source_initial_pose_mismatch"
    if "actor_load_failed" in text:
        return "actor_load_failed"
    if "receiver_state_profile" in text or "missing_profile" in text:
        return "missing_profile"
    return "setup_failed"


def _failure(
    args: argparse.Namespace,
    reason: str,
    exc: Exception,
    timings: dict[str, Any],
    total_started: float,
) -> dict[str, Any]:
    _record_total(timings, total_started)
    return {
        "attempt_id": getattr(args, "attempt_id", None),
        "job_id": getattr(args, "job_id", None),
        "job_index": getattr(args, "job_index", None),
        "pair_key": getattr(args, "pair_key", None),
        "relation": getattr(args, "relation", None),
        "seed": int(args.seed),
        "plan_success": False,
        "released": False,
        "settled": False,
        "checker_success": False,
        "success": False,
        "failure_reason": reason,
        "exception_type": type(exc).__name__,
        "exception": str(exc),
        "traceback_tail": traceback.format_exc()[-4000:],
        "timings": timings,
        "duration_sec": timings.get("total_worker_sec"),
        "warmup": bool(getattr(args, "warmup", False)),
    }


def _runtime_failure(
    args: argparse.Namespace,
    probe_job: dict[str, Any],
    env: Any,
    reason: str,
    exc: Exception,
    timings: dict[str, Any],
    total_started: float,
    *,
    failure_phase: str,
    info: Any | None = None,
    observed_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Preserve observed rollout state when runtime/checker code raises."""

    observed = dict(observed_state or {})
    observed.setdefault(
        "plan_success", bool(getattr(env, "plan_success", False))
    )
    observed.setdefault("released", bool(getattr(env, "released", False)))
    observed.setdefault("settled", bool(getattr(env, "settled", False)))
    observed.setdefault("checker_success", False)
    observed.setdefault(
        "metrics", dict(getattr(env, "probe_metrics", {}) or {})
    )
    diagnostics: dict[str, Any] = {}
    _merge_env_diagnostics(diagnostics, env)
    _merge_env_timings(timings, env)
    _close_env_timed(env, timings)
    result = _failure(args, reason, exc, timings, total_started)
    result.update(
        {
            "subject": probe_job.get("subject"),
            "receiver": probe_job.get("receiver"),
            "receiver_state_profile_id": probe_job.get(
                "receiver_state_profile_id"
            ),
            "receiver_state_profile_hash": probe_job.get(
                "receiver_state_profile_hash"
            ),
            "plan_success": bool(observed["plan_success"]),
            "released": bool(observed["released"]),
            "settled": bool(observed["settled"]),
            "checker_success": bool(observed["checker_success"]),
            "success": False,
            "failure_phase": failure_phase,
            "info": info or {},
            "metrics": dict(observed["metrics"] or {}),
            **diagnostics,
        }
    )
    return _jsonable(result)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-file", required=True)
    parser.add_argument("--batch-output", required=True)
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--reuse-task-env", action="store_true")
    args = parser.parse_args(argv)
    run_batch(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
