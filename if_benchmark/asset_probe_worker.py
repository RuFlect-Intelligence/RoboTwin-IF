"""RoboTwin worker for Milestone 4 canonical asset probes.

The worker supports the original one-shot mode and a batch mode.  Batch mode
keeps one Python process alive for multiple attempts, and can optionally reuse a
single task object across attempts while each attempt still calls setup_demo and
close_env.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import yaml

from if_benchmark.asset_validation_contract import M4_DISABLED_USES
from if_benchmark.execution_policy import (
    m4_worker_execution_policy_manifest,
)
from if_benchmark.initial_pose.core import InitialPoseContractError
from if_benchmark.runtime_errors import PhysicalCollisionValidationError


USE_TO_CANDIDATE = {
    "pick_left": "pick_subject",
    "pick_right": "pick_subject",
    "open": "articulated_open_close",
    "close": "articulated_open_close",
    "switch_on": "activate_subject",
    "switch_off": "activate_subject",
    "click_subject": "click_subject",
    "press_subject": "press_subject",
    "rotate_to_orientation_subject": "rotate_to_orientation_subject",
    "adjust_orientation_subject": "adjust_orientation_subject",
    "bimanual_grab_subject": "bimanual_grab_subject",
}
M4_NON_EVALUATED_USES = {
    "handover": "m4_capability_explicitly_not_evaluated",
    **{
        use: "m4_capability_explicitly_disabled"
        for use in M4_DISABLED_USES
    },
}


def run_worker(args: argparse.Namespace) -> dict[str, Any]:
    return _run_attempt(args)


def run_batch(args: argparse.Namespace) -> list[dict[str, Any]]:
    batch = _load_json_required(Path(args.batch_file))
    attempts = batch.get("attempts") if isinstance(batch, dict) else batch
    if not isinstance(attempts, list):
        raise ValueError(f"expected attempts list in {args.batch_file}")

    needs_catalog = _batch_needs_catalog(attempts)
    catalog = _load_yaml_required(Path(args.asset_catalog)) if needs_catalog else {}
    return _run_batch_attempts(args, attempts, catalog)


def _batch_needs_catalog(attempts: list[Any]) -> bool:
    """A payload shell without its candidate still requires catalog lookup."""

    return any(
        not isinstance(attempt, dict)
        or not isinstance(attempt.get("probe_job"), dict)
        or not isinstance(attempt["probe_job"].get("candidate"), dict)
        for attempt in attempts
    )


def _run_batch_attempts(
    args: argparse.Namespace,
    attempts: list[dict[str, Any]],
    catalog: dict[str, Any],
) -> list[dict[str, Any]]:
    runtime_args_template = _build_runtime_args(args.task_config, args.run_dir)
    from envs.if_asset_probe import if_asset_probe

    reusable_env = if_asset_probe() if args.reuse_task_env else None
    results: list[dict[str, Any]] = []
    output_path = Path(args.batch_output) if args.batch_output else None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")

    for index, attempt in enumerate(attempts):
        if not isinstance(attempt, dict):
            raise ValueError(f"expected attempt mapping at batch index {index}")
        attempt_args = argparse.Namespace(
            asset_catalog=args.asset_catalog,
            asset=str(attempt.get("asset")),
            model_id=str(attempt.get("model_id")),
            use=str(attempt.get("use")),
            requested_arm=attempt.get("requested_arm"),
            seed=int(attempt.get("seed")),
            task_config=args.task_config,
            run_dir=args.run_dir,
            attempt_id=attempt.get("attempt_id"),
            job_id=attempt.get("job_id"),
            job_index=attempt.get("job_index"),
            warmup=bool(attempt.get("warmup", False)),
            probe_job=attempt.get("probe_job"),
            probe_job_file=None,
            diagnostic_video_dir=getattr(args, "diagnostic_video_dir", None),
        )
        result = _run_attempt(
            attempt_args,
            catalog=catalog,
            runtime_args_template=runtime_args_template,
            env_cls=if_asset_probe,
            reusable_env=reusable_env,
        )
        results.append(result)
        if output_path is not None:
            with output_path.open("a", encoding="utf-8") as f:
                json.dump(_jsonable(result), f, sort_keys=True, ensure_ascii=False)
                f.write("\n")
    return results


def _run_attempt(
    args: argparse.Namespace,
    *,
    catalog: dict[str, Any] | None = None,
    runtime_args_template: dict[str, Any] | None = None,
    env_cls: Any | None = None,
    reusable_env: Any | None = None,
) -> dict[str, Any]:
    timings: dict[str, Any] = {}
    total_started = time.monotonic()
    if str(args.use) in M4_NON_EVALUATED_USES:
        return _failure(
            args,
            M4_NON_EVALUATED_USES[str(args.use)],
            ValueError(f"M4 capability is disabled or not evaluated: {args.use!r}"),
            timings=timings,
            total_started=total_started,
        )
    try:
        started = time.monotonic()
        probe_job = _build_probe_job_from_catalog(catalog, args) if catalog is not None else _build_probe_job(args)
        _record_timing(timings, "build_probe_job_sec", started)

        started = time.monotonic()
        if runtime_args_template is None:
            runtime_args = _build_runtime_args(args.task_config, args.run_dir)
        else:
            runtime_args = copy.deepcopy(runtime_args_template)
        _record_timing(timings, "build_runtime_args_sec", started)
    except Exception as exc:
        return _failure(args, "missing_candidate_metadata", exc, timings=timings, total_started=total_started)
    try:
        if env_cls is None:
            started = time.monotonic()
            from envs.if_asset_probe import if_asset_probe
            env_cls = if_asset_probe
            _record_timing(timings, "import_env_sec", started)
        else:
            timings["import_env_sec"] = 0.0
    except Exception as exc:
        return _failure(args, "worker_import_failed", exc, timings=timings, total_started=total_started)

    env = None
    diagnostic_video: dict[str, Any] | None = None
    try:
        if reusable_env is None:
            started = time.monotonic()
            env = env_cls()
            _record_timing(timings, "env_construct_sec", started)
            timings["task_env_reused"] = False
        else:
            env = reusable_env
            timings["env_construct_sec"] = 0.0
            timings["task_env_reused"] = True
        started = time.monotonic()
        env.setup_demo(
            now_ep_num=0,
            seed=int(args.seed),
            task_variant=args.use,
            probe_job=probe_job,
            **runtime_args,
        )
        video_dir = getattr(args, "diagnostic_video_dir", None)
        if video_dir:
            label = str(
                getattr(args, "attempt_id", None)
                or f"{args.asset}_{args.model_id}_{args.use}_seed{args.seed}"
            )
            safe_label = "".join(
                character if character.isalnum() or character in "-_." else "_"
                for character in label
            )
            env.begin_diagnostic_video(Path(video_dir) / f"{safe_label}.mp4")
        _record_timing(timings, "setup_demo_sec", started)
        _merge_env_timings(timings, env)
        _reset_reused_planner_state(env, timings, enabled=reusable_env is not None)
    except Exception as exc:
        if env is not None:
            _merge_env_timings(timings, env)
            _close_env_timed(env, timings)
        reason = (
            "source_initial_pose_mismatch"
            if isinstance(exc, InitialPoseContractError)
            or "source_initial_pose" in str(exc)
            else "setup_failed"
        )
        result = _failure(
            args,
            reason,
            exc,
            timings=timings,
            total_started=total_started,
        )
        _merge_env_diagnostics(result, env)
        return result

    try:
        started = time.monotonic()
        info = env.play_once()
        _record_timing(timings, "play_once_sec", started)
        plan_success = bool(getattr(env, "plan_success", False))
        started = time.monotonic()
        # The physical checker is evidence, not a branch of the execution
        # gate.  Always observe the terminal state, including after planner or
        # execution failure, so every completed runtime attempt carries a
        # SuccessSnapshot and canonical contract identity.
        checker_success = bool(env.check_success())
        success = bool(plan_success and checker_success)
        _record_timing(timings, "check_success_sec", started)
        failure_reason = (
            "ok" if success else ("planner_failed" if not plan_success else "checker_false")
        )
        articulation_code = _articulation_failure_code(env)
        if not success and articulation_code is not None:
            failure_reason = articulation_code
        result = {
            "attempt_id": getattr(args, "attempt_id", None),
            "job_id": getattr(args, "job_id", None),
            "job_index": getattr(args, "job_index", None),
            "asset": args.asset,
            "model_id": str(args.model_id),
            "use": args.use,
            "requested_arm": getattr(args, "requested_arm", None),
            "seed": int(args.seed),
            "plan_success": plan_success,
            "checker_success": checker_success,
            "success": success,
            "failure_reason": failure_reason,
            "info": info or {},
            "metrics": getattr(env, "probe_metrics", {}),
            "warmup": bool(getattr(args, "warmup", False)),
            "execution_policy_manifest": (
                m4_worker_execution_policy_manifest()
            ),
        }
        if getattr(env, "_diagnostic_video_writer", None) is not None:
            diagnostic_video = env.finish_diagnostic_video(
                outcome="passed" if success else "failed"
            )
            result["diagnostic_video"] = diagnostic_video
        predicate_result = getattr(env, "success_checker_result", None)
        if isinstance(predicate_result, dict):
            result["predicate_result"] = predicate_result
            for key in (
                "success_contract_version",
                "success_contract_hash",
                "relation_contract_hash",
                "success_snapshot_hash",
            ):
                value = predicate_result.get(key)
                if value is not None:
                    result[key] = value
        _merge_env_diagnostics(result, env)
        _close_env_timed(env, timings)
        _record_total(timings, total_started)
        result["timings"] = timings
        result["duration_sec"] = timings["total_worker_sec"]
        return result
    except Exception as exc:
        if env is not None and getattr(env, "_diagnostic_video_writer", None) is not None:
            try:
                diagnostic_video = env.finish_diagnostic_video(outcome="exception")
            except Exception:
                diagnostic_video = None
        reason = (
            "worker_physical_collision_validation_failed"
            if isinstance(exc, PhysicalCollisionValidationError)
            else "worker_exception"
        )
        result = _failure(args, reason, exc, timings=timings, total_started=total_started)
        _merge_env_diagnostics(result, env)
        _close_env_timed(env, timings)
        if diagnostic_video is not None:
            result["diagnostic_video"] = diagnostic_video
        return result


def _build_probe_job(args: argparse.Namespace) -> dict[str, Any]:
    probe_job_file = getattr(args, "probe_job_file", None)
    if probe_job_file:
        probe_job = _load_json_required(Path(probe_job_file))
        catalog = _load_yaml_required(Path(args.asset_catalog))
        return _ensure_probe_job_candidate(catalog, args, probe_job)
    probe_job = getattr(args, "probe_job", None)
    if isinstance(probe_job, dict):
        catalog = _load_yaml_required(Path(args.asset_catalog))
        return _ensure_probe_job_candidate(catalog, args, probe_job)
    catalog = _load_yaml_required(Path(args.asset_catalog))
    return _build_probe_job_from_catalog(catalog, args)


def _build_probe_job_from_catalog(catalog: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    probe_job = getattr(args, "probe_job", None)
    if isinstance(probe_job, dict):
        return _ensure_probe_job_candidate(catalog, args, probe_job)
    model = ((catalog.get("assets") or {}).get(args.asset) or {}).get("models", {}).get(str(args.model_id))
    if not model:
        raise ValueError(f"missing model in asset_catalog: {args.asset}:{args.model_id}")
    candidate_use = USE_TO_CANDIDATE.get(args.use)
    if candidate_use is None:
        raise ValueError(f"unsupported use: {args.use}")
    candidate = (model.get("candidates") or {}).get(candidate_use)
    if not candidate:
        raise ValueError(f"missing candidate {candidate_use}: {args.asset}:{args.model_id}")
    result = {
        "scope": "asset",
        "stage": "canonical_skill",
        "asset": args.asset,
        "model_id": str(args.model_id),
        "use": args.use,
        "candidate_use": candidate_use,
        "candidate": candidate,
    }
    requested_arm = getattr(args, "requested_arm", None)
    if requested_arm is not None:
        result["requested_arm"] = str(requested_arm)
    return result


def _ensure_probe_job_candidate(
    catalog: dict[str, Any], args: argparse.Namespace, probe_job: dict[str, Any]
) -> dict[str, Any]:
    """Fill immutable candidate metadata without dropping pose bindings."""

    result = copy.deepcopy(probe_job)
    requested_arm = getattr(args, "requested_arm", None)
    if requested_arm is not None:
        result["requested_arm"] = str(requested_arm)
    if isinstance(result.get("candidate"), dict):
        return result
    model = ((catalog.get("assets") or {}).get(str(args.asset)) or {}).get(
        "models", {}
    ).get(str(args.model_id))
    if not model:
        raise ValueError(f"missing model in asset_catalog: {args.asset}:{args.model_id}")
    candidate_use = str(result.get("candidate_use") or USE_TO_CANDIDATE.get(args.use) or "")
    candidate = (model.get("candidates") or {}).get(candidate_use)
    if not candidate:
        raise ValueError(f"missing candidate {candidate_use}: {args.asset}:{args.model_id}")
    result["candidate_use"] = candidate_use
    result["candidate"] = candidate
    return result


def _build_runtime_args(task_config: str, run_dir: str) -> dict[str, Any]:
    config_path = Path("task_config") / f"{task_config}.yml"
    with config_path.open("r", encoding="utf-8") as f:
        runtime_args = yaml.safe_load(f)
    if not isinstance(runtime_args, dict):
        raise ValueError(f"expected mapping in {config_path}")
    embodiment_type = runtime_args.get("embodiment")
    with Path("task_config/_embodiment_config.yml").open("r", encoding="utf-8") as f:
        embodiment_types = yaml.safe_load(f)
    if len(embodiment_type) != 1:
        raise ValueError("asset probes require a single dual-arm embodiment")
    robot_file = str(embodiment_types[embodiment_type[0]]["file_path"])
    embodiment_config = _load_embodiment_config(robot_file)
    runtime_robot_file = _prepare_probe_embodiment_runtime(robot_file, Path(run_dir))
    runtime_args["left_robot_file"] = runtime_robot_file
    runtime_args["right_robot_file"] = runtime_robot_file
    runtime_args["left_embodiment_config"] = copy.deepcopy(embodiment_config)
    runtime_args["right_embodiment_config"] = copy.deepcopy(embodiment_config)
    runtime_args["dual_arm_embodied"] = True
    runtime_args["embodiment_name"] = str(embodiment_type[0])
    runtime_args["task_config"] = task_config
    runtime_args["task_name"] = "if_asset_probe"
    runtime_args["save_path"] = str(Path(run_dir) / "sim_data")
    runtime_args["save_data"] = False
    runtime_args["collect_data"] = False
    runtime_args["eval_mode"] = False
    runtime_args["eval_video_log"] = False
    runtime_args["save_video"] = False
    runtime_args["render_freq"] = 0
    runtime_args["save_freq"] = None
    runtime_args["need_plan"] = True
    runtime_args["skip_instructions"] = True
    return runtime_args


def _load_embodiment_config(robot_file: str) -> dict[str, Any]:
    config_path = Path(robot_file) / "config.yml"
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected mapping in {config_path}")
    return data


def _prepare_probe_embodiment_runtime(robot_file: str, run_dir: Path) -> str:
    source_dir = Path(robot_file)
    if not source_dir.is_absolute():
        source_dir = (Path.cwd() / source_dir).resolve()
    template_paths = sorted(source_dir.glob("*_tmp.yml"))
    if not template_paths:
        return str(source_dir)

    runtime_dir = run_dir / "runtime_embodiments" / source_dir.name
    runtime_dir.mkdir(parents=True, exist_ok=True)
    _ensure_probe_symlink(runtime_dir / "config.yml", source_dir / "config.yml")
    for child_name in ("meshes", "srdf", "urdf"):
        child = source_dir / child_name
        if child.exists():
            _ensure_probe_symlink(runtime_dir / child_name, child)

    repo_root = Path.cwd().resolve()
    for template_path in template_paths:
        target_path = runtime_dir / template_path.name.replace("_tmp.yml", ".yml")
        content = template_path.read_text(encoding="utf-8")
        content = content.replace("${ASSETS_PATH}", str(repo_root))
        content = content.replace("$ASSETS_PATH", str(repo_root))
        target_path.write_text(content, encoding="utf-8")
    return str(runtime_dir)


def _ensure_probe_symlink(link_path: Path, target_path: Path) -> None:
    target = target_path.resolve()
    if link_path.is_symlink():
        if link_path.resolve() == target:
            return
        link_path.unlink()
    elif link_path.exists():
        # ``is_symlink`` and ``exists`` are separate filesystem reads.  A
        # parallel worker can create the expected link between them.
        if link_path.is_symlink() and link_path.resolve() == target:
            return
        raise ValueError(f"probe runtime path already exists and is not a symlink: {link_path}")
    try:
        link_path.symlink_to(target, target_is_directory=target.is_dir())
    except FileExistsError:
        # Parallel GPU workers prepare the same run-local embodiment directory.
        # Another worker may create the identical link after the checks above.
        if link_path.is_symlink() and link_path.resolve() == target:
            return
        raise


def _failure(
    args: argparse.Namespace,
    reason: str,
    exc: Exception,
    *,
    timings: dict[str, Any] | None = None,
    total_started: float | None = None,
) -> dict[str, Any]:
    timings = timings or {}
    if total_started is not None:
        _record_total(timings, total_started)
    return {
        "attempt_id": getattr(args, "attempt_id", None),
        "job_id": getattr(args, "job_id", None),
        "job_index": getattr(args, "job_index", None),
        "asset": args.asset,
        "model_id": str(args.model_id),
        "use": args.use,
        "seed": int(args.seed),
        "plan_success": False,
        "success": False,
        "failure_reason": reason,
        "exception_type": type(exc).__name__,
        "exception": str(exc),
        "traceback_tail": traceback.format_exc()[-4000:],
        "timings": timings,
        "duration_sec": timings.get("total_worker_sec"),
        "warmup": bool(getattr(args, "warmup", False)),
        "execution_policy_manifest": m4_worker_execution_policy_manifest(),
    }


def _close_env(env: Any) -> None:
    try:
        env.close_env()
    except Exception:
        pass


def _close_env_timed(env: Any, timings: dict[str, Any]) -> None:
    started = time.monotonic()
    _close_env(env)
    _record_timing(timings, "close_env_sec", started)


def _reset_reused_planner_state(env: Any, timings: dict[str, Any], *, enabled: bool) -> None:
    if not enabled:
        return
    started = time.monotonic()
    reset_count = 0
    errors: list[str] = []
    robot = getattr(env, "robot", None)
    for planner_name in ("left_planner", "right_planner"):
        planner = getattr(robot, planner_name, None)
        if planner is None:
            continue
        for motion_gen_name in ("motion_gen", "motion_gen_batch"):
            motion_gen = getattr(planner, motion_gen_name, None)
            reset = getattr(motion_gen, "reset", None)
            if not callable(reset):
                continue
            try:
                reset(reset_seed=True)
            except TypeError:
                try:
                    reset()
                except Exception as exc:
                    errors.append(f"{planner_name}.{motion_gen_name}:{type(exc).__name__}:{exc}")
                    continue
            except Exception as exc:
                errors.append(f"{planner_name}.{motion_gen_name}:{type(exc).__name__}:{exc}")
                continue
            reset_count += 1
    timings["planner_reset_count"] = reset_count
    if errors:
        timings["planner_reset_errors"] = errors
    _record_timing(timings, "planner_reset_sec", started)


def _merge_env_timings(timings: dict[str, Any], env: Any) -> None:
    env_timings = getattr(env, "probe_timings", None)
    if not isinstance(env_timings, dict):
        return
    for key, value in env_timings.items():
        timings[str(key)] = _jsonable(value)


def _merge_env_diagnostics(result: dict[str, Any], env: Any) -> None:
    diagnostics = getattr(env, "probe_diagnostics", None)
    if isinstance(diagnostics, dict) and diagnostics:
        result["diagnostics"] = _jsonable(diagnostics)
        placement = diagnostics.get("canonical_placement_witness")
        if isinstance(placement, dict):
            # Attempt-owned evidence: the runner may validate and aggregate
            # this value but must never synthesize or repair it.
            result["canonical_placement_witness"] = _jsonable(placement)
        articulation = diagnostics.get("articulation_action")
        if isinstance(articulation, dict):
            for key in (
                "execution_profile_version",
                "execution_profile_hash",
                "execution_profile_source_hash",
                "motion_policy_version",
                "motion_policy_hash",
                "policy_version",
            ):
                if articulation.get(key) is not None:
                    result[f"articulation_{key}"] = _jsonable(
                        articulation[key]
                    )


def _articulation_failure_code(env: Any) -> str | None:
    diagnostics = getattr(env, "probe_diagnostics", None)
    if not isinstance(diagnostics, dict):
        return None
    action = diagnostics.get("articulation_action")
    if not isinstance(action, dict):
        return None
    direct_code = str(action.get("failure_code") or "")
    if direct_code.startswith("articulation_"):
        return direct_code
    attempts = action.get("arm_attempts") or []
    if not attempts:
        return None
    result = attempts[-1].get("result") or {}
    code = str(result.get("failure_code") or "")
    return code if code.startswith("articulation_") else None


def _record_timing(timings: dict[str, Any], key: str, started: float) -> None:
    timings[key] = round(float(time.monotonic() - started), 12)


def _record_total(timings: dict[str, Any], started: float) -> None:
    timings["total_worker_sec"] = round(float(time.monotonic() - started), 12)


def _load_yaml_required(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected YAML mapping in {path}")
    return data


def _load_json_required(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON mapping in {path}")
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Milestone 4 canonical asset probe attempts.")
    parser.add_argument("--asset-catalog", required=True)
    parser.add_argument("--asset")
    parser.add_argument("--model-id")
    parser.add_argument("--use")
    parser.add_argument("--requested-arm", choices=("left", "right"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--run-dir", default="if_benchmark/data/asset_validation/runs")
    parser.add_argument("--batch-file")
    parser.add_argument("--batch-output")
    parser.add_argument("--reuse-task-env", action="store_true")
    parser.add_argument("--probe-job-file")
    parser.add_argument("--diagnostic-video-dir")
    args = parser.parse_args(argv)
    os.makedirs(args.run_dir, exist_ok=True)
    if args.batch_file:
        results = run_batch(args)
        print(
            json.dumps(
                {
                    "batch_output": args.batch_output,
                    "batch_result_count": len(results),
                    "reuse_task_env": bool(args.reuse_task_env),
                },
                sort_keys=True,
                ensure_ascii=False,
            )
        )
        return 0
    missing = [name for name in ("asset", "model_id", "use", "seed") if getattr(args, name) is None]
    if missing:
        parser.error(f"missing required one-shot arguments: {', '.join(missing)}")
    result = run_worker(args)
    print(json.dumps(_jsonable(result), sort_keys=True, ensure_ascii=False))
    reason = str(result.get("failure_reason") or "")
    return 0 if result.get("success") or reason in {"planner_failed", "checker_false"} or reason.startswith("articulation_") else 1


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return _jsonable(value.tolist())
        except Exception:
            pass
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
