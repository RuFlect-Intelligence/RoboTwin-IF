"""Deterministic runner for the post-approval Initial Pose support sweep."""

from __future__ import annotations

from pathlib import Path
import traceback
from typing import Any, Mapping

from if_benchmark.asset_probe_worker import _build_runtime_args
from if_benchmark.schema import artifact_hash

from .core import (
    InitialPoseContractError,
    _load_mapping,
    atomic_json,
    validate_support_report,
)


def run_support_plan(
    *,
    plan_path: Path,
    reports_dir: Path,
    runtime_dir: Path,
    task_config: str = "demo_clean",
    state_ids: set[str] | None = None,
    limit: int | None = None,
    skip_existing: bool = False,
    shard_index: int = 0,
    shard_count: int = 1,
) -> dict[str, Any]:
    plan = _load_mapping(plan_path)
    if plan.get("artifact_kind") != "initial_pose_support_probe_plan":
        raise InitialPoseContractError("unexpected Initial Pose support plan kind")
    if plan.get("plan_hash") != artifact_hash(plan, omit_keys=("plan_hash",)):
        raise InitialPoseContractError("Initial Pose support plan self-hash mismatch")
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise InitialPoseContractError("invalid support shard index/count")
    jobs = [
        dict(row)
        for index, row in enumerate(plan.get("jobs") or [])
        if index % shard_count == shard_index
        if state_ids is None or str(row.get("initial_state_id")) in state_ids
    ]
    if limit is not None:
        jobs = jobs[: max(0, int(limit))]
    reports_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    counts = {"pass": 0, "fail": 0, "source_failure": 0, "internal_failure": 0, "skipped": 0}
    for job in jobs:
        output_path = reports_dir / f"{job['job_id']}.json"
        if skip_existing and output_path.is_file():
            try:
                existing_report = _load_mapping(output_path)
                validate_support_report(
                    existing_report,
                    job,
                    expected_plan_hash=str(plan["plan_hash"]),
                )
                # An internal failure is an execution failure, not immutable
                # physical evidence.  Retrying a support run must therefore
                # replace it instead of treating it as a completed job.
                if existing_report.get("outcome") != "internal_failure":
                    counts["skipped"] += 1
                    continue
            except InitialPoseContractError:
                pass
        report = run_support_job(
            job=job,
            plan_hash=str(plan["plan_hash"]),
            runtime_dir=runtime_dir / str(job["job_id"]),
            task_config=task_config,
        )
        atomic_json(output_path, report)
        counts[str(report["outcome"])] += 1
    return {"selected_job_count": len(jobs), **counts}


def run_support_job(
    *,
    job: Mapping[str, Any],
    plan_hash: str,
    runtime_dir: Path,
    task_config: str,
) -> dict[str, Any]:
    env = None
    outcome = "internal_failure"
    error: dict[str, Any] | None = None
    try:
        from envs.if_initial_pose_support import if_initial_pose_support

        runtime_args = _build_runtime_args(task_config, str(runtime_dir))
        runtime_args["task_name"] = "if_initial_pose_support"
        runtime_args["need_plan"] = False
        env = if_initial_pose_support()
        try:
            env.setup_demo(
                now_ep_num=0,
                seed=int(job["seed"]),
                support_job=dict(job),
                **runtime_args,
            )
            outcome = "pass"
        except Exception as exc:
            metrics = getattr(env, "support_metrics", {}) or {}
            if metrics.get("failed_checks"):
                outcome = "fail"
            elif isinstance(exc, InitialPoseContractError) or "source_initial_pose" in str(exc):
                outcome = "source_failure"
            else:
                outcome = "internal_failure"
            error = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback_tail": traceback.format_exc()[-4000:],
            }
        metrics = dict(getattr(env, "support_metrics", {}) or {})
        trajectory = list(getattr(env, "support_trajectory", []) or [])
        preflight = dict(getattr(env, "preflight", {}) or {})
    except Exception as exc:
        metrics = {}
        trajectory = []
        preflight = {}
        outcome = (
            "source_failure"
            if isinstance(exc, InitialPoseContractError)
            or "source_initial_pose" in str(exc)
            else "internal_failure"
        )
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback_tail": traceback.format_exc()[-4000:],
        }
    finally:
        if env is not None:
            try:
                env.close_env()
            except Exception:
                pass
    report: dict[str, Any] = {
        "artifact_kind": "initial_pose_support_report",
        "schema_version": 2,
        "job_id": job["job_id"],
        "initial_state_id": job["initial_state_id"],
        "revision": job["revision"],
        "approval_hash": job["approval_hash"],
        "geometry_hash": job["geometry_hash"],
        "support_contract_version": job["support_contract_version"],
        "support_config_hash": job["support_config_hash"],
        "support_mode": job["support_mode"],
        "source_plan_hash": plan_hash,
        "asset": job["asset"],
        "model_id": str(job["model_id"]),
        "initial_state_profile_id": job["initial_state_profile_id"],
        "yaw_deg": float(job["yaw_deg"]),
        "seed": int(job["seed"]),
        "fallback_pose_used": False,
        "preflight": preflight,
        "metrics": metrics,
        "trajectory": trajectory,
        "outcome": outcome,
        "error": error,
    }
    report["report_hash"] = artifact_hash(report)
    return report
