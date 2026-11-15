"""Fresh canonical asset-skill probe runner for Milestone 4.

The runner consumes ``probe_plan.yml`` jobs and launches fresh worker attempts.
It can run one isolated worker process per attempt, or batch several attempts
inside one worker process for import/planner reuse.  Each attempt still calls
``setup_demo -> play_once -> check_success -> close_env`` and aggregation writes
one normalized ``canonical_skill_asset_report``; legacy smoke summaries are not
read here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml

from if_benchmark.asset_semantic_exclusions import (
    DEFAULT_SEMANTIC_EXCLUSIONS_PATH,
    expand_asset_semantic_exclusions,
    load_asset_semantic_exclusions,
    semantic_exclusion_identity,
)
from if_benchmark.asset_probe_evidence import (
    m4_attempt_is_blocking,
    m4_attempt_is_retryable,
    m4_report_attempt_accounting_errors,
)
from if_benchmark.asset_validation_contract import (
    ASSET_VALIDATION_REPORT_SCHEMA_VERSION,
    ASSET_VALIDATION_SCHEMA_VERSION,
    M4_DISABLED_USES,
    M4_EXECUTABLE_USES,
)
from if_benchmark.execution_policy import (
    m4_worker_execution_policy_manifest,
    m4_worker_execution_policy_manifest_errors,
)
from if_benchmark.operation_placement import (
    M4_TOUCH_PLACEMENT_POLICY_HASH,
    M4_TOUCH_PLACEMENT_POLICY_VERSION,
    PICK_OPERATION_PLACEMENT_POLICY_HASH,
    PICK_OPERATION_PLACEMENT_POLICY_VERSION,
    m4_canonical_placement_witness_errors,
)
from if_benchmark.grasp_validation.contact_contract import (
    resolved_contact_preflight_policy_version,
)
from if_benchmark.success_contract import SuccessContract
from if_benchmark.articulation_capability import (
    ARTICULATION_ARM_SAMPLING_VERSION,
    ARTICULATION_REQUESTED_ARMS,
    aggregate_articulation_capabilities,
)
from if_benchmark.touch_capability import (
    TOUCH_ARM_SAMPLING_VERSION,
    TOUCH_CAPABILITY_VERSION,
    TOUCH_EXECUTABLE_USES,
    TOUCH_REQUESTED_ARMS,
    aggregate_touch_capabilities,
)
from if_benchmark.scene_validation.articulation_execution import (
    load_articulation_execution_catalog,
    resolve_articulation_execution_profile,
)
# Schema 3 requires runtime-backed per-arm capability rows. The M4 projection
# loader still accepts schema 2 only through its strict legacy report rebind,
# so new runs must never identify themselves as legacy.
REPORT_SCHEMA_VERSION = ASSET_VALIDATION_REPORT_SCHEMA_VERSION
SCHEMA_VERSION = ASSET_VALIDATION_SCHEMA_VERSION
STAGE_CANONICAL = "canonical_skill"
FRESH_CANONICAL_SOURCE_KIND = "m4_fresh_canonical_skill_probe"
BASE_EXECUTABLE_USES = (
    "pick_left",
    "pick_right",
    "open",
    "close",
    "switch_on",
    "switch_off",
)
ADDITIONAL_EXECUTABLE_USES = tuple(
    use
    for use in (
        "click_subject",
        "press_subject",
        "rotate_to_orientation_subject",
        "adjust_orientation_subject",
        "bimanual_grab_subject",
    )
    if use in M4_EXECUTABLE_USES
)
EXECUTABLE_USES = M4_EXECUTABLE_USES
M4_NON_EVALUATED_USES = {
    "handover": "m4_capability_explicitly_not_evaluated",
    **{
        use: "m4_capability_explicitly_disabled"
        for use in M4_DISABLED_USES
    },
}
ARTICULATION_EXECUTABLE_USES = {"open", "close", "switch_on", "switch_off"}
ARTICULATION_PROVENANCE_FIELDS = (
    "articulation_execution_profile_version",
    "articulation_execution_profile_hash",
    "articulation_execution_profile_source_hash",
    "articulation_motion_policy_version",
    "articulation_motion_policy_hash",
)
DEFAULT_PROBE_SEEDS = (0, 1, 2)
DEFAULT_PROBE_THRESHOLDS = {use: 0.6 for use in M4_EXECUTABLE_USES}
DEFAULT_TIMEOUT_SECONDS = 240
DEFAULT_WARMUP_TIMEOUT_SECONDS = 1800
DEFAULT_BATCH_SIZE = 1
EQUIVALENCE_REPORT_FILE = "probe_equivalence_report.yml"
DEFAULT_ASSET_CATALOG_PATH = Path("if_benchmark/assets/asset_catalog.yml")
M4_RESUME_CONFIG_VERSION = "m4_probe_resume_config_v1"
M4_CHECKPOINT_INDEX_VERSION = "m4_probe_checkpoint_index_v1"
M4_GPU_QUEUE_POLICY_VERSION = "m4_fixed_gpu_slot_queue_v1"
M4_COST_BALANCE_POLICY_VERSION = "m4_cost_balanced_batches_v1"
M4_RESUME_SOURCE_MIGRATIONS_PATH = (
    Path(__file__).with_name("assets") / "m4_resume_source_migrations.yml"
)


def _requested_arms_for_job(job: dict[str, Any]) -> tuple[str | None, ...]:
    use = str(job.get("use") or "")
    if use in TOUCH_EXECUTABLE_USES:
        arms = tuple(str(arm) for arm in job.get("requested_arms") or ())
        if arms != TOUCH_REQUESTED_ARMS:
            raise ValueError(
                f"touch job must request {list(TOUCH_REQUESTED_ARMS)}: "
                f"{job.get('job_id')}"
            )
        if job.get("touch_arm_sampling_version") != (
            TOUCH_ARM_SAMPLING_VERSION
        ):
            raise ValueError(
                f"touch job sampling contract is stale: {job.get('job_id')}"
            )
        if (
            job.get("touch_placement_policy_version")
            != M4_TOUCH_PLACEMENT_POLICY_VERSION
            or job.get("touch_placement_policy_hash")
            != M4_TOUCH_PLACEMENT_POLICY_HASH
        ):
            raise ValueError(
                f"touch job placement contract is stale: {job.get('job_id')}"
            )
        return tuple(arms)
    if use not in ARTICULATION_EXECUTABLE_USES:
        return (None,)
    arms = tuple(str(arm) for arm in job.get("requested_arms") or ())
    if arms != ARTICULATION_REQUESTED_ARMS:
        raise ValueError(
            f"articulation job must request {list(ARTICULATION_REQUESTED_ARMS)}: "
            f"{job.get('job_id')}"
        )
    if job.get("articulation_arm_sampling_version") != (
        ARTICULATION_ARM_SAMPLING_VERSION
    ):
        raise ValueError(
            f"articulation job sampling contract is stale: {job.get('job_id')}"
        )
    return tuple(arms)


def _expected_attempt_count(job: dict[str, Any], seeds: list[int]) -> int:
    return len(seeds) * len(_requested_arms_for_job(job))


def run_probe_plan(
    *,
    asset_catalog_path: Path,
    plan_path: Path,
    reports_dir: Path,
    run_dir: Path,
    python_executable: str | None = None,
    task_config: str = "demo_clean",
    seeds: list[int] | None = None,
    thresholds: dict[str, float] | None = None,
    uses: set[str] | None = None,
    start_index: int = 0,
    max_jobs: int | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    warmup_timeout_seconds: int = DEFAULT_WARMUP_TIMEOUT_SECONDS,
    torch_extensions_dir: Path | None = None,
    clear_existing_reports: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
    reuse_task_env: bool = False,
    parallel_workers: int = 1,
    cuda_devices: list[str] | None = None,
    source_kind: str = FRESH_CANONICAL_SOURCE_KIND,
    resume: bool = False,
    job_cost_history_path: Path | None = None,
) -> dict[str, Any]:
    """Run fresh canonical probes and write one normalized report."""

    plan = _load_yaml_required(plan_path)
    if plan.get("probe_plan_hash") != _artifact_hash(plan, "probe_plan_hash"):
        raise ValueError("probe plan self-hash mismatch")
    _validate_pick_operation_placement_binding(plan)
    if not plan.get("source_validated_pose_set_hash"):
        raise ValueError("formal M4 probe plan is missing Initial Pose Set binding")
    if not plan.get("initial_state_selection_hash"):
        raise ValueError("formal M4 probe plan is missing state selection binding")
    all_jobs = plan.get("jobs") or []
    forbidden_policy_jobs = [
        str(job.get("job_id"))
        for job in all_jobs
        if str(job.get("use")) in M4_NON_EVALUATED_USES
    ]
    if forbidden_policy_jobs:
        raise ValueError(
            "probe plan contains disabled/non-evaluated M4 jobs; regenerate "
            "the plan: " + ", ".join(forbidden_policy_jobs)
        )
    semantic_policy = _validate_semantic_exclusion_binding(
        plan, asset_catalog_path
    )
    if semantic_policy is not None:
        excluded = set(expand_asset_semantic_exclusions(semantic_policy))
        forbidden_jobs = [
            str(job.get("job_id"))
            for job in all_jobs
            if (
                str(job.get("asset")),
                str(job.get("model_id")),
                str(job.get("use")),
            )
            in excluded
        ]
        if forbidden_jobs:
            raise ValueError(
                "probe plan contains semantically unsupported jobs: "
                + ", ".join(forbidden_jobs)
            )
    for job in all_jobs:
        if not isinstance(job.get("initial_pose_binding"), dict) or not isinstance(
            job.get("canonical_initial_state"), dict
        ):
            raise ValueError(
                f"M4 job lacks approved Initial Pose binding: {job.get('job_id')}"
            )
        if job.get("initial_state_selection_hash") != plan.get(
            "initial_state_selection_hash"
        ):
            raise ValueError(
                f"M4 job state selection mismatch: {job.get('job_id')}"
            )
        _requested_arms_for_job(job)
    selected_jobs = [
        dict(job)
        for job in _select_jobs(
            all_jobs, uses=uses, start_index=start_index, max_jobs=max_jobs
        )
    ]
    plan_job_indices = _job_indices(all_jobs)
    selected_job_indices = {
        str(job["job_id"]): plan_job_indices[str(job["job_id"])]
        for job in selected_jobs
    }
    probe_seeds = _validated_probe_seeds(
        list(DEFAULT_PROBE_SEEDS if seeds is None else seeds)
    )
    probe_thresholds = _validated_probe_thresholds(
        {**DEFAULT_PROBE_THRESHOLDS, **(thresholds or {})}
    )
    python = python_executable or sys.executable
    # Resolve every selected articulation profile before creating run files or
    # launching GPU workers.  A stale Success Contract/profile binding is a
    # source error, not a planner result, and must never be discovered only
    # after a full sweep reaches ``freeze``.
    articulation_catalog = load_articulation_execution_catalog()
    success_contract = SuccessContract.load()
    _bind_articulation_profile_provenance(
        selected_jobs,
        catalog=articulation_catalog,
        contract=success_contract,
    )
    # Leave TORCH_EXTENSIONS_DIR unset by default so PyTorch can reuse its
    # versioned user-level cache across independent benchmark runs.  A
    # run-local cache made every new sweep recompile the same CuRobo kernels
    # and routinely consumed the entire warmup timeout.  Explicit callers can
    # still request isolation with ``torch_extensions_dir``.
    torch_cache = (
        Path(torch_extensions_dir) if torch_extensions_dir is not None else None
    )
    worker_count = max(1, int(parallel_workers))
    device_slots = _worker_device_slots(cuda_devices, worker_count)
    cuda_devices = list(
        dict.fromkeys(
            str(device) for device in device_slots if device is not None
        )
    ) or None
    effective_batch_size = max(1, int(batch_size))
    asset_catalog_sha256 = _sha256_file(asset_catalog_path)
    job_costs, cost_history = _resolve_job_costs(
        selected_jobs,
        seeds=probe_seeds,
        history_path=job_cost_history_path,
    )
    resume_config = _build_resume_config(
        plan=plan,
        selected_jobs=selected_jobs,
        selected_job_indices=selected_job_indices,
        asset_catalog_sha256=asset_catalog_sha256,
        python_executable=python,
        task_config=task_config,
        seeds=probe_seeds,
        thresholds=probe_thresholds,
        timeout_seconds=int(timeout_seconds),
        warmup_timeout_seconds=int(warmup_timeout_seconds),
        torch_extensions_dir=torch_cache,
        batch_size=effective_batch_size,
        reuse_task_env=bool(reuse_task_env),
        parallel_workers=worker_count,
        cuda_devices=cuda_devices,
        device_slots=device_slots,
        source_kind=source_kind,
        success_contract=success_contract,
        articulation_catalog=articulation_catalog,
        job_costs=job_costs,
        cost_history=cost_history,
    )
    resume_config_hash = str(resume_config["resume_config_hash"])

    reports_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    if torch_cache is not None:
        torch_cache.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / "checkpoints"
    attempts_checkpoint = checkpoint_dir / "attempts.jsonl"
    records_checkpoint = checkpoint_dir / "records.jsonl"
    timing_summary_path = run_dir / "timing_summary.yml"
    metadata_path = checkpoint_dir / "run_metadata.json"
    checkpoint_index_path = checkpoint_dir / "checkpoint_index.json"
    prior_run_metadata: dict[str, Any] | None = None
    if resume:
        if clear_existing_reports:
            raise ValueError("M4 --resume cannot be combined with clear_existing_reports")
        prior_run_metadata = _load_json_required(metadata_path)
        source_migration = _validate_resume_metadata(
            prior_run_metadata, resume_config
        )
        if (
            not attempts_checkpoint.is_file()
            or not records_checkpoint.is_file()
            or not checkpoint_index_path.is_file()
        ):
            raise ValueError(
                "M4 resume requires attempts.jsonl, records.jsonl, and "
                "checkpoint_index.json"
            )
        if source_migration is not None:
            _migrate_resume_source_revision(
                metadata=prior_run_metadata,
                metadata_path=metadata_path,
                expected_config=resume_config,
                checkpoint_index_path=checkpoint_index_path,
                attempts_checkpoint=attempts_checkpoint,
                records_checkpoint=records_checkpoint,
                reports_dir=reports_dir,
                migration=source_migration,
            )
        _validate_resume_reports_directory(
            reports_dir,
            expected_resume_config_hash=resume_config_hash,
        )
        checkpoint_index = _load_checkpoint_index(
            checkpoint_index_path,
            resume_config_hash=resume_config_hash,
        )
        retryable_checkpoint_count = sum(
            1
            for row in _committed_checkpoint_rows(
                attempts_checkpoint,
                row_kind="attempt",
                committed_hashes=checkpoint_index["attempts"],
            )
            if m4_attempt_is_retryable(row)
        )
        if retryable_checkpoint_count:
            _archive_retryable_resume_reports(reports_dir)
            completed = None
        else:
            completed = _completed_report_result(
                prior_run_metadata,
                expected_resume_config_hash=resume_config_hash,
                expected_reports_dir=reports_dir,
            )
            if completed is None:
                discovered = _discover_completed_report_result(
                    reports_dir,
                    expected_resume_config_hash=resume_config_hash,
                )
                if discovered is not None:
                    completed, discovered_path, discovered_hash = discovered
                    prior_run_metadata.update(
                        {
                            "completed_report_path": _path_string(discovered_path),
                            "completed_report_hash": discovered_hash,
                            "completed_at": time.strftime(
                                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                            ),
                        }
                    )
                    _write_json_atomic(metadata_path, prior_run_metadata)
        if completed is not None:
            return completed
    else:
        _clear_batch_artifacts(run_dir)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if not resume:
        for checkpoint_path in (attempts_checkpoint, records_checkpoint):
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            checkpoint_path.touch()
        _commit_checkpoint_index(
            checkpoint_index_path,
            _empty_checkpoint_index(resume_config_hash),
            checkpoint_paths=(attempts_checkpoint, records_checkpoint),
        )
    if clear_existing_reports:
        for report_path in sorted(reports_dir.glob("*.json")):
            report_path.unlink()

    now_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    started_at = str((prior_run_metadata or {}).get("started_at") or now_utc)
    resume_count = int((prior_run_metadata or {}).get("resume_count") or 0)
    if resume:
        resume_count += 1
    run_metadata = {
        "schema_version": 2,
        "source_kind": source_kind,
        "source_probe_plan_path": _path_string(plan_path),
        "source_probe_plan_hash": plan.get("probe_plan_hash"),
        "pick_operation_placement_policy_version": plan.get(
            "pick_operation_placement_policy_version"
        ),
        "pick_operation_placement_policy_hash": plan.get(
            "pick_operation_placement_policy_hash"
        ),
        "worker_execution_policy_manifest": m4_worker_execution_policy_manifest(),
        "source_validated_pose_set_hash": plan.get("source_validated_pose_set_hash"),
        "initial_state_selection_version": plan.get("initial_state_selection_version"),
        "initial_state_selection_hash": plan.get("initial_state_selection_hash"),
        "source_asset_catalog_path": _path_string(asset_catalog_path),
        "source_asset_catalog_sha256": asset_catalog_sha256,
        "selected_job_count": len(selected_jobs),
        "selected_job_ids": [str(job["job_id"]) for job in selected_jobs],
        "seeds": probe_seeds,
        "thresholds": probe_thresholds,
        "timeout_seconds": timeout_seconds,
        "warmup_timeout_seconds": warmup_timeout_seconds,
        "torch_extensions_dir": _path_string(torch_cache),
        "batch_size": effective_batch_size,
        "reuse_task_env": bool(reuse_task_env),
        "parallel_workers": worker_count,
        "cuda_devices": list(cuda_devices or []),
        "device_slots": device_slots,
        "gpu_queue_policy_version": M4_GPU_QUEUE_POLICY_VERSION,
        "cost_balance_policy_version": M4_COST_BALANCE_POLICY_VERSION,
        "job_cost_history": cost_history,
        "job_costs_hash": _artifact_hash({"job_costs": job_costs}, "job_costs_hash"),
        "resume_config": resume_config,
        "resume_config_hash": resume_config_hash,
        "checkpoint_index_path": _path_string(checkpoint_index_path),
        "started_at": started_at,
        "resume": bool(resume),
        "resume_count": resume_count,
        "resumed_at": now_utc if resume else None,
        "resume_source_migrations": list(
            (prior_run_metadata or {}).get("resume_source_migrations") or []
        ),
    }
    _write_json_atomic(metadata_path, run_metadata)
    if (
        effective_batch_size == 1
        and not reuse_task_env
        and worker_count == 1
        and not resume
    ):
        batch_result = _run_isolated_attempts_fresh(
            python_executable=python,
            asset_catalog_path=asset_catalog_path,
            jobs=selected_jobs,
            job_indices=selected_job_indices,
            seeds=probe_seeds,
            thresholds=probe_thresholds,
            task_config=task_config,
            run_dir=run_dir,
            timeout_seconds=timeout_seconds,
            warmup_timeout_seconds=warmup_timeout_seconds,
            torch_extensions_dir=torch_cache,
            attempts_checkpoint=attempts_checkpoint,
            records_checkpoint=records_checkpoint,
            resume_config_hash=resume_config_hash,
        )
    else:
        batch_result = _run_attempt_batches(
            python_executable=python,
            asset_catalog_path=asset_catalog_path,
            jobs=selected_jobs,
            job_indices=selected_job_indices,
            seeds=probe_seeds,
            thresholds=probe_thresholds,
            task_config=task_config,
            run_dir=run_dir,
            timeout_seconds=timeout_seconds,
            warmup_timeout_seconds=warmup_timeout_seconds,
            torch_extensions_dir=torch_cache,
            batch_size=effective_batch_size,
            reuse_task_env=reuse_task_env,
            parallel_workers=worker_count,
            device_slots=device_slots,
            attempts_checkpoint=attempts_checkpoint,
            records_checkpoint=records_checkpoint,
            resume=resume,
            resume_config_hash=resume_config_hash,
            job_costs=job_costs,
        )
    records = batch_result["records"]
    attempt_rows = batch_result["attempts"]
    accounting_errors = m4_report_attempt_accounting_errors(
        records,
        attempt_rows,
        require_seed_coverage=True,
    )
    if accounting_errors:
        raise ValueError(
            "M4 runner produced invalid attempt/record accounting: "
            + ", ".join(accounting_errors[:8])
        )

    timing_summary = _write_timing_summary(
        timing_summary_path,
        attempts=attempt_rows,
        records=records,
        batch_size=max(1, int(batch_size)),
        reuse_task_env=reuse_task_env,
        parallel_workers=worker_count,
        cuda_devices=cuda_devices,
        source_kind=source_kind,
    )

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "kind": "canonical_skill_asset_report",
        "scope": "asset",
        "stage": STAGE_CANONICAL,
        "source_kind": source_kind,
        "source_probe_plan_path": _path_string(plan_path),
        "source_probe_plan_hash": plan.get("probe_plan_hash"),
        "pick_operation_placement_policy_version": plan.get(
            "pick_operation_placement_policy_version"
        ),
        "pick_operation_placement_policy_hash": plan.get(
            "pick_operation_placement_policy_hash"
        ),
        "source_validated_pose_set_hash": plan.get("source_validated_pose_set_hash"),
        "source_initial_state_selection_path": plan.get(
            "source_initial_state_selection_path"
        ),
        "initial_state_selection_version": plan.get(
            "initial_state_selection_version"
        ),
        "initial_state_selection_hash": plan.get("initial_state_selection_hash"),
        "source_asset_catalog_path": _path_string(asset_catalog_path),
        "source_asset_catalog_sha256": asset_catalog_sha256,
        "semantic_exclusion_policy": plan.get(
            "semantic_exclusion_policy"
        ),
        **success_contract.identity,
        "worker_execution_policy_manifest": (
            m4_worker_execution_policy_manifest()
        ),
        "articulation_execution_catalog_hash": articulation_catalog[
            "profile_hash"
        ],
        "articulation_arm_sampling_version": (
            ARTICULATION_ARM_SAMPLING_VERSION
        ),
        "touch_arm_sampling_version": TOUCH_ARM_SAMPLING_VERSION,
        "touch_placement_policy_version": M4_TOUCH_PLACEMENT_POLICY_VERSION,
        "touch_placement_policy_hash": M4_TOUCH_PLACEMENT_POLICY_HASH,
        "runner": {
            "python_executable": python,
            "task_config": task_config,
            "seeds": probe_seeds,
            "thresholds": probe_thresholds,
            "timeout_seconds": timeout_seconds,
            "warmup_timeout_seconds": warmup_timeout_seconds,
            "torch_extensions_dir": _path_string(torch_cache),
            "batch_size": int(batch_size),
            "reuse_task_env": bool(reuse_task_env),
            "parallel_workers": worker_count,
            "cuda_devices": list(cuda_devices or []),
            "device_slots": device_slots,
            "gpu_queue_policy_version": M4_GPU_QUEUE_POLICY_VERSION,
            "cost_balance_policy_version": M4_COST_BALANCE_POLICY_VERSION,
            "job_cost_history": cost_history,
            "job_costs_hash": run_metadata["job_costs_hash"],
            "resume": bool(resume),
            "resume_count": resume_count,
            "resume_config_hash": resume_config_hash,
            "resume_summary": batch_result["resume_summary"],
            "checkpoint_index_path": run_metadata["checkpoint_index_path"],
            "timing_summary_path": _path_string(timing_summary_path),
            "started_at": started_at,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "summary": _summary(records, attempt_rows),
        "timing_summary": timing_summary,
        "records": records,
        "attempts": attempt_rows,
    }
    report["evidence_id"] = _evidence_id(report)
    report["report_hash"] = _artifact_hash(report, "report_hash")
    report_path = reports_dir / f"{report['evidence_id']}.json"
    _write_json_atomic(report_path, report)
    run_metadata.update(
        {
            "completed_report_path": _path_string(report_path),
            "completed_report_hash": report["report_hash"],
            "completed_at": report["runner"]["finished_at"],
        }
    )
    _write_json_atomic(metadata_path, run_metadata)
    return {
        "evidence_id": report["evidence_id"],
        "evidence_path": _path_string(report_path),
        "record_count": len(records),
        "attempt_count": len(attempt_rows),
        "summary": report["summary"],
        "timing_summary_path": _path_string(timing_summary_path),
        "resume_summary": batch_result["resume_summary"],
    }


def run_probe_equivalence_check(
    *,
    asset_catalog_path: Path,
    plan_path: Path,
    output_dir: Path,
    python_executable: str | None = None,
    task_config: str = "demo_clean",
    seeds: list[int] | None = None,
    thresholds: dict[str, float] | None = None,
    uses: set[str] | None = None,
    start_index: int = 0,
    max_jobs: int | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    warmup_timeout_seconds: int = DEFAULT_WARMUP_TIMEOUT_SECONDS,
    torch_extensions_dir: Path | None = None,
    batch_size: int = 2,
) -> dict[str, Any]:
    """Run isolated and batch-reuse probes on the same slice and compare results."""

    root = output_dir / "probe_equivalence"
    isolated_dir = root / "isolated"
    batch_dir = root / "batch_reuse"
    isolated = run_probe_plan(
        asset_catalog_path=asset_catalog_path,
        plan_path=plan_path,
        reports_dir=isolated_dir / "reports",
        run_dir=isolated_dir / "runs",
        python_executable=python_executable,
        task_config=task_config,
        seeds=seeds,
        thresholds=thresholds,
        uses=uses,
        start_index=start_index,
        max_jobs=max_jobs,
        timeout_seconds=timeout_seconds,
        warmup_timeout_seconds=warmup_timeout_seconds,
        torch_extensions_dir=torch_extensions_dir,
        clear_existing_reports=True,
        batch_size=1,
        reuse_task_env=False,
    )
    batch = run_probe_plan(
        asset_catalog_path=asset_catalog_path,
        plan_path=plan_path,
        reports_dir=batch_dir / "reports",
        run_dir=batch_dir / "runs",
        python_executable=python_executable,
        task_config=task_config,
        seeds=seeds,
        thresholds=thresholds,
        uses=uses,
        start_index=start_index,
        max_jobs=max_jobs,
        timeout_seconds=timeout_seconds,
        warmup_timeout_seconds=warmup_timeout_seconds,
        torch_extensions_dir=torch_extensions_dir,
        clear_existing_reports=True,
        batch_size=max(2, int(batch_size)),
        reuse_task_env=True,
    )
    isolated_attempts = _read_jsonl(isolated_dir / "runs" / "checkpoints" / "attempts.jsonl")
    batch_attempts = _read_jsonl(batch_dir / "runs" / "checkpoints" / "attempts.jsonl")
    comparison = compare_attempt_equivalence(isolated_attempts, batch_attempts)
    report = {
        "schema_version": 1,
        "source_kind": FRESH_CANONICAL_SOURCE_KIND,
        "pass": not comparison["mismatches"] and not comparison["missing_in_batch"] and not comparison["missing_in_isolated"],
        "config": {
            "source_probe_plan_path": _path_string(plan_path),
            "source_asset_catalog_path": _path_string(asset_catalog_path),
            "seeds": list(DEFAULT_PROBE_SEEDS if seeds is None else seeds),
            "uses": sorted(uses) if uses else None,
            "start_index": int(start_index),
            "max_jobs": max_jobs,
            "task_config": task_config,
            "timeout_seconds": timeout_seconds,
            "warmup_timeout_seconds": warmup_timeout_seconds,
            "torch_extensions_dir": _path_string(torch_extensions_dir),
            "batch_size": max(2, int(batch_size)),
        },
        "isolated": isolated,
        "batch_reuse": batch,
        "comparison": comparison,
    }
    report["report_hash"] = _artifact_hash(report, "report_hash")
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / EQUIVALENCE_REPORT_FILE
    with report_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(report, f, sort_keys=False, allow_unicode=True, default_flow_style=False, width=100)
    return {
        "pass": report["pass"],
        "report_path": _path_string(report_path),
        "attempt_count": comparison["attempt_count"],
        "mismatch_count": len(comparison["mismatches"]),
        "missing_in_batch_count": len(comparison["missing_in_batch"]),
        "missing_in_isolated_count": len(comparison["missing_in_isolated"]),
    }


def _select_jobs(
    jobs: list[dict[str, Any]],
    *,
    uses: set[str] | None,
    start_index: int,
    max_jobs: int | None,
) -> list[dict[str, Any]]:
    selected = [job for job in jobs if job.get("use") in EXECUTABLE_USES and (uses is None or job.get("use") in uses)]
    if start_index:
        selected = selected[start_index:]
    if max_jobs is not None:
        selected = selected[:max_jobs]
    return selected


def _job_indices(jobs: list[dict[str, Any]]) -> dict[str, int]:
    indices: dict[str, int] = {}
    for index, job in enumerate(jobs):
        job_id = str(job.get("job_id") or "")
        if not job_id:
            raise ValueError(f"M4 probe job at index {index} has no job_id")
        if job_id in indices:
            raise ValueError(f"M4 probe plan contains duplicate job_id: {job_id}")
        indices[job_id] = index
    return indices


def _resolve_job_costs(
    jobs: list[dict[str, Any]],
    *,
    seeds: list[int],
    history_path: Path | None,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Resolve deterministic scheduling-only costs without changing evidence."""

    history_rows: list[dict[str, Any]] = []
    history_identity: dict[str, Any] = {
        "path": None,
        "sha256": None,
        "attempt_count": 0,
        "usable_duration_count": 0,
    }
    if history_path is not None:
        history_path = Path(history_path)
        if not history_path.is_file():
            raise ValueError(f"M4 job-cost history does not exist: {history_path}")
        history_rows = _read_jsonl_strict(history_path, allow_truncated_tail=False)
        history_identity = {
            "path": _path_string(history_path),
            "sha256": _sha256_file(history_path),
            "attempt_count": len(history_rows),
            "usable_duration_count": 0,
        }

    by_job_id: dict[str, list[float]] = {}
    by_identity: dict[tuple[str, str, str], list[float]] = {}
    by_use: dict[str, list[float]] = {}
    for row in history_rows:
        duration = _attempt_cost_duration(row)
        if duration is None:
            continue
        history_identity["usable_duration_count"] += 1
        job_id = str(row.get("job_id") or "")
        if job_id:
            by_job_id.setdefault(job_id, []).append(duration)
        identity = (
            str(row.get("asset")),
            str(row.get("model_id")),
            str(row.get("use")),
        )
        by_identity.setdefault(identity, []).append(duration)
        by_use.setdefault(identity[2], []).append(duration)

    seed_count = max(1, len(seeds))
    costs: dict[str, float] = {}
    for job in jobs:
        job_id = str(job["job_id"])
        explicit = _positive_float(job.get("estimated_cost_sec"))
        values = by_job_id.get(job_id)
        if not values:
            values = by_identity.get(
                (
                    str(job.get("asset")),
                    str(job.get("model_id")),
                    str(job.get("use")),
                )
            )
        if not values:
            values = by_use.get(str(job.get("use")))
        if explicit is not None:
            cost = explicit
        elif values:
            cost = (sum(values) / len(values)) * seed_count
        else:
            cost = 1.0
        costs[job_id] = _clean_float(max(cost, 1e-6))
    return costs, history_identity


def _attempt_cost_duration(row: dict[str, Any]) -> float | None:
    timings = row.get("timings")
    value = timings.get("total_worker_sec") if isinstance(timings, dict) else None
    if value is None:
        value = row.get("duration_sec")
    return _positive_float(value)


def _positive_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0.0:
        return None
    return number


def _build_resume_config(
    *,
    plan: dict[str, Any],
    selected_jobs: list[dict[str, Any]],
    selected_job_indices: dict[str, int],
    asset_catalog_sha256: str,
    python_executable: str,
    task_config: str,
    seeds: list[int],
    thresholds: dict[str, float],
    timeout_seconds: int,
    warmup_timeout_seconds: int,
    torch_extensions_dir: Path | None,
    batch_size: int,
    reuse_task_env: bool,
    parallel_workers: int,
    cuda_devices: list[str] | None,
    device_slots: list[str | None],
    source_kind: str,
    success_contract: SuccessContract,
    articulation_catalog: dict[str, Any],
    job_costs: dict[str, float],
    cost_history: dict[str, Any],
) -> dict[str, Any]:
    config = {
        "version": M4_RESUME_CONFIG_VERSION,
        "source_kind": source_kind,
        "source_probe_plan_hash": plan.get("probe_plan_hash"),
        "pick_operation_placement_policy_version": plan.get(
            "pick_operation_placement_policy_version"
        ),
        "pick_operation_placement_policy_hash": plan.get(
            "pick_operation_placement_policy_hash"
        ),
        "source_validated_pose_set_hash": plan.get(
            "source_validated_pose_set_hash"
        ),
        "initial_state_selection_version": plan.get(
            "initial_state_selection_version"
        ),
        "initial_state_selection_hash": plan.get(
            "initial_state_selection_hash"
        ),
        "source_asset_catalog_sha256": asset_catalog_sha256,
        "runner_source_sha256": _sha256_file(Path(__file__)),
        "worker_source_sha256": _sha256_file(
            Path(__file__).with_name("asset_probe_worker.py")
        ),
        "selected_jobs_hash": _sha256_bytes(
            _canonical_json(selected_jobs).encode("utf-8")
        ),
        "selected_job_indices": {
            str(job["job_id"]): int(selected_job_indices[str(job["job_id"])])
            for job in selected_jobs
        },
        "python_executable": python_executable,
        "task_config": task_config,
        "seeds": [int(seed) for seed in seeds],
        "thresholds": thresholds,
        "timeout_seconds": int(timeout_seconds),
        "warmup_timeout_seconds": int(warmup_timeout_seconds),
        "torch_extensions_dir": _path_string(torch_extensions_dir),
        "batch_size": int(batch_size),
        "reuse_task_env": bool(reuse_task_env),
        "parallel_workers": int(parallel_workers),
        "cuda_devices": list(cuda_devices or []),
        "device_slots": device_slots,
        "gpu_queue_policy_version": M4_GPU_QUEUE_POLICY_VERSION,
        "cost_balance_policy_version": M4_COST_BALANCE_POLICY_VERSION,
        "job_costs": job_costs,
        "job_cost_history": cost_history,
        "success_contract": success_contract.identity,
        "worker_execution_policy_manifest": (
            m4_worker_execution_policy_manifest()
        ),
        "articulation_execution_catalog_hash": articulation_catalog.get(
            "profile_hash"
        ),
    }
    config["resume_config_hash"] = _artifact_hash(config, "resume_config_hash")
    return config


def _validate_resume_metadata(
    metadata: dict[str, Any],
    expected_config: dict[str, Any],
) -> dict[str, Any] | None:
    observed = metadata.get("resume_config")
    if not isinstance(observed, dict):
        raise ValueError(
            "M4 resume checkpoint predates the strict resume contract; "
            "start a fresh output directory"
        )
    observed_hash = observed.get("resume_config_hash")
    if observed_hash != _artifact_hash(observed, "resume_config_hash"):
        raise ValueError("M4 resume checkpoint config self-hash mismatch")
    if metadata.get("resume_config_hash") != observed_hash:
        raise ValueError("M4 resume checkpoint metadata/config hash mismatch")
    if observed_hash != expected_config.get("resume_config_hash"):
        keys = sorted(
            key
            for key in set(observed) | set(expected_config)
            if observed.get(key) != expected_config.get(key)
        )
        migration = _approved_resume_source_migration(
            observed,
            expected_config,
            differing_keys=keys,
        )
        if migration is not None:
            return migration
        raise ValueError(
            "M4 resume checkpoint is incompatible with the requested run: "
            + ", ".join(keys[:12])
        )
    return None


def _approved_resume_source_migration(
    observed: dict[str, Any],
    expected: dict[str, Any],
    *,
    differing_keys: list[str],
) -> dict[str, Any] | None:
    """Resolve one explicitly reviewed source-only checkpoint migration."""

    if differing_keys != ["resume_config_hash", "runner_source_sha256"]:
        return None
    migrations = _load_yaml_required(M4_RESUME_SOURCE_MIGRATIONS_PATH)
    if migrations.get("schema_version") != 1:
        raise ValueError("M4 resume source migration schema is stale")
    for migration in migrations.get("migrations") or []:
        if not isinstance(migration, dict):
            continue
        if (
            migration.get("from_resume_config_hash")
            == observed.get("resume_config_hash")
            and migration.get("from_runner_source_sha256")
            == observed.get("runner_source_sha256")
            and migration.get("to_resume_config_hash")
            == expected.get("resume_config_hash")
            and migration.get("to_runner_source_sha256")
            == expected.get("runner_source_sha256")
        ):
            return dict(migration)
    return None


def _migrate_resume_source_revision(
    *,
    metadata: dict[str, Any],
    metadata_path: Path,
    expected_config: dict[str, Any],
    checkpoint_index_path: Path,
    attempts_checkpoint: Path,
    records_checkpoint: Path,
    reports_dir: Path,
    migration: dict[str, Any],
) -> None:
    """Atomically rebind an audited checkpoint before transient retries."""

    old_hash = str(migration["from_resume_config_hash"])
    new_hash = str(migration["to_resume_config_hash"])
    _validate_resume_reports_directory(
        reports_dir,
        expected_resume_config_hash=old_hash,
    )
    checkpoint_index = _load_checkpoint_index(
        checkpoint_index_path,
        resume_config_hash=old_hash,
    )
    committed_attempts = _committed_checkpoint_rows(
        attempts_checkpoint,
        row_kind="attempt",
        committed_hashes=checkpoint_index["attempts"],
    )
    retryable_count = sum(
        1 for row in committed_attempts if m4_attempt_is_retryable(row)
    )
    migration_kind = str(migration.get("migration_kind") or "")
    contact_rebind = _contact_policy_projection_rebind_audit(
        committed_attempts,
        records_checkpoint=records_checkpoint,
        committed_record_hashes=checkpoint_index["records"],
    )
    if migration_kind == "retryable_attempt_repair" and retryable_count == 0:
        raise ValueError(
            "M4 resume source migration has no retryable checkpoint attempts"
        )
    if migration_kind == "contact_policy_projection_rebind":
        if retryable_count != 0:
            raise ValueError(
                "M4 Contact projection migration unexpectedly contains "
                "retryable attempts"
            )
        if (
            contact_rebind["attempt_count"]
            != migration.get("expected_contact_policy_rebind_attempt_count")
            or contact_rebind["job_count"]
            != migration.get("expected_contact_policy_rebind_job_count")
        ):
            raise ValueError(
                "M4 Contact projection migration audit count mismatch"
            )
    if migration_kind not in {
        "retryable_attempt_repair",
        "contact_policy_projection_rebind",
    }:
        raise ValueError("M4 resume source migration kind is unsupported")
    _archive_retryable_resume_reports(reports_dir)
    checkpoint_index["resume_config_hash"] = new_hash
    _commit_checkpoint_index(
        checkpoint_index_path,
        checkpoint_index,
        checkpoint_paths=(attempts_checkpoint, records_checkpoint),
    )
    audit = {
        "migration_id": migration.get("migration_id"),
        "from_resume_config_hash": old_hash,
        "to_resume_config_hash": new_hash,
        "retryable_attempt_count": retryable_count,
        "contact_policy_rebind_attempt_count": contact_rebind[
            "attempt_count"
        ],
        "contact_policy_rebind_job_count": contact_rebind["job_count"],
        "migrated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    history = list(metadata.get("resume_source_migrations") or [])
    history.append(audit)
    metadata.update(
        {
            "resume_config": expected_config,
            "resume_config_hash": new_hash,
            "resume_source_migrations": history,
        }
    )
    metadata.pop("completed_report_path", None)
    metadata.pop("completed_report_hash", None)
    metadata.pop("completed_at", None)
    _write_json_atomic(metadata_path, metadata)


def _contact_policy_projection_rebind_audit(
    attempts: list[dict[str, Any]],
    *,
    records_checkpoint: Path,
    committed_record_hashes: dict[str, str],
) -> dict[str, int]:
    """Count only cryptographically committed omitted Contact policy fields."""

    bindings: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in _committed_checkpoint_rows(
        records_checkpoint,
        row_kind="record",
        committed_hashes=committed_record_hashes,
    ):
        binding = record.get("contact_contract_binding")
        if isinstance(binding, dict):
            bindings[
                (
                    str(record.get("asset")),
                    str(record.get("model_id")),
                    str(record.get("use")),
                )
            ] = binding
    attempt_ids: set[str] = set()
    jobs: set[tuple[str, str, str]] = set()
    for attempt in attempts:
        if m4_attempt_is_blocking(attempt):
            continue
        key = (
            str(attempt.get("asset")),
            str(attempt.get("model_id")),
            str(attempt.get("use")),
        )
        binding = bindings.get(key)
        if binding is None:
            continue
        observed = (attempt.get("diagnostics") or {}).get(
            "contact_contract_preflight"
        )
        if not isinstance(observed, dict) or observed.get("policy_version") is not None:
            continue
        if resolved_contact_preflight_policy_version(
            observed, binding
        ) != binding.get("policy_version"):
            continue
        attempt_ids.add(str(attempt.get("attempt_id")))
        jobs.add(key)
    return {"attempt_count": len(attempt_ids), "job_count": len(jobs)}


def _completed_report_result(
    metadata: dict[str, Any],
    *,
    expected_resume_config_hash: str,
    expected_reports_dir: Path,
) -> dict[str, Any] | None:
    raw_path = metadata.get("completed_report_path")
    if not raw_path:
        return None
    path = Path(str(raw_path))
    if not path.is_file():
        return None
    if path.resolve().parent != expected_reports_dir.resolve():
        raise ValueError("completed M4 report is outside the target reports directory")
    report = _load_json_required(path)
    if report.get("report_hash") != _artifact_hash(report, "report_hash"):
        raise ValueError("completed M4 resume report self-hash mismatch")
    if report.get("report_hash") != metadata.get("completed_report_hash"):
        raise ValueError("completed M4 report does not match checkpoint metadata")
    runner = report.get("runner") or {}
    if runner.get("resume_config_hash") != expected_resume_config_hash:
        raise ValueError("completed M4 report resume config is stale")
    accounting_errors = m4_report_attempt_accounting_errors(
        report.get("records"),
        report.get("attempts"),
        require_seed_coverage=True,
    )
    if accounting_errors:
        raise ValueError(
            "completed M4 report attempt accounting is invalid: "
            + ", ".join(accounting_errors[:8])
        )
    return {
        "evidence_id": report["evidence_id"],
        "evidence_path": _path_string(path),
        "record_count": len(report.get("records") or []),
        "attempt_count": len(report.get("attempts") or []),
        "summary": report.get("summary") or {},
        "timing_summary_path": runner.get("timing_summary_path"),
        "resume_summary": {
            "resumed": True,
            "completed_report_reused": True,
            "reused_attempt_count": len(report.get("attempts") or []),
            "pending_attempt_count": 0,
        },
    }


def _discover_completed_report_result(
    reports_dir: Path,
    *,
    expected_resume_config_hash: str,
) -> tuple[dict[str, Any], Path, str] | None:
    matches: list[tuple[dict[str, Any], Path, str]] = []
    for path in sorted(reports_dir.glob("*.json")):
        report = _load_json_required(path)
        runner = report.get("runner") or {}
        if runner.get("resume_config_hash") != expected_resume_config_hash:
            continue
        result = _completed_report_result(
            {
                "completed_report_path": _path_string(path),
                "completed_report_hash": report.get("report_hash"),
            },
            expected_resume_config_hash=expected_resume_config_hash,
            expected_reports_dir=reports_dir,
        )
        if result is not None:
            matches.append((result, path, str(report["report_hash"])))
    if len(matches) > 1:
        raise ValueError(
            "M4 resume found multiple completed reports for one exact run config"
        )
    return matches[0] if matches else None


def _validate_resume_reports_directory(
    reports_dir: Path,
    *,
    expected_resume_config_hash: str,
) -> None:
    """Fail before workers run if an existing report would break freeze."""

    report_count = 0
    for path in sorted(reports_dir.glob("*.json")):
        try:
            report = _load_json_required(path)
            if (
                report.get("kind") != "canonical_skill_asset_report"
                or report.get("scope") != "asset"
            ):
                raise ValueError("expected normalized canonical asset report")
            if report.get("schema_version") != REPORT_SCHEMA_VERSION:
                raise ValueError("canonical report schema is stale")
            if report.get("source_kind") != FRESH_CANONICAL_SOURCE_KIND:
                raise ValueError("canonical report source kind is invalid")
            if (
                report.get("pick_operation_placement_policy_version")
                != PICK_OPERATION_PLACEMENT_POLICY_VERSION
                or report.get("pick_operation_placement_policy_hash")
                != PICK_OPERATION_PLACEMENT_POLICY_HASH
            ):
                raise ValueError("canonical report Pick placement policy is stale")
            manifest_errors = m4_worker_execution_policy_manifest_errors(
                report.get("worker_execution_policy_manifest")
            )
            if manifest_errors:
                raise ValueError(
                    "canonical report worker execution manifest is stale: "
                    + ", ".join(manifest_errors)
                )
            contract = SuccessContract.load()
            if (
                report.get("success_contract_version") != contract.version
                or report.get("success_contract_hash") != contract.hash
            ):
                raise ValueError("canonical report success contract is stale")
            if report.get("report_hash") != _artifact_hash(report, "report_hash"):
                raise ValueError("canonical report hash mismatch")
            if report.get("evidence_id") != _evidence_id(report):
                raise ValueError("canonical report evidence id mismatch")
            if path.stem != str(report.get("evidence_id") or ""):
                raise ValueError("canonical report filename does not match evidence id")
            runner = report.get("runner")
            if not isinstance(runner, dict) or runner.get(
                "resume_config_hash"
            ) != expected_resume_config_hash:
                raise ValueError(
                    "canonical report belongs to a different M4 resume config"
                )
            accounting_errors = m4_report_attempt_accounting_errors(
                report.get("records"),
                report.get("attempts"),
                require_seed_coverage=True,
            )
            if accounting_errors:
                raise ValueError(
                    "canonical report attempt accounting is invalid: "
                    + ", ".join(accounting_errors[:8])
                )
            report_count += 1
        except ValueError as exc:
            raise ValueError(
                f"M4 resume reports directory contains invalid report {path}: {exc}"
            ) from exc
    if report_count > 1:
        raise ValueError(
            "M4 resume reports directory contains multiple reports for one exact run"
        )


def _archive_retryable_resume_reports(reports_dir: Path) -> None:
    """Move completed reports aside before transient attempts are replaced."""

    archive_dir = reports_dir / "superseded_retryable_resume"
    for path in sorted(reports_dir.glob("*.json")):
        archive_dir.mkdir(parents=True, exist_ok=True)
        target = archive_dir / path.name
        suffix = 1
        while target.exists():
            target = archive_dir / f"{path.stem}.{suffix}{path.suffix}"
            suffix += 1
        os.replace(path, target)


def _bind_articulation_profile_provenance(
    jobs: list[dict[str, Any]],
    *,
    catalog: dict[str, Any],
    contract: SuccessContract,
) -> None:
    """Preflight and bind the exact reviewed profile used by each A2O job."""

    for job in jobs:
        use = str(job.get("use") or "")
        if use not in ARTICULATION_EXECUTABLE_USES:
            continue
        try:
            profile = resolve_articulation_execution_profile(
                str(job.get("asset")),
                str(job.get("model_id")),
                use,
                contract=contract,
                catalog=catalog,
            )
        except Exception as exc:
            identity = str(
                job.get("job_id")
                or f"{job.get('asset')}|{job.get('model_id')}|{use}"
            )
            raise ValueError(
                f"articulation execution profile preflight failed for "
                f"{identity}: {exc}"
            ) from exc
        job.update(
            {
                "articulation_execution_profile_version": profile[
                    "profile_version"
                ],
                "articulation_execution_profile_hash": profile["profile_hash"],
                "articulation_execution_profile_source_hash": profile[
                    "source_asset_catalog_sha256"
                ],
                "articulation_motion_policy_version": profile[
                    "motion_policy_version"
                ],
                "articulation_motion_policy_hash": profile[
                    "motion_policy_hash"
                ],
            }
        )


def _clear_batch_artifacts(run_dir: Path) -> None:
    batches_dir = run_dir / "batches"
    if not batches_dir.is_dir():
        return
    for pattern in ("batch_*.json", "batch_*.results.jsonl"):
        for path in batches_dir.glob(pattern):
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def _next_batch_index(run_dir: Path) -> int:
    batches_dir = run_dir / "batches"
    indices: list[int] = []
    if batches_dir.is_dir():
        for path in batches_dir.glob("batch_*.json"):
            suffix = path.stem.removeprefix("batch_")
            if suffix.isdecimal():
                indices.append(int(suffix))
    return max(indices, default=-1) + 1


def _worker_device_slots(
    cuda_devices: list[str] | None,
    parallel_workers: int,
) -> list[str | None]:
    worker_count = max(1, int(parallel_workers))
    if not cuda_devices:
        if worker_count != 1:
            raise ValueError(
                "parallel M4 probes require explicit CUDA devices so per-GPU "
                "concurrency can be bounded"
            )
        return [None]
    devices = _normalize_cuda_device_selectors(cuda_devices)
    if not devices:
        if worker_count != 1:
            raise ValueError(
                "parallel M4 probes require at least one explicit CUDA device"
            )
        return [None]
    if worker_count < len(devices) or worker_count % len(devices):
        raise ValueError(
            "M4 parallel_workers must be an integer multiple of the unique "
            "CUDA device count so every GPU has one explicit concurrency cap"
        )
    return [devices[index % len(devices)] for index in range(worker_count)]


def _normalize_cuda_device_selectors(
    cuda_devices: list[str],
) -> list[str]:
    selectors = [
        str(device).strip() for device in cuda_devices if str(device).strip()
    ]
    normalized = [
        str(int(selector, 10)) if selector.isdecimal() else selector
        for selector in selectors
    ]
    if len(set(normalized)) != len(normalized):
        raise ValueError("M4 CUDA device ids must be unique")
    return normalized


def _build_cost_balanced_slot_batches(
    attempt_specs: list[dict[str, Any]],
    *,
    batch_size: int,
    device_slots: list[str | None],
    job_costs: dict[str, float],
    expected_seed_count: int,
) -> list[list[list[dict[str, Any]]]]:
    """Pack complete job seed groups, then LPT-balance batches over GPU slots."""

    if not device_slots:
        raise ValueError("M4 scheduling requires at least one fixed GPU slot")
    capacity = max(1, int(batch_size))
    default_attempt_count = max(1, int(expected_seed_count))

    specs_by_job: dict[str, list[dict[str, Any]]] = {}
    job_first_order: dict[str, tuple[int, str]] = {}
    for spec in attempt_specs:
        job_id = str(spec["job_id"])
        specs_by_job.setdefault(job_id, []).append(spec)
        job_first_order.setdefault(
            job_id,
            (int(spec.get("job_index") or 0), job_id),
        )
    for specs in specs_by_job.values():
        specs.sort(key=lambda spec: int(spec["seed"]))

    groups: list[dict[str, Any]] = []
    for job_id, specs in specs_by_job.items():
        full_cost = float(job_costs.get(job_id, 1.0))
        expected_attempt_count = max(
            (
                int(spec.get("job_attempt_count") or default_attempt_count)
                for spec in specs
            ),
            default=default_attempt_count,
        )
        spec_groups = (
            [specs]
            if len(specs) <= capacity
            else _chunks(specs, capacity)
        )
        for group_index, grouped_specs in enumerate(spec_groups):
            groups.append(
                {
                    "job_id": job_id,
                    "specs": grouped_specs,
                    "cost": (
                        full_cost
                        * len(grouped_specs)
                        / expected_attempt_count
                    ),
                    "order": (*job_first_order[job_id], group_index),
                }
            )
    groups.sort(key=lambda group: (-group["cost"], group["order"]))

    initial_batch_count = max(
        1,
        math.ceil(len(attempt_specs) / capacity),
    ) if attempt_specs else 0
    bins: list[dict[str, Any]] = [
        {"groups": [], "size": 0, "cost": 0.0, "index": index}
        for index in range(initial_batch_count)
    ]
    for group in groups:
        candidates = [
            batch
            for batch in bins
            if batch["size"] == 0
            or batch["size"] + len(group["specs"]) <= capacity
        ]
        if not candidates:
            batch = {
                "groups": [],
                "size": 0,
                "cost": 0.0,
                "index": len(bins),
            }
            bins.append(batch)
            candidates = [batch]
        target = min(
            candidates,
            key=lambda batch: (batch["cost"], batch["size"], batch["index"]),
        )
        target["groups"].append(group)
        target["size"] += len(group["specs"])
        target["cost"] += group["cost"]

    packed_batches: list[dict[str, Any]] = []
    for batch in bins:
        if not batch["groups"]:
            continue
        ordered_groups = sorted(batch["groups"], key=lambda group: group["order"])
        packed_batches.append(
            {
                "index": batch["index"],
                "cost": batch["cost"],
                "specs": [
                    spec
                    for group in ordered_groups
                    for spec in group["specs"]
                ],
            }
        )

    slot_rows = [
        {"index": index, "cost": 0.0, "batches": []}
        for index in range(len(device_slots))
    ]
    for batch in sorted(
        packed_batches,
        key=lambda row: (-row["cost"], row["index"]),
    ):
        slot = min(
            slot_rows,
            key=lambda row: (row["cost"], len(row["batches"]), row["index"]),
        )
        slot["batches"].append(batch["specs"])
        slot["cost"] += batch["cost"]
    return [row["batches"] for row in slot_rows]


def _run_isolated_attempts_fresh(
    *,
    python_executable: str,
    asset_catalog_path: Path,
    jobs: list[dict[str, Any]],
    job_indices: dict[str, int],
    seeds: list[int],
    thresholds: dict[str, float],
    task_config: str,
    run_dir: Path,
    timeout_seconds: int,
    warmup_timeout_seconds: int,
    torch_extensions_dir: Path | None,
    attempts_checkpoint: Path,
    records_checkpoint: Path,
    resume_config_hash: str,
) -> dict[str, Any]:
    """Preserve the original one-process-per-attempt default execution path."""

    attempts: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    checkpoint_index_path = attempts_checkpoint.parent / "checkpoint_index.json"
    checkpoint_index = _empty_checkpoint_index(resume_config_hash)
    _commit_checkpoint_index(
        checkpoint_index_path,
        checkpoint_index,
        checkpoint_paths=(attempts_checkpoint, records_checkpoint),
    )
    first_attempt = True
    for job in jobs:
        job_attempts: list[dict[str, Any]] = []
        for requested_arm in _requested_arms_for_job(job):
            for seed in seeds:
                attempt = _run_attempt(
                    python_executable=python_executable,
                    asset_catalog_path=asset_catalog_path,
                    job=job,
                    seed=seed,
                    requested_arm=requested_arm,
                    task_config=task_config,
                    run_dir=run_dir,
                    timeout_seconds=(
                        warmup_timeout_seconds
                        if first_attempt
                        else timeout_seconds
                    ),
                    torch_extensions_dir=torch_extensions_dir,
                    warmup=first_attempt,
                )
                first_attempt = False
                attempt["job_index"] = int(job_indices[str(job["job_id"])])
                attempts.append(attempt)
                job_attempts.append(attempt)
                _append_checkpoint_row(attempts_checkpoint, attempt)
                _checkpoint_index_add(
                    checkpoint_index,
                    attempt,
                    row_kind="attempt",
                )
                _commit_checkpoint_index(
                    checkpoint_index_path,
                    checkpoint_index,
                    checkpoint_paths=(attempts_checkpoint, records_checkpoint),
                )
        record = _record_from_attempts(
            job,
            job_attempts,
            thresholds[job["use"]],
            seeds,
        )
        records.append(record)
        _append_checkpoint_row(records_checkpoint, record)
        _checkpoint_index_add(checkpoint_index, record, row_kind="record")
        _commit_checkpoint_index(
            checkpoint_index_path,
            checkpoint_index,
            checkpoint_paths=(attempts_checkpoint, records_checkpoint),
        )
    _rewrite_checkpoint_jsonl(attempts_checkpoint, attempts)
    _rewrite_checkpoint_jsonl(records_checkpoint, records)
    return {
        "attempts": attempts,
        "records": records,
        "resume_summary": {
            "resumed": False,
            "reused_attempt_count": 0,
            "pending_attempt_count": len(attempts),
        },
    }


def _run_attempt_batches(
    *,
    python_executable: str,
    asset_catalog_path: Path,
    jobs: list[dict[str, Any]],
    job_indices: dict[str, int],
    seeds: list[int],
    thresholds: dict[str, float],
    task_config: str,
    run_dir: Path,
    timeout_seconds: int,
    warmup_timeout_seconds: int,
    torch_extensions_dir: Path | None,
    batch_size: int,
    reuse_task_env: bool,
    parallel_workers: int,
    device_slots: list[str | None],
    attempts_checkpoint: Path,
    records_checkpoint: Path,
    resume: bool,
    resume_config_hash: str,
    job_costs: dict[str, float],
) -> dict[str, Any]:
    if len(device_slots) != max(1, int(parallel_workers)):
        raise ValueError("M4 fixed GPU slot count does not match parallel_workers")
    attempts_by_job: dict[str, list[dict[str, Any]]] = {
        str(job["job_id"]): [] for job in jobs
    }
    jobs_by_id = {str(job["job_id"]): job for job in jobs}
    all_attempt_specs = _build_attempt_specs(
        jobs,
        seeds,
        start_index=0,
        job_indices=job_indices,
    )
    expected_order = {
        str(spec["attempt_id"]): index
        for index, spec in enumerate(all_attempt_specs)
    }
    if len(expected_order) != len(all_attempt_specs):
        raise ValueError("M4 probe plan produced duplicate attempt ids")

    attempts: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    emitted_jobs: set[str] = set()
    checkpoint_index_path = attempts_checkpoint.parent / "checkpoint_index.json"
    if resume:
        checkpoint_index = _load_checkpoint_index(
            checkpoint_index_path,
            resume_config_hash=resume_config_hash,
        )
        attempts = _load_resume_attempts(
            attempts_checkpoint,
            all_attempt_specs,
            committed_hashes=checkpoint_index["attempts"],
        )
        for attempt in attempts:
            attempts_by_job[str(attempt["job_id"])].append(attempt)
        records = _validate_resume_records(
            records_checkpoint,
            jobs=jobs,
            attempts_by_job=attempts_by_job,
            thresholds=thresholds,
            seeds=seeds,
            committed_hashes=checkpoint_index["records"],
        )
        emitted_jobs = {
            str(job["job_id"])
            for job in jobs
            if len(attempts_by_job[str(job["job_id"])])
            == _expected_attempt_count(job, seeds)
        }
        _rewrite_checkpoint_jsonl(attempts_checkpoint, attempts)
        _rewrite_checkpoint_jsonl(records_checkpoint, records)
        checkpoint_index["attempts"] = {}
        for attempt in attempts:
            _checkpoint_index_add(checkpoint_index, attempt, row_kind="attempt")
        checkpoint_index["records"] = {}
        for record in records:
            _checkpoint_index_add(checkpoint_index, record, row_kind="record")
        _commit_checkpoint_index(
            checkpoint_index_path,
            checkpoint_index,
            checkpoint_paths=(attempts_checkpoint, records_checkpoint),
        )
    else:
        checkpoint_index = _empty_checkpoint_index(resume_config_hash)
        _commit_checkpoint_index(
            checkpoint_index_path,
            checkpoint_index,
            checkpoint_paths=(attempts_checkpoint, records_checkpoint),
        )

    completed_attempt_ids = {str(attempt["attempt_id"]) for attempt in attempts}
    pending_specs = [
        dict(spec)
        for spec in all_attempt_specs
        if str(spec["attempt_id"]) not in completed_attempt_ids
    ]
    resume_summary = {
        "resumed": bool(resume),
        "reused_attempt_count": len(attempts),
        "pending_attempt_count": len(pending_specs),
    }

    batches_dir = run_dir / "batches"
    batches_dir.mkdir(parents=True, exist_ok=True)
    slot_batches = _build_cost_balanced_slot_batches(
        pending_specs,
        batch_size=max(1, int(batch_size)),
        device_slots=device_slots,
        job_costs=job_costs,
        expected_seed_count=max(
            (_expected_attempt_count(job, seeds) for job in jobs),
            default=len(seeds),
        ),
    )
    batch_index_offset = _next_batch_index(run_dir) if resume else 0

    def accept_batch(batch_attempts: list[dict[str, Any]]) -> None:
        for attempt in batch_attempts:
            attempt_id = str(attempt.get("attempt_id"))
            if attempt_id not in expected_order:
                raise ValueError(
                    f"M4 worker returned unexpected attempt id: {attempt_id}"
                )
            attempts.append(attempt)
            _append_checkpoint_row(attempts_checkpoint, attempt)
            _checkpoint_index_add(checkpoint_index, attempt, row_kind="attempt")
            job_id = str(attempt["job_id"])
            attempts_by_job[job_id].append(attempt)
            if (
                len(attempts_by_job[job_id])
                == _expected_attempt_count(jobs_by_id[job_id], seeds)
                and job_id not in emitted_jobs
            ):
                job = jobs_by_id[job_id]
                ordered_attempts = sorted(
                    attempts_by_job[job_id], key=lambda row: int(row["seed"])
                )
                record = _record_from_attempts(
                    job,
                    ordered_attempts,
                    thresholds[job["use"]],
                    seeds,
                )
                records.append(record)
                emitted_jobs.add(job_id)
                _append_checkpoint_row(records_checkpoint, record)
                _checkpoint_index_add(checkpoint_index, record, row_kind="record")
        _commit_checkpoint_index(
            checkpoint_index_path,
            checkpoint_index,
            checkpoint_paths=(attempts_checkpoint, records_checkpoint),
        )

    submitted: list[tuple[int, int, list[dict[str, Any]]]] = []
    logical_batch_index = 0
    for slot_index, batches in enumerate(slot_batches):
        for slot_batch_index, chunk in enumerate(batches):
            if chunk:
                chunk[0]["warmup"] = slot_batch_index == 0
            submitted.append((logical_batch_index, slot_index, chunk))
            logical_batch_index += 1

    if submitted:
        workers_by_device: dict[str | None, int] = {}
        for device in device_slots:
            workers_by_device[device] = workers_by_device.get(device, 0) + 1
        executors = {
            device: ThreadPoolExecutor(max_workers=worker_limit)
            for device, worker_limit in workers_by_device.items()
        }
        try:
            future_to_batch = {}
            for batch_index, slot_index, chunk in submitted:
                device = device_slots[slot_index]
                future = executors[device].submit(
                    _run_attempt_batch_safe,
                    python_executable=python_executable,
                    asset_catalog_path=asset_catalog_path,
                    attempts=chunk,
                    task_config=task_config,
                    run_dir=run_dir,
                    torch_extensions_dir=torch_extensions_dir,
                    timeout_seconds=_chunk_timeout(
                        chunk, timeout_seconds, warmup_timeout_seconds
                    ),
                    reuse_task_env=reuse_task_env,
                    batch_index=batch_index_offset + batch_index,
                    batches_dir=batches_dir,
                    cuda_device=device,
                    warp_cache_path=(
                        run_dir / "warp_cache" / f"slot_{slot_index:03d}"
                    ),
                )
                future_to_batch[future] = batch_index
            for future in as_completed(future_to_batch):
                accept_batch(future.result())
        finally:
            for executor in executors.values():
                executor.shutdown(wait=True, cancel_futures=False)

    for job in jobs:
        job_id = str(job["job_id"])
        if job_id in emitted_jobs:
            continue
        job_attempts = attempts_by_job[job_id]
        expected_attempts = _expected_attempt_count(job, seeds)
        if len(job_attempts) != expected_attempts:
            raise ValueError(
                f"M4 runner is missing attempts for {job_id}: "
                f"{len(job_attempts)} != {expected_attempts}"
            )
        records.append(
            _record_from_attempts(
                job,
                sorted(job_attempts, key=lambda row: int(row["seed"])),
                thresholds[job["use"]],
                seeds,
            )
        )
        emitted_jobs.add(job_id)

    attempts.sort(key=lambda row: expected_order[str(row["attempt_id"])])
    record_order = {
        (str(job["asset"]), str(job["model_id"]), str(job["use"])): index
        for index, job in enumerate(jobs)
    }
    records.sort(
        key=lambda row: record_order[
            (str(row["asset"]), str(row["model_id"]), str(row["use"]))
        ]
    )
    _rewrite_checkpoint_jsonl(attempts_checkpoint, attempts)
    _rewrite_checkpoint_jsonl(records_checkpoint, records)
    checkpoint_index["attempts"] = {}
    checkpoint_index["records"] = {}
    for attempt in attempts:
        _checkpoint_index_add(checkpoint_index, attempt, row_kind="attempt")
    for record in records:
        _checkpoint_index_add(checkpoint_index, record, row_kind="record")
    _commit_checkpoint_index(
        checkpoint_index_path,
        checkpoint_index,
        checkpoint_paths=(attempts_checkpoint, records_checkpoint),
    )
    return {
        "attempts": attempts,
        "records": records,
        "resume_summary": resume_summary,
    }


def _build_attempt_specs(
    jobs: list[dict[str, Any]],
    seeds: list[int],
    *,
    start_index: int,
    job_indices: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for job_offset, job in enumerate(jobs, start=start_index):
        job_index = (
            int(job_indices[str(job["job_id"])])
            if job_indices is not None
            else job_offset
        )
        probe_job = _probe_job_payload(job)
        for requested_arm in _requested_arms_for_job(job):
            for seed in seeds:
                arm_suffix = (
                    f"|arm:{requested_arm}" if requested_arm is not None else ""
                )
                spec = {
                    "attempt_id": f"{job['job_id']}{arm_suffix}|seed{seed}",
                    "job_id": job["job_id"],
                    "job_index": job_index,
                    "asset": str(job["asset"]),
                    "model_id": str(job["model_id"]),
                    "use": str(job["use"]),
                    "seed": int(seed),
                    "job_attempt_count": _expected_attempt_count(job, seeds),
                }
                if requested_arm is not None:
                    spec["requested_arm"] = requested_arm
                for field in ARTICULATION_PROVENANCE_FIELDS:
                    if job.get(field) is not None:
                        spec[field] = job[field]
                if probe_job is not None:
                    spec["probe_job"] = {
                        **probe_job,
                        **(
                            {"requested_arm": requested_arm}
                            if requested_arm is not None
                            else {}
                        ),
                    }
                specs.append(spec)
    return specs


def _probe_job_payload(job: dict[str, Any]) -> dict[str, Any] | None:
    if isinstance(job.get("probe_job"), dict):
        return dict(job["probe_job"])
    candidate = job.get("candidate")
    if not isinstance(candidate, dict) and not isinstance(
        job.get("canonical_initial_state"), dict
    ):
        return None
    return {
        "scope": job.get("scope", "asset"),
        "stage": STAGE_CANONICAL,
        "asset": str(job["asset"]),
        "model_id": str(job["model_id"]),
        "use": str(job["use"]),
        "candidate_use": job.get("candidate_use"),
        "candidate": candidate,
        "initial_pose_binding": job.get("initial_pose_binding"),
        "canonical_initial_state": job.get("canonical_initial_state"),
        "initial_state_selection_version": job.get(
            "initial_state_selection_version"
        ),
        "initial_state_selection_hash": job.get("initial_state_selection_hash"),
        "contact_contract_binding": job.get("contact_contract_binding"),
        "articulation_arm_sampling_version": job.get(
            "articulation_arm_sampling_version"
        ),
        "touch_arm_sampling_version": job.get(
            "touch_arm_sampling_version"
        ),
        "touch_placement_policy_version": job.get(
            "touch_placement_policy_version"
        ),
        "touch_placement_policy_hash": job.get(
            "touch_placement_policy_hash"
        ),
        "requested_arms": job.get("requested_arms"),
    }


def _run_attempt_batch(
    *,
    python_executable: str,
    asset_catalog_path: Path,
    attempts: list[dict[str, Any]],
    task_config: str,
    run_dir: Path,
    torch_extensions_dir: Path | None,
    timeout_seconds: int,
    reuse_task_env: bool,
    batch_index: int,
    batches_dir: Path,
    cuda_device: str | None = None,
    warp_cache_path: Path | None = None,
) -> list[dict[str, Any]]:
    started = time.monotonic()
    batch_file = batches_dir / f"batch_{batch_index:06d}.json"
    batch_output = batches_dir / f"batch_{batch_index:06d}.results.jsonl"
    stdout_path = run_dir / "logs" / f"batch_{batch_index:06d}.stdout.log"
    stderr_path = run_dir / "logs" / f"batch_{batch_index:06d}.stderr.log"
    _write_json(batch_file, {"attempts": attempts})
    cmd = [
        python_executable,
        "-m",
        "if_benchmark.asset_probe_worker",
        "--asset-catalog",
        str(asset_catalog_path),
        "--task-config",
        task_config,
        "--run-dir",
        str(run_dir),
        "--batch-file",
        str(batch_file),
        "--batch-output",
        str(batch_output),
    ]
    if reuse_task_env:
        cmd.append("--reuse-task-env")
    env = os.environ.copy()
    env.setdefault("PYTHONWARNINGS", "ignore::UserWarning")
    if cuda_device is None:
        env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    else:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
    _apply_torch_extensions_dir(env, torch_extensions_dir)
    if warp_cache_path is not None:
        _apply_warp_cache_path(env, warp_cache_path)
    env.setdefault("MPLCONFIGDIR", str(run_dir / "matplotlib"))
    timeout = False
    returncode: int | None = None
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout_f, stderr_path.open("w", encoding="utf-8") as stderr_f:
            proc = subprocess.Popen(
                cmd,
                cwd=Path.cwd(),
                env=env,
                text=True,
                stdout=stdout_f,
                stderr=stderr_f,
            )
            try:
                returncode = proc.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                timeout = True
                proc.kill()
                proc.wait()
    except subprocess.TimeoutExpired:
        timeout = True
    duration_sec = _clean_float(time.monotonic() - started)
    parsed_results = _read_batch_results(batch_output)
    parsed_by_attempt = {str(item.get("attempt_id")): item for item in parsed_results if item.get("attempt_id")}
    results: list[dict[str, Any]] = []
    for spec in attempts:
        attempt_id = str(spec["attempt_id"])
        if attempt_id in parsed_by_attempt:
            result = dict(parsed_by_attempt[attempt_id])
        else:
            result = _missing_batch_attempt(
                spec,
                reason="worker_timeout" if timeout else ("worker_no_json_result" if returncode == 0 else "worker_process_failed"),
                duration_sec=duration_sec,
            )
        result.setdefault("job_id", spec["job_id"])
        result.setdefault("job_index", spec["job_index"])
        result.setdefault("asset", spec["asset"])
        result.setdefault("model_id", str(spec["model_id"]))
        result.setdefault("use", spec["use"])
        result.setdefault("seed", int(spec["seed"]))
        result.setdefault("warmup", bool(spec.get("warmup", False)))
        _apply_expected_articulation_provenance(result, spec)
        result["worker_returncode"] = returncode
        result["cuda_device"] = cuda_device
        result["stdout_log"] = _path_string(stdout_path)
        result["stderr_log"] = _path_string(stderr_path)
        result["stdout_tail"] = _tail_file(stdout_path)
        result["stderr_tail"] = _tail_file(stderr_path)
        result.setdefault("duration_sec", (result.get("timings") or {}).get("total_worker_sec", duration_sec))
        results.append(result)
    return results


def _run_attempt_batch_safe(
    *,
    python_executable: str,
    asset_catalog_path: Path,
    attempts: list[dict[str, Any]],
    task_config: str,
    run_dir: Path,
    torch_extensions_dir: Path | None,
    timeout_seconds: int,
    reuse_task_env: bool,
    batch_index: int,
    batches_dir: Path,
    cuda_device: str | None = None,
    warp_cache_path: Path | None = None,
) -> list[dict[str, Any]]:
    try:
        return _run_attempt_batch(
            python_executable=python_executable,
            asset_catalog_path=asset_catalog_path,
            attempts=attempts,
            task_config=task_config,
            run_dir=run_dir,
            torch_extensions_dir=torch_extensions_dir,
            timeout_seconds=timeout_seconds,
            reuse_task_env=reuse_task_env,
            batch_index=batch_index,
            batches_dir=batches_dir,
            cuda_device=cuda_device,
            warp_cache_path=warp_cache_path,
        )
    except Exception as exc:
        results = []
        for spec in attempts:
            result = _missing_batch_attempt(spec, reason="worker_process_failed", duration_sec=0.0)
            result["exception_type"] = type(exc).__name__
            result["exception"] = str(exc)
            result["traceback_tail"] = traceback.format_exc()[-4000:]
            result["cuda_device"] = cuda_device
            results.append(result)
        return results


def _missing_batch_attempt(spec: dict[str, Any], *, reason: str, duration_sec: float) -> dict[str, Any]:
    result = {
        "attempt_id": spec.get("attempt_id"),
        "job_id": spec.get("job_id"),
        "job_index": spec.get("job_index"),
        "asset": spec.get("asset"),
        "model_id": str(spec.get("model_id")),
        "use": spec.get("use"),
        "requested_arm": spec.get("requested_arm"),
        "seed": int(spec.get("seed")),
        "plan_success": False,
        "success": False,
        "failure_reason": reason,
        "duration_sec": duration_sec,
        "warmup": bool(spec.get("warmup", False)),
        "timings": {"total_worker_sec": duration_sec},
    }
    _apply_expected_articulation_provenance(result, spec)
    return result


def _apply_expected_articulation_provenance(
    result: dict[str, Any], source: dict[str, Any]
) -> None:
    """Keep source-bound profile identity on success and infrastructure failure."""

    expected = {
        field: source[field]
        for field in ARTICULATION_PROVENANCE_FIELDS
        if source.get(field) is not None
    }
    if not expected:
        return
    mismatches = {
        field: {"expected": value, "actual": result.get(field)}
        for field, value in expected.items()
        if result.get(field) is not None and result.get(field) != value
    }
    if mismatches:
        result["plan_success"] = False
        result["checker_success"] = False
        result["success"] = False
        result["failure_reason"] = (
            "internal_articulation_profile_provenance_mismatch"
        )
        result["articulation_profile_provenance_mismatch"] = mismatches
    result.update(expected)


def _chunk_timeout(attempts: list[dict[str, Any]], timeout_seconds: int, warmup_timeout_seconds: int) -> int:
    total = 0
    for attempt in attempts:
        total += warmup_timeout_seconds if attempt.get("warmup") else timeout_seconds
    return max(timeout_seconds, total)


def _chunks(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [items[index:index + size] for index in range(0, len(items), size)]


def _read_batch_results(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    results: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                results.append(data)
    return results


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return _read_batch_results(path)


def _read_jsonl_strict(
    path: Path,
    *,
    allow_truncated_tail: bool,
) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="strict")
    lines = text.splitlines()
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            if allow_truncated_tail and line_number == len(lines) and not text.endswith("\n"):
                break
            raise ValueError(
                f"invalid JSONL checkpoint row {path}:{line_number}"
            ) from exc
        if not isinstance(row, dict):
            raise ValueError(
                f"checkpoint row must be a mapping: {path}:{line_number}"
            )
        rows.append(row)
    return rows


def _append_checkpoint_row(path: Path, row: dict[str, Any]) -> None:
    """Append a raw row; integrity lives in the atomic sidecar index."""

    _append_jsonl(path, row)


def _rewrite_checkpoint_jsonl(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False))
            stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _checkpoint_key(row: dict[str, Any], row_kind: str) -> str:
    if row_kind == "attempt":
        key = str(row.get("attempt_id") or "")
        if not key:
            raise ValueError("M4 attempt checkpoint row has no attempt_id")
        return key
    if row_kind == "record":
        return _canonical_json(
            [str(row.get("asset")), str(row.get("model_id")), str(row.get("use"))]
        )
    raise ValueError(f"unsupported M4 checkpoint row kind: {row_kind}")


def _checkpoint_payload_hash(row: dict[str, Any]) -> str:
    return _sha256_bytes(_canonical_json(row).encode("utf-8"))


def _empty_checkpoint_index(resume_config_hash: str) -> dict[str, Any]:
    index = {
        "version": M4_CHECKPOINT_INDEX_VERSION,
        "resume_config_hash": resume_config_hash,
        "attempts": {},
        "records": {},
    }
    index["index_hash"] = _artifact_hash(index, "index_hash")
    return index


def _write_checkpoint_index(path: Path, index: dict[str, Any]) -> None:
    payload = {
        key: value for key, value in index.items() if key != "index_hash"
    }
    payload["index_hash"] = _artifact_hash(payload, "index_hash")
    index.clear()
    index.update(payload)
    _write_json_atomic(path, index)


def _commit_checkpoint_index(
    path: Path,
    index: dict[str, Any],
    *,
    checkpoint_paths: tuple[Path, ...],
) -> None:
    """Durably publish raw rows before their hashes become committed."""

    _sync_checkpoint_files(*checkpoint_paths)
    _write_checkpoint_index(path, index)


def _sync_checkpoint_files(*paths: Path) -> None:
    """Flush raw checkpoint files once per commit, not once per row."""

    for path in paths:
        with path.open("a", encoding="utf-8") as stream:
            stream.flush()
            os.fsync(stream.fileno())


def _load_checkpoint_index(
    path: Path,
    *,
    resume_config_hash: str,
) -> dict[str, Any]:
    index = _load_json_required(path)
    if index.get("index_hash") != _artifact_hash(index, "index_hash"):
        raise ValueError("M4 checkpoint index self-hash mismatch")
    if index.get("version") != M4_CHECKPOINT_INDEX_VERSION:
        raise ValueError("M4 checkpoint index version is stale")
    if index.get("resume_config_hash") != resume_config_hash:
        raise ValueError("M4 checkpoint index config hash mismatch")
    for field in ("attempts", "records"):
        values = index.get(field)
        if not isinstance(values, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in values.items()
        ):
            raise ValueError(f"M4 checkpoint index {field} mapping is invalid")
    return index


def _checkpoint_index_add(
    index: dict[str, Any],
    row: dict[str, Any],
    *,
    row_kind: str,
) -> None:
    field = "attempts" if row_kind == "attempt" else "records"
    key = _checkpoint_key(row, row_kind)
    row_hash = _checkpoint_payload_hash(row)
    prior = index[field].get(key)
    if prior is not None and prior != row_hash:
        raise ValueError(f"M4 checkpoint {row_kind} key changed payload: {key}")
    index[field][key] = row_hash


def _committed_checkpoint_rows(
    path: Path,
    *,
    row_kind: str,
    committed_hashes: dict[str, str],
) -> list[dict[str, Any]]:
    rows = _read_jsonl_strict(path, allow_truncated_tail=True)
    by_key: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = _checkpoint_key(row, row_kind)
        if key in by_key:
            raise ValueError(f"M4 checkpoint contains duplicate {row_kind}: {key}")
        by_key[key] = row
    missing = sorted(set(committed_hashes) - set(by_key))
    if missing:
        raise ValueError(
            f"M4 checkpoint index references missing {row_kind}: {missing[0]}"
        )
    committed: list[dict[str, Any]] = []
    for row in rows:
        key = _checkpoint_key(row, row_kind)
        expected_hash = committed_hashes.get(key)
        if expected_hash is None:
            continue
        if _checkpoint_payload_hash(row) != expected_hash:
            raise ValueError(f"M4 checkpoint {row_kind} payload hash mismatch: {key}")
        committed.append(row)
    return committed


def _load_resume_attempts(
    checkpoint_path: Path,
    expected_specs: list[dict[str, Any]],
    *,
    committed_hashes: dict[str, str],
) -> list[dict[str, Any]]:
    expected_by_id = {
        str(spec["attempt_id"]): spec for spec in expected_specs
    }
    selected: dict[str, dict[str, Any]] = {}
    for row in _committed_checkpoint_rows(
        checkpoint_path,
        row_kind="attempt",
        committed_hashes=committed_hashes,
    ):
        attempt_id = str(row.get("attempt_id") or "")
        if attempt_id not in expected_by_id:
            raise ValueError(
                f"M4 resume checkpoint contains an unknown attempt: {attempt_id}"
            )
        if attempt_id in selected:
            raise ValueError(
                f"M4 resume checkpoint contains duplicate attempt: {attempt_id}"
            )
        spec = expected_by_id[attempt_id]
        for field in (
            "job_id",
            "job_index",
            "asset",
            "model_id",
            "use",
            "seed",
            "requested_arm",
        ):
            observed = str(row.get(field)) if field != "seed" else row.get(field)
            expected = str(spec.get(field)) if field != "seed" else spec.get(field)
            if observed != expected:
                raise ValueError(
                    f"M4 resume checkpoint identity mismatch for {attempt_id}: {field}"
                )
        if not m4_attempt_is_blocking(row):
            manifest_errors = m4_worker_execution_policy_manifest_errors(
                row.get("execution_policy_manifest"),
                use=str(spec.get("use") or ""),
            )
            if manifest_errors:
                raise ValueError(
                    f"M4 resume attempt has stale worker policy evidence: "
                    f"{attempt_id}: {', '.join(manifest_errors)}"
                )
            expected_state_hash = (
                ((spec.get("probe_job") or {}).get("canonical_initial_state") or {}).get(
                    "canonical_state_hash"
                )
            )
            placement_errors = m4_canonical_placement_witness_errors(
                row.get("canonical_placement_witness"),
                expected_use=str(spec.get("use") or ""),
                expected_canonical_state_hash=(
                    str(expected_state_hash)
                    if expected_state_hash is not None
                    else None
                ),
            )
            if expected_state_hash is not None and placement_errors:
                raise ValueError(
                    "M4 resume attempt has stale placement evidence: "
                    f"{attempt_id}: {', '.join(placement_errors)}"
                )
        if m4_attempt_is_retryable(row):
            continue
        selected[attempt_id] = row
    return [
        selected[str(spec["attempt_id"])]
        for spec in expected_specs
        if str(spec["attempt_id"]) in selected
    ]


def _validate_resume_records(
    checkpoint_path: Path,
    *,
    jobs: list[dict[str, Any]],
    attempts_by_job: dict[str, list[dict[str, Any]]],
    thresholds: dict[str, float],
    seeds: list[int],
    committed_hashes: dict[str, str],
) -> list[dict[str, Any]]:
    expected_by_identity: dict[tuple[str, str, str], dict[str, Any]] = {}
    known_identities = {
        (str(job["asset"]), str(job["model_id"]), str(job["use"]))
        for job in jobs
    }
    expected_records: list[dict[str, Any]] = []
    for job in jobs:
        job_id = str(job["job_id"])
        attempts = attempts_by_job[job_id]
        if len(attempts) != _expected_attempt_count(job, seeds):
            continue
        record = _record_from_attempts(
            job,
            sorted(attempts, key=lambda row: int(row["seed"])),
            thresholds[job["use"]],
            seeds,
        )
        identity = (
            str(record["asset"]),
            str(record["model_id"]),
            str(record["use"]),
        )
        expected_by_identity[identity] = record
        expected_records.append(record)

    observed_identities: set[tuple[str, str, str]] = set()
    for row in _committed_checkpoint_rows(
        checkpoint_path,
        row_kind="record",
        committed_hashes=committed_hashes,
    ):
        identity = (
            str(row.get("asset")),
            str(row.get("model_id")),
            str(row.get("use")),
        )
        if identity in observed_identities:
            raise ValueError(
                f"M4 resume checkpoint contains duplicate record: {identity}"
            )
        observed_identities.add(identity)
        expected = expected_by_identity.get(identity)
        if expected is None:
            if identity in known_identities:
                # A transient attempt for this known job was intentionally
                # dropped. Its old projection is stale and will be regenerated
                # after the missing attempt has been retried.
                continue
            raise ValueError(
                f"M4 resume record has no complete attempt group: {identity}"
            )
        if not _resume_record_matches_or_policy_migration(row, expected):
            raise ValueError(
                f"M4 resume record does not match recomputed attempts: {identity}"
            )
    return expected_records


def _resume_record_matches_or_policy_migration(
    observed: dict[str, Any], expected: dict[str, Any]
) -> bool:
    """Accept only reviewed derived-record migrations over fixed attempts."""

    if _canonical_json(observed) == _canonical_json(expected):
        return True
    if _resume_record_matches_contact_policy_rebind(observed, expected):
        return True
    # Global v18/v19 labels are no longer part of record identity.  Older
    # checkpoints may still contain the historical derived field.
    ignored = {
        "config_hash",
        "skill_execution_policy_versions",
    }
    observed_stable = {
        key: value for key, value in observed.items() if key not in ignored
    }
    expected_stable = {
        key: value for key, value in expected.items() if key not in ignored
    }
    return _canonical_json(observed_stable) == _canonical_json(expected_stable)


def _resume_record_matches_contact_policy_rebind(
    observed: dict[str, Any], expected: dict[str, Any]
) -> bool:
    """Accept removal of only the redundant missing-policy failure."""

    observed_contact_failures = set(observed.get("contact_evidence_failures") or [])
    expected_contact_failures = set(expected.get("contact_evidence_failures") or [])
    if observed_contact_failures != {
        *expected_contact_failures,
        "contact_contract_policy_version_mismatch",
    }:
        return False
    if not isinstance(expected.get("contact_contract_binding"), dict):
        return False
    contact_ignored = {
        "config_hash",
        "contact_evidence_failures",
        "outcome",
        "pass",
        "verdict",
    }
    observed_contact_stable = {
        key: value for key, value in observed.items() if key not in contact_ignored
    }
    expected_contact_stable = {
        key: value for key, value in expected.items() if key not in contact_ignored
    }
    return _canonical_json(observed_contact_stable) == _canonical_json(
        expected_contact_stable
    )


def _apply_torch_extensions_dir(
    env: dict[str, str], torch_extensions_dir: Path | None
) -> None:
    """Apply an explicit cache override without defeating PyTorch defaults.

    When no CLI override is supplied, preserving the inherited environment is
    important: an exported ``TORCH_EXTENSIONS_DIR`` remains authoritative, and
    an otherwise-unset variable lets PyTorch use its stable versioned user
    cache.
    """

    if torch_extensions_dir is not None:
        env["TORCH_EXTENSIONS_DIR"] = str(torch_extensions_dir)


def _apply_warp_cache_path(env: dict[str, str], warp_cache_path: Path) -> None:
    """Give one serial GPU slot an isolated, reusable Warp compile cache."""

    warp_cache_path.mkdir(parents=True, exist_ok=True)
    env["WARP_CACHE_PATH"] = str(warp_cache_path)


def _run_attempt(
    *,
    python_executable: str,
    asset_catalog_path: Path,
    job: dict[str, Any],
    seed: int,
    requested_arm: str | None,
    task_config: str,
    run_dir: Path,
    timeout_seconds: int,
    torch_extensions_dir: Path | None,
    warmup: bool = False,
) -> dict[str, Any]:
    started = time.monotonic()
    cmd = [
        python_executable,
        "-m",
        "if_benchmark.asset_probe_worker",
        "--asset-catalog",
        str(asset_catalog_path),
        "--asset",
        str(job["asset"]),
        "--model-id",
        str(job["model_id"]),
        "--use",
        str(job["use"]),
        "--seed",
        str(seed),
        "--task-config",
        task_config,
        "--run-dir",
        str(run_dir),
    ]
    if requested_arm is not None:
        cmd.extend(["--requested-arm", str(requested_arm)])
    probe_job = _probe_job_payload(job)
    if probe_job is not None:
        probe_jobs_dir = run_dir / "probe_jobs"
        probe_jobs_dir.mkdir(parents=True, exist_ok=True)
        probe_job = {
            **probe_job,
            **(
                {"requested_arm": requested_arm}
                if requested_arm is not None
                else {}
            ),
        }
        probe_job_path = probe_jobs_dir / f"{_safe_log_stem(job, seed, requested_arm)}.json"
        _write_json(probe_job_path, probe_job)
        cmd.extend(["--probe-job-file", str(probe_job_path)])
    base = {
        "job_id": job.get("job_id"),
        "asset": job.get("asset"),
        "model_id": str(job.get("model_id")),
        "use": job.get("use"),
        "requested_arm": requested_arm,
        "seed": int(seed),
        "plan_success": False,
        "success": False,
        "failure_reason": None,
        "worker_returncode": None,
        "duration_sec": None,
        "warmup": bool(warmup),
    }
    _apply_expected_articulation_provenance(base, job)
    log_stem = _safe_log_stem(job, seed, requested_arm)
    stdout_path = run_dir / "logs" / f"{log_stem}.stdout.log"
    stderr_path = run_dir / "logs" / f"{log_stem}.stderr.log"
    base["stdout_log"] = _path_string(stdout_path)
    base["stderr_log"] = _path_string(stderr_path)
    env = os.environ.copy()
    env.setdefault("PYTHONWARNINGS", "ignore::UserWarning")
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    _apply_torch_extensions_dir(env, torch_extensions_dir)
    env.setdefault("MPLCONFIGDIR", str(run_dir / "matplotlib"))
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout_f, stderr_path.open("w", encoding="utf-8") as stderr_f:
            proc = subprocess.Popen(
                cmd,
                cwd=Path.cwd(),
                env=env,
                text=True,
                stdout=stdout_f,
                stderr=stderr_f,
            )
            try:
                returncode = proc.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                raise
    except subprocess.TimeoutExpired:
        base.update(
            {
                "failure_reason": "worker_timeout",
                "stdout_tail": _tail_file(stdout_path),
                "stderr_tail": _tail_file(stderr_path),
                "duration_sec": _clean_float(time.monotonic() - started),
            }
        )
        return base
    base["worker_returncode"] = returncode
    stdout_text = _read_text(stdout_path)
    stderr_text = _read_text(stderr_path)
    parsed = _parse_worker_json(stdout_text)
    if parsed is None:
        base.update(
            {
                "failure_reason": "worker_no_json_result" if returncode == 0 else "worker_process_failed",
                "stdout_tail": _tail(stdout_text),
                "stderr_tail": _tail(stderr_text),
                "duration_sec": _clean_float(time.monotonic() - started),
            }
        )
        return base
    base.update(parsed)
    _apply_expected_articulation_provenance(base, job)
    if returncode != 0 and base.get("failure_reason") in {None, "ok"}:
        base["failure_reason"] = "worker_process_failed"
    base["job_id"] = job.get("job_id")
    base["asset"] = job.get("asset")
    base["model_id"] = str(job.get("model_id"))
    base["use"] = job.get("use")
    base["requested_arm"] = requested_arm
    base["seed"] = int(seed)
    base["warmup"] = bool(warmup)
    if not base.get("attempt_id"):
        arm_suffix = f"|arm:{requested_arm}" if requested_arm is not None else ""
        base["attempt_id"] = f"{job.get('job_id')}{arm_suffix}|seed{seed}"
    base["stdout_tail"] = _tail(stdout_text)
    base["stderr_tail"] = _tail(stderr_text)
    base["duration_sec"] = _clean_float(time.monotonic() - started)
    return base


def _record_from_attempts(
    job: dict[str, Any],
    attempts: list[dict[str, Any]],
    threshold: float,
    seeds: list[int],
) -> dict[str, Any]:
    attempt_count = len(attempts)
    success_contract = SuccessContract.load()
    excluded_attempts = [attempt for attempt in attempts if _attempt_is_blocking(attempt)]
    denominator_rows = [
        attempt for attempt in attempts if not _attempt_is_blocking(attempt)
    ]
    execution_policy_evidence_failures = sorted(
        {
            failure
            for attempt in denominator_rows
            for failure in m4_worker_execution_policy_manifest_errors(
                attempt.get("execution_policy_manifest"),
                use=str(job.get("use") or ""),
            )
        }
    )
    # Blocking attempts still ran under a concrete worker policy. Preserve
    # that provenance even when the statistical denominator is empty so
    # freeze can distinguish a current-policy blocker from a stale record.
    execution_policy_manifests = [
        attempt["execution_policy_manifest"]
        for attempt in attempts
        if isinstance(attempt.get("execution_policy_manifest"), dict)
    ]
    contact_binding = job.get("contact_contract_binding")
    contact_evidence_failures: list[str] = []
    if isinstance(contact_binding, dict):
        for attempt in denominator_rows:
            observed = (attempt.get("diagnostics") or {}).get(
                "contact_contract_preflight"
            )
            if not isinstance(observed, dict):
                contact_evidence_failures.append(
                    "contact_contract_preflight_missing"
                )
                continue
            for field in (
                "policy_version",
                "contact_contract_id",
                "contact_contract_hash",
                "contact_projection_hash",
                "candidate_use",
                "use_profile",
                "contact_count",
                "contact_groups",
            ):
                observed_value = (
                    resolved_contact_preflight_policy_version(
                        observed, contact_binding
                    )
                    if field == "policy_version"
                    else observed.get(field)
                )
                if observed_value != contact_binding.get(field):
                    contact_evidence_failures.append(
                        f"contact_contract_{field}_mismatch"
                    )
    contact_evidence_failures = sorted(set(contact_evidence_failures))
    articulation_evidence_failures: list[str] = []
    if str(job.get("use") or "") in ARTICULATION_EXECUTABLE_USES:
        expected_action = {
            "execution_profile_version": job.get(
                "articulation_execution_profile_version"
            ),
            "execution_profile_hash": job.get(
                "articulation_execution_profile_hash"
            ),
            "execution_profile_source_hash": job.get(
                "articulation_execution_profile_source_hash"
            ),
            "motion_policy_version": job.get(
                "articulation_motion_policy_version"
            ),
            "motion_policy_hash": job.get(
                "articulation_motion_policy_hash"
            ),
        }
        for attempt in denominator_rows:
            observed = (attempt.get("diagnostics") or {}).get(
                "articulation_action"
            )
            if not isinstance(observed, dict):
                articulation_evidence_failures.append(
                    "articulation_execution_preflight_missing"
                )
                continue
            for field, expected in expected_action.items():
                if expected is None or observed.get(field) != expected:
                    articulation_evidence_failures.append(
                        f"articulation_execution_{field}_mismatch"
                    )
    articulation_evidence_failures = sorted(
        set(articulation_evidence_failures)
    )
    expected_state_hash = (
        (job.get("canonical_initial_state") or {}).get(
            "canonical_state_hash"
        )
    )
    placement_evidence_failures = (
        sorted(
            {
                failure
                for attempt in denominator_rows
                for failure in m4_canonical_placement_witness_errors(
                    attempt.get("canonical_placement_witness"),
                    expected_use=str(job.get("use") or ""),
                    expected_canonical_state_hash=str(expected_state_hash),
                )
            }
        )
        if expected_state_hash is not None
        else []
    )
    placement_witness_hashes = (
        sorted(
            {
                str(
                    (attempt.get("canonical_placement_witness") or {}).get(
                        "witness_hash"
                    )
                )
                for attempt in denominator_rows
                if (attempt.get("canonical_placement_witness") or {}).get(
                    "witness_hash"
                )
            }
        )
        if expected_state_hash is not None
        else []
    )
    if (
        expected_state_hash is not None
        and denominator_rows
        and len(placement_witness_hashes) != 1
    ):
        placement_evidence_failures = sorted(
            {
                *placement_evidence_failures,
                "canonical_placement_witness_inconsistent",
            }
        )
    articulation_capability_projection: dict[str, Any] | None = None
    touch_capability_projection: dict[str, Any] | None = None
    articulation_requested_arm_sampling = (
        str(job.get("use") or "") in ARTICULATION_EXECUTABLE_USES
        and job.get("articulation_arm_sampling_version")
        == ARTICULATION_ARM_SAMPLING_VERSION
    )
    touch_requested_arm_sampling = (
        str(job.get("use") or "") in TOUCH_EXECUTABLE_USES
        and job.get("touch_arm_sampling_version")
        == TOUCH_ARM_SAMPLING_VERSION
    )
    requested_arm_sampling = bool(
        articulation_requested_arm_sampling or touch_requested_arm_sampling
    )
    if str(job.get("use") or "") in ARTICULATION_EXECUTABLE_USES:
        articulation_capability_projection = aggregate_articulation_capabilities(
            denominator_rows,
            use=str(job["use"]),
            denominator=len(denominator_rows),
            threshold=float(threshold),
            require_explicit_runtime_capability=True,
        )
        articulation_evidence_failures = sorted(
            {
                *articulation_evidence_failures,
                *articulation_capability_projection["failures"],
            }
        )
    touch_evidence_failures: list[str] = []
    if str(job.get("use") or "") in TOUCH_EXECUTABLE_USES:
        touch_capability_projection = aggregate_touch_capabilities(
            denominator_rows,
            use=str(job["use"]),
            threshold=float(threshold),
        )
        touch_evidence_failures = list(
            touch_capability_projection["failures"]
        )
    evidence_failures = sorted(
        {
            failure
            for attempt in denominator_rows
            for failure in _success_evidence_failures(attempt, success_contract)
        }
    )
    predicate_blocked_attempts = [
        attempt
        for attempt in denominator_rows
        if isinstance(attempt.get("predicate_result"), dict)
        and attempt["predicate_result"].get("status") == "blocked"
    ]
    denominator = len(denominator_rows)
    plan_success = sum(1 for attempt in denominator_rows if attempt.get("plan_success"))
    success = sum(1 for attempt in denominator_rows if attempt.get("success"))
    rate = success / denominator if denominator else 0.0
    articulation_arm_results: dict[str, dict[str, Any]] = {}
    if articulation_requested_arm_sampling:
        assert articulation_capability_projection is not None
        capable_arms = {
            str(row["arm"])
            for row in articulation_capability_projection["arm_capabilities"]
        }
        for arm in ARTICULATION_REQUESTED_ARMS:
            arm_attempts = [
                attempt
                for attempt in attempts
                if str(attempt.get("requested_arm") or "") == arm
            ]
            arm_blocked = [
                attempt for attempt in arm_attempts if _attempt_is_blocking(attempt)
            ]
            arm_denominator = [
                attempt
                for attempt in arm_attempts
                if not _attempt_is_blocking(attempt)
            ]
            arm_success = sum(
                1 for attempt in arm_denominator if attempt.get("success")
            )
            arm_rate = (
                arm_success / len(arm_denominator) if arm_denominator else 0.0
            )
            arm_passed = arm in capable_arms and not arm_blocked
            articulation_arm_results[arm] = {
                "attempts": len(arm_attempts),
                "denominator": len(arm_denominator),
                "excluded_blocking_attempts": len(arm_blocked),
                "plan_success": sum(
                    1
                    for attempt in arm_denominator
                    if attempt.get("plan_success")
                ),
                "success": arm_success,
                "rate": _clean_float(arm_rate),
                "threshold": _clean_float(threshold),
                "outcome": (
                    "blocked"
                    if arm_blocked
                    else "pass"
                    if arm_passed
                    else "fail"
                ),
                "pass": bool(arm_passed),
            }
        passing_arms = {
            arm for arm, result in articulation_arm_results.items() if result["pass"]
        }
        articulation_capability_projection["arm_capabilities"] = [
            row
            for row in articulation_capability_projection["arm_capabilities"]
            if str(row["arm"]) in passing_arms
        ]
        eligible_rates = [
            float(result["rate"])
            for result in articulation_arm_results.values()
            if result["excluded_blocking_attempts"] == 0
        ]
        rate = max(eligible_rates, default=0.0)
    touch_arm_results: dict[str, dict[str, Any]] = {}
    if touch_requested_arm_sampling:
        assert touch_capability_projection is not None
        touch_capable_arms = {
            str(row["arm"])
            for row in touch_capability_projection["arm_capabilities"]
        }
        for arm in TOUCH_REQUESTED_ARMS:
            arm_attempts = [
                attempt
                for attempt in attempts
                if str(attempt.get("requested_arm") or "") == arm
            ]
            arm_blocked = [
                attempt for attempt in arm_attempts if _attempt_is_blocking(attempt)
            ]
            arm_denominator = [
                attempt
                for attempt in arm_attempts
                if not _attempt_is_blocking(attempt)
            ]
            arm_success = sum(
                1 for attempt in arm_denominator if attempt.get("success")
            )
            arm_rate = (
                arm_success / len(arm_denominator) if arm_denominator else 0.0
            )
            arm_passed = arm in touch_capable_arms and not arm_blocked
            touch_arm_results[arm] = {
                "attempts": len(arm_attempts),
                "denominator": len(arm_denominator),
                "excluded_blocking_attempts": len(arm_blocked),
                "plan_success": sum(
                    1
                    for attempt in arm_denominator
                    if attempt.get("plan_success")
                ),
                "success": arm_success,
                "rate": _clean_float(arm_rate),
                "min_rate": _clean_float(arm_rate),
                "threshold": _clean_float(threshold),
                "outcome": (
                    "blocked"
                    if arm_blocked
                    else "pass"
                    if arm_passed
                    else "fail"
                ),
                "pass": bool(arm_passed),
            }
        passing_touch_arms = {
            arm for arm, result in touch_arm_results.items() if result["pass"]
        }
        touch_capability_projection["arm_capabilities"] = [
            row
            for row in touch_capability_projection["arm_capabilities"]
            if str(row["arm"]) in passing_touch_arms
        ]
        touch_capability_projection["arm_results"] = touch_arm_results
        eligible_rates = [
            float(result["rate"])
            for result in touch_arm_results.values()
            if result["excluded_blocking_attempts"] == 0
        ]
        rate = max(eligible_rates, default=0.0)
    fail_reasons = Counter(
        _normalize_failure_reason(attempt.get("failure_reason"))
        for attempt in attempts
        if not attempt.get("success")
    )
    fail_reasons.pop("ok", None)
    variant_name = f"{job['use']}({job['asset']}:{job['model_id']})"
    global_evidence_blocked = bool(
        evidence_failures
        or predicate_blocked_attempts
        or contact_evidence_failures
        or articulation_evidence_failures
        or touch_evidence_failures
        or placement_evidence_failures
        or execution_policy_evidence_failures
    )
    articulation_has_capability = bool(
        articulation_requested_arm_sampling
        and articulation_capability_projection
        and articulation_capability_projection["arm_capabilities"]
    )
    touch_has_capability = bool(
        touch_requested_arm_sampling
        and touch_capability_projection
        and touch_capability_projection["arm_capabilities"]
    )
    requested_arm_has_capability = bool(
        articulation_has_capability or touch_has_capability
    )
    blocked = bool(
        global_evidence_blocked
        or (
            excluded_attempts
            and (not requested_arm_sampling or not requested_arm_has_capability)
        )
    )
    verdict = (
        _dominant_failure(fail_reasons)
        if blocked
        else "ok"
        if rate >= threshold
        else _dominant_failure(fail_reasons)
    )
    row = {
        "scope": "asset",
        "stage": STAGE_CANONICAL,
        "asset": str(job["asset"]),
        "model_id": str(job["model_id"]),
        "use": str(job["use"]),
        "outcome": "blocked" if blocked else ("pass" if rate >= threshold else "fail"),
        "pass": bool(not blocked and rate >= threshold),
        "verdict": verdict,
        "threshold": _clean_float(threshold),
        "seeds": [int(seed) for seed in seeds],
        "attempts": attempt_count,
        "denominator": denominator,
        "excluded_source_or_internal_attempts": sum(
            1
            for attempt in excluded_attempts
            if str(attempt.get("failure_reason") or "").startswith(
                ("source_", "internal_")
            )
        ),
        "excluded_blocking_attempts": len(excluded_attempts),
        "predicate_blocked_attempts": len(predicate_blocked_attempts),
        "plan_success": plan_success,
        "success": success,
        "per_variant": (
            {
                f"{variant_name}|arm:{arm}": dict(result)
                for arm, result in (
                    articulation_arm_results or touch_arm_results
                ).items()
            }
            if requested_arm_sampling
            else {
                variant_name: {
                "attempts": attempt_count,
                "denominator": denominator,
                "plan_success": plan_success,
                "success": success,
                "rate": _clean_float(rate),
                }
            }
        ),
        "min_rate": _clean_float(rate),
        "fail_reasons": dict(sorted(fail_reasons.items())),
        "initial_state_selection_version": job.get(
            "initial_state_selection_version"
        ),
        "initial_state_selection_hash": job.get("initial_state_selection_hash"),
        "success_contract_versions": sorted(
            {
                str(attempt.get("success_contract_version"))
                for attempt in denominator_rows
                if attempt.get("success_contract_version") is not None
            }
        ),
        "success_contract_hashes": sorted(
            {
                str(attempt.get("success_contract_hash"))
                for attempt in denominator_rows
                if attempt.get("success_contract_hash") is not None
            }
        ),
        "success_snapshot_hashes": sorted(
            {
                str(attempt.get("success_snapshot_hash"))
                for attempt in denominator_rows
                if attempt.get("success_snapshot_hash") is not None
            }
        ),
        "success_evidence_failures": evidence_failures,
        "canonical_placement_evidence_failures": (
            placement_evidence_failures
        ),
        "canonical_placement_policy_version": (
            PICK_OPERATION_PLACEMENT_POLICY_VERSION
        ),
        "canonical_placement_policy_hash": (
            PICK_OPERATION_PLACEMENT_POLICY_HASH
        ),
        "canonical_placement_witness_hashes": placement_witness_hashes,
        "execution_policy_evidence_failures": (
            execution_policy_evidence_failures
        ),
        "worker_execution_policy_manifest_hashes": sorted(
            {
                str(manifest.get("manifest_hash"))
                for manifest in execution_policy_manifests
                if manifest.get("manifest_hash")
            }
        ),
    }
    if isinstance(contact_binding, dict):
        row["contact_contract_binding"] = dict(contact_binding)
        row["contact_contract_hash"] = contact_binding.get(
            "contact_contract_hash"
        )
        row["contact_evidence_failures"] = contact_evidence_failures
    if str(job["use"]) in TOUCH_EXECUTABLE_USES:
        assert touch_capability_projection is not None
        row["touch_evidence_failures"] = touch_evidence_failures
        row["touch_capability_version"] = TOUCH_CAPABILITY_VERSION
        row["touch_arm_sampling_version"] = TOUCH_ARM_SAMPLING_VERSION
        row["touch_placement_policy_version"] = (
            M4_TOUCH_PLACEMENT_POLICY_VERSION
        )
        row["touch_placement_policy_hash"] = M4_TOUCH_PLACEMENT_POLICY_HASH
        row["touch_requested_arms"] = list(TOUCH_REQUESTED_ARMS)
        row["touch_arm_results"] = touch_arm_results
        row["touch_arm_capabilities"] = touch_capability_projection[
            "arm_capabilities"
        ]
        row["touch_canonical_placement_witness_hash"] = (
            touch_capability_projection[
                "canonical_placement_witness_hash"
            ]
        )
    if str(job["use"]) in {"open", "close", "switch_on", "switch_off"}:
        profile_versions = sorted(
            {
                str(attempt.get("articulation_execution_profile_version") or "")
                for attempt in attempts
                if attempt.get("articulation_execution_profile_version")
            }
        )
        profile_hashes = sorted(
            {
                str(attempt.get("articulation_execution_profile_hash") or "")
                for attempt in attempts
                if attempt.get("articulation_execution_profile_hash")
            }
        )
        source_hashes = sorted(
            {
                str(attempt.get("articulation_execution_profile_source_hash") or "")
                for attempt in attempts
                if attempt.get("articulation_execution_profile_source_hash")
            }
        )
        row["articulation_execution_profile_versions"] = profile_versions
        row["articulation_execution_profile_hashes"] = profile_hashes
        row["articulation_execution_profile_source_hashes"] = source_hashes
        row["articulation_motion_policy_versions"] = sorted(
            {
                str(attempt.get("articulation_motion_policy_version") or "")
                for attempt in attempts
                if attempt.get("articulation_motion_policy_version")
            }
        )
        row["articulation_motion_policy_hashes"] = sorted(
            {
                str(attempt.get("articulation_motion_policy_hash") or "")
                for attempt in attempts
                if attempt.get("articulation_motion_policy_hash")
            }
        )
        row["articulation_evidence_failures"] = (
            articulation_evidence_failures
        )
        assert articulation_capability_projection is not None
        row["articulation_capability_version"] = (
            articulation_capability_projection["capability_version"]
        )
        if requested_arm_sampling:
            row["articulation_arm_sampling_version"] = (
                ARTICULATION_ARM_SAMPLING_VERSION
            )
            row["articulation_requested_arms"] = list(
                ARTICULATION_REQUESTED_ARMS
            )
            row["articulation_arm_results"] = articulation_arm_results
        row["articulation_arm_capabilities"] = (
            articulation_capability_projection["arm_capabilities"]
        )
    row["config_hash"] = _row_config_hash(row)
    return row


def _attempt_is_blocking(attempt: dict[str, Any]) -> bool:
    """Return whether an attempt stopped before terminal checker evidence."""

    return m4_attempt_is_blocking(attempt)


def _success_evidence_failures(
    attempt: dict[str, Any], contract: SuccessContract
) -> list[str]:
    """Audit one completed runtime attempt's canonical checker binding."""

    failures: list[str] = []
    if attempt.get("success_contract_version") != contract.version:
        failures.append("success_contract_version_missing_or_mismatch")
    if attempt.get("success_contract_hash") != contract.hash:
        failures.append("success_contract_hash_missing_or_mismatch")
    snapshot_hash = attempt.get("success_snapshot_hash")
    if not isinstance(snapshot_hash, str) or not snapshot_hash:
        failures.append("success_snapshot_hash_missing")
    predicate = attempt.get("predicate_result")
    if not isinstance(predicate, dict):
        failures.append("predicate_result_missing")
        return failures
    if predicate.get("success_contract_version") != contract.version:
        failures.append("predicate_contract_version_mismatch")
    if predicate.get("success_contract_hash") != contract.hash:
        failures.append("predicate_contract_hash_mismatch")
    if predicate.get("success_snapshot_hash") != snapshot_hash:
        failures.append("predicate_snapshot_hash_mismatch")
    status = predicate.get("status")
    if status not in {"passed", "failed", "blocked"}:
        failures.append("predicate_status_invalid")
    checker_success = status == "passed"
    if attempt.get("checker_success") is not checker_success:
        failures.append("checker_success_mismatch")
    if bool(attempt.get("success")) != bool(
        attempt.get("plan_success") and checker_success
    ):
        failures.append("attempt_gate_mismatch")
    return failures


def _summary(records: list[dict[str, Any]], attempts: list[dict[str, Any]]) -> dict[str, Any]:
    by_use = Counter(record["use"] for record in records)
    validated = sum(1 for record in records if record.get("pass"))
    failure_reasons = Counter(
        _normalize_failure_reason(attempt.get("failure_reason"))
        for attempt in attempts
        if not attempt.get("success")
    )
    failure_reasons.pop("ok", None)
    return {
        "record_count": len(records),
        "attempt_count": len(attempts),
        "validated_count": validated,
        "rejected_count": len(records) - validated,
        "record_count_by_use": {use: by_use.get(use, 0) for use in EXECUTABLE_USES},
        "attempt_failure_reasons": dict(sorted(failure_reasons.items())),
    }


def compare_attempt_equivalence(
    isolated_attempts: list[dict[str, Any]],
    batch_attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    isolated_by_key = {_attempt_equivalence_key(attempt): attempt for attempt in isolated_attempts}
    batch_by_key = {_attempt_equivalence_key(attempt): attempt for attempt in batch_attempts}
    isolated_keys = set(isolated_by_key)
    batch_keys = set(batch_by_key)
    common_keys = sorted(isolated_keys & batch_keys)
    mismatches: list[dict[str, Any]] = []
    for key in common_keys:
        isolated = isolated_by_key[key]
        batch = batch_by_key[key]
        field_mismatches = _attempt_field_mismatches(isolated, batch)
        if field_mismatches:
            mismatches.append(
                {
                    "attempt_key": list(key),
                    "asset": isolated.get("asset"),
                    "model_id": str(isolated.get("model_id")),
                    "use": isolated.get("use"),
                    "seed": isolated.get("seed"),
                    "field_mismatches": field_mismatches,
                }
            )
    return {
        "attempt_count": len(common_keys),
        "isolated_attempt_count": len(isolated_attempts),
        "batch_attempt_count": len(batch_attempts),
        "missing_in_batch": [list(key) for key in sorted(isolated_keys - batch_keys)],
        "missing_in_isolated": [list(key) for key in sorted(batch_keys - isolated_keys)],
        "mismatches": mismatches,
    }


def _attempt_equivalence_key(attempt: dict[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(attempt.get("asset")),
        str(attempt.get("model_id")),
        str(attempt.get("use")),
        int(attempt.get("seed")),
    )


def _attempt_field_mismatches(isolated: dict[str, Any], batch: dict[str, Any]) -> dict[str, Any]:
    mismatches: dict[str, Any] = {}
    for key in ("plan_success", "success", "failure_reason"):
        if isolated.get(key) != batch.get(key):
            mismatches[key] = {
                "isolated": isolated.get(key),
                "batch_reuse": batch.get(key),
            }
    isolated_metrics = isolated.get("metrics") or {}
    batch_metrics = batch.get("metrics") or {}
    if _canonical_json(isolated_metrics) != _canonical_json(batch_metrics):
        mismatches["metrics"] = {
            "isolated": isolated_metrics,
            "batch_reuse": batch_metrics,
        }
    if _canonical_json(isolated.get("execution_policy_manifest")) != (
        _canonical_json(batch.get("execution_policy_manifest"))
    ):
        mismatches["execution_policy_manifest"] = {
            "isolated": isolated.get("execution_policy_manifest"),
            "batch_reuse": batch.get("execution_policy_manifest"),
        }
    return mismatches


def _write_timing_summary(
    path: Path,
    *,
    attempts: list[dict[str, Any]],
    records: list[dict[str, Any]],
    batch_size: int,
    reuse_task_env: bool,
    parallel_workers: int = 1,
    cuda_devices: list[str] | None = None,
    source_kind: str = FRESH_CANONICAL_SOURCE_KIND,
) -> dict[str, Any]:
    timing_summary = {
        "schema_version": 1,
        "source_kind": source_kind,
        "attempt_count": len(attempts),
        "record_count": len(records),
        "batch_size": int(batch_size),
        "reuse_task_env": bool(reuse_task_env),
        "parallel_workers": int(parallel_workers),
        "cuda_devices": list(cuda_devices or []),
        "timing_keys": _timing_key_stats(attempts),
        "by_use": _timing_stats_by_use(attempts),
        "slowest_attempts": _slowest_attempts(attempts),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(timing_summary, f, sort_keys=False, allow_unicode=True, default_flow_style=False, width=100)
    return timing_summary


def _timing_key_stats(attempts: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    values_by_key: dict[str, list[float]] = {}
    for attempt in attempts:
        timings = attempt.get("timings") or {}
        if not isinstance(timings, dict):
            continue
        for key, value in timings.items():
            if isinstance(value, bool):
                continue
            try:
                values_by_key.setdefault(str(key), []).append(float(value))
            except (TypeError, ValueError):
                continue
    return {key: _numeric_stats(values) for key, values in sorted(values_by_key.items())}


def _timing_stats_by_use(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    by_use: dict[str, list[dict[str, Any]]] = {}
    for attempt in attempts:
        by_use.setdefault(str(attempt.get("use")), []).append(attempt)
    return {
        use: {
            "attempt_count": len(use_attempts),
            "timing_keys": _timing_key_stats(use_attempts),
        }
        for use, use_attempts in sorted(by_use.items())
    }


def _numeric_stats(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": _clean_float(ordered[0]),
        "p50": _clean_float(_percentile(ordered, 0.50)),
        "p90": _clean_float(_percentile(ordered, 0.90)),
        "max": _clean_float(ordered[-1]),
        "mean": _clean_float(sum(ordered) / len(ordered)),
    }


def _percentile(values: list[float], percentile: float) -> float:
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def _slowest_attempts(attempts: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    def duration(attempt: dict[str, Any]) -> float:
        try:
            return float(attempt.get("duration_sec") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    rows: list[dict[str, Any]] = []
    for attempt in sorted(attempts, key=duration, reverse=True)[:limit]:
        rows.append(
            {
                "job_id": attempt.get("job_id"),
                "asset": attempt.get("asset"),
                "model_id": str(attempt.get("model_id")),
                "use": attempt.get("use"),
                "seed": attempt.get("seed"),
                "success": bool(attempt.get("success")),
                "failure_reason": attempt.get("failure_reason"),
                "duration_sec": _clean_float(duration(attempt)),
            }
        )
    return rows


def _parse_worker_json(stdout: str | bytes | None) -> dict[str, Any] | None:
    if isinstance(stdout, bytes):
        stdout = stdout.decode("utf-8", errors="replace")
    for line in reversed((stdout or "").splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        return data if isinstance(data, dict) else None
    return None


def _normalize_failure_reason(reason: Any) -> str:
    raw = str(reason or "unknown_failure").strip().lower()
    raw = raw.replace(" ", "_")
    allowed = {
        "ok",
        "worker_import_failed",
        "worker_exception",
        "worker_physical_collision_validation_failed",
        "worker_process_failed",
        "worker_timeout",
        "worker_no_json_result",
        "setup_failed",
        "actor_load_failed",
        "planner_failed",
        "checker_false",
        "missing_candidate_metadata",
        "unsupported_use",
    }
    if raw in allowed:
        return raw
    if raw.startswith("articulation_"):
        return raw
    if "curobo" in raw or "planner" in raw or "cuda" in raw:
        return "planner_failed"
    if "unstable" in raw:
        return "setup_failed"
    return "worker_exception"


def _dominant_failure(fail_reasons: Counter[str]) -> str:
    if not fail_reasons:
        return "checker_false"
    reason, _ = fail_reasons.most_common(1)[0]
    return reason


def _tail(text: str | bytes | None, limit: int = 4000) -> str:
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    return (text or "")[-limit:]


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def _tail_file(path: Path, limit: int = 4000) -> str:
    return _tail(_read_text(path), limit=limit)


def _safe_log_stem(
    job: dict[str, Any], seed: int, requested_arm: str | None = None
) -> str:
    arm_suffix = f"|arm:{requested_arm}" if requested_arm is not None else ""
    raw = f"{job.get('job_id', 'probe')}{arm_suffix}|seed{seed}"
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in raw)


def _load_yaml_required(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected YAML mapping in {path}")
    return data


def _load_json_required(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"required JSON file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON file: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON mapping in {path}")
    return data


def _write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, sort_keys=True, ensure_ascii=False, indent=2)
        f.write("\n")


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, sort_keys=True, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _append_jsonl(path: Path, data: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        json.dump(data, f, sort_keys=True, ensure_ascii=False)
        f.write("\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _artifact_hash(data: dict[str, Any], hash_key: str) -> str:
    payload = {key: value for key, value in data.items() if key != hash_key}
    return _sha256_bytes(_canonical_json(payload).encode("utf-8"))


def _validate_pick_operation_placement_binding(plan: dict[str, Any]) -> None:
    if plan.get("pick_operation_placement_policy_version") != (
        PICK_OPERATION_PLACEMENT_POLICY_VERSION
    ):
        raise ValueError("probe plan Pick operation-placement policy is stale")
    if plan.get("pick_operation_placement_policy_hash") != (
        PICK_OPERATION_PLACEMENT_POLICY_HASH
    ):
        raise ValueError("probe plan Pick operation-placement policy is stale")
    if any(
        str(job.get("use") or "") in TOUCH_EXECUTABLE_USES
        for job in plan.get("jobs") or []
        if isinstance(job, dict)
    ) and (
        plan.get("touch_placement_policy_version")
        != M4_TOUCH_PLACEMENT_POLICY_VERSION
        or plan.get("touch_placement_policy_hash")
        != M4_TOUCH_PLACEMENT_POLICY_HASH
    ):
        raise ValueError("probe plan touch placement policy is stale")


def _evidence_id(report: dict[str, Any]) -> str:
    payload = {key: value for key, value in report.items() if key not in {"evidence_id", "report_hash"}}
    digest = _sha256_bytes(_canonical_json(payload).encode("utf-8"))[:20]
    return f"canonical_skill_{digest}"


def _row_config_hash(row: dict[str, Any]) -> str:
    keys = (
        "scope",
        "stage",
        "use",
        "threshold",
        "seeds",
        "articulation_arm_sampling_version",
        "articulation_requested_arms",
        "touch_arm_sampling_version",
        "touch_requested_arms",
        "touch_placement_policy_version",
        "touch_placement_policy_hash",
    )
    return _sha256_bytes(_canonical_json({key: row.get(key) for key in keys}).encode("utf-8"))


def _clean_float(value: float) -> float:
    return round(float(value), 12)


def _validate_semantic_exclusion_binding(
    plan: dict[str, Any], asset_catalog_path: Path
) -> dict[str, Any] | None:
    try:
        uses_default_catalog = (
            asset_catalog_path.resolve() == DEFAULT_ASSET_CATALOG_PATH.resolve()
        )
    except OSError:
        uses_default_catalog = asset_catalog_path == DEFAULT_ASSET_CATALOG_PATH
    binding = plan.get("semantic_exclusion_policy")
    if not uses_default_catalog:
        if binding is not None:
            raise ValueError(
                "custom asset catalog cannot use the default semantic-exclusion binding"
            )
        return None
    policy = load_asset_semantic_exclusions(
        DEFAULT_SEMANTIC_EXCLUSIONS_PATH,
        asset_catalog_path=asset_catalog_path,
    )
    if binding != semantic_exclusion_identity(policy):
        raise ValueError("probe plan semantic-exclusion policy is missing or stale")
    return policy


def _path_string(path: Path | None) -> str | None:
    if path is None:
        return None
    return str(path).replace("\\", "/")


def parse_seed_list(value: str) -> list[int]:
    try:
        seeds = [
            int(part.strip()) for part in value.split(",") if part.strip()
        ]
        return _validated_probe_seeds(seeds)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_cuda_devices(value: str | None) -> list[str] | None:
    if value is None:
        return None
    devices = [part.strip() for part in value.split(",") if part.strip()]
    return _normalize_cuda_device_selectors(devices) or None


def parse_thresholds(values: list[str] | None) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for value in values or []:
        try:
            use, raw_threshold = value.split("=", 1)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"threshold must be USE=VALUE: {value}"
            ) from exc
        use = use.strip()
        if use not in EXECUTABLE_USES:
            raise argparse.ArgumentTypeError(f"unsupported use threshold: {use}")
        if use in thresholds:
            raise argparse.ArgumentTypeError(f"duplicate use threshold: {use}")
        try:
            thresholds[use] = float(raw_threshold)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"invalid threshold for {use}: {raw_threshold}"
            ) from exc
    try:
        return _validated_probe_thresholds(thresholds)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _validated_probe_seeds(seeds: list[int]) -> list[int]:
    if not seeds:
        raise ValueError("M4 probe seeds must not be empty")
    if any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds):
        raise ValueError("M4 probe seeds must be integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("M4 probe seeds must be unique")
    return list(seeds)


def _validated_probe_thresholds(
    thresholds: dict[str, float],
) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for use, value in thresholds.items():
        if use not in EXECUTABLE_USES:
            raise ValueError(f"unsupported M4 probe threshold use: {use}")
        if isinstance(value, bool):
            raise ValueError(f"M4 probe threshold for {use} must be numeric")
        try:
            threshold = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"M4 probe threshold for {use} must be numeric"
            ) from exc
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"M4 probe threshold for {use} must be finite in [0, 1]"
            )
        normalized[use] = threshold
    return normalized
