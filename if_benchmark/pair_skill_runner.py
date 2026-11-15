"""Batch runner for Milestone 5 canonical pair-skill validation."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import yaml

from .pair_skill_contract import INFRASTRUCTURE_FAILURES, aggregate_skill_attempts
from .pair_validation import (
    ACTIVE_RELATIONS,
    CANONICAL_SOURCE_KIND,
    RELATIONS,
    REPORT_SCHEMA_VERSION,
    STAGE_CANONICAL,
    _artifact_hash,
    _evidence_id,
    _path_string,
    _row_config_hash,
    config_hash,
    require_active_relation_selection,
    skill_relation_policy_hash,
    skill_probe_policy_hash,
)


DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_THRESHOLDS = {relation: 0.6 for relation in RELATIONS}
DEFAULT_TIMEOUT_SECONDS = 300
DEFAULT_WARMUP_TIMEOUT_SECONDS = 1800
# Canonical skill probes pay a large Python/CuRobo cold-start cost.  The old
# default of four attempts restarted that stack thousands of times on a full
# run.  Ninety-six keeps failure/retry payloads bounded while amortizing one
# cold start over 32 complete three-seed jobs.
DEFAULT_BATCH_SIZE = 96
PROGRESS_EMIT_INTERVAL_SECONDS = 5.0
# CuRobo applies a ten-second wall-clock budget inside each motion query.
# Above eight concurrent workers on one A800, fixed-sample canaries returned
# load-induced no-path results instead of the baseline trajectories.  Keep the
# evidence runner inside the validated semantic envelope; higher outer
# concurrency requires a separately validated phase-level planner gate.
MAX_VALIDATED_WORKERS_PER_CUDA_DEVICE = 8
RESUME_IDENTITY_FIELDS = (
    "schema_version",
    "source_kind",
    "source_probe_plan_path",
    "source_probe_plan_hash",
    "source_validated_pose_set_hash",
    "source_initial_state_selection_path",
    "initial_state_selection_version",
    "initial_state_selection_hash",
    "config_hash",
    "source_rubric_hash",
    "skill_probe_policy_hash",
    "skill_relation_policy_hashes",
    "selected_job_ids_hash",
    "selected_job_count",
    "seeds",
    "thresholds",
    "task_config",
)


def run_skill_probe_plan(
    *,
    plan_path: Path,
    reports_dir: Path,
    run_dir: Path,
    python_executable: str | None = None,
    task_config: str = "demo_clean",
    seeds: list[int] | None = None,
    thresholds: dict[str, float] | None = None,
    relations: set[str] | None = None,
    pair_keys: set[str] | None = None,
    profile_ids: set[str] | None = None,
    start_index: int = 0,
    max_jobs: int | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    warmup_timeout_seconds: int = DEFAULT_WARMUP_TIMEOUT_SECONDS,
    torch_extensions_dir: Path | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    reuse_task_env: bool = True,
    parallel_workers: int = 1,
    cuda_devices: list[str] | None = None,
    resume: bool = False,
    clear_existing_reports: bool = False,
) -> dict[str, Any]:
    require_active_relation_selection(relations=relations, pair_keys=pair_keys)
    plan = _load_yaml(plan_path)
    _validate_plan(plan)
    jobs = _select_jobs(
        plan.get("jobs") or [],
        relations=set(ACTIVE_RELATIONS) if relations is None else relations,
        pair_keys=pair_keys,
        profile_ids=profile_ids,
        start_index=start_index,
        max_jobs=max_jobs,
    )
    probe_seeds = [int(seed) for seed in (seeds or DEFAULT_SEEDS)]
    probe_thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    run_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / "checkpoints"
    batch_dir = run_dir / "batches"
    log_dir = run_dir / "logs"
    for path in (checkpoint_dir, batch_dir, log_dir):
        path.mkdir(parents=True, exist_ok=True)
    attempts_path = checkpoint_dir / "attempts.jsonl"
    records_path = checkpoint_dir / "records.jsonl"
    metadata_path = checkpoint_dir / "run_metadata.json"
    progress_path = run_dir / "progress.json"
    requested_batch_size = max(1, int(batch_size))
    worker_count = max(1, int(parallel_workers))
    devices = list(cuda_devices or [None])
    worker_devices = [
        devices[slot_index % len(devices)]
        for slot_index in range(worker_count)
    ]
    workers_per_device = Counter(
        str(device if device is not None else 0)
        for device in worker_devices
    )
    overloaded_devices = {
        device: count
        for device, count in workers_per_device.items()
        if count > MAX_VALIDATED_WORKERS_PER_CUDA_DEVICE
    }
    if overloaded_devices:
        detail = ", ".join(
            f"{device}={count}"
            for device, count in sorted(overloaded_devices.items())
        )
        raise ValueError(
            "canonical M5 CuRobo concurrency exceeds the validated limit of "
            f"{MAX_VALIDATED_WORKERS_PER_CUDA_DEVICE} workers per CUDA device: "
            f"{detail}; higher concurrency causes planner-timeout false negatives"
        )
    metadata = {
        "schema_version": 1,
        "source_kind": CANONICAL_SOURCE_KIND,
        "source_probe_plan_path": _path_string(plan_path),
        "source_probe_plan_hash": plan.get("probe_plan_hash"),
        "source_validated_pose_set_hash": plan.get(
            "source_validated_pose_set_hash"
        ),
        "source_initial_state_selection_path": plan.get(
            "source_initial_state_selection_path"
        ),
        "initial_state_selection_version": plan.get(
            "initial_state_selection_version"
        ),
        "initial_state_selection_hash": plan.get("initial_state_selection_hash"),
        "config_hash": plan.get("config_hash"),
        "source_rubric_hash": plan.get("source_rubric_hash"),
        "skill_probe_policy_hash": plan.get("skill_probe_policy_hash"),
        "skill_relation_policy_hashes": plan.get(
            "skill_relation_policy_hashes"
        ),
        "selected_job_ids_hash": _hash([job["job_id"] for job in jobs]),
        "selected_job_count": len(jobs),
        "seeds": probe_seeds,
        "thresholds": probe_thresholds,
        "task_config": task_config,
        "torch_extensions_dir": _path_string(torch_extensions_dir) if torch_extensions_dir else None,
        "timeout_seconds": int(timeout_seconds),
        "warmup_timeout_seconds": int(warmup_timeout_seconds),
        "batch_size": requested_batch_size,
        "requested_batch_size": requested_batch_size,
        "effective_batch_size": requested_batch_size,
        "reuse_task_env": bool(reuse_task_env),
        "parallel_workers": worker_count,
        "cuda_devices": list(cuda_devices or []),
        "workers_per_cuda_device": dict(sorted(workers_per_device.items())),
    }
    if resume:
        prior = _load_json(metadata_path)
        mismatches = [
            key
            for key in RESUME_IDENTITY_FIELDS
            if prior.get(key) != metadata.get(key)
        ]
        if mismatches:
            raise ValueError(f"incompatible skill resume checkpoint: {', '.join(sorted(mismatches))}")
        started_at = prior.get("started_at")
        resume_count = int(prior.get("resume_count") or 0) + 1
    else:
        started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        resume_count = 0
        _write_jsonl(attempts_path, [])
        _write_jsonl(records_path, [])
    if clear_existing_reports:
        for path in reports_dir.glob("canonical_skill_pair_*.json"):
            path.unlink()

    specs = _attempt_specs(jobs, probe_seeds)
    if resume:
        _repair_jsonl_append_boundary(attempts_path)
    prior_attempts = _load_reusable_attempts(attempts_path, specs) if resume else {}
    # A retryable infrastructure row must not remain beside the replacement
    # row: canonical audit treats duplicate attempt ids as invalid evidence.
    # Compact once at resume, then append normally for the rest of the run.
    if resume:
        reusable_rows = [
            prior_attempts[str(spec["attempt_id"])]
            for spec in specs
            if str(spec["attempt_id"]) in prior_attempts
        ]
        _write_jsonl(attempts_path, reusable_rows)
    pending = [spec for spec in specs if spec["attempt_id"] not in prior_attempts]
    effective_batch_size = _effective_batch_size(
        pending_attempt_count=len(pending),
        worker_count=worker_count,
        requested_batch_size=requested_batch_size,
    )
    chunks = _arm_homogeneous_chunks(pending, effective_batch_size)
    _mark_chunk_warmups(chunks, reuse_task_env=reuse_task_env)
    metadata["batch_size"] = effective_batch_size
    metadata["effective_batch_size"] = effective_batch_size
    metadata["pending_attempt_count"] = len(pending)
    warp_cache_root = run_dir / "runtime_caches" / "warp"
    warp_cache_paths = [
        warp_cache_root / f"slot_{slot_index:03d}"
        for slot_index in range(worker_count)
    ]
    for warp_cache_path in warp_cache_paths:
        warp_cache_path.mkdir(parents=True, exist_ok=True)
    metadata["warp_cache_layout"] = "exclusive_logical_worker_slots_v1"
    metadata["warp_cache_root"] = _path_string(warp_cache_root)
    metadata["warp_cache_slot_count"] = worker_count
    metadata["started_at"] = started_at
    metadata["resume_count"] = resume_count
    metadata["resumed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) if resume else None
    _write_json_atomic(metadata_path, metadata)
    python = python_executable or sys.executable
    # Batch/log payloads are intentionally retained for failed diagnostics.
    # Always allocate a fresh index, even for a non-resume run in an existing
    # directory, so a new worker can never inherit an old sidecar filename.
    batch_index_offset = _next_batch_index(run_dir)
    completed: list[dict[str, Any]] = []
    progress_started_monotonic = time.monotonic()
    initial_reused_attempt_count = len(prior_attempts)
    completed_attempts_by_job = Counter(
        str(row.get("job_id")) for row in prior_attempts.values()
    )
    completed_job_count = sum(
        completed_attempts_by_job.get(str(job["job_id"]), 0) >= len(probe_seeds)
        for job in jobs
    )
    last_progress_emit = 0.0

    def emit_progress(*, state: str) -> None:
        nonlocal last_progress_emit
        finished_attempt_count = initial_reused_attempt_count + len(completed)
        snapshot = _progress_snapshot(
            state=state,
            total_attempt_count=len(specs),
            finished_attempt_count=finished_attempt_count,
            checkpointed_attempt_count=finished_attempt_count,
            total_job_count=len(jobs),
            completed_job_count=completed_job_count,
            started_monotonic=progress_started_monotonic,
            now_monotonic=time.monotonic(),
            now_wall_time=time.time(),
            parallel_workers=worker_count,
            batch_size=effective_batch_size,
            baseline_finished_attempt_count=initial_reused_attempt_count,
        )
        _write_json_atomic(progress_path, snapshot)
        last_progress_emit = time.monotonic()
        print(_format_progress(snapshot), flush=True)

    def checkpoint_batch(
        batch_index: int,
        batch_results: list[dict[str, Any]],
        attempts_output: Any,
    ) -> None:
        nonlocal completed_job_count
        completed.extend(batch_results)
        _append_jsonl_rows(attempts_output, batch_results)
        _durable_flush(attempts_output)
        for row in batch_results:
            job_id = str(row.get("job_id"))
            previous_count = completed_attempts_by_job.get(job_id, 0)
            completed_attempts_by_job[job_id] = previous_count + 1
            if previous_count < len(probe_seeds) <= previous_count + 1:
                completed_job_count += 1
        _discard_checkpointed_batch_payloads(
            batch_dir=batch_dir,
            batch_index=batch_index,
            attempts=batch_results,
        )
        if time.monotonic() - last_progress_emit >= PROGRESS_EMIT_INTERVAL_SECONDS:
            emit_progress(state="running")

    def launch(
        index: int,
        chunk: list[dict[str, Any]],
        slot_index: int,
    ) -> list[dict[str, Any]]:
        return _run_batch_safe(
            index=index,
            specs=chunk,
            python_executable=python,
            task_config=task_config,
            run_dir=run_dir,
            batch_dir=batch_dir,
            log_dir=log_dir,
            timeout_seconds=int(timeout_seconds),
            warmup_timeout_seconds=int(warmup_timeout_seconds),
            torch_extensions_dir=torch_extensions_dir,
            reuse_task_env=reuse_task_env,
            cuda_device=worker_devices[slot_index],
            warp_cache_path=warp_cache_paths[slot_index],
        )

    emit_progress(state="running")
    with attempts_path.open("a", encoding="utf-8") as attempts_output:
        if worker_count == 1:
            for index, chunk in enumerate(chunks):
                absolute_index = batch_index_offset + index
                batch_results = launch(absolute_index, chunk, 0)
                checkpoint_batch(absolute_index, batch_results, attempts_output)
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                pending_chunks = iter(enumerate(chunks))
                futures: dict[Any, tuple[int, int]] = {}

                def submit_next(slot_index: int) -> None:
                    try:
                        chunk_index, chunk = next(pending_chunks)
                    except StopIteration:
                        return
                    absolute_index = batch_index_offset + chunk_index
                    future = executor.submit(
                        launch,
                        absolute_index,
                        chunk,
                        slot_index,
                    )
                    futures[future] = (absolute_index, slot_index)

                # Keep exactly one subprocess in flight per logical slot.  A
                # completed slot immediately claims the next chunk, preserving
                # dynamic load balancing without ever sharing its Warp cache
                # with another live worker.
                for slot_index in range(min(worker_count, len(chunks))):
                    submit_next(slot_index)
                while futures:
                    done, _ = wait(
                        tuple(futures),
                        return_when=FIRST_COMPLETED,
                    )
                    for future in done:
                        absolute_index, slot_index = futures.pop(future)
                        batch_results = future.result()
                        checkpoint_batch(
                            absolute_index,
                            batch_results,
                            attempts_output,
                        )
                        submit_next(slot_index)
    all_attempts = list(prior_attempts.values()) + completed
    all_attempts.sort(key=lambda row: (int(row.get("job_index") or 0), int(row.get("seed") or 0)))
    by_job: dict[str, list[dict[str, Any]]] = {}
    for attempt in all_attempts:
        by_job.setdefault(str(attempt["job_id"]), []).append(attempt)
    records = []
    for job in jobs:
        row = aggregate_skill_attempts(
            job,
            by_job.get(str(job["job_id"]), []),
            seeds=probe_seeds,
            threshold=float(probe_thresholds[str(job["relation"])]),
        )
        row["config_hash"] = _row_config_hash(row)
        records.append(row)
    _write_jsonl(records_path, records)
    timing_summary_path = run_dir / "timing_summary.yml"
    timing_summary = _write_timing_summary(
        timing_summary_path,
        attempts=all_attempts,
        records=records,
        batch_size=effective_batch_size,
        requested_batch_size=requested_batch_size,
        reuse_task_env=reuse_task_env,
        parallel_workers=worker_count,
        cuda_devices=cuda_devices,
    )
    finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "kind": "canonical_skill_pair_report",
        "scope": "pair",
        "stage": STAGE_CANONICAL,
        "source_kind": CANONICAL_SOURCE_KIND,
        "source_probe_plan_path": _path_string(plan_path),
        "source_probe_plan_hash": plan.get("probe_plan_hash"),
        "source_validated_pose_set_hash": plan.get(
            "source_validated_pose_set_hash"
        ),
        "source_initial_state_selection_path": plan.get(
            "source_initial_state_selection_path"
        ),
        "initial_state_selection_version": plan.get(
            "initial_state_selection_version"
        ),
        "initial_state_selection_hash": plan.get("initial_state_selection_hash"),
        "config_hash": plan.get("config_hash"),
        "source_rubric_hash": plan.get("source_rubric_hash"),
        "skill_probe_policy_hash": plan.get("skill_probe_policy_hash"),
        "skill_relation_policy_hashes": plan.get(
            "skill_relation_policy_hashes"
        ),
        "runner": {
            **metadata,
            "python_executable": python,
            "finished_at": finished_at,
            "resume": bool(resume),
        },
        "summary": {
            "record_count": len(records),
            "attempt_count": len(all_attempts),
            "outcomes": dict(sorted(Counter(str(row["outcome"]) for row in records).items())),
            "relations": dict(sorted(Counter(str(row["relation"]) for row in records).items())),
        },
        "timing_summary": timing_summary,
        "checkpoints": {
            "attempts": _path_string(attempts_path),
            "records": _path_string(records_path),
            "progress": _path_string(progress_path),
            "timing_summary": _path_string(timing_summary_path),
        },
        "records": records,
    }
    report["evidence_id"] = _evidence_id(report)
    report["report_hash"] = _artifact_hash(report, "report_hash")
    report_path = reports_dir / f"{report['evidence_id']}.json"
    _write_json_atomic(report_path, report)
    emit_progress(state="completed")
    return {
        "evidence_id": report["evidence_id"],
        "evidence_path": _path_string(report_path),
        "record_count": len(records),
        "attempt_count": len(all_attempts),
        "summary": report["summary"],
        "timing_summary_path": _path_string(timing_summary_path),
    }


def audit_skill_probe_run(
    *,
    plan_path: Path,
    checkpoint_dir: Path,
    output_path: Path,
    receiver_profile_path: Path | None = None,
    relations: set[str] | None = None,
    pair_keys: set[str] | None = None,
    profile_ids: set[str] | None = None,
) -> dict[str, Any]:
    require_active_relation_selection(relations=relations, pair_keys=pair_keys)
    plan = _load_yaml(plan_path)
    _validate_plan(plan)
    plan_jobs = _select_jobs(
        plan.get("jobs") or [],
        relations=relations,
        pair_keys=pair_keys,
        profile_ids=profile_ids,
        start_index=0,
        max_jobs=None,
    )
    jobs = {str(job["job_id"]): job for job in plan_jobs}

    metadata_path = checkpoint_dir / "run_metadata.json"
    metadata = _load_json(metadata_path) if metadata_path.is_file() else {}
    metadata_mismatches: list[dict[str, Any]] = []
    expected_metadata = {
        "source_probe_plan_hash": plan.get("probe_plan_hash"),
        "skill_probe_policy_hash": plan.get("skill_probe_policy_hash"),
        "selected_job_count": len(plan_jobs),
        "selected_job_ids_hash": _hash([job["job_id"] for job in plan_jobs]),
    }
    if plan.get("initial_state_selection_hash"):
        expected_metadata["source_validated_pose_set_hash"] = plan.get(
            "source_validated_pose_set_hash"
        )
        expected_metadata["initial_state_selection_hash"] = plan[
            "initial_state_selection_hash"
        ]
    if not metadata:
        metadata_mismatches.append(
            {"field": "run_metadata", "expected": "present", "actual": "missing"}
        )
    else:
        for field, expected in expected_metadata.items():
            actual = metadata.get(field)
            if actual != expected:
                metadata_mismatches.append(
                    {"field": field, "expected": expected, "actual": actual}
                )
    raw_seeds = metadata.get("seeds")
    seeds = (
        [int(seed) for seed in raw_seeds]
        if isinstance(raw_seeds, list) and raw_seeds
        else list(DEFAULT_SEEDS)
    )
    if not isinstance(raw_seeds, list) or not raw_seeds:
        metadata_mismatches.append(
            {"field": "seeds", "expected": list(DEFAULT_SEEDS), "actual": raw_seeds}
        )
    thresholds = {**DEFAULT_THRESHOLDS, **(metadata.get("thresholds") or {})}

    progress_path = checkpoint_dir.parent / "progress.json"
    progress = _load_json(progress_path) if progress_path.is_file() else {}
    progress_mismatches: list[dict[str, Any]] = []
    expected_progress = {
        "state": "completed",
        "total_attempt_count": len(plan_jobs) * len(seeds),
        "finished_attempt_count": len(plan_jobs) * len(seeds),
        "checkpointed_attempt_count": len(plan_jobs) * len(seeds),
        "total_job_count": len(plan_jobs),
        "completed_job_count": len(plan_jobs),
    }
    if not progress:
        progress_mismatches.append(
            {"field": "progress", "expected": "present", "actual": "missing"}
        )
    else:
        for field, expected in expected_progress.items():
            actual = progress.get(field)
            if actual != expected:
                progress_mismatches.append(
                    {"field": field, "expected": expected, "actual": actual}
                )

    attempts = _read_jsonl(checkpoint_dir / "attempts.jsonl")
    expected_attempt_specs = _attempt_specs(plan_jobs, seeds)
    expected_attempts = {
        str(spec["attempt_id"]): spec for spec in expected_attempt_specs
    }
    expected_attempt_ids_by_job: dict[str, set[str]] = {}
    for spec in expected_attempt_specs:
        expected_attempt_ids_by_job.setdefault(str(spec["job_id"]), set()).add(
            str(spec["attempt_id"])
        )
    attempt_ids = [str(row.get("attempt_id")) for row in attempts]
    attempt_counts = Counter(attempt_ids)
    attempt_id_set = set(attempt_ids)
    missing_attempt_ids = sorted(set(expected_attempts) - attempt_id_set)
    extra_attempt_ids = sorted(attempt_id_set - set(expected_attempts))
    duplicate_attempt_ids = sorted(
        attempt_id for attempt_id, count in attempt_counts.items() if count > 1
    )
    duplicate_attempt_count = sum(
        count - 1 for count in attempt_counts.values() if count > 1
    )
    attempt_identity_mismatches: list[dict[str, Any]] = []
    infrastructure_attempts: list[dict[str, Any]] = []
    invalid_attempt_ids: set[str] = set(extra_attempt_ids) | set(duplicate_attempt_ids)
    attempts_by_job: dict[str, list[dict[str, Any]]] = {}
    for attempt_index, row in enumerate(attempts):
        attempt_id = str(row.get("attempt_id"))
        job_id = str(row.get("job_id"))
        attempts_by_job.setdefault(job_id, []).append(row)
        expected = expected_attempts.get(attempt_id)
        mismatches: dict[str, dict[str, Any]] = {}
        if expected is not None:
            for field in ("job_id", "job_index", "pair_key", "relation", "seed"):
                actual = row.get(field)
                expected_value = expected.get(field)
                if actual != expected_value:
                    mismatches[field] = {
                        "expected": expected_value,
                        "actual": actual,
                    }
            job = jobs.get(str(expected["job_id"])) or {}
            expected_profile = job.get("receiver_state_profile_id") or None
            actual_profile = row.get("receiver_state_profile_id") or None
            if actual_profile != expected_profile:
                mismatches["receiver_state_profile_id"] = {
                    "expected": expected_profile,
                    "actual": actual_profile,
                }
        if mismatches:
            invalid_attempt_ids.add(attempt_id)
            attempt_identity_mismatches.append(
                {
                    "attempt_index": attempt_index,
                    "attempt_id": attempt_id,
                    "fields": mismatches,
                }
            )
        failure_reason = str(row.get("failure_reason") or "")
        if failure_reason in INFRASTRUCTURE_FAILURES or row.get("outcome") == "blocked":
            infrastructure_attempts.append(
                {
                    "attempt_index": attempt_index,
                    "attempt_id": attempt_id,
                    "job_id": job_id,
                    "failure_reason": failure_reason or "blocked",
                }
            )

    records = _read_jsonl(checkpoint_dir / "records.jsonl")
    record_job_ids = [str(row.get("job_id")) for row in records]
    record_counts = Counter(record_job_ids)
    record_job_id_set = set(record_job_ids)
    missing_job_ids = sorted(set(jobs) - record_job_id_set)
    extra_record_job_ids = sorted(record_job_id_set - set(jobs))
    duplicate_record_job_ids = sorted(
        job_id for job_id, count in record_counts.items() if count > 1
    )
    duplicate_record_count = sum(
        count - 1 for count in record_counts.values() if count > 1
    )
    invalid_outcome_records: list[dict[str, Any]] = []
    identity_mismatches: list[dict[str, Any]] = []
    base_record_contract_valid: list[bool] = []
    unique_records_by_job: dict[str, dict[str, Any]] = {}
    for record_index, row in enumerate(records):
        job_id = str(row.get("job_id"))
        outcome = str(row.get("outcome") or "")
        if outcome not in {"pass", "fail", "blocked"}:
            invalid_outcome_records.append(
                {
                    "record_index": record_index,
                    "job_id": job_id,
                    "outcome": row.get("outcome"),
                }
            )
        mismatches: dict[str, dict[str, Any]] = {}
        job = jobs.get(job_id)
        if job is not None:
            for field in ("pair_key", "relation", "receiver_state_profile_id"):
                expected = job.get(field)
                actual = row.get(field)
                if field == "receiver_state_profile_id":
                    expected = expected or None
                    actual = actual or None
                if actual != expected:
                    mismatches[field] = {
                        "expected": expected,
                        "actual": actual,
                    }
        if mismatches:
            identity_mismatches.append(
                {
                    "record_index": record_index,
                    "job_id": job_id,
                    "fields": mismatches,
                }
            )
        contract_valid = (
            job is not None
            and record_counts[job_id] == 1
            and outcome in {"pass", "fail", "blocked"}
            and not mismatches
        )
        base_record_contract_valid.append(contract_valid)
        if contract_valid:
            unique_records_by_job[job_id] = row

    aggregate_mismatches: list[dict[str, Any]] = []
    aggregate_mismatch_job_ids: set[str] = set()
    for job_id, job in jobs.items():
        record = unique_records_by_job.get(job_id)
        expected_ids = expected_attempt_ids_by_job.get(job_id, set())
        if (
            record is None
            or not expected_ids
            or any(attempt_id in invalid_attempt_ids for attempt_id in expected_ids)
            or not expected_ids.issubset(attempt_id_set)
            or any(attempt_counts[attempt_id] != 1 for attempt_id in expected_ids)
        ):
            continue
        expected_record = aggregate_skill_attempts(
            job,
            attempts_by_job.get(job_id, []),
            seeds=seeds,
            threshold=float(thresholds[str(job["relation"])]),
        )
        fields: dict[str, dict[str, Any]] = {}
        for field in (
            "outcome",
            "pass",
            "verdict",
            "attempts",
            "valid_attempts",
            "plan_success",
            "success",
            "min_rate",
            "fail_reasons",
            "validated_skill_recipe_id",
            "validated_skill_band_ids",
            "receiver_state_profile_id",
            "receiver_state_profile_hash",
        ):
            expected_value = expected_record.get(field)
            actual = record.get(field)
            if actual != expected_value:
                fields[field] = {"expected": expected_value, "actual": actual}
        if fields:
            aggregate_mismatch_job_ids.add(job_id)
            aggregate_mismatches.append({"job_id": job_id, "fields": fields})

    record_contract_valid = [
        contract_valid
        and str(row.get("job_id")) not in aggregate_mismatch_job_ids
        for row, contract_valid in zip(records, base_record_contract_valid)
    ]
    outcome_counts = Counter(str(row.get("outcome") or "missing") for row in records)
    passed_by_relation = Counter(
        str(row.get("relation"))
        for row, contract_valid in zip(records, record_contract_valid)
        if contract_valid and row.get("outcome") == "pass"
    )
    required_relations = (
        sorted(relations)
        if relations is not None
        else list(ACTIVE_RELATIONS)
    )
    missing_reference_relations = [
        relation
        for relation in required_relations
        if passed_by_relation.get(relation, 0) == 0
    ]
    passed_by_profile = Counter(
        str(row.get("receiver_state_profile_id"))
        for row, contract_valid in zip(records, record_contract_valid)
        if contract_valid
        and row.get("outcome") == "pass"
        and row.get("receiver_state_profile_id")
    )
    reviewed_profile_ids: list[str] = []
    if receiver_profile_path is not None:
        profile_catalog = _load_yaml(receiver_profile_path)
        reviewed_profile_ids = sorted(
            str(profile_id)
            for profile_id, profile in (profile_catalog.get("profiles") or {}).items()
            if isinstance(profile, dict) and profile.get("review_status") == "reviewed"
            and profile.get("acceptance_required") is not False
            and (
                profile_ids is None
                or str(profile_id) in profile_ids
            )
        )
    missing_reviewed_profile_positives = [
        profile_id
        for profile_id in reviewed_profile_ids
        if passed_by_profile.get(profile_id, 0) == 0
    ]
    blocked_record_count = int(outcome_counts.get("blocked", 0))
    precondition_blocked_jobs = [
        row
        for row in (plan.get("blocked_jobs") or [])
        if (
            (relations is None or str(row.get("relation")) in relations)
            and (
                pair_keys is None
                or str(row.get("pair_key")) in pair_keys
            )
            and (
                profile_ids is None
                or str(row.get("receiver_state_profile_id")) in profile_ids
            )
        )
    ]
    acceptance_ready = bool(
        not metadata_mismatches
        and not progress_mismatches
        and not missing_attempt_ids
        and not extra_attempt_ids
        and duplicate_attempt_count == 0
        and not attempt_identity_mismatches
        and not infrastructure_attempts
        and not missing_job_ids
        and not extra_record_job_ids
        and duplicate_record_count == 0
        and not invalid_outcome_records
        and not identity_mismatches
        and not aggregate_mismatches
        and blocked_record_count == 0
        and not missing_reference_relations
        and not missing_reviewed_profile_positives
    )
    failures = []
    for row in records:
        if row.get("outcome") == "pass":
            continue
        job = jobs.get(str(row.get("job_id"))) or {}
        failures.append(
            {
                "pair_key": row.get("pair_key"),
                "relation": row.get("relation"),
                "outcome": row.get("outcome"),
                "verdict": row.get("verdict"),
                "fail_reasons": row.get("fail_reasons") or {},
                "selected_arm": job.get("selected_arm"),
                "yaw_witness": job.get("yaw_witness") or {},
                "skill_preconditions": job.get("skill_preconditions") or {},
                "receiver_state_profile_id": job.get("receiver_state_profile_id"),
            }
        )
    audit = {
        "schema_version": 1,
        "kind": "canonical_skill_failure_audit",
        "source_probe_plan_hash": plan.get("probe_plan_hash"),
        "summary": {
            "acceptance_ready": acceptance_ready,
            "plan_job_count": len(jobs),
            "expected_attempt_count": len(expected_attempt_specs),
            "attempt_count": len(attempts),
            "missing_attempt_count": len(missing_attempt_ids),
            "extra_attempt_count": len(extra_attempt_ids),
            "duplicate_attempt_count": duplicate_attempt_count,
            "attempt_identity_mismatch_count": len(attempt_identity_mismatches),
            "infrastructure_attempt_count": len(infrastructure_attempts),
            "metadata_mismatch_count": len(metadata_mismatches),
            "progress_mismatch_count": len(progress_mismatches),
            "record_count": len(records),
            "missing_record_count": len(missing_job_ids),
            "extra_record_count": len(extra_record_job_ids),
            "duplicate_record_count": duplicate_record_count,
            "invalid_outcome_count": len(invalid_outcome_records),
            "identity_mismatch_count": len(identity_mismatches),
            "aggregate_mismatch_count": len(aggregate_mismatches),
            "blocked_record_count": blocked_record_count,
            "precondition_blocked_job_count": len(precondition_blocked_jobs),
            "failure_count": len(failures),
            "outcomes": dict(sorted(outcome_counts.items())),
            "passed_by_relation": {
                relation: int(passed_by_relation.get(relation, 0))
                for relation in RELATIONS
            },
            "missing_reference_relations": missing_reference_relations,
            "reviewed_profile_count": len(reviewed_profile_ids),
            "passed_by_reviewed_profile": {
                profile_id: int(passed_by_profile.get(profile_id, 0))
                for profile_id in reviewed_profile_ids
            },
            "missing_reviewed_profile_positives": missing_reviewed_profile_positives,
        },
        "coverage_gaps": {
            "metadata_mismatches": metadata_mismatches,
            "progress_mismatches": progress_mismatches,
            "missing_attempt_ids": missing_attempt_ids,
            "extra_attempt_ids": extra_attempt_ids,
            "duplicate_attempt_ids": duplicate_attempt_ids,
            "attempt_identity_mismatches": attempt_identity_mismatches,
            "infrastructure_attempts": infrastructure_attempts,
            "missing_job_ids": missing_job_ids,
            "extra_record_job_ids": extra_record_job_ids,
            "duplicate_record_job_ids": duplicate_record_job_ids,
            "invalid_outcome_records": invalid_outcome_records,
            "identity_mismatches": identity_mismatches,
            "aggregate_mismatches": aggregate_mismatches,
            "precondition_blocked_jobs": precondition_blocked_jobs,
        },
        "failures": failures,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(audit, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return audit


def render_skill_probe_job(
    *,
    plan_path: Path,
    pair_key: str,
    seed: int,
    output_dir: Path,
    task_config: str = "demo_clean",
    profile_id: str | None = None,
) -> dict[str, Any]:
    """Run one rollout with headless offscreen capture enabled for audit."""

    from .asset_probe_worker import _build_runtime_args
    from .pair_skill_worker import run_attempt

    require_active_relation_selection(pair_keys={pair_key})
    plan = _load_yaml(plan_path)
    _validate_plan(plan)
    matches = [
        job
        for job in plan.get("jobs") or []
        if str(job.get("pair_key")) == pair_key
        and (profile_id is None or str(job.get("receiver_state_profile_id")) == profile_id)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one skill job for {pair_key}; found {len(matches)}")
    selected_job = matches[0]
    selected_profile_id = selected_job.get("receiver_state_profile_id")
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_args = _build_runtime_args(task_config, str(output_dir))
    runtime_args.update(
        {
            "task_name": "if_pair_skill_probe",
            "save_video": False,
            "eval_video_log": False,
            "render_freq": 0,
            "save_path": str(output_dir / "sim_data"),
            "skill_render_dir": str(output_dir / "frames"),
        }
    )
    result = run_attempt(
        selected_job,
        seed=int(seed),
        task_config=task_config,
        run_dir=output_dir,
        runtime_args_template=runtime_args,
    )
    frames = sorted(
        [
            str(path)
            for suffix in ("*.png", "*.jpg", "*.mp4")
            for path in output_dir.rglob(suffix)
        ]
    )
    manifest = {
        "schema_version": 1,
        "kind": "canonical_skill_render",
        "pair_key": pair_key,
        "seed": int(seed),
        "receiver_state_profile_id": selected_profile_id,
        "frames": frames,
        "result": result,
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    return {"manifest_path": str(manifest_path), **manifest}


def _select_jobs(
    jobs: list[dict[str, Any]],
    *,
    relations: set[str] | None,
    pair_keys: set[str] | None,
    profile_ids: set[str] | None,
    start_index: int,
    max_jobs: int | None,
) -> list[dict[str, Any]]:
    selected = [
        job
        for job in jobs
        if (relations is None or job.get("relation") in relations)
        and (pair_keys is None or str(job.get("pair_key")) in pair_keys)
        and (profile_ids is None or str(job.get("receiver_state_profile_id")) in profile_ids)
    ]
    selected = selected[max(0, int(start_index)) :]
    return selected if max_jobs is None else selected[: max(0, int(max_jobs))]


def _attempt_specs(jobs: list[dict[str, Any]], seeds: list[int]) -> list[dict[str, Any]]:
    return [
        {
            "attempt_id": _hash({"job_id": job["job_id"], "seed": int(seed)})[:24],
            "job_id": job["job_id"],
            "job_index": job_index,
            "pair_key": job["pair_key"],
            "relation": job["relation"],
            "seed": int(seed),
            "probe_job": job,
            "warmup": False,
        }
        for job_index, job in enumerate(jobs)
        for seed in seeds
    ]


def _load_reusable_attempts(path: Path, specs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    expected = {str(spec["attempt_id"]): spec for spec in specs}
    rows = {}
    for row in _read_jsonl(path):
        attempt_id = str(row.get("attempt_id"))
        if attempt_id not in expected:
            continue
        if str(row.get("failure_reason") or "") in INFRASTRUCTURE_FAILURES:
            continue
        spec = expected[attempt_id]
        # ``job_index`` is the ordinal within the selected run, not physical
        # evidence identity.  It legitimately changes when a relation is
        # disabled and an otherwise identical checkpoint is resumed through a
        # narrower selection.  Preserve the attempt while rebinding that one
        # scheduler field; reject every substantive identity mismatch.
        if any(
            field in spec and row.get(field) != spec.get(field)
            for field in ("job_id", "pair_key", "relation", "seed")
        ):
            continue
        normalized = dict(row)
        if "job_index" in spec:
            normalized["job_index"] = spec["job_index"]
        rows[attempt_id] = normalized
    return rows


def _run_batch(
    *,
    index: int,
    specs: list[dict[str, Any]],
    python_executable: str,
    task_config: str,
    run_dir: Path,
    batch_dir: Path,
    log_dir: Path,
    timeout_seconds: int,
    torch_extensions_dir: Path | None,
    reuse_task_env: bool,
    cuda_device: str | None,
    warmup_timeout_seconds: int = DEFAULT_WARMUP_TIMEOUT_SECONDS,
    warp_cache_path: Path | None = None,
) -> list[dict[str, Any]]:
    batch_path = batch_dir / f"skill_batch_{index:06d}.json"
    output_path = batch_dir / f"skill_batch_{index:06d}.results.jsonl"
    stdout_path = log_dir / f"skill_batch_{index:06d}.stdout.log"
    stderr_path = log_dir / f"skill_batch_{index:06d}.stderr.log"
    _write_json(batch_path, {"attempts": specs})
    # Fail closed at the worker boundary.  A diagnostic sidecar from a prior
    # invocation must never be interpreted as output from this subprocess.
    try:
        output_path.unlink()
    except FileNotFoundError:
        pass
    command = [
        python_executable,
        "-m",
        "if_benchmark.pair_skill_worker",
        "--batch-file",
        str(batch_path),
        "--batch-output",
        str(output_path),
        "--task-config",
        task_config,
        "--run-dir",
        str(run_dir),
    ]
    if reuse_task_env:
        command.append("--reuse-task-env")
    env = os.environ.copy()
    env.setdefault("PYTHONWARNINGS", "ignore::UserWarning")
    # Each GPU worker owns its own physics/planning loop.  Prevent NumPy/BLAS
    # from multiplying the worker count again on the CPU side.
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    matplotlib_cache = run_dir / "runtime_caches" / "matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    env.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    env["CUDA_VISIBLE_DEVICES"] = str(cuda_device if cuda_device is not None else 0)
    env.setdefault(
        "TORCH_EXTENSIONS_DIR",
        str(torch_extensions_dir or (run_dir / "torch_extensions")),
    )
    if warp_cache_path is not None:
        _apply_warp_cache_path(env, warp_cache_path)
    timed_out = False
    returncode: int | None = None
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            command,
            cwd=Path.cwd(),
            env=env,
            stdout=stdout,
            stderr=stderr,
            text=True,
            start_new_session=True,
        )
        returncode, timed_out = _wait_for_batch_progress(
            process,
            output_path=output_path,
            specs=specs,
            timeout_seconds=max(1, int(timeout_seconds)),
            warmup_timeout_seconds=max(1, int(warmup_timeout_seconds)),
        )
    worker_rows = _read_worker_results(output_path)
    worker_row_counts = Counter(str(row.get("attempt_id")) for row in worker_rows)
    specs_by_id = {str(spec["attempt_id"]): spec for spec in specs}
    parsed = {
        attempt_id: row
        for row in worker_rows
        if (attempt_id := str(row.get("attempt_id"))) in specs_by_id
        and worker_row_counts[attempt_id] == 1
        and _worker_result_matches_spec(row, specs_by_id[attempt_id])
    }
    results = []
    for spec in specs:
        attempt_id = str(spec["attempt_id"])
        row = parsed.get(attempt_id)
        if row is None:
            reason = "worker_timeout" if timed_out else ("worker_no_json_result" if returncode == 0 else "worker_process_failed")
            row = {
                **{key: spec.get(key) for key in ("attempt_id", "job_id", "job_index", "pair_key", "relation", "seed")},
                "plan_success": False,
                "released": False,
                "settled": False,
                "checker_success": False,
                "success": False,
                "failure_reason": reason,
                "warmup": bool(spec.get("warmup", False)),
            }
        row["worker_returncode"] = returncode
        row["cuda_device"] = cuda_device
        row["stdout_log"] = _path_string(stdout_path)
        row["stderr_log"] = _path_string(stderr_path)
        results.append(row)
    return results


def _apply_warp_cache_path(
    env: dict[str, str],
    warp_cache_path: Path,
) -> None:
    """Give one serial logical worker an isolated, reusable Warp cache."""

    warp_cache_path.mkdir(parents=True, exist_ok=True)
    # Override an inherited shared cache.  ``setdefault`` would retain the
    # exact XDG/Warp path that caused concurrent PCH creation/deletion races.
    env["WARP_CACHE_PATH"] = str(warp_cache_path)


def _run_batch_safe(**kwargs: Any) -> list[dict[str, Any]]:
    specs = list(kwargs.get("specs") or [])
    try:
        return _run_batch(**kwargs)
    except (OSError, subprocess.SubprocessError) as exc:
        # Expected worker-transport failures are retryable infrastructure
        # evidence.  Programming/aggregation errors must still escape and
        # make the command fail non-zero instead of masquerading as a run.
        cuda_device = kwargs.get("cuda_device")
        rows = []
        for spec in specs:
            row = _missing_attempt(spec, reason="worker_process_failed")
            row.update(
                {
                    "exception_type": type(exc).__name__,
                    "exception": str(exc),
                    "traceback_tail": traceback.format_exc()[-4000:],
                    "cuda_device": cuda_device,
                }
            )
            rows.append(row)
        return rows


def _missing_attempt(spec: dict[str, Any], *, reason: str) -> dict[str, Any]:
    return {
        **{
            key: spec.get(key)
            for key in (
                "attempt_id",
                "job_id",
                "job_index",
                "pair_key",
                "relation",
                "seed",
            )
        },
        "plan_success": False,
        "released": False,
        "settled": False,
        "checker_success": False,
        "success": False,
        "failure_reason": reason,
        "warmup": bool(spec.get("warmup", False)),
    }


def _wait_for_batch_progress(
    process: subprocess.Popen[Any],
    *,
    output_path: Path,
    specs: list[dict[str, Any]],
    timeout_seconds: int,
    warmup_timeout_seconds: int,
    poll_interval_seconds: float = 0.25,
) -> tuple[int | None, bool]:
    """Watch each emitted attempt instead of assigning one huge batch timeout."""

    completed_lines = 0
    observed_size = 0
    partial = b""

    def attempt_timeout(index: int) -> int:
        if index < len(specs) and specs[index].get("warmup"):
            return max(1, int(warmup_timeout_seconds))
        return max(1, int(timeout_seconds))

    deadline = time.monotonic() + attempt_timeout(0)
    while True:
        if output_path.is_file():
            size = output_path.stat().st_size
            if size < observed_size:
                observed_size = 0
                partial = b""
                completed_lines = 0
            if size > observed_size:
                with output_path.open("rb") as stream:
                    stream.seek(observed_size)
                    partial += stream.read(size - observed_size)
                observed_size = size
                complete, separator, partial = partial.rpartition(b"\n")
                if separator:
                    newly_completed = complete.count(b"\n") + 1
                    completed_lines += newly_completed
                    deadline = time.monotonic() + attempt_timeout(completed_lines)

        returncode = process.poll()
        if returncode is not None:
            return returncode, False

        now = time.monotonic()
        if now >= deadline:
            return _terminate_process_group(process), True
        time.sleep(
            min(
                max(0.01, float(poll_interval_seconds)),
                max(0.01, deadline - now),
            )
        )


def _terminate_process_group(process: subprocess.Popen[Any]) -> int | None:
    """Stop a timed-out worker and any compiler/planner descendants."""

    pid = getattr(process, "pid", None)
    group_signal_sent = False
    if isinstance(pid, int) and pid > 0:
        try:
            os.killpg(pid, signal.SIGTERM)
            group_signal_sent = True
        except ProcessLookupError:
            pass
        except OSError:
            group_signal_sent = False
    if not group_signal_sent:
        try:
            process.kill()
        except ProcessLookupError:
            pass
    try:
        return process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if isinstance(pid, int) and pid > 0:
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError:
                process.kill()
        else:
            process.kill()
        return process.wait()


def _worker_result_matches_spec(
    row: dict[str, Any],
    spec: dict[str, Any],
) -> bool:
    """Require a worker row to belong exactly to the launched attempt."""

    return all(
        row.get(field) == spec.get(field)
        for field in (
            "attempt_id",
            "job_id",
            "job_index",
            "pair_key",
            "relation",
            "seed",
        )
    )


def _next_batch_index(run_dir: Path) -> int:
    indices: list[int] = []
    for directory, patterns in (
        (run_dir, ("skill_batch_*.json", "skill_batch_*.results.jsonl")),
        (run_dir / "batches", ("skill_batch_*.json", "skill_batch_*.results.jsonl")),
        (run_dir / "logs", ("skill_batch_*.stdout.log", "skill_batch_*.stderr.log")),
    ):
        for pattern in patterns:
            for path in directory.glob(pattern):
                index_text = path.name.removeprefix("skill_batch_").split(".", 1)[0]
                try:
                    indices.append(int(index_text))
                except ValueError:
                    continue
    return max(indices, default=-1) + 1


def _validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("stage") != STAGE_CANONICAL:
        raise ValueError("expected canonical-skill probe plan")
    if plan.get("config_hash") != config_hash():
        raise ValueError("canonical-skill plan config is stale; regenerate plan-skill-probes")
    if plan.get("skill_probe_policy_hash") != skill_probe_policy_hash():
        raise ValueError("canonical-skill plan policy is stale; regenerate plan-skill-probes")
    plan_relation_hashes = plan.get("skill_relation_policy_hashes") or {}
    for job in plan.get("jobs") or []:
        relation = str(job.get("relation") or "")
        expected_relation_hash = skill_relation_policy_hash(relation)
        if (
            job.get("skill_relation_policy_hash")
            != expected_relation_hash
        ):
            raise ValueError(
                "canonical-skill job relation policy is stale; "
                "regenerate plan-skill-probes"
            )
        if plan_relation_hashes.get(relation) != expected_relation_hash:
            raise ValueError(
                "canonical-skill plan relation policy is stale; "
                "regenerate plan-skill-probes"
            )
    if plan.get("probe_plan_hash") != _artifact_hash(plan, "probe_plan_hash"):
        raise ValueError("canonical-skill probe plan hash mismatch")
    if not plan.get("source_validated_pose_set_hash"):
        raise ValueError("canonical-skill plan has no validated Initial Pose Set")
    for job in plan.get("jobs") or []:
        if set((job.get("initial_pose_bindings") or {})) != {"subject", "receiver"}:
            raise ValueError(
                f"canonical-skill job lacks Initial Pose bindings: {job.get('job_id')}"
            )
    job_ids = [str(job.get("job_id")) for job in plan.get("jobs") or []]
    duplicate_job_ids = sorted(
        job_id for job_id, count in Counter(job_ids).items() if count > 1
    )
    if duplicate_job_ids:
        raise ValueError(
            "canonical-skill probe plan contains duplicate job ids: "
            + ", ".join(duplicate_job_ids)
        )


def _progress_snapshot(
    *,
    state: str,
    total_attempt_count: int,
    finished_attempt_count: int,
    checkpointed_attempt_count: int,
    total_job_count: int,
    completed_job_count: int,
    started_monotonic: float,
    now_monotonic: float,
    now_wall_time: float,
    parallel_workers: int,
    batch_size: int,
    baseline_finished_attempt_count: int = 0,
) -> dict[str, Any]:
    elapsed_seconds = max(0.0, float(now_monotonic) - float(started_monotonic))
    total_attempts = max(0, int(total_attempt_count))
    finished_attempts = min(total_attempts, max(0, int(finished_attempt_count)))
    remaining_attempts = max(0, total_attempts - finished_attempts)
    baseline_finished = min(finished_attempts, max(0, int(baseline_finished_attempt_count)))
    newly_finished = max(0, finished_attempts - baseline_finished)
    attempts_per_second = newly_finished / elapsed_seconds if elapsed_seconds > 0.0 else 0.0
    if remaining_attempts == 0:
        eta_seconds: float | None = 0.0
    elif attempts_per_second > 0.0:
        eta_seconds = remaining_attempts / attempts_per_second
    else:
        eta_seconds = None
    estimated_finish_at = (
        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now_wall_time + eta_seconds))
        if eta_seconds is not None
        else None
    )
    progress_percent = 100.0 if total_attempts == 0 else 100.0 * finished_attempts / total_attempts
    return {
        "schema_version": 1,
        "kind": "canonical_skill_run_progress",
        "state": str(state),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now_wall_time)),
        "estimated_finish_at": estimated_finish_at,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "eta_seconds": None if eta_seconds is None else round(eta_seconds, 3),
        "progress_percent": round(progress_percent, 3),
        "attempts_per_second": round(attempts_per_second, 4),
        "completed_attempt_count": finished_attempts,
        "finished_attempt_count": finished_attempts,
        "resumed_attempt_count": baseline_finished,
        "newly_finished_attempt_count": newly_finished,
        "checkpointed_attempt_count": max(0, int(checkpointed_attempt_count)),
        "remaining_attempt_count": remaining_attempts,
        "total_attempt_count": total_attempts,
        "total_job_count": max(0, int(total_job_count)),
        "completed_job_count": max(0, int(completed_job_count)),
        "parallel_workers": max(1, int(parallel_workers)),
        "batch_size": max(1, int(batch_size)),
    }


def _format_progress(progress: dict[str, Any]) -> str:
    eta_seconds = progress.get("eta_seconds")
    eta = "estimating" if eta_seconds is None else _format_duration(float(eta_seconds))
    return (
        "progress: "
        f"{progress['finished_attempt_count']}/{progress['total_attempt_count']} attempts "
        f"({progress['progress_percent']:.1f}%), "
        f"{progress['checkpointed_attempt_count']} checkpointed, "
        f"{progress['completed_job_count']}/{progress['total_job_count']} jobs, "
        f"{progress['attempts_per_second']:.2f} attempts/s, ETA {eta}"
    )


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.load(stream, Loader=yaml.CSafeLoader)
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML mapping in {path}")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON mapping in {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _read_worker_results(path: Path) -> list[dict[str, Any]]:
    """Read complete worker rows while tolerating only a crash-truncated tail."""

    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                # The parent may have killed a timed-out worker between bytes.
                # That partial row has no completed attempt boundary and is
                # represented by the synthesized infrastructure result.
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_text_replace(
        path,
        json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_text_replace(
        path,
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows),
    )


def _atomic_text_replace(path: Path, content: str) -> None:
    """Durably replace a checkpoint without exposing a truncated file."""

    temporary = path.parent / (
        f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    with temporary.open("w", encoding="utf-8") as output:
        output.write(content)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)
    try:
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    except (AttributeError, OSError):
        return
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Durably checkpoint one completed batch before launching more work."""

    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _append_jsonl_rows(output: Any, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        json.dump(row, output, sort_keys=True, ensure_ascii=False)
        output.write("\n")


def _durable_flush(output: Any) -> None:
    output.flush()
    os.fsync(output.fileno())


def _chunks(
    rows: list[dict[str, Any]], size: int
) -> list[list[dict[str, Any]]]:
    chunk_size = max(1, int(size))
    return [
        rows[index : index + chunk_size]
        for index in range(0, len(rows), chunk_size)
    ]


def _arm_homogeneous_chunks(
    rows: list[dict[str, Any]],
    size: int,
) -> list[list[dict[str, Any]]]:
    """Keep each reusable CuRobo process bound to one selected arm."""

    by_arm: dict[str, list[dict[str, Any]]] = {"left": [], "right": []}
    for row in rows:
        arm = str((row.get("probe_job") or {}).get("selected_arm") or "")
        if arm not in by_arm:
            raise ValueError(
                f"canonical skill attempt has invalid selected_arm: {arm!r}"
            )
        by_arm[arm].append(row)
    chunks: list[list[dict[str, Any]]] = []
    for arm in ("left", "right"):
        chunks.extend(_chunks(by_arm[arm], size))
    return chunks


def _effective_batch_size(
    *,
    pending_attempt_count: int,
    worker_count: int,
    requested_batch_size: int,
) -> int:
    """Avoid extra cold-start waves on small focused/canary runs."""

    requested = max(1, int(requested_batch_size))
    pending = max(0, int(pending_attempt_count))
    workers = max(1, int(worker_count))
    if pending == 0:
        return requested
    attempts_per_worker = max(1, (pending + workers - 1) // workers)
    return min(requested, attempts_per_worker)


def _mark_chunk_warmups(
    chunks: list[list[dict[str, Any]]],
    *,
    reuse_task_env: bool,
) -> None:
    """Mark attempts that construct a new robot/planner stack."""

    for chunk in chunks:
        for index, spec in enumerate(chunk):
            spec["warmup"] = bool(not reuse_task_env or index == 0)


def _repair_jsonl_append_boundary(path: Path) -> None:
    """Drop only an invalid crash-truncated tail before a resume append."""

    if not path.is_file():
        return
    with path.open("r+b") as output:
        output.seek(0, os.SEEK_END)
        size = output.tell()
        if size == 0:
            return
        output.seek(-1, os.SEEK_END)
        if output.read(1) == b"\n":
            return
        cursor = size
        tail_start = 0
        while cursor > 0:
            start = max(0, cursor - 64 * 1024)
            output.seek(start)
            chunk = output.read(cursor - start)
            newline_index = chunk.rfind(b"\n")
            if newline_index >= 0:
                tail_start = start + newline_index + 1
                break
            cursor = start
        output.seek(tail_start)
        tail = output.read(size - tail_start)
        try:
            decoded = json.loads(tail.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            decoded = None
        if isinstance(decoded, dict):
            output.seek(size)
            output.write(b"\n")
        else:
            output.truncate(tail_start)
        output.flush()
        os.fsync(output.fileno())


def _discard_checkpointed_batch_payloads(
    *,
    batch_dir: Path,
    batch_index: int,
    attempts: list[dict[str, Any]],
) -> None:
    """Delete duplicate payloads after their semantic rows are durable."""

    diagnostic_reasons = set(INFRASTRUCTURE_FAILURES) | {
        "source_initial_pose_mismatch"
    }
    if not attempts or any(
        attempt.get("worker_returncode") != 0
        or str(attempt.get("failure_reason") or "") in diagnostic_reasons
        for attempt in attempts
    ):
        return
    for suffix in (".json", ".results.jsonl"):
        path = batch_dir / f"skill_batch_{int(batch_index):06d}{suffix}"
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            # Cleanup is operational only; the durable checkpoint remains the
            # source of truth if a shared filesystem refuses this unlink.
            pass


def _write_timing_summary(
    path: Path,
    *,
    attempts: list[dict[str, Any]],
    records: list[dict[str, Any]],
    batch_size: int,
    requested_batch_size: int,
    reuse_task_env: bool,
    parallel_workers: int,
    cuda_devices: list[str] | None,
) -> dict[str, Any]:
    summary = {
        "schema_version": 1,
        "source_kind": CANONICAL_SOURCE_KIND,
        "attempt_count": len(attempts),
        "record_count": len(records),
        "batch_size": int(batch_size),
        "requested_batch_size": int(requested_batch_size),
        "effective_batch_size": int(batch_size),
        "reuse_task_env": bool(reuse_task_env),
        "parallel_workers": int(parallel_workers),
        "cuda_devices": list(cuda_devices or []),
        "timing_keys": _timing_key_stats(attempts),
        "by_relation": _timing_stats_by_relation(attempts),
        "by_warmup": {
            label: {
                "attempt_count": len(rows),
                "timing_keys": _timing_key_stats(rows),
            }
            for label, rows in (
                ("cold", [row for row in attempts if row.get("warmup")]),
                ("warm", [row for row in attempts if not row.get("warmup")]),
            )
        },
        "slowest_attempts": _slowest_attempts(attempts),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(summary, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return summary


def _timing_key_stats(
    attempts: list[dict[str, Any]],
) -> dict[str, dict[str, float | int]]:
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
    return {
        key: _numeric_stats(values)
        for key, values in sorted(values_by_key.items())
        if values
    }


def _timing_stats_by_relation(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for attempt in attempts:
        grouped.setdefault(str(attempt.get("relation")), []).append(attempt)
    return {
        relation: {
            "attempt_count": len(rows),
            "timing_keys": _timing_key_stats(rows),
        }
        for relation, rows in sorted(grouped.items())
    }


def _numeric_stats(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": _clean_timing(ordered[0]),
        "p50": _clean_timing(_percentile(ordered, 0.50)),
        "p90": _clean_timing(_percentile(ordered, 0.90)),
        "max": _clean_timing(ordered[-1]),
        "mean": _clean_timing(sum(ordered) / len(ordered)),
    }


def _percentile(values: list[float], percentile: float) -> float:
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def _slowest_attempts(
    attempts: list[dict[str, Any]], limit: int = 10
) -> list[dict[str, Any]]:
    def duration(row: dict[str, Any]) -> float:
        try:
            return float(row.get("duration_sec") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    return [
        {
            "job_id": row.get("job_id"),
            "pair_key": row.get("pair_key"),
            "relation": row.get("relation"),
            "seed": row.get("seed"),
            "success": bool(row.get("success")),
            "failure_reason": row.get("failure_reason"),
            "duration_sec": _clean_timing(duration(row)),
        }
        for row in sorted(attempts, key=duration, reverse=True)[:limit]
    ]


def _clean_timing(value: float) -> float:
    return round(float(value), 12)


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
