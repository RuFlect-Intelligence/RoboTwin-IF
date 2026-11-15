"""Parallel runner for Milestone 5 pair terminal-settle probes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml

from .asset_probe_pose import canonical_pose_policy_hash
from .pair_validation import (
    ACTIVE_RELATIONS,
    DISABLED_RELATIONS,
    RELATIONS,
    REPORT_SCHEMA_VERSION,
    SCHEMA_VERSION as PAIR_PLAN_SCHEMA_VERSION,
    STAGE_TERMINAL,
    TERMINAL_SOURCE_KIND,
    _artifact_hash,
    _clean_float,
    _evidence_id,
    _expected_terminal_job_recipe_id,
    _path_string,
    _row_config_hash,
    config_hash,
    require_active_relation_selection,
    terminal_probe_policy_hash,
)
from .pair_yaw_geometry import yaw_solver_policy_hash


SCHEMA_VERSION = 2
DEFAULT_PROBE_SEEDS = (0, 1, 2)
DEFAULT_PROBE_THRESHOLDS = {relation: 0.6 for relation in RELATIONS}
DEFAULT_TIMEOUT_SECONDS = 180
DEFAULT_WARMUP_TIMEOUT_SECONDS = 1800
DEFAULT_BATCH_SIZE = 64
PROGRESS_EMIT_INTERVAL_SECONDS = 5.0

RUN_BLOCKING_REASONS = {
    "worker_import_failed",
    "worker_exception",
    "worker_process_failed",
    "worker_timeout",
    "worker_no_json_result",
    "actor_load_failed",
    "source_initial_pose_mismatch",
}
SPEC_BLOCKING_REASONS = {
    "missing_pair_metadata",
    "unsupported_relation",
}


def run_terminal_settle_plan(
    *,
    asset_catalog_path: Path,
    plan_path: Path,
    reports_dir: Path,
    run_dir: Path,
    python_executable: str | None = None,
    seeds: list[int] | None = None,
    thresholds: dict[str, float] | None = None,
    relations: set[str] | None = None,
    pair_keys: set[str] | None = None,
    start_index: int = 0,
    max_jobs: int | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    warmup_timeout_seconds: int = DEFAULT_WARMUP_TIMEOUT_SECONDS,
    torch_extensions_dir: Path | None = None,
    clear_existing_reports: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
    reuse_task_env: bool = True,
    parallel_workers: int = 1,
    cuda_devices: list[str] | None = None,
    source_kind: str = TERMINAL_SOURCE_KIND,
    resume: bool = False,
) -> dict[str, Any]:
    """Run terminal-settle pair probes and write one normalized pair report."""

    require_active_relation_selection(relations=relations, pair_keys=pair_keys)
    plan = _load_yaml_required(plan_path)
    _validate_probe_plan_freshness(plan)
    all_jobs = plan.get("jobs") or []
    selected_jobs = _select_jobs(
        all_jobs,
        relations=set(ACTIVE_RELATIONS) if relations is None else relations,
        pair_keys=pair_keys,
        start_index=start_index,
        max_jobs=max_jobs,
    )
    probe_seeds = list(DEFAULT_PROBE_SEEDS if seeds is None else seeds)
    probe_thresholds = {**DEFAULT_PROBE_THRESHOLDS, **(thresholds or {})}
    python = python_executable or sys.executable
    torch_cache = torch_extensions_dir or (run_dir / "torch_extensions")
    worker_count = max(1, int(parallel_workers))
    device_slots = _worker_device_slots(cuda_devices, worker_count)
    requested_batch_size = max(1, int(batch_size))

    reports_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / "checkpoints"
    attempts_checkpoint = checkpoint_dir / "attempts.jsonl"
    records_checkpoint = checkpoint_dir / "records.jsonl"
    prior_run_metadata: dict[str, Any] | None = None
    if resume:
        if clear_existing_reports:
            raise ValueError("--resume cannot be combined with --clear-existing-reports")
        prior_run_metadata = _load_json_required(checkpoint_dir / "run_metadata.json")
        _validate_resume_metadata(
            prior_run_metadata,
            plan=plan,
            selected_jobs=selected_jobs,
            seeds=probe_seeds,
            thresholds=probe_thresholds,
            source_kind=source_kind,
        )
        if not attempts_checkpoint.is_file():
            raise ValueError(f"resume checkpoint does not exist: {attempts_checkpoint}")
        archived_run = None
    else:
        archived_run = _archive_existing_run_artifacts(run_dir)
    torch_cache.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    if not resume:
        _clear_batch_artifacts(run_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    timing_summary_path = run_dir / "timing_summary.yml"
    progress_path = run_dir / "progress.json"
    if not resume:
        for checkpoint_path in (attempts_checkpoint, records_checkpoint):
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            checkpoint_path.touch()
    if clear_existing_reports:
        clear_terminal_reports(reports_dir)

    now_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    started_at = str((prior_run_metadata or {}).get("started_at") or now_utc)
    resume_count = int((prior_run_metadata or {}).get("resume_count") or 0) + (1 if resume else 0)
    run_metadata = {
        "schema_version": SCHEMA_VERSION,
        "source_kind": source_kind,
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
        "terminal_probe_policy_hash": plan.get("terminal_probe_policy_hash"),
        "canonical_pose_policy_hash": plan.get("canonical_pose_policy_hash"),
        "yaw_solver_policy_hash": plan.get("yaw_solver_policy_hash"),
        "source_asset_catalog_path": _path_string(asset_catalog_path),
        "selected_job_count": len(selected_jobs),
        "seeds": probe_seeds,
        "thresholds": probe_thresholds,
        "timeout_seconds": int(timeout_seconds),
        "warmup_timeout_seconds": int(warmup_timeout_seconds),
        "torch_extensions_dir": _path_string(torch_cache),
        # ``batch_size`` remains the operational/effective value for backward
        # compatibility.  It is refined once resume filtering reveals the
        # exact pending-attempt count.
        "batch_size": requested_batch_size,
        "requested_batch_size": requested_batch_size,
        "effective_batch_size": requested_batch_size,
        "reuse_task_env": bool(reuse_task_env),
        "parallel_workers": worker_count,
        "cuda_devices": list(cuda_devices or []),
        "progress_path": _path_string(progress_path),
        "started_at": started_at,
        "archived_previous_run": (
            (prior_run_metadata or {}).get("archived_previous_run")
            if resume
            else _path_string(archived_run)
        ),
        "resume": bool(resume),
        "resume_count": resume_count,
        "resumed_at": now_utc if resume else None,
    }
    _write_json_atomic(checkpoint_dir / "run_metadata.json", run_metadata)

    batch_result = _run_attempt_batches(
        python_executable=python,
        asset_catalog_path=asset_catalog_path,
        jobs=selected_jobs,
        start_index=start_index,
        seeds=probe_seeds,
        thresholds=probe_thresholds,
        run_dir=run_dir,
        timeout_seconds=int(timeout_seconds),
        warmup_timeout_seconds=int(warmup_timeout_seconds),
        torch_extensions_dir=torch_cache,
        batch_size=requested_batch_size,
        reuse_task_env=reuse_task_env,
        parallel_workers=worker_count,
        device_slots=device_slots,
        attempts_checkpoint=attempts_checkpoint,
        records_checkpoint=records_checkpoint,
        resume=resume,
    )
    band_records = batch_result["records"]
    attempt_rows = batch_result["attempts"]
    effective_batch_size = int(
        batch_result.get("effective_batch_size") or requested_batch_size
    )
    run_metadata["batch_size"] = effective_batch_size
    run_metadata["requested_batch_size"] = requested_batch_size
    run_metadata["effective_batch_size"] = effective_batch_size
    run_metadata["pending_attempt_count"] = int(
        batch_result.get("pending_attempt_count") or 0
    )
    _write_json_atomic(checkpoint_dir / "run_metadata.json", run_metadata)
    records = _aggregate_pair_records(band_records, selected_jobs)

    timing_summary = _write_timing_summary(
        timing_summary_path,
        attempts=attempt_rows,
        records=band_records,
        batch_size=effective_batch_size,
        requested_batch_size=requested_batch_size,
        reuse_task_env=reuse_task_env,
        parallel_workers=worker_count,
        cuda_devices=cuda_devices,
        source_kind=source_kind,
    )

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "kind": "terminal_settle_pair_report",
        "scope": "pair",
        "stage": STAGE_TERMINAL,
        "source_kind": source_kind,
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
        "terminal_probe_policy_hash": plan.get("terminal_probe_policy_hash"),
        "canonical_pose_policy_hash": plan.get("canonical_pose_policy_hash"),
        "yaw_solver_policy_hash": plan.get("yaw_solver_policy_hash"),
        "source_asset_catalog_path": _path_string(asset_catalog_path),
        "source_asset_catalog_sha256": _sha256_file(asset_catalog_path),
        "runner": {
            **run_metadata,
            "python_executable": python,
            "timing_summary_path": _path_string(timing_summary_path),
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "resume_summary": batch_result.get("resume_summary") or {},
        },
        "summary": _summary(records, attempt_rows),
        "timing_summary": timing_summary,
        "checkpoints": {
            "attempts": _path_string(attempts_checkpoint),
            "records": _path_string(records_checkpoint),
            "progress": _path_string(progress_path),
        },
        "records": records,
    }
    report["evidence_id"] = _evidence_id(report)
    report["report_hash"] = _artifact_hash(report, "report_hash")
    report_path = reports_dir / f"{report['evidence_id']}.json"
    _write_json(report_path, report)
    return {
        "evidence_id": report["evidence_id"],
        "evidence_path": _path_string(report_path),
        "record_count": len(records),
        "band_record_count": len(band_records),
        "attempt_count": len(attempt_rows),
        "summary": report["summary"],
        "timing_summary_path": _path_string(timing_summary_path),
        "resume_summary": batch_result.get("resume_summary") or {},
    }


def migrate_terminal_settle_evidence(
    *,
    asset_catalog_path: Path,
    plan_path: Path,
    source_attempts_path: Path,
    replacement_attempts_paths: list[Path],
    reports_dir: Path,
    run_dir: Path,
    seeds: list[int] | None = None,
    thresholds: dict[str, float] | None = None,
    source_kind: str = TERMINAL_SOURCE_KIND,
) -> dict[str, Any]:
    """Migrate complete v7 attempts into a complete v8 report without rerunning unchanged physics."""

    from .pair_terminal_settle_worker import place_in_success_metrics

    plan = _load_yaml_required(plan_path)
    _validate_probe_plan_freshness(plan)
    jobs = [
        job
        for job in (plan.get("jobs") or [])
        if str(job.get("relation")) in ACTIVE_RELATIONS
    ]
    probe_seeds = list(DEFAULT_PROBE_SEEDS if seeds is None else seeds)
    probe_thresholds = {**DEFAULT_PROBE_THRESHOLDS, **(thresholds or {})}
    expected_specs = _build_attempt_specs(
        jobs,
        probe_seeds,
        start_index=0,
        worker_count=1,
        batch_size=1,
    )
    expected_by_id = {str(spec["attempt_id"]): spec for spec in expected_specs}
    if len(expected_by_id) != len(expected_specs):
        raise ValueError("v8 terminal plan produced duplicate attempt ids")
    jobs_by_id = {str(job["job_id"]): job for job in jobs}

    source_metadata = _load_json_required(source_attempts_path.parent / "run_metadata.json")
    source_policy_hash = source_metadata.get("terminal_probe_policy_hash")
    if source_policy_hash == plan.get("terminal_probe_policy_hash"):
        raise ValueError("source attempts already use the target terminal policy; migration is unnecessary")

    replacements: dict[str, dict[str, Any]] = {}
    replacement_sources: dict[str, str] = {}
    replacement_metadata_rows: list[dict[str, Any]] = []
    for path in replacement_attempts_paths:
        metadata = _load_json_required(path.parent / "run_metadata.json")
        invariant_keys = (
            "config_hash",
            "source_rubric_hash",
            "canonical_pose_policy_hash",
            "yaw_solver_policy_hash",
        )
        mismatches = [
            key
            for key in invariant_keys
            if plan.get(key) is not None and metadata.get(key) != plan.get(key)
        ]
        if mismatches:
            raise ValueError(
                f"replacement attempts are incompatible with the target plan ({', '.join(mismatches)}): {path}"
            )
        replacement_metadata_rows.append(
            {
                "path": _path_string(path),
                "attempt_count": 0,
                "source_probe_plan_hash": metadata.get("source_probe_plan_hash"),
                "terminal_probe_policy_hash": metadata.get("terminal_probe_policy_hash"),
            }
        )
        for row in _iter_jsonl_required(path):
            attempt_id = str(row.get("attempt_id"))
            if attempt_id not in expected_by_id:
                if str(row.get("relation") or "") in DISABLED_RELATIONS:
                    continue
                raise ValueError(f"replacement attempt is not present in target plan: {attempt_id}")
            if attempt_id in replacements:
                raise ValueError(f"duplicate replacement attempt id: {attempt_id}")
            replacements[attempt_id] = row
            replacement_sources[attempt_id] = _path_string(path) or str(path)
            replacement_metadata_rows[-1]["attempt_count"] += 1

    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    attempts_checkpoint = checkpoint_dir / "attempts.jsonl"
    records_checkpoint = checkpoint_dir / "records.jsonl"
    temporary_attempts = attempts_checkpoint.parent / f".{attempts_checkpoint.name}.tmp"
    compact_attempts_by_job: dict[str, list[dict[str, Any]]] = {}
    seen_attempt_ids: set[str] = set()
    migration_counts: Counter[str] = Counter()
    place_in_outcome_changes: Counter[str] = Counter()

    with temporary_attempts.open("w", encoding="utf-8") as output:
        for source_row in _iter_jsonl_required(source_attempts_path):
            attempt_id = str(source_row.get("attempt_id"))
            if attempt_id not in expected_by_id or attempt_id in replacements:
                continue
            if attempt_id in seen_attempt_ids:
                raise ValueError(f"duplicate source attempt id: {attempt_id}")
            spec = expected_by_id[attempt_id]
            relation = str(spec.get("relation"))
            migrated = dict(source_row)
            mode = "rechecked_place_in" if relation == "place_in" else "reused_unchanged_relation"
            if relation == "dump_into":
                raise ValueError(f"dump_into attempt lacks fresh v8 replacement: {attempt_id}")
            if relation == "place_in" and migrated.get("plan_success"):
                before = bool(migrated.get("success"))
                migrated = _reevaluate_place_in_attempt(migrated, place_in_success_metrics)
                after = bool(migrated.get("success"))
                place_in_outcome_changes[f"{'pass' if before else 'fail'}_to_{'pass' if after else 'fail'}"] += 1
            migrated = _bind_attempt_to_target_spec(
                migrated,
                spec=spec,
                job=jobs_by_id[str(spec["job_id"])],
                migration={
                    "mode": mode,
                    "source_attempts_path": _path_string(source_attempts_path),
                    "source_terminal_probe_policy_hash": source_policy_hash,
                    "target_terminal_probe_policy_hash": plan.get("terminal_probe_policy_hash"),
                },
            )
            _append_jsonl_row(output, migrated)
            seen_attempt_ids.add(attempt_id)
            compact_attempts_by_job.setdefault(str(spec["job_id"]), []).append(_compact_attempt(migrated))
            migration_counts[mode] += 1

        for attempt_id in sorted(replacements, key=lambda value: int(expected_by_id[value]["job_index"]) * 10 + int(expected_by_id[value]["seed"])):
            if attempt_id in seen_attempt_ids:
                raise ValueError(f"replacement attempt collides with migrated source attempt: {attempt_id}")
            spec = expected_by_id[attempt_id]
            job = jobs_by_id[str(spec["job_id"])]
            relation = str(spec.get("relation"))
            migrated = dict(replacements[attempt_id])
            source_path = replacement_sources[attempt_id]
            source_replacement_policy = next(
                row["terminal_probe_policy_hash"]
                for row in replacement_metadata_rows
                if row["path"] == source_path
            )
            if relation == "place_in":
                migrated = _reevaluate_place_in_attempt(migrated, place_in_success_metrics)
                mode = "rechecked_place_in_replacement"
            elif relation == "dump_into":
                _validate_dump_into_replacement_attempt(migrated)
                mode = "validated_dump_into_replacement"
            else:
                mode = "reused_unchanged_replacement_relation"
            migrated = _bind_attempt_to_target_spec(
                migrated,
                spec=spec,
                job=job,
                migration={
                    "mode": mode,
                    "source_attempts_path": source_path,
                    "source_terminal_probe_policy_hash": source_replacement_policy,
                    "target_terminal_probe_policy_hash": plan.get("terminal_probe_policy_hash"),
                },
            )
            _append_jsonl_row(output, migrated)
            seen_attempt_ids.add(attempt_id)
            compact_attempts_by_job.setdefault(str(spec["job_id"]), []).append(_compact_attempt(migrated))
            migration_counts[mode] += 1

    missing_attempt_ids = sorted(set(expected_by_id) - seen_attempt_ids)
    extra_attempt_ids = sorted(seen_attempt_ids - set(expected_by_id))
    if missing_attempt_ids or extra_attempt_ids:
        temporary_attempts.unlink(missing_ok=True)
        raise ValueError(
            "terminal evidence migration coverage mismatch: "
            f"missing={len(missing_attempt_ids)} extra={len(extra_attempt_ids)} "
            f"missing_examples={missing_attempt_ids[:5]} extra_examples={extra_attempt_ids[:5]}"
        )
    temporary_attempts.replace(attempts_checkpoint)

    band_records = [
        _record_from_attempts(
            job,
            compact_attempts_by_job.get(str(job["job_id"]), []),
            float(probe_thresholds[str(job["relation"])]),
            probe_seeds,
        )
        for job in jobs
    ]
    with records_checkpoint.open("w", encoding="utf-8") as output:
        for record in band_records:
            _append_jsonl_row(output, record)
    records = _aggregate_pair_records(band_records, jobs)
    compact_attempts = [
        attempt
        for job in jobs
        for attempt in compact_attempts_by_job.get(str(job["job_id"]), [])
    ]
    if any(record.get("outcome") == "blocked" for record in band_records):
        blocked = [record["job_id"] for record in band_records if record.get("outcome") == "blocked"]
        raise ValueError(f"migrated report contains blocked band records: {blocked[:10]}")

    reports_dir.mkdir(parents=True, exist_ok=True)
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    migration_summary = {
        "kind": "terminal_settle_policy_migration",
        "source_attempts_path": _path_string(source_attempts_path),
        "source_terminal_probe_policy_hash": source_policy_hash,
        "target_terminal_probe_policy_hash": plan.get("terminal_probe_policy_hash"),
        "replacement_sources": replacement_metadata_rows,
        "migration_counts": dict(sorted(migration_counts.items())),
        "place_in_outcome_changes": dict(sorted(place_in_outcome_changes.items())),
        "expected_attempt_count": len(expected_by_id),
        "migrated_attempt_count": len(seen_attempt_ids),
        "expected_job_count": len(jobs),
        "migrated_job_count": len(band_records),
        "coverage_complete": True,
    }
    run_metadata = {
        "schema_version": SCHEMA_VERSION,
        "source_kind": source_kind,
        "source_probe_plan_path": _path_string(plan_path),
        "source_probe_plan_hash": plan.get("probe_plan_hash"),
        "config_hash": plan.get("config_hash"),
        "source_rubric_hash": plan.get("source_rubric_hash"),
        "terminal_probe_policy_hash": plan.get("terminal_probe_policy_hash"),
        "canonical_pose_policy_hash": plan.get("canonical_pose_policy_hash"),
        "yaw_solver_policy_hash": plan.get("yaw_solver_policy_hash"),
        "source_asset_catalog_path": _path_string(asset_catalog_path),
        "selected_job_count": len(jobs),
        "seeds": probe_seeds,
        "thresholds": probe_thresholds,
        "started_at": now,
        "finished_at": now,
        "migration": migration_summary,
    }
    _write_json(checkpoint_dir / "run_metadata.json", run_metadata)
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "kind": "terminal_settle_pair_report",
        "scope": "pair",
        "stage": STAGE_TERMINAL,
        "source_kind": source_kind,
        "source_probe_plan_path": _path_string(plan_path),
        "source_probe_plan_hash": plan.get("probe_plan_hash"),
        "config_hash": plan.get("config_hash"),
        "source_rubric_hash": plan.get("source_rubric_hash"),
        "terminal_probe_policy_hash": plan.get("terminal_probe_policy_hash"),
        "canonical_pose_policy_hash": plan.get("canonical_pose_policy_hash"),
        "yaw_solver_policy_hash": plan.get("yaw_solver_policy_hash"),
        "source_asset_catalog_path": _path_string(asset_catalog_path),
        "source_asset_catalog_sha256": _sha256_file(asset_catalog_path),
        "runner": run_metadata,
        "summary": _summary(records, compact_attempts),
        "migration": migration_summary,
        "checkpoints": {
            "attempts": _path_string(attempts_checkpoint),
            "records": _path_string(records_checkpoint),
        },
        "records": records,
    }
    report["evidence_id"] = _evidence_id(report)
    report["report_hash"] = _artifact_hash(report, "report_hash")
    report_path = reports_dir / f"{report['evidence_id']}.json"
    _write_json(report_path, report)
    return {
        "evidence_id": report["evidence_id"],
        "evidence_path": _path_string(report_path),
        "record_count": len(records),
        "band_record_count": len(band_records),
        "attempt_count": len(compact_attempts),
        "summary": report["summary"],
        "migration": migration_summary,
    }


def _reevaluate_place_in_attempt(
    attempt: dict[str, Any],
    checker: Any,
) -> dict[str, Any]:
    metrics = dict(attempt.get("metrics") or {})
    required = ("subject_aabb", "receiver_aabb", "contacts", "terminal_stability")
    missing = [key for key in required if not isinstance(metrics.get(key), dict)]
    if missing:
        raise ValueError(f"valid place_in attempt lacks migration metrics {missing}: {attempt.get('attempt_id')}")
    reevaluated = checker(
        {"aabb": metrics["subject_aabb"]},
        {"aabb": metrics["receiver_aabb"]},
        metrics["contacts"],
        metrics["terminal_stability"],
    )
    metrics.update(reevaluated)
    clearance_attempts = []
    for clearance_attempt in list(metrics.get("clearance_attempts") or []):
        nested = dict(clearance_attempt)
        nested_metrics = dict(nested.get("metrics") or {})
        if all(isinstance(nested_metrics.get(key), dict) for key in required):
            nested_result = checker(
                {"aabb": nested_metrics["subject_aabb"]},
                {"aabb": nested_metrics["receiver_aabb"]},
                nested_metrics["contacts"],
                nested_metrics["terminal_stability"],
            )
            nested_metrics.update(nested_result)
            nested["metrics"] = nested_metrics
            nested["success"] = bool(nested_result["success"])
            nested["failure_reason"] = str(nested_result["failure_reason"])
        clearance_attempts.append(nested)
    if clearance_attempts:
        metrics["clearance_attempts"] = clearance_attempts
    attempt["metrics"] = metrics
    attempt["success"] = bool(reevaluated["success"])
    attempt["failure_reason"] = str(reevaluated["failure_reason"])
    return attempt


def _bind_attempt_to_target_spec(
    attempt: dict[str, Any],
    *,
    spec: dict[str, Any],
    job: dict[str, Any],
    migration: dict[str, Any],
) -> dict[str, Any]:
    attempt.update(
        {
            "attempt_id": spec["attempt_id"],
            "job_id": spec["job_id"],
            "job_index": spec["job_index"],
            "pair_key": spec["pair_key"],
            "relation": spec["relation"],
            "seed": int(spec["seed"]),
            "subject": _endpoint_identity(job.get("subject") or {}),
            "receiver": _endpoint_identity(job.get("receiver") or {}),
            "migration": migration,
        }
    )
    info = dict(attempt.get("info") or {})
    info["recipe_id"] = job.get("recipe_id")
    info["yaw_witness"] = job.get("yaw_witness") or {}
    attempt["info"] = info
    return attempt


def _validate_dump_into_replacement_attempt(attempt: dict[str, Any]) -> None:
    metrics = dict(attempt.get("metrics") or {})
    proxy_contents = list(metrics.get("proxy_contents") or [])
    release_layout = dict(metrics.get("proxy_release_layout") or {})
    checks = dict(metrics.get("checks") or {})
    compatible = (
        metrics.get("source_container_in_terminal_scene") is False
        and len(proxy_contents) == 5
        and int(release_layout.get("count") or 0) == 5
        and bool(release_layout.get("non_overlapping"))
        and "all_proxy_contents_inside_receiver_xy" in checks
        and "all_proxy_contents_above_table" in checks
    )
    if not compatible:
        raise ValueError(
            "dump_into replacement does not carry the reviewed five-proxy terminal-retention contract: "
            f"{attempt.get('attempt_id')}"
        )


def _compact_attempt(attempt: dict[str, Any]) -> dict[str, Any]:
    return {
        key: attempt.get(key)
        for key in (
            "attempt_id",
            "job_id",
            "job_index",
            "pair_key",
            "relation",
            "subject",
            "receiver",
            "seed",
            "plan_success",
            "success",
            "failure_reason",
            "duration_sec",
            "warmup",
            "migration",
        )
    }


def _iter_jsonl_required(path: Path):
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL row at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            yield row


def _append_jsonl_row(output: Any, row: dict[str, Any]) -> None:
    json.dump(row, output, sort_keys=True, ensure_ascii=False)
    output.write("\n")


def clear_terminal_reports(reports_dir: Path) -> int:
    removed = 0
    if not reports_dir.is_dir():
        return removed
    for path in sorted(reports_dir.glob("terminal_settle_pair_*.json")):
        try:
            path.unlink()
            removed += 1
        except FileNotFoundError:
            pass
    return removed


def _validate_probe_plan_freshness(plan: dict[str, Any]) -> None:
    if plan.get("probe_plan_hash") != _artifact_hash(plan, "probe_plan_hash"):
        raise ValueError("terminal-settle probe plan hash mismatch")
    if int(plan.get("schema_version") or 0) != PAIR_PLAN_SCHEMA_VERSION:
        raise ValueError(
            "terminal-settle plan schema is stale; regenerate with "
            "plan-terminal-settle"
        )
    if str(plan.get("stage") or "") != STAGE_TERMINAL:
        raise ValueError("terminal-settle plan stage is invalid")
    if not plan.get("source_validated_pose_set_hash"):
        raise ValueError("terminal-settle plan has no validated Initial Pose Set")
    for job in plan.get("jobs") or []:
        if set((job.get("initial_pose_bindings") or {})) != {"subject", "receiver"}:
            raise ValueError(f"terminal-settle job lacks Initial Pose bindings: {job.get('job_id')}")
        if int(job.get("expected_report_schema") or 0) != REPORT_SCHEMA_VERSION:
            raise ValueError(
                "terminal-settle job report schema is stale: "
                f"{job.get('job_id')}"
            )
        if str(job.get("recipe_id") or "") != (
            _expected_terminal_job_recipe_id(job)
        ):
            raise ValueError(
                "terminal-settle job recipe binding is stale: "
                f"{job.get('job_id')}"
            )
    if plan.get("config_hash") != config_hash():
        raise ValueError("terminal-settle plan config hash is stale; regenerate with plan-terminal-settle")
    if plan.get("terminal_probe_policy_hash") != terminal_probe_policy_hash():
        raise ValueError("terminal-settle plan policy hash is stale; regenerate with plan-terminal-settle")
    if plan.get("canonical_pose_policy_hash") is not None and plan.get("canonical_pose_policy_hash") != canonical_pose_policy_hash():
        raise ValueError("terminal-settle plan canonical pose policy is stale; regenerate with plan-terminal-settle")
    if plan.get("yaw_solver_policy_hash") is not None and plan.get("yaw_solver_policy_hash") != yaw_solver_policy_hash():
        raise ValueError("terminal-settle plan yaw solver policy is stale; regenerate with plan-terminal-settle")
    if not plan.get("source_rubric_hash"):
        raise ValueError("terminal-settle plan has no rubric hash; regenerate with plan-terminal-settle")


def _validate_resume_metadata(
    metadata: dict[str, Any],
    *,
    plan: dict[str, Any],
    selected_jobs: list[dict[str, Any]],
    seeds: list[int],
    thresholds: dict[str, float],
    source_kind: str,
) -> None:
    expected = {
        "source_probe_plan_hash": plan.get("probe_plan_hash"),
        "source_validated_pose_set_hash": plan.get(
            "source_validated_pose_set_hash"
        ),
        "initial_state_selection_version": plan.get(
            "initial_state_selection_version"
        ),
        "initial_state_selection_hash": plan.get("initial_state_selection_hash"),
        "config_hash": plan.get("config_hash"),
        "source_rubric_hash": plan.get("source_rubric_hash"),
        "terminal_probe_policy_hash": plan.get("terminal_probe_policy_hash"),
        "canonical_pose_policy_hash": plan.get("canonical_pose_policy_hash"),
        "yaw_solver_policy_hash": plan.get("yaw_solver_policy_hash"),
        "selected_job_count": len(selected_jobs),
        "seeds": [int(seed) for seed in seeds],
        "thresholds": thresholds,
        "source_kind": source_kind,
    }
    mismatches = [
        key
        for key, value in expected.items()
        if metadata.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            "resume checkpoint is incompatible with the requested run: "
            + ", ".join(sorted(mismatches))
        )


def audit_terminal_settle_run(*, plan_path: Path, checkpoint_dir: Path, output_path: Path) -> dict[str, Any]:
    """Join failed checkpoint records back to their rubric and geometry evidence."""

    plan = _load_yaml_required(plan_path)
    jobs_by_id = {str(job.get("job_id")): job for job in plan.get("jobs") or []}
    records = _read_batch_results(checkpoint_dir / "records.jsonl")
    failures: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    relations: Counter[str] = Counter()
    expectation_counts: Counter[str] = Counter()
    for record in records:
        if record.get("outcome") == "pass" or record.get("pass"):
            continue
        pair_key = str(record.get("pair_key"))
        job = jobs_by_id.get(str(record.get("job_id")))
        if job is None:
            expectation = "not_in_current_plan"
            rubric_review = {}
            job = {}
        else:
            rubric_review = job.get("rubric_review") or {}
            decision = str(rubric_review.get("verdict") or "")
            if rubric_review.get("pair_expectation") == "positive":
                expectation_scope = str(rubric_review.get("pair_expectation_scope") or "")
                evidence = str(rubric_review.get("pair_expectation_evidence") or "")
                if expectation_scope == "model_pair":
                    expectation = "should_succeed"
                elif expectation_scope == "asset_family" or evidence.startswith("envs/"):
                    expectation = "asset_family_positive"
                else:
                    expectation = "should_succeed"
            elif decision == "review_required":
                expectation = "manual_review"
            else:
                expectation = "geometry_survivor"
        row_reasons = record.get("fail_reasons") or {}
        reasons.update({str(key): int(value) for key, value in row_reasons.items()})
        relations[str(record.get("relation"))] += 1
        expectation_counts[expectation] += 1
        failures.append(
            {
                "pair_key": pair_key,
                "relation": record.get("relation"),
                "expectation": expectation,
                "verdict": record.get("verdict"),
                "success": record.get("success"),
                "attempts": record.get("attempts"),
                "outcome": record.get("outcome"),
                "band_id": record.get("band_id"),
                "yaw_witness": job.get("yaw_witness") or {},
                "recipe_id": job.get("recipe_id"),
                "fail_reasons": row_reasons,
                "rubric_review": rubric_review,
                "geometry_metrics": job.get("geometry_metrics") or {},
                "probe_spec": job.get("probe_spec") or {},
            }
        )
    expectation_rank = {
        "should_succeed": 0,
        "asset_family_positive": 1,
        "manual_review": 2,
        "geometry_survivor": 3,
        "not_in_current_plan": 4,
    }
    failures.sort(
        key=lambda row: (
            expectation_rank.get(str(row["expectation"]), 99),
            str(row["relation"]),
            str(row["pair_key"]),
        )
    )
    audit = {
        "schema_version": 1,
        "kind": "terminal_settle_failure_audit",
        "source_probe_plan_path": _path_string(plan_path),
        "source_probe_plan_hash": plan.get("probe_plan_hash"),
        "source_checkpoint_dir": _path_string(checkpoint_dir),
        "summary": {
            "record_count": len(records),
            "failure_count": len(failures),
            "failure_count_by_expectation": dict(sorted(expectation_counts.items())),
            "failure_count_by_relation": dict(sorted(relations.items())),
            "attempt_failure_reasons": dict(reasons.most_common()),
        },
        "failures": failures,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(audit, f, sort_keys=False, allow_unicode=True, default_flow_style=False, width=120)
    return audit


def _select_jobs(
    jobs: list[dict[str, Any]],
    *,
    relations: set[str] | None,
    pair_keys: set[str] | None,
    start_index: int,
    max_jobs: int | None,
) -> list[dict[str, Any]]:
    selected = [
        job
        for job in jobs
        if job.get("relation") in RELATIONS
        and (relations is None or job.get("relation") in relations)
        and (pair_keys is None or str(job.get("pair_key")) in pair_keys)
    ]
    if start_index:
        selected = selected[start_index:]
    if max_jobs is not None:
        selected = selected[:max_jobs]
    return selected


def _run_attempt_batches(
    *,
    python_executable: str,
    asset_catalog_path: Path,
    jobs: list[dict[str, Any]],
    start_index: int,
    seeds: list[int],
    thresholds: dict[str, float],
    run_dir: Path,
    timeout_seconds: int,
    warmup_timeout_seconds: int,
    torch_extensions_dir: Path,
    batch_size: int,
    reuse_task_env: bool,
    parallel_workers: int,
    device_slots: list[str | None],
    attempts_checkpoint: Path,
    records_checkpoint: Path,
    resume: bool = False,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    attempts_by_job: dict[str, list[dict[str, Any]]] = {str(job["job_id"]): [] for job in jobs}
    jobs_by_id = {str(job["job_id"]): job for job in jobs}
    emitted_jobs: set[str] = set()
    all_attempt_specs = _build_attempt_specs(
        jobs,
        seeds,
        start_index=start_index,
        worker_count=max(1, int(parallel_workers)),
        batch_size=max(1, int(batch_size)),
    )
    resume_summary = {
        "resumed": bool(resume),
        "reused_attempt_count": 0,
        "retry_attempt_count": 0,
        "pending_attempt_count": len(all_attempt_specs),
    }
    retry_attempt_ids: set[str] = set()
    if resume:
        _repair_jsonl_append_boundary(attempts_checkpoint)
        attempts, retry_attempt_ids = _load_resume_attempts(
            attempts_checkpoint,
            all_attempt_specs,
        )
        resume_summary = {
            "resumed": True,
            "reused_attempt_count": len(attempts),
            "retry_attempt_count": len(retry_attempt_ids),
            "pending_attempt_count": len(all_attempt_specs) - len(attempts),
        }
        for attempt in attempts:
            job_id = str(attempt.get("job_id"))
            if job_id in attempts_by_job:
                attempts_by_job[job_id].append(attempt)
        for job in jobs:
            job_id = str(job["job_id"])
            if len(attempts_by_job[job_id]) != len(seeds):
                continue
            record = _record_from_attempts(
                job,
                attempts_by_job[job_id],
                thresholds[job["relation"]],
                seeds,
            )
            records.append(record)
            emitted_jobs.add(job_id)
        _rewrite_jsonl(records_checkpoint, records)
    completed_attempt_ids = {str(attempt["attempt_id"]) for attempt in attempts}
    attempt_specs = [
        spec
        for spec in all_attempt_specs
        if str(spec["attempt_id"]) not in completed_attempt_ids
    ]
    requested_batch_size = max(1, int(batch_size))
    worker_count = max(1, int(parallel_workers))
    effective_batch_size = _effective_batch_size(
        pending_attempt_count=len(attempt_specs),
        worker_count=worker_count,
        requested_batch_size=requested_batch_size,
    )
    _mark_warmup_specs(
        attempt_specs,
        worker_count=worker_count,
        batch_size=effective_batch_size,
    )
    chunks = _chunks(attempt_specs, effective_batch_size)
    batches_dir = run_dir / "batches"
    batches_dir.mkdir(parents=True, exist_ok=True)
    progress_path = run_dir / "progress.json"
    progress_started_monotonic = time.monotonic()
    initial_reused_attempt_count = len(attempts)
    finished_attempt_count = initial_reused_attempt_count
    last_progress_emit = 0.0
    batch_index_offset = _next_batch_index(run_dir) if resume else 0

    _update_run_batch_metadata(
        attempts_checkpoint.parent / "run_metadata.json",
        requested_batch_size=requested_batch_size,
        effective_batch_size=effective_batch_size,
        pending_attempt_count=len(attempt_specs),
    )

    def checkpoint_completed_batch(
        *,
        batch_index: int,
        batch_attempts: list[dict[str, Any]],
        attempts_output: Any,
        records_output: Any,
    ) -> None:
        """Durably checkpoint one completed batch without completion-order HOL blocking."""

        checkpoint_attempts: list[dict[str, Any]] = []
        new_records: list[dict[str, Any]] = []
        for attempt in batch_attempts:
            attempts.append(attempt)
            # Blocker retries are rare and remain part of the permanent audit
            # trail.  Persist the latest row as well as retaining its payload
            # and logs, so this run's report is reproducible from the durable
            # attempt journal.
            checkpoint_attempts.append(attempt)
            job_id = str(attempt.get("job_id"))
            if job_id not in attempts_by_job:
                continue
            attempts_by_job[job_id].append(attempt)
            if len(attempts_by_job[job_id]) >= len(seeds) and job_id not in emitted_jobs:
                job = jobs_by_id[job_id]
                record = _record_from_attempts(
                    job,
                    attempts_by_job[job_id],
                    thresholds[job["relation"]],
                    seeds,
                )
                records.append(record)
                new_records.append(record)
                emitted_jobs.add(job_id)

        if checkpoint_attempts:
            _append_jsonl_rows(attempts_output, checkpoint_attempts)
            _durable_flush(attempts_output)
        _append_jsonl_rows(records_output, new_records)
        # Records are a compact derivative that resume rebuilds from the
        # durable attempt journal.  A normal flush preserves process-crash
        # visibility without imposing a second GPFS fsync per batch.
        records_output.flush()
        _discard_checkpointed_batch_payloads(
            batches_dir=batches_dir,
            batch_index=batch_index,
            attempts=batch_attempts,
        )

    def emit_progress(*, state: str, force: bool = False) -> None:
        nonlocal last_progress_emit
        now_monotonic = time.monotonic()
        if not force and now_monotonic - last_progress_emit < PROGRESS_EMIT_INTERVAL_SECONDS:
            return
        snapshot = _progress_snapshot(
            state=state,
            total_attempt_count=len(all_attempt_specs),
            finished_attempt_count=finished_attempt_count,
            checkpointed_attempt_count=len(attempts),
            total_job_count=len(jobs),
            completed_job_count=len(records),
            started_monotonic=progress_started_monotonic,
            now_monotonic=now_monotonic,
            now_wall_time=time.time(),
            parallel_workers=worker_count,
            batch_size=effective_batch_size,
            baseline_finished_attempt_count=initial_reused_attempt_count,
        )
        _write_json_atomic(progress_path, snapshot)
        print(_format_progress(snapshot), flush=True)
        last_progress_emit = now_monotonic

    emit_progress(state="running", force=True)
    if not chunks:
        emit_progress(state="completed", force=True)
        return {
            "attempts": attempts,
            "records": records,
            "resume_summary": resume_summary,
            "requested_batch_size": requested_batch_size,
            "effective_batch_size": effective_batch_size,
            "pending_attempt_count": len(attempt_specs),
        }

    with (
        attempts_checkpoint.open("a", encoding="utf-8") as attempts_output,
        records_checkpoint.open("a", encoding="utf-8") as records_output,
    ):
        if parallel_workers <= 1:
            for batch_index, chunk in enumerate(chunks):
                absolute_batch_index = batch_index_offset + batch_index
                batch_attempts = _run_attempt_batch_safe(
                    python_executable=python_executable,
                    asset_catalog_path=asset_catalog_path,
                    attempts=chunk,
                    run_dir=run_dir,
                    torch_extensions_dir=torch_extensions_dir,
                    timeout_seconds=_chunk_timeout(chunk, timeout_seconds, warmup_timeout_seconds),
                    reuse_task_env=reuse_task_env,
                    batch_index=absolute_batch_index,
                    batches_dir=batches_dir,
                    cuda_device=device_slots[0] if device_slots else None,
                )
                checkpoint_completed_batch(
                    batch_index=absolute_batch_index,
                    batch_attempts=batch_attempts,
                    attempts_output=attempts_output,
                    records_output=records_output,
                )
                finished_attempt_count += len(batch_attempts)
                emit_progress(state="running")
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                future_to_batch = {
                    executor.submit(
                        _run_attempt_batch_safe,
                        python_executable=python_executable,
                        asset_catalog_path=asset_catalog_path,
                        attempts=chunk,
                        run_dir=run_dir,
                        torch_extensions_dir=torch_extensions_dir,
                        timeout_seconds=_chunk_timeout(chunk, timeout_seconds, warmup_timeout_seconds),
                        reuse_task_env=reuse_task_env,
                        batch_index=batch_index_offset + batch_index,
                        batches_dir=batches_dir,
                        cuda_device=device_slots[batch_index % len(device_slots)] if device_slots else None,
                    ): batch_index_offset + batch_index
                    for batch_index, chunk in enumerate(chunks)
                }
                for future in as_completed(future_to_batch):
                    absolute_batch_index = future_to_batch[future]
                    batch_attempts = future.result()
                    checkpoint_completed_batch(
                        batch_index=absolute_batch_index,
                        batch_attempts=batch_attempts,
                        attempts_output=attempts_output,
                        records_output=records_output,
                    )
                    finished_attempt_count += len(batch_attempts)
                    emit_progress(state="running")

        final_records: list[dict[str, Any]] = []
        for job in jobs:
            job_id = str(job["job_id"])
            if job_id in emitted_jobs:
                continue
            record = _record_from_attempts(
                job,
                attempts_by_job[job_id],
                thresholds[job["relation"]],
                seeds,
            )
            records.append(record)
            final_records.append(record)
        _append_jsonl_rows(records_output, final_records)
        records_output.flush()

    attempt_order = {
        str(spec["attempt_id"]): index
        for index, spec in enumerate(all_attempt_specs)
    }
    attempts.sort(key=lambda item: attempt_order.get(str(item.get("attempt_id")), len(attempt_order)))
    job_order = {str(job["job_id"]): index for index, job in enumerate(jobs)}
    records.sort(key=lambda item: job_order.get(str(item.get("job_id")), len(job_order)))
    # The record checkpoint is small and derived from attempts.  Canonicalize
    # it once after out-of-order completion without rewriting the potentially
    # multi-gigabyte attempts checkpoint.
    _rewrite_jsonl(records_checkpoint, records)
    emit_progress(state="completed", force=True)
    return {
        "attempts": attempts,
        "records": records,
        "resume_summary": resume_summary,
        "requested_batch_size": requested_batch_size,
        "effective_batch_size": effective_batch_size,
        "pending_attempt_count": len(attempt_specs),
    }


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
        "kind": "terminal_settle_run_progress",
        "state": str(state),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now_wall_time)),
        "estimated_finish_at": estimated_finish_at,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "eta_seconds": None if eta_seconds is None else round(eta_seconds, 3),
        "progress_percent": round(progress_percent, 3),
        "attempts_per_second": round(attempts_per_second, 4),
        "total_attempt_count": total_attempts,
        "finished_attempt_count": finished_attempts,
        "resumed_attempt_count": baseline_finished,
        "newly_finished_attempt_count": newly_finished,
        "checkpointed_attempt_count": max(0, int(checkpointed_attempt_count)),
        "remaining_attempt_count": remaining_attempts,
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


def _build_attempt_specs(
    jobs: list[dict[str, Any]],
    seeds: list[int],
    *,
    start_index: int,
    worker_count: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for job_offset, job in enumerate(jobs, start=start_index):
        probe_job = _probe_job_payload(job)
        for seed in seeds:
            specs.append(
                {
                    "attempt_id": f"{job['job_id']}|seed{seed}",
                    "job_id": job["job_id"],
                    "job_index": job_offset,
                    "pair_key": job["pair_key"],
                    "relation": job["relation"],
                    "seed": int(seed),
                    "warmup": False,
                    "probe_job": probe_job,
                }
            )
    _mark_warmup_specs(specs, worker_count=worker_count, batch_size=batch_size)
    return specs


def _mark_warmup_specs(
    specs: list[dict[str, Any]],
    *,
    worker_count: int,
    batch_size: int,
) -> None:
    for spec in specs:
        spec["warmup"] = False
    warmed_slots: set[int] = set()
    for batch_index, chunk in enumerate(_chunks(specs, max(1, int(batch_size)))):
        slot = batch_index % max(1, int(worker_count))
        if slot in warmed_slots or not chunk:
            continue
        chunk[0]["warmup"] = True
        warmed_slots.add(slot)


def _probe_job_payload(job: dict[str, Any]) -> dict[str, Any]:
    canonical_initial_states = job.get("canonical_initial_states")
    if not isinstance(canonical_initial_states, dict):
        raise ValueError(
            f"terminal-settle job lacks canonical Initial State payload: {job.get('job_id')}"
        )
    for endpoint_name in ("subject", "receiver"):
        endpoint_state = canonical_initial_states.get(endpoint_name)
        if not isinstance(endpoint_state, dict):
            raise ValueError(
                "terminal-settle job lacks canonical Initial State endpoint "
                f"{endpoint_name}: {job.get('job_id')}"
            )
    return {
        "scope": job.get("scope", "pair"),
        "stage": STAGE_TERMINAL,
        "relation": str(job["relation"]),
        "subject": _endpoint_identity(job.get("subject") or {}),
        "receiver": _endpoint_identity(job.get("receiver") or {}),
        "pair_key": str(job["pair_key"]),
        "probe_spec": job.get("probe_spec") or {},
        "geometry_metrics": job.get("geometry_metrics") or {},
        "rubric_review": job.get("rubric_review") or {},
        "source_candidate_id": job.get("source_candidate_id"),
        "yaw_witness": job.get("yaw_witness") or {},
        "placement_recipe": job.get("placement_recipe") or {},
        "recipe_id": job.get("recipe_id"),
        "canonical_initial_states": canonical_initial_states,
        "expected_report_schema": job.get("expected_report_schema", REPORT_SCHEMA_VERSION),
    }


def _run_attempt_batch(
    *,
    python_executable: str,
    asset_catalog_path: Path,
    attempts: list[dict[str, Any]],
    run_dir: Path,
    torch_extensions_dir: Path,
    timeout_seconds: int,
    reuse_task_env: bool,
    batch_index: int,
    batches_dir: Path,
    cuda_device: str | None = None,
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
        "if_benchmark.pair_terminal_settle_worker",
        "--asset-catalog",
        str(asset_catalog_path),
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
    env["TORCH_EXTENSIONS_DIR"] = str(torch_extensions_dir)
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
        result.setdefault("attempt_id", attempt_id)
        result.setdefault("job_id", spec["job_id"])
        result.setdefault("job_index", spec["job_index"])
        result.setdefault("pair_key", spec["pair_key"])
        result.setdefault("relation", spec["relation"])
        result.setdefault("subject", _endpoint_identity((spec.get("probe_job") or {}).get("subject") or {}))
        result.setdefault("receiver", _endpoint_identity((spec.get("probe_job") or {}).get("receiver") or {}))
        result.setdefault("seed", int(spec["seed"]))
        result.setdefault("warmup", bool(spec.get("warmup", False)))
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
    run_dir: Path,
    torch_extensions_dir: Path,
    timeout_seconds: int,
    reuse_task_env: bool,
    batch_index: int,
    batches_dir: Path,
    cuda_device: str | None = None,
) -> list[dict[str, Any]]:
    try:
        return _run_attempt_batch(
            python_executable=python_executable,
            asset_catalog_path=asset_catalog_path,
            attempts=attempts,
            run_dir=run_dir,
            torch_extensions_dir=torch_extensions_dir,
            timeout_seconds=timeout_seconds,
            reuse_task_env=reuse_task_env,
            batch_index=batch_index,
            batches_dir=batches_dir,
            cuda_device=cuda_device,
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
    probe_job = spec.get("probe_job") or {}
    return {
        "attempt_id": spec.get("attempt_id"),
        "job_id": spec.get("job_id"),
        "job_index": spec.get("job_index"),
        "pair_key": spec.get("pair_key"),
        "relation": spec.get("relation"),
        "subject": _endpoint_identity(probe_job.get("subject") or {}),
        "receiver": _endpoint_identity(probe_job.get("receiver") or {}),
        "seed": int(spec.get("seed")),
        "plan_success": False,
        "success": False,
        "failure_reason": reason,
        "duration_sec": duration_sec,
        "warmup": bool(spec.get("warmup", False)),
        "timings": {"total_worker_sec": duration_sec},
    }


def _record_from_attempts(
    job: dict[str, Any],
    attempts: list[dict[str, Any]],
    threshold: float,
    seeds: list[int],
) -> dict[str, Any]:
    attempt_count = len(attempts)
    normalized_reasons = [_normalize_failure_reason(attempt.get("failure_reason")) for attempt in attempts]
    blocking_reasons = [
        reason
        for reason in normalized_reasons
        if reason in RUN_BLOCKING_REASONS or reason in SPEC_BLOCKING_REASONS
    ]
    expected_seeds = {int(seed) for seed in seeds}
    completed_seeds = {int(attempt.get("seed", -1)) for attempt in attempts}
    complete = expected_seeds == completed_seeds and attempt_count == len(seeds)
    valid_attempts = [
        attempt
        for attempt, reason in zip(attempts, normalized_reasons)
        if reason not in RUN_BLOCKING_REASONS and reason not in SPEC_BLOCKING_REASONS
    ]
    plan_success = len(valid_attempts)
    success = sum(1 for attempt in valid_attempts if attempt.get("success"))
    rate = success / len(valid_attempts) if valid_attempts else 0.0
    fail_reasons = Counter(
        _normalize_failure_reason(attempt.get("failure_reason"))
        for attempt in attempts
        if not attempt.get("success")
    )
    fail_reasons.pop("ok", None)
    if not complete or blocking_reasons or len(valid_attempts) != len(seeds):
        outcome = "blocked"
        verdict = blocking_reasons[0] if blocking_reasons else "incomplete_terminal_attempts"
        passed: bool | None = None
    elif rate >= threshold:
        outcome = "pass"
        verdict = "ok"
        passed = True
    else:
        outcome = "fail"
        verdict = _dominant_failure(fail_reasons)
        passed = False
    row = {
        "scope": "pair",
        "stage": STAGE_TERMINAL,
        "relation": str(job["relation"]),
        "subject": _endpoint_identity(job.get("subject") or {}),
        "receiver": _endpoint_identity(job.get("receiver") or {}),
        "pair_key": str(job["pair_key"]),
        "job_id": str(job["job_id"]),
        "band_id": str((job.get("yaw_witness") or {}).get("band_id") or "fixed"),
        "yaw_witness": job.get("yaw_witness") or {},
        "recipe_id": job.get("recipe_id"),
        "placement_recipe": job.get("placement_recipe") or {},
        "outcome": outcome,
        "pass": passed,
        "verdict": verdict,
        "threshold": _clean_float(float(threshold)),
        "seeds": [int(seed) for seed in seeds],
        "attempts": attempt_count,
        "valid_attempts": len(valid_attempts),
        "plan_success": plan_success,
        "success": success,
        "min_rate": _clean_float(rate),
        "fail_reasons": dict(sorted(fail_reasons.items())),
        "initial_state_selection_version": job.get(
            "initial_state_selection_version"
        ),
        "initial_state_selection_hash": job.get("initial_state_selection_hash"),
        "metrics": {
            "rate": _clean_float(rate),
            "attempts": attempt_count,
            "valid_attempts": len(valid_attempts),
            "plan_success": plan_success,
            "success": success,
            "per_seed": _per_seed_attempt_summary(attempts),
            "attempt_failure_reasons": dict(sorted(fail_reasons.items())),
        },
    }
    row["config_hash"] = _row_config_hash(row)
    return row


def _aggregate_pair_records(
    band_records: list[dict[str, Any]],
    jobs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    jobs_by_id = {str(job["job_id"]): job for job in jobs}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in band_records:
        grouped.setdefault(str(record["pair_key"]), []).append(record)
    rows: list[dict[str, Any]] = []
    for pair_key in sorted(grouped):
        bands = sorted(grouped[pair_key], key=lambda item: str(item.get("band_id")))
        first = bands[0]
        passed_bands = [band for band in bands if band.get("outcome") == "pass"]
        blocked_bands = [band for band in bands if band.get("outcome") == "blocked"]
        if passed_bands:
            outcome = "pass"
            passed: bool | None = True
            verdict = "ok"
            selected = passed_bands[0]
        elif blocked_bands:
            outcome = "blocked"
            passed = None
            verdict = str(blocked_bands[0].get("verdict") or "incomplete_terminal_attempts")
            selected = None
        else:
            outcome = "fail"
            passed = False
            verdict = _dominant_failure(
                Counter(
                    reason
                    for band in bands
                    for reason, count in (band.get("fail_reasons") or {}).items()
                    for _ in range(int(count))
                )
            )
            selected = None

        fail_reasons: Counter[str] = Counter()
        for band in bands:
            fail_reasons.update({str(key): int(value) for key, value in (band.get("fail_reasons") or {}).items()})
        first_job = jobs_by_id.get(str(first.get("job_id"))) or {}
        pair_jobs = [
            jobs_by_id[str(band.get("job_id"))]
            for band in bands
            if str(band.get("job_id")) in jobs_by_id
        ]
        relative_yaw = dict((first_job.get("geometry_metrics") or {}).get("relative_yaw") or {})
        source_bands_by_id: dict[str, dict[str, Any]] = {}
        for job in pair_jobs:
            job_relative_yaw = (job.get("geometry_metrics") or {}).get("relative_yaw") or {}
            for source_band in job_relative_yaw.get("geometric_bands") or []:
                source_bands_by_id[str(source_band.get("band_id"))] = source_band
            if not relative_yaw and job_relative_yaw:
                relative_yaw = dict(job_relative_yaw)
        source_bands = [source_bands_by_id[key] for key in sorted(source_bands_by_id)]
        if not source_bands and relative_yaw.get("fallback_witness"):
            source_bands = [relative_yaw["fallback_witness"]]
        status_by_band: dict[str, str] = {}
        for band in bands:
            status = (
                "validated_terminal"
                if band.get("outcome") == "pass"
                else "probe_run_blocked"
                if band.get("outcome") == "blocked"
                else "rejected"
            )
            covered_band_ids = list((band.get("yaw_witness") or {}).get("covered_band_ids") or [])
            for band_id in covered_band_ids or [str(band.get("band_id"))]:
                status_by_band[str(band_id)] = status
        relative_yaw["geometric_bands"] = [
            {
                **band,
                "witness_status": status_by_band.get(str(band.get("band_id")), band.get("witness_status")),
                "sampling_eligible": bool(
                    band.get("sampling_eligible")
                    and status_by_band.get(str(band.get("band_id"))) == "validated_terminal"
                ),
            }
            for band in source_bands
        ]
        row = {
            "scope": "pair",
            "stage": STAGE_TERMINAL,
            "relation": first["relation"],
            "subject": first["subject"],
            "receiver": first["receiver"],
            "pair_key": pair_key,
            "outcome": outcome,
            "pass": passed,
            "verdict": verdict,
            "threshold": first["threshold"],
            "seeds": first["seeds"],
            "attempts": sum(int(band.get("attempts") or 0) for band in bands),
            "valid_attempts": sum(int(band.get("valid_attempts") or 0) for band in bands),
            "plan_success": sum(int(band.get("plan_success") or 0) for band in bands),
            "success": sum(int(band.get("success") or 0) for band in bands),
            "min_rate": max((float(band.get("min_rate") or 0.0) for band in bands), default=0.0),
            "fail_reasons": dict(sorted(fail_reasons.items())),
            "initial_state_selection_version": first.get(
                "initial_state_selection_version"
            ),
            "initial_state_selection_hash": first.get(
                "initial_state_selection_hash"
            ),
            "validated_recipe_id": selected.get("recipe_id") if selected else None,
            "validated_placement_recipe": (
                selected.get("placement_recipe") if selected else None
            ),
            "validated_band_ids": sorted(
                {
                    str(band_id)
                    for band in passed_bands
                    for band_id in (
                        list((band.get("yaw_witness") or {}).get("covered_band_ids") or [])
                        or [str(band.get("band_id"))]
                    )
                }
            ),
            "canonical_pose_policy_hash": ((first_job.get("probe_spec") or {}).get("canonical_pose_policy_hash")),
            "yaw_solver_policy_hash": ((first_job.get("probe_spec") or {}).get("yaw_solver_policy_hash")),
            "relative_yaw": relative_yaw,
            "metrics": {
                "band_results": bands,
                "relative_yaw": relative_yaw,
            },
        }
        row["config_hash"] = _row_config_hash(row)
        rows.append(row)
    return rows


def _per_seed_attempt_summary(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for attempt in sorted(attempts, key=lambda item: int(item.get("seed", 0))):
        rows[str(int(attempt.get("seed", 0)))] = {
            "success": bool(attempt.get("success")),
            "plan_success": bool(attempt.get("plan_success")),
            "failure_reason": _normalize_failure_reason(attempt.get("failure_reason")),
            "warmup": bool(attempt.get("warmup", False)),
            "duration_sec": _clean_float(float(attempt.get("duration_sec") or 0.0)),
        }
    return rows


def _summary(records: list[dict[str, Any]], attempts: list[dict[str, Any]]) -> dict[str, Any]:
    by_relation = Counter(record["relation"] for record in records)
    validated = sum(1 for record in records if record.get("outcome") == "pass")
    blocked = sum(1 for record in records if record.get("outcome") == "blocked")
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
        "rejected_count": len(records) - validated - blocked,
        "blocked_count": blocked,
        "record_count_by_relation": {relation: by_relation.get(relation, 0) for relation in RELATIONS},
        "attempt_failure_reasons": dict(sorted(failure_reasons.items())),
    }


def _write_timing_summary(
    path: Path,
    *,
    attempts: list[dict[str, Any]],
    records: list[dict[str, Any]],
    batch_size: int,
    reuse_task_env: bool,
    parallel_workers: int,
    cuda_devices: list[str] | None,
    source_kind: str,
    requested_batch_size: int | None = None,
) -> dict[str, Any]:
    requested_batch = (
        int(batch_size)
        if requested_batch_size is None
        else int(requested_batch_size)
    )
    timing_summary = {
        "schema_version": 1,
        "source_kind": source_kind,
        "attempt_count": len(attempts),
        "record_count": len(records),
        "batch_size": int(batch_size),
        "requested_batch_size": requested_batch,
        "effective_batch_size": int(batch_size),
        "reuse_task_env": bool(reuse_task_env),
        "parallel_workers": int(parallel_workers),
        "cuda_devices": list(cuda_devices or []),
        "timing_keys": _timing_key_stats(attempts),
        "by_relation": _timing_stats_by_relation(attempts),
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


def _timing_stats_by_relation(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    by_relation: dict[str, list[dict[str, Any]]] = {}
    for attempt in attempts:
        by_relation.setdefault(str(attempt.get("relation")), []).append(attempt)
    return {
        relation: {
            "attempt_count": len(relation_attempts),
            "timing_keys": _timing_key_stats(relation_attempts),
        }
        for relation, relation_attempts in sorted(by_relation.items())
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
                "pair_key": attempt.get("pair_key"),
                "relation": attempt.get("relation"),
                "seed": attempt.get("seed"),
                "success": bool(attempt.get("success")),
                "failure_reason": attempt.get("failure_reason"),
                "duration_sec": _clean_float(duration(attempt)),
            }
        )
    return rows


def _chunk_timeout(attempts: list[dict[str, Any]], timeout_seconds: int, warmup_timeout_seconds: int) -> int:
    total = 0
    for attempt in attempts:
        total += warmup_timeout_seconds if attempt.get("warmup") else timeout_seconds
    return max(timeout_seconds, total)


def _chunks(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [items[index:index + size] for index in range(0, len(items), size)]


def _effective_batch_size(
    *,
    pending_attempt_count: int,
    worker_count: int,
    requested_batch_size: int,
) -> int:
    """Balance a small run without creating a second cold-start wave."""

    requested = max(1, int(requested_batch_size))
    pending = max(0, int(pending_attempt_count))
    workers = max(1, int(worker_count))
    if pending == 0:
        return requested
    attempts_per_worker = max(1, (pending + workers - 1) // workers)
    return min(requested, attempts_per_worker)


def _append_jsonl_rows(output: Any, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        json.dump(row, output, sort_keys=True, ensure_ascii=False)
        output.write("\n")


def _durable_flush(output: Any) -> None:
    """Make a completed batch durable before removing its duplicate payload."""

    output.flush()
    os.fsync(output.fileno())


def _update_run_batch_metadata(
    path: Path,
    *,
    requested_batch_size: int,
    effective_batch_size: int,
    pending_attempt_count: int,
) -> None:
    if not path.is_file():
        return
    metadata = _load_json_required(path)
    metadata["batch_size"] = int(effective_batch_size)
    metadata["requested_batch_size"] = int(requested_batch_size)
    metadata["effective_batch_size"] = int(effective_batch_size)
    metadata["pending_attempt_count"] = int(pending_attempt_count)
    _write_json_atomic(path, metadata)


def _repair_jsonl_append_boundary(path: Path) -> None:
    """Drop only an invalid crash-truncated tail before appending resumed rows."""

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
        block_size = 64 * 1024
        while cursor > 0:
            start = max(0, cursor - block_size)
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
    batches_dir: Path,
    batch_index: int,
    attempts: list[dict[str, Any]],
) -> None:
    """Remove duplicate successful worker payloads after durable checkpointing.

    Semantic terminal failures are complete probe results and remain fully
    represented in ``attempts.jsonl`` and the per-batch logs.  Infrastructure,
    source, and spec blockers retain their input/output payloads for diagnosis.
    """

    if not attempts:
        return
    diagnostic_reasons = RUN_BLOCKING_REASONS | SPEC_BLOCKING_REASONS
    if any(
        attempt.get("worker_returncode") != 0
        or _normalize_failure_reason(attempt.get("failure_reason")) in diagnostic_reasons
        for attempt in attempts
    ):
        return
    for suffix in (".json", ".results.jsonl"):
        path = batches_dir / f"batch_{int(batch_index):06d}{suffix}"
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            # Payload cleanup is an optimization.  A cleanup failure must not
            # invalidate the already durable semantic evidence.
            pass


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
    indexes: list[int] = []
    # Successful batch payloads are intentionally discarded after their rows
    # are durable, while logs remain as permanent diagnostics.  Inspect both
    # trees so resume never reuses an old log name and silently retargets the
    # log paths frozen in prior attempt rows.
    for directory, patterns in (
        (run_dir / "batches", ("batch_*.json", "batch_*.results.jsonl")),
        (run_dir / "logs", ("batch_*.stdout.log", "batch_*.stderr.log")),
    ):
        for pattern in patterns:
            for path in directory.glob(pattern):
                index_text = path.name.removeprefix("batch_").split(".", 1)[0]
                try:
                    indexes.append(int(index_text))
                except ValueError:
                    continue
    return max(indexes, default=-1) + 1


def _archive_existing_run_artifacts(run_dir: Path) -> Path | None:
    """Preserve the previous checkpoints/logs before starting a fresh run."""

    material_names = ("checkpoints", "logs", "batches", "timing_summary.yml", "progress.json")
    material = [run_dir / name for name in material_names if (run_dir / name).exists()]
    if not material:
        return None
    history_root = run_dir / "history"
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    archive_dir = history_root / timestamp
    suffix = 1
    while archive_dir.exists():
        archive_dir = history_root / f"{timestamp}_{suffix}"
        suffix += 1
    archive_dir.mkdir(parents=True, exist_ok=False)
    for path in material:
        shutil.move(str(path), str(archive_dir / path.name))
    return archive_dir


def _worker_device_slots(cuda_devices: list[str] | None, parallel_workers: int) -> list[str | None]:
    worker_count = max(1, int(parallel_workers))
    if not cuda_devices:
        return [None for _ in range(worker_count)]
    devices = [str(device).strip() for device in cuda_devices if str(device).strip()]
    if not devices:
        return [None for _ in range(worker_count)]
    return [devices[index % len(devices)] for index in range(worker_count)]


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


def _load_resume_attempts(
    checkpoint_path: Path,
    expected_specs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], set[str]]:
    expected_by_id = {str(spec["attempt_id"]): spec for spec in expected_specs}
    rows = _read_batch_results(checkpoint_path)
    unknown_ids = sorted(
        {
            str(row.get("attempt_id"))
            for row in rows
            if str(row.get("attempt_id")) not in expected_by_id
        }
    )
    if unknown_ids:
        preview = ", ".join(unknown_ids[:3])
        raise ValueError(
            f"resume checkpoint contains {len(unknown_ids)} attempts outside the selected plan: {preview}"
        )

    selected_by_id: dict[str, dict[str, Any]] = {}
    retry_attempt_ids: set[str] = set()
    for row in rows:
        attempt_id = str(row["attempt_id"])
        spec = expected_by_id[attempt_id]
        if str(row.get("job_id")) != str(spec["job_id"]) or int(row.get("seed", -1)) != int(spec["seed"]):
            raise ValueError(f"resume checkpoint identity mismatch for {attempt_id}")
        reason = _normalize_failure_reason(row.get("failure_reason"))
        if reason in RUN_BLOCKING_REASONS:
            retry_attempt_ids.add(attempt_id)
            continue
        selected_by_id[attempt_id] = row
        retry_attempt_ids.discard(attempt_id)

    retry_attempt_ids.difference_update(selected_by_id)
    attempts = [
        selected_by_id[str(spec["attempt_id"])]
        for spec in expected_specs
        if str(spec["attempt_id"]) in selected_by_id
    ]
    return attempts, retry_attempt_ids


def _normalize_failure_reason(reason: Any) -> str:
    raw = str(reason or "unknown_failure").strip().lower().replace(" ", "_")
    allowed = {
        "ok",
        "worker_import_failed",
        "worker_exception",
        "worker_process_failed",
        "worker_timeout",
        "worker_no_json_result",
        "missing_pair_metadata",
        "actor_load_failed",
        "receiver_unstable",
        "unsupported_relation",
        "checker_false",
        "stable",
        "subject_receiver_contact",
        "subject_not_table_contact",
        # Current SuccessContract predicates may be emitted directly by
        # checkpoints produced before the worker's semantic-reason adapter.
        # They are terminal-physics failures, not worker infrastructure
        # failures, and therefore remain in the statistical denominator.
        "relation_entities_stable",
        "subject_meaningfully_inside_receiver",
        "xy_aligned",
        "height_gap_ok",
        "xy_inside_receiver",
        "subject_center_below_receiver_top",
        "subject_center_above_receiver_top",
        "subject_not_meaningfully_inside_receiver",
        "xy_midpoint_aligned",
        "mug_functional_point_high",
        "stamp_contact_ids",
        "all_proxy_contents_inside_receiver_xy",
        "all_proxy_contents_above_table",
        "terminal_unstable",
        "missing_subject_receiver_contact",
        "subject_resting_on_table",
        "outside_receiver_cavity",
        "subject_inside_receiver",
        "terminal_misaligned",
        "hang_height_too_low",
        "invalid_stamp_contact_face",
        "proxy_contents_outside_receiver",
        "proxy_contents_on_table",
    }
    if raw in allowed:
        return raw
    if "timeout" in raw:
        return "worker_timeout"
    if "load" in raw or "mesh" in raw or "instantiate" in raw:
        return "actor_load_failed"
    if "unstable" in raw:
        return "receiver_unstable"
    return "worker_exception"


def _dominant_failure(fail_reasons: Counter[str]) -> str:
    if not fail_reasons:
        return "checker_false"
    reason, _ = fail_reasons.most_common(1)[0]
    return reason


def _endpoint_identity(endpoint: dict[str, Any]) -> dict[str, Any]:
    row = {
        "asset": str(endpoint.get("asset")),
        "model_id": str(endpoint.get("model_id")),
    }
    if endpoint.get("category") is not None:
        row["category"] = endpoint.get("category")
    if endpoint.get("inventory_id") is not None:
        row["inventory_id"] = endpoint.get("inventory_id")
    return row


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


def _load_yaml_required(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=yaml.CSafeLoader)
    if not isinstance(data, dict):
        raise ValueError(f"expected YAML mapping in {path}")
    return data


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, sort_keys=True, ensure_ascii=False, indent=2)
        f.write("\n")


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, sort_keys=True, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(temporary_path, path)


def _rewrite_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary_path.open("w", encoding="utf-8") as f:
        for row in rows:
            json.dump(row, f, sort_keys=True, ensure_ascii=False)
            f.write("\n")
    os.replace(temporary_path, path)


def _load_json_required(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON mapping in {path}")
    return data


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_seed_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_cuda_devices(value: str | None) -> list[str] | None:
    if value is None:
        return None
    devices = [part.strip() for part in value.split(",") if part.strip()]
    return devices or None


def parse_thresholds(values: list[str] | None) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for value in values or []:
        relation, raw_threshold = value.split("=", 1)
        relation = relation.strip()
        if relation not in RELATIONS:
            raise argparse.ArgumentTypeError(f"unsupported relation threshold: {relation}")
        thresholds[relation] = float(raw_threshold)
    return thresholds
