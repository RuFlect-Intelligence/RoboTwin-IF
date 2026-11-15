"""State-conditioned pair planning for Milestone 5 articulated receivers.

This module is intentionally not imported by the rigid M5 runner.  It can be
developed and tested while a rigid sweep is running without changing the code
loaded by that sweep.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import yaml

from .asset_probe_pose import resolve_canonical_pose
from .pair_recipe import (
    ARTICULATED_TERMINAL_SETTLE_POLICY_VERSION,
    canonical_articulated_terminal_placement_recipe,
    receiver_state_profile_hash as _shared_receiver_state_profile_hash,
)
from .pair_yaw_geometry import pose_aware_box_geometry, solve_relative_yaw_domain


SCHEMA_VERSION = 1
PLAN_SCHEMA_VERSION = 1
M5_AR_POLICY_VERSION = ARTICULATED_TERMINAL_SETTLE_POLICY_VERSION
SUPPORTED_RELATIONS = {"place_in"}
PROFILE_READY_STATUS = "reviewed"
DEFAULT_EDGE_CLEARANCE_M = 0.004
RUN_BLOCKING_REASONS = {
    "worker_import_failed",
    "worker_exception",
    "worker_process_failed",
    "worker_timeout",
    "worker_no_json_result",
    "actor_load_failed",
}
PROGRESS_EMIT_INTERVAL_SECONDS = 5.0


def articulated_policy_hash() -> str:
    payload = {
        "version": M5_AR_POLICY_VERSION,
        "supported_relations": sorted(SUPPORTED_RELATIONS),
        "profile_ready_status": PROFILE_READY_STATUS,
        "default_edge_clearance_m": DEFAULT_EDGE_CLEARANCE_M,
        "pair_identity": "relation|subject|receiver|receiver_state_profile_id",
        "geometry_frame": "reviewed_cavity_link",
        "terminal_evidence": "pointwise_band_witness",
    }
    return _hash_payload(payload)


def load_receiver_profiles(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=yaml.CSafeLoader)
    if not isinstance(data, dict):
        raise ValueError(f"expected YAML mapping in {path}")
    if int(data.get("schema_version") or 0) != SCHEMA_VERSION:
        raise ValueError(f"unsupported articulated receiver profile schema in {path}")
    profiles = data.get("profiles")
    if not isinstance(profiles, dict):
        raise ValueError(f"articulated receiver profiles must be a mapping in {path}")
    if not data.get("catalog_hash"):
        data["catalog_hash"] = _hash_payload({key: value for key, value in data.items() if key != "catalog_hash"})
    return data


def validate_receiver_profile(profile_id: str, profile: dict[str, Any]) -> list[str]:
    """Return deterministic blockers; an empty list means probe-ready."""

    blockers: list[str] = []
    receiver = profile.get("receiver") or {}
    if not receiver.get("asset"):
        blockers.append("missing_receiver_asset")
    if receiver.get("model_id") is None:
        blockers.append("missing_receiver_model_id")
    relation = str(profile.get("relation") or "")
    if relation not in SUPPORTED_RELATIONS:
        blockers.append("unsupported_relation")
    if str(profile.get("review_status") or "") != PROFILE_READY_STATUS:
        blockers.append("receiver_profile_not_reviewed")

    fixture = profile.get("fixture") or {}
    if not _vector(fixture.get("root_q0_wxyz"), 4):
        blockers.append("missing_fixture_root_q0")
    if not _vector(fixture.get("table_xy"), 2):
        blockers.append("missing_fixture_table_xy")

    state = profile.get("state") or {}
    joint_targets = state.get("joint_targets")
    if not isinstance(joint_targets, list) or not joint_targets:
        blockers.append("missing_reviewed_joint_targets")
    else:
        for index, target in enumerate(joint_targets):
            if not isinstance(target, dict):
                blockers.append(f"invalid_joint_target_{index}")
                continue
            if target.get("joint_index") is None:
                blockers.append(f"missing_joint_index_{index}")
            ratio = _float_or_none(target.get("target_ratio"))
            if ratio is None or ratio < 0.0 or ratio > 1.0:
                blockers.append(f"invalid_target_ratio_{index}")
            tolerance = _float_or_none(target.get("tolerance_ratio"))
            if tolerance is None or tolerance <= 0.0:
                blockers.append(f"invalid_joint_tolerance_{index}")

    cavity = profile.get("cavity") or {}
    if not cavity.get("frame_link"):
        blockers.append("missing_cavity_frame_link")
    if not cavity.get("yaw_reference_link"):
        blockers.append("missing_yaw_reference_link")
    frame_transform = cavity.get("frame_transform")
    if frame_transform is not None and not (
        isinstance(frame_transform, list)
        and len(frame_transform) == 4
        and all(_vector(row, 4) for row in frame_transform)
    ):
        blockers.append("invalid_cavity_frame_transform")
    region = cavity.get("usable_region") or {}
    if not _vector(region.get("center_xyz"), 3):
        blockers.append("missing_reviewed_usable_region_center")
    half_extents = _vector(region.get("half_extents_xyz"), 3)
    if not half_extents or min(half_extents) <= 0.0:
        blockers.append("missing_reviewed_usable_region_half_extents")
    containment = cavity.get("containment_region")
    if containment is not None:
        if not isinstance(containment, dict) or not _vector(
            containment.get("center_xyz"), 3
        ):
            blockers.append("invalid_reviewed_containment_region_center")
        containment_half_extents = (
            _vector(containment.get("half_extents_xyz"), 3)
            if isinstance(containment, dict)
            else None
        )
        if (
            not containment_half_extents
            or min(containment_half_extents) <= 0.0
        ):
            blockers.append(
                "invalid_reviewed_containment_region_half_extents"
            )
    if _float_or_none(cavity.get("support_plane_z")) is None:
        blockers.append("missing_reviewed_support_plane")
    opening = cavity.get("opening_plane") or {}
    if not _vector(opening.get("point_xyz"), 3) or not _nonzero_vector(opening.get("outward_normal_xyz")):
        blockers.append("missing_reviewed_opening_plane")
    allowed = cavity.get("allowed_support_links")
    if not isinstance(allowed, list) or not allowed:
        blockers.append("missing_allowed_support_links")
    forbidden = cavity.get("forbidden_support_links")
    if not isinstance(forbidden, list) or not forbidden:
        blockers.append("missing_forbidden_support_links")
    if profile.get("allow_generic_place_on") is not False:
        blockers.append("generic_place_on_must_be_explicitly_disabled")
    skill = profile.get("skill") or {}
    if skill.get("entry_mode") not in {"top_down", "front_insertion"}:
        blockers.append("missing_or_invalid_skill_entry_mode")
    for key in ("approach_distance_m", "retreat_distance_m"):
        distance = _float_or_none(skill.get(key))
        if distance is None or distance <= 0.0:
            blockers.append(f"missing_or_invalid_skill_{key}")

    return _deduplicate(blockers)


def resolve_joint_targets(
    profile: dict[str, Any],
    qlimits: Iterable[Iterable[float]],
    current_qpos: Iterable[float],
) -> tuple[list[float], list[dict[str, Any]]]:
    """Resolve reviewed ratios against runtime joint limits."""

    limits = [[float(value) for value in row] for row in qlimits]
    qpos = [float(value) for value in current_qpos]
    resolved: list[dict[str, Any]] = []
    for target in (profile.get("state") or {}).get("joint_targets") or []:
        joint_index = int(target["joint_index"])
        if joint_index < 0 or joint_index >= len(limits) or joint_index >= len(qpos):
            raise ValueError(f"joint_index_out_of_range:{joint_index}")
        if len(limits[joint_index]) < 2:
            raise ValueError(f"invalid_joint_limits:{joint_index}")
        lower, upper = limits[joint_index][:2]
        span = upper - lower
        if not span > 0.0:
            raise ValueError(f"invalid_joint_limits:{joint_index}")
        ratio = float(target["target_ratio"])
        tolerance_ratio = float(target["tolerance_ratio"])
        value = lower + span * ratio
        qpos[joint_index] = value
        resolved.append(
            {
                "joint_index": joint_index,
                "joint_name": target.get("joint_name"),
                "target_ratio": ratio,
                "target_qpos": value,
                "tolerance_ratio": tolerance_ratio,
                "tolerance_qpos": span * tolerance_ratio,
                "lower": lower,
                "upper": upper,
            }
        )
    return qpos, resolved


def state_conditioned_pair_key(
    relation: str,
    subject: dict[str, Any],
    receiver: dict[str, Any],
    receiver_state_profile_id: str,
) -> str:
    return (
        f"{relation}|{subject['asset']}:{subject['model_id']}|"
        f"{receiver['asset']}:{receiver['model_id']}|state:{receiver_state_profile_id}"
    )


def build_articulated_probe_plan(
    asset_catalog: dict[str, Any],
    profile_catalog: dict[str, Any],
    *,
    subject_assets: set[str] | None = None,
    receiver_profile_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Build an isolated M5-AR plan from reviewed receiver-state profiles."""

    subjects = list(_iter_rigid_pick_subjects(asset_catalog, subject_assets))
    profiles = profile_catalog.get("profiles") or {}
    jobs: list[dict[str, Any]] = []
    blocked_profiles: list[dict[str, Any]] = []
    pruned_pairs: list[dict[str, Any]] = []
    pair_count = 0
    for profile_id in sorted(profiles):
        if receiver_profile_ids is not None and profile_id not in receiver_profile_ids:
            continue
        profile = profiles[profile_id]
        blockers = validate_receiver_profile(profile_id, profile)
        if blockers:
            blocked_profiles.append(
                {
                    "receiver_state_profile_id": profile_id,
                    "receiver": profile.get("receiver") or {},
                    "status": "probe_spec_blocked",
                    "blocked_reasons": blockers,
                }
            )
            continue
        profile_hash = receiver_state_profile_hash(profile_id, profile)
        receiver = _endpoint(profile["receiver"])
        cavity = profile["cavity"]
        half_extents = [float(value) for value in cavity["usable_region"]["half_extents_xyz"]]
        receiver_geometry = _box_geometry(2.0 * half_extents[0], 2.0 * half_extents[1])
        for subject, subject_candidate in subjects:
            canonical_pose = resolve_canonical_pose(
                subject_candidate,
                "pick_subject",
                relation="place_in",
                role="subject",
            )
            if not canonical_pose.get("resolved"):
                continue
            subject_geometry = pose_aware_box_geometry(subject, canonical_pose)
            if subject_geometry is None:
                continue
            yaw_domain = solve_relative_yaw_domain(
                "place_in",
                subject_geometry,
                receiver_geometry,
                usable_factor=1.0,
                edge_clearance_m=float(profile.get("edge_clearance_m") or DEFAULT_EDGE_CLEARANCE_M),
            )
            _ground_yaw_domain(yaw_domain, profile_id)
            pair_key = state_conditioned_pair_key("place_in", subject, receiver, profile_id)
            pair_count += 1
            if yaw_domain.get("outer_impossible"):
                pruned_pairs.append(
                    {
                        "pair_key": pair_key,
                        "relation": "place_in",
                        "receiver_state_profile_id": profile_id,
                        "status": "pruned",
                        "evidence_class": "heuristic",
                        "reason": "canonical_footprint_exceeds_reviewed_cavity",
                        "outer_best_margin_m": yaw_domain.get("outer_best_margin_m"),
                    }
                )
                continue
            bands = list(yaw_domain.get("geometric_bands") or [])
            fallback = yaw_domain.get("fallback_witness")
            if not bands and isinstance(fallback, dict):
                bands = [fallback]
            for band in bands:
                jobs.append(
                    _probe_job(
                        pair_key=pair_key,
                        subject=subject,
                        receiver=receiver,
                        canonical_pose=canonical_pose,
                        profile_id=profile_id,
                        profile=profile,
                        profile_hash=profile_hash,
                        yaw_domain=yaw_domain,
                        band=band,
                    )
                )

    plan = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "kind": "m5_articulated_receiver_probe_plan",
        "scope": "state_conditioned_pair",
        "stage": "articulated_terminal_settle",
        "source_asset_catalog_hash": asset_catalog.get("catalog_hash"),
        "source_receiver_profile_catalog_hash": profile_catalog.get("catalog_hash"),
        "articulated_policy_hash": articulated_policy_hash(),
        "summary": {
            "subject_count": len(subjects),
            "reviewed_profile_count": len(
                [profile_id for profile_id in profiles if not validate_receiver_profile(profile_id, profiles[profile_id])]
            ),
            "blocked_profile_count": len(blocked_profiles),
            "state_conditioned_pair_count": pair_count,
            "pruned_pair_count": len(pruned_pairs),
            "probe_job_count": len(jobs),
        },
        "blocked_profiles": blocked_profiles,
        "pruned_pairs": pruned_pairs,
        "jobs": jobs,
    }
    plan["plan_hash"] = _hash_payload(plan)
    return plan


def aggregate_band_attempts(
    job: dict[str, Any],
    attempts: list[dict[str, Any]],
    *,
    seeds: list[int],
    threshold: float = 0.6,
) -> dict[str, Any]:
    by_seed = {int(attempt["seed"]): attempt for attempt in attempts if attempt.get("seed") is not None}
    ordered = [by_seed[seed] for seed in seeds if seed in by_seed]
    blocked = [
        attempt
        for attempt in ordered
        if attempt.get("outcome") == "blocked" or attempt.get("failure_reason") in RUN_BLOCKING_REASONS
    ]
    valid = [attempt for attempt in ordered if attempt not in blocked]
    success = sum(bool(attempt.get("success")) for attempt in valid)
    complete = len(ordered) == len(seeds)
    if blocked or not complete:
        outcome = "blocked"
        witness_status = "probe_run_blocked"
        passed: bool | None = None
    else:
        passed = bool(valid) and success / len(valid) >= float(threshold)
        outcome = "pass" if passed else "fail"
        witness_status = "validated_terminal" if passed else "rejected"
    return {
        "job_id": job["job_id"],
        "pair_key": job["pair_key"],
        "relation": job["relation"],
        "receiver_state_profile_id": job["receiver_state_profile_id"],
        "band_id": job["band_id"],
        "outcome": outcome,
        "pass": passed,
        "witness_status": witness_status,
        "seeds": list(seeds),
        "attempts": len(ordered),
        "valid_attempts": len(valid),
        "success": success,
        "threshold": float(threshold),
    }


def aggregate_pair_bands(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot aggregate an empty articulated pair")
    passed = [record for record in records if record.get("outcome") == "pass"]
    blocked = [record for record in records if record.get("outcome") == "blocked"]
    if passed:
        status = "validated_terminal"
    elif blocked:
        status = "probe_run_blocked"
    else:
        status = "rejected"
    return {
        "pair_key": records[0]["pair_key"],
        "relation": records[0]["relation"],
        "receiver_state_profile_id": records[0]["receiver_state_profile_id"],
        "status": status,
        "validated_band_ids": [record["band_id"] for record in passed],
        "band_records": records,
    }


def write_articulated_plan(path: Path, plan: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(plan, f, sort_keys=False, allow_unicode=True, width=120)


def run_articulated_probe_plan(
    plan: dict[str, Any],
    *,
    reports_dir: Path,
    run_dir: Path,
    rubric_hash: str,
    seeds: list[int] | None = None,
    threshold: float = 0.6,
    resume: bool = False,
) -> dict[str, Any]:
    """Run the isolated M5-AR Terminal plan and emit a common pair report."""

    from .articulated_terminal_worker import run_articulated_attempt
    probe_seeds = [int(seed) for seed in (seeds or [0, 1, 2])]
    run_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    attempts_path = run_dir / "attempts.jsonl"
    prior: dict[str, dict[str, Any]] = {}
    if resume and attempts_path.is_file():
        for row in _read_jsonl(attempts_path):
            if row.get("outcome") != "blocked" and row.get("failure_reason") not in RUN_BLOCKING_REASONS:
                prior[str(row["attempt_id"])] = row
    attempts_by_id = dict(prior)
    if not resume:
        _write_jsonl(attempts_path, [])
    for job_index, job in enumerate(plan.get("jobs") or []):
        for seed in probe_seeds:
            attempt_id = _hash_payload({"job_id": job["job_id"], "seed": seed})[:24]
            if attempt_id in prior:
                continue
            result = run_articulated_attempt(job, seed=seed)
            result.update(
                {
                    "attempt_id": attempt_id,
                    "job_id": job["job_id"],
                    "job_index": job_index,
                    "pair_key": job["pair_key"],
                    "seed": seed,
                }
            )
            attempts_by_id[attempt_id] = result
            _append_jsonl(attempts_path, result)
    attempts = sorted(
        attempts_by_id.values(),
        key=lambda item: (
            int(item.get("job_index") or 0),
            int(item.get("seed") or 0),
        ),
    )
    _write_jsonl(attempts_path, attempts)
    return _write_articulated_probe_report(
        plan,
        attempts=attempts,
        reports_dir=reports_dir,
        attempts_path=attempts_path,
        rubric_hash=rubric_hash,
        seeds=probe_seeds,
        threshold=threshold,
    )


def run_articulated_probe_plan_parallel(
    plan: dict[str, Any],
    *,
    reports_dir: Path,
    run_dir: Path,
    rubric_hash: str,
    python_executable: str | None = None,
    seeds: list[int] | None = None,
    threshold: float = 0.6,
    resume: bool = False,
    cuda_devices: list[str] | None = None,
    parallel_workers: int = 1,
    batch_size: int = 1,
    timeout_seconds: int = 180,
) -> dict[str, Any]:
    """Run independent M5-AR attempts concurrently with resumable progress."""

    probe_seeds = [int(seed) for seed in (seeds or [0, 1, 2])]
    jobs = list(plan.get("jobs") or [])
    devices = list(cuda_devices or ["0"])
    worker_count = max(1, int(parallel_workers))
    effective_batch_size = max(1, int(batch_size))
    run_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    jobs_dir = run_dir / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    batches_dir = run_dir / "batches"
    batches_dir.mkdir(parents=True, exist_ok=True)
    attempts_path = run_dir / "attempts.jsonl"
    progress_path = run_dir / "progress.json"
    metadata_path = run_dir / "run_metadata.json"
    identity = {
        "source_probe_plan_hash": plan.get("plan_hash"),
        "selected_job_ids_hash": _hash_payload([str(job["job_id"]) for job in jobs]),
        "seeds": probe_seeds,
        "threshold": float(threshold),
    }
    prior_metadata = _load_json(metadata_path) if resume else {}
    mismatches = [
        key for key, value in identity.items()
        if prior_metadata and prior_metadata.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            "incompatible articulated resume checkpoint: "
            + ", ".join(sorted(mismatches))
        )
    started_at = (
        prior_metadata.get("started_at")
        or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    metadata = {
        **identity,
        "started_at": started_at,
        "resumed_at": (
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            if resume else None
        ),
        "parallel_workers": worker_count,
        "batch_size": effective_batch_size,
        "cuda_devices": devices,
        "timeout_seconds": int(timeout_seconds),
        "python_executable": python_executable or sys.executable,
    }
    _write_json(metadata_path, metadata)
    if not resume:
        _write_jsonl(attempts_path, [])

    attempts_by_id: dict[str, dict[str, Any]] = {}
    if resume and attempts_path.is_file():
        for row in _read_jsonl(attempts_path):
            attempts_by_id[str(row["attempt_id"])] = row
    specs: list[dict[str, Any]] = []
    for job_index, job in enumerate(jobs):
        job_path = jobs_dir / f"{_hash_payload(job)[:20]}.json"
        _write_json(job_path, job)
        for seed in probe_seeds:
            attempt_id = _hash_payload(
                {"job_id": job["job_id"], "seed": seed}
            )[:24]
            prior = attempts_by_id.get(attempt_id)
            if (
                prior is not None
                and prior.get("outcome") != "blocked"
                and prior.get("failure_reason") not in RUN_BLOCKING_REASONS
            ):
                continue
            specs.append(
                {
                    "attempt_id": attempt_id,
                    "job": job,
                    "job_index": job_index,
                    "job_path": job_path,
                    "seed": seed,
                }
            )

    total_attempts = len(jobs) * len(probe_seeds)
    started_monotonic = time.monotonic()
    pending_attempt_ids = {str(spec["attempt_id"]) for spec in specs}
    completed_attempts_by_job: dict[str, int] = {}
    for attempt_id, row in attempts_by_id.items():
        if attempt_id in pending_attempt_ids:
            continue
        job_id = str(row.get("job_id"))
        completed_attempts_by_job[job_id] = completed_attempts_by_job.get(job_id, 0) + 1
    completed_job_count = sum(
        count >= len(probe_seeds)
        for count in completed_attempts_by_job.values()
    )
    last_progress_emit = 0.0

    def emit_progress(state: str, newly_finished: int) -> None:
        nonlocal last_progress_emit
        elapsed = max(0.0, time.monotonic() - started_monotonic)
        finished = total_attempts - len(pending_attempt_ids)
        rate = newly_finished / elapsed if elapsed > 0.0 else 0.0
        remaining = max(0, total_attempts - finished)
        eta = 0.0 if remaining == 0 else (remaining / rate if rate > 0.0 else None)
        payload = {
            "schema_version": 1,
            "kind": "m5_articulated_terminal_run_progress",
            "state": state,
            "total_attempt_count": total_attempts,
            "finished_attempt_count": finished,
            "remaining_attempt_count": remaining,
            "completed_job_count": completed_job_count,
            "total_job_count": len(jobs),
            "parallel_workers": worker_count,
            "cuda_devices": devices,
            "attempts_per_second": rate,
            "elapsed_seconds": elapsed,
            "eta_seconds": eta,
            "progress_percent": (
                100.0 if total_attempts == 0
                else 100.0 * finished / total_attempts
            ),
            "updated_at": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            ),
        }
        _write_json(progress_path, payload)
        last_progress_emit = time.monotonic()
        eta_text = _format_eta(eta)
        print(
            "m5-ar progress: "
            f"{finished}/{total_attempts} attempts "
            f"({payload['progress_percent']:.1f}%), "
            f"{completed_job_count}/{len(jobs)} jobs, "
            f"{rate:.2f} attempts/s, ETA {eta_text}",
            flush=True,
        )

    def maybe_emit_progress(newly_finished: int, *, force: bool = False) -> None:
        if force or time.monotonic() - last_progress_emit >= PROGRESS_EMIT_INTERVAL_SECONDS:
            emit_progress("running", newly_finished)

    emit_progress("running", 0)
    python = python_executable or sys.executable
    completed_new = 0
    chunks = [
        specs[index : index + effective_batch_size]
        for index in range(0, len(specs), effective_batch_size)
    ]
    batch_index_offset = _next_articulated_batch_index(batches_dir) if resume else 0
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = {}
        for index, chunk in enumerate(chunks):
            device = devices[index % len(devices)]
            if effective_batch_size == 1:
                future = pool.submit(
                    _run_articulated_subprocess_attempt,
                    chunk[0],
                    python=python,
                    cuda_device=device,
                    timeout_seconds=int(timeout_seconds),
                )
            else:
                future = pool.submit(
                    _run_articulated_subprocess_batch,
                    chunk,
                    python=python,
                    cuda_device=device,
                    timeout_seconds=int(timeout_seconds),
                    batch_index=batch_index_offset + index,
                    batches_dir=batches_dir,
                )
            futures[future] = chunk
        for future in as_completed(futures):
            result = future.result()
            rows = [result] if isinstance(result, dict) else result
            for row in rows:
                attempt_id = str(row["attempt_id"])
                attempts_by_id[attempt_id] = row
                was_pending = attempt_id in pending_attempt_ids
                pending_attempt_ids.discard(attempt_id)
                if was_pending:
                    job_id = str(row.get("job_id"))
                    previous_count = completed_attempts_by_job.get(job_id, 0)
                    completed_attempts_by_job[job_id] = previous_count + 1
                    if previous_count < len(probe_seeds) <= previous_count + 1:
                        completed_job_count += 1
                completed_new += 1
                _append_jsonl(attempts_path, row)
                maybe_emit_progress(completed_new)
    ordered_attempts = sorted(
        attempts_by_id.values(),
        key=lambda item: (
            int(item.get("job_index") or 0),
            int(item.get("seed") or 0),
        ),
    )
    _write_jsonl(attempts_path, ordered_attempts)
    emit_progress("complete", completed_new)
    metadata["finished_at"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
    )
    _write_json(metadata_path, metadata)
    return _write_articulated_probe_report(
        plan,
        attempts=ordered_attempts,
        reports_dir=reports_dir,
        attempts_path=attempts_path,
        rubric_hash=rubric_hash,
        seeds=probe_seeds,
        threshold=threshold,
        runner_metadata=metadata,
    )


def _write_articulated_probe_report(
    plan: dict[str, Any],
    *,
    attempts: list[dict[str, Any]],
    reports_dir: Path,
    attempts_path: Path,
    rubric_hash: str,
    seeds: list[int],
    threshold: float,
    runner_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from .pair_validation import (
        REPORT_SCHEMA_VERSION,
        STAGE_TERMINAL,
        TERMINAL_SOURCE_KIND,
        _artifact_hash,
        _evidence_id,
        _row_config_hash,
        config_hash,
    )

    probe_seeds = [int(seed) for seed in seeds]
    attempts_by_job: dict[str, list[dict[str, Any]]] = {}
    for attempt in attempts:
        attempts_by_job.setdefault(str(attempt["job_id"]), []).append(attempt)
    band_records = [
        aggregate_band_attempts(
            job,
            attempts_by_job.get(str(job["job_id"]), []),
            seeds=probe_seeds,
            threshold=threshold,
        )
        for job in plan.get("jobs") or []
    ]
    jobs_by_id = {str(job["job_id"]): job for job in plan.get("jobs") or []}
    by_pair: dict[str, list[dict[str, Any]]] = {}
    for row in band_records:
        by_pair.setdefault(str(row["pair_key"]), []).append(row)
    records = []
    for pair_key, rows in sorted(by_pair.items()):
        first_job = jobs_by_id[str(rows[0]["job_id"])]
        passed = [row for row in rows if row["outcome"] == "pass"]
        selected_job = (
            jobs_by_id[str(passed[0]["job_id"])] if passed else None
        )
        blocked = [row for row in rows if row["outcome"] == "blocked"]
        outcome = "pass" if passed else ("blocked" if blocked else "fail")
        valid_attempts = sum(int(row["valid_attempts"]) for row in rows)
        success = sum(int(row["success"]) for row in rows)
        record = {
            "scope": "pair",
            "stage": STAGE_TERMINAL,
            "relation": "place_in",
            "subject": _identity(first_job["subject"]),
            "receiver": _identity(first_job["receiver"]),
            "receiver_state_profile_id": first_job["receiver_state_profile_id"],
            "receiver_state_profile_hash": first_job["receiver_state_profile_hash"],
            "outcome": outcome,
            "pass": True if outcome == "pass" else (None if outcome == "blocked" else False),
            "verdict": "ok" if outcome == "pass" else str((blocked or rows)[0].get("witness_status")),
            "threshold": float(threshold),
            "seeds": probe_seeds,
            "attempts": sum(int(row["attempts"]) for row in rows),
            "valid_attempts": valid_attempts,
            "plan_success": valid_attempts,
            "success": success,
            "min_rate": success / valid_attempts if valid_attempts else 0.0,
            "validated_recipe_id": (
                (selected_job.get("placement_recipe") or {}).get("recipe_id")
                if selected_job
                else None
            ),
            "validated_placement_recipe": (
                selected_job.get("placement_recipe")
                if selected_job
                else None
            ),
            "validated_band_ids": [row["band_id"] for row in passed],
            "relative_yaw": first_job.get("relative_yaw") or {},
            "metrics": {
                "band_results": [
                    {
                        **row,
                        "yaw_witness": jobs_by_id[str(row["job_id"])].get("yaw_witness") or {},
                        "recipe_id": jobs_by_id[str(row["job_id"])].get("placement_recipe", {}).get("recipe_id"),
                    }
                    for row in rows
                ]
            },
        }
        record["config_hash"] = _row_config_hash(record)
        records.append(record)
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "kind": "terminal_settle_pair_report",
        "scope": "pair",
        "stage": STAGE_TERMINAL,
        "source_kind": TERMINAL_SOURCE_KIND,
        "config_hash": config_hash(),
        "source_rubric_hash": rubric_hash,
        "articulated_policy_hash": articulated_policy_hash(),
        "runner": {
            "started_at": now,
            "finished_at": now,
            "source_probe_plan_hash": plan.get("plan_hash"),
            "attempts_path": str(attempts_path),
            **(runner_metadata or {}),
        },
        "records": records,
    }
    report["evidence_id"] = _evidence_id(report)
    report["report_hash"] = _artifact_hash(report, "report_hash")
    report_path = reports_dir / f"{report['evidence_id']}.json"
    report_path.write_text(json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "evidence_id": report["evidence_id"],
        "evidence_path": str(report_path),
        "record_count": len(records),
        "attempt_count": len(attempts),
    }


def _run_articulated_subprocess_attempt(
    spec: dict[str, Any],
    *,
    python: str,
    cuda_device: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
    command = [
        python,
        "-m",
        "if_benchmark.articulated_pair_cli",
        "run-probe",
        "--probe-job",
        str(spec["job_path"]),
        "--seed",
        str(spec["seed"]),
    ]
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=max(1, int(timeout_seconds)),
            env=env,
        )
        result = _parse_json_output(completed.stdout)
        if result is None:
            reason = (
                "worker_process_failed"
                if completed.returncode
                else "worker_no_json_result"
            )
            result = {
                "outcome": "blocked",
                "success": False,
                "failure_reason": reason,
            }
    except subprocess.TimeoutExpired as exc:
        completed = None
        result = {
            "outcome": "blocked",
            "success": False,
            "failure_reason": "worker_timeout",
            "stderr_tail": str(exc.stderr or "")[-2000:],
        }
    result.update(
        {
            "attempt_id": spec["attempt_id"],
            "job_id": spec["job"]["job_id"],
            "job_index": spec["job_index"],
            "pair_key": spec["job"]["pair_key"],
            "seed": spec["seed"],
            "cuda_device": str(cuda_device),
            "duration_sec": float(
                result.get("duration_sec")
                or (time.monotonic() - started)
            ),
        }
    )
    if completed is not None:
        result["worker_returncode"] = int(completed.returncode)
        result["stderr_tail"] = str(completed.stderr or "")[-2000:]
    return result


def _run_articulated_subprocess_batch(
    specs: list[dict[str, Any]],
    *,
    python: str,
    cuda_device: str,
    timeout_seconds: int,
    batch_index: int,
    batches_dir: Path,
) -> list[dict[str, Any]]:
    batch_path = batches_dir / f"batch_{batch_index:06d}.json"
    output_path = batches_dir / f"batch_{batch_index:06d}.results.jsonl"
    _write_json(
        batch_path,
        {
            "attempts": [
                {
                    "attempt_id": spec["attempt_id"],
                    "job": spec["job"],
                    "job_id": spec["job"]["job_id"],
                    "job_index": spec["job_index"],
                    "pair_key": spec["job"]["pair_key"],
                    "seed": spec["seed"],
                }
                for spec in specs
            ]
        },
    )
    if output_path.exists():
        output_path.unlink()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
    command = [
        python,
        "-m",
        "if_benchmark.articulated_terminal_worker",
        "--batch-file",
        str(batch_path),
        "--batch-output",
        str(output_path),
    ]
    started = time.monotonic()
    timed_out = False
    completed: subprocess.CompletedProcess[str] | None = None
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=max(1, int(timeout_seconds)) * max(1, len(specs)),
            env=env,
        )
    except subprocess.TimeoutExpired:
        timed_out = True
    parsed_rows = _read_jsonl(output_path) if output_path.is_file() else []
    parsed_by_id = {
        str(row.get("attempt_id")): row
        for row in parsed_rows
        if row.get("attempt_id") is not None
    }
    elapsed = time.monotonic() - started
    stderr_tail = str(completed.stderr or "")[-2000:] if completed is not None else ""
    results: list[dict[str, Any]] = []
    for spec in specs:
        attempt_id = str(spec["attempt_id"])
        result = parsed_by_id.get(attempt_id)
        if result is None:
            result = {
                "outcome": "blocked",
                "success": False,
                "failure_reason": (
                    "worker_timeout"
                    if timed_out
                    else (
                        "worker_no_json_result"
                        if completed is not None and completed.returncode == 0
                        else "worker_process_failed"
                    )
                ),
            }
        result.update(
            {
                "attempt_id": attempt_id,
                "job_id": spec["job"]["job_id"],
                "job_index": spec["job_index"],
                "pair_key": spec["job"]["pair_key"],
                "seed": spec["seed"],
                "cuda_device": str(cuda_device),
                "duration_sec": float(result.get("duration_sec") or elapsed),
                "worker_returncode": (
                    int(completed.returncode) if completed is not None else None
                ),
                "stderr_tail": stderr_tail,
            }
        )
        results.append(result)
    return results


def _parse_json_output(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _format_eta(seconds: float | None) -> str:
    if seconds is None:
        return "estimating"
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _iter_rigid_pick_subjects(
    catalog: dict[str, Any],
    subject_assets: set[str] | None,
) -> Iterable[tuple[dict[str, Any], dict[str, Any]]]:
    for asset_name in sorted((catalog.get("assets") or {})):
        if subject_assets is not None and asset_name not in subject_assets:
            continue
        asset = catalog["assets"][asset_name]
        for model_id in sorted((asset.get("models") or {}), key=str):
            model = asset["models"][model_id]
            candidate = (model.get("candidates") or {}).get("pick_subject")
            if not isinstance(candidate, dict) or not candidate.get("candidate"):
                continue
            metadata = candidate.get("metadata") or {}
            if metadata.get("has_articulation"):
                continue
            yield _endpoint(candidate), candidate


def _probe_job(
    *,
    pair_key: str,
    subject: dict[str, Any],
    receiver: dict[str, Any],
    canonical_pose: dict[str, Any],
    profile_id: str,
    profile: dict[str, Any],
    profile_hash: str,
    yaw_domain: dict[str, Any],
    band: dict[str, Any],
) -> dict[str, Any]:
    band_id = str(band["band_id"])
    recipe = canonical_articulated_terminal_placement_recipe(
        profile_id=profile_id,
        profile=profile,
        band_id=band_id,
    )
    if recipe["receiver_state_profile_hash"] != profile_hash:
        raise ValueError("articulated receiver profile hash mismatch")
    return {
        "job_id": f"articulated_terminal_settle|{pair_key}|{band_id}",
        "scope": "state_conditioned_pair",
        "stage": "articulated_terminal_settle",
        "relation": "place_in",
        "pair_key": pair_key,
        "subject": subject,
        "receiver": receiver,
        "receiver_state_profile_id": profile_id,
        "receiver_state_profile_hash": profile_hash,
        "receiver_state_profile": profile,
        "canonical_poses": {"subject": canonical_pose},
        "relative_yaw": yaw_domain,
        "band_id": band_id,
        "yaw_witness": band,
        "placement_recipe": recipe,
        "recipe_hash": recipe["recipe_hash"],
        "articulated_policy_hash": articulated_policy_hash(),
        "seeds": [0, 1, 2],
        "threshold": 0.6,
    }


def _ground_yaw_domain(domain: dict[str, Any], profile_id: str) -> None:
    domain["geometry_source"] = "reviewed_cavity_usable_region"
    domain["receiver_state_profile_id"] = profile_id
    for band in domain.get("geometric_bands") or []:
        band["geometry_source"] = "reviewed_cavity_usable_region"
        band["sampling_eligible"] = bool(band.get("continuous"))
    fallback = domain.get("fallback_witness")
    if isinstance(fallback, dict):
        fallback["geometry_source"] = "reviewed_cavity_usable_region"
        fallback["sampling_eligible"] = False


def _box_geometry(x: float, y: float) -> dict[str, Any]:
    return {
        "x": float(x),
        "y": float(y),
        "area": float(x * y),
        "points_xy": [
            [-x / 2.0, -y / 2.0],
            [-x / 2.0, y / 2.0],
            [x / 2.0, -y / 2.0],
            [x / 2.0, y / 2.0],
        ],
        "source": "reviewed_cavity_usable_region",
    }


def _endpoint(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "asset": str(row["asset"]),
        "model_id": str(row["model_id"]),
        "category": str(row.get("category") or row["asset"]),
        "inventory_id": str(row.get("inventory_id") or f"{row['asset']}|{row['model_id']}"),
        "metadata": row.get("metadata") or {},
    }


def receiver_state_profile_hash(profile_id: str, profile: dict[str, Any]) -> str:
    """Hash the complete inline fixture/state/cavity/skill contract."""

    return _shared_receiver_state_profile_hash(profile_id, profile)


def _hash_payload(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _vector(value: Any, length: int) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def _nonzero_vector(value: Any) -> bool:
    vector = _vector(value, 3)
    return bool(vector and sum(item * item for item in vector) > 1e-12)


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _deduplicate(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _identity(endpoint: dict[str, Any]) -> dict[str, str]:
    return {"asset": str(endpoint["asset"]), "model_id": str(endpoint["model_id"])}


def _next_articulated_batch_index(batches_dir: Path) -> int:
    indices: list[int] = []
    for path in batches_dir.glob("batch_*.json"):
        try:
            indices.append(int(path.stem.rsplit("_", 1)[-1]))
        except ValueError:
            continue
    for path in batches_dir.glob("batch_*.results.jsonl"):
        prefix = path.name.removesuffix(".results.jsonl")
        try:
            indices.append(int(prefix.rsplit("_", 1)[-1]))
        except ValueError:
            continue
    return max(indices, default=-1) + 1


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON mapping in {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
