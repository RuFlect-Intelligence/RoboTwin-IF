"""Staleness and deterministic migration audit for pre-contract evidence."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from if_benchmark.schema.serialization import canonical_data

from .contract import SuccessContract
from .relation import RelationContract
from .snapshot import SuccessSnapshotError, normalize_success_snapshot


def evidence_staleness(
    evidence: Mapping[str, Any], *, contract: SuccessContract
) -> dict[str, Any]:
    """Return a strict identity audit; verdicts never make evidence fresh."""

    relation = RelationContract.from_success_contract(contract)
    failures: list[str] = []
    for key, expected in {
        "success_contract_version": contract.version,
        "success_contract_hash": contract.hash,
    }.items():
        if evidence.get(key) != expected:
            failures.append(f"{key}_missing_or_mismatch")
    predicate = str(evidence.get("predicate") or "")
    if predicate in {"relation_holds", "placement_completed"}:
        if evidence.get("relation_contract_hash") != relation.hash:
            failures.append("relation_contract_hash_missing_or_mismatch")
    snapshot = evidence.get("success_snapshot")
    if not isinstance(snapshot, Mapping):
        failures.append("success_snapshot_missing")
    else:
        try:
            normalized = normalize_success_snapshot(snapshot, contract)
        except SuccessSnapshotError:
            failures.append("success_snapshot_invalid")
        else:
            if evidence.get("success_snapshot_hash") != normalized.get(
                "success_snapshot_hash"
            ):
                failures.append("success_snapshot_hash_missing_or_mismatch")
            if evidence.get("profile_hashes") != normalized.get("profile_hashes"):
                failures.append("profile_hashes_missing_or_mismatch")
            if evidence.get("source_hashes") != normalized.get("source_hashes"):
                failures.append("source_hashes_missing_or_mismatch")
    return {
        "status": "fresh" if not failures else "stale",
        "fresh": not failures,
        "failures": failures,
        **contract.identity,
    }


def audit_legacy_success_evidence(
    rows: Iterable[Mapping[str, Any]], *, contract: SuccessContract
) -> dict[str, Any]:
    """Account old evidence as rejudgable, replay-required, or blocked."""

    entries: list[dict[str, Any]] = []
    counts = {"rejudgable": 0, "replay_required": 0, "blocked": 0}
    for index, raw in enumerate(rows):
        row = canonical_data(dict(raw))
        record_id = str(row.get("id") or row.get("record_id") or index)
        snapshot = row.get("success_snapshot")
        if isinstance(snapshot, Mapping):
            try:
                normalize_success_snapshot(snapshot, contract)
            except SuccessSnapshotError as exc:
                classification = "blocked"
                reasons = ["canonical_snapshot_invalid", str(exc)]
            else:
                classification = "rejudgable"
                reasons = ["canonical_snapshot_complete"]
        else:
            signals = _available_signals(row)
            unrecoverable = bool(
                row.get("source_unavailable")
                or row.get("profile_unavailable")
            )
            if unrecoverable:
                classification = "blocked"
                reasons = ["source_or_profile_unrecoverable"]
            elif _raw_metrics_sufficient(row, signals):
                classification = "rejudgable"
                reasons = ["raw_metrics_sufficient"]
            else:
                classification = "replay_required"
                reasons = [
                    "missing_actual_contact_gripper_se3_or_stability",
                    *sorted(signals),
                ]
        counts[classification] += 1
        entries.append(
            {
                "record_id": record_id,
                "classification": classification,
                "reasons": reasons,
            }
        )
    return {
        "status": "complete",
        "counts": counts,
        "entries": entries,
        **contract.identity,
    }


def _available_signals(row: Mapping[str, Any]) -> set[str]:
    metrics = row.get("raw_metrics") or row.get("metrics") or {}
    if not isinstance(metrics, Mapping):
        return set()
    names = {
        "actual_contacts": ("active_contacts", "contact_manifolds"),
        "actual_gripper": ("actual_width_m", "actual_qpos"),
        "relative_se3": ("translation_drift_m", "rotation_drift_deg"),
        "stability": ("stability", "stable"),
        "geometry": ("aabb", "entities", "poses"),
        "articulation": ("qpos", "articulations"),
        "joint_limits": ("lower", "upper", "limits"),
        "state_profile": ("state_profile_hash",),
    }
    return {
        name
        for name, keys in names.items()
        if any(key in metrics for key in keys)
    }


def _raw_metrics_sufficient(row: Mapping[str, Any], signals: set[str]) -> bool:
    predicate = str(row.get("predicate") or "")
    if predicate == "articulation_state_success":
        return {"articulation", "joint_limits", "state_profile", "stability"} <= signals
    if predicate == "relation_holds":
        return {"geometry", "actual_contacts", "stability"} <= signals
    if predicate == "placement_completed":
        return {
            "geometry",
            "actual_contacts",
            "actual_gripper",
            "stability",
        } <= signals
    if predicate in {"held_by", "pick_success"}:
        return {
            "actual_contacts",
            "actual_gripper",
            "relative_se3",
            "stability",
        } <= signals
    return False


__all__ = ["audit_legacy_success_evidence", "evidence_staleness"]
