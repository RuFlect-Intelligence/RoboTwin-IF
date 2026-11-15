"""Versioned pure data boundary consumed by all success predicates."""

from __future__ import annotations

from typing import Any, Mapping

from if_benchmark.schema.serialization import artifact_hash, canonical_data

from .contract import SuccessContract
from .relation import RelationContract


SUCCESS_SNAPSHOT_VERSION = "benchmarkv1_success_snapshot_v1"


class SuccessSnapshotError(ValueError):
    """A SuccessSnapshot is incomplete, stale, or malformed."""


def build_success_snapshot(
    *,
    initial: Mapping[str, Any],
    terminal: Mapping[str, Any],
    stability: Mapping[str, Any],
    execution_events: Mapping[str, Any] | None,
    source_hashes: Mapping[str, str],
    contract: SuccessContract,
    profiles: Mapping[str, Any] | None = None,
    profile_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build one canonical, self-hashed SuccessSnapshot.

    The builder records observations only.  It deliberately performs no
    simulation action, recovery, target selection, or success evaluation.
    """

    if not source_hashes or any(not str(value) for value in source_hashes.values()):
        raise SuccessSnapshotError("success snapshot requires non-empty source hashes")
    relation_contract = RelationContract.from_success_contract(contract)
    hashes = dict(profile_hashes or {})
    for name, profile in (profiles or {}).items():
        if isinstance(profile, Mapping) and profile.get("profile_hash"):
            hashes.setdefault(str(name), str(profile["profile_hash"]))
    value: dict[str, Any] = {
        "snapshot_version": SUCCESS_SNAPSHOT_VERSION,
        **contract.identity,
        **relation_contract.identity,
        "initial": canonical_data(dict(initial)),
        "terminal": canonical_data(dict(terminal)),
        "stability": canonical_data(dict(stability)),
        "execution_events": canonical_data(dict(execution_events or {})),
        "profiles": canonical_data(dict(profiles or {})),
        "profile_hashes": canonical_data(hashes),
        "source_hashes": canonical_data(dict(source_hashes)),
    }
    value["success_snapshot_hash"] = artifact_hash(value)
    return value


def normalize_success_snapshot(
    snapshot: Mapping[str, Any], contract: SuccessContract
) -> dict[str, Any]:
    """Validate identity/self-hash and return a detached canonical snapshot."""

    value = canonical_data(dict(snapshot))
    if value.get("snapshot_version") != SUCCESS_SNAPSHOT_VERSION:
        raise SuccessSnapshotError("unsupported success snapshot version")
    if value.get("success_contract_version") != contract.version:
        raise SuccessSnapshotError("success snapshot contract version mismatch")
    if value.get("success_contract_hash") != contract.hash:
        raise SuccessSnapshotError("success snapshot contract hash mismatch")
    expected = artifact_hash(value, omit_keys=("success_snapshot_hash",))
    if value.get("success_snapshot_hash") != expected:
        raise SuccessSnapshotError("success snapshot self-hash mismatch")
    relation_contract = RelationContract.from_success_contract(contract)
    if value.get("relation_contract_hash") != relation_contract.hash:
        raise SuccessSnapshotError("success snapshot relation contract hash mismatch")
    for key in (
        "initial",
        "terminal",
        "stability",
        "source_hashes",
        "profile_hashes",
    ):
        if not isinstance(value.get(key), Mapping):
            raise SuccessSnapshotError(f"success snapshot missing {key}")
    return value


__all__ = [
    "SUCCESS_SNAPSHOT_VERSION",
    "SuccessSnapshotError",
    "build_success_snapshot",
    "normalize_success_snapshot",
]
