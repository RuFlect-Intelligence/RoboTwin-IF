"""Narrow read-only compatibility for one reviewed operational-budget patch.

This is not a version bypass or an identity migration writer.  The sole
allowed source deltas are exact reviewed old/new ``layout_core`` and
``pick_planning_contract`` file digests;
all task, source, config, contract, physics and planner identities remain exact.
The current digest must match the final reviewed build, so a later core edit
automatically disables the exception.  An unfilled digest disables migration.

Callers still validate the immutable cache artifact's self-hash and failure
classification.  Planner-negative entries additionally require strict current
provider identity and complete frozen-witness replay.  Limited-search failures
must never be admitted as deterministic unsat through this compatibility path.
"""

from __future__ import annotations

from typing import Any, Mapping

from if_benchmark.schema.serialization import artifact_hash, canonical_data


REVIEWED_OLD_LAYOUT_CORE_SHA256 = (
    "beb863c1c5c803798655061596ac45560034b25c7ede104bf1fd76e219554530"
)
# Filled only after the operational-budget implementation and focused
# regression have been reviewed.  Do not compute this from the live source:
# doing so would silently accept every future, possibly physical change.
REVIEWED_NEW_LAYOUT_CORE_SHA256: str | None = (
    "0dc48814a89d59ebd1d72e2472e94d0eff31b5dd7effeea6f452d28705a9fbdf"
)
REVIEWED_OLD_PICK_PLANNING_CONTRACT_SHA256 = (
    "a698e77ea01dfa9abaa16c2cc28da69214d199031e955a4bd9c7ed7ff3c39f73"
)
REVIEWED_NEW_PICK_PLANNING_CONTRACT_SHA256: str | None = (
    "d9f2bbab2310e0939b7ef2eb1ee56312851dd59360e4b4c28fecf5cf9976eb09"
)


def _valid_identity(value: Any) -> bool:
    return isinstance(value, Mapping) and value.get("identity_hash") == artifact_hash(
        value, omit_keys=("identity_hash",)
    )


def equivalent_attempt_identity(cached: Any, current: Any) -> bool:
    """Compare complete identities, allowing only the pinned source deltas.

    A detached in-memory comparison updates its *derived* identity hash only;
    neither input mapping nor any cache file is modified or republished.
    """

    try:
        if not _valid_identity(cached) or not _valid_identity(current):
            return False
        if canonical_data(cached) == canonical_data(current):
            return True
        transitions = {
            "layout_core": (REVIEWED_OLD_LAYOUT_CORE_SHA256, REVIEWED_NEW_LAYOUT_CORE_SHA256),
            "pick_planning_contract": (
                REVIEWED_OLD_PICK_PLANNING_CONTRACT_SHA256,
                REVIEWED_NEW_PICK_PLANNING_CONTRACT_SHA256,
            ),
        }
        old_sources = cached["solver_identity"]["source_hashes"]
        new_sources = current["solver_identity"]["source_hashes"]
        compared = canonical_data(cached)
        for name, (reviewed_old, reviewed_new) in transitions.items():
            if (
                not isinstance(reviewed_new, str)
                or len(reviewed_new) != 64
                or any(character not in "0123456789abcdef" for character in reviewed_new)
                or old_sources[name] != reviewed_old
                or new_sources[name] != reviewed_new
            ):
                return False
            compared["solver_identity"]["source_hashes"][name] = reviewed_new
        compared["identity_hash"] = artifact_hash(compared, omit_keys=("identity_hash",))
        return compared == canonical_data(current)
    except (KeyError, TypeError, ValueError):
        return False


def equivalent_provider_identity(cached: Any, current: Any) -> bool:
    """No provider/runtime/physics compatibility exception is authorized."""

    try:
        return (
            _valid_identity(cached)
            and _valid_identity(current)
            and canonical_data(cached) == canonical_data(current)
        )
    except (TypeError, ValueError):
        return False
