"""Reviewed project-local exclusions from the BenchmarkV1 task universe.

These exclusions describe semantic ineligibility, not simulator or skill
failure.  They deliberately live outside ``assets/**`` so the shared asset
catalog remains immutable.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import yaml

from if_benchmark.schema import artifact_hash


DEFAULT_SEMANTIC_EXCLUSIONS_PATH = Path(
    "if_benchmark/data/asset_validation/semantic_exclusions_v1.yml"
)
POLICY_SCHEMA_VERSION = 1


class AssetSemanticExclusionError(ValueError):
    """The reviewed semantic-exclusion policy is missing or stale."""


def load_asset_semantic_exclusions(
    path: Path = DEFAULT_SEMANTIC_EXCLUSIONS_PATH,
    *,
    asset_catalog_path: Path | None = None,
) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.load(stream, Loader=yaml.CSafeLoader)
    if not isinstance(value, dict):
        raise AssetSemanticExclusionError(
            f"expected semantic-exclusion mapping: {path}"
        )
    if value.get("schema_version") != POLICY_SCHEMA_VERSION:
        raise AssetSemanticExclusionError("semantic-exclusion schema is stale")
    if value.get("review_status") != "reviewed":
        raise AssetSemanticExclusionError(
            "semantic-exclusion policy is not reviewed"
        )
    expected_hash = artifact_hash(value, omit_keys=("policy_hash",))
    if value.get("policy_hash") != expected_hash:
        raise AssetSemanticExclusionError(
            "semantic-exclusion policy hash mismatch"
        )
    if asset_catalog_path is not None:
        expected_catalog_hash = str(
            value.get("source_asset_catalog_sha256") or ""
        )
        actual_catalog_hash = _sha256_file(asset_catalog_path)
        if not expected_catalog_hash or expected_catalog_hash != actual_catalog_hash:
            raise AssetSemanticExclusionError(
                "semantic-exclusion source asset catalog hash mismatch"
            )
    # Expanding here also validates every rule and catches duplicate coverage.
    expand_asset_semantic_exclusions(value)
    return value


def expand_asset_semantic_exclusions(
    policy: Mapping[str, Any],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    expanded: dict[tuple[str, str, str], dict[str, Any]] = {}
    for rule in policy.get("rules") or []:
        if not isinstance(rule, Mapping):
            raise AssetSemanticExclusionError(
                "semantic-exclusion rule must be a mapping"
            )
        required = (
            "rule_id",
            "asset",
            "model_ids",
            "uses",
            "disposition",
            "reason_code",
            "rationale",
        )
        missing = [key for key in required if not rule.get(key)]
        if missing:
            raise AssetSemanticExclusionError(
                "semantic-exclusion rule missing " + ", ".join(missing)
            )
        if rule.get("disposition") != "unsupported":
            raise AssetSemanticExclusionError(
                "semantic exclusions must use disposition=unsupported"
            )
        model_ids = rule.get("model_ids")
        uses = rule.get("uses")
        if not isinstance(model_ids, list) or not isinstance(uses, list):
            raise AssetSemanticExclusionError(
                "semantic-exclusion model_ids and uses must be lists"
            )
        for model_id in model_ids:
            for use in uses:
                key = (str(rule["asset"]), str(model_id), str(use))
                if key in expanded:
                    raise AssetSemanticExclusionError(
                        "duplicate semantic-exclusion coverage: " + "|".join(key)
                    )
                expanded[key] = {
                    "rule_id": str(rule["rule_id"]),
                    "asset": key[0],
                    "model_id": key[1],
                    "use": key[2],
                    "disposition": "unsupported",
                    "reason_code": str(rule["reason_code"]),
                    "rationale": str(rule["rationale"]),
                    "policy_version": str(policy.get("policy_version") or ""),
                    "policy_hash": str(policy.get("policy_hash") or ""),
                }
    return expanded


def semantic_exclusion_for(
    policy: Mapping[str, Any] | None,
    asset: str,
    model_id: str,
    use: str,
) -> dict[str, Any] | None:
    if policy is None:
        return None
    return expand_asset_semantic_exclusions(policy).get(
        (str(asset), str(model_id), str(use))
    )


def semantic_exclusion_identity(
    policy: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if policy is None:
        return None
    return {
        "semantic_exclusion_policy_version": policy.get("policy_version"),
        "semantic_exclusion_policy_hash": policy.get("policy_hash"),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
