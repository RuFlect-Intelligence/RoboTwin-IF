"""Deterministic serialization and provenance helpers for BenchmarkV1.

These helpers are deliberately simulator-independent.  They define the stable
byte/content boundary used by schema artifacts and by later milestone
generators without making YAML key order part of object semantics.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from .contracts import SCHEMA_VERSION


def _identity_v1(document: Mapping[str, Any]) -> dict[str, Any]:
    return canonical_data(dict(document))


MIGRATION_REGISTRY = {
    (SCHEMA_VERSION, SCHEMA_VERSION): _identity_v1,
}



def canonical_json(value: Any) -> str:
    """Return stable compact JSON for a JSON-compatible value."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_data(value: Any) -> Any:
    """Return a detached JSON-compatible copy with canonical scalar semantics."""

    return json.loads(canonical_json(value))


def canonical_yaml(value: Any) -> str:
    """Return deterministic YAML.

    YAML is a human-facing export.  Hashing and ids use :func:`canonical_json`
    so emitter formatting changes never alter semantic identity.
    """

    return yaml.safe_dump(
        canonical_data(value),
        sort_keys=True,
        allow_unicode=True,
        default_flow_style=False,
        width=100,
    )


def artifact_hash(value: Any, *, omit_keys: Iterable[str] = ()) -> str:
    """Hash canonical content, optionally omitting top-level self-hash fields."""

    omitted = set(omit_keys)
    if isinstance(value, Mapping) and omitted:
        payload = {key: item for key, item in value.items() if key not in omitted}
    else:
        payload = value
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def deterministic_id(prefix: str, payload: Any, *, length: int = 16) -> str:
    """Build a stable content-derived identifier."""

    if not prefix or not prefix[0].isalpha() or not all(
        character.isalnum() or character in "_-" for character in prefix
    ):
        raise ValueError(f"invalid deterministic id prefix: {prefix!r}")
    if length < 8 or length > 64:
        raise ValueError("deterministic id length must be between 8 and 64")
    return f"{prefix}_{artifact_hash(payload)[:length]}"


def build_provenance(
    generator: str,
    config_hash: str,
    sources: Sequence[Mapping[str, Any]],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build deterministic artifact provenance.

    Every source must identify its kind and content hash.  ``path`` and
    ``record_id`` are optional because some sources are logical records rather
    than standalone files.
    """

    if not generator:
        raise ValueError("provenance generator must be non-empty")
    if not config_hash:
        raise ValueError("provenance config_hash must be non-empty")
    normalized_sources: list[dict[str, Any]] = []
    allowed_source_keys = {"kind", "hash", "path", "record_id", "metadata"}
    for index, source in enumerate(sources):
        row = canonical_data(dict(source))
        if not row.get("kind"):
            raise ValueError(f"provenance source {index} is missing kind")
        if not row.get("hash"):
            raise ValueError(f"provenance source {index} is missing hash")
        extra_keys = sorted(set(row) - allowed_source_keys)
        if extra_keys:
            raise ValueError(
                f"provenance source {index} has unsupported keys: {extra_keys}"
            )
        normalized_sources.append(row)
    normalized_sources.sort(
        key=lambda row: (
            str(row.get("kind", "")),
            str(row.get("path", "")),
            str(row.get("record_id", "")),
            str(row.get("hash", "")),
        )
    )
    result: dict[str, Any] = {
        "generator": generator,
        "config_hash": config_hash,
        "sources": normalized_sources,
    }
    if metadata:
        result["metadata"] = canonical_data(dict(metadata))
    return result


def migrate_document(
    document: Mapping[str, Any],
    *,
    from_version: int | None = None,
    target_version: int = SCHEMA_VERSION,
) -> dict[str, Any]:
    """Migrate a BenchmarkV1 document through the registered version path.

    Version 1 is the first formal contract, so its only current migration is a
    detached identity copy.  The explicit registry boundary prevents callers
    from silently accepting unknown future or legacy versions.
    """

    source_version = int(
        from_version if from_version is not None else document.get("schema_version", 0)
    )
    migration = MIGRATION_REGISTRY.get((source_version, target_version))
    if migration is not None:
        return migration(document)
    raise ValueError(
        f"unsupported BenchmarkV1 schema migration: {source_version} -> {target_version}"
    )


def load_document(
    path: Path,
    *,
    migrate: bool = True,
    expected_kind: str | None = None,
) -> dict[str, Any]:
    """Load one JSON/YAML BenchmarkV1 document."""

    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
    else:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"BenchmarkV1 document must be a mapping: {path}")
    result = migrate_document(value) if migrate else canonical_data(value)
    if expected_kind is not None and result.get("kind") != expected_kind:
        raise ValueError(
            f"BenchmarkV1 document kind {result.get('kind')!r} does not match "
            f"{expected_kind!r}: {path}"
        )
    return result


def dump_document(
    path: Path,
    document: Mapping[str, Any],
    *,
    validate: bool = True,
) -> None:
    """Validate and deterministically write one BenchmarkV1 document."""

    normalized = migrate_document(document)
    if validate:
        # Local import avoids making serialization depend on validator startup.
        from .cli import validate_document

        report = validate_document(normalized)
        if not report["valid"]:
            raise ValueError(f"invalid BenchmarkV1 document: {report['errors']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        path.write_text(canonical_json(normalized) + "\n", encoding="utf-8")
    else:
        path.write_text(canonical_yaml(normalized), encoding="utf-8")
