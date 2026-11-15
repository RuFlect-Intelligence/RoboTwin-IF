"""Filesystem helpers for frozen BenchmarkV1 schema artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


def write_yaml(path: Path, data: Mapping[str, Any]) -> None:
    """Write a human-readable deterministic generated artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(
            dict(data),
            stream,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
            width=100,
        )


def load_yaml(path: Path) -> dict[str, Any] | None:
    """Load one generated mapping artifact, returning ``None`` when absent."""

    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if data is None:
        return None
    if not isinstance(data, dict):
        raise ValueError(f"expected YAML mapping in {path}")
    return data


def load_examples(examples_dir: Path) -> dict[str, dict[str, Any]]:
    """Load all frozen canonical example documents."""

    if not examples_dir.is_dir():
        return {}
    examples: dict[str, dict[str, Any]] = {}
    for path in sorted(examples_dir.glob("*.yml")):
        data = load_yaml(path)
        if data is not None:
            examples[path.name] = data
    return examples


def write_artifact_set(
    schema_path: Path,
    examples_dir: Path,
    audit_path: Path,
    *,
    schema: Mapping[str, Any],
    examples: Mapping[str, Mapping[str, Any]],
    audit: Mapping[str, Any],
) -> None:
    """Atomically-at-the-file-level write the canonical M6 artifact set."""

    write_yaml(schema_path, schema)
    examples_dir.mkdir(parents=True, exist_ok=True)
    for name, example in sorted(examples.items()):
        write_yaml(examples_dir / name, example)
    write_yaml(audit_path, audit)
