"""M16 content-addressed I/O through explicit, repository-local data roots."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import yaml

from if_benchmark.schema.serialization import artifact_hash


class IntegrityError(ValueError):
    """An immutable input cannot establish its declared identity."""


def parallel_map(function, values, *, workers: int = 32, processes: bool = False):
    """Ordered, bounded work queue; never enqueue a million file futures."""
    if workers < 1:
        raise ValueError("workers must be positive")
    if workers == 1:
        yield from map(function, values)
        return
    factory = ProcessPoolExecutor if processes else ThreadPoolExecutor
    options = {"mp_context": multiprocessing.get_context("fork")} if processes else {}
    with factory(max_workers=workers, **options) as executor:
        pending, iterator = deque(), iter(values)
        for _ in range(workers * 2):
            try:
                pending.append(executor.submit(function, next(iterator)))
            except StopIteration:
                break
        while pending:
            yield pending.popleft().result()
            try:
                pending.append(executor.submit(function, next(iterator)))
            except StopIteration:
                pass


def source_stamps(files, *, workers: int = 32) -> dict[str, tuple]:
    return dict(parallel_map(lambda row: (row["path"], stamp(Path(row["path"]))), files, workers=workers))


def verify_stamps(expected: Mapping[str, tuple], *, workers: int = 32) -> None:
    def unchanged(row):
        path, before = row
        if stamp(Path(path)) != tuple(before):
            raise IntegrityError(f"input changed during audit: {path}")
    for _ in parallel_map(unchanged, expected.items(), workers=workers):
        pass


def source_file_error(row: Mapping[str, Any]) -> str | None:
    path = Path(row["path"])
    try:
        before = stamp(path)
        if before[2] != row["size_bytes"] or file_hash(path) != row["sha256"] or stamp(path) != before:
            return f"source changed: {path}"
    except OSError as exc:
        return f"source unavailable: {path}: {exc}"
    return None


def seal(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result.pop("artifact_hash", None)
    result["artifact_hash"] = artifact_hash(result)
    return result


def verify(value: Mapping[str, Any], key: str = "artifact_hash") -> None:
    if not value.get(key) or value[key] != artifact_hash(value, omit_keys=(key,)):
        raise IntegrityError(f"invalid {key}: {value.get('artifact_kind', 'record')}")


def stamp(path: Path) -> tuple[int, ...]:
    s = path.stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream) if path.suffix == ".json" else yaml.load(stream, Loader=yaml.CSafeLoader)
    if not isinstance(value, dict):
        raise IntegrityError(f"expected mapping: {path}")
    return value


def atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as stream:
            name = stream.name
            if path.suffix == ".json":
                json.dump(value, stream, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
                stream.write("\n")
            else:
                yaml.dump(dict(value), stream, Dumper=yaml.CSafeDumper, sort_keys=True,
                          allow_unicode=True, width=100)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if name is not None and os.path.exists(name):
            os.unlink(name)


def relative_reference(recorded: str, recorded_parent: str, local_parent: Path) -> Path:
    """Resolve a recorded child relative to its declared parent, never a host map.

    Original strings remain in every hash/ID. The two parents are explicit
    artifact bindings; no historical mount prefixes are recognized here.
    """
    path = Path(recorded)
    relative = path.relative_to(Path(recorded_parent)) if path.is_absolute() else path
    if not relative.parts or ".." in relative.parts:
        raise IntegrityError(f"invalid child reference: {recorded}")
    return local_parent / relative


@dataclass
class InputLedger:
    files: dict[str, dict[str, Any]] = field(default_factory=dict)

    def pin(self, path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
        key = str(path.absolute())  # Do not erase the user-facing symlink entry.
        before = stamp(path)
        previous = self.files.get(key)
        if previous is not None:
            if tuple(previous["stat"]) != before:
                raise IntegrityError(f"input changed during read: {path}")
            row = previous
        else:
            digest = file_hash(path)
            if stamp(path) != before:
                raise IntegrityError(f"input changed while hashing: {path}")
            row = {"path": key, "sha256": digest, "size_bytes": before[2], "stat": list(before)}
            self.files[key] = row
        if expected_sha256 is not None and row["sha256"] != expected_sha256:
            raise IntegrityError(f"file hash mismatch: {path}")
        return row

    def load(self, path: Path, hash_key: str | None = "artifact_hash") -> dict[str, Any]:
        row = self.pin(path)
        value = read(path)
        if stamp(path) != tuple(row["stat"]):
            raise IntegrityError(f"input changed while parsing: {path}")
        if hash_key is not None:
            verify(value, hash_key)
        return value

    def stable(self) -> None:
        for row in self.files.values():
            if stamp(Path(row["path"])) != tuple(row["stat"]):
                raise IntegrityError(f"input changed during build: {row['path']}")

    def records(self) -> list[dict[str, Any]]:
        # Filesystem timestamps/inodes are resume hints, never semantic identity.
        return [{k: v for k, v in row.items() if k != "stat"}
                for _, row in sorted(self.files.items())]
