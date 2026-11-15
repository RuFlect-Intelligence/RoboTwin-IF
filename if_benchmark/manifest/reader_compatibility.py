"""Finite, exact-file compatibility for the non-semantic snapshot reader change.

The raw implementation hash always includes the current reader and this record.
The legacy projection is accepted only when every changed reader file matches
the reviewed migration AND every unchanged semantic file reconstructs the exact
previous whole implementation hash. No arbitrary version or hash is accepted.
"""
import os
from pathlib import Path

from if_benchmark.schema.serialization import artifact_hash
from .io import file_hash, read, verify


M16_FILES = {'manifest/core.py', 'manifest/cli.py', 'manifest/source_snapshots.py',
             'manifest/reader_compatibility.py'}
EVALUATION_FILES = {'evaluation/contracts.py', 'evaluation/plan.py'}
RECORD = 'manifest/source_reader_compatibility.json'


def code_files(root):
    root = Path(root).absolute()
    files = {}
    for folder, directories, names in os.walk(root, followlinks=False):
        directories[:] = sorted(d for d in directories if not d.startswith('.')
                                and d not in {'data', 'docs', 'tests', 'evaluation', '__pycache__'})
        for name in sorted(names):
            path = Path(folder) / name
            if path.suffix not in {'.py', '.json'}:
                continue
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError('local code escapes workspace')
            files[str(path.relative_to(root))] = file_hash(path)
    return files


def _legacy_projection(root, files):
    try:
        record = read(root / RECORD)
        verify(record)
        if record.get('policy') != 'exact_snapshot_reader_only_migration_v1':
            return None
        if set(record['m16_files']) != M16_FILES or set(record['evaluation_files']) != EVALUATION_FILES:
            return None
        projected = dict(files)
        projected.pop(RECORD, None)
        for name, binding in record['m16_files'].items():
            if files.get(name) != binding['after_sha256']:
                return None
            if binding['before_sha256'] is None:
                projected.pop(name)
            else:
                projected[name] = binding['before_sha256']
        if artifact_hash(projected) != record['legacy_m16_implementation_hash']:
            return None
        return record
    except (OSError, ValueError, KeyError, TypeError):
        return None


def implementation_matches(recorded_hash, *, code_root=None):
    root = Path(code_root or Path(__file__).parents[1]).absolute()
    files = code_files(root)
    if artifact_hash(files) == recorded_hash:
        return True
    record = _legacy_projection(root, files)
    return record is not None and record['legacy_m16_implementation_hash'] == recorded_hash


def evaluation_identity_files(files, *, code_root=None):
    """Keep existing physical plans usable only for this exact reader-only change."""
    root = Path(code_root or Path(__file__).parents[1]).absolute()
    record = _legacy_projection(root, code_files(root))
    if record is None:
        return files
    for name, binding in record['evaluation_files'].items():
        if files.get('if_benchmark/' + name) != binding['after_sha256']:
            return files
    projected = dict(files)
    for name, binding in record['evaluation_files'].items():
        projected['if_benchmark/' + name] = binding['before_sha256']
    projected['m16_semantic_implementation'] = record['legacy_m16_implementation_hash']
    return projected
