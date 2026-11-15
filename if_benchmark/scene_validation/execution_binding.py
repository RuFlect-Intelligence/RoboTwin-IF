"""Immutable per-scene execution identities for a streamed M15 dispatch.

Children are execution-only, never publishable validation plans. A normally
constructed global plan can bind those identities after every route is ready,
without changing any already checkpointed attempt's id or content hash.
"""

from __future__ import annotations

from collections import OrderedDict
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from if_benchmark.schema.serialization import artifact_hash, canonical_data, deterministic_id


EXECUTION_KIND = "scene_validation_execution_plan"
DISPATCH_KIND = "scene_validation_fused_dispatch"
BINDING_POLICY = "immutable_scene_execution_children"
# One dispatch is shared by every scene. Do not repeatedly parse/hash its O(N)
# scene list, or hash unchanged multi-MB source reports for every child.
_DISPATCH_CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()


def _stamp(path: Path) -> tuple[int, ...]:
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = yaml.load(stream, Loader=yaml.CSafeLoader)
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping: {path}")
    return value


def _self_hash(value: Mapping[str, Any]) -> bool:
    return value.get("artifact_hash") == artifact_hash(value, omit_keys=("artifact_hash",))


def _dispatch(path: Path) -> dict[str, Any]:
    path = path.resolve()
    key, stamp = str(path), _stamp(path)
    cached = _DISPATCH_CACHE.get(key)
    if cached is None or cached["stamp"] != stamp:
        value = _load(path)
        if not _self_hash(value) or value.get("artifact_kind") != DISPATCH_KIND:
            raise ValueError(f"invalid dispatch identity: {path}")
        paths = value.get("scene_paths")
        pins = value.get("source_file_hashes")
        if (not isinstance(paths, list) or not all(isinstance(item, str) for item in paths)
                or len(paths) != len(set(paths)) or not isinstance(pins, Mapping)
                or not isinstance(value.get("source_roots"), Mapping)
                or not value.get("config_hash") or value.get("scope") not in {"full", "diagnostic"}):
            raise ValueError(f"invalid dispatch source selection: {path}")
        cached = {"stamp": stamp, "value": value, "paths": frozenset(paths), "pin_stamps": {}}
        _DISPATCH_CACHE[key] = cached
        _DISPATCH_CACHE.move_to_end(key)
        while len(_DISPATCH_CACHE) > 4:
            _DISPATCH_CACHE.popitem(last=False)
    for filename, expected_hash in cached["value"]["source_file_hashes"].items():
        source = Path(str(filename))
        source_stamp = _stamp(source)
        if cached["pin_stamps"].get(filename) != source_stamp:
            if _sha(source) != expected_hash:
                raise ValueError(f"dispatch source pin changed: {source}")
            cached["pin_stamps"][filename] = source_stamp
    return cached


def load_execution_dispatch(path: Path) -> Mapping[str, Any]:
    """Read-only cached parent; callers must not mutate the returned mapping."""
    return _dispatch(path)["value"]


def _child_error(child: Mapping[str, Any], parent: Mapping[str, Any],
                 parent_path: Path, candidate: Mapping[str, Any] | None = None) -> None:
    value = parent["value"]
    scenes = child.get("scenes")
    if (child.get("artifact_kind") != EXECUTION_KIND or child.get("execution_only") is not True
            or child.get("publication_ready") is not False or not _self_hash(child)
            or child.get("parent_dispatch_hash") != value["artifact_hash"]
            or Path(str(child.get("parent_dispatch_path") or "")).resolve() != parent_path.resolve()
            or child.get("config_hash") != value["config_hash"]
            or child.get("source_roots") != value["source_roots"]
            or child.get("scope") != value["scope"]
            or not isinstance(scenes, list) or len(scenes) != 1):
        raise ValueError("execution child has an invalid parent/source/config identity")
    scene = scenes[0]
    if not isinstance(scene, Mapping) or str(scene.get("scene_path")) not in parent["paths"]:
        raise ValueError("execution child scene is outside the dispatch selection")
    if candidate is not None and scene != candidate:
        raise ValueError("execution child candidate/task/route records differ from the global plan")


def build_execution_plan(
    dispatch: Mapping[str, Any], candidate: Mapping[str, Any], *, dispatch_path: Path,
) -> dict[str, Any]:
    """Freeze one candidate and its exact routes under a verified parent dispatch."""
    parent = _dispatch(dispatch_path)
    saved = parent["value"]
    # The on-disk self-hash-valid parent is authoritative. Callers commonly hold
    # their own copy; comparing its already-frozen hash is O(1), not O(N scenes).
    if dispatch.get("artifact_hash") != saved["artifact_hash"]:
        raise ValueError("provided dispatch hash differs from its saved identity")
    if str(candidate.get("scene_path")) not in parent["paths"]:
        raise ValueError("execution scene is outside the dispatch selection")
    scene_id = str(candidate.get("scene_spec_id") or "")
    if not scene_id or not candidate.get("task_records"):
        raise ValueError("execution child requires one scene with at least one task")
    roots = saved["source_roots"]
    family_path = str(roots.get("family_library_path") or "")
    family_hash = saved["source_file_hashes"].get(family_path)
    child = {
        "artifact_kind": EXECUTION_KIND,
        "schema_version": saved.get("schema_version", 1),
        "execution_only": True,
        "publication_ready": False,
        "scope": saved["scope"],
        "config_hash": saved["config_hash"],
        "source_roots": canonical_data(roots),
        "parent_dispatch_path": str(dispatch_path.resolve()),
        "parent_dispatch_hash": saved["artifact_hash"],
        "scenes": [canonical_data(candidate)],
        "source_failures": [],
        "provenance": {"sources": ([{"kind": "abstract_family_library", "hash": family_hash}]
                                   if family_hash else [])},
    }
    child["plan_id"] = deterministic_id("svx", {
        "dispatch_hash": saved["artifact_hash"], "scene_spec_id": scene_id,
        "candidate_hash": artifact_hash(candidate),
    })
    child["artifact_hash"] = artifact_hash(child)
    return child


def execution_plan_hash(plan: Mapping[str, Any], scene_spec_id: str) -> Any:
    """Return the original execution identity, retaining legacy plan behavior."""
    bindings = plan.get("execution_bindings")
    if bindings is None:
        return plan.get("artifact_hash")
    if not isinstance(bindings, Mapping) or bindings.get("policy") != BINDING_POLICY:
        raise ValueError("invalid execution binding policy")
    scenes = bindings.get("scenes")
    row = scenes.get(scene_spec_id) if isinstance(scenes, Mapping) else None
    if not isinstance(row, Mapping) or not row.get("child_hash"):
        raise ValueError(f"missing execution binding for scene {scene_spec_id}")
    return row["child_hash"]


def bind_execution_plans(
    final_plan: Mapping[str, Any], dispatch_path: Path, child_paths: Sequence[Path],
) -> dict[str, Any]:
    """Bind an ordinarily constructed global plan to existing execution children."""
    if final_plan.get("artifact_kind") != "scene_validation_plan" or not _self_hash(final_plan):
        raise ValueError("execution bindings require a self-hash-valid ordinary global plan")
    parent = _dispatch(dispatch_path)
    scenes: dict[str, Any] = {}
    for path in child_paths:
        path = Path(path).resolve()
        child = _load(path)
        _child_error(child, parent, dispatch_path)
        scene_id = str(child["scenes"][0]["scene_spec_id"])
        if scene_id in scenes:
            raise ValueError(f"duplicate execution child for {scene_id}")
        scenes[scene_id] = {"child_path": str(path), "child_hash": child["artifact_hash"]}
    result = dict(final_plan)
    result.pop("artifact_hash", None)
    result["execution_bindings"] = {
        "policy": BINDING_POLICY, "dispatch_path": str(dispatch_path.resolve()),
        "dispatch_hash": parent["value"]["artifact_hash"], "scenes": scenes,
    }
    result["artifact_hash"] = artifact_hash(result)
    errors = verify_execution_bindings(result)
    if errors:
        raise ValueError("invalid execution bindings: " + "; ".join(errors[:20]))
    return result


def verify_execution_bindings(plan: Mapping[str, Any]) -> list[str]:
    """Check parent pins, closed selection, and every child's original bytes/identity."""
    if plan.get("artifact_kind") != EXECUTION_KIND and "execution_bindings" not in plan:
        return []
    try:
        if plan.get("artifact_kind") == EXECUTION_KIND:
            path = Path(str(plan.get("parent_dispatch_path") or ""))
            _child_error(plan, _dispatch(path), path)
            return []
        binding = plan.get("execution_bindings")
        if (not isinstance(binding, Mapping) or binding.get("policy") != BINDING_POLICY
                or not isinstance(binding.get("scenes"), Mapping)
                or plan.get("artifact_kind") != "scene_validation_plan" or not _self_hash(plan)):
            raise ValueError("invalid global execution binding envelope")
        path = Path(str(binding.get("dispatch_path") or ""))
        parent = _dispatch(path)
        dispatch = parent["value"]
        if (binding.get("dispatch_hash") != dispatch["artifact_hash"]
                or plan.get("config_hash") != dispatch["config_hash"]
                or plan.get("source_roots") != dispatch["source_roots"]):
            raise ValueError("global plan differs from its parent dispatch source/config identity")
        global_scenes = plan.get("scenes", [])
        by_id = {str(row["scene_spec_id"]): row for row in global_scenes}
        global_paths = {str(row.get("scene_path")) for row in global_scenes}
        if (len(by_id) != len(global_scenes) or set(by_id) != set(binding["scenes"])
                or not global_paths.issubset(parent["paths"])
                or len(global_paths) != len(global_scenes)):
            raise ValueError("dispatch/global/child scene selections are not closed and identical")
        omitted = parent["paths"] - global_paths
        if len(omitted) != int((plan.get("summary") or {}).get("accounted_no_task_scene_count", 0)):
            raise ValueError("omitted dispatch scenes do not match the accounted no-task count")
        if omitted:
            from .core import strict_source_join

            roots = dispatch["source_roots"]
            for scene_path in sorted(omitted):
                joined = strict_source_join(
                    Path(scene_path), realization_dir=Path(str(roots["realization_dir"])),
                    unit_dir=Path(str(roots["unit_dir"])), task_dir=Path(str(roots["task_dir"])),
                    route_dir=None,
                )
                if joined["tasks"]:
                    raise ValueError(f"omitted dispatch scene still contains M13 tasks: {scene_path}")
        if dispatch["scope"] == "diagnostic" and not plan.get("execution_only"):
            raise ValueError("diagnostic dispatch may not become a publishable full plan")
        if dispatch["scope"] == "full" and (
                plan.get("scope") != "full" or plan.get("source_failures")
                or not isinstance(plan.get("coverage"), Mapping)
                or plan["coverage"].get("complete") is not True):
            raise ValueError("global execution plan has not passed ordinary full-plan source/coverage gates")
        for scene_id, descriptor in binding["scenes"].items():
            if not isinstance(descriptor, Mapping):
                raise ValueError(f"invalid child descriptor for {scene_id}")
            child = _load(Path(str(descriptor.get("child_path") or "")))
            if child.get("artifact_hash") != descriptor.get("child_hash"):
                raise ValueError(f"execution child hash changed for {scene_id}")
            _child_error(child, parent, path, by_id[scene_id])
        return []
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as error:
        return [f"execution bindings: {error}"]
