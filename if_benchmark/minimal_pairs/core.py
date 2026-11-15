"""Milestone 14 minimal-pair validation and matrix-ready IndexIF.

The subsystem is simulator-independent.  It joins frozen M10, M12, and M13
artifacts, proves pairwise single-D-code differences inside one initial
observation, groups complete conditional contrast sets, and exposes pure
numeric IndexIF estimators for later rollout consumers.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from itertools import combinations
import math
import multiprocessing
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence

import yaml

from if_benchmark.schema import artifact_hash, build_provenance, deterministic_id
from if_benchmark.schema.cli import validate_document
from if_benchmark.schema.serialization import canonical_data


SCHEMA_VERSION = 1
MINIMAL_PAIR_VERSION = 3
INDEX_IF_VERSION = 1
RESIDUAL_KEYS = ("failure", "other", "ambiguous")
AXIS_ORDER = ("D1a", "D1b", "D1c", "D2a", "D2b", "D4")
DESCRIPTOR_AXIS = {
    "attribute": "D1a",
    "spatial": "D1b",
    "ordinal": "D1c",
}

DEFAULT_SCENE_SPEC_DIR = Path("if_benchmark/data/scene_specs")
DEFAULT_REALIZATION_DIR = Path("if_benchmark/data/realized_families")
DEFAULT_TASK_DIR = Path("if_benchmark/data/tasks")
DEFAULT_OUTPUT_DIR = Path("if_benchmark/data/minimal_pairs")
DEFAULT_PROCESS_WORKERS = min(16, os.cpu_count() or 1)
SCENE_BATCH_SIZE = 16

MINIMAL_PAIRS_FILE = "minimal_pairs.yml"
AUDIT_FILE = "minimal_pair_audit.yml"
INVALID_REPORT_FILE = "invalid_pair_report.yml"
PUBLICATION_POLICY = "matrix_ready_contrast_subset_with_audited_exclusions_v1"
_PUBLISHABLE_EXCLUSION_CODES = frozenset(
    {
        "upstream_task_missing",
        "contrast_set_task_missing",
        "contrast_set_valid_pairs_incomplete",
    }
)

CONFIG = {
    "schema_version": SCHEMA_VERSION,
    "minimal_pair_version": MINIMAL_PAIR_VERSION,
    "index_if_version": INDEX_IF_VERSION,
    "axis_order": list(AXIS_ORDER),
    "residual_keys": list(RESIDUAL_KEYS),
    "contrast_policy": "fixed_non_axis_context_complete_clique_v1",
    "task_endpoint_policy": "required",
    "instruction_style_policy": "scene_shared",
    "style_failure_requires_complete_tasks": True,
    "failure_attribution_policy": "merge_m13_instruction_and_checker_reports_v1",
    "publication_policy": PUBLICATION_POLICY,
    "publishable_exclusion_codes": sorted(_PUBLISHABLE_EXCLUSION_CODES),
}


class MinimalPairError(ValueError):
    """Raised when an M14 input or numeric contract is invalid."""


@dataclass(frozen=True)
class AxisEdgeResult:
    """Structured semantic difference between two ConcreteGoals."""

    actual_axes: tuple[str, ...]
    actual_dcodes: tuple[str, ...]
    unmapped_semantic_deltas: tuple[str, ...]
    left_projections: Mapping[str, Any]
    right_projections: Mapping[str, Any]

    @property
    def is_single_axis(self) -> bool:
        return (
            len(self.actual_dcodes) == 1
            and not self.unmapped_semantic_deltas
        )

    def to_record(self) -> dict[str, Any]:
        return canonical_data(
            {
                "actual_axes": list(self.actual_axes),
                "actual_dcodes": list(self.actual_dcodes),
                "unmapped_semantic_deltas": list(
                    self.unmapped_semantic_deltas
                ),
            }
        )


def _load_yaml(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = yaml.load(stream, Loader=yaml.CSafeLoader)
    except (OSError, yaml.YAMLError):
        return None
    return value if isinstance(value, dict) else None


def _write_yaml(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            yaml.dump(
                value,
                stream,
                Dumper=yaml.CSafeDumper,
                sort_keys=True,
                allow_unicode=True,
                default_flow_style=False,
                width=100,
            )
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _self_hash_valid(value: Mapping[str, Any], key: str = "artifact_hash") -> bool:
    expected = value.get(key)
    return isinstance(expected, str) and expected == artifact_hash(
        value, omit_keys=(key,)
    )


def _axis_subtypes(
    left_axes: Sequence[Mapping[str, Any]] | None,
    right_axes: Sequence[Mapping[str, Any]] | None,
) -> tuple[dict[str, str], list[str]]:
    values: dict[str, set[str]] = defaultdict(set)
    for row in [*(left_axes or []), *(right_axes or [])]:
        axis = str(row.get("axis", ""))
        subtype = str(row.get("subtype", ""))
        if axis:
            values[axis].add(subtype)
    result: dict[str, str] = {}
    failures: list[str] = []
    for axis, subtypes in sorted(values.items()):
        if len(subtypes) != 1 or "" in subtypes:
            failures.append(f"ambiguous_axis_subtype:{axis}")
        else:
            result[axis] = next(iter(subtypes))
    return result, failures


def _descriptor_projection(goal: Mapping[str, Any], kind: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for slot, bundle in sorted((goal.get("descriptors") or {}).items()):
        if kind not in bundle:
            continue
        result.append(
            {
                "slot": str(slot),
                "entity": goal.get(slot),
                "descriptor": canonical_data(bundle[kind]),
            }
        )
    return result


def _axis_projections(goal: Mapping[str, Any]) -> dict[str, Any]:
    state = goal.get("state")
    relation = goal.get("relation")
    return {
        "D1a": _descriptor_projection(goal, "attribute"),
        "D1b": _descriptor_projection(goal, "spatial"),
        "D1c": _descriptor_projection(goal, "ordinal"),
        "D2a": {
            "state": state,
            "verb": goal.get("verb") if state is not None else None,
            "params": (
                canonical_data(goal.get("params") or {})
                if state is not None
                else {}
            ),
        },
        "D2b": {
            "relation": relation,
            "verb": goal.get("verb") if relation is not None else None,
        },
        "D4": {"arm": goal.get("arm")},
    }


def _semantic_skeleton(goal: Mapping[str, Any]) -> dict[str, Any]:
    skeleton = canonical_data(dict(goal))
    descriptors = skeleton.pop("descriptors", {}) or {}
    skeleton["arm"] = "<D4>"
    if skeleton.get("state") is not None:
        skeleton["state"] = "<D2a>"
        skeleton["verb"] = "<D2a>"
        skeleton["params"] = "<D2a>"
    if skeleton.get("relation") is not None:
        skeleton["relation"] = "<D2b>"
        skeleton["verb"] = "<D2b>"
    for slot, bundle in descriptors.items():
        axes = {
            DESCRIPTOR_AXIS[kind]
            for kind in bundle
            if kind in DESCRIPTOR_AXIS
        }
        if len(axes) == 1:
            skeleton[str(slot)] = f"<{next(iter(axes))}:{slot}>"
        elif axes:
            skeleton[str(slot)] = f"<MULTI_D1:{slot}>"
    return skeleton


def _difference_paths(left: Any, right: Any, path: str = "goal") -> list[str]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        result: list[str] = []
        for key in sorted(set(left) | set(right), key=str):
            child = f"{path}.{key}"
            if key not in left or key not in right:
                result.append(child)
            else:
                result.extend(_difference_paths(left[key], right[key], child))
        return result
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [path]
        result = []
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            result.extend(
                _difference_paths(
                    left_item, right_item, f"{path}[{index}]"
                )
            )
        return result
    return [] if left == right else [path]


def edge_axis(
    left_goal: Mapping[str, Any],
    right_goal: Mapping[str, Any],
    *,
    left_intended_axes: Sequence[Mapping[str, Any]] | None = None,
    right_intended_axes: Sequence[Mapping[str, Any]] | None = None,
) -> AxisEdgeResult:
    """Return every changed D-code and every unassigned semantic delta."""

    left_projection = _axis_projections(left_goal)
    right_projection = _axis_projections(right_goal)
    changed_axes = tuple(
        axis
        for axis in AXIS_ORDER
        if left_projection[axis] != right_projection[axis]
    )
    subtypes, subtype_failures = _axis_subtypes(
        left_intended_axes, right_intended_axes
    )
    dcodes: list[str] = []
    unmapped = list(subtype_failures)
    for axis in changed_axes:
        subtype = subtypes.get(axis)
        if subtype is None:
            unmapped.append(f"missing_axis_subtype:{axis}")
        else:
            dcodes.append(f"{axis}.{subtype}")
    unmapped.extend(
        _difference_paths(
            _semantic_skeleton(left_goal),
            _semantic_skeleton(right_goal),
        )
    )
    return AxisEdgeResult(
        actual_axes=changed_axes,
        actual_dcodes=tuple(dcodes),
        unmapped_semantic_deltas=tuple(sorted(set(unmapped))),
        left_projections=canonical_data(left_projection),
        right_projections=canonical_data(right_projection),
    )


def _dcode(edge: Mapping[str, Any]) -> str:
    axis = str(edge.get("axis", ""))
    subtype = str(edge.get("subtype", ""))
    return f"{axis}.{subtype}" if subtype else axis


def _context_signature(
    goal: Mapping[str, Any],
    *,
    target_axis: str,
) -> str:
    projections = _axis_projections(goal)
    context = {
        "non_axis_projections": {
            axis: projections[axis]
            for axis in AXIS_ORDER
            if axis != target_axis
        },
        "semantic_skeleton": _semantic_skeleton(goal),
    }
    return artifact_hash(context)


def _selected_style(task_artifact: Mapping[str, Any]) -> str | None:
    task = task_artifact.get("task_record") or {}
    instruction = task.get("instruction") or {}
    selected = instruction.get("selected") or {}
    return (
        str(selected.get("style_id"))
        if selected.get("style_id") is not None
        else None
    )


def _task_endpoint(
    task_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    task = task_artifact["task_record"]
    instruction = task["instruction"]["selected"]
    return {
        "goal_variant_id": str(task["goal_variant_id"]),
        "task_record_id": str(task["id"]),
        "task_artifact_hash": str(task_artifact["artifact_hash"]),
        "goal_hash": str(task["goal_provenance"]["goal_hash"]),
        "instruction_hash": artifact_hash(instruction),
        "checker_hash": artifact_hash(task["checker_config"]),
        "style_id": str(instruction["style_id"]),
    }


def _initial_observation_failures(
    scene_artifact: Mapping[str, Any],
) -> list[dict[str, str]]:
    scene = scene_artifact.get("scene_spec") or {}
    observation = scene.get("initial_observation") or {}
    failures: list[dict[str, str]] = []
    state = {
        "actor": scene.get("actor"),
        "camera": scene.get("camera"),
        "entities": scene.get("entities"),
    }
    if observation.get("variant_invariant") is not True:
        failures.append(
            {
                "code": "initial_observation_not_variant_invariant",
                "stage": "layout",
                "message": "SceneSpec initial observation is not variant-invariant",
            }
        )
    if observation.get("scene_state_hash") != artifact_hash(state):
        failures.append(
            {
                "code": "initial_observation_hash_mismatch",
                "stage": "layout",
                "message": "SceneSpec state hash does not match actor/camera/entities",
            }
        )
    return failures


def _descriptor_scene_failures(
    goal: Mapping[str, Any],
    scene: Mapping[str, Any],
) -> list[dict[str, str]]:
    """Check scene-resolvable descriptor facts without perception."""

    failures: list[dict[str, str]] = []
    entities = scene.get("entities") or {}
    for slot, bundle in (goal.get("descriptors") or {}).items():
        subject = goal.get(slot)
        if subject not in entities:
            failures.append(
                {
                    "code": "descriptor_entity_missing",
                    "stage": "layout",
                    "message": f"descriptor slot {slot} entity is absent from SceneSpec",
                }
            )
            continue
        for kind in ("spatial", "ordinal"):
            descriptor = bundle.get(kind)
            if not isinstance(descriptor, Mapping):
                continue
            reference = descriptor.get("reference")
            if reference is not None and reference not in entities:
                failures.append(
                    {
                        "code": "descriptor_reference_missing",
                        "stage": "layout",
                        "message": f"{kind} reference {reference!r} is absent from SceneSpec",
                    }
                )
            if kind != "spatial" or reference not in entities:
                continue
            subject_row = entities[subject]
            reference_row = entities[reference]
            sx, sy = subject_row["pose"]["position"][:2]
            rx, ry = reference_row["pose"]["position"][:2]
            for predicate in descriptor.get("predicates", []):
                valid = {
                    "left_of": sx < rx,
                    "right_of": sx > rx,
                    "in_front_of": sy < ry,
                    "behind": sy > ry,
                    "in": subject_row.get("unit_id")
                    == reference_row.get("unit_id"),
                    "on": subject_row.get("unit_id")
                    == reference_row.get("unit_id"),
                }.get(str(predicate), True)
                if not valid:
                    failures.append(
                        {
                            "code": "descriptor_scene_fact_mismatch",
                            "stage": "layout",
                            "message": f"{subject} does not satisfy {predicate}({reference})",
                        }
                    )
    return failures


def _task_failures(
    task_artifact: Mapping[str, Any] | None,
    *,
    scene_artifact: Mapping[str, Any],
    realized_artifact: Mapping[str, Any],
    variant: Mapping[str, Any],
) -> list[dict[str, str]]:
    if task_artifact is None:
        return [
            {
                "code": "upstream_task_missing",
                "stage": "task",
                "message": "M13 task endpoint is missing",
            }
        ]
    task = task_artifact.get("task_record") or {}
    scene = scene_artifact["scene_spec"]
    realized = realized_artifact["realized_scene_family"]
    expected = {
        "artifact_kind": task_artifact.get("artifact_kind")
        == "rendered_task_record",
        "artifact_hash": _self_hash_valid(task_artifact),
        "preflight": task_artifact.get("preflight", {}).get("status")
        == "passed",
        "scene_spec_id": task_artifact.get("scene_spec_id") == scene.get("id")
        and task.get("scene_spec_id") == scene.get("id"),
        "realized_family_id": task_artifact.get("realized_family_id")
        == realized.get("id")
        and task.get("realized_family_id") == realized.get("id"),
        "goal_variant_id": task_artifact.get("goal_variant_id")
        == variant.get("id")
        and task.get("goal_variant_id") == variant.get("id"),
        "source_scene_hash": task_artifact.get("source_scene_hash")
        == scene_artifact.get("artifact_hash"),
        "source_realization_hash": task_artifact.get("source_realization_hash")
        == realized_artifact.get("artifact_hash"),
        "goal": canonical_data(task.get("goal"))
        == canonical_data(variant.get("goal")),
        "assignment_id": task.get("goal_provenance", {}).get("assignment_id")
        == realized.get("assignment_id"),
        "goal_hash": task.get("goal_provenance", {}).get("goal_hash")
        == artifact_hash(variant.get("goal")),
    }
    return [
        {
            "code": f"task_{key}_mismatch",
            "stage": "task" if key not in {"source_scene_hash"} else "source",
            "message": f"M13 task endpoint failed {key} alignment",
        }
        for key, passed in expected.items()
        if not passed
    ]


def _pair_failure(
    *,
    code: str,
    stage: str,
    message: str,
    scene_id: str,
    realized_id: str,
    left_id: str,
    right_id: str,
    intended_dcode: str | None = None,
    actual_dcodes: Sequence[str] = (),
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "code": code,
        "stage": stage,
        "message": message,
        "scene_spec_id": scene_id,
        "realized_family_id": realized_id,
        "from_goal_variant_id": left_id,
        "to_goal_variant_id": right_id,
        "actual_dcodes": list(actual_dcodes),
    }
    if intended_dcode is not None:
        row["intended_dcode"] = intended_dcode
    if details:
        row["details"] = canonical_data(dict(details))
    return canonical_data(row)


def _source_failures(
    scene_artifact: Mapping[str, Any],
    realized_artifact: Mapping[str, Any],
) -> list[dict[str, str]]:
    scene = scene_artifact.get("scene_spec") or {}
    realized = realized_artifact.get("realized_scene_family") or {}
    from if_benchmark.layout.manipulation import valid_domain
    checks = {
        "scene_artifact_kind": scene_artifact.get("artifact_kind")
        == "solved_scene_spec",
        "scene_self_hash": _self_hash_valid(scene_artifact),
        "scene_preflight": scene_artifact.get("preflight", {}).get("status")
        == "passed",
        "realization_artifact_kind": realized_artifact.get("artifact_kind")
        == "realized_scene_family",
        "realization_self_hash": _self_hash_valid(realized_artifact),
        "realization_preflight": realized_artifact.get("preflight", {}).get(
            "status"
        )
        == "passed",
        "realized_family_id": scene.get("realized_family_id")
        == realized.get("id")
        == scene_artifact.get("realized_family_id"),
        "source_realization_hash": scene_artifact.get(
            "source_realization_hash"
        )
        == realized_artifact.get("artifact_hash"),
        "goal_variant_ids": valid_domain(realized, scene),
    }
    return [
        {
            "code": f"source_{key}_mismatch",
            "stage": "source",
            "message": f"M14 source join failed {key}",
        }
        for key, passed in checks.items()
        if not passed
    ]


def _components(
    edges: Sequence[Mapping[str, Any]],
) -> list[set[str]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        left = str(edge["from"])
        right = str(edge["to"])
        adjacency[left].add(right)
        adjacency[right].add(left)
    unseen = set(adjacency)
    result: list[set[str]] = []
    while unseen:
        stack = [min(unseen)]
        component: set[str] = set()
        while stack:
            value = stack.pop()
            if value in component:
                continue
            component.add(value)
            unseen.discard(value)
            stack.extend(sorted(adjacency[value] - component, reverse=True))
        result.append(component)
    return sorted(result, key=lambda values: sorted(values))


def build_contrast_sets(
    *,
    scene_spec: Mapping[str, Any],
    realized_family: Mapping[str, Any],
    task_artifacts: Mapping[str, Mapping[str, Any]],
    pair_records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build complete matrix-ready conditional axis contrast sets."""

    from if_benchmark.layout.manipulation import selected_realized
    realized_family = selected_realized(realized_family, scene_spec)
    scene_id = str(scene_spec["id"])
    realized_id = str(realized_family["id"])
    variants = {
        str(row["id"]): row for row in realized_family.get("goal_variants", [])
    }
    valid_pairs = {
        frozenset(
            (
                str(row["from_goal_variant_id"]),
                str(row["to_goal_variant_id"]),
            )
        ): row
        for row in pair_records
        if row.get("status") == "valid"
    }
    by_dcode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in realized_family.get("minimal_pair_edges", []):
        by_dcode[_dcode(edge)].append(dict(edge))
    sets: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    observation = scene_spec["initial_observation"]
    for dcode, edges in sorted(by_dcode.items()):
        axis, subtype = dcode.split(".", 1)
        for component in _components(edges):
            ordered = sorted(
                component,
                key=lambda variant_id: (
                    artifact_hash(
                        _axis_projections(variants[variant_id]["goal"])[axis]
                    ),
                    variant_id,
                ),
            )
            context_rows = [
                _context_signature(variants[value]["goal"], target_axis=axis)
                for value in ordered
            ]
            context_hashes = set(context_rows)
            expected_pairs = {
                frozenset(values) for values in combinations(ordered, 2)
            }
            declared_pairs = {
                frozenset((str(edge["from"]), str(edge["to"])))
                for edge in edges
                if edge["from"] in component and edge["to"] in component
            }
            available_pairs = expected_pairs & set(valid_pairs)
            endpoint_tasks = {
                value: task_artifacts.get(value) for value in ordered
            }
            style_ids = {
                _selected_style(task)
                for task in endpoint_tasks.values()
                if task is not None
            }
            set_failures: list[dict[str, Any]] = []
            if len(ordered) < 2:
                set_failures.append(
                    {"code": "contrast_set_too_small", "stage": "gate"}
                )
            if declared_pairs != expected_pairs:
                set_failures.append(
                    {"code": "contrast_set_not_declared_clique", "stage": "family"}
                )
            if available_pairs != expected_pairs:
                set_failures.append(
                    {"code": "contrast_set_valid_pairs_incomplete", "stage": "gate"}
                )
            if len(context_hashes) != 1:
                set_failures.append(
                    {"code": "contrast_set_context_mismatch", "stage": "materialization"}
                )
            tasks_missing = any(
                value is None for value in endpoint_tasks.values()
            )
            if tasks_missing:
                set_failures.append(
                    {"code": "contrast_set_task_missing", "stage": "task"}
                )
            if not tasks_missing and (len(style_ids) != 1 or None in style_ids):
                set_failures.append(
                    {"code": "contrast_set_style_mismatch", "stage": "task"}
                )
            context_hash = (
                next(iter(context_hashes))
                if len(context_hashes) == 1
                else artifact_hash(sorted(context_hashes))
            )
            set_id = deterministic_id(
                "cs",
                {
                    "scene_spec_id": scene_id,
                    "dcode": dcode,
                    "context_signature": context_hash,
                    "members": ordered,
                },
            )
            member_rows = []
            for matrix_index, variant_id in enumerate(ordered):
                task = endpoint_tasks[variant_id]
                member = {
                    "matrix_index": matrix_index,
                    "goal_variant_id": variant_id,
                    "axis_value": canonical_data(
                        _axis_projections(variants[variant_id]["goal"])[axis]
                    ),
                    "intent_goal_hash": artifact_hash(
                        variants[variant_id]["goal"]
                    ),
                }
                if task is not None:
                    member.update(_task_endpoint(task))
                member_rows.append(member)
            pair_ids = sorted(
                str(valid_pairs[pair]["id"])
                for pair in available_pairs
            )
            record = {
                "id": set_id,
                "scene_spec_id": scene_id,
                "realized_family_id": realized_id,
                "layout_seed": int(scene_spec["layout_seed"]),
                "observation_id": str(observation["observation_id"]),
                "scene_state_hash": str(observation["scene_state_hash"]),
                "axis": axis,
                "subtype": subtype,
                "dcode": dcode,
                "context_signature": context_hash,
                "k": len(ordered),
                "members": member_rows,
                "pair_ids": pair_ids,
                "matrix_ready": not set_failures,
            }
            record["id"] = set_id
            sets.append(canonical_data(record))
            for failure in set_failures:
                failures.append(
                    {
                        "code": failure["code"],
                        "stage": failure["stage"],
                        "message": f"contrast set {set_id} is not matrix-ready",
                        "scene_spec_id": scene_id,
                        "realized_family_id": realized_id,
                        "contrast_set_id": set_id,
                        "dcode": dcode,
                        "goal_variant_ids": ordered,
                    }
                )
    return sets, canonical_data(failures)


def _finite_nonnegative(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise MinimalPairError(f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise MinimalPairError(f"{label} must be numeric") from exc
    if not math.isfinite(result) or result < 0:
        raise MinimalPairError(f"{label} must be finite and non-negative")
    return result


def _numeric_matrix(values: Sequence[Sequence[Any]]) -> list[list[float]]:
    rows = [list(row) for row in values]
    if len(rows) < 2 or any(len(row) != len(rows) for row in rows):
        raise MinimalPairError("intent probability matrix must be square with K >= 2")
    return [
        [
            _finite_nonnegative(value, label=f"matrix[{i}][{j}]")
            for j, value in enumerate(row)
        ]
        for i, row in enumerate(rows)
    ]


def _residual_rows(
    residuals: Sequence[Mapping[str, Any]] | None,
    *,
    k: int,
) -> list[dict[str, float]]:
    if residuals is None:
        return [{key: 0.0 for key in RESIDUAL_KEYS} for _ in range(k)]
    if len(residuals) != k:
        raise MinimalPairError("residual row count must equal K")
    result = []
    for index, row in enumerate(residuals):
        extra = sorted(set(row) - set(RESIDUAL_KEYS))
        if extra:
            raise MinimalPairError(f"unsupported residual keys: {extra}")
        result.append(
            {
                key: _finite_nonnegative(
                    row.get(key, 0.0), label=f"residuals[{index}].{key}"
                )
                for key in RESIDUAL_KEYS
            }
        )
    return result


def compute_index_if(
    probabilities: Sequence[Sequence[Any]],
    residuals: Sequence[Mapping[str, Any]] | None = None,
    *,
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    """Compute the point-estimate IndexIF from unconditional intent mass."""

    matrix = _numeric_matrix(probabilities)
    k = len(matrix)
    residual_rows = _residual_rows(residuals, k=k)
    for index, row in enumerate(matrix):
        total = sum(row) + sum(residual_rows[index].values())
        if abs(total - 1.0) > tolerance:
            raise MinimalPairError(
                f"row {index} probability mass is {total}, expected 1"
            )
    row_margins = [
        row[index] - max(value for j, value in enumerate(row) if j != index)
        for index, row in enumerate(matrix)
    ]
    column_margins = [
        matrix[index][index]
        - max(matrix[row][index] for row in range(k) if row != index)
        for index in range(k)
    ]
    action_advantage = sum(row_margins) / k
    instruction_advantage = sum(column_margins) / k
    index_if = 0.5 * (action_advantage + instruction_advantage)
    return canonical_data(
        {
            "index_if_version": INDEX_IF_VERSION,
            "k": k,
            "probability_matrix": matrix,
            "residual_rates": residual_rows,
            "row_margins": row_margins,
            "column_margins": column_margins,
            "correct_action_advantage": action_advantage,
            "correct_instruction_advantage": instruction_advantage,
            "index_if": index_if,
        }
    )


def compute_index_if_from_counts(
    intent_counts: Sequence[Sequence[Any]],
    residual_counts: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Normalize all rollout outcomes per instruction and compute IndexIF."""

    counts = _numeric_matrix(intent_counts)
    k = len(counts)
    residual = _residual_rows(residual_counts, k=k)
    totals = [
        sum(counts[index]) + sum(residual[index].values())
        for index in range(k)
    ]
    if any(total <= 0 for total in totals):
        raise MinimalPairError("every instruction row must have at least one rollout")
    probabilities = [
        [value / totals[index] for value in row]
        for index, row in enumerate(counts)
    ]
    residual_rates = [
        {key: value / totals[index] for key, value in row.items()}
        for index, row in enumerate(residual)
    ]
    result = compute_index_if(probabilities, residual_rates, tolerance=1e-8)
    result["intent_counts"] = counts
    result["residual_counts"] = residual
    result["row_sample_counts"] = totals
    return canonical_data(result)


def aggregate_scene_axis_index(
    set_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Equal-weight complete contrast-set IndexIF results by scene and D-code."""

    if not set_results:
        raise MinimalPairError("at least one contrast-set result is required")
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in set_results:
        if row.get("matrix_ready") is False:
            raise MinimalPairError("cannot aggregate a non-matrix-ready contrast set")
        scene_id = str(row.get("scene_spec_id", ""))
        dcode = str(row.get("dcode", ""))
        if not scene_id or not dcode or "index_if" not in row:
            raise MinimalPairError(
                "each set result requires scene_spec_id, dcode, and index_if"
            )
        index_value = float(row["index_if"])
        if not math.isfinite(index_value) or not -1.0 <= index_value <= 1.0:
            raise MinimalPairError("index_if must be finite and within [-1, 1]")
        groups[(scene_id, dcode)].append(row)
    aggregates = []
    for (scene_id, dcode), rows in sorted(groups.items()):
        values = [float(row["index_if"]) for row in rows]
        aggregates.append(
            {
                "scene_spec_id": scene_id,
                "dcode": dcode,
                "contrast_set_count": len(rows),
                "contrast_set_ids": sorted(
                    str(row.get("contrast_set_id", row.get("id", "")))
                    for row in rows
                ),
                "index_if": sum(values) / len(values),
            }
        )
    return {
        "aggregation": "uniform_over_contrast_sets",
        "scene_axis_results": canonical_data(aggregates),
    }


def _m13_failure_index(task_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for name in ("instruction_report.yml", "checker_alignment_report.yml"):
        report = _load_yaml(task_dir / name) or {}
        for row in report.get("failures", []):
            key = (
                str(row.get("scene_spec_id", "")),
                str(row.get("goal_variant_id", "")),
            )
            result.setdefault(key, row)
    return result


def _upstream_status(
    scene_dir: Path,
    realization_dir: Path,
    task_dir: Path,
) -> tuple[dict[str, str], bool]:
    layout = _load_yaml(scene_dir / "layout_report.yml") or {}
    task = _load_yaml(task_dir / "instruction_report.yml") or {}
    realization = _load_yaml(
        realization_dir / "realization_audit.yml"
    ) or {}
    hashes = {
        "realized_families": str(realization.get("artifact_set_hash", "")),
        "scene_specs": str(layout.get("artifact_set_hash", "")),
        "tasks": str(task.get("artifact_set_hash", "")),
    }
    return hashes, bool(layout.get("publication_ready")) and bool(
        task.get("publication_ready")
    )


def _scene_paths(scene_dir: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in scene_dir.glob("rsf_*/*.yml")
            if path.name not in {"layout_report.yml", "layout_unsat_report.yml"}
        ),
        key=lambda path: (path.parent.name, path.stem),
    )


def _assemble_artifacts(
    *,
    source_hashes: Mapping[str, str],
    upstream_publication_ready: bool,
    scene_spec_count: int,
    expected_edge_count: int,
    internal_failure_count: int,
    pairs: Sequence[Mapping[str, Any]],
    valid_sets: Sequence[Mapping[str, Any]],
    invalid_sets: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
    config_hash: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build the three global M14 documents in the parent process."""

    pair_axis_counts = Counter(row["intended_axis"] for row in pairs)
    pair_dcode_counts = Counter(row["intended_dcode"] for row in pairs)
    set_dcode_counts = Counter(row["dcode"] for row in valid_sets)
    set_k_counts = Counter(
        str(row["k"]) for row in [*valid_sets, *invalid_sets]
    )
    failure_code_counts = Counter(str(row["code"]) for row in failures)
    failure_stage_counts = Counter(str(row["stage"]) for row in failures)
    summary = {
        "scene_spec_count": scene_spec_count,
        "expected_pair_count": expected_edge_count,
        "valid_pair_count": len(pairs),
        "invalid_pair_count": expected_edge_count - len(pairs),
        "contrast_set_count": len(valid_sets) + len(invalid_sets),
        "matrix_ready_contrast_set_count": len(valid_sets),
        "incomplete_contrast_set_count": len(invalid_sets),
        "failure_record_count": len(failures),
        "internal_failure_count": internal_failure_count,
    }
    implementation_complete = internal_failure_count == 0
    blocking_failure_codes = sorted(
        set(failure_code_counts) - _PUBLISHABLE_EXCLUSION_CODES
    )
    coverage_complete = (
        implementation_complete
        and summary["invalid_pair_count"] == 0
        and not invalid_sets
        and not failures
    )
    stage_acceptance_ready = (
        implementation_complete
        and bool(valid_sets)
        and not blocking_failure_codes
    )
    publication_ready = stage_acceptance_ready and upstream_publication_ready
    common = {
        "schema_version": SCHEMA_VERSION,
        "minimal_pair_version": MINIMAL_PAIR_VERSION,
        "index_if_version": INDEX_IF_VERSION,
        "config_hash": config_hash,
        "source_artifact_set_hashes": dict(source_hashes),
        "implementation_complete": implementation_complete,
        "publication_policy": PUBLICATION_POLICY,
        "publication_scope": "matrix_ready_contrast_sets",
        "coverage_complete": coverage_complete,
        "stage_acceptance_ready": stage_acceptance_ready,
        "publication_ready": publication_ready,
    }
    artifact = {
        **common,
        "artifact_kind": "validated_minimal_pairs",
        "pairs": sorted(pairs, key=lambda row: row["id"]),
        "contrast_sets": sorted(valid_sets, key=lambda row: row["id"]),
        "provenance": build_provenance(
            "if_benchmark.minimal_pairs",
            config_hash,
            [
                {"kind": kind, "hash": value}
                for kind, value in source_hashes.items()
                if value
            ],
            metadata={
                "contrast_policy": CONFIG["contrast_policy"],
                "index_if_version": INDEX_IF_VERSION,
            },
        ),
    }
    artifact["artifact_hash"] = artifact_hash(artifact)
    audit = {
        **common,
        "artifact_kind": "minimal_pair_audit",
        "artifact_set_hash": artifact["artifact_hash"],
        "summary": summary,
        "valid_pair_axis_counts": dict(sorted(pair_axis_counts.items())),
        "valid_pair_dcode_counts": dict(sorted(pair_dcode_counts.items())),
        "matrix_ready_set_dcode_counts": dict(sorted(set_dcode_counts.items())),
        "contrast_set_k_distribution": dict(sorted(set_k_counts.items())),
        "failure_code_counts": dict(sorted(failure_code_counts.items())),
        "failure_stage_counts": dict(sorted(failure_stage_counts.items())),
        "publishable_exclusion_codes": sorted(_PUBLISHABLE_EXCLUSION_CODES),
        "blocking_failure_codes": blocking_failure_codes,
    }
    audit["audit_hash"] = artifact_hash(audit)
    invalid = {
        **common,
        "artifact_kind": "invalid_pair_report",
        "summary": {
            "invalid_pair_count": summary["invalid_pair_count"],
            "incomplete_contrast_set_count": len(invalid_sets),
            "failure_record_count": len(failures),
        },
        "failure_code_counts": dict(sorted(failure_code_counts.items())),
        "failure_stage_counts": dict(sorted(failure_stage_counts.items())),
        "failures": canonical_data(failures),
        "incomplete_contrast_sets": canonical_data(invalid_sets),
    }
    invalid["report_hash"] = artifact_hash(invalid)
    return artifact, audit, invalid


_M14_WORKER_FAILURE_INDEX: Mapping[tuple[str, str], Mapping[str, Any]] = {}
_M14_WORKER_SOURCE_STATUS: tuple[Mapping[str, str], bool] = ({}, False)
_M14_WORKER_REALIZATION_DIR = Path()
_M14_WORKER_TASK_DIR = Path()


def _init_m14_build_worker(
    failure_index: Mapping[tuple[str, str], Mapping[str, Any]],
    source_status: tuple[Mapping[str, str], bool],
    realization_dir: str,
    task_dir: str,
) -> None:
    global _M14_WORKER_FAILURE_INDEX
    global _M14_WORKER_SOURCE_STATUS
    global _M14_WORKER_REALIZATION_DIR
    global _M14_WORKER_TASK_DIR
    _M14_WORKER_FAILURE_INDEX = failure_index
    _M14_WORKER_SOURCE_STATUS = source_status
    _M14_WORKER_REALIZATION_DIR = Path(realization_dir)
    _M14_WORKER_TASK_DIR = Path(task_dir)


def _validated_worker_count(workers: int) -> int:
    worker_count = int(workers)
    if worker_count < 1:
        raise ValueError("workers must be a positive integer")
    return worker_count


def _fork_context() -> multiprocessing.context.BaseContext | None:
    try:
        return multiprocessing.get_context("fork")
    except ValueError:  # pragma: no cover - non-POSIX safe fallback
        return None


def _scene_batches(scene_paths: Sequence[Path]) -> list[tuple[str, ...]]:
    return [
        tuple(str(path) for path in scene_paths[index : index + SCENE_BATCH_SIZE])
        for index in range(0, len(scene_paths), SCENE_BATCH_SIZE)
    ]


def _build_scene_batch(paths: tuple[str, ...]) -> dict[str, Any]:
    progress_ids: list[str] = []

    def collect_progress(_index: int, _total: int, scene_id: str) -> None:
        progress_ids.append(scene_id)

    artifact, audit, invalid = _build_artifacts(
        scene_spec_dir=Path(paths[0]).parent.parent,
        realization_dir=_M14_WORKER_REALIZATION_DIR,
        task_dir=_M14_WORKER_TASK_DIR,
        progress=collect_progress,
        workers=1,
        _scene_paths_override=[Path(value) for value in paths],
        _failure_index_override=_M14_WORKER_FAILURE_INDEX,
        _source_status_override=_M14_WORKER_SOURCE_STATUS,
    )
    return {
        "progress_ids": progress_ids,
        "pairs": artifact["pairs"],
        "valid_sets": artifact["contrast_sets"],
        "invalid_sets": invalid["incomplete_contrast_sets"],
        "failures": invalid["failures"],
        "summary": audit["summary"],
    }


def _bounded_process_map(
    executor: ProcessPoolExecutor,
    function: Callable[[Any], Any],
    tasks: Sequence[Any],
    *,
    max_in_flight: int,
) -> Iterable[Any]:
    """Yield process results in input order with bounded queued payloads."""

    pending: deque[Future[Any]] = deque()
    next_index = 0
    while next_index < len(tasks) and len(pending) < max_in_flight:
        pending.append(executor.submit(function, tasks[next_index]))
        next_index += 1
    while pending:
        yield pending.popleft().result()
        if next_index < len(tasks):
            pending.append(executor.submit(function, tasks[next_index]))
            next_index += 1


def _build_artifacts_parallel(
    *,
    scene_paths: Sequence[Path],
    realization_dir: Path,
    task_dir: Path,
    source_status: tuple[Mapping[str, str], bool],
    failure_index: Mapping[tuple[str, str], Mapping[str, Any]],
    config_hash: str,
    workers: int,
    progress: Callable[[int, int, str], None] | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    batches = _scene_batches(scene_paths)
    worker_count = min(_validated_worker_count(workers), max(1, len(batches)))
    context = _fork_context()
    if context is None or worker_count == 1:
        return _build_artifacts(
            scene_spec_dir=scene_paths[0].parent.parent if scene_paths else Path(),
            realization_dir=realization_dir,
            task_dir=task_dir,
            progress=progress,
            workers=1,
            _scene_paths_override=scene_paths,
            _failure_index_override=failure_index,
            _source_status_override=source_status,
        )

    pairs: list[dict[str, Any]] = []
    valid_sets: list[dict[str, Any]] = []
    invalid_sets: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    expected_edge_count = 0
    internal_failure_count = 0
    progress_index = 0
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=context,
        initializer=_init_m14_build_worker,
        initargs=(
            failure_index,
            source_status,
            str(realization_dir),
            str(task_dir),
        ),
    ) as executor:
        results = _bounded_process_map(
            executor,
            _build_scene_batch,
            batches,
            max_in_flight=max(1, 2 * worker_count),
        )
        for result in results:
            pairs.extend(result["pairs"])
            valid_sets.extend(result["valid_sets"])
            invalid_sets.extend(result["invalid_sets"])
            failures.extend(result["failures"])
            summary = result["summary"]
            expected_edge_count += int(summary["expected_pair_count"])
            internal_failure_count += int(summary["internal_failure_count"])
            for scene_id in result["progress_ids"]:
                progress_index += 1
                if progress is not None:
                    progress(progress_index, len(scene_paths), scene_id)
    return _assemble_artifacts(
        source_hashes=source_status[0],
        upstream_publication_ready=source_status[1],
        scene_spec_count=len(scene_paths),
        expected_edge_count=expected_edge_count,
        internal_failure_count=internal_failure_count,
        pairs=pairs,
        valid_sets=valid_sets,
        invalid_sets=invalid_sets,
        failures=failures,
        config_hash=config_hash,
    )


def _build_artifacts(
    *,
    scene_spec_dir: Path,
    realization_dir: Path,
    task_dir: Path,
    progress: Callable[[int, int, str], None] | None,
    workers: int = 1,
    _scene_paths_override: Sequence[Path] | None = None,
    _failure_index_override: Mapping[
        tuple[str, str], Mapping[str, Any]
    ] | None = None,
    _source_status_override: tuple[Mapping[str, str], bool] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    _validated_worker_count(workers)
    source_hashes, upstream_publication_ready = (
        _source_status_override
        if _source_status_override is not None
        else _upstream_status(scene_spec_dir, realization_dir, task_dir)
    )
    failure_index = (
        _failure_index_override
        if _failure_index_override is not None
        else _m13_failure_index(task_dir)
    )
    pairs: list[dict[str, Any]] = []
    contrast_sets: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    scene_paths = (
        list(_scene_paths_override)
        if _scene_paths_override is not None
        else _scene_paths(scene_spec_dir)
    )
    expected_edge_count = 0
    internal_failure_count = 0
    config_hash = artifact_hash(CONFIG)
    if workers > 1 and _scene_paths_override is None and scene_paths:
        return _build_artifacts_parallel(
            scene_paths=scene_paths,
            realization_dir=realization_dir,
            task_dir=task_dir,
            source_status=(source_hashes, upstream_publication_ready),
            failure_index=failure_index,
            config_hash=config_hash,
            workers=workers,
            progress=progress,
        )
    for index, scene_path in enumerate(scene_paths, start=1):
        scene_artifact = _load_yaml(scene_path)
        scene_id = scene_path.stem
        if scene_artifact is None:
            failures.append(
                {
                    "code": "scene_artifact_invalid",
                    "stage": "source",
                    "message": f"cannot load {scene_path}",
                    "scene_spec_id": scene_id,
                }
            )
            internal_failure_count += 1
            if progress:
                progress(index, len(scene_paths), scene_id)
            continue
        scene = scene_artifact.get("scene_spec") or {}
        scene_id = str(scene.get("id", scene_id))
        realized_id = str(scene.get("realized_family_id", scene_path.parent.name))
        realized_path = realization_dir / f"{realized_id}.yml"
        realized_artifact = _load_yaml(realized_path)
        if realized_artifact is None:
            failures.append(
                {
                    "code": "realization_artifact_missing",
                    "stage": "source",
                    "message": f"cannot load {realized_path}",
                    "scene_spec_id": scene_id,
                    "realized_family_id": realized_id,
                }
            )
            internal_failure_count += 1
            if progress:
                progress(index, len(scene_paths), scene_id)
            continue
        realized = realized_artifact.get("realized_scene_family") or {}
        from if_benchmark.layout.manipulation import selected_realized, valid_domain
        # Source validation below uses the original artifact; consumers use the
        # explicitly selected, provenance-checked induced task domain.
        if valid_domain(realized, scene):
            realized = selected_realized(realized, scene)
        variants = {
            str(row["id"]): row for row in realized.get("goal_variants", [])
        }
        expected_edges = list(realized.get("minimal_pair_edges", []))
        expected_edge_count += len(expected_edges)
        source_failures = [
            *_source_failures(scene_artifact, realized_artifact),
            *_initial_observation_failures(scene_artifact),
        ]
        if source_failures:
            for edge in expected_edges:
                for failure in source_failures:
                    failures.append(
                        _pair_failure(
                            code=failure["code"],
                            stage=failure["stage"],
                            message=failure["message"],
                            scene_id=scene_id,
                            realized_id=realized_id,
                            left_id=str(edge.get("from", "")),
                            right_id=str(edge.get("to", "")),
                            intended_dcode=_dcode(edge),
                        )
                    )
            internal_failure_count += len(source_failures)
            if progress:
                progress(index, len(scene_paths), scene_id)
            continue
        tasks = {
            variant_id: _load_yaml(task_dir / scene_id / f"{variant_id}.yml")
            for variant_id in variants
        }
        actual_single_edges: dict[frozenset[str], AxisEdgeResult] = {}
        for left_id, right_id in combinations(sorted(variants), 2):
            left = variants[left_id]
            right = variants[right_id]
            actual = edge_axis(
                left["goal"],
                right["goal"],
                left_intended_axes=left.get("intended_axes"),
                right_intended_axes=right.get("intended_axes"),
            )
            if actual.is_single_axis:
                actual_single_edges[frozenset((left_id, right_id))] = actual
        expected_map = {
            frozenset((str(edge["from"]), str(edge["to"]))): edge
            for edge in expected_edges
        }
        for endpoints in sorted(
            set(actual_single_edges) - set(expected_map),
            key=lambda value: sorted(value),
        ):
            left_id, right_id = sorted(endpoints)
            actual = actual_single_edges[endpoints]
            failures.append(
                _pair_failure(
                    code="unexpected_actual_single_axis_edge",
                    stage="family",
                    message="actual single-D-code edge is absent from family contract",
                    scene_id=scene_id,
                    realized_id=realized_id,
                    left_id=left_id,
                    right_id=right_id,
                    actual_dcodes=actual.actual_dcodes,
                )
            )
        scene_pairs: list[dict[str, Any]] = []
        for edge in expected_edges:
            left_id = str(edge["from"])
            right_id = str(edge["to"])
            intended_dcode = _dcode(edge)
            pair_id = deterministic_id(
                "mp",
                {
                    "scene_spec_id": scene_id,
                    "from": left_id,
                    "to": right_id,
                    "intended_dcode": intended_dcode,
                    "scene_hash": scene_artifact["artifact_hash"],
                    "realization_hash": realized_artifact["artifact_hash"],
                },
            )
            pair_failures: list[dict[str, Any]] = []
            if left_id == right_id or left_id not in variants or right_id not in variants:
                pair_failures.append(
                    _pair_failure(
                        code="intended_edge_variant_invalid",
                        stage="family",
                        message="intended edge has missing or identical endpoints",
                        scene_id=scene_id,
                        realized_id=realized_id,
                        left_id=left_id,
                        right_id=right_id,
                        intended_dcode=intended_dcode,
                    )
                )
                actual = AxisEdgeResult((), (), (), {}, {})
            else:
                left = variants[left_id]
                right = variants[right_id]
                actual = edge_axis(
                    left["goal"],
                    right["goal"],
                    left_intended_axes=left.get("intended_axes"),
                    right_intended_axes=right.get("intended_axes"),
                )
                if tuple(actual.actual_dcodes) != (intended_dcode,) or (
                    actual.unmapped_semantic_deltas
                ):
                    pair_failures.append(
                        _pair_failure(
                            code="edge_axis_mismatch",
                            stage="materialization",
                            message="actual semantic edge is not exactly the intended D-code",
                            scene_id=scene_id,
                            realized_id=realized_id,
                            left_id=left_id,
                            right_id=right_id,
                            intended_dcode=intended_dcode,
                            actual_dcodes=actual.actual_dcodes,
                            details={
                                "unmapped_semantic_deltas": list(
                                    actual.unmapped_semantic_deltas
                                )
                            },
                        )
                    )
                for variant_id in (left_id, right_id):
                    for failure in _task_failures(
                        tasks.get(variant_id),
                        scene_artifact=scene_artifact,
                        realized_artifact=realized_artifact,
                        variant=variants[variant_id],
                    ):
                        details = failure_index.get((scene_id, variant_id))
                        pair_failures.append(
                            _pair_failure(
                                code=failure["code"],
                                stage=failure["stage"],
                                message=failure["message"],
                                scene_id=scene_id,
                                realized_id=realized_id,
                                left_id=left_id,
                                right_id=right_id,
                                intended_dcode=intended_dcode,
                                actual_dcodes=actual.actual_dcodes,
                                details=details,
                            )
                        )
                for variant_id in (left_id, right_id):
                    for failure in _descriptor_scene_failures(
                        variants[variant_id]["goal"], scene
                    ):
                        pair_failures.append(
                            _pair_failure(
                                code=failure["code"],
                                stage=failure["stage"],
                                message=failure["message"],
                                scene_id=scene_id,
                                realized_id=realized_id,
                                left_id=left_id,
                                right_id=right_id,
                                intended_dcode=intended_dcode,
                                actual_dcodes=actual.actual_dcodes,
                            )
                        )
                if (
                    tasks.get(left_id) is not None
                    and tasks.get(right_id) is not None
                    and _selected_style(tasks[left_id])
                    != _selected_style(tasks[right_id])
                ):
                    pair_failures.append(
                        _pair_failure(
                            code="instruction_style_mismatch",
                            stage="task",
                            message="minimal-pair endpoints do not share instruction style",
                            scene_id=scene_id,
                            realized_id=realized_id,
                            left_id=left_id,
                            right_id=right_id,
                            intended_dcode=intended_dcode,
                            actual_dcodes=actual.actual_dcodes,
                        )
                    )
            observation = scene["initial_observation"]
            record: dict[str, Any] = {
                "id": pair_id,
                "scene_spec_id": scene_id,
                "realized_family_id": realized_id,
                "layout_seed": int(scene["layout_seed"]),
                "assignment_id": str(realized.get("assignment_id", "")),
                "observation_id": str(observation["observation_id"]),
                "scene_state_hash": str(observation["scene_state_hash"]),
                "from_goal_variant_id": left_id,
                "to_goal_variant_id": right_id,
                "intended_axis": str(edge.get("axis", "")),
                "intended_subtype": str(edge.get("subtype", "")),
                "intended_dcode": intended_dcode,
                **actual.to_record(),
                "status": "invalid" if pair_failures else "valid",
                "source_scene_hash": str(scene_artifact["artifact_hash"]),
                "source_realization_hash": str(realized_artifact["artifact_hash"]),
            }
            if tasks.get(left_id) is not None:
                record["from_endpoint"] = _task_endpoint(tasks[left_id])
            if tasks.get(right_id) is not None:
                record["to_endpoint"] = _task_endpoint(tasks[right_id])
            scene_pairs.append(canonical_data(record))
            if pair_failures:
                failures.extend(pair_failures)
            else:
                pairs.append(canonical_data(record))
        sets, set_failures = build_contrast_sets(
            scene_spec=scene,
            realized_family=realized_artifact["realized_scene_family"],
            task_artifacts={
                key: value for key, value in tasks.items() if value is not None
            },
            pair_records=scene_pairs,
        )
        contrast_sets.extend(sets)
        failures.extend(set_failures)
        if progress:
            progress(index, len(scene_paths), scene_id)

    valid_sets = [row for row in contrast_sets if row["matrix_ready"]]
    invalid_sets = [row for row in contrast_sets if not row["matrix_ready"]]
    return _assemble_artifacts(
        source_hashes=source_hashes,
        upstream_publication_ready=upstream_publication_ready,
        scene_spec_count=len(scene_paths),
        expected_edge_count=expected_edge_count,
        internal_failure_count=internal_failure_count,
        pairs=pairs,
        valid_sets=valid_sets,
        invalid_sets=invalid_sets,
        failures=failures,
        config_hash=config_hash,
    )


def freeze_minimal_pairs(
    *,
    scene_spec_dir: Path = DEFAULT_SCENE_SPEC_DIR,
    realization_dir: Path = DEFAULT_REALIZATION_DIR,
    task_dir: Path = DEFAULT_TASK_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    workers: int = 1,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    artifacts = _build_artifacts(
        scene_spec_dir=scene_spec_dir,
        realization_dir=realization_dir,
        task_dir=task_dir,
        progress=progress,
        workers=workers,
    )
    for name, value in zip(
        (MINIMAL_PAIRS_FILE, AUDIT_FILE, INVALID_REPORT_FILE), artifacts
    ):
        _write_yaml(output_dir / name, value)
    artifact, audit, invalid = artifacts
    return {
        "implementation_complete": audit["implementation_complete"],
        "coverage_complete": audit["coverage_complete"],
        "stage_acceptance_ready": audit["stage_acceptance_ready"],
        "publication_ready": audit["publication_ready"],
        "summary": audit["summary"],
        MINIMAL_PAIRS_FILE: artifact,
        AUDIT_FILE: audit,
        INVALID_REPORT_FILE: invalid,
    }


def check_minimal_pairs(
    *,
    scene_spec_dir: Path = DEFAULT_SCENE_SPEC_DIR,
    realization_dir: Path = DEFAULT_REALIZATION_DIR,
    task_dir: Path = DEFAULT_TASK_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    workers: int = 1,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    expected = _build_artifacts(
        scene_spec_dir=scene_spec_dir,
        realization_dir=realization_dir,
        task_dir=task_dir,
        progress=progress,
        workers=workers,
    )
    stale_paths = []
    for name, value in zip(
        (MINIMAL_PAIRS_FILE, AUDIT_FILE, INVALID_REPORT_FILE), expected
    ):
        current = _load_yaml(output_dir / name)
        if current != value:
            stale_paths.append(str(output_dir / name))
    artifact, audit, _ = expected
    return {
        "stale": bool(stale_paths),
        "stale_paths": stale_paths,
        "valid_pair_count": len(artifact["pairs"]),
        "matrix_ready_contrast_set_count": sum(
            row["matrix_ready"] for row in artifact["contrast_sets"]
        ),
        "implementation_complete": audit["implementation_complete"],
    }


def _validate_pair_record(row: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    document = {
        "schema_version": SCHEMA_VERSION,
        "kind": "minimal_pair",
        "minimal_pair": canonical_data(row),
    }
    report = validate_document(document)
    errors.extend(f"{item['path']}: {item['message']}" for item in report["errors"])
    if row.get("status") != "valid":
        errors.append(f"pair {row.get('id')} in valid artifact is not valid")
    if row.get("actual_dcodes") != [row.get("intended_dcode")]:
        errors.append(f"pair {row.get('id')} actual/intended D-code mismatch")
    if row.get("unmapped_semantic_deltas"):
        errors.append(f"pair {row.get('id')} has unmapped semantic deltas")
    return errors


def _validate_set_record(row: Mapping[str, Any], pair_ids: set[str]) -> list[str]:
    errors: list[str] = []
    document = {
        "schema_version": SCHEMA_VERSION,
        "kind": "axis_contrast_set",
        "axis_contrast_set": canonical_data(row),
    }
    report = validate_document(document)
    errors.extend(f"{item['path']}: {item['message']}" for item in report["errors"])
    k = int(row.get("k", 0))
    if row.get("matrix_ready"):
        if len(row.get("members", [])) != k:
            errors.append(f"contrast set {row.get('id')} member count differs from K")
        if len(row.get("pair_ids", [])) != k * (k - 1) // 2:
            errors.append(f"contrast set {row.get('id')} is not a complete clique")
        if not set(row.get("pair_ids", [])) <= pair_ids:
            errors.append(f"contrast set {row.get('id')} references unknown pair")
        indices = [member.get("matrix_index") for member in row.get("members", [])]
        if indices != list(range(k)):
            errors.append(f"contrast set {row.get('id')} matrix indices are unstable")
    return errors


def _validate_frozen_sources(
    artifact: Mapping[str, Any],
    *,
    scene_spec_dir: Path,
    realization_dir: Path,
    task_dir: Path,
) -> list[str]:
    errors: list[str] = []
    pairs_by_scene: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    sets_by_scene: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in artifact.get("pairs", []):
        pairs_by_scene[str(row.get("scene_spec_id", ""))].append(row)
    for row in artifact.get("contrast_sets", []):
        sets_by_scene[str(row.get("scene_spec_id", ""))].append(row)
    for scene_id in sorted(set(pairs_by_scene) | set(sets_by_scene)):
        rows = pairs_by_scene.get(scene_id) or sets_by_scene[scene_id]
        first = rows[0]
        realized_id = str(first.get("realized_family_id", ""))
        layout_seed = int(first.get("layout_seed", -1))
        scene_path = (
            scene_spec_dir / realized_id / f"{layout_seed}.yml"
        )
        realized_path = realization_dir / f"{realized_id}.yml"
        scene_artifact = _load_yaml(scene_path)
        realized_artifact = _load_yaml(realized_path)
        if scene_artifact is None:
            errors.append(f"{scene_id}: missing source SceneSpec {scene_path}")
            continue
        if realized_artifact is None:
            errors.append(
                f"{scene_id}: missing source realization {realized_path}"
            )
            continue
        source_failures = [
            *_source_failures(scene_artifact, realized_artifact),
            *_initial_observation_failures(scene_artifact),
        ]
        errors.extend(
            f"{scene_id}: {row['code']}: {row['message']}"
            for row in source_failures
        )
        scene = scene_artifact["scene_spec"]
        realized = realized_artifact["realized_scene_family"]
        from if_benchmark.layout.manipulation import selected_realized
        if not source_failures:
            realized = selected_realized(realized, scene)
        if scene.get("id") != scene_id:
            errors.append(f"{scene_id}: source SceneSpec id mismatch")
            continue
        variants = {
            str(row["id"]): row
            for row in realized.get("goal_variants", [])
        }
        tasks = {
            variant_id: _load_yaml(
                task_dir / scene_id / f"{variant_id}.yml"
            )
            for variant_id in variants
        }
        for pair in pairs_by_scene.get(scene_id, []):
            if pair.get("source_scene_hash") != scene_artifact.get(
                "artifact_hash"
            ):
                errors.append(f"{pair.get('id')}: source SceneSpec hash mismatch")
            if pair.get("source_realization_hash") != realized_artifact.get(
                "artifact_hash"
            ):
                errors.append(f"{pair.get('id')}: source realization hash mismatch")
            left_id = str(pair.get("from_goal_variant_id", ""))
            right_id = str(pair.get("to_goal_variant_id", ""))
            if left_id not in variants or right_id not in variants:
                errors.append(f"{pair.get('id')}: source goal endpoint missing")
                continue
            result = edge_axis(
                variants[left_id]["goal"],
                variants[right_id]["goal"],
                left_intended_axes=variants[left_id].get("intended_axes"),
                right_intended_axes=variants[right_id].get("intended_axes"),
            )
            if list(result.actual_dcodes) != [pair.get("intended_dcode")] or (
                result.unmapped_semantic_deltas
            ):
                errors.append(f"{pair.get('id')}: source edge_axis mismatch")
            for endpoint_key, variant_id in (
                ("from_endpoint", left_id),
                ("to_endpoint", right_id),
            ):
                task = tasks.get(variant_id)
                if task is None:
                    errors.append(
                        f"{pair.get('id')}: source task {variant_id} is missing"
                    )
                    continue
                task_failures = _task_failures(
                    task,
                    scene_artifact=scene_artifact,
                    realized_artifact=realized_artifact,
                    variant=variants[variant_id],
                )
                errors.extend(
                    f"{pair.get('id')}: {row['code']}"
                    for row in task_failures
                )
                if canonical_data(pair.get(endpoint_key)) != _task_endpoint(task):
                    errors.append(
                        f"{pair.get('id')}: frozen {endpoint_key} is stale"
                    )
        rebuilt_sets, _ = build_contrast_sets(
            scene_spec=scene,
            realized_family=realized_artifact["realized_scene_family"],
            task_artifacts={
                key: value for key, value in tasks.items() if value is not None
            },
            pair_records=pairs_by_scene.get(scene_id, []),
        )
        rebuilt_sets = [
            row for row in rebuilt_sets if bool(row.get("matrix_ready"))
        ]
        frozen_sets = sorted(
            sets_by_scene.get(scene_id, []), key=lambda row: row["id"]
        )
        if canonical_data(sorted(rebuilt_sets, key=lambda row: row["id"])) != (
            canonical_data(frozen_sets)
        ):
            errors.append(f"{scene_id}: frozen contrast sets are stale")
    return errors


_M14_VALIDATE_ARTIFACT: Mapping[str, Any] = {}
_M14_VALIDATE_PAIR_IDS: set[str] = set()
_M14_VALIDATE_PAIRS_BY_SCENE: Mapping[
    str, Sequence[Mapping[str, Any]]
] = {}
_M14_VALIDATE_SETS_BY_SCENE: Mapping[
    str, Sequence[Mapping[str, Any]]
] = {}
_M14_VALIDATE_SCENE_SPEC_DIR = Path()
_M14_VALIDATE_REALIZATION_DIR = Path()
_M14_VALIDATE_TASK_DIR = Path()
VALIDATION_RECORD_BATCH_SIZE = 256


def _init_m14_validate_worker(
    artifact: Mapping[str, Any],
    pair_ids: set[str],
    pairs_by_scene: Mapping[str, Sequence[Mapping[str, Any]]],
    sets_by_scene: Mapping[str, Sequence[Mapping[str, Any]]],
    scene_spec_dir: str,
    realization_dir: str,
    task_dir: str,
) -> None:
    global _M14_VALIDATE_ARTIFACT
    global _M14_VALIDATE_PAIR_IDS
    global _M14_VALIDATE_PAIRS_BY_SCENE
    global _M14_VALIDATE_SETS_BY_SCENE
    global _M14_VALIDATE_SCENE_SPEC_DIR
    global _M14_VALIDATE_REALIZATION_DIR
    global _M14_VALIDATE_TASK_DIR
    _M14_VALIDATE_ARTIFACT = artifact
    _M14_VALIDATE_PAIR_IDS = pair_ids
    _M14_VALIDATE_PAIRS_BY_SCENE = pairs_by_scene
    _M14_VALIDATE_SETS_BY_SCENE = sets_by_scene
    _M14_VALIDATE_SCENE_SPEC_DIR = Path(scene_spec_dir)
    _M14_VALIDATE_REALIZATION_DIR = Path(realization_dir)
    _M14_VALIDATE_TASK_DIR = Path(task_dir)


def _validate_record_batch(task: tuple[str, int, int]) -> list[str]:
    kind, start, end = task
    if kind == "pair":
        rows = _M14_VALIDATE_ARTIFACT.get("pairs", [])[start:end]
        return [error for row in rows for error in _validate_pair_record(row)]
    rows = _M14_VALIDATE_ARTIFACT.get("contrast_sets", [])[start:end]
    return [
        error
        for row in rows
        for error in _validate_set_record(row, _M14_VALIDATE_PAIR_IDS)
    ]


def _validate_source_batch(scene_ids: tuple[str, ...]) -> list[str]:
    partial = {
        "pairs": [
            row
            for scene_id in scene_ids
            for row in _M14_VALIDATE_PAIRS_BY_SCENE.get(scene_id, [])
        ],
        "contrast_sets": [
            row
            for scene_id in scene_ids
            for row in _M14_VALIDATE_SETS_BY_SCENE.get(scene_id, [])
        ],
    }
    return _validate_frozen_sources(
        partial,
        scene_spec_dir=_M14_VALIDATE_SCENE_SPEC_DIR,
        realization_dir=_M14_VALIDATE_REALIZATION_DIR,
        task_dir=_M14_VALIDATE_TASK_DIR,
    )


def _validation_record_tasks(kind: str, count: int) -> list[tuple[str, int, int]]:
    return [
        (kind, start, min(count, start + VALIDATION_RECORD_BATCH_SIZE))
        for start in range(0, count, VALIDATION_RECORD_BATCH_SIZE)
    ]


def _parallel_validation_errors(
    artifact: Mapping[str, Any],
    pair_ids: set[str],
    *,
    scene_spec_dir: Path,
    realization_dir: Path,
    task_dir: Path,
    workers: int,
) -> list[str]:
    pairs_by_scene: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    sets_by_scene: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in artifact.get("pairs", []):
        pairs_by_scene[str(row.get("scene_spec_id", ""))].append(row)
    for row in artifact.get("contrast_sets", []):
        sets_by_scene[str(row.get("scene_spec_id", ""))].append(row)
    scene_ids = sorted(set(pairs_by_scene) | set(sets_by_scene))
    source_tasks = [
        tuple(scene_ids[index : index + SCENE_BATCH_SIZE])
        for index in range(0, len(scene_ids), SCENE_BATCH_SIZE)
    ]
    pair_tasks = _validation_record_tasks(
        "pair", len(artifact.get("pairs", []))
    )
    set_tasks = _validation_record_tasks(
        "set", len(artifact.get("contrast_sets", []))
    )
    task_count = len(pair_tasks) + len(set_tasks) + len(source_tasks)
    context = _fork_context()
    worker_count = min(_validated_worker_count(workers), max(1, task_count))
    if context is None or worker_count == 1:
        errors = [
            error
            for row in artifact.get("pairs", [])
            for error in _validate_pair_record(row)
        ]
        errors.extend(
            error
            for row in artifact.get("contrast_sets", [])
            for error in _validate_set_record(row, pair_ids)
        )
        errors.extend(
            _validate_frozen_sources(
                artifact,
                scene_spec_dir=scene_spec_dir,
                realization_dir=realization_dir,
                task_dir=task_dir,
            )
        )
        return errors

    errors: list[str] = []
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=context,
        initializer=_init_m14_validate_worker,
        initargs=(
            artifact,
            pair_ids,
            pairs_by_scene,
            sets_by_scene,
            str(scene_spec_dir),
            str(realization_dir),
            str(task_dir),
        ),
    ) as executor:
        for tasks, function in (
            (pair_tasks, _validate_record_batch),
            (set_tasks, _validate_record_batch),
            (source_tasks, _validate_source_batch),
        ):
            for rows in _bounded_process_map(
                executor,
                function,
                tasks,
                max_in_flight=max(1, 2 * worker_count),
            ):
                errors.extend(rows)
    return errors


def validate_minimal_pairs(
    *,
    scene_spec_dir: Path = DEFAULT_SCENE_SPEC_DIR,
    realization_dir: Path = DEFAULT_REALIZATION_DIR,
    task_dir: Path = DEFAULT_TASK_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    workers: int = 1,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    del progress
    _validated_worker_count(workers)
    artifact = _load_yaml(output_dir / MINIMAL_PAIRS_FILE)
    audit = _load_yaml(output_dir / AUDIT_FILE)
    invalid = _load_yaml(output_dir / INVALID_REPORT_FILE)
    errors: list[str] = []
    if artifact is None or audit is None or invalid is None:
        errors.append("one or more M14 artifact files are missing or invalid YAML")
        return {
            "valid": False,
            "errors": errors,
            "valid_pair_count": 0,
            "matrix_ready_contrast_set_count": 0,
        }
    if not _self_hash_valid(artifact):
        errors.append("minimal_pairs.yml artifact_hash is invalid")
    if not _self_hash_valid(audit, "audit_hash"):
        errors.append("minimal_pair_audit.yml audit_hash is invalid")
    if not _self_hash_valid(invalid, "report_hash"):
        errors.append("invalid_pair_report.yml report_hash is invalid")
    if audit.get("artifact_set_hash") != artifact.get("artifact_hash"):
        errors.append("audit artifact_set_hash does not match minimal_pairs artifact")
    pair_ids = {str(row.get("id")) for row in artifact.get("pairs", [])}
    if len(pair_ids) != len(artifact.get("pairs", [])):
        errors.append("duplicate minimal-pair ids")
    set_ids = {
        str(row.get("id")) for row in artifact.get("contrast_sets", [])
    }
    if len(set_ids) != len(artifact.get("contrast_sets", [])):
        errors.append("duplicate contrast-set ids")
    errors.extend(
        _parallel_validation_errors(
            artifact,
            pair_ids,
            scene_spec_dir=scene_spec_dir,
            realization_dir=realization_dir,
            task_dir=task_dir,
            workers=workers,
        )
    )
    return {
        "valid": not errors,
        "errors": errors,
        "valid_pair_count": len(artifact.get("pairs", [])),
        "matrix_ready_contrast_set_count": sum(
            bool(row.get("matrix_ready"))
            for row in artifact.get("contrast_sets", [])
        ),
    }
