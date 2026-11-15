"""Pure M13/M15 scene-integrity contract and evaluator.

The primary SuccessContract predicates prove the intended terminal fact.  This
module adds the orthogonal guarantee that entities not explicitly referenced by
the instruction remain at their shared initial pose.  It deliberately lives
outside the asset/pair SuccessContract so this scene-level policy does not
invalidate M4/M5 evidence.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from if_benchmark.schema import artifact_hash
from if_benchmark.schema.serialization import canonical_data
from if_benchmark.success_contract import (
    SuccessContract,
    SuccessSnapshotError,
    normalize_success_snapshot,
)


SCENE_INTEGRITY_POLICY_VERSION = (
    "m13_m15_scene_integrity_v1_explicit_goal_entities_exempt"
)
SCENE_INTEGRITY_ARTICULATION_NORMALIZED_LIMIT = 0.02


def scene_integrity_policy(contract: SuccessContract) -> dict[str, Any]:
    """Return the exact policy frozen into every M13 checker binding."""

    stability = contract.section("stability")
    value: dict[str, Any] = {
        "policy_version": SCENE_INTEGRITY_POLICY_VERSION,
        "participant_source": (
            "concrete_goal_recursive_scene_entity_references_v1"
        ),
        "translation_limit_m": float(stability["translation_drift_m"]),
        "rotation_limit_deg": float(stability["rotation_drift_deg"]),
        "articulation_normalized_limit": (
            SCENE_INTEGRITY_ARTICULATION_NORMALIZED_LIMIT
        ),
        "limits_inclusive": True,
    }
    value["policy_hash"] = artifact_hash(value)
    return canonical_data(value)


def instruction_participating_entities(
    goal: Mapping[str, Any], scene_entity_keys: Sequence[str]
) -> list[str]:
    """Collect every SceneSpec entity explicitly referenced by ConcreteGoal.

    Recursive matching covers direct subject/target slots as well as typed
    descriptor and parameter references such as a spatial landmark.  Values
    that are not exact SceneSpec entity keys are ignored, so ordinary strings
    such as verbs, relations, arms, and descriptor labels cannot become
    exemptions.
    """

    known = {str(key) for key in scene_entity_keys}
    participants: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for item in value.values():
                visit(item)
            return
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            for item in value:
                visit(item)
            return
        if isinstance(value, str) and value in known:
            participants.add(value)

    visit(goal)
    return sorted(participants)


def build_scene_integrity_binding(
    goal: Mapping[str, Any],
    scene_entities: Mapping[str, Any],
    contract: SuccessContract,
) -> dict[str, Any]:
    """Build the deterministic M13 binding consumed by M15."""

    entity_keys = sorted({str(key) for key in scene_entities})
    exempt_entities = instruction_participating_entities(goal, entity_keys)
    protected_entities = sorted(set(entity_keys) - set(exempt_entities))
    articulated_entities = {
        str(key)
        for key, row in scene_entities.items()
        if _is_articulated_scene_entity(row)
    }
    return {
        "policy": scene_integrity_policy(contract),
        "exempt_entities": exempt_entities,
        "protected_entities": protected_entities,
        "protected_articulated_entities": sorted(
            articulated_entities & set(protected_entities)
        ),
    }


def evaluate_scene_integrity(
    binding: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    contract: SuccessContract,
) -> dict[str, Any]:
    """Compare initial/terminal poses for every non-exempt scene entity."""

    expected_policy = scene_integrity_policy(contract)
    if not isinstance(binding, Mapping):
        return _blocked(
            contract,
            snapshot,
            "scene_integrity_binding_missing",
        )
    if binding.get("policy") != expected_policy:
        return _blocked(
            contract,
            snapshot,
            "scene_integrity_policy_mismatch",
            metrics={"expected_policy": expected_policy},
        )
    entity_sets: dict[str, list[str]] = {}
    for field in (
        "exempt_entities",
        "protected_entities",
        "protected_articulated_entities",
    ):
        raw_values = binding.get(field)
        if (
            not isinstance(raw_values, Sequence)
            or isinstance(raw_values, (str, bytes, bytearray))
            or any(
                not isinstance(value, str) or not value
                for value in raw_values
            )
        ):
            return _blocked(
                contract,
                snapshot,
                f"scene_integrity_{field}_invalid",
            )
        values = [str(value) for value in raw_values]
        if values != sorted(set(values)):
            return _blocked(
                contract,
                snapshot,
                f"scene_integrity_{field}_not_canonical",
            )
        entity_sets[field] = values
    exempt = entity_sets["exempt_entities"]
    protected = entity_sets["protected_entities"]
    protected_articulated = entity_sets["protected_articulated_entities"]
    protected_articulated_set = set(protected_articulated)
    overlap = sorted(set(exempt) & set(protected))
    if overlap:
        return _blocked(
            contract,
            snapshot,
            "scene_integrity_partition_overlap",
            metrics={"overlap_entities": overlap},
        )
    unexpected_articulated = sorted(set(protected_articulated) - set(protected))
    if unexpected_articulated:
        return _blocked(
            contract,
            snapshot,
            "scene_integrity_articulated_entities_not_protected",
            metrics={"unexpected_entities": unexpected_articulated},
        )
    try:
        normalized = normalize_success_snapshot(snapshot, contract)
    except SuccessSnapshotError as exc:
        return _blocked(
            contract,
            snapshot,
            "scene_integrity_snapshot_invalid",
            metrics={"error": str(exc)},
        )

    initial_entities = (normalized["initial"].get("entities") or {})
    terminal_entities = (normalized["terminal"].get("entities") or {})
    if not isinstance(initial_entities, Mapping) or not isinstance(
        terminal_entities, Mapping
    ):
        return _blocked(
            contract,
            normalized,
            "scene_integrity_entities_missing",
        )
    initial_keys = set(initial_entities)
    terminal_keys = set(terminal_entities)
    declared_keys = set(exempt) | set(protected)
    if initial_keys != terminal_keys or declared_keys != initial_keys:
        return _blocked(
            contract,
            normalized,
            "scene_integrity_entity_partition_mismatch",
            metrics={
                "declared_entities": sorted(declared_keys),
                "initial_entities": sorted(initial_keys),
                "terminal_entities": sorted(terminal_keys),
            },
        )

    initial_articulations = normalized["initial"].get("articulations") or {}
    terminal_articulations = normalized["terminal"].get("articulations") or {}
    if not isinstance(initial_articulations, Mapping) or not isinstance(
        terminal_articulations, Mapping
    ):
        return _blocked(
            contract,
            normalized,
            "scene_integrity_articulations_malformed",
        )

    checked_entities = protected
    rows: list[dict[str, Any]] = []
    incomplete: list[str] = []
    failed_entities: list[str] = []
    for key in checked_entities:
        initial_row = initial_entities.get(key)
        terminal_row = terminal_entities.get(key)
        if not isinstance(initial_row, Mapping) or not isinstance(
            terminal_row, Mapping
        ):
            incomplete.append(key)
            continue
        try:
            initial_position, initial_quat = _pose(initial_row)
            terminal_position, terminal_quat = _pose(terminal_row)
        except (KeyError, TypeError, ValueError):
            incomplete.append(key)
            continue
        translation = math.dist(initial_position, terminal_position)
        rotation = _quaternion_distance_deg(initial_quat, terminal_quat)
        row: dict[str, Any] = {
            "entity_key": key,
            "translation_delta_m": translation,
            "rotation_delta_deg": rotation,
            "translation_within_limit": (
                translation <= float(expected_policy["translation_limit_m"])
            ),
            "rotation_within_limit": (
                rotation <= float(expected_policy["rotation_limit_deg"])
            ),
        }

        initial_has_articulation = key in initial_articulations
        terminal_has_articulation = key in terminal_articulations
        expects_articulation = key in protected_articulated_set
        if expects_articulation:
            if not initial_has_articulation or not terminal_has_articulation:
                incomplete.append(key)
                continue
            try:
                articulation_delta = _normalized_articulation_delta(
                    initial_articulations[key], terminal_articulations[key]
                )
            except (KeyError, TypeError, ValueError):
                incomplete.append(key)
                continue
            row["articulation_normalized_delta"] = articulation_delta
            row["articulation_within_limit"] = articulation_delta <= float(
                expected_policy["articulation_normalized_limit"]
            )
        elif initial_has_articulation or terminal_has_articulation:
            incomplete.append(key)
            continue

        row["passed"] = bool(
            row["translation_within_limit"]
            and row["rotation_within_limit"]
            and row.get("articulation_within_limit", True)
        )
        if not row["passed"]:
            failed_entities.append(key)
        rows.append(row)

    metrics = {
        "policy": expected_policy,
        "exempt_entities": exempt,
        "checked_entities": checked_entities,
        "checked_articulated_entities": protected_articulated,
        "failed_entities": sorted(failed_entities),
        "per_entity": rows,
    }
    if incomplete:
        return _blocked(
            contract,
            normalized,
            "scene_integrity_observation_incomplete",
            metrics={**metrics, "incomplete_entities": sorted(set(incomplete))},
        )
    passed = not failed_entities
    return canonical_data(
        {
            "status": "passed" if passed else "failed",
            "passed": passed,
            "predicate": "scene_integrity",
            "failed_predicates": [] if passed else ["scene_integrity"],
            "metrics": metrics,
            **contract.identity,
            "success_snapshot_hash": normalized["success_snapshot_hash"],
        }
    )


def _pose(entity: Mapping[str, Any]) -> tuple[list[float], list[float]]:
    pose = entity.get("pose")
    if not isinstance(pose, Mapping):
        raise KeyError("pose")
    position = pose.get("position")
    quaternion = pose.get("quat_wxyz")
    if (
        not isinstance(position, Sequence)
        or isinstance(position, (str, bytes, bytearray))
        or len(position) != 3
        or not isinstance(quaternion, Sequence)
        or isinstance(quaternion, (str, bytes, bytearray))
        or len(quaternion) != 4
    ):
        raise ValueError("malformed pose")
    p = [float(value) for value in position]
    q = [float(value) for value in quaternion]
    if any(not math.isfinite(value) for value in (*p, *q)):
        raise ValueError("non-finite pose")
    return p, q


def _quaternion_distance_deg(left: Sequence[float], right: Sequence[float]) -> float:
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        raise ValueError("zero quaternion")
    dot = abs(
        sum(float(a) * float(b) for a, b in zip(left, right))
        / (left_norm * right_norm)
    )
    return math.degrees(2.0 * math.acos(min(1.0, max(-1.0, dot))))


def _qpos(value: Any) -> float:
    if isinstance(value, Mapping):
        value = value.get("qpos")
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        if len(value) != 1:
            raise ValueError("scene integrity requires one selected joint")
        value = value[0]
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("non-finite articulation qpos")
    return result


def _normalized_articulation_delta(initial: Any, terminal: Any) -> float:
    if not isinstance(initial, Mapping) or not isinstance(terminal, Mapping):
        raise TypeError("articulation rows must be mappings")
    initial_qpos = _qpos(initial)
    terminal_qpos = _qpos(terminal)
    if any(
        row.get(field) is None
        for row in (initial, terminal)
        for field in ("lower", "upper")
    ):
        raise KeyError("articulation limits")
    initial_limits = (float(initial["lower"]), float(initial["upper"]))
    terminal_limits = (float(terminal["lower"]), float(terminal["upper"]))
    if any(
        not math.isfinite(value)
        for value in (*initial_limits, *terminal_limits)
    ):
        raise ValueError("non-finite articulation limits")
    if any(
        not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)
        for left, right in zip(initial_limits, terminal_limits)
    ):
        raise ValueError("articulation limits changed")
    lower, upper = initial_limits
    span = upper - lower
    if not math.isfinite(span) or span <= 1e-12:
        raise ValueError("invalid articulation limits")
    return abs(terminal_qpos - initial_qpos) / span


def _is_articulated_scene_entity(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    initial_pose = value.get("initial_pose") or {}
    if not isinstance(initial_pose, Mapping):
        return False
    joint_state = initial_pose.get("articulated_joint_state") or {}
    if not isinstance(joint_state, Mapping):
        return False
    mode = str(joint_state.get("mode") or "")
    return bool(mode and mode != "not_articulated")


def _blocked(
    contract: SuccessContract,
    snapshot: Mapping[str, Any],
    code: str,
    *,
    metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return canonical_data(
        {
            "status": "blocked",
            "passed": None,
            "predicate": "scene_integrity",
            "failed_predicates": [code],
            "metrics": dict(metrics or {}),
            **contract.identity,
            "success_snapshot_hash": snapshot.get("success_snapshot_hash"),
        }
    )


__all__ = [
    "SCENE_INTEGRITY_ARTICULATION_NORMALIZED_LIMIT",
    "SCENE_INTEGRITY_POLICY_VERSION",
    "build_scene_integrity_binding",
    "evaluate_scene_integrity",
    "instruction_participating_entities",
    "scene_integrity_policy",
]
