"""Milestone 10 RealizedSceneFamily construction.

This stage is simulator-independent.  It binds one complete Milestone 8
assignment to its frozen Milestone 9 goals, emits deterministic placement and
Unit Builder inputs, and performs structural preflights.  It does not build
Unit geometry or sample a layout.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
import multiprocessing
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence

import yaml

from if_benchmark.assignments.core import validate_assignment_domain
from if_benchmark.goals import (
    canonical_relation_value,
    canonical_state_value,
    decode_legacy_variant,
    encode_legacy_variant,
)
from if_benchmark.goals.core import (
    GOAL_VERSION,
    SCHEMA_VERSION as GOAL_SCHEMA_VERSION,
)
from if_benchmark.schema import (
    artifact_hash,
    canonical_yaml,
    deterministic_id,
    validate_document,
)
from if_benchmark.schema.serialization import canonical_data
from if_benchmark.touch_capability import validated_touch_arm_capability
from if_benchmark.orientation_adjustment import (
    ORIENTATION_ADJUSTMENT_USE,
    ORIENTATION_ADJUSTMENT_VERB,
)


SCHEMA_VERSION = 1
REALIZATION_VERSION = 5

DEFAULT_ARCHETYPES_PATH = Path("if_benchmark/data/families/archetypes.yml")
DEFAULT_FAMILIES_PATH = Path("if_benchmark/data/families/abstract_families.yml")
DEFAULT_ASSIGNMENT_DIR = Path("if_benchmark/data/assignment_domains")
DEFAULT_GOAL_DIR = Path("if_benchmark/data/goal_variants")
DEFAULT_OUTPUT_DIR = Path("if_benchmark/data/realized_families")
DEFAULT_PROCESS_WORKERS = min(32, os.cpu_count() or 1)

AUDIT_FILE = "realization_audit.yml"
FAIL_FILE = "preflight_fail_report.yml"

AXIS_PROJECTION_ORDER = ("D1a", "D1b", "D1c", "D2a", "D2b", "D4")
PAIR_REQUIRED_RELATIONS = frozenset(("in", "on", "stack"))
PAIR_RELATION_BY_GOAL_RELATION = {
    "in": "place_in",
    "on": "place_on",
    "stack": "stack",
}
TWO_TARGET_RELATIONS = frozenset(("between", "left_of_pair", "right_of_pair"))
SUPPORTED_A2O = frozenset(
    (
        "pick",
        "open",
        "close",
        "switch_on",
        "switch_off",
        "click",
        "press",
        ORIENTATION_ADJUSTMENT_VERB,
    )
)
TOUCH_USE_BY_VERB = {"click": "click_subject", "press": "press_subject"}
SUPPORTED_O2O = frozenset(
    (
        "place_in",
        "place_on",
        "place_next_to",
        "stack",
        "place_left_of",
        "place_right_of",
        "place_in_front_of",
        "place_behind",
        "place_between",
        "place_left_of_pair",
        "place_right_of_pair",
    )
)

CONFIG: dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "realization_version": REALIZATION_VERSION,
    "unit_boundary": "unit_inputs_only",
    "axis_projection_order": list(AXIS_PROJECTION_ORDER),
    "asset_validation_evidence_join": "asset_model_use_role_v2",
    "supported_a2o": sorted(SUPPORTED_A2O),
    "supported_o2o": sorted(SUPPORTED_O2O),
    "pair_required_relations": sorted(PAIR_REQUIRED_RELATIONS),
    "legacy_export_fields": ["templates", "variants", "meta"],
}


@dataclass
class RealizationError(ValueError):
    """Stable failure attributed to the M10 realization boundary."""

    code: str
    message: str
    family_id: str | None = None
    assignment_id: str | None = None
    path: str | None = None

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"

    def to_record(self) -> dict[str, Any]:
        result: dict[str, Any] = {"code": self.code, "message": self.message}
        for key in ("family_id", "assignment_id", "path"):
            value = getattr(self, key)
            if value is not None:
                result[key] = value
        return result


def _error(
    code: str,
    message: str,
    *,
    family_id: str | None = None,
    assignment_id: str | None = None,
    path: str | None = None,
) -> RealizationError:
    return RealizationError(
        code,
        message,
        family_id=family_id,
        assignment_id=assignment_id,
        path=path,
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.load(stream, Loader=yaml.CSafeLoader)
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return value


def _try_load_yaml(path: Path) -> dict[str, Any] | None:
    try:
        return _load_yaml(path)
    except (OSError, ValueError, yaml.YAMLError):
        return None


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
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(canonical_yaml(value))
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _realized_id(family_id: str, assignment_id: str) -> str:
    return deterministic_id(
        "rsf",
        {"abstract_family_id": family_id, "assignment_id": assignment_id},
    )


def _constraint_roles(constraint: Mapping[str, Any]) -> set[str]:
    return {str(role) for role in constraint.get("roles", [])}


def _connected_local_groups(
    constraints: Sequence[Mapping[str, Any]],
) -> list[tuple[list[str], list[Mapping[str, Any]]]]:
    local = [row for row in constraints if row.get("type") == "local_relation"]
    remaining = list(local)
    result: list[tuple[list[str], list[Mapping[str, Any]]]] = []
    while remaining:
        selected = [remaining.pop(0)]
        roles = _constraint_roles(selected[0])
        changed = True
        while changed:
            changed = False
            for row in list(remaining):
                if roles & _constraint_roles(row):
                    remaining.remove(row)
                    selected.append(row)
                    roles.update(_constraint_roles(row))
                    changed = True
        result.append((sorted(roles), selected))
    return result


def _unit_input(
    unit_type: str,
    roles: Sequence[str],
    candidate: Mapping[str, Any],
    requirements: Sequence[Mapping[str, Any]],
    *,
    params: Mapping[str, Any] | None = None,
    extra_source_constraints: Sequence[str] = (),
    pair_precondition_ids: Sequence[str] = (),
) -> dict[str, Any]:
    entity_by_role = candidate["materialization_params"]["entity_key_by_role"]
    resolved = candidate.get("resolved_values") or {}
    role_set = set(roles)
    sources = [
        row
        for row in requirements
        if role_set & _constraint_roles(row)
    ]
    source_ids = sorted(
        {str(row["id"]) for row in sources}
        | {str(value) for value in extra_source_constraints}
    )
    all_pairs = (
        (candidate.get("materialization_params") or {}).get(
            "pair_preconditions", {}
        )
        or {}
    )
    pair_preconditions = {
        key: deepcopy(value)
        for key, value in sorted(all_pairs.items())
        if key in set(pair_precondition_ids) | set(source_ids)
    }
    identity = {"type": unit_type, "roles": sorted(roles)}
    row = {
        "id": deterministic_id("unit_input", identity),
        "type": unit_type,
        "roles": sorted(roles),
        "entity_keys": [str(entity_by_role[role]) for role in sorted(roles)],
        "source_constraints": source_ids,
        "geometry_refs": {
            role: str(resolved[role]["geometry_ref"]) for role in sorted(roles)
        },
        "pair_preconditions": pair_preconditions,
        "params": canonical_data(params or {}),
    }
    initial_state_refs = {
        role: deepcopy(resolved[role]["initial_state_ref"])
        for role in sorted(roles)
        if isinstance(resolved[role].get("initial_state_ref"), Mapping)
    }
    if initial_state_refs:
        row["initial_state_refs"] = initial_state_refs
    return row


def _receiver_relation_endpoints(
    archetype: Mapping[str, Any],
    receiver_roles: Sequence[str],
    relations: Sequence[str],
) -> dict[str, dict[str, list[str]]]:
    receiver_set = set(receiver_roles)
    result: dict[str, dict[str, list[str]]] = {}
    for relation in relations:
        subject_roles: set[str] = set()
        endpoint_receivers: set[str] = set()
        for variant in archetype.get("variant_schemas", []):
            action = variant.get("action") or {}
            if str(action.get("relation") or "") != relation:
                continue
            bindings = variant.get("role_bindings") or {}
            subject = bindings.get("subject")
            targets = [
                bindings.get("target"),
                bindings.get("target2"),
            ]
            targets = [str(value) for value in targets if value is not None]
            if subject is None or not targets or not set(targets) <= receiver_set:
                continue
            subject_roles.add(str(subject))
            endpoint_receivers.update(targets)

        pair_relation = PAIR_RELATION_BY_GOAL_RELATION.get(relation)
        pair_constraint_ids: list[str] = []
        if pair_relation is not None:
            for constraint in archetype.get("constraints", []):
                if constraint.get("type") != "pair_validated":
                    continue
                if str((constraint.get("params") or {}).get("relation")) != pair_relation:
                    continue
                constraint_roles = _constraint_roles(constraint)
                if (
                    constraint_roles & subject_roles
                    and constraint_roles & endpoint_receivers
                ):
                    pair_constraint_ids.append(str(constraint["id"]))
        result[relation] = {
            "subject_roles": sorted(subject_roles),
            "receiver_roles": sorted(endpoint_receivers),
            "pair_constraint_ids": sorted(pair_constraint_ids),
        }
    return result


def _build_unit_inputs(
    archetype: Mapping[str, Any], candidate: Mapping[str, Any]
) -> list[dict[str, Any]]:
    roles = {
        str(row["id"]): row for row in archetype.get("roles", [])
    }
    requirements = [
        row
        for row in candidate.get("layout_requirements", [])
        if row.get("phase") in {"unit_build", "layout"}
    ]
    result: list[dict[str, Any]] = []
    consumed: set[str] = set()
    for group_roles, constraints in _connected_local_groups(requirements):
        reference_roles = sorted(
            {
                str(row.get("params", {}).get("reference_role"))
                for row in constraints
                if row.get("params", {}).get("reference_role") is not None
            }
        )
        result.append(
            _unit_input(
                "LandmarkRelationUnit",
                group_roles,
                candidate,
                requirements,
                params={
                    "reference_roles": reference_roles,
                    "relations": [
                        {
                            "source_role": row["params"]["source_role"],
                            "reference_role": row["params"]["reference_role"],
                            "predicates": list(row["params"]["predicates"]),
                        }
                        for row in constraints
                    ],
                },
            )
        )
        consumed.update(group_roles)

    receiver_roles = sorted(
        role
        for role, definition in roles.items()
        if definition.get("layout_role") == "receiver_reservation"
    )
    if receiver_roles:
        relations = sorted(
            {
                str(value)
                for row in requirements
                if row.get("type")
                in {"terminal_relation_supported", "receiver_reservation"}
                for value in row.get("params", {}).get("relations", [])
            }
        )
        endpoints = _receiver_relation_endpoints(
            archetype, receiver_roles, relations
        )
        pair_ids = sorted(
            {
                constraint_id
                for endpoint in endpoints.values()
                for constraint_id in endpoint["pair_constraint_ids"]
            }
        )
        subject_roles = sorted(
            {
                role
                for endpoint in endpoints.values()
                for role in endpoint["subject_roles"]
            }
        )
        result.append(
            _unit_input(
                "ReceiverReservationUnit",
                receiver_roles,
                candidate,
                requirements,
                params={
                    "subject_roles": subject_roles,
                    "receiver_roles": receiver_roles,
                    "relations": relations,
                    "relation_endpoints": endpoints,
                },
                extra_source_constraints=pair_ids,
                pair_precondition_ids=pair_ids,
            )
        )
        consumed.update(receiver_roles)

    for role in sorted(set(roles) - consumed):
        result.append(
            _unit_input(
                "ObjectUnit", [role], candidate, requirements, params={}
            )
        )
    return sorted(result, key=lambda row: row["id"])


def _descriptor_projection(
    goal: Mapping[str, Any], kind: str
) -> tuple[Any, ...]:
    rows: list[Any] = []
    for slot, bundle in sorted((goal.get("descriptors") or {}).items()):
        if kind in bundle:
            rows.append((slot, canonical_data(bundle[kind])))
    return tuple(rows)


def _axis_projections(goal: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "D1a": _descriptor_projection(goal, "attribute"),
        "D1b": _descriptor_projection(goal, "spatial"),
        "D1c": _descriptor_projection(goal, "ordinal"),
        "D2a": {
            "state": canonical_state_value(goal),
            "verb": goal.get("verb") if goal.get("state") is not None else None,
            "params": (
                canonical_data(goal.get("params") or {})
                if str(goal.get("verb") or "")
                == ORIENTATION_ADJUSTMENT_VERB
                else {}
            ),
        },
        "D2b": canonical_relation_value(goal),
        "D4": goal.get("arm"),
    }


def _static_edge_axis(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> str | None:
    left_projection = _axis_projections(left)
    right_projection = _axis_projections(right)
    changed = [
        axis
        for axis in AXIS_PROJECTION_ORDER
        if left_projection[axis] != right_projection[axis]
    ]
    return changed[0] if len(changed) == 1 else None


def _matching_use_evidence(
    realized: Mapping[str, Any], entity_key: str, use: str
) -> list[dict[str, Any]]:
    role = next(
        (
            role_id
            for role_id, binding in realized["role_bindings"].items()
            if binding["entity_key"] == entity_key
        ),
        None,
    )
    if role is None:
        return []
    binding = realized["role_bindings"][role]
    return [
        deepcopy(row)
        for row in realized["evidence"]["assignment"]["evidence_refs"]
        if row.get("kind") == "asset_validation"
        and row.get("asset") == binding["asset"]
        and str(row.get("model_id")) == str(binding["model_id"])
        and row.get("use") == use
        and row.get("status") == "validated"
        and str((row.get("metadata") or {}).get("role") or "") == role
    ]


def _checker_plan(
    realized: Mapping[str, Any], variant: Mapping[str, Any]
) -> dict[str, Any]:
    goal = variant["goal"]
    verb = str(goal["verb"])
    action_family = str(goal.get("action_family"))
    subject = str(goal["subject"])
    if action_family in {"A2O_state", "A2O_event"}:
        if verb not in SUPPORTED_A2O:
            raise _error("unsupported_checker_action", f"unsupported A2O verb: {verb}")
        if any(goal.get(slot) is not None for slot in ("target", "subject2", "target2")):
            raise _error("checker_arity_mismatch", f"{verb} must only reference subject")
        arm = goal.get("arm")
        if verb in {"click", "press"} and arm not in {"left", "right"}:
            raise _error(
                "checker_arm_mismatch", f"{verb} requires a concrete arm"
            )
        if verb == "pick" and arm not in {None, "left", "right"}:
            raise _error(
                "checker_arm_mismatch", f"{verb} has an invalid arm: {arm}"
            )
        if verb not in {"pick", "click", "press"} and arm not in {
            None,
            "left",
            "right",
        }:
            raise _error(
                "checker_arm_mismatch", f"{verb} has an invalid arm: {arm}"
            )
        uses = (
            (f"pick_{arm}",)
            if verb == "pick" and arm in {"left", "right"}
            else ("pick_left", "pick_right")
            if verb == "pick"
            else (ORIENTATION_ADJUSTMENT_USE,)
            if verb == ORIENTATION_ADJUSTMENT_VERB
            else (TOUCH_USE_BY_VERB[verb],)
            if verb in TOUCH_USE_BY_VERB
            else (verb,)
        )
        evidence = [
            row
            for use in uses
            for row in _matching_use_evidence(realized, subject, use)
        ]
        if not evidence:
            required_use = " or ".join(uses)
            raise _error(
                "checker_evidence_missing",
                f"{verb} lacks validated use evidence {required_use} for {subject}",
            )
        use = (
            f"pick_{arm}"
            if verb == "pick" and arm in {"left", "right"}
            else ORIENTATION_ADJUSTMENT_USE
            if verb == ORIENTATION_ADJUSTMENT_VERB
            else TOUCH_USE_BY_VERB[verb]
            if verb in TOUCH_USE_BY_VERB
            else verb
        )
        result = {
            "variant_id": variant["id"],
            "checker_family": action_family,
            "verb": verb,
            "entity_refs": {"subject": subject},
            "arm": arm,
            "state": canonical_state_value(goal),
            "params": canonical_data(goal.get("params") or {}),
            "required_evidence": evidence,
        }
        if verb in TOUCH_USE_BY_VERB:
            if len(evidence) != 1:
                raise _error(
                    "checker_touch_evidence_ambiguous",
                    f"{verb} requires exactly one M4 evidence row for {subject}",
                )
            binding = (evidence[0].get("metadata") or {}).get(
                "touch_execution_binding"
            )
            capability = (
                validated_touch_arm_capability(binding, use=use, arm=str(arm))
                if isinstance(binding, Mapping)
                else None
            )
            if capability is None:
                raise _error(
                    "checker_touch_arm_capability_missing",
                    f"{verb} lacks exact M4 {arm}-arm capability for {subject}",
                )
            result["contact_id"] = int(capability["contact_id"])
            result["touch_execution_binding"] = canonical_data(dict(binding))
        return result

    if action_family != "O2O_relation" or verb not in SUPPORTED_O2O:
        raise _error(
            "unsupported_checker_action",
            f"unsupported checker action: {action_family}/{verb}",
        )
    relation = canonical_relation_value(goal)
    if goal.get("target") is None:
        raise _error("checker_arity_mismatch", f"{verb} requires target")
    requires_target2 = relation in TWO_TARGET_RELATIONS
    if requires_target2 != (goal.get("target2") is not None):
        raise _error(
            "checker_arity_mismatch",
            f"{verb} target2 presence does not match relation arity",
        )
    if goal.get("subject2") is not None:
        raise _error("checker_arity_mismatch", f"{verb} does not support subject2")
    pair_preconditions = realized["evidence"]["assignment"][
        "pair_preconditions"
    ]
    if relation in PAIR_REQUIRED_RELATIONS and not any(
        row.get("stage") == "canonical_skill"
        for row in pair_preconditions.values()
    ):
        raise _error(
            "checker_pair_evidence_missing",
            f"{relation} requires canonical-skill pair evidence",
        )
    entity_refs = {
        key: goal[key]
        for key in ("subject", "target", "target2")
        if goal.get(key) is not None
    }
    return {
        "variant_id": variant["id"],
        "checker_family": "O2O_relation",
        "verb": verb,
        "relation": relation,
        "entity_refs": entity_refs,
        "arm": goal.get("arm"),
        "required_evidence": deepcopy(pair_preconditions),
    }


def realize_scene_family(
    family: Mapping[str, Any],
    archetype: Mapping[str, Any],
    candidate: Mapping[str, Any],
    goal_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Join one abstract family, complete assignment, and frozen goal set."""

    family_id = str(family.get("id", ""))
    assignment_id = str(candidate.get("id", ""))
    if candidate.get("abstract_family_id") != family_id:
        raise _error(
            "assignment_family_mismatch",
            "assignment does not belong to abstract family",
            family_id=family_id,
            assignment_id=assignment_id,
        )
    if goal_artifact.get("abstract_family_id") != family_id or goal_artifact.get(
        "assignment_id"
    ) != assignment_id:
        raise _error(
            "goal_assignment_mismatch",
            "goal artifact does not match family and assignment",
            family_id=family_id,
            assignment_id=assignment_id,
        )
    expected_archetypes = list(family.get("archetypes") or [])
    if (
        list(candidate.get("archetype_ids") or []) != expected_archetypes
        or str(archetype.get("id", "")) not in expected_archetypes
    ):
        raise _error(
            "archetype_mismatch",
            "family, assignment, and archetype ids are not aligned",
            family_id=family_id,
            assignment_id=assignment_id,
        )
    entity_by_role = (
        (candidate.get("materialization_params") or {}).get(
            "entity_key_by_role", {}
        )
        or {}
    )
    roles = candidate.get("roles") or {}
    if set(entity_by_role) != set(roles) or len(set(entity_by_role.values())) != len(
        roles
    ):
        raise _error(
            "entity_key_map_mismatch",
            "entity_key_by_role must uniquely cover every assigned role",
            family_id=family_id,
            assignment_id=assignment_id,
        )
    role_bindings = {
        role: {
            "entity_key": str(entity_by_role[role]),
            "asset": str(binding["asset"]),
            "model_id": str(binding["model_id"]),
            "category": str(binding["category"]),
        }
        for role, binding in sorted(roles.items())
    }
    templates = [
        {
            "key": binding["entity_key"],
            "role_ref": role,
            "asset": binding["asset"],
            "model_candidates": [binding["model_id"]],
            "roles": list(roles[role].get("roles") or []),
        }
        for role, binding in sorted(role_bindings.items())
    ]
    goals = deepcopy(goal_artifact.get("goal_variants") or [])
    layout_constraints = [
        deepcopy(row)
        for row in candidate.get("layout_requirements", [])
        if row.get("phase") == "layout"
    ]
    realized_id = _realized_id(family_id, assignment_id)
    assignment_evidence = {
        "candidate_id": assignment_id,
        "candidate_hash": artifact_hash(candidate),
        "proven_facts": deepcopy(candidate.get("proven_facts") or []),
        "evidence_refs": deepcopy(candidate.get("evidence_refs") or []),
        "resolved_values": deepcopy(candidate.get("resolved_values") or {}),
        "layout_requirements": deepcopy(
            candidate.get("layout_requirements") or []
        ),
        "pair_preconditions": deepcopy(
            (candidate.get("materialization_params") or {}).get(
                "pair_preconditions", {}
            )
        ),
    }
    result: dict[str, Any] = {
        "id": realized_id,
        "abstract_family_id": family_id,
        "assignment_id": assignment_id,
        "archetype_ids": expected_archetypes,
        "role_bindings": role_bindings,
        "templates": templates,
        "goal_variants": goals,
        "unit_inputs": _build_unit_inputs(archetype, candidate),
        "units": [],
        "layout_constraints": layout_constraints,
        "minimal_pair_edges": deepcopy(
            archetype.get("minimal_pair_edges") or []
        ),
        "evidence": {
            "assignment": assignment_evidence,
            "minimal_pair_contracts": deepcopy(
                archetype.get("minimal_pair_edges") or []
            ),
        },
        "export": {
            "legacy_scene_family": {
                "templates": deepcopy(templates),
                "variants": [
                    str(row["export"]["variant"]) for row in goals
                ],
                "meta": {
                    "realized_family_id": realized_id,
                    "abstract_family_id": family_id,
                    "assignment_id": assignment_id,
                    "archetype_ids": expected_archetypes,
                },
            }
        },
        "provenance": {
            "generator": "if_benchmark.realization",
            "config_hash": artifact_hash(CONFIG),
            "sources": [
                {
                    "kind": "assignment_candidate",
                    "hash": artifact_hash(candidate),
                    "record_id": assignment_id,
                },
                {
                    "kind": "goal_variant_assignment",
                    "hash": str(goal_artifact.get("artifact_hash", "")),
                    "record_id": assignment_id,
                },
            ],
        },
    }
    return canonical_data(result)


def preflight_realized_family(
    realized: Mapping[str, Any],
) -> dict[str, Any]:
    """Run the four M10 structural preflights."""

    checks: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    def record(name: str, callback: Callable[[], Any]) -> Any:
        try:
            detail = callback()
            checks.append({"name": name, "status": "passed", "detail": detail})
            return detail
        except RealizationError as exc:
            failures.append(exc.to_record())
        except Exception as exc:
            failures.append({"code": f"{name}_failed", "message": str(exc)})
        checks.append({"name": name, "status": "failed"})
        return None

    def goal_roundtrip() -> dict[str, int]:
        count = 0
        for variant in realized.get("goal_variants", []):
            encoded = encode_legacy_variant(variant["goal"])
            if encoded != variant["export"]["variant"]:
                raise _error(
                    "goal_export_mismatch",
                    f"variant {variant['id']} export differs from structured goal",
                )
            if canonical_data(decode_legacy_variant(encoded)) != canonical_data(
                variant["goal"]
            ):
                raise _error(
                    "goal_roundtrip_mismatch",
                    f"variant {variant['id']} failed structured round trip",
                )
            if variant.get("assignment_id") != realized.get("assignment_id"):
                raise _error(
                    "cross_assignment_variant",
                    f"variant {variant['id']} belongs to another assignment",
                )
            count += 1
        return {"goal_variant_count": count}

    def minimal_pairs() -> dict[str, int]:
        variants = {
            str(row["id"]): row for row in realized.get("goal_variants", [])
        }
        for edge in realized.get("minimal_pair_edges", []):
            left = variants.get(str(edge.get("from")))
            right = variants.get(str(edge.get("to")))
            if left is None or right is None:
                raise _error(
                    "minimal_pair_variant_missing",
                    f"edge references missing variants: {edge}",
                )
            actual = _static_edge_axis(left["goal"], right["goal"])
            if actual != edge.get("axis"):
                raise _error(
                    "minimal_pair_axis_mismatch",
                    f"edge {edge['from']}->{edge['to']} expected "
                    f"{edge.get('axis')} but static projection found {actual}",
                )
        return {"minimal_pair_edge_count": len(realized.get("minimal_pair_edges", []))}

    def checker_alignment() -> dict[str, Any]:
        plans = [
            _checker_plan(realized, variant)
            for variant in realized.get("goal_variants", [])
        ]
        return {"checker_plan_count": len(plans), "plans": plans}

    def unit_builder() -> dict[str, int]:
        inputs = list(realized.get("unit_inputs", []))
        role_ids = set(realized.get("role_bindings", {}))
        covered_roles = {
            str(role) for row in inputs for role in row.get("roles", [])
        }
        if covered_roles != role_ids:
            raise _error(
                "unit_role_coverage_mismatch",
                f"unit inputs cover {sorted(covered_roles)}, expected {sorted(role_ids)}",
            )
        local_roles = {
            str(role)
            for row in inputs
            if row.get("type") == "LandmarkRelationUnit"
            for role in row.get("roles", [])
        }
        object_roles = {
            str(role)
            for row in inputs
            if row.get("type") == "ObjectUnit"
            for role in row.get("roles", [])
        }
        if local_roles & object_roles:
            raise _error(
                "unit_role_double_placement",
                f"local relation roles also have ObjectUnit inputs: "
                f"{sorted(local_roles & object_roles)}",
            )
        requirement_ids = {
            str(row["id"])
            for row in realized["evidence"]["assignment"][
                "layout_requirements"
            ]
        }
        covered_constraints = {
            str(value)
            for row in inputs
            for value in row.get("source_constraints", [])
        }
        if requirement_ids - covered_constraints:
            raise _error(
                "unit_constraint_unmapped",
                f"requirements lack a unit input mapping: "
                f"{sorted(requirement_ids - covered_constraints)}",
            )
        for row in inputs:
            if row.get("type") == "ReceiverReservationUnit":
                receiver_roles = set(row.get("roles", []))
                manipulated = {
                    variant["role_refs"]["subject"]
                    for variant in realized.get("goal_variants", [])
                    if variant["goal"].get("action_family") == "O2O_relation"
                }
                if receiver_roles & manipulated:
                    raise _error(
                        "receiver_contains_subject",
                        "ReceiverReservationUnit input contains a manipulated subject",
                    )
                params = row.get("params") or {}
                declared_receivers = {
                    str(role) for role in params.get("receiver_roles", [])
                }
                declared_subjects = {
                    str(role) for role in params.get("subject_roles", [])
                }
                if declared_receivers != receiver_roles:
                    raise _error(
                        "receiver_endpoint_mismatch",
                        "ReceiverReservationUnit receiver_roles do not match membership",
                    )
                if not declared_subjects or declared_subjects - role_ids:
                    raise _error(
                        "receiver_subject_endpoint_missing",
                        "ReceiverReservationUnit must reference valid subject roles",
                    )
                if declared_subjects & receiver_roles:
                    raise _error(
                        "receiver_contains_subject",
                        "ReceiverReservationUnit subject endpoints overlap receiver membership",
                    )
                relations = {
                    str(value) for value in params.get("relations", [])
                }
                endpoints = params.get("relation_endpoints") or {}
                if set(endpoints) != relations:
                    raise _error(
                        "receiver_relation_endpoint_mismatch",
                        "ReceiverReservationUnit relations and endpoints differ",
                    )
                pair_preconditions = row.get("pair_preconditions") or {}
                for relation, endpoint in endpoints.items():
                    endpoint_subjects = {
                        str(role)
                        for role in endpoint.get("subject_roles", [])
                    }
                    endpoint_receivers = {
                        str(role)
                        for role in endpoint.get("receiver_roles", [])
                    }
                    if (
                        not endpoint_subjects
                        or endpoint_subjects - declared_subjects
                        or not endpoint_receivers
                        or endpoint_receivers - receiver_roles
                    ):
                        raise _error(
                            "receiver_relation_endpoint_invalid",
                            f"invalid endpoints for receiver relation {relation}",
                        )
                    pair_ids = [
                        str(value)
                        for value in endpoint.get("pair_constraint_ids", [])
                    ]
                    if relation in PAIR_REQUIRED_RELATIONS:
                        if not pair_ids:
                            raise _error(
                                "receiver_pair_precondition_missing",
                                f"{relation} receiver endpoint lacks pair constraint ids",
                            )
                        missing = [
                            value
                            for value in pair_ids
                            if value not in pair_preconditions
                        ]
                        if missing:
                            raise _error(
                                "receiver_pair_precondition_missing",
                                f"{relation} receiver endpoint lacks pair evidence "
                                f"for {missing}",
                            )
                        expected_prefix = (
                            PAIR_RELATION_BY_GOAL_RELATION[relation] + "|"
                        )
                        for pair_id in pair_ids:
                            pair_key = str(
                                pair_preconditions[pair_id].get("pair_key") or ""
                            )
                            if not pair_key.startswith(expected_prefix):
                                raise _error(
                                    "receiver_pair_relation_mismatch",
                                    f"{pair_id} does not prove {relation}",
                                )
                    elif pair_ids:
                        raise _error(
                            "unexpected_receiver_pair_precondition",
                            f"{relation} must not invent pair constraints",
                        )
        return {
            "unit_input_count": len(inputs),
            "mapped_constraint_count": len(covered_constraints),
        }

    record("goal_roundtrip", goal_roundtrip)
    record("minimal_pair_contract", minimal_pairs)
    checker_detail = record("checker_alignment", checker_alignment)
    if checker_detail is not None:
        checks[-1]["detail"] = {
            "checker_plan_count": checker_detail["checker_plan_count"]
        }
    record("unit_builder", unit_builder)
    schema_result = validate_document(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "realized_scene_family",
            "realized_scene_family": canonical_data(realized),
        }
    )
    if schema_result["valid"]:
        checks.append({"name": "schema", "status": "passed"})
    else:
        failures.extend(
            {
                "code": "realized_schema_invalid",
                "message": f"{row['path']}: {row['message']}",
            }
            for row in schema_result["errors"]
        )
        checks.append({"name": "schema", "status": "failed"})
    return {
        "status": "passed" if not failures else "failed",
        "checks": checks,
        "checker_plans": (
            checker_detail.get("plans", []) if checker_detail else []
        ),
        "failures": failures,
    }


def _source_hashes(
    domain: Mapping[str, Any],
    goal_artifact: Mapping[str, Any],
    *,
    families_hash: str,
    archetypes_hash: str,
    families_path: Path,
    archetypes_path: Path,
    domain_path: Path,
    goal_path: Path,
) -> dict[str, str]:
    return dict(
        sorted(
            {
                str(families_path): families_hash,
                str(archetypes_path): archetypes_hash,
                str(domain_path): str(domain.get("domain_hash", "")),
                str(goal_path): str(goal_artifact.get("artifact_hash", "")),
            }.items()
        )
    )


def _validate_goal_artifact_source(
    artifact: Mapping[str, Any],
    *,
    family_id: str,
    assignment_id: str,
    expected_source_hashes: Mapping[str, str],
) -> None:
    if artifact.get("schema_version") != GOAL_SCHEMA_VERSION:
        raise ValueError("goal artifact schema is stale")
    if artifact.get("artifact_kind") != "goal_variant_assignment":
        raise ValueError("goal artifact kind is invalid")
    if artifact.get("goal_version") != GOAL_VERSION:
        raise ValueError("goal artifact version is stale")
    if str(artifact.get("abstract_family_id")) != family_id:
        raise ValueError("goal artifact family join mismatch")
    if str(artifact.get("assignment_id")) != assignment_id:
        raise ValueError("goal artifact assignment join mismatch")
    if artifact.get("artifact_hash") != artifact_hash(
        artifact, omit_keys=("artifact_hash",)
    ):
        raise ValueError("goal artifact self hash mismatch")
    observed_sources = artifact.get("source_hashes") or {}
    mismatches = {
        path: (observed_sources.get(path), expected)
        for path, expected in expected_source_hashes.items()
        if observed_sources.get(path) != expected
    }
    if mismatches:
        raise ValueError(f"goal artifact source join mismatch: {mismatches}")


def _artifact(
    realized: Mapping[str, Any],
    preflight: Mapping[str, Any],
    *,
    source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "realized_scene_family",
        "realization_version": REALIZATION_VERSION,
        "realized_family_id": realized["id"],
        "abstract_family_id": realized["abstract_family_id"],
        "assignment_id": realized["assignment_id"],
        "source_hashes": dict(source_hashes),
        "config_hash": artifact_hash(CONFIG),
        "preflight": canonical_data(preflight),
        "realized_scene_family": canonical_data(realized),
    }
    result["artifact_hash"] = artifact_hash(result)
    return result


def _self_hash_valid(artifact: Mapping[str, Any]) -> bool:
    return artifact.get("artifact_hash") == artifact_hash(
        artifact, omit_keys=("artifact_hash",)
    )


def _reusable(
    artifact: Mapping[str, Any],
    *,
    realized_id: str,
    family_id: str,
    assignment_id: str,
    source_hashes: Mapping[str, str],
) -> bool:
    return (
        artifact.get("schema_version") == SCHEMA_VERSION
        and artifact.get("artifact_kind") == "realized_scene_family"
        and artifact.get("realization_version") == REALIZATION_VERSION
        and artifact.get("realized_family_id") == realized_id
        and artifact.get("abstract_family_id") == family_id
        and artifact.get("assignment_id") == assignment_id
        and artifact.get("source_hashes") == dict(source_hashes)
        and artifact.get("config_hash") == artifact_hash(CONFIG)
        and artifact.get("preflight", {}).get("status") == "passed"
        and _self_hash_valid(artifact)
    )


def _library_inputs(
    families_path: Path, archetypes_path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    families_doc = _load_yaml(families_path)
    archetypes_doc = _load_yaml(archetypes_path)
    family_by_id = {
        str(row["id"]): row for row in families_doc.get("families", [])
    }
    archetype_by_id = {
        str(row["id"]): row for row in archetypes_doc.get("archetypes", [])
    }
    return families_doc, archetypes_doc, family_by_id, archetype_by_id


def _reports(
    rows: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
    *,
    source_hashes: Mapping[str, str],
    expected_count: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    family_counts = Counter(str(row["abstract_family_id"]) for row in rows)
    axis_counts = Counter(
        str(axis)
        for row in rows
        for axis in row.get("minimal_pair_axes", [])
    )
    summary = {
        "expected_realized_family_count": expected_count,
        "realized_family_count": len(rows),
        "goal_variant_count": sum(
            int(row["goal_variant_count"]) for row in rows
        ),
        "unit_input_count": sum(int(row["unit_input_count"]) for row in rows),
        "minimal_pair_edge_count": sum(
            int(row["minimal_pair_edge_count"]) for row in rows
        ),
        "preflight_pass_count": sum(
            row.get("preflight_status") == "passed" for row in rows
        ),
        "failure_count": len(failures),
    }
    acceptance_ready = (
        not failures
        and summary["realized_family_count"] == expected_count
        and summary["preflight_pass_count"] == expected_count
    )
    artifact_rows = sorted(
        (
            {
                "id": row["realized_family_id"],
                "hash": row["artifact_hash"],
            }
            for row in rows
        ),
        key=lambda item: item["id"],
    )
    audit: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "realization_audit",
        "realization_version": REALIZATION_VERSION,
        "config_hash": artifact_hash(CONFIG),
        "source_hashes": dict(source_hashes),
        "summary": summary,
        "family_counts": dict(sorted(family_counts.items())),
        "minimal_pair_axis_counts": dict(sorted(axis_counts.items())),
        "artifact_set_hash": artifact_hash(artifact_rows),
        "acceptance_ready": acceptance_ready,
    }
    audit["audit_hash"] = artifact_hash(audit)
    fail_report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "preflight_fail_report",
        "realization_version": REALIZATION_VERSION,
        "config_hash": artifact_hash(CONFIG),
        "failure_count": len(failures),
        "failures": canonical_data(list(failures)),
        "acceptance_ready": acceptance_ready,
    }
    fail_report["report_hash"] = artifact_hash(fail_report)
    return audit, fail_report


_REALIZATION_WORKER_CONTEXT: dict[str, Any] = {}


def _init_realization_worker(context: Mapping[str, Any]) -> None:
    global _REALIZATION_WORKER_CONTEXT
    _REALIZATION_WORKER_CONTEXT = dict(context)


def _validated_worker_count(workers: int) -> int:
    worker_count = int(workers)
    if worker_count < 1:
        raise ValueError("workers must be a positive integer")
    return worker_count


def _run_family_workers(
    function: Callable[[Mapping[str, Any]], dict[str, Any]],
    families: Sequence[Mapping[str, Any]],
    *,
    workers: int,
    context: Mapping[str, Any],
) -> Iterable[dict[str, Any]]:
    """Run one independent task per abstract family in source order."""

    worker_count = min(
        _validated_worker_count(workers), max(1, len(families))
    )
    if worker_count == 1:
        _init_realization_worker(context)
        for family in families:
            yield function(family)
        return

    try:
        process_context = multiprocessing.get_context("fork")
    except ValueError:  # pragma: no cover - non-POSIX compatibility fallback
        process_context = multiprocessing.get_context()
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=process_context,
        initializer=_init_realization_worker,
        initargs=(context,),
    ) as executor:
        # executor.map preserves the canonical AbstractSceneFamily order even
        # when a later family completes first.
        yield from executor.map(function, families, chunksize=1)


def _realization_worker_context(
    *,
    archetype_by_id: Mapping[str, Mapping[str, Any]],
    assignment_dir: Path,
    goal_dir: Path,
    output_dir: Path,
    families_path: Path,
    archetypes_path: Path,
    report_sources: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "archetype_by_id": dict(archetype_by_id),
        "assignment_dir": assignment_dir,
        "goal_dir": goal_dir,
        "output_dir": output_dir,
        "families_path": families_path,
        "archetypes_path": archetypes_path,
        "report_sources": dict(report_sources),
    }


def _family_inputs(
    family: Mapping[str, Any],
) -> tuple[str, Mapping[str, Any], Mapping[str, Any], Path]:
    context = _REALIZATION_WORKER_CONTEXT
    family_id = str(family["id"])
    domain_path = Path(context["assignment_dir"]) / f"{family_id}.yml"
    domain = _load_yaml(domain_path)
    archetype = context["archetype_by_id"][str(family["archetypes"][0])]
    validate_assignment_domain(
        domain,
        family=family,
        archetype=archetype,
    )
    return family_id, archetype, domain, domain_path


def _candidate_inputs(
    *,
    family_id: str,
    domain: Mapping[str, Any],
    domain_path: Path,
    candidate: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Path, dict[str, str]]:
    context = _REALIZATION_WORKER_CONTEXT
    assignment_id = str(candidate["id"])
    goal_path = (
        Path(context["goal_dir"])
        / family_id
        / f"{assignment_id}.yml"
    )
    goal_artifact = _load_yaml(goal_path)
    report_sources = context["report_sources"]
    families_path = Path(context["families_path"])
    archetypes_path = Path(context["archetypes_path"])
    _validate_goal_artifact_source(
        goal_artifact,
        family_id=family_id,
        assignment_id=assignment_id,
        expected_source_hashes={
            str(families_path): report_sources[str(families_path)],
            str(archetypes_path): report_sources[str(archetypes_path)],
            str(domain_path): str(domain["domain_hash"]),
        },
    )
    sources = _source_hashes(
        domain,
        goal_artifact,
        families_hash=report_sources[str(families_path)],
        archetypes_hash=report_sources[str(archetypes_path)],
        families_path=families_path,
        archetypes_path=archetypes_path,
        domain_path=domain_path,
        goal_path=goal_path,
    )
    return goal_artifact, goal_path, sources


def _freeze_realization_family(
    family: Mapping[str, Any],
) -> dict[str, Any]:
    family_id = str(family.get("id") or "<unknown-family>")
    domain_path = (
        Path(_REALIZATION_WORKER_CONTEXT["assignment_dir"])
        / f"{family_id}.yml"
    )
    try:
        family_id, archetype, domain, domain_path = _family_inputs(family)
    except Exception as exc:
        return {
            "family_id": family_id,
            "expected_names": [],
            "expected_count": 0,
            "rows": [],
            "failures": [
                _error(
                    "realization_input_missing",
                    str(exc),
                    family_id=family_id,
                    path=str(domain_path),
                ).to_record()
            ],
        }

    output_dir = Path(_REALIZATION_WORKER_CONTEXT["output_dir"])
    expected_names: list[str] = []
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    candidates = list(domain.get("candidates", []))
    for candidate in candidates:
        assignment_id = str(candidate.get("id", ""))
        realized_id = _realized_id(family_id, assignment_id)
        name = f"{realized_id}.yml"
        expected_names.append(name)
        path = output_dir / name
        try:
            goal_artifact, _, sources = _candidate_inputs(
                family_id=family_id,
                domain=domain,
                domain_path=domain_path,
                candidate=candidate,
            )
            existing = _try_load_yaml(path) if path.is_file() else None
            if existing is not None and _reusable(
                existing,
                realized_id=realized_id,
                family_id=family_id,
                assignment_id=assignment_id,
                source_hashes=sources,
            ):
                artifact = existing
            else:
                realized = realize_scene_family(
                    family, archetype, candidate, goal_artifact
                )
                preflight = preflight_realized_family(realized)
                if preflight["status"] != "passed":
                    raise _error(
                        "preflight_failed",
                        str(preflight["failures"]),
                        family_id=family_id,
                        assignment_id=assignment_id,
                    )
                realized["evidence"]["checker_plans"] = deepcopy(
                    preflight["checker_plans"]
                )
                artifact = _artifact(
                    realized, preflight, source_hashes=sources
                )
                _write_yaml(path, artifact)
            realized = artifact["realized_scene_family"]
            rows.append(
                {
                    "realized_family_id": realized_id,
                    "abstract_family_id": family_id,
                    "artifact_hash": artifact["artifact_hash"],
                    "goal_variant_count": len(realized["goal_variants"]),
                    "unit_input_count": len(realized["unit_inputs"]),
                    "minimal_pair_edge_count": len(
                        realized["minimal_pair_edges"]
                    ),
                    "minimal_pair_axes": [
                        str(edge["axis"])
                        for edge in realized["minimal_pair_edges"]
                    ],
                    "preflight_status": artifact["preflight"]["status"],
                }
            )
        except RealizationError as exc:
            if exc.family_id is None:
                exc.family_id = family_id
            if exc.assignment_id is None:
                exc.assignment_id = assignment_id
            failures.append(exc.to_record())
            path.unlink(missing_ok=True)
        except Exception as exc:
            failures.append(
                _error(
                    "realization_internal_error",
                    str(exc),
                    family_id=family_id,
                    assignment_id=assignment_id,
                ).to_record()
            )
            path.unlink(missing_ok=True)
    return {
        "family_id": family_id,
        "expected_names": expected_names,
        "expected_count": len(candidates),
        "rows": rows,
        "failures": failures,
    }


def freeze_realized_families(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    assignment_dir: Path = DEFAULT_ASSIGNMENT_DIR,
    goal_dir: Path = DEFAULT_GOAL_DIR,
    families_path: Path = DEFAULT_FAMILIES_PATH,
    archetypes_path: Path = DEFAULT_ARCHETYPES_PATH,
    workers: int = 1,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Freeze one M10 artifact per published assignment."""

    (
        families_doc,
        archetypes_doc,
        _,
        archetype_by_id,
    ) = _library_inputs(families_path, archetypes_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_names: set[str] = set()
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    expected_count = 0
    report_sources = {
        str(families_path): artifact_hash(families_doc),
        str(archetypes_path): artifact_hash(archetypes_doc),
    }
    families = list(families_doc.get("families", []))
    context = _realization_worker_context(
        archetype_by_id=archetype_by_id,
        assignment_dir=assignment_dir,
        goal_dir=goal_dir,
        output_dir=output_dir,
        families_path=families_path,
        archetypes_path=archetypes_path,
        report_sources=report_sources,
    )
    results = _run_family_workers(
        _freeze_realization_family,
        families,
        workers=workers,
        context=context,
    )
    for index, result in enumerate(results, start=1):
        expected_names.update(result["expected_names"])
        expected_count += int(result["expected_count"])
        rows.extend(result["rows"])
        failures.extend(result["failures"])
        if progress is not None:
            progress(index, len(families), str(result["family_id"]))
    for stale_path in sorted(output_dir.glob("rsf_*.yml")):
        if stale_path.name not in expected_names:
            stale_path.unlink()
    audit, fail_report = _reports(
        rows,
        failures,
        source_hashes=report_sources,
        expected_count=expected_count,
    )
    _write_yaml(output_dir / AUDIT_FILE, audit)
    _write_yaml(output_dir / FAIL_FILE, fail_report)
    return {
        "acceptance_ready": audit["acceptance_ready"],
        "summary": audit["summary"],
        AUDIT_FILE: audit,
        FAIL_FILE: fail_report,
    }


def _expected_artifacts(
    *,
    assignment_dir: Path,
    goal_dir: Path,
    families_path: Path,
    archetypes_path: Path,
    progress: Callable[[int, int, str], None] | None,
) -> Iterable[
    tuple[
        Mapping[str, Any],
        Mapping[str, Any],
        Mapping[str, Any],
        Mapping[str, Any],
        str,
        dict[str, str],
    ]
]:
    families_doc, archetypes_doc, _, archetype_by_id = _library_inputs(
        families_path, archetypes_path
    )
    families_hash = artifact_hash(families_doc)
    archetypes_hash = artifact_hash(archetypes_doc)
    families = list(families_doc.get("families", []))
    for index, family in enumerate(families, start=1):
        family_id = str(family["id"])
        if progress is not None:
            progress(index, len(families), family_id)
        domain_path = assignment_dir / f"{family_id}.yml"
        domain = _load_yaml(domain_path)
        archetype = archetype_by_id[str(family["archetypes"][0])]
        validate_assignment_domain(
            domain,
            family=family,
            archetype=archetype,
        )
        for candidate in domain.get("candidates", []):
            assignment_id = str(candidate["id"])
            goal_path = goal_dir / family_id / f"{assignment_id}.yml"
            goal_artifact = _load_yaml(goal_path)
            _validate_goal_artifact_source(
                goal_artifact,
                family_id=family_id,
                assignment_id=assignment_id,
                expected_source_hashes={
                    str(families_path): families_hash,
                    str(archetypes_path): archetypes_hash,
                    str(domain_path): str(domain["domain_hash"]),
                },
            )
            sources = _source_hashes(
                domain,
                goal_artifact,
                families_hash=families_hash,
                archetypes_hash=archetypes_hash,
                families_path=families_path,
                archetypes_path=archetypes_path,
                domain_path=domain_path,
                goal_path=goal_path,
            )
            yield family, archetype, candidate, goal_artifact, _realized_id(
                family_id, assignment_id
            ), sources


def _check_realization_family(
    family: Mapping[str, Any],
) -> dict[str, Any]:
    family_id = str(family.get("id") or "<unknown-family>")
    try:
        family_id, _, domain, domain_path = _family_inputs(family)
    except Exception as exc:
        return {
            "family_id": family_id,
            "expected_names": [],
            "stale_paths": [f"family {family_id}: {exc}"],
            "count": 0,
            "hash_rows": [],
        }

    output_dir = Path(_REALIZATION_WORKER_CONTEXT["output_dir"])
    expected_names: list[str] = []
    stale_paths: list[str] = []
    count = 0
    hash_rows: list[dict[str, str]] = []
    for candidate in domain.get("candidates", []):
        assignment_id = str(candidate.get("id", ""))
        realized_id = _realized_id(family_id, assignment_id)
        name = f"{realized_id}.yml"
        expected_names.append(name)
        path = output_dir / name
        try:
            _, _, sources = _candidate_inputs(
                family_id=family_id,
                domain=domain,
                domain_path=domain_path,
                candidate=candidate,
            )
            artifact = _try_load_yaml(path)
            if artifact is None or not _reusable(
                artifact,
                realized_id=realized_id,
                family_id=family_id,
                assignment_id=assignment_id,
                source_hashes=sources,
            ):
                stale_paths.append(str(path))
            else:
                count += 1
                hash_rows.append(
                    {
                        "id": realized_id,
                        "hash": str(artifact["artifact_hash"]),
                    }
                )
        except Exception as exc:
            stale_paths.append(
                f"family {family_id} assignment {assignment_id}: {exc}"
            )
    return {
        "family_id": family_id,
        "expected_names": expected_names,
        "stale_paths": stale_paths,
        "count": count,
        "hash_rows": hash_rows,
    }


def check_realized_families(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    assignment_dir: Path = DEFAULT_ASSIGNMENT_DIR,
    goal_dir: Path = DEFAULT_GOAL_DIR,
    families_path: Path = DEFAULT_FAMILIES_PATH,
    archetypes_path: Path = DEFAULT_ARCHETYPES_PATH,
    workers: int = 1,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    families_doc, archetypes_doc, _, archetype_by_id = _library_inputs(
        families_path, archetypes_path
    )
    report_sources = {
        str(families_path): artifact_hash(families_doc),
        str(archetypes_path): artifact_hash(archetypes_doc),
    }
    families = list(families_doc.get("families", []))
    context = _realization_worker_context(
        archetype_by_id=archetype_by_id,
        assignment_dir=assignment_dir,
        goal_dir=goal_dir,
        output_dir=output_dir,
        families_path=families_path,
        archetypes_path=archetypes_path,
        report_sources=report_sources,
    )
    stale_paths: list[str] = []
    expected_names: set[str] = set()
    count = 0
    hash_rows: list[dict[str, str]] = []
    results = _run_family_workers(
        _check_realization_family,
        families,
        workers=workers,
        context=context,
    )
    for index, result in enumerate(results, start=1):
        expected_names.update(result["expected_names"])
        stale_paths.extend(result["stale_paths"])
        count += int(result["count"])
        hash_rows.extend(result["hash_rows"])
        if progress is not None:
            progress(index, len(families), str(result["family_id"]))
    actual_names = {path.name for path in output_dir.glob("rsf_*.yml")}
    stale_paths.extend(
        str(output_dir / name) for name in sorted(actual_names - expected_names)
    )
    audit = _try_load_yaml(output_dir / AUDIT_FILE)
    fail_report = _try_load_yaml(output_dir / FAIL_FILE)
    if (
        audit is None
        or not audit.get("acceptance_ready")
        or audit.get("summary", {}).get("realized_family_count") != len(
            expected_names
        )
        or audit.get("source_hashes") != report_sources
        or audit.get("config_hash") != artifact_hash(CONFIG)
        or audit.get("artifact_set_hash")
        != artifact_hash(sorted(hash_rows, key=lambda row: row["id"]))
        or audit.get("audit_hash")
        != artifact_hash(audit, omit_keys=("audit_hash",))
    ):
        stale_paths.append(str(output_dir / AUDIT_FILE))
    if (
        fail_report is None
        or not fail_report.get("acceptance_ready")
        or fail_report.get("config_hash") != artifact_hash(CONFIG)
        or fail_report.get("report_hash")
        != artifact_hash(fail_report, omit_keys=("report_hash",))
    ):
        stale_paths.append(str(output_dir / FAIL_FILE))
    return {
        "stale": bool(stale_paths),
        "stale_paths": sorted(set(stale_paths)),
        "realized_family_count": count,
    }


def _validate_realization_family(
    family: Mapping[str, Any],
) -> dict[str, Any]:
    family_id = str(family.get("id") or "<unknown-family>")
    try:
        family_id, archetype, domain, domain_path = _family_inputs(family)
    except Exception as exc:
        return {
            "family_id": family_id,
            "errors": [f"family {family_id}: {exc}"],
            "count": 0,
        }

    output_dir = Path(_REALIZATION_WORKER_CONTEXT["output_dir"])
    errors: list[str] = []
    count = 0
    for candidate in domain.get("candidates", []):
        assignment_id = str(candidate.get("id", ""))
        realized_id = _realized_id(family_id, assignment_id)
        path = output_dir / f"{realized_id}.yml"
        try:
            goal_artifact, _, sources = _candidate_inputs(
                family_id=family_id,
                domain=domain,
                domain_path=domain_path,
                candidate=candidate,
            )
            artifact = _try_load_yaml(path)
            if artifact is None:
                errors.append(f"missing artifact: {path}")
                continue
            if not _reusable(
                artifact,
                realized_id=realized_id,
                family_id=family_id,
                assignment_id=assignment_id,
                source_hashes=sources,
            ):
                errors.append(f"stale or invalid envelope: {path}")
                continue
            rebuilt = realize_scene_family(
                family, archetype, candidate, goal_artifact
            )
            rebuilt_preflight = preflight_realized_family(rebuilt)
            if rebuilt_preflight["status"] != "passed":
                errors.append(
                    f"rebuilt preflight failed: {path}: "
                    f"{rebuilt_preflight['failures']}"
                )
                continue
            rebuilt["evidence"]["checker_plans"] = deepcopy(
                rebuilt_preflight["checker_plans"]
            )
            if canonical_data(rebuilt) != canonical_data(
                artifact["realized_scene_family"]
            ):
                errors.append(f"realized payload mismatch: {path}")
                continue
            count += 1
        except Exception as exc:
            errors.append(
                f"family {family_id} assignment {assignment_id} "
                f"({path}): {exc}"
            )
    return {"family_id": family_id, "errors": errors, "count": count}


def validate_realized_families(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    assignment_dir: Path = DEFAULT_ASSIGNMENT_DIR,
    goal_dir: Path = DEFAULT_GOAL_DIR,
    families_path: Path = DEFAULT_FAMILIES_PATH,
    archetypes_path: Path = DEFAULT_ARCHETYPES_PATH,
    workers: int = 1,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    families_doc, archetypes_doc, _, archetype_by_id = _library_inputs(
        families_path, archetypes_path
    )
    report_sources = {
        str(families_path): artifact_hash(families_doc),
        str(archetypes_path): artifact_hash(archetypes_doc),
    }
    families = list(families_doc.get("families", []))
    context = _realization_worker_context(
        archetype_by_id=archetype_by_id,
        assignment_dir=assignment_dir,
        goal_dir=goal_dir,
        output_dir=output_dir,
        families_path=families_path,
        archetypes_path=archetypes_path,
        report_sources=report_sources,
    )
    errors: list[str] = []
    count = 0
    results = _run_family_workers(
        _validate_realization_family,
        families,
        workers=workers,
        context=context,
    )
    for index, result in enumerate(results, start=1):
        errors.extend(result["errors"])
        count += int(result["count"])
        if progress is not None:
            progress(index, len(families), str(result["family_id"]))
    check = check_realized_families(
        output_dir=output_dir,
        assignment_dir=assignment_dir,
        goal_dir=goal_dir,
        families_path=families_path,
        archetypes_path=archetypes_path,
        workers=workers,
        progress=None,
    )
    errors.extend(f"stale: {path}" for path in check["stale_paths"])
    return {
        "valid": not errors,
        "errors": errors,
        "realized_family_count": count,
    }
