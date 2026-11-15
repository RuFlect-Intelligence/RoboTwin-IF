"""Milestone 9 GoalVariant realization and artifact lifecycle.

This module is deliberately simulator-independent.  It consumes the frozen
Milestone 7 family library and Milestone 8 complete assignment candidates.  It
does not reopen assignment decisions and does not construct layouts.
"""

from __future__ import annotations

import multiprocessing
import os
import re
import shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence

import jsonschema
import yaml

from if_benchmark.assignments.core import validate_assignment_domain
from if_benchmark.schema import artifact_hash, canonical_yaml
from if_benchmark.schema.cli import build_schema_artifact
from if_benchmark.schema.serialization import canonical_data
from if_benchmark.orientation_adjustment import (
    ORIENTATION_ADJUSTMENT_STATE,
    ORIENTATION_ADJUSTMENT_VERB,
    normalize_orientation_adjustment_params,
)


SCHEMA_VERSION = 1
GOAL_VERSION = 3

DEFAULT_ARCHETYPES_PATH = Path("if_benchmark/data/families/archetypes.yml")
DEFAULT_FAMILIES_PATH = Path("if_benchmark/data/families/abstract_families.yml")
DEFAULT_ASSIGNMENT_DIR = Path("if_benchmark/data/assignment_domains")
DEFAULT_OUTPUT_DIR = Path("if_benchmark/data/goal_variants")
DEFAULT_PROCESS_WORKERS = min(32, os.cpu_count() or 1)

ROUNDTRIP_FILE = "goal_roundtrip_report.yml"
FAIL_FILE = "materialization_fail_report.yml"

GOAL_SLOTS = ("subject", "target", "subject2", "target2")
DESCRIPTOR_KINDS = ("attribute", "spatial", "ordinal", "part")
SPATIAL_PREDICATE_ORDER = (
    "left_of",
    "right_of",
    "in_front_of",
    "behind",
    "in",
    "on",
)
ATTRIBUTE_SUBTYPES = ("type", "color", "size", "shape", "material")
ORDINAL_BASES = ("position", "size", "robot", "landmark")
ARMS = ("left", "right")

RELATION_BY_VERB = {
    "place_in": "in",
    "place_on": "on",
    "place_next_to": "next_to",
    "stack": "stack",
    "place_left_of": "left_of",
    "place_right_of": "right_of",
    "place_in_front_of": "in_front_of",
    "place_behind": "behind",
    "place_between": "between",
    "place_left_of_pair": "left_of_pair",
    "place_right_of_pair": "right_of_pair",
}
VERB_BY_RELATION = {relation: verb for verb, relation in RELATION_BY_VERB.items()}
TWO_TARGET_RELATIONS = frozenset(("between", "left_of_pair", "right_of_pair"))
A2O_STATE_BY_VERB = {
    "pick": "acquired",
    "open": "open",
    "close": "close",
    "switch_on": "on",
    "switch_off": "off",
    ORIENTATION_ADJUSTMENT_VERB: ORIENTATION_ADJUSTMENT_STATE,
}
A2O_EVENT_VERBS = frozenset(("click", "press"))

CONFIG: dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "goal_version": GOAL_VERSION,
    "action_families": {
        "A2O_state": A2O_STATE_BY_VERB,
        "A2O_event": sorted(A2O_EVENT_VERBS),
        "O2O_relation": RELATION_BY_VERB,
    },
    "goal_slots": list(GOAL_SLOTS),
    "descriptor_kinds": list(DESCRIPTOR_KINDS),
    "spatial_predicate_order": list(SPATIAL_PREDICATE_ORDER),
    "legacy_codec": {
        "descriptor_slots": list(GOAL_SLOTS),
        "descriptor_order": list(DESCRIPTOR_KINDS),
        "arm_prefixed_state": True,
        "strict": True,
    },
}

_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9_.-]+$")
_ENTITY_KEY = re.compile(r"^obj_[A-Za-z0-9_]+$")
_SCHEMA_VALIDATOR: jsonschema.Draft202012Validator | None = None


@dataclass
class GoalMaterializationError(ValueError):
    """Stable, reportable failure at the goal realization boundary."""

    code: str
    message: str
    family_id: str | None = None
    assignment_id: str | None = None
    schema_id: str | None = None
    path: str | None = None

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"

    def to_record(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
        }
        for key in ("family_id", "assignment_id", "schema_id", "path"):
            value = getattr(self, key)
            if value is not None:
                row[key] = value
        return row


def _error(
    code: str,
    message: str,
    *,
    family_id: str | None = None,
    assignment_id: str | None = None,
    schema_id: str | None = None,
    path: str | None = None,
) -> GoalMaterializationError:
    return GoalMaterializationError(
        code,
        message,
        family_id=family_id,
        assignment_id=assignment_id,
        schema_id=schema_id,
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


def _goal_schema_validator() -> jsonschema.Draft202012Validator:
    global _SCHEMA_VALIDATOR
    if _SCHEMA_VALIDATOR is None:
        _SCHEMA_VALIDATOR = jsonschema.Draft202012Validator(
            build_schema_artifact()
        )
    return _SCHEMA_VALIDATOR


def _validate_goal_variant_contract(variant: Mapping[str, Any]) -> None:
    document = {
        "schema_version": SCHEMA_VERSION,
        "kind": "goal_variant",
        "goal_variant": canonical_data(variant),
    }
    errors = sorted(
        _goal_schema_validator().iter_errors(document),
        key=lambda item: list(item.path),
    )
    if errors:
        first = errors[0]
        path = "$." + ".".join(str(part) for part in first.path)
        raise _error(
            "schema_validation_error",
            first.message,
            assignment_id=str(variant.get("assignment_id", "")) or None,
            schema_id=str(variant.get("schema_id", "")) or None,
            path=path,
        )


def canonical_action_family(goal: Mapping[str, Any]) -> str:
    """Return the canonical single-step action family."""

    verb = str(goal.get("verb", ""))
    relation = goal.get("relation")
    state = goal.get("state")
    declared_family = goal.get("action_family")
    if verb in A2O_STATE_BY_VERB:
        if relation is not None:
            raise _error(
                "action_relation_mismatch",
                f"A2O verb {verb!r} cannot carry relation {relation!r}",
            )
        expected_state = None if verb == "pick" else A2O_STATE_BY_VERB[verb]
        if state not in (None, expected_state):
            raise _error(
                "action_state_mismatch",
                f"verb {verb!r} requires state {expected_state!r}, got {state!r}",
            )
        if verb == ORIENTATION_ADJUSTMENT_VERB:
            try:
                normalized_params = normalize_orientation_adjustment_params(
                    goal.get("params") or {}
                )
            except ValueError as exc:
                raise _error(
                    "orientation_params_invalid", str(exc), path="params"
                ) from exc
            if canonical_data(goal.get("params") or {}) != canonical_data(
                normalized_params
            ):
                raise _error(
                    "orientation_params_not_canonical",
                    "orientation adjustment params are not canonical",
                    path="params",
                )
        family = "A2O_state"
        if declared_family not in (None, "", family):
            raise _error(
                "action_family_mismatch",
                f"verb {verb!r} requires action_family {family!r}, "
                f"got {declared_family!r}",
            )
        return family
    if verb in A2O_EVENT_VERBS:
        if relation is not None:
            raise _error(
                "action_relation_mismatch",
                f"A2O event verb {verb!r} cannot carry relation {relation!r}",
            )
        if state is not None:
            raise _error(
                "action_state_mismatch",
                f"A2O event verb {verb!r} cannot carry terminal state {state!r}",
            )
        family = "A2O_event"
        if declared_family not in (None, "", family):
            raise _error(
                "action_family_mismatch",
                f"verb {verb!r} requires action_family {family!r}, "
                f"got {declared_family!r}",
            )
        return family
    expected_relation = RELATION_BY_VERB.get(verb)
    if expected_relation is not None:
        if relation != expected_relation:
            raise _error(
                "action_relation_mismatch",
                f"verb {verb!r} requires relation {expected_relation!r}, got {relation!r}",
            )
        if state is not None:
            raise _error(
                "action_state_mismatch",
                f"O2O verb {verb!r} cannot carry state {state!r}",
            )
        family = "O2O_relation"
        if declared_family not in (None, "", family):
            raise _error(
                "action_family_mismatch",
                f"verb {verb!r} requires action_family {family!r}, "
                f"got {declared_family!r}",
            )
        return family
    raise _error("unsupported_action", f"unsupported canonical verb: {verb!r}")


def canonical_relation_value(goal: Mapping[str, Any]) -> str | None:
    """Return the canonical D2b relation, or ``None`` for A2O actions."""

    family = canonical_action_family(goal)
    return str(goal["relation"]) if family == "O2O_relation" else None


def canonical_state_value(goal: Mapping[str, Any]) -> str | None:
    """Return the canonical D2a value, or ``None`` for O2O actions."""

    family = canonical_action_family(goal)
    return A2O_STATE_BY_VERB[str(goal["verb"])] if family == "A2O_state" else None


def _safe_token(value: Any, *, field: str) -> str:
    text = str(value)
    if not text or _SAFE_TOKEN.fullmatch(text) is None:
        raise _error(
            "legacy_codec_unsafe_token",
            f"{field} contains a reserved or unsupported token: {text!r}",
            path=field,
        )
    return text


def _canonical_predicates(values: Iterable[Any]) -> list[str]:
    predicates = [str(value) for value in values]
    if not predicates or len(predicates) != len(set(predicates)):
        raise _error(
            "invalid_spatial_descriptor",
            "spatial predicates must be non-empty and unique",
        )
    unknown = sorted(set(predicates) - set(SPATIAL_PREDICATE_ORDER))
    if unknown:
        raise _error(
            "invalid_spatial_descriptor",
            f"unsupported spatial predicates: {unknown}",
        )
    if any(item in {"in", "on"} for item in predicates) and len(predicates) != 1:
        raise _error(
            "invalid_spatial_descriptor",
            "topological in/on predicates must stand alone",
        )
    return [item for item in SPATIAL_PREDICATE_ORDER if item in predicates]


def _encode_descriptor(kind: str, descriptor: Mapping[str, Any]) -> str:
    if kind == "attribute":
        subtype = _safe_token(descriptor.get("subtype"), field="attribute.subtype")
        if subtype not in ATTRIBUTE_SUBTYPES:
            raise _error(
                "invalid_attribute_descriptor",
                f"unsupported attribute subtype: {subtype!r}",
            )
        value = _safe_token(descriptor.get("value"), field="attribute.value")
        return f"{subtype}:{value}"
    if kind == "spatial":
        predicates = _canonical_predicates(descriptor.get("predicates", []))
        reference = _safe_token(
            descriptor.get("reference"), field="spatial.reference"
        )
        return f"{'+'.join(predicates)}:{reference}"
    if kind == "ordinal":
        basis = _safe_token(descriptor.get("basis"), field="ordinal.basis")
        if basis not in ORDINAL_BASES:
            raise _error(
                "invalid_ordinal_descriptor",
                f"unsupported ordinal basis: {basis!r}",
            )
        order = _safe_token(descriptor.get("order"), field="ordinal.order")
        reference = descriptor.get("reference")
        if reference is None:
            return f"{basis}:{order}"
        return (
            f"{basis}:{order}:"
            f"{_safe_token(reference, field='ordinal.reference')}"
        )
    if kind == "part":
        return _safe_token(descriptor.get("region"), field="part.region")
    raise _error("unsupported_descriptor", f"unsupported descriptor kind: {kind!r}")


def _encode_descriptors(descriptors: Mapping[str, Any]) -> str:
    parts: list[str] = []
    unknown_slots = sorted(set(descriptors) - set(GOAL_SLOTS))
    if unknown_slots:
        raise _error(
            "legacy_codec_error",
            f"unsupported descriptor slots: {unknown_slots}",
        )
    for slot in GOAL_SLOTS:
        bundle = descriptors.get(slot)
        if bundle is None:
            continue
        if not isinstance(bundle, Mapping):
            raise _error(
                "legacy_codec_error", f"descriptor bundle {slot!r} is not a mapping"
            )
        unknown_kinds = sorted(set(bundle) - set(DESCRIPTOR_KINDS))
        if unknown_kinds:
            raise _error(
                "legacy_codec_error",
                f"unsupported descriptor kinds in {slot}: {unknown_kinds}",
            )
        for kind in DESCRIPTOR_KINDS:
            descriptor = bundle.get(kind)
            if descriptor is not None:
                parts.append(
                    f"{slot}.{kind}={_encode_descriptor(kind, descriptor)}"
                )
    return "" if not parts else "|desc=" + ";".join(parts)


def encode_legacy_variant(goal: Mapping[str, Any]) -> str:
    """Encode a structured ConcreteGoal into the compatibility string."""

    family = canonical_action_family(goal)
    subject = _safe_token(goal.get("subject"), field="subject")
    if _ENTITY_KEY.fullmatch(subject) is None:
        raise _error(
            "legacy_codec_error", f"subject is not a concrete entity key: {subject!r}"
        )
    arm = goal.get("arm")
    if arm not in (None, *ARMS):
        raise _error("legacy_codec_error", f"unsupported arm: {arm!r}")
    prefix = f"{arm}_arm_" if arm is not None else ""
    if family in {"A2O_state", "A2O_event"}:
        verb = str(goal["verb"])
        if any(goal.get(slot) is not None for slot in ("target", "subject2", "target2")):
            raise _error(
                "action_arity_mismatch",
                f"A2O verb {verb!r} accepts only a subject",
            )
        if verb == ORIENTATION_ADJUSTMENT_VERB:
            params = normalize_orientation_adjustment_params(
                goal.get("params") or {}
            )
            base = (
                f"{prefix}{verb}({subject},"
                f"{params['orientation_axis']},"
                f"{params['rotation_delta_deg']},"
                f"{params['orientation_reference']})"
            )
        else:
            base = (
                f"{prefix}pick({subject})"
                if verb == "pick"
                else f"{prefix}{verb}({subject})"
            )
    else:
        relation = canonical_relation_value(goal)
        target = goal.get("target")
        target2 = goal.get("target2")
        if target is None:
            raise _error(
                "action_arity_mismatch",
                f"relation {relation!r} requires target",
            )
        target_text = _safe_token(target, field="target")
        if relation in TWO_TARGET_RELATIONS:
            if target2 is None:
                raise _error(
                    "action_arity_mismatch",
                    f"relation {relation!r} requires target2",
                )
            target2_text = _safe_token(target2, field="target2")
            arguments = f"{target_text},{target2_text}"
        else:
            if target2 is not None:
                raise _error(
                    "action_arity_mismatch",
                    f"relation {relation!r} does not accept target2",
                )
            arguments = target_text
        if goal.get("subject2") is not None:
            raise _error(
                "action_arity_mismatch",
                "single-step O2O actions do not accept subject2",
            )
        base = (
            f"{prefix}pick({subject})+{goal['verb']}({arguments})"
        )
    return base + _encode_descriptors(goal.get("descriptors") or {})


def _decode_descriptor(kind: str, value: str) -> dict[str, Any]:
    if kind == "attribute":
        subtype, separator, label = value.partition(":")
        if not separator or subtype not in ATTRIBUTE_SUBTYPES or not label:
            raise _error(
                "legacy_codec_error", f"invalid attribute descriptor: {value!r}"
            )
        _safe_token(label, field="attribute.value")
        return {"subtype": subtype, "value": label}
    if kind == "spatial":
        predicate_text, separator, reference = value.partition(":")
        if not separator or not reference:
            raise _error(
                "legacy_codec_error", f"invalid spatial descriptor: {value!r}"
            )
        predicates = _canonical_predicates(predicate_text.split("+"))
        _safe_token(reference, field="spatial.reference")
        return {"predicates": predicates, "reference": reference}
    if kind == "ordinal":
        parts = value.split(":")
        if len(parts) not in (2, 3) or parts[0] not in ORDINAL_BASES:
            raise _error(
                "legacy_codec_error", f"invalid ordinal descriptor: {value!r}"
            )
        result: dict[str, Any] = {"basis": parts[0], "order": parts[1]}
        _safe_token(parts[1], field="ordinal.order")
        if len(parts) == 3:
            result["reference"] = _safe_token(
                parts[2], field="ordinal.reference"
            )
        return result
    if kind == "part":
        return {"region": _safe_token(value, field="part.region")}
    raise _error("legacy_codec_error", f"unsupported descriptor kind: {kind!r}")


def _decode_descriptors(text: str) -> dict[str, Any]:
    if not text:
        raise _error("legacy_codec_error", "empty descriptor suffix")
    result: dict[str, dict[str, Any]] = {}
    seen: set[tuple[str, str]] = set()
    for token in text.split(";"):
        lhs, separator, value = token.partition("=")
        slot, dot, kind = lhs.partition(".")
        if (
            not separator
            or not dot
            or slot not in GOAL_SLOTS
            or kind not in DESCRIPTOR_KINDS
            or not value
        ):
            raise _error(
                "legacy_codec_error", f"invalid descriptor token: {token!r}"
            )
        key = (slot, kind)
        if key in seen:
            raise _error(
                "legacy_codec_error",
                f"duplicate descriptor token: {slot}.{kind}",
            )
        seen.add(key)
        result.setdefault(slot, {})[kind] = _decode_descriptor(kind, value)
    return result


def _base_goal(
    *,
    verb: str,
    subject: str,
    arm: str | None,
    target: str | None = None,
    target2: str | None = None,
    descriptors: Mapping[str, Any] | None = None,
    params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    relation = RELATION_BY_VERB.get(verb)
    state = None if verb in {"pick", *A2O_EVENT_VERBS} else A2O_STATE_BY_VERB.get(verb)
    goal = {
        "verb": verb,
        "action_family": (
            "O2O_relation"
            if relation is not None
            else "A2O_event"
            if verb in A2O_EVENT_VERBS
            else "A2O_state"
        ),
        "relation": relation,
        "state": state,
        "subject": subject,
        "arm": arm,
        "target": target,
        "subject2": None,
        "target2": target2,
        "descriptors": canonical_data(descriptors or {}),
        "params": canonical_data(params or {}),
    }
    canonical_action_family(goal)
    return goal


def decode_legacy_variant(variant: str) -> dict[str, Any]:
    """Strictly decode a supported compatibility string to ConcreteGoal."""

    if not isinstance(variant, str) or not variant.strip():
        raise _error("legacy_codec_error", "variant must be a non-empty string")
    text = variant.strip()
    if text.count("|desc=") > 1 or ("|" in text and "|desc=" not in text):
        raise _error("legacy_codec_error", "invalid legacy variant suffix")
    if "|desc=" in text:
        base, descriptor_text = text.split("|desc=", 1)
        descriptors = _decode_descriptors(descriptor_text)
    else:
        base = text
        descriptors = {}

    state_match = re.fullmatch(
        r"(?:(left|right)_arm_)?(open|close|switch_on|switch_off|click|press)"
        r"\((obj_[A-Za-z0-9_]+)\)",
        base,
    )
    if state_match:
        arm, verb, subject = state_match.groups()
        return _base_goal(
            verb=verb,
            subject=subject,
            arm=arm,
            descriptors=descriptors,
        )

    orientation_match = re.fullmatch(
        r"(?:(left|right)_arm_)?adjust_orientation"
        r"\((obj_[A-Za-z0-9_]+),(support_normal),(-?[0-9]+),"
        r"(initial_orientation)\)",
        base,
    )
    if orientation_match:
        arm, subject, axis, delta, reference = orientation_match.groups()
        params = normalize_orientation_adjustment_params(
            {
                "orientation_axis": axis,
                "rotation_delta_deg": int(delta),
                "orientation_reference": reference,
            }
        )
        return _base_goal(
            verb=ORIENTATION_ADJUSTMENT_VERB,
            subject=subject,
            arm=arm,
            descriptors=descriptors,
            params=params,
        )

    pick_match = re.fullmatch(
        r"(?:(left|right)_arm_)?pick\((obj_[A-Za-z0-9_]+)\)(?:\+(.+))?",
        base,
    )
    if not pick_match:
        raise _error("legacy_codec_error", f"unsupported variant: {base!r}")
    arm, subject, suffix = pick_match.groups()
    if suffix is None:
        return _base_goal(
            verb="pick",
            subject=subject,
            arm=arm,
            descriptors=descriptors,
        )
    place_match = re.fullmatch(
        r"([A-Za-z][A-Za-z0-9_]*)\(([^()]*)\)", suffix
    )
    if not place_match:
        raise _error("legacy_codec_error", f"invalid action suffix: {suffix!r}")
    verb, argument_text = place_match.groups()
    relation = RELATION_BY_VERB.get(verb)
    if relation is None:
        raise _error("unsupported_action", f"unsupported legacy action: {verb!r}")
    arguments = [item.strip() for item in argument_text.split(",")]
    expected_count = 2 if relation in TWO_TARGET_RELATIONS else 1
    if (
        len(arguments) != expected_count
        or any(_ENTITY_KEY.fullmatch(item) is None for item in arguments)
    ):
        raise _error(
            "action_arity_mismatch",
            f"relation {relation!r} requires {expected_count} concrete target(s)",
        )
    return _base_goal(
        verb=verb,
        subject=subject,
        arm=arm,
        target=arguments[0],
        target2=arguments[1] if expected_count == 2 else None,
        descriptors=descriptors,
    )


def _materialize_descriptors(
    variant_schema: Mapping[str, Any],
    candidate: Mapping[str, Any],
    entity_by_role: Mapping[str, str],
    *,
    family_id: str,
    assignment_id: str,
    schema_id: str,
) -> dict[str, Any]:
    role_bindings = variant_schema.get("role_bindings") or {}
    resolved = candidate.get("resolved_values") or {}
    result: dict[str, dict[str, Any]] = {}
    for slot, constructors in (variant_schema.get("descriptors") or {}).items():
        if slot not in GOAL_SLOTS or slot not in role_bindings:
            raise _error(
                "descriptor_slot_mismatch",
                f"descriptor slot {slot!r} is not a bound goal slot",
                family_id=family_id,
                assignment_id=assignment_id,
                schema_id=schema_id,
                path=f"descriptors.{slot}",
            )
        if not isinstance(constructors, Sequence):
            raise _error(
                "invalid_descriptor_constructor",
                f"descriptor constructors for {slot!r} are not a list",
                family_id=family_id,
                assignment_id=assignment_id,
                schema_id=schema_id,
                path=f"descriptors.{slot}",
            )
        bundle: dict[str, Any] = {}
        for index, constructor in enumerate(constructors):
            if not isinstance(constructor, Mapping):
                raise _error(
                    "invalid_descriptor_constructor",
                    "descriptor constructor is not a mapping",
                    family_id=family_id,
                    assignment_id=assignment_id,
                    schema_id=schema_id,
                    path=f"descriptors.{slot}.{index}",
                )
            kind = str(constructor.get("kind", ""))
            if kind not in DESCRIPTOR_KINDS:
                raise _error(
                    "unsupported_descriptor",
                    f"unsupported descriptor kind: {kind!r}",
                    family_id=family_id,
                    assignment_id=assignment_id,
                    schema_id=schema_id,
                )
            if kind in bundle:
                raise _error(
                    "descriptor_kind_conflict",
                    f"duplicate {kind!r} descriptor for slot {slot!r}",
                    family_id=family_id,
                    assignment_id=assignment_id,
                    schema_id=schema_id,
                )
            source_role = str(constructor.get("source_role", ""))
            if source_role != role_bindings[slot]:
                raise _error(
                    "descriptor_source_mismatch",
                    f"{slot}.{kind} source {source_role!r} does not match "
                    f"bound role {role_bindings[slot]!r}",
                    family_id=family_id,
                    assignment_id=assignment_id,
                    schema_id=schema_id,
                )
            if source_role not in entity_by_role:
                raise _error(
                    "unknown_role",
                    f"descriptor source role is not assigned: {source_role!r}",
                    family_id=family_id,
                    assignment_id=assignment_id,
                    schema_id=schema_id,
                )
            if kind == "attribute":
                subtype = str(constructor.get("subtype", ""))
                value = (
                    (resolved.get(source_role) or {})
                    .get("attributes", {})
                    .get(subtype)
                )
                if value is None or str(value) == "":
                    raise _error(
                        "missing_resolved_attribute",
                        f"missing reviewed value for {source_role}.{subtype}",
                        family_id=family_id,
                        assignment_id=assignment_id,
                        schema_id=schema_id,
                    )
                descriptor = {"subtype": subtype, "value": str(value)}
            elif kind == "spatial":
                reference_role = str(constructor.get("reference_role", ""))
                reference = entity_by_role.get(reference_role)
                if reference is None:
                    raise _error(
                        "unknown_role",
                        f"spatial reference role is not assigned: {reference_role!r}",
                        family_id=family_id,
                        assignment_id=assignment_id,
                        schema_id=schema_id,
                    )
                descriptor = {
                    "predicates": _canonical_predicates(
                        constructor.get("predicates", [])
                    ),
                    "reference": reference,
                }
            elif kind == "ordinal":
                descriptor = {
                    "basis": str(constructor.get("basis", "")),
                    "order": str(constructor.get("order", "")),
                }
                reference_role = constructor.get("reference_role")
                if reference_role is not None:
                    reference = entity_by_role.get(str(reference_role))
                    if reference is None:
                        raise _error(
                            "unknown_role",
                            f"ordinal reference role is not assigned: "
                            f"{reference_role!r}",
                            family_id=family_id,
                            assignment_id=assignment_id,
                            schema_id=schema_id,
                        )
                    descriptor["reference"] = reference
            else:
                region = (constructor.get("params") or {}).get("region")
                if not region:
                    raise _error(
                        "invalid_part_descriptor",
                        "part descriptor requires params.region",
                        family_id=family_id,
                        assignment_id=assignment_id,
                        schema_id=schema_id,
                    )
                descriptor = {"region": str(region)}
            _encode_descriptor(kind, descriptor)
            bundle[kind] = descriptor
        if bundle:
            result[str(slot)] = bundle
    return result


def _intended_axes(family: Mapping[str, Any]) -> list[dict[str, Any]]:
    axes: list[dict[str, Any]] = []
    for item in family.get("axis_plan", []):
        row = {
            key: deepcopy(item[key])
            for key in ("axis", "subtype", "role")
            if key in item
        }
        axes.append(row)
    return axes


def materialize_goal_variant(
    family: Mapping[str, Any],
    variant_schema: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    validate_schema: bool = True,
) -> dict[str, Any]:
    """Materialize one abstract VariantSchema for one complete assignment."""

    family_id = str(family.get("id", ""))
    assignment_id = str(candidate.get("id", ""))
    schema_id = str(variant_schema.get("id", ""))
    if candidate.get("abstract_family_id") != family_id:
        raise _error(
            "family_assignment_mismatch",
            f"candidate family {candidate.get('abstract_family_id')!r} "
            f"does not match {family_id!r}",
            family_id=family_id,
            assignment_id=assignment_id,
            schema_id=schema_id,
        )
    entity_by_role = {
        str(role): str(entity)
        for role, entity in (
            (candidate.get("materialization_params") or {})
            .get("entity_key_by_role", {})
            .items()
        )
    }
    assigned_roles = set((candidate.get("roles") or {}))
    if set(entity_by_role) != assigned_roles:
        raise _error(
            "entity_key_map_mismatch",
            "entity_key_by_role must cover every assigned role exactly",
            family_id=family_id,
            assignment_id=assignment_id,
            schema_id=schema_id,
        )
    if (
        len(set(entity_by_role.values())) != len(entity_by_role)
        or any(_ENTITY_KEY.fullmatch(value) is None for value in entity_by_role.values())
    ):
        raise _error(
            "invalid_entity_key",
            "entity_key_by_role values must be unique concrete entity keys",
            family_id=family_id,
            assignment_id=assignment_id,
            schema_id=schema_id,
        )

    role_refs = {
        str(slot): str(role)
        for slot, role in (variant_schema.get("role_bindings") or {}).items()
    }
    unknown_slots = sorted(set(role_refs) - set(GOAL_SLOTS))
    if unknown_slots or "subject" not in role_refs:
        raise _error(
            "action_arity_mismatch",
            f"invalid goal role slots: {unknown_slots or sorted(role_refs)}",
            family_id=family_id,
            assignment_id=assignment_id,
            schema_id=schema_id,
        )
    unknown_roles = sorted(set(role_refs.values()) - set(entity_by_role))
    if unknown_roles:
        raise _error(
            "unknown_role",
            f"variant references unassigned roles: {unknown_roles}",
            family_id=family_id,
            assignment_id=assignment_id,
            schema_id=schema_id,
        )
    entity_refs = {
        slot: entity_by_role[role] for slot, role in role_refs.items()
    }
    action = dict(variant_schema.get("action") or {})
    verb = str(action.get("verb", ""))
    relation = action.get("relation")
    state = action.get("state")
    arm = action.get("arm")
    if arm not in (None, *ARMS):
        raise _error(
            "unsupported_arm",
            f"unsupported arm: {arm!r}",
            family_id=family_id,
            assignment_id=assignment_id,
            schema_id=schema_id,
        )
    goal = {
        "verb": verb,
        "action_family": "",
        "relation": relation,
        "state": state,
        "subject": entity_refs["subject"],
        "arm": arm,
        "target": entity_refs.get("target"),
        "subject2": entity_refs.get("subject2"),
        "target2": entity_refs.get("target2"),
        "descriptors": _materialize_descriptors(
            variant_schema,
            candidate,
            entity_by_role,
            family_id=family_id,
            assignment_id=assignment_id,
            schema_id=schema_id,
        ),
        "params": canonical_data(variant_schema.get("params") or {}),
    }
    try:
        goal["action_family"] = canonical_action_family(goal)
        encoded = encode_legacy_variant(goal)
        decoded = decode_legacy_variant(encoded)
    except GoalMaterializationError as exc:
        exc.family_id = exc.family_id or family_id
        exc.assignment_id = exc.assignment_id or assignment_id
        exc.schema_id = exc.schema_id or schema_id
        raise
    if canonical_data(decoded) != canonical_data(goal):
        raise _error(
            "roundtrip_mismatch",
            "legacy encode/decode changed the structured ConcreteGoal",
            family_id=family_id,
            assignment_id=assignment_id,
            schema_id=schema_id,
        )
    result = {
        "id": schema_id,
        "schema_id": schema_id,
        "assignment_id": assignment_id,
        "role_refs": role_refs,
        "entity_refs": entity_refs,
        "intended_axes": _intended_axes(family),
        "goal": goal,
        "export": {"variant": encoded},
    }
    if validate_schema:
        try:
            _validate_goal_variant_contract(result)
        except GoalMaterializationError as exc:
            exc.family_id = family_id
            exc.assignment_id = assignment_id
            exc.schema_id = schema_id
            raise
    return canonical_data(result)


def materialize_assignment_goals(
    family: Mapping[str, Any],
    archetype: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    validate_schema: bool = True,
) -> list[dict[str, Any]]:
    """Materialize every variant schema for one complete assignment."""

    expected_archetypes = list(family.get("archetypes") or [])
    if list(candidate.get("archetype_ids") or []) != expected_archetypes:
        raise _error(
            "archetype_assignment_mismatch",
            "candidate archetype_ids do not match the abstract family",
            family_id=str(family.get("id", "")) or None,
            assignment_id=str(candidate.get("id", "")) or None,
        )
    if str(archetype.get("id", "")) not in expected_archetypes:
        raise _error(
            "unknown_archetype",
            f"archetype {archetype.get('id')!r} is not part of the family",
            family_id=str(family.get("id", "")) or None,
            assignment_id=str(candidate.get("id", "")) or None,
        )
    variants = [
        materialize_goal_variant(
            family,
            schema,
            candidate,
            validate_schema=validate_schema,
        )
        for schema in archetype.get("variant_schemas", [])
    ]
    if len({variant["id"] for variant in variants}) != len(variants):
        raise _error(
            "duplicate_goal_variant_id",
            "materialized goal variant ids are not unique",
            family_id=str(family.get("id", "")) or None,
            assignment_id=str(candidate.get("id", "")) or None,
        )
    return variants


def _source_hashes(
    *,
    family_doc: Mapping[str, Any],
    archetype_doc: Mapping[str, Any],
    domain: Mapping[str, Any],
    families_path: Path,
    archetypes_path: Path,
    domain_path: Path,
) -> dict[str, str]:
    return dict(
        sorted(
            {
                str(families_path): artifact_hash(family_doc),
                str(archetypes_path): artifact_hash(archetype_doc),
                str(domain_path): str(domain.get("domain_hash", "")),
            }.items()
        )
    )


def _assignment_artifact(
    family: Mapping[str, Any],
    archetype: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    source_hashes: Mapping[str, str],
    config_hash: str,
    validate_schema: bool = True,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "goal_variant_assignment",
        "goal_version": GOAL_VERSION,
        "abstract_family_id": family["id"],
        "assignment_id": candidate["id"],
        "archetype_ids": list(candidate["archetype_ids"]),
        "source_hashes": dict(source_hashes),
        "config_hash": config_hash,
        "goal_variants": materialize_assignment_goals(
            family,
            archetype,
            candidate,
            validate_schema=validate_schema,
        ),
    }
    result["artifact_hash"] = artifact_hash(result)
    return result


def _artifact_self_hash_valid(artifact: Mapping[str, Any]) -> bool:
    return artifact.get("artifact_hash") == artifact_hash(
        artifact, omit_keys=("artifact_hash",)
    )


def _artifact_reusable(
    artifact: Mapping[str, Any],
    *,
    family_id: str,
    assignment_id: str,
    archetype_ids: Sequence[str],
    source_hashes: Mapping[str, str],
    config_hash: str,
) -> bool:
    return (
        artifact.get("schema_version") == SCHEMA_VERSION
        and artifact.get("artifact_kind") == "goal_variant_assignment"
        and artifact.get("goal_version") == GOAL_VERSION
        and artifact.get("abstract_family_id") == family_id
        and artifact.get("assignment_id") == assignment_id
        and artifact.get("archetype_ids") == list(archetype_ids)
        and artifact.get("source_hashes") == dict(source_hashes)
        and artifact.get("config_hash") == config_hash
        and _artifact_self_hash_valid(artifact)
    )


def _new_stats() -> dict[str, Any]:
    return {
        "assignment_artifact_count": 0,
        "goal_variant_count": 0,
        "roundtrip_count": 0,
        "action_family_counts": Counter(),
        "relation_counts": Counter(),
        "state_counts": Counter(),
        "axis_counts": Counter(),
        "descriptor_kind_counts": Counter(),
        "families": {},
        "artifact_hash_rows": [],
    }


def _collect_artifact_stats(
    stats: dict[str, Any],
    artifact: Mapping[str, Any],
    relative_path: str,
) -> None:
    family_id = str(artifact["abstract_family_id"])
    variants = list(artifact.get("goal_variants") or [])
    stats["assignment_artifact_count"] += 1
    stats["goal_variant_count"] += len(variants)
    stats["roundtrip_count"] += len(variants)
    family_row = stats["families"].setdefault(
        family_id, {"assignment_artifact_count": 0, "goal_variant_count": 0}
    )
    family_row["assignment_artifact_count"] += 1
    family_row["goal_variant_count"] += len(variants)
    stats["artifact_hash_rows"].append(
        {"path": relative_path, "hash": artifact["artifact_hash"]}
    )
    for variant in variants:
        goal = variant["goal"]
        stats["action_family_counts"][str(goal["action_family"])] += 1
        if goal.get("relation") is not None:
            stats["relation_counts"][str(goal["relation"])] += 1
        state = canonical_state_value(goal)
        if state is not None:
            stats["state_counts"][state] += 1
        for axis in variant.get("intended_axes", []):
            key = str(axis["axis"])
            if axis.get("subtype"):
                key += f".{axis['subtype']}"
            stats["axis_counts"][key] += 1
        for bundle in (goal.get("descriptors") or {}).values():
            for kind in bundle:
                stats["descriptor_kind_counts"][kind] += 1


def _summary_from_stats(
    stats: Mapping[str, Any],
    *,
    expected_assignment_count: int,
    expected_goal_count: int,
    failures: Sequence[Mapping[str, Any]],
    skipped_empty: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "expected_assignment_artifact_count": expected_assignment_count,
        "assignment_artifact_count": stats["assignment_artifact_count"],
        "expected_goal_variant_count": expected_goal_count,
        "goal_variant_count": stats["goal_variant_count"],
        "roundtrip_count": stats["roundtrip_count"],
        "failure_count": len(failures),
        "skipped_empty_family_count": len(skipped_empty),
    }


def _build_reports(
    stats: Mapping[str, Any],
    *,
    config_hash: str,
    source_hashes: Mapping[str, str],
    expected_assignment_count: int,
    expected_goal_count: int,
    failures: Sequence[Mapping[str, Any]],
    skipped_empty: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = _summary_from_stats(
        stats,
        expected_assignment_count=expected_assignment_count,
        expected_goal_count=expected_goal_count,
        failures=failures,
        skipped_empty=skipped_empty,
    )
    acceptance_ready = (
        not failures
        and summary["assignment_artifact_count"] == expected_assignment_count
        and summary["goal_variant_count"] == expected_goal_count
        and summary["roundtrip_count"] == expected_goal_count
    )
    artifact_rows = sorted(
        stats["artifact_hash_rows"], key=lambda row: row["path"]
    )
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "goal_roundtrip_report",
        "goal_version": GOAL_VERSION,
        "config_hash": config_hash,
        "source_hashes": dict(source_hashes),
        "acceptance_ready": acceptance_ready,
        "summary": summary,
        "coverage": {
            "action_family_counts": dict(
                sorted(stats["action_family_counts"].items())
            ),
            "relation_counts": dict(sorted(stats["relation_counts"].items())),
            "state_counts": dict(sorted(stats["state_counts"].items())),
            "axis_counts": dict(sorted(stats["axis_counts"].items())),
            "descriptor_kind_counts": dict(
                sorted(stats["descriptor_kind_counts"].items())
            ),
        },
        "families": [
            {"abstract_family_id": family_id, **row}
            for family_id, row in sorted(stats["families"].items())
        ],
        "artifact_set_hash": artifact_hash(artifact_rows),
    }
    report["report_hash"] = artifact_hash(report)
    failure_counts = Counter(str(row["code"]) for row in failures)
    fail_report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "goal_materialization_fail_report",
        "goal_version": GOAL_VERSION,
        "config_hash": config_hash,
        "source_hashes": dict(source_hashes),
        "acceptance_ready": acceptance_ready,
        "summary": {
            "failure_count": len(failures),
            "failure_counts": dict(sorted(failure_counts.items())),
            "skipped_empty_family_count": len(skipped_empty),
        },
        "skipped_empty_domains": list(skipped_empty),
        "failures": list(failures),
    }
    fail_report["report_hash"] = artifact_hash(fail_report)
    return report, fail_report


def _library_inputs(
    *,
    families_path: Path,
    archetypes_path: Path,
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


_GOAL_WORKER_FAMILIES_DOC: Mapping[str, Any] = {}
_GOAL_WORKER_ARCHETYPES_DOC: Mapping[str, Any] = {}
_GOAL_WORKER_FAMILY_BY_ID: Mapping[str, Mapping[str, Any]] = {}
_GOAL_WORKER_ARCHETYPE_BY_ID: Mapping[str, Mapping[str, Any]] = {}
_GOAL_WORKER_ASSIGNMENT_DIR = Path()
_GOAL_WORKER_OUTPUT_DIR = Path()
_GOAL_WORKER_FAMILIES_PATH = Path()
_GOAL_WORKER_ARCHETYPES_PATH = Path()
_GOAL_WORKER_CONFIG_HASH = ""


def _init_goal_worker(
    families_doc: Mapping[str, Any],
    archetypes_doc: Mapping[str, Any],
    assignment_dir: str,
    output_dir: str,
    families_path: str,
    archetypes_path: str,
    config_hash: str,
) -> None:
    global _GOAL_WORKER_FAMILIES_DOC
    global _GOAL_WORKER_ARCHETYPES_DOC
    global _GOAL_WORKER_FAMILY_BY_ID
    global _GOAL_WORKER_ARCHETYPE_BY_ID
    global _GOAL_WORKER_ASSIGNMENT_DIR
    global _GOAL_WORKER_OUTPUT_DIR
    global _GOAL_WORKER_FAMILIES_PATH
    global _GOAL_WORKER_ARCHETYPES_PATH
    global _GOAL_WORKER_CONFIG_HASH

    _GOAL_WORKER_FAMILIES_DOC = families_doc
    _GOAL_WORKER_ARCHETYPES_DOC = archetypes_doc
    _GOAL_WORKER_FAMILY_BY_ID = {
        str(row["id"]): row for row in families_doc.get("families", [])
    }
    _GOAL_WORKER_ARCHETYPE_BY_ID = {
        str(row["id"]): row for row in archetypes_doc.get("archetypes", [])
    }
    _GOAL_WORKER_ASSIGNMENT_DIR = Path(assignment_dir)
    _GOAL_WORKER_OUTPUT_DIR = Path(output_dir)
    _GOAL_WORKER_FAMILIES_PATH = Path(families_path)
    _GOAL_WORKER_ARCHETYPES_PATH = Path(archetypes_path)
    _GOAL_WORKER_CONFIG_HASH = config_hash


def _validated_worker_count(workers: int) -> int:
    worker_count = int(workers)
    if worker_count < 1:
        raise ValueError("workers must be a positive integer")
    return worker_count


def _run_goal_family_workers(
    function: Callable[[str], dict[str, Any]],
    family_ids: Sequence[str],
    *,
    workers: int,
    families_doc: Mapping[str, Any],
    archetypes_doc: Mapping[str, Any],
    assignment_dir: Path,
    output_dir: Path,
    families_path: Path,
    archetypes_path: Path,
    config_hash: str,
) -> Iterable[dict[str, Any]]:
    """Run independent abstract families and preserve library order."""

    worker_count = min(
        _validated_worker_count(workers), max(1, len(family_ids))
    )
    initargs = (
        families_doc,
        archetypes_doc,
        str(assignment_dir),
        str(output_dir),
        str(families_path),
        str(archetypes_path),
        config_hash,
    )
    if worker_count == 1:
        _init_goal_worker(*initargs)
        for family_id in family_ids:
            yield function(family_id)
        return

    try:
        process_context = multiprocessing.get_context("fork")
    except ValueError:  # pragma: no cover - non-POSIX compatibility fallback
        process_context = multiprocessing.get_context()
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=process_context,
        initializer=_init_goal_worker,
        initargs=initargs,
    ) as executor:
        yield from executor.map(function, family_ids, chunksize=1)


def _merge_stats(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    for key in (
        "assignment_artifact_count",
        "goal_variant_count",
        "roundtrip_count",
    ):
        target[key] += int(source[key])
    for key in (
        "action_family_counts",
        "relation_counts",
        "state_counts",
        "axis_counts",
        "descriptor_kind_counts",
    ):
        target[key].update(source[key])
    for family_id, row in source["families"].items():
        target_row = target["families"].setdefault(
            family_id,
            {"assignment_artifact_count": 0, "goal_variant_count": 0},
        )
        target_row["assignment_artifact_count"] += int(
            row["assignment_artifact_count"]
        )
        target_row["goal_variant_count"] += int(row["goal_variant_count"])
    target["artifact_hash_rows"].extend(source["artifact_hash_rows"])


def _goal_family_inputs(
    family_id: str,
) -> tuple[Mapping[str, Any], Path, dict[str, Any] | None, Mapping[str, Any] | None]:
    family = _GOAL_WORKER_FAMILY_BY_ID[family_id]
    domain_path = _GOAL_WORKER_ASSIGNMENT_DIR / f"{family_id}.yml"
    domain = _try_load_yaml(domain_path)
    archetype_ids = list(family.get("archetypes") or [])
    archetype = (
        _GOAL_WORKER_ARCHETYPE_BY_ID.get(str(archetype_ids[0]))
        if archetype_ids
        else None
    )
    return family, domain_path, domain, archetype


def _freeze_goal_family(family_id: str) -> dict[str, Any]:
    family, domain_path, domain, archetype = _goal_family_inputs(family_id)
    stats = _new_stats()
    failures: list[dict[str, Any]] = []
    skipped_empty: list[dict[str, Any]] = []
    if domain is None:
        failures.append(
            _error(
                "missing_assignment_domain",
                f"expected YAML mapping: {domain_path}",
                family_id=family_id,
                path=str(domain_path),
            ).to_record()
        )
        return {
            "family_id": family_id,
            "stats": stats,
            "failures": failures,
            "skipped_empty": skipped_empty,
            "expected_assignment_count": 0,
            "expected_goal_count": 0,
        }
    if archetype is None:
        failures.append(
            _error(
                "unknown_archetype",
                f"missing archetype {list(family.get('archetypes') or [])!r}",
                family_id=family_id,
            ).to_record()
        )
        return {
            "family_id": family_id,
            "stats": stats,
            "failures": failures,
            "skipped_empty": skipped_empty,
            "expected_assignment_count": 0,
            "expected_goal_count": 0,
        }
    try:
        validate_assignment_domain(
            domain,
            family=family,
            archetype=archetype,
        )
    except Exception as exc:
        failures.append(
            _error(
                "missing_assignment_domain",
                str(exc),
                family_id=family_id,
                path=str(domain_path),
            ).to_record()
        )
        return {
            "family_id": family_id,
            "stats": stats,
            "failures": failures,
            "skipped_empty": skipped_empty,
            "expected_assignment_count": 0,
            "expected_goal_count": 0,
        }

    candidate_rows = list(domain.get("candidates") or [])
    expected_assignment_count = len(candidate_rows)
    expected_goal_count = len(candidate_rows) * len(
        archetype.get("variant_schemas", [])
    )
    if not candidate_rows:
        skipped_empty.append(
            {
                "abstract_family_id": family_id,
                "reason": "empty_assignment_domain",
                "feasible_candidate_count": int(
                    domain.get("feasible_candidate_count", 0)
                ),
                "published_candidate_count": 0,
            }
        )
    family_dir = _GOAL_WORKER_OUTPUT_DIR / family_id
    family_dir.mkdir(parents=True, exist_ok=True)
    expected_names: set[str] = set()
    sources = _source_hashes(
        family_doc=_GOAL_WORKER_FAMILIES_DOC,
        archetype_doc=_GOAL_WORKER_ARCHETYPES_DOC,
        domain=domain,
        families_path=_GOAL_WORKER_FAMILIES_PATH,
        archetypes_path=_GOAL_WORKER_ARCHETYPES_PATH,
        domain_path=domain_path,
    )
    for candidate in candidate_rows:
        assignment_id = str(candidate.get("id", ""))
        name = f"{assignment_id}.yml"
        expected_names.add(name)
        path = family_dir / name
        existing = _try_load_yaml(path) if path.is_file() else None
        if existing is not None and _artifact_reusable(
            existing,
            family_id=family_id,
            assignment_id=assignment_id,
            archetype_ids=candidate.get("archetype_ids") or [],
            source_hashes=sources,
            config_hash=_GOAL_WORKER_CONFIG_HASH,
        ):
            artifact = existing
        else:
            try:
                artifact = _assignment_artifact(
                    family,
                    archetype,
                    candidate,
                    source_hashes=sources,
                    config_hash=_GOAL_WORKER_CONFIG_HASH,
                )
                _write_yaml(path, artifact)
            except GoalMaterializationError as exc:
                failures.append(exc.to_record())
                path.unlink(missing_ok=True)
                continue
            except Exception as exc:
                failures.append(
                    _error(
                        "materialization_internal_error",
                        str(exc),
                        family_id=family_id,
                        assignment_id=assignment_id,
                    ).to_record()
                )
                path.unlink(missing_ok=True)
                continue
        _collect_artifact_stats(stats, artifact, f"{family_id}/{name}")
    for stale_path in sorted(family_dir.glob("*.yml")):
        if stale_path.name not in expected_names:
            stale_path.unlink()
    return {
        "family_id": family_id,
        "stats": stats,
        "failures": failures,
        "skipped_empty": skipped_empty,
        "expected_assignment_count": expected_assignment_count,
        "expected_goal_count": expected_goal_count,
    }


def _check_goal_family(family_id: str) -> dict[str, Any]:
    family, domain_path, domain, archetype = _goal_family_inputs(family_id)
    stale_paths: list[str] = []
    stats = _new_stats()
    skipped_empty: list[dict[str, Any]] = []
    if domain is None:
        stale_paths.append(str(domain_path))
        return {
            "family_id": family_id,
            "stats": stats,
            "stale_paths": stale_paths,
            "skipped_empty": skipped_empty,
            "expected_assignment_count": 0,
            "expected_goal_count": 0,
        }
    if archetype is None:
        stale_paths.append(
            f"missing archetype:{list(family.get('archetypes') or [])!r}"
        )
        return {
            "family_id": family_id,
            "stats": stats,
            "stale_paths": stale_paths,
            "skipped_empty": skipped_empty,
            "expected_assignment_count": 0,
            "expected_goal_count": 0,
        }
    try:
        validate_assignment_domain(
            domain,
            family=family,
            archetype=archetype,
        )
    except ValueError:
        stale_paths.append(str(domain_path))
        return {
            "family_id": family_id,
            "stats": stats,
            "stale_paths": stale_paths,
            "skipped_empty": skipped_empty,
            "expected_assignment_count": 0,
            "expected_goal_count": 0,
        }

    candidates = list(domain.get("candidates") or [])
    expected_assignment_count = len(candidates)
    expected_goal_count = len(candidates) * len(
        archetype.get("variant_schemas", [])
    )
    if not candidates:
        skipped_empty.append(
            {
                "abstract_family_id": family_id,
                "reason": "empty_assignment_domain",
                "feasible_candidate_count": int(
                    domain.get("feasible_candidate_count", 0)
                ),
                "published_candidate_count": 0,
            }
        )
    sources = _source_hashes(
        family_doc=_GOAL_WORKER_FAMILIES_DOC,
        archetype_doc=_GOAL_WORKER_ARCHETYPES_DOC,
        domain=domain,
        families_path=_GOAL_WORKER_FAMILIES_PATH,
        archetypes_path=_GOAL_WORKER_ARCHETYPES_PATH,
        domain_path=domain_path,
    )
    family_dir = _GOAL_WORKER_OUTPUT_DIR / family_id
    expected_names = {f"{candidate['id']}.yml" for candidate in candidates}
    actual_names = (
        {path.name for path in family_dir.glob("*.yml")}
        if family_dir.is_dir()
        else set()
    )
    stale_paths.extend(
        str(family_dir / name) for name in sorted(expected_names - actual_names)
    )
    stale_paths.extend(
        str(family_dir / name) for name in sorted(actual_names - expected_names)
    )
    candidate_by_name = {
        f"{candidate['id']}.yml": candidate for candidate in candidates
    }
    for name in sorted(expected_names & actual_names):
        path = family_dir / name
        artifact = _try_load_yaml(path)
        candidate = candidate_by_name[name]
        if artifact is None or not _artifact_reusable(
            artifact,
            family_id=family_id,
            assignment_id=str(candidate["id"]),
            archetype_ids=candidate.get("archetype_ids") or [],
            source_hashes=sources,
            config_hash=_GOAL_WORKER_CONFIG_HASH,
        ):
            stale_paths.append(str(path))
            continue
        _collect_artifact_stats(stats, artifact, f"{family_id}/{name}")
    return {
        "family_id": family_id,
        "stats": stats,
        "stale_paths": stale_paths,
        "skipped_empty": skipped_empty,
        "expected_assignment_count": expected_assignment_count,
        "expected_goal_count": expected_goal_count,
    }


def _validate_goal_family(family_id: str) -> dict[str, Any]:
    family, domain_path, domain, archetype = _goal_family_inputs(family_id)
    errors: list[str] = []
    assignment_count = 0
    goal_count = 0
    if domain is None or archetype is None:
        return {
            "family_id": family_id,
            "errors": errors,
            "assignment_artifact_count": assignment_count,
            "goal_variant_count": goal_count,
        }
    try:
        validate_assignment_domain(
            domain,
            family=family,
            archetype=archetype,
        )
    except ValueError as exc:
        errors.append(f"invalid assignment domain {domain_path}: {exc}")
        return {
            "family_id": family_id,
            "errors": errors,
            "assignment_artifact_count": assignment_count,
            "goal_variant_count": goal_count,
        }
    sources = _source_hashes(
        family_doc=_GOAL_WORKER_FAMILIES_DOC,
        archetype_doc=_GOAL_WORKER_ARCHETYPES_DOC,
        domain=domain,
        families_path=_GOAL_WORKER_FAMILIES_PATH,
        archetypes_path=_GOAL_WORKER_ARCHETYPES_PATH,
        domain_path=domain_path,
    )
    for candidate in domain.get("candidates") or []:
        assignment_count += 1
        path = _GOAL_WORKER_OUTPUT_DIR / family_id / f"{candidate['id']}.yml"
        actual = _try_load_yaml(path)
        if actual is None:
            continue
        try:
            expected = _assignment_artifact(
                family,
                archetype,
                candidate,
                source_hashes=sources,
                config_hash=_GOAL_WORKER_CONFIG_HASH,
            )
        except GoalMaterializationError as exc:
            errors.append(str(exc))
            continue
        goal_count += len(expected["goal_variants"])
        if path.read_text(encoding="utf-8") != canonical_yaml(actual):
            errors.append(f"artifact is not canonical YAML: {path}")
        if canonical_data(actual) != canonical_data(expected):
            errors.append(f"artifact does not match materialization: {path}")
    return {
        "family_id": family_id,
        "errors": errors,
        "assignment_artifact_count": assignment_count,
        "goal_variant_count": goal_count,
    }


def freeze_goal_artifacts(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    assignment_dir: Path = DEFAULT_ASSIGNMENT_DIR,
    families_path: Path = DEFAULT_FAMILIES_PATH,
    archetypes_path: Path = DEFAULT_ARCHETYPES_PATH,
    workers: int = 1,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Stream, reuse, and freeze all published M9 assignment artifacts."""

    (
        families_doc,
        archetypes_doc,
        _family_by_id,
        _archetype_by_id,
    ) = _library_inputs(
        families_path=families_path,
        archetypes_path=archetypes_path,
    )
    config_hash = artifact_hash(CONFIG)
    report_source_hashes = {
        str(families_path): artifact_hash(families_doc),
        str(archetypes_path): artifact_hash(archetypes_doc),
    }
    stats = _new_stats()
    failures: list[dict[str, Any]] = []
    skipped_empty: list[dict[str, Any]] = []
    expected_assignment_count = 0
    expected_goal_count = 0
    families = list(families_doc.get("families", []))
    family_ids = [str(family["id"]) for family in families]
    output_dir.mkdir(parents=True, exist_ok=True)
    results = _run_goal_family_workers(
        _freeze_goal_family,
        family_ids,
        workers=workers,
        families_doc=families_doc,
        archetypes_doc=archetypes_doc,
        assignment_dir=assignment_dir,
        output_dir=output_dir,
        families_path=families_path,
        archetypes_path=archetypes_path,
        config_hash=config_hash,
    )
    for index, result in enumerate(results, start=1):
        family_id = str(result["family_id"])
        _merge_stats(stats, result["stats"])
        failures.extend(result["failures"])
        skipped_empty.extend(result["skipped_empty"])
        expected_assignment_count += int(result["expected_assignment_count"])
        expected_goal_count += int(result["expected_goal_count"])
        if progress is not None:
            progress(index, len(family_ids), family_id)

    expected_family_ids = set(family_ids)
    for path in sorted(output_dir.iterdir()):
        if (
            path.is_dir()
            and path.name.startswith("asf_")
            and path.name not in expected_family_ids
        ):
            shutil.rmtree(path)
    report, fail_report = _build_reports(
        stats,
        config_hash=config_hash,
        source_hashes=report_source_hashes,
        expected_assignment_count=expected_assignment_count,
        expected_goal_count=expected_goal_count,
        failures=failures,
        skipped_empty=skipped_empty,
    )
    _write_yaml(output_dir / ROUNDTRIP_FILE, report)
    _write_yaml(output_dir / FAIL_FILE, fail_report)
    return {
        "acceptance_ready": report["acceptance_ready"],
        "summary": report["summary"],
        ROUNDTRIP_FILE: report,
        FAIL_FILE: fail_report,
    }


def check_goal_artifacts(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    assignment_dir: Path = DEFAULT_ASSIGNMENT_DIR,
    families_path: Path = DEFAULT_FAMILIES_PATH,
    archetypes_path: Path = DEFAULT_ARCHETYPES_PATH,
    workers: int = 1,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Check M9 source freshness, expected paths, and artifact self-hashes."""

    try:
        (
            families_doc,
            archetypes_doc,
            _family_by_id,
            _archetype_by_id,
        ) = _library_inputs(
            families_path=families_path,
            archetypes_path=archetypes_path,
        )
    except Exception as exc:
        return {
            "stale": True,
            "stale_paths": [str(exc)],
            "assignment_artifact_count": 0,
            "goal_variant_count": 0,
        }
    config_hash = artifact_hash(CONFIG)
    report_source_hashes = {
        str(families_path): artifact_hash(families_doc),
        str(archetypes_path): artifact_hash(archetypes_doc),
    }
    stale_paths: list[str] = []
    stats = _new_stats()
    skipped_empty: list[dict[str, Any]] = []
    expected_assignment_count = 0
    expected_goal_count = 0
    expected_family_ids: set[str] = set()
    families = list(families_doc.get("families", []))
    family_ids = [str(family["id"]) for family in families]
    results = _run_goal_family_workers(
        _check_goal_family,
        family_ids,
        workers=workers,
        families_doc=families_doc,
        archetypes_doc=archetypes_doc,
        assignment_dir=assignment_dir,
        output_dir=output_dir,
        families_path=families_path,
        archetypes_path=archetypes_path,
        config_hash=config_hash,
    )
    for index, result in enumerate(results, start=1):
        family_id = str(result["family_id"])
        expected_family_ids.add(family_id)
        _merge_stats(stats, result["stats"])
        stale_paths.extend(result["stale_paths"])
        skipped_empty.extend(result["skipped_empty"])
        expected_assignment_count += int(result["expected_assignment_count"])
        expected_goal_count += int(result["expected_goal_count"])
        if progress is not None:
            progress(index, len(family_ids), family_id)
    if output_dir.is_dir():
        for path in output_dir.iterdir():
            if (
                path.is_dir()
                and path.name.startswith("asf_")
                and path.name not in expected_family_ids
            ):
                stale_paths.append(str(path))
    expected_report, expected_fail = _build_reports(
        stats,
        config_hash=config_hash,
        source_hashes=report_source_hashes,
        expected_assignment_count=expected_assignment_count,
        expected_goal_count=expected_goal_count,
        failures=[],
        skipped_empty=skipped_empty,
    )
    for name, expected in (
        (ROUNDTRIP_FILE, expected_report),
        (FAIL_FILE, expected_fail),
    ):
        path = output_dir / name
        actual = _try_load_yaml(path)
        if actual is None or canonical_data(actual) != canonical_data(expected):
            stale_paths.append(str(path))
    return {
        "stale": bool(stale_paths),
        "stale_paths": sorted(set(stale_paths)),
        "assignment_artifact_count": stats["assignment_artifact_count"],
        "goal_variant_count": stats["goal_variant_count"],
    }


def validate_goal_artifacts(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    assignment_dir: Path = DEFAULT_ASSIGNMENT_DIR,
    families_path: Path = DEFAULT_FAMILIES_PATH,
    archetypes_path: Path = DEFAULT_ARCHETYPES_PATH,
    workers: int = 1,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Exhaustively rebuild and compare every frozen M9 assignment artifact."""

    check = check_goal_artifacts(
        output_dir=output_dir,
        assignment_dir=assignment_dir,
        families_path=families_path,
        archetypes_path=archetypes_path,
        workers=workers,
        progress=None,
    )
    errors = [f"stale artifact: {path}" for path in check["stale_paths"]]
    (
        families_doc,
        archetypes_doc,
        _family_by_id,
        _archetype_by_id,
    ) = _library_inputs(
        families_path=families_path,
        archetypes_path=archetypes_path,
    )
    config_hash = artifact_hash(CONFIG)
    assignment_count = 0
    goal_count = 0
    families = list(families_doc.get("families", []))
    family_ids = [str(family["id"]) for family in families]
    results = _run_goal_family_workers(
        _validate_goal_family,
        family_ids,
        workers=workers,
        families_doc=families_doc,
        archetypes_doc=archetypes_doc,
        assignment_dir=assignment_dir,
        output_dir=output_dir,
        families_path=families_path,
        archetypes_path=archetypes_path,
        config_hash=config_hash,
    )
    for index, result in enumerate(results, start=1):
        family_id = str(result["family_id"])
        assignment_count += int(result["assignment_artifact_count"])
        goal_count += int(result["goal_variant_count"])
        errors.extend(result["errors"])
        if progress is not None:
            progress(index, len(family_ids), family_id)
    return {
        "valid": not errors,
        "errors": errors,
        "assignment_artifact_count": assignment_count,
        "goal_variant_count": goal_count,
    }
