"""Milestone 13 instruction, checker, skill, and legacy task rendering.

The implementation is deliberately simulator-independent.  It joins one
frozen Milestone 12 SceneSpec to its exact Milestone 10 RealizedSceneFamily and
materializes variant-specific execution records without changing scene state,
assets, layout, or goal semantics.
"""

from __future__ import annotations

from collections import Counter, deque, defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
import hashlib
from itertools import combinations
import multiprocessing
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence

import yaml

from if_benchmark.articulation_capability import (
    ARTICULATION_CAPABILITY_VERSION,
    articulation_capability_errors,
    validated_articulation_arm_capability,
)
from if_benchmark.touch_capability import (
    TOUCH_CAPABILITY_VERSION,
    TOUCH_REQUESTED_ARMS,
    touch_capability_errors,
    validated_touch_arm_capability,
)
from if_benchmark.goals import (
    canonical_action_family,
    canonical_relation_value,
    canonical_state_value,
    encode_legacy_variant,
)
from if_benchmark.grasp_validation.planning_contract import (
    preferred_non_d4_arm,
    validated_scene_pick_planning_contract,
)
from if_benchmark.operation_anchors import OPERATION_ANCHOR_POLICY_VERSION
from if_benchmark.orientation_adjustment import (
    ORIENTATION_ADJUSTMENT_USE,
    ORIENTATION_ADJUSTMENT_VERB,
    ORIENTATION_ADJUSTMENT_VERSION,
    normalize_orientation_adjustment_params,
)
from if_benchmark.scene_integrity import build_scene_integrity_binding
from if_benchmark.schema import (
    artifact_hash,
    build_provenance,
    canonical_json,
    deterministic_id,
)
from if_benchmark.schema.cli import validate_document
from if_benchmark.schema.serialization import canonical_data
from if_benchmark.success_contract import RelationContract, SuccessContract


SCHEMA_VERSION = 1
TASK_RENDER_VERSION = 12
INSTRUCTION_RENDERER_VERSION = 6
TEMPLATE_LIBRARY_VERSION = 2
CHECKER_CONFIG_VERSION = 5
SKILL_CONFIG_VERSION = 10
LEGACY_EXPORT_VERSION = 1
SELECTION_POLICY = "sha256_scene_shared_style_v1"
STYLE_IDS = ("style_0", "style_1", "style_2", "style_3")
PUBLICATION_POLICY = "validated_task_subset_with_audited_exclusions_v1"
_STRUCTURAL_TASK_FAILURE_CODES = frozenset(
    {
        "instruction_reference_ambiguous",
        "instruction_shared_style_empty",
        "skill_route_unreachable",
        "skill_articulation_arm_capability_missing",
        "skill_touch_arm_capability_missing",
        "skill_semantic_arm_unreachable",
        "skill_subject_unreachable",
        "pick_planning_task_arm_missing",
        "skill_m12_execution_arm_mismatch",
    }
)

DEFAULT_SCENE_SPEC_DIR = Path("if_benchmark/data/scene_specs")
DEFAULT_REALIZATION_DIR = Path("if_benchmark/data/realized_families")
DEFAULT_OUTPUT_DIR = Path("if_benchmark/data/tasks")
DEFAULT_PROCESS_WORKERS = min(32, os.cpu_count() or 1)

INSTRUCTION_REPORT_FILE = "instruction_report.yml"
CHECKER_REPORT_FILE = "checker_alignment_report.yml"
LEGACY_REPORT_FILE = "legacy_export_report.yml"

REPORT_FILES = frozenset(
    {
        INSTRUCTION_REPORT_FILE,
        CHECKER_REPORT_FILE,
        LEGACY_REPORT_FILE,
    }
)

_DIRECTION_WORDS = {
    "left_of": "to the left of",
    "right_of": "to the right of",
    "in_front_of": "in front of",
    "behind": "behind",
    "in": "in",
    "on": "on",
}
_ORDINAL_WORDS = {
    "leftmost": "leftmost",
    "rightmost": "rightmost",
    "nearest": "nearest",
    "farthest": "farthest",
    "smallest": "smallest",
    "largest": "largest",
    "second": "second nearest",
    "2": "second from the left",
    "2_from_left": "second from the left",
    "second_from_left": "second from the left",
    "3": "third from the left",
    "3_from_left": "third from the left",
    "third_from_left": "third from the left",
}
_RELATION_PHRASES = {
    "in": "inside {target}",
    "on": "on {target}",
    "next_to": "next to {target}",
    "stack": "on top of {target}",
    "left_of": "to the left of {target}",
    "right_of": "to the right of {target}",
    "in_front_of": "in front of {target}",
    "behind": "behind {target}",
    "between": "between {target} and {target2}",
    "left_of_pair": "to the left of the pair formed by {target} and {target2}",
    "right_of_pair": "to the right of the pair formed by {target} and {target2}",
}
_TWO_TARGET_RELATIONS = frozenset(("between", "left_of_pair", "right_of_pair"))
_DIRECTIONAL_RELATIONS = frozenset(
    ("left_of", "right_of", "in_front_of", "behind")
)
_PAIR_REQUIRED_RELATIONS = frozenset(("in", "on", "stack"))
_FORBIDDEN_CHECKER_KEYS = frozenset(
    {
        "world",
        "world_pose",
        "world_position",
        "position",
        "pose",
        "xyz",
        "x",
        "y",
        "z",
        "cell",
        "zone_target",
    }
)

SUCCESS_CONTRACT = SuccessContract.load()
RELATION_CONTRACT = RelationContract.from_success_contract(SUCCESS_CONTRACT)

TASK_CONFIG: dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "task_render_version": TASK_RENDER_VERSION,
    "instruction_renderer_version": INSTRUCTION_RENDERER_VERSION,
    "template_library_version": TEMPLATE_LIBRARY_VERSION,
    "checker_config_version": CHECKER_CONFIG_VERSION,
    "skill_config_version": SKILL_CONFIG_VERSION,
    "operation_anchor_policy_version": OPERATION_ANCHOR_POLICY_VERSION,
    "legacy_export_version": LEGACY_EXPORT_VERSION,
    "selection_policy": SELECTION_POLICY,
    "style_ids": list(STYLE_IDS),
    "success_contract": SUCCESS_CONTRACT.identity,
    "relation_contract": RELATION_CONTRACT.identity,
    "skill_parameters": {
        "pre_grasp_m": 0.06,
        # The shared M4/M12 Pick chain already proves the guarded acquisition
        # lift at roughly this scale.  M13 must not silently strengthen that
        # witness into the former 70 mm execution-only requirement.
        "lift_m": 0.05,
        "retreat_m": 0.05,
        "retreat_frame": "arm",
        "retreat_axis": "negative_ee_local_x",
        "stack_lift_m": 0.13,
    },
}


@dataclass
class TaskRenderError(ValueError):
    """Stable failure attributed to one M13 rendering stage."""

    code: str
    message: str
    stage: str = "task"
    scene_spec_id: str | None = None
    realized_family_id: str | None = None
    goal_variant_id: str | None = None
    path: str | None = None

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"

    def to_record(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "stage": self.stage,
        }
        for key in (
            "scene_spec_id",
            "realized_family_id",
            "goal_variant_id",
            "path",
        ):
            value = getattr(self, key)
            if value is not None:
                row[key] = value
        return row


def _error(
    code: str,
    message: str,
    *,
    stage: str = "task",
    scene_spec_id: str | None = None,
    realized_family_id: str | None = None,
    goal_variant_id: str | None = None,
    path: str | None = None,
) -> TaskRenderError:
    return TaskRenderError(
        code,
        message,
        stage=stage,
        scene_spec_id=scene_spec_id,
        realized_family_id=realized_family_id,
        goal_variant_id=goal_variant_id,
        path=path,
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.load(stream, Loader=yaml.CSafeLoader)
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return value


def _try_load_yaml(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return _load_yaml(path)
    except (OSError, ValueError, yaml.YAMLError):
        return None


def _write_yaml(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = canonical_data(dict(value))
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
                normalized,
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


def _self_hash_valid(value: Mapping[str, Any]) -> bool:
    return value.get("artifact_hash") == artifact_hash(
        value, omit_keys=("artifact_hash",)
    )


def _report_hash_valid(value: Mapping[str, Any]) -> bool:
    return value.get("report_hash") == artifact_hash(
        value, omit_keys=("report_hash",)
    )


def _plural(noun: str) -> str:
    if noun.endswith(("s", "x", "z", "ch", "sh")):
        return noun + "es"
    if noun.endswith("y") and len(noun) > 1 and noun[-2].lower() not in "aeiou":
        return noun[:-1] + "ies"
    return noun + "s"


def _resolved_entities(realized: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    assignment = (realized.get("evidence") or {}).get("assignment") or {}
    resolved_by_role = assignment.get("resolved_values") or {}
    result: dict[str, dict[str, Any]] = {}
    for role, binding in (realized.get("role_bindings") or {}).items():
        entity_key = str(binding["entity_key"])
        resolved = canonical_data(resolved_by_role.get(role) or {})
        noun = str(
            resolved.get("noun")
            or binding.get("category")
            or str(binding.get("asset", "object")).split("_", 1)[-1]
        ).replace("_", " ")
        resolved["noun"] = noun
        resolved["role_ref"] = role
        resolved["category"] = str(binding.get("category") or noun)
        result[entity_key] = resolved
    return result


def _bare_description(
    entity_key: str,
    resolved: Mapping[str, Mapping[str, Any]],
    *,
    allow_duplicate: bool = False,
) -> str:
    if entity_key not in resolved:
        raise _error(
            "instruction_entity_missing",
            f"no reviewed realization facts for {entity_key}",
            stage="instruction",
        )
    noun = str(resolved[entity_key]["noun"])
    duplicates = [
        key
        for key, row in resolved.items()
        if key != entity_key and str(row.get("noun")) == noun
    ]
    if duplicates and not allow_duplicate:
        attributes = resolved[entity_key].get("attributes") or {}
        ordered_keys = [
            key
            for key in ("size", "color", "material", "shape", "type")
            if attributes.get(key) not in (None, "", noun)
        ]
        for count in range(1, len(ordered_keys) + 1):
            for keys in combinations(ordered_keys, count):
                if all(
                    any(
                        (resolved[other].get("attributes") or {}).get(key)
                        not in (None, "", attributes[key])
                        for key in keys
                    )
                    for other in duplicates
                ):
                    words = [
                        str(attributes[key]).replace("_", " ") for key in keys
                    ]
                    return f"the {' '.join(words)} {noun}"
        raise _error(
            "instruction_reference_ambiguous",
            f"reviewed noun and attributes do not uniquely identify {entity_key}",
            stage="instruction",
        )
    return f"the {noun}"


def _spatial_phrase(
    descriptor: Mapping[str, Any],
    resolved: Mapping[str, Mapping[str, Any]],
) -> str:
    predicates = [str(value) for value in descriptor.get("predicates", [])]
    if not predicates:
        raise _error(
            "instruction_descriptor_invalid",
            "spatial descriptor has no predicates",
            stage="instruction",
        )
    words = [_DIRECTION_WORDS.get(value, value.replace("_", " ")) for value in predicates]
    relation = " and ".join(words)
    reference = str(descriptor.get("reference", ""))
    return f"{relation} {_bare_description(reference, resolved)}"


def _ordinal_phrase(
    descriptor: Mapping[str, Any],
    resolved: Mapping[str, Mapping[str, Any]],
) -> str:
    order = str(descriptor.get("order", ""))
    basis = str(descriptor.get("basis", ""))
    phrase = _ORDINAL_WORDS.get(order, order.replace("_", " "))
    if basis == "robot":
        return f"{phrase} to the robot"
    if basis == "landmark":
        reference = str(descriptor.get("reference", ""))
        return f"{phrase} to {_bare_description(reference, resolved)}"
    return phrase


def _describe_slot(
    slot: str,
    goal: Mapping[str, Any],
    resolved: Mapping[str, Mapping[str, Any]],
    *,
    allow_duplicate: bool = False,
) -> str:
    entity_key = goal.get(slot)
    if entity_key is None:
        raise _error(
            "instruction_goal_arity",
            f"goal slot {slot} is null",
            stage="instruction",
        )
    entity_key = str(entity_key)
    bundle = (goal.get("descriptors") or {}).get(slot) or {}
    if not bundle:
        return _bare_description(
            entity_key, resolved, allow_duplicate=allow_duplicate
        )
    if entity_key not in resolved:
        raise _error(
            "instruction_entity_missing",
            f"descriptor subject {entity_key} has no realization facts",
            stage="instruction",
        )
    noun = str(resolved[entity_key]["noun"])
    if "attribute" in bundle:
        descriptor = bundle["attribute"]
        subtype = str(descriptor.get("subtype", ""))
        value = str(descriptor.get("value", "")).replace("_", " ")
        if not subtype or not value:
            raise _error(
                "instruction_descriptor_invalid",
                f"invalid attribute descriptor on {slot}",
                stage="instruction",
            )
        return f"the {value}" if subtype == "type" else f"the {value} {noun}"
    if "spatial" in bundle:
        return f"the {noun} {_spatial_phrase(bundle['spatial'], resolved)}"
    if "ordinal" in bundle:
        return f"the {_ordinal_phrase(bundle['ordinal'], resolved)} {noun}"
    if "part" in bundle:
        region = str(bundle["part"].get("region", "")).replace("_", " ")
        return f"the {region} of the {noun}"
    raise _error(
        "instruction_descriptor_unsupported",
        f"unsupported descriptor bundle on {slot}",
        stage="instruction",
    )


def _semantic_frame(goal: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "verb": str(goal["verb"]),
        "action_family": canonical_action_family(goal),
        "relation": canonical_relation_value(goal),
        "state": canonical_state_value(goal),
        "arm": goal.get("arm"),
        "entity_refs": {
            slot: goal.get(slot)
            for slot in ("subject", "target", "subject2", "target2")
            if goal.get(slot) is not None
        },
        "descriptors": canonical_data(goal.get("descriptors") or {}),
        "params": canonical_data(goal.get("params") or {}),
    }


def _goal_signature(goal: Mapping[str, Any]) -> str:
    return artifact_hash(_semantic_frame(goal))


def _arm_clause(arm: Any, style_id: str) -> tuple[str, str]:
    if arm is None:
        return "", ""
    text = str(arm)
    if style_id == "style_0":
        return f"Use the {text} arm to ", ""
    if style_id == "style_1":
        return f"With the {text} arm, ", ""
    if style_id == "style_2":
        return "", f" using the {text} arm"
    return f"Using your {text} arm, ", ""


def _sentence(text: str) -> str:
    """Normalize one rendered template into a publication sentence."""

    return text[:1].upper() + text[1:] + "."


def _render_style(
    goal: Mapping[str, Any],
    resolved: Mapping[str, Mapping[str, Any]],
    style_id: str,
) -> str:
    verb = str(goal["verb"])
    subject = _describe_slot("subject", goal, resolved)
    prefix, suffix = _arm_clause(goal.get("arm"), style_id)

    if verb == "pick":
        bodies = {
            "style_0": f"pick up {subject}",
            "style_1": f"lift {subject} from the table",
            "style_2": f"pick up {subject}",
            "style_3": f"grasp and lift {subject}",
        }
        return _sentence(prefix + bodies[style_id] + suffix)

    if verb in {"open", "close"}:
        bodies = {
            "style_0": f"{verb} {subject}",
            "style_1": f"move {subject} to the {verb} position",
            "style_2": f"set {subject} to the {verb} state",
            "style_3": f"change {subject} so that it is {verb}",
        }
        return _sentence(prefix + bodies[style_id] + suffix)

    if verb in {"switch_on", "switch_off"}:
        state = "on" if verb == "switch_on" else "off"
        bodies = {
            "style_0": f"turn {subject} {state}",
            "style_1": f"switch {subject} {state}",
            "style_2": f"set {subject} to {state}",
            "style_3": f"change {subject} to the {state} state",
        }
        return _sentence(prefix + bodies[style_id] + suffix)

    if verb in {"click", "press"}:
        bodies = (
            {
                "style_0": f"click {subject}",
                "style_1": f"tap {subject}",
                "style_2": f"click the control on {subject}",
                "style_3": f"activate {subject} with a click",
            }
            if verb == "click"
            else {
                "style_0": f"press {subject}",
                "style_1": f"push down on {subject}",
                "style_2": f"press the control on {subject}",
                "style_3": f"apply a press to {subject}",
            }
        )
        return _sentence(prefix + bodies[style_id] + suffix)

    if verb == ORIENTATION_ADJUSTMENT_VERB:
        params = normalize_orientation_adjustment_params(goal.get("params") or {})
        delta = int(params["rotation_delta_deg"])
        direction = "counterclockwise" if delta > 0 else "clockwise"
        magnitude = abs(delta)
        bodies = {
            "style_0": (
                f"rotate {subject} {direction} by {magnitude} degrees"
            ),
            "style_1": (
                f"turn {subject} {magnitude} degrees {direction}"
            ),
            "style_2": (
                f"adjust {subject} by rotating it {direction} "
                f"{magnitude} degrees"
            ),
            "style_3": (
                f"change the orientation of {subject} by {magnitude} degrees "
                f"{direction}"
            ),
        }
        return _sentence(prefix + bodies[style_id] + suffix)

    relation = canonical_relation_value(goal)
    if relation is None or relation not in _RELATION_PHRASES:
        raise _error(
            "instruction_action_unsupported",
            f"no instruction templates for {verb}",
            stage="instruction",
        )
    target2 = goal.get("target2")
    pair_collective = False
    if relation in _TWO_TARGET_RELATIONS and target2 is not None:
        first_key = str(goal["target"])
        second_key = str(target2)
        first_noun = str(resolved.get(first_key, {}).get("noun", ""))
        second_noun = str(resolved.get(second_key, {}).get("noun", ""))
        pair_collective = bool(first_noun and first_noun == second_noun)
    if pair_collective:
        noun = str(resolved[str(goal["target"])]["noun"])
        collective = f"the two {_plural(noun)}"
        if relation == "between":
            relation_text = f"between {collective}"
        elif relation == "left_of_pair":
            relation_text = f"to the left of {collective}"
        else:
            relation_text = f"to the right of {collective}"
    else:
        target = _describe_slot("target", goal, resolved)
        target2_text = (
            _describe_slot("target2", goal, resolved)
            if target2 is not None
            else ""
        )
        relation_text = _RELATION_PHRASES[relation].format(
            target=target, target2=target2_text
        )
    bodies = {
        "style_0": f"pick up {subject} and place it {relation_text}",
        "style_1": f"move {subject} {relation_text}",
        "style_2": f"pick up {subject}, then set it {relation_text}",
        "style_3": f"transfer {subject} so that it ends {relation_text}",
    }
    return _sentence(prefix + bodies[style_id] + suffix)


def build_instruction_candidates(
    goal: Mapping[str, Any],
    scene_spec: Mapping[str, Any],
    realized_family: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return the audited, deterministic instruction candidate pool."""

    del scene_spec  # Scene identity matters for selection, not target invention.
    resolved = _resolved_entities(realized_family)
    frame = _semantic_frame(goal)
    signature = _goal_signature(goal)
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for style_id in STYLE_IDS:
        text = _render_style(goal, resolved, style_id)
        normalized = " ".join(text.split())
        if normalized in seen:
            continue
        seen.add(normalized)
        candidates.append(
            {
                "template_id": f"tpl_{str(goal['verb'])}_{style_id}",
                "style_id": style_id,
                "text": normalized,
                "semantic_frame": deepcopy(frame),
                "goal_signature": signature,
            }
        )
    return canonical_data(candidates)


def select_instruction(
    candidates: Sequence[Mapping[str, Any]],
    *,
    instruction_seed: int,
    scene_spec_id: str,
    shared_style_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Select one style reproducibly for a publication SceneSpec."""

    if instruction_seed < 0:
        raise ValueError("instruction_seed must be non-negative")
    by_style = {str(row["style_id"]): canonical_data(row) for row in candidates}
    available = sorted(
        set(by_style)
        & set(shared_style_ids if shared_style_ids is not None else by_style)
    )
    if not available:
        raise _error(
            "instruction_shared_style_empty",
            f"no common instruction style for {scene_spec_id}",
            stage="instruction",
            scene_spec_id=scene_spec_id,
        )
    digest = hashlib.sha256(
        canonical_json(
            {
                "policy": SELECTION_POLICY,
                "instruction_seed": instruction_seed,
                "scene_spec_id": scene_spec_id,
                "style_ids": available,
            }
        ).encode("utf-8")
    ).hexdigest()
    selected_style = available[int(digest[:16], 16) % len(available)]
    return {
        "selected": by_style[selected_style],
        "selection": {
            "policy": SELECTION_POLICY,
            "instruction_seed": instruction_seed,
            "scene_spec_id": scene_spec_id,
            "shared_style_ids": available,
            "selected_style_id": selected_style,
            "selection_digest": digest,
            "candidate_set_hash": artifact_hash(canonical_data(list(candidates))),
        },
    }


def render_instruction(
    goal: Mapping[str, Any],
    scene_spec: Mapping[str, Any],
    realized_family: Mapping[str, Any],
    *,
    instruction_seed: int = 0,
    shared_style_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    candidates = build_instruction_candidates(goal, scene_spec, realized_family)
    selected = select_instruction(
        candidates,
        instruction_seed=instruction_seed,
        scene_spec_id=str(scene_spec["id"]),
        shared_style_ids=shared_style_ids,
    )
    return {
        "language": "en",
        "renderer_version": INSTRUCTION_RENDERER_VERSION,
        "candidates": candidates,
        **selected,
    }


def build_checker_config(
    goal: Mapping[str, Any],
    *,
    scene_entities: Mapping[str, Any],
    scene_spec: Mapping[str, Any] | None = None,
    execution_arm: str | None = None,
) -> dict[str, Any]:
    """Bind one ConcreteGoal to its predicate and scene-integrity guard.

    The complete joined SceneSpec entity set is mandatory because an omitted
    distractor would otherwise disappear from the protected partition.
    """

    family = canonical_action_family(goal)
    signature = _goal_signature(goal)
    subject = str(goal["subject"])
    arm = goal.get("arm")
    predicate: str
    arguments: dict[str, Any]
    if family == "A2O_state":
        verb = str(goal["verb"])
        if verb == "pick":
            requested_arm = arm if arm in {"left", "right"} else execution_arm
            if requested_arm not in {"left", "right"}:
                raise _error(
                    "checker_arm_missing",
                    "pick checker requires a semantic or M12 execution arm",
                    stage="checker",
                )
            predicate = "pick_success"
            arguments = {
                "subject": subject,
                "requested_arm": str(requested_arm),
            }
        elif verb in {"open", "close", "switch_on", "switch_off"}:
            predicate = "articulation_state_success"
            arguments = {
                "subject": subject,
                "state": canonical_state_value(goal),
            }
        elif verb == ORIENTATION_ADJUSTMENT_VERB:
            params = normalize_orientation_adjustment_params(
                goal.get("params") or {}
            )
            predicate = "orientation_success"
            arguments = {
                "subject": subject,
                "orientation_adjustment_version": (
                    ORIENTATION_ADJUSTMENT_VERSION
                ),
                **params,
            }
        else:
            raise _error(
                "checker_action_unsupported",
                f"unsupported A2O action {verb}",
                stage="checker",
            )
    elif family == "A2O_event":
        if scene_spec is None:
            raise _error(
                "checker_touch_capability_missing",
                "click/press checker requires the exact joined SceneSpec",
                stage="checker",
            )
        touch = _resolve_touch_execution(goal, scene_spec)
        predicate = "click_press_success"
        arguments = {
            "subject": subject,
            "requested_arm": touch["arm"],
            "use": touch["use"],
            "event": touch["event"],
            "contact_id": touch["contact_id"],
            "operation_anchor_id": touch["operation_anchor_id"],
        }
    else:
        relation = canonical_relation_value(goal)
        target = str(goal["target"])
        if relation not in {
            "in", "on", "stack", "next_to", *_DIRECTIONAL_RELATIONS,
            "between", "left_of_pair", "right_of_pair",
        }:
            raise _error(
                "checker_relation_unsupported",
                f"unsupported relation {relation}",
                stage="checker",
            )
        predicate = "placement_completed"
        arguments = {"subject": subject, "target": target, "relation": relation}
        if relation in _TWO_TARGET_RELATIONS:
            arguments["target2"] = str(goal["target2"])
    checker = {
        "version": CHECKER_CONFIG_VERSION,
        "family": family,
        "goal_signature": signature,
        "entity_refs": _semantic_frame(goal)["entity_refs"],
        "success_contract": SUCCESS_CONTRACT.identity,
        "predicate": {"name": predicate, "arguments": arguments},
        "scene_integrity": build_scene_integrity_binding(
            goal, scene_entities, SUCCESS_CONTRACT
        ),
    }
    if family == "O2O_relation":
        checker["relation_contract"] = RELATION_CONTRACT.identity
    return checker


def _reachable_arms(scene_spec: Mapping[str, Any], entity_key: str) -> tuple[str, ...]:
    row = (scene_spec.get("reach") or {}).get(entity_key) or {}
    return tuple(
        arm
        for arm in ("left", "right")
        if arm in set(row.get("reachable_arms") or [])
    )


def _terminal_reachable_arms(
    scene_spec: Mapping[str, Any], subject: str, relation: str
) -> tuple[str, ...]:
    """Return arms valid for every matching frozen terminal reservation.

    M13 is direct-only.  A receiver's initial object-center reach zone is not
    evidence that an arm can place the held subject at the relation terminal.
    Consume the M12 terminal-reach certificate for the exact subject/relation
    instead.  Multiple matching rows are intersected so an ambiguous
    reservation set cannot silently broaden the direct-route certificate.
    """

    arm_sets: list[set[str]] = []
    for row in (scene_spec.get("terminal_reach") or {}).values():
        if (
            not isinstance(row, Mapping)
            or str(row.get("subject") or "") != str(subject)
            or str(row.get("relation") or "") != str(relation)
            or row.get("proxy_passed") is not True
        ):
            continue
        arm_sets.append(
            {
                str(arm)
                for arm in row.get("reachable_arms") or []
                if str(arm) in {"left", "right"}
            }
        )
    if not arm_sets:
        return ()
    available = set.intersection(*arm_sets)
    return tuple(arm for arm in ("left", "right") if arm in available)


def _stable_arm(
    scene_spec: Mapping[str, Any], subject: str, arms: Sequence[str]
) -> str:
    scene_spec_id = str(scene_spec.get("id") or "")
    try:
        return preferred_non_d4_arm(
            str(scene_spec.get("realized_family_id") or ""),
            subject,
            int(scene_spec.get("layout_seed") or 0),
            arms,
        )
    except ValueError as exc:
        raise _error(
            "skill_subject_unreachable",
            f"{subject} is unreachable by both arms",
            stage="skill",
            scene_spec_id=scene_spec_id,
        ) from exc


def _articulation_use_from_goal(goal: Mapping[str, Any]) -> str | None:
    return {
        "open": "open",
        "close": "close",
        "on": "switch_on",
        "off": "switch_off",
        "switch_on": "switch_on",
        "switch_off": "switch_off",
    }.get(str(goal.get("verb") or goal.get("state") or ""))


def _articulation_capability_for_use(
    scene_spec: Mapping[str, Any], entity_key: str, use: str
) -> tuple[dict[str, Any], tuple[str, ...]]:
    projection = (
        ((scene_spec.get("reach") or {}).get(entity_key) or {}).get(
            "articulation_capability"
        )
        or {}
    )
    use_row = ((projection.get("uses") or {}).get(str(use)) or {})
    binding = {
        "capability_version": projection.get("capability_version"),
        "sampling_version": use_row.get("sampling_version"),
        "requested_arms": use_row.get("requested_arms"),
        "arm_results": use_row.get("arm_results"),
        "failures": use_row.get("failures"),
        "arm_capabilities": use_row.get("arm_capabilities"),
    }
    failures = articulation_capability_errors(binding, use=str(use))
    capable_arms = tuple(
        arm
        for arm in ("left", "right")
        if arm in set(use_row.get("capable_arms") or [])
    )
    if (
        projection.get("capability_version")
        != ARTICULATION_CAPABILITY_VERSION
        or not projection.get("evidence")
        or failures
        or capable_arms
        != tuple(
            arm
            for arm in ("left", "right")
            if arm
            in {
                str(row.get("arm") or "")
                for row in use_row.get("arm_capabilities") or []
                if isinstance(row, Mapping)
            }
        )
    ):
        raise _error(
            "skill_articulation_arm_capability_missing",
            f"{entity_key} lacks frozen exact per-arm capability for {use}",
            stage="skill",
            scene_spec_id=str(scene_spec.get("id") or ""),
        )
    return canonical_data(dict(projection)), capable_arms


def _touch_use_from_goal(goal: Mapping[str, Any]) -> str | None:
    return {
        "click": "click_subject",
        "press": "press_subject",
    }.get(str(goal.get("verb") or ""))


def _touch_capability_for_use(
    scene_spec: Mapping[str, Any], entity_key: str, use: str
) -> tuple[dict[str, Any], tuple[str, ...], tuple[str, ...]]:
    """Resolve one exact M11-projected touch capability and M12 path set.

    ``reachable_arms`` is intentionally insufficient here: it is the union of
    every operation use on an entity.  Click/Press instead consumes the exact
    ``touch_reach:<entity>:<use>:<arm>`` group whose Contact agrees with the
    selected M4 arm capability.
    """

    reach = ((scene_spec.get("reach") or {}).get(entity_key) or {})
    projection = reach.get("touch_capability") or {}
    use_row = ((projection.get("uses") or {}).get(str(use)) or {})
    binding = {
        "capability_version": projection.get("capability_version"),
        "sampling_version": use_row.get("sampling_version"),
        "placement_policy_version": use_row.get(
            "placement_policy_version"
        ),
        "placement_policy_hash": use_row.get("placement_policy_hash"),
        "requested_arms": use_row.get("requested_arms"),
        "canonical_placement_witness_hash": use_row.get(
            "canonical_placement_witness_hash"
        ),
        "arm_results": use_row.get("arm_results"),
        "arm_capabilities": use_row.get("arm_capabilities"),
        "failures": use_row.get("failures"),
    }
    arm_rows = list(use_row.get("arm_capabilities") or [])
    arm_capabilities: dict[str, dict[str, Any]] = {}
    contact_ids: set[int] = set()
    valid_rows = True
    for row in arm_rows:
        if not isinstance(row, Mapping):
            valid_rows = False
            continue
        arm = str(row.get("arm") or "")
        try:
            contact_id = int(row.get("contact_id"))
        except (TypeError, ValueError, OverflowError):
            valid_rows = False
            continue
        if (
            row.get("version") != TOUCH_CAPABILITY_VERSION
            or row.get("use") != str(use)
            or arm not in TOUCH_REQUESTED_ARMS
            or isinstance(row.get("contact_id"), bool)
            or contact_id < 0
            or float(row.get("contact_id")) != float(contact_id)
            or arm in arm_capabilities
        ):
            valid_rows = False
            continue
        arm_capabilities[arm] = canonical_data(dict(row))
        contact_ids.add(contact_id)

    declared_capable_arms = tuple(
        arm
        for arm in TOUCH_REQUESTED_ARMS
        if arm in set(use_row.get("capable_arms") or [])
    )
    observed_capable_arms = tuple(
        arm for arm in TOUCH_REQUESTED_ARMS if arm in arm_capabilities
    )
    arm_results = use_row.get("arm_results") or {}
    passing_result_arms = tuple(
        arm
        for arm in TOUCH_REQUESTED_ARMS
        if isinstance(arm_results.get(arm), Mapping)
        and arm_results[arm].get("pass") is True
    )
    witness_hash = str(
        use_row.get("canonical_placement_witness_hash") or ""
    )
    if (
        projection.get("capability_version") != TOUCH_CAPABILITY_VERSION
        or projection.get("evidence")
        != "m11_exact_m4_touch_capability_v1"
        or touch_capability_errors(binding, use=str(use))
        or not valid_rows
        or not arm_capabilities
        or declared_capable_arms != observed_capable_arms
        or passing_result_arms != observed_capable_arms
        or len(contact_ids) != 1
        or len(witness_hash) != 64
    ):
        raise _error(
            "skill_touch_arm_capability_missing",
            f"{entity_key} lacks frozen exact per-arm capability for {use}",
            stage="skill",
            scene_spec_id=str(scene_spec.get("id") or ""),
        )

    exact_path_arms: list[str] = []
    operation_groups = reach.get("touch_operation_groups") or []
    for arm in observed_capable_arms:
        capability = arm_capabilities[arm]
        expected_contact_id = int(capability["contact_id"])
        expected_group_id = f"touch_reach:{entity_key}:{use}:{arm}"
        matches = [
            group
            for group in operation_groups
            if isinstance(group, Mapping)
            and str(group.get("id") or "") == expected_group_id
            and str(group.get("use") or "") == str(use)
        ]
        if len(matches) != 1 or matches[0].get("passed") is not True:
            continue
        options = matches[0].get("options") or []
        def option_matches(option: Any) -> bool:
            if (
                not isinstance(option, Mapping)
                or str(option.get("arm") or "") != arm
                or option.get("reachable") is not True
                or isinstance(option.get("contact_id"), bool)
            ):
                return False
            try:
                observed_contact_id = int(option.get("contact_id"))
            except (TypeError, ValueError, OverflowError):
                return False
            return observed_contact_id == expected_contact_id

        if any(option_matches(option) for option in options):
            exact_path_arms.append(arm)
    return (
        canonical_data(dict(projection)),
        observed_capable_arms,
        tuple(exact_path_arms),
    )


def _resolve_touch_execution(
    goal: Mapping[str, Any], scene_spec: Mapping[str, Any]
) -> dict[str, Any]:
    subject = str(goal["subject"])
    scene_spec_id = str(scene_spec.get("id") or "")
    use = _touch_use_from_goal(goal)
    if use is None:
        raise _error(
            "skill_touch_arm_capability_missing",
            f"{subject} has unsupported touch event {goal.get('verb')!r}",
            stage="skill",
            scene_spec_id=scene_spec_id,
        )
    projection, capable_arms, exact_path_arms = _touch_capability_for_use(
        scene_spec, subject, use
    )
    reachable = set(_reachable_arms(scene_spec, subject))
    eligible_arms = tuple(
        arm
        for arm in TOUCH_REQUESTED_ARMS
        if arm in set(capable_arms)
        and arm in set(exact_path_arms)
        and arm in reachable
    )
    semantic_arm = goal.get("arm")
    if semantic_arm is not None and str(semantic_arm) not in capable_arms:
        raise _error(
            "skill_touch_arm_capability_missing",
            f"{subject} use {use} is not validated for {semantic_arm}",
            stage="skill",
            scene_spec_id=scene_spec_id,
        )
    if semantic_arm is not None:
        execution_arm = str(semantic_arm)
        if execution_arm not in eligible_arms:
            raise _error(
                "skill_semantic_arm_unreachable",
                f"{execution_arm} has no exact {use} operation path to {subject}",
                stage="skill",
                scene_spec_id=scene_spec_id,
            )
    else:
        execution_arm = _stable_arm(scene_spec, subject, eligible_arms)

    use_row = projection["uses"][use]
    binding = {
        "capability_version": projection["capability_version"],
        "sampling_version": use_row["sampling_version"],
        "placement_policy_version": use_row["placement_policy_version"],
        "placement_policy_hash": use_row["placement_policy_hash"],
        "requested_arms": use_row["requested_arms"],
        "canonical_placement_witness_hash": use_row[
            "canonical_placement_witness_hash"
        ],
        "arm_results": use_row["arm_results"],
        "arm_capabilities": use_row["arm_capabilities"],
        "failures": use_row["failures"],
    }
    arm_capability = validated_touch_arm_capability(
        binding, use=use, arm=execution_arm
    )
    if arm_capability is None:
        raise _error(
            "skill_touch_arm_capability_missing",
            f"{subject} use {use} is not validated for {execution_arm}",
            stage="skill",
            scene_spec_id=scene_spec_id,
        )
    contact_id = int(arm_capability["contact_id"])
    operation_anchor_id = f"grasp_contact:{contact_id}"
    return {
        "use": use,
        "event": str(goal["verb"]),
        "arm": execution_arm,
        "contact_id": contact_id,
        "operation_anchor_id": operation_anchor_id,
        "touch_capability": {
            "capability_version": projection["capability_version"],
            "use": use,
            "arm_capability": arm_capability,
            "evidence": projection["evidence"],
        },
    }


def compose_skill_config(
    goal: Mapping[str, Any],
    scene_spec: Mapping[str, Any],
    *,
    pick_plannable_arms: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compose a symbolic actor program using only goal and scene reach facts."""

    family = canonical_action_family(goal)
    subject = str(goal["subject"])
    scene_spec_id = str(scene_spec["id"])
    subject_arms = _reachable_arms(scene_spec, subject)
    if pick_plannable_arms is not None:
        planned = set(str(arm) for arm in pick_plannable_arms)
        subject_arms = tuple(arm for arm in subject_arms if arm in planned)
    semantic_arm = goal.get("arm")
    articulation_projection: dict[str, Any] | None = None
    articulation_use_value: str | None = None
    touch_execution: dict[str, Any] | None = None
    if family == "A2O_state" and str(goal.get("verb") or "") not in {
        "pick",
        ORIENTATION_ADJUSTMENT_VERB,
    }:
        articulation_use_value = _articulation_use_from_goal(goal)
        if articulation_use_value is None:
            raise _error(
                "skill_articulation_arm_capability_missing",
                f"{subject} has unsupported articulated use",
                stage="skill",
                scene_spec_id=scene_spec_id,
            )
        articulation_projection, capable_arms = (
            _articulation_capability_for_use(
                scene_spec, subject, articulation_use_value
            )
        )
        if semantic_arm is not None and str(semantic_arm) not in capable_arms:
            raise _error(
                "skill_articulation_arm_capability_missing",
                f"{subject} use {articulation_use_value} is not validated "
                f"for {semantic_arm}",
                stage="skill",
                scene_spec_id=scene_spec_id,
            )
        subject_arms = tuple(
            arm for arm in subject_arms if arm in set(capable_arms)
        )
        if not subject_arms:
            raise _error(
                "skill_articulation_arm_capability_missing",
                f"{subject} has no geometrically reachable M4-capable arm "
                f"for {articulation_use_value}",
                stage="skill",
                scene_spec_id=scene_spec_id,
            )
    elif family == "A2O_event":
        touch_execution = _resolve_touch_execution(goal, scene_spec)
        subject_arms = (str(touch_execution["arm"]),)
    elif family == "O2O_relation":
        relation = str(canonical_relation_value(goal))
        terminal_arms = _terminal_reachable_arms(
            scene_spec, subject, relation
        )
        direct_arms = tuple(
            arm for arm in subject_arms if arm in set(terminal_arms)
        )
        if not direct_arms:
            raise _error(
                "skill_route_unreachable",
                f"no single direct arm can both acquire {subject} and reach "
                f"its {relation} terminal reservation",
                stage="skill",
                scene_spec_id=scene_spec_id,
            )
        subject_arms = direct_arms
    if touch_execution is not None:
        execution_arm = str(touch_execution["arm"])
    elif semantic_arm is not None:
        execution_arm = str(semantic_arm)
        if execution_arm not in subject_arms:
            raise _error(
                "skill_semantic_arm_unreachable",
                f"{execution_arm} cannot reach {subject}",
                stage="skill",
                scene_spec_id=scene_spec_id,
            )
    else:
        execution_arm = _stable_arm(scene_spec, subject, subject_arms)

    params = TASK_CONFIG["skill_parameters"]
    steps: list[dict[str, Any]]
    route = "direct"
    if family == "A2O_state":
        if str(goal["verb"]) == "pick":
            steps = [
                {
                    "op": "grasp",
                    "subject": subject,
                    "arm": execution_arm,
                    "pre_grasp_m": params["pre_grasp_m"],
                    "total_lift_delta_m": params["lift_m"],
                    "operation_anchor_kind": "grasp_contact",
                }
            ]
        elif str(goal["verb"]) == ORIENTATION_ADJUSTMENT_VERB:
            orientation_params = normalize_orientation_adjustment_params(
                goal.get("params") or {}
            )
            steps = [
                {
                    "op": "grasp",
                    "subject": subject,
                    "arm": execution_arm,
                    "pre_grasp_m": params["pre_grasp_m"],
                    "total_lift_delta_m": params["lift_m"],
                    "operation_anchor_kind": "grasp_contact",
                    "candidate_use": ORIENTATION_ADJUSTMENT_USE,
                },
                {
                    "op": "adjust_orientation",
                    "subject": subject,
                    "arm": execution_arm,
                    "orientation_adjustment_version": (
                        ORIENTATION_ADJUSTMENT_VERSION
                    ),
                    **orientation_params,
                },
                {"op": "release", "arm": execution_arm},
                {
                    "op": "retreat_relative",
                    "arm": execution_arm,
                    "delta_m": params["retreat_m"],
                    "frame": params["retreat_frame"],
                    "axis": params["retreat_axis"],
                },
            ]
        else:
            assert articulation_projection is not None
            assert articulation_use_value is not None
            capability_binding = {
                "capability_version": articulation_projection[
                    "capability_version"
                ],
                "sampling_version": articulation_projection["uses"][
                    articulation_use_value
                ].get("sampling_version"),
                "requested_arms": articulation_projection["uses"][
                    articulation_use_value
                ].get("requested_arms"),
                "arm_results": articulation_projection["uses"][
                    articulation_use_value
                ].get("arm_results"),
                "failures": articulation_projection["uses"][
                    articulation_use_value
                ].get("failures"),
                "arm_capabilities": (
                    articulation_projection["uses"][
                        articulation_use_value
                    ]["arm_capabilities"]
                ),
            }
            arm_capability = validated_articulation_arm_capability(
                capability_binding,
                use=articulation_use_value,
                arm=execution_arm,
            )
            if arm_capability is None:
                raise _error(
                    "skill_articulation_arm_capability_missing",
                    f"{subject} use {articulation_use_value} is not validated "
                    f"for {execution_arm}",
                    stage="skill",
                    scene_spec_id=scene_spec_id,
                )
            steps = [
                {
                    "op": "actuate_state",
                    "subject": subject,
                    "arm": execution_arm,
                    "state": canonical_state_value(goal),
                    "profile_source": "validated_asset_use",
                    "articulation_capability": {
                        "capability_version": articulation_projection[
                            "capability_version"
                        ],
                        "use": articulation_use_value,
                        "arm_capability": arm_capability,
                        "evidence": articulation_projection["evidence"],
                    },
                    "operation_anchor_kind": "functional_or_contact",
                }
            ]
    elif family == "A2O_event":
        assert touch_execution is not None
        steps = [
            {
                "op": "touch_event",
                "subject": subject,
                "arm": execution_arm,
                "use": touch_execution["use"],
                "event": touch_execution["event"],
                "contact_id": touch_execution["contact_id"],
                "operation_anchor_kind": "functional_or_contact",
                "operation_anchor_id": touch_execution[
                    "operation_anchor_id"
                ],
                "touch_capability": touch_execution[
                    "touch_capability"
                ],
            }
        ]
    else:
        relation = str(canonical_relation_value(goal))
        place_arm = execution_arm
        lift_m = params["stack_lift_m"] if relation == "stack" else params["lift_m"]
        steps = [
            {
                "op": "grasp",
                "subject": subject,
                "arm": execution_arm,
                "pre_grasp_m": params["pre_grasp_m"],
                "total_lift_delta_m": lift_m,
                "operation_anchor_kind": "grasp_contact",
            }
        ]
        carry: dict[str, Any] = {
            "op": "carry_to_relation",
            "subject": subject,
            "target": str(goal["target"]),
            "arm": place_arm,
            "relation": relation,
            "anchor": (
                "receiver_reservation"
                if relation in _PAIR_REQUIRED_RELATIONS
                else "relative_entity"
            ),
            "operation_anchor_kind": "receiver_reservation",
        }
        if goal.get("target2") is not None:
            carry["target2"] = str(goal["target2"])
            carry["anchor"] = "relative_anchor_pair"
        steps.append(carry)
        if relation == "stack":
            steps.append(
                {
                    "op": "settle_stack",
                    "subject": subject,
                    "target": str(goal["target"]),
                    "arm": place_arm,
                    "anchor": "geometry_top",
                }
            )
        steps.extend(
            [
                {"op": "release", "arm": place_arm},
                {
                    "op": "retreat_relative",
                    "arm": place_arm,
                    "delta_m": params["retreat_m"],
                    "frame": params["retreat_frame"],
                    "axis": params["retreat_axis"],
                },
            ]
        )
    return {
        "version": SKILL_CONFIG_VERSION,
        "operation_anchor_policy_version": OPERATION_ANCHOR_POLICY_VERSION,
        "goal_signature": _goal_signature(goal),
        "semantic_arm": semantic_arm,
        "execution_arm": execution_arm,
        "route": route,
        "steps": steps,
    }


def build_legacy_export(
    goal_variant: Mapping[str, Any],
    scene_spec: Mapping[str, Any],
    instruction: Mapping[str, Any],
    *,
    task_record_id: str,
) -> dict[str, Any]:
    goal = goal_variant["goal"]
    encoded = encode_legacy_variant(goal)
    expected = str((goal_variant.get("export") or {}).get("variant", ""))
    if encoded != expected:
        raise _error(
            "legacy_variant_mismatch",
            "GoalVariant.export.variant differs from the structured goal",
            stage="legacy",
            scene_spec_id=str(scene_spec["id"]),
            goal_variant_id=str(goal_variant["id"]),
        )
    return {
        "version": LEGACY_EXPORT_VERSION,
        "task_name": "if_scene",
        "task_variant": expected,
        "setup_payload": {
            "task_record_id": task_record_id,
            "scene_spec_id": str(scene_spec["id"]),
            "realized_family_id": str(scene_spec["realized_family_id"]),
            "layout_seed": int(scene_spec["layout_seed"]),
            "embodiment": str(scene_spec["actor"]["embodiment"]),
            "instruction": str(instruction["selected"]["text"]),
            "task_variant": expected,
        },
    }


def _source_join_failures(
    scene_artifact: Mapping[str, Any],
    realized_artifact: Mapping[str, Any],
) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []
    scene = scene_artifact.get("scene_spec") or {}
    realized = realized_artifact.get("realized_scene_family") or {}

    def fail(code: str, message: str) -> None:
        failures.append({"code": code, "message": message})

    if not _self_hash_valid(scene_artifact):
        fail("scene_self_hash_mismatch", "SceneSpec artifact self hash is invalid")
    if not _self_hash_valid(realized_artifact):
        fail("realization_self_hash_mismatch", "realized artifact self hash is invalid")
    if scene_artifact.get("preflight", {}).get("status") != "passed":
        fail("scene_preflight_failed", "SceneSpec source preflight did not pass")
    if realized_artifact.get("preflight", {}).get("status") != "passed":
        fail("realization_preflight_failed", "realization source preflight did not pass")
    if (
        scene_artifact.get("realized_family_id")
        != realized_artifact.get("realized_family_id")
        or scene.get("realized_family_id") != realized.get("id")
    ):
        fail("source_realized_family_mismatch", "realized family ids do not align")
    if scene_artifact.get("source_realization_hash") != realized_artifact.get(
        "artifact_hash"
    ):
        fail("source_realization_hash_mismatch", "SceneSpec references a stale realization")
    try:
        validated_scene_pick_planning_contract(
            scene_artifact,
            realized_artifact=realized_artifact,
        )
    except ValueError as exc:
        fail(
            "pick_planning_evidence_invalid",
            f"SceneSpec Pick planning evidence is invalid: {exc}",
        )
    scene_entities = scene.get("entities") or {}
    for role, binding in (realized.get("role_bindings") or {}).items():
        entity_key = str(binding["entity_key"])
        entity = scene_entities.get(entity_key)
        if not isinstance(entity, dict):
            fail("scene_entity_missing", f"SceneSpec lacks {entity_key}")
            continue
        expected = (
            str(role),
            str(binding["asset"]),
            str(binding["model_id"]),
        )
        actual = (
            str(entity.get("role_ref")),
            str(entity.get("asset")),
            str(entity.get("model_id")),
        )
        if actual != expected:
            fail("scene_entity_binding_mismatch", f"binding mismatch for {entity_key}")
    variant_ids = {str(row["id"]) for row in realized.get("goal_variants", [])}
    from if_benchmark.layout.manipulation import valid_domain
    if not valid_domain(realized, scene):
        fail("scene_variant_set_mismatch", "SceneSpec variant ids differ from realization")
    return failures


def _attributed_failure(
    error: TaskRenderError,
    *,
    scene_spec_id: str,
    realized_family_id: str,
    goal_variant_id: str,
) -> dict[str, Any]:
    row = error.to_record()
    row["scene_spec_id"] = error.scene_spec_id or scene_spec_id
    row["realized_family_id"] = (
        error.realized_family_id or realized_family_id
    )
    row["goal_variant_id"] = error.goal_variant_id or goal_variant_id
    return canonical_data(row)


def render_task_record(
    scene_artifact: Mapping[str, Any],
    realized_artifact: Mapping[str, Any],
    goal_variant: Mapping[str, Any],
    *,
    instruction_seed: int = 0,
    shared_style_ids: Sequence[str] | None = None,
    instruction_candidates: Sequence[Mapping[str, Any]] | None = None,
    _skip_preflight: bool = False,
) -> dict[str, Any]:
    """Render one complete task record from a strict M10/M12 join."""

    join_failures = _source_join_failures(scene_artifact, realized_artifact)
    if join_failures:
        first = join_failures[0]
        raise _error(first["code"], first["message"], stage="source")
    scene = scene_artifact["scene_spec"]
    realized = realized_artifact["realized_scene_family"]
    variant_id = str(goal_variant["id"])
    if variant_id not in set(scene["goal_variant_ids"]):
        raise _error(
            "goal_variant_not_in_scene",
            f"{variant_id} is not part of {scene['id']}",
            stage="source",
            scene_spec_id=str(scene["id"]),
            goal_variant_id=variant_id,
        )
    goal = canonical_data(goal_variant["goal"])
    if instruction_candidates is None:
        instruction = render_instruction(
            goal,
            scene,
            realized,
            instruction_seed=instruction_seed,
            shared_style_ids=shared_style_ids,
        )
    else:
        candidates = canonical_data(list(instruction_candidates))
        instruction = {
            "language": "en",
            "renderer_version": INSTRUCTION_RENDERER_VERSION,
            "candidates": candidates,
            **select_instruction(
                candidates,
                instruction_seed=instruction_seed,
                scene_spec_id=str(scene["id"]),
                shared_style_ids=shared_style_ids,
            ),
        }
    pick_contract = validated_scene_pick_planning_contract(
        scene_artifact,
        realized_artifact=realized_artifact,
    )
    family = canonical_action_family(goal)
    needs_pick = family == "O2O_relation" or str(goal.get("verb") or "") in {
        "pick",
        ORIENTATION_ADJUSTMENT_VERB,
    }
    selected_arm = (
        (pick_contract.get("selected_arm_by_goal_variant") or {}).get(
            variant_id
        )
        if needs_pick
        else None
    )
    if needs_pick and selected_arm not in {"left", "right"}:
        raise _error(
            "pick_planning_task_arm_missing",
            f"M12 did not freeze one direct execution arm for {variant_id}",
            stage="source",
            scene_spec_id=str(scene["id"]),
            realized_family_id=str(realized["id"]),
            goal_variant_id=variant_id,
        )
    checker = build_checker_config(
        goal,
        scene_entities=scene["entities"],
        scene_spec=scene,
        execution_arm=(str(selected_arm) if needs_pick else None),
    )
    skill = compose_skill_config(
        goal,
        scene,
        pick_plannable_arms=(
            [str(selected_arm)]
            if needs_pick
            else None
        ),
    )
    config_hash = artifact_hash(
        {**TASK_CONFIG, "instruction_seed": instruction_seed}
    )
    record_id = deterministic_id(
        "task",
        {
            "scene_spec_id": scene["id"],
            "goal_variant_id": variant_id,
            "source_scene_hash": scene_artifact["artifact_hash"],
            "source_realization_hash": realized_artifact["artifact_hash"],
            "config_hash": config_hash,
            "selected_style_id": instruction["selected"]["style_id"],
        },
    )
    legacy = build_legacy_export(
        goal_variant, scene, instruction, task_record_id=record_id
    )
    goal_hash = artifact_hash(goal)
    task_record = {
        "id": record_id,
        "scene_spec_id": str(scene["id"]),
        "realized_family_id": str(realized["id"]),
        "goal_variant_id": variant_id,
        "goal": goal,
        "goal_provenance": {
            "abstract_family_id": str(realized["abstract_family_id"]),
            "assignment_id": str(goal_variant["assignment_id"]),
            "schema_id": str(goal_variant["schema_id"]),
            "role_refs": canonical_data(goal_variant["role_refs"]),
            "entity_refs": canonical_data(goal_variant["entity_refs"]),
            "scene_entity_keys": sorted(scene["entities"]),
            "intended_axes": canonical_data(goal_variant["intended_axes"]),
            "goal_hash": goal_hash,
            "source_scene_hash": str(scene_artifact["artifact_hash"]),
            "source_realization_hash": str(realized_artifact["artifact_hash"]),
        },
        "instruction": instruction,
        "success_contract": SUCCESS_CONTRACT.identity,
        "checker_config": checker,
        "skill_config": skill,
        "legacy_export": legacy,
        "provenance": build_provenance(
            "if_benchmark.tasks",
            config_hash,
            [
                {
                    "kind": "realized_scene_family",
                    "record_id": str(realized["id"]),
                    "hash": str(realized_artifact["artifact_hash"]),
                },
                {
                    "kind": "scene_spec",
                    "record_id": str(scene["id"]),
                    "hash": str(scene_artifact["artifact_hash"]),
                },
                {
                    "kind": "goal_variant",
                    "record_id": variant_id,
                    "hash": artifact_hash(goal_variant),
                },
            ],
            metadata={
                "task_render_version": TASK_RENDER_VERSION,
                "instruction_seed": instruction_seed,
            },
        ),
    }
    if checker.get("relation_contract") is not None:
        task_record["relation_contract"] = checker["relation_contract"]
    if not _skip_preflight:
        preflight = preflight_task_record(
            task_record,
            scene_spec=scene,
            realized_family=realized,
            goal_variant=goal_variant,
        )
        if preflight["status"] != "passed":
            first = preflight["failures"][0]
            raise _error(
                first["code"],
                first["message"],
                stage=str(first.get("stage", "alignment")),
                scene_spec_id=str(scene["id"]),
                realized_family_id=str(realized["id"]),
                goal_variant_id=variant_id,
            )
    return canonical_data(task_record)


def _forbidden_checker_paths(
    value: Any, path: tuple[str, ...] = ()
) -> list[str]:
    failures: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            next_path = path + (key_text,)
            if key_text in _FORBIDDEN_CHECKER_KEYS:
                failures.append(".".join(next_path))
            failures.extend(_forbidden_checker_paths(item, next_path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            failures.extend(_forbidden_checker_paths(item, path + (str(index),)))
    return failures


def preflight_task_record(
    task_record: Mapping[str, Any],
    *,
    scene_spec: Mapping[str, Any] | None = None,
    realized_family: Mapping[str, Any] | None = None,
    goal_variant: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Check cross-component semantic alignment and leakage invariants."""

    failures: list[dict[str, str]] = []

    def fail(code: str, message: str, stage: str = "alignment") -> None:
        failures.append({"code": code, "message": message, "stage": stage})

    goal = task_record.get("goal") or {}
    try:
        signature = _goal_signature(goal)
    except Exception as exc:
        fail("goal_signature_invalid", str(exc))
        signature = ""
    provenance = task_record.get("goal_provenance") or {}
    if provenance.get("goal_hash") != artifact_hash(goal):
        fail("goal_hash_mismatch", "goal provenance hash differs from goal")
    instruction = task_record.get("instruction") or {}
    candidates = instruction.get("candidates") or []
    unique_text = {str(row.get("text")) for row in candidates}
    if not candidates or len(unique_text) != len(candidates):
        fail(
            "instruction_candidate_contract",
            "instruction pool is empty or contains duplicate text",
            "instruction",
        )
    selected = instruction.get("selected") or {}
    if selected not in candidates:
        fail(
            "instruction_selection_missing",
            "selected instruction is not in candidates",
            "instruction",
        )
    selection = instruction.get("selection") or {}
    if selected.get("style_id") != selection.get("selected_style_id"):
        fail(
            "instruction_style_mismatch",
            "selection style does not match selected instruction",
            "instruction",
        )
    for row in candidates:
        if row.get("goal_signature") != signature:
            fail(
                "instruction_goal_mismatch",
                "candidate goal signature differs from ConcreteGoal",
                "instruction",
            )
            break
    arm = goal.get("arm")
    if arm is not None and any(
        f"{arm} arm" not in str(row.get("text", "")).lower()
        for row in candidates
    ):
        fail(
            "instruction_arm_omitted",
            f"semantic arm {arm} is absent from an instruction candidate",
            "instruction",
        )
    checker = task_record.get("checker_config") or {}
    skill = task_record.get("skill_config") or {}
    for name, config in (("checker", checker), ("skill", skill)):
        if config.get("goal_signature") != signature:
            fail(
                f"{name}_goal_mismatch",
                f"{name} goal signature differs from ConcreteGoal",
                name,
            )
    if checker.get("version") != CHECKER_CONFIG_VERSION:
        fail(
            "checker_config_version_stale",
            f"checker version must be {CHECKER_CONFIG_VERSION}",
            "checker",
        )
    if skill.get("version") != SKILL_CONFIG_VERSION:
        fail(
            "skill_config_version_stale",
            f"skill version must be {SKILL_CONFIG_VERSION}",
            "skill",
        )
    if str(skill.get("route") or "") != "direct" or any(
        isinstance(step, Mapping) and str(step.get("op") or "") == "handover"
        for step in skill.get("steps") or []
    ):
        fail(
            "skill_handover_disabled",
            "M13 publishes direct-only Skill IR; handover is disabled",
            "skill",
        )
    predicate = checker.get("predicate")
    if not isinstance(predicate, Mapping) or checker.get("conditions") is not None:
        fail(
            "checker_v5_shape_invalid",
            "checker v5 must bind one global predicate and scene-integrity guard",
            "checker",
        )
    expected_success = SUCCESS_CONTRACT.identity
    if checker.get("success_contract") != expected_success:
        fail(
            "checker_success_contract_stale",
            "checker is not bound to the canonical success contract",
            "checker",
        )
    if task_record.get("success_contract") != expected_success:
        fail(
            "task_success_contract_stale",
            "TaskRecord is not bound to the canonical success contract",
            "checker",
        )
    try:
        goal_family = canonical_action_family(goal)
    except Exception:
        goal_family = ""
    needs_pick_planning = goal_family == "O2O_relation" or str(
        goal.get("verb") or ""
    ) in {"pick", ORIENTATION_ADJUSTMENT_VERB}
    if scene_spec is not None and needs_pick_planning:
        subject = str(goal.get("subject") or "")
        variant_id = str(task_record.get("goal_variant_id") or "")
        reach = ((scene_spec.get("reach") or {}).get(subject) or {})
        matching_requirements = [
            row
            for row in reach.get("pick_planning_requirements") or []
            if isinstance(row, Mapping)
            and str(row.get("goal_variant_id") or "") == variant_id
        ]
        if len(matching_requirements) != 1:
            fail(
                "pick_planning_task_arm_missing",
                "joined M12 SceneSpec must freeze exactly one planning arm "
                f"for GoalVariant {variant_id}",
                "skill",
            )
        else:
            requirement = matching_requirements[0]
            expected_arm = str(requirement.get("selected_arm") or "")
            step_arms = {
                str(step.get("arm"))
                for step in skill.get("steps") or []
                if isinstance(step, Mapping) and step.get("arm") is not None
            }
            if (
                expected_arm not in {"left", "right"}
                or requirement.get("required_arms") != [expected_arm]
                or requirement.get("requirement_mode") != "all"
                or requirement.get("route") != "direct"
                or skill.get("execution_arm") != expected_arm
                or step_arms != {expected_arm}
            ):
                fail(
                    "skill_m12_execution_arm_mismatch",
                    "M13 execution must use exactly the singleton direct arm "
                    f"frozen by M12 for GoalVariant {variant_id}",
                    "skill",
                )
            if str(goal.get("verb") or "") == "pick":
                checker_arguments = (
                    (checker.get("predicate") or {}).get("arguments") or {}
                )
                if checker_arguments.get("requested_arm") != expected_arm:
                    fail(
                        "checker_m12_execution_arm_mismatch",
                        "Pick checker must use the singleton execution arm "
                        f"frozen by M12 for GoalVariant {variant_id}",
                        "checker",
                    )
    if goal_family == "A2O_event":
        expected_use = _touch_use_from_goal(goal)
        expected_event = str(goal.get("verb") or "")
        steps = skill.get("steps") or []
        touch_step = steps[0] if len(steps) == 1 else {}
        if (
            not isinstance(touch_step, Mapping)
            or touch_step.get("op") != "touch_event"
        ):
            fail(
                "skill_touch_event_shape_invalid",
                "click/press must compile to exactly one touch_event step",
                "skill",
            )
            touch_step = {}
        step_arm = str(touch_step.get("arm") or "")
        step_contact_id = touch_step.get("contact_id")
        expected_anchor_id = (
            f"grasp_contact:{step_contact_id}"
            if isinstance(step_contact_id, int)
            and not isinstance(step_contact_id, bool)
            and step_contact_id >= 0
            else None
        )
        expected_arguments = {
            "subject": str(goal.get("subject") or ""),
            "requested_arm": step_arm,
            "use": expected_use,
            "event": expected_event,
            "contact_id": step_contact_id,
            "operation_anchor_id": expected_anchor_id,
        }
        touch_capability = touch_step.get("touch_capability") or {}
        arm_capability = (
            touch_capability.get("arm_capability")
            if isinstance(touch_capability, Mapping)
            else {}
        ) or {}
        if (
            expected_use is None
            or step_arm not in TOUCH_REQUESTED_ARMS
            or touch_step.get("subject") != goal.get("subject")
            or touch_step.get("use") != expected_use
            or touch_step.get("event") != expected_event
            or touch_step.get("operation_anchor_kind")
            != "functional_or_contact"
            or touch_step.get("operation_anchor_id") != expected_anchor_id
            or skill.get("semantic_arm") != goal.get("arm")
            or skill.get("execution_arm") != step_arm
            or not isinstance(touch_capability, Mapping)
            or set(touch_capability)
            != {"capability_version", "use", "arm_capability", "evidence"}
            or touch_capability.get("capability_version")
            != TOUCH_CAPABILITY_VERSION
            or touch_capability.get("use") != expected_use
            or touch_capability.get("evidence")
            != "m11_exact_m4_touch_capability_v1"
            or not isinstance(arm_capability, Mapping)
            or arm_capability.get("version") != TOUCH_CAPABILITY_VERSION
            or arm_capability.get("use") != expected_use
            or arm_capability.get("arm") != step_arm
            or arm_capability.get("contact_id") != step_contact_id
        ):
            fail(
                "skill_touch_event_binding_mismatch",
                "touch_event differs from semantic arm/use/event/Contact capability",
                "skill",
            )
        if (
            not isinstance(predicate, Mapping)
            or predicate.get("name") != "click_press_success"
            or predicate.get("arguments") != expected_arguments
        ):
            fail(
                "checker_touch_event_binding_mismatch",
                "click_press_success does not bind the exact touch_event",
                "checker",
            )
        if scene_spec is not None:
            try:
                expected_skill = compose_skill_config(goal, scene_spec)
                expected_checker = build_checker_config(
                    goal,
                    scene_entities=scene_spec.get("entities") or {},
                    scene_spec=scene_spec,
                )
            except TaskRenderError as exc:
                fail(exc.code, exc.message, exc.stage)
            else:
                if canonical_data(skill) != canonical_data(expected_skill):
                    fail(
                        "skill_touch_scene_binding_mismatch",
                        "touch skill differs from exact M12 capability/path projection",
                        "skill",
                    )
                if canonical_data(predicate) != canonical_data(
                    expected_checker["predicate"]
                ):
                    fail(
                        "checker_touch_scene_binding_mismatch",
                        "touch checker differs from exact M12 capability/path projection",
                        "checker",
                    )
    elif (
        goal_family == "A2O_state"
        and str(goal.get("verb") or "") == ORIENTATION_ADJUSTMENT_VERB
    ):
        try:
            expected_params = normalize_orientation_adjustment_params(
                goal.get("params") or {}
            )
        except ValueError as exc:
            fail(
                "orientation_params_invalid", str(exc), "goal"
            )
            expected_params = {}
        steps = skill.get("steps") or []
        step_ops = [
            str(row.get("op") or "")
            for row in steps
            if isinstance(row, Mapping)
        ]
        execution_arm = str(skill.get("execution_arm") or "")
        grasp_step = (
            steps[0]
            if len(steps) == 4 and isinstance(steps[0], Mapping)
            else {}
        )
        orientation_step = (
            steps[1]
            if len(steps) == 4 and isinstance(steps[1], Mapping)
            else {}
        )
        if (
            step_ops
            != [
                "grasp",
                "adjust_orientation",
                "release",
                "retreat_relative",
            ]
            or grasp_step.get("subject") != goal.get("subject")
            or grasp_step.get("arm") != execution_arm
            or grasp_step.get("candidate_use")
            != ORIENTATION_ADJUSTMENT_USE
            or orientation_step.get("subject") != goal.get("subject")
            or orientation_step.get("arm") != execution_arm
            or orientation_step.get("orientation_adjustment_version")
            != ORIENTATION_ADJUSTMENT_VERSION
            or any(
                orientation_step.get(key) != value
                for key, value in expected_params.items()
            )
        ):
            fail(
                "skill_orientation_binding_mismatch",
                "orientation skill differs from the parameterized ConcreteGoal",
                "skill",
            )
        expected_arguments = {
            "subject": str(goal.get("subject") or ""),
            "orientation_adjustment_version": (
                ORIENTATION_ADJUSTMENT_VERSION
            ),
            **expected_params,
        }
        if (
            not isinstance(predicate, Mapping)
            or predicate.get("name") != "orientation_success"
            or predicate.get("arguments") != expected_arguments
        ):
            fail(
                "checker_orientation_binding_mismatch",
                "orientation_success differs from the parameterized ConcreteGoal",
                "checker",
            )
    provenance_entity_keys = sorted(
        str(value)
        for value in (
            (task_record.get("goal_provenance") or {}).get(
                "scene_entity_keys"
            )
            or []
        )
    )
    scene_entity_keys = provenance_entity_keys
    expected_scene_entities: Mapping[str, Any] = {
        key: {} for key in provenance_entity_keys
    }
    if scene_spec is not None:
        actual_scene_entity_keys = sorted(
            str(value) for value in (scene_spec.get("entities") or {})
        )
        if provenance_entity_keys != actual_scene_entity_keys:
            fail(
                "checker_scene_integrity_entity_universe_mismatch",
                "goal provenance entity universe differs from the joined SceneSpec",
                "checker",
            )
        scene_entity_keys = actual_scene_entity_keys
        expected_scene_entities = scene_spec.get("entities") or {}
    if not scene_entity_keys:
        fail(
            "checker_scene_integrity_entity_universe_missing",
            "scene-integrity requires the complete SceneSpec entity universe",
            "checker",
        )
    else:
        expected_integrity = build_scene_integrity_binding(
            goal, expected_scene_entities, SUCCESS_CONTRACT
        )
        observed_integrity = checker.get("scene_integrity") or {}
        exact_integrity_match = observed_integrity == expected_integrity
        if scene_spec is None and isinstance(observed_integrity, Mapping):
            exact_integrity_match = all(
                observed_integrity.get(field) == expected_integrity.get(field)
                for field in (
                    "policy",
                    "exempt_entities",
                    "protected_entities",
                )
            ) and set(
                observed_integrity.get("protected_articulated_entities") or []
            ).issubset(set(expected_integrity["protected_entities"]))
        if not exact_integrity_match:
            fail(
                "checker_scene_integrity_binding_mismatch",
                "scene-integrity exemptions/protected entities differ from ConcreteGoal",
                "checker",
            )
    if canonical_action_family(goal) == "O2O_relation":
        expected_relation = RELATION_CONTRACT.identity
        if checker.get("relation_contract") != expected_relation:
            fail(
                "checker_relation_contract_stale",
                "relation checker is not bound to the canonical relation contract",
                "checker",
            )
        if task_record.get("relation_contract") != expected_relation:
            fail(
                "task_relation_contract_stale",
                "TaskRecord relation contract binding is stale",
                "checker",
            )
    leaked_paths = _forbidden_checker_paths(checker)
    if leaked_paths:
        fail(
            "checker_absolute_coordinate_leak",
            f"forbidden checker coordinate keys: {leaked_paths[:10]}",
            "checker",
        )
    legacy = task_record.get("legacy_export") or {}
    try:
        if legacy.get("task_variant") != encode_legacy_variant(goal):
            fail(
                "legacy_goal_mismatch",
                "legacy task_variant differs from ConcreteGoal",
                "legacy",
            )
    except Exception as exc:
        fail("legacy_goal_invalid", str(exc), "legacy")
    if (
        (legacy.get("setup_payload") or {}).get("instruction")
        != selected.get("text")
    ):
        fail(
            "legacy_instruction_mismatch",
            "legacy payload instruction differs from selected instruction",
            "legacy",
        )
    if scene_spec is not None:
        if task_record.get("scene_spec_id") != scene_spec.get("id"):
            fail("task_scene_mismatch", "task references another SceneSpec")
        entity_keys = set(scene_spec.get("entities") or {})
        for slot, entity in _semantic_frame(goal).get("entity_refs", {}).items():
            if entity not in entity_keys:
                fail(
                    "task_entity_missing",
                    f"goal {slot} references missing scene entity {entity}",
                )
    if realized_family is not None and task_record.get(
        "realized_family_id"
    ) != realized_family.get("id"):
        fail("task_realization_mismatch", "task references another realization")
    if goal_variant is not None:
        if task_record.get("goal_variant_id") != goal_variant.get("id"):
            fail("task_variant_mismatch", "task references another GoalVariant")
        if canonical_data(goal) != canonical_data(goal_variant.get("goal")):
            fail("task_goal_mismatch", "task goal differs from GoalVariant")
        if legacy.get("task_variant") != (
            (goal_variant.get("export") or {}).get("variant")
        ):
            fail(
                "legacy_export_mismatch",
                "legacy task_variant differs from GoalVariant export",
                "legacy",
            )
    schema_result = validate_document(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "task_record",
            "task_record": canonical_data(task_record),
        }
    )
    if not schema_result["valid"]:
        for row in schema_result["errors"][:10]:
            fail(
                "task_schema_invalid",
                f"{row['path']}: {row['message']}",
                "schema",
            )
    return {
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "goal_signature": signature,
        "instruction_candidate_count": len(candidates),
    }


def _artifact(
    task_record: Mapping[str, Any],
    preflight: Mapping[str, Any],
    *,
    scene_artifact: Mapping[str, Any],
    realized_artifact: Mapping[str, Any],
    instruction_seed: int,
) -> dict[str, Any]:
    value = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "rendered_task_record",
        "task_render_version": TASK_RENDER_VERSION,
        "scene_spec_id": str(task_record["scene_spec_id"]),
        "realized_family_id": str(task_record["realized_family_id"]),
        "goal_variant_id": str(task_record["goal_variant_id"]),
        "source_scene_hash": str(scene_artifact["artifact_hash"]),
        "source_realization_hash": str(realized_artifact["artifact_hash"]),
        "config_hash": artifact_hash(
            {**TASK_CONFIG, "instruction_seed": instruction_seed}
        ),
        "task_record": canonical_data(task_record),
        "preflight": canonical_data(preflight),
    }
    value["artifact_hash"] = artifact_hash(value)
    return value


def _reusable(
    artifact: Mapping[str, Any],
    *,
    scene_artifact: Mapping[str, Any],
    realized_artifact: Mapping[str, Any],
    goal_variant_id: str,
    instruction_seed: int,
) -> bool:
    task = artifact.get("task_record") or {}
    return (
        artifact.get("artifact_kind") == "rendered_task_record"
        and artifact.get("task_render_version") == TASK_RENDER_VERSION
        and artifact.get("source_scene_hash") == scene_artifact.get("artifact_hash")
        and artifact.get("source_realization_hash")
        == realized_artifact.get("artifact_hash")
        and artifact.get("goal_variant_id") == goal_variant_id
        and artifact.get("config_hash")
        == artifact_hash({**TASK_CONFIG, "instruction_seed": instruction_seed})
        and artifact.get("preflight", {}).get("status") == "passed"
        and task.get("goal_variant_id") == goal_variant_id
        and _self_hash_valid(artifact)
    )


def _report_row(
    artifact: Mapping[str, Any],
    *,
    scene_spec_id: str,
    goal_variant_id: str,
    shared_style_ids: Sequence[str],
    instruction_seed: int,
) -> dict[str, Any]:
    task = artifact["task_record"]
    instruction = task["instruction"]
    selected_style = str(instruction["selected"]["style_id"])
    repeated = select_instruction(
        instruction["candidates"],
        instruction_seed=instruction_seed,
        scene_spec_id=scene_spec_id,
        shared_style_ids=shared_style_ids,
    )
    alternate = select_instruction(
        instruction["candidates"],
        instruction_seed=instruction_seed + 1,
        scene_spec_id=scene_spec_id,
        shared_style_ids=shared_style_ids,
    )
    return {
        "task_record_id": task["id"],
        "scene_spec_id": scene_spec_id,
        "goal_variant_id": goal_variant_id,
        "artifact_hash": artifact["artifact_hash"],
        "style_id": selected_style,
        "template_id": instruction["selected"]["template_id"],
        "candidate_count": len(instruction["candidates"]),
        "checker_family": task["checker_config"]["family"],
        "skill_route": task["skill_config"]["route"],
        "shared_style_valid": selected_style in shared_style_ids,
        "same_seed_reproduced": (
            repeated["selected"]["style_id"] == selected_style
            and repeated["selected"]["text"] == instruction["selected"]["text"]
        ),
        "alternate_seed_style_changed": (
            alternate["selected"]["style_id"] != selected_style
        ),
    }


def _scene_paths(scene_spec_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in scene_spec_dir.glob("rsf_*/*.yml")
        if path.name not in REPORT_FILES
    )


def _layout_publication_ready(scene_spec_dir: Path) -> bool:
    report = _try_load_yaml(scene_spec_dir / "layout_report.yml") or {}
    return bool(report.get("publication_ready"))


def _common_styles(
    realized: Mapping[str, Any],
    scene: Mapping[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    pools: dict[str, list[dict[str, Any]]] = {}
    common = set(STYLE_IDS)
    variant_by_id = {
        str(row["id"]): row for row in realized.get("goal_variants", [])
    }
    for variant_id in scene.get("goal_variant_ids", []):
        variant = variant_by_id.get(str(variant_id))
        if variant is None:
            raise _error(
                "goal_variant_missing",
                f"realization lacks {variant_id}",
                stage="source",
                scene_spec_id=str(scene.get("id", "")),
                goal_variant_id=str(variant_id),
            )
        candidates = build_instruction_candidates(
            variant["goal"], scene, realized
        )
        pools[str(variant_id)] = candidates
        common &= {str(row["style_id"]) for row in candidates}
    if not common:
        raise _error(
            "instruction_shared_style_empty",
            f"SceneSpec {scene.get('id')} variants share no style",
            stage="instruction",
            scene_spec_id=str(scene.get("id", "")),
        )
    return pools, sorted(common)


def _reports(
    rows: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
    *,
    scene_count: int,
    expected_task_count: int,
    instruction_seed: int,
    upstream_publication_ready: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config_hash = artifact_hash({**TASK_CONFIG, "instruction_seed": instruction_seed})
    artifact_rows = sorted(
        (
            {
                "task_record_id": str(row["task_record_id"]),
                "artifact_hash": str(row["artifact_hash"]),
            }
            for row in rows
        ),
        key=lambda row: row["task_record_id"],
    )
    artifact_set_hash = artifact_hash(artifact_rows)
    implementation_failure_count = sum(
        str(row.get("code")) not in _STRUCTURAL_TASK_FAILURE_CODES
        for row in failures
    )
    structural_failure_count = len(failures) - implementation_failure_count
    implementation_complete = (
        len(rows) + len(failures) == expected_task_count
        and implementation_failure_count == 0
    )
    coverage_complete = implementation_complete and structural_failure_count == 0
    stage_ready = implementation_complete and bool(rows)
    publication_ready = stage_ready and upstream_publication_ready
    common = {
        "schema_version": SCHEMA_VERSION,
        "task_render_version": TASK_RENDER_VERSION,
        "config_hash": config_hash,
        "instruction_seed": instruction_seed,
        "artifact_set_hash": artifact_set_hash,
        "implementation_complete": implementation_complete,
        "publication_policy": PUBLICATION_POLICY,
        "publication_scope": "rendered_task_records",
        "coverage_complete": coverage_complete,
        "stage_acceptance_ready": stage_ready,
        "publication_ready": publication_ready,
        "upstream_publication_ready": upstream_publication_ready,
        "summary": {
            "scene_spec_count": scene_count,
            "expected_task_count": expected_task_count,
            "rendered_task_count": len(rows),
            "failure_count": len(failures),
            "structural_failure_count": structural_failure_count,
            "implementation_failure_count": implementation_failure_count,
            "accounted_task_count": len(rows) + len(failures),
        },
        "failure_code_counts": dict(
            sorted(Counter(str(row.get("code")) for row in failures).items())
        ),
    }
    style_counts = Counter(str(row["style_id"]) for row in rows)
    template_counts = Counter(str(row["template_id"]) for row in rows)
    candidate_counts = Counter(int(row["candidate_count"]) for row in rows)
    scene_styles: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        scene_styles[str(row["scene_spec_id"])].add(str(row["style_id"]))
    instruction_failures = [
        canonical_data(row)
        for row in failures
        if str(row.get("stage")) in {"instruction", "source", "task", "schema"}
    ]
    instruction = {
        **common,
        "artifact_kind": "instruction_report",
        "selection_policy": SELECTION_POLICY,
        "style_counts": dict(sorted(style_counts.items())),
        "template_counts": dict(sorted(template_counts.items())),
        "candidate_count_distribution": {
            str(key): value for key, value in sorted(candidate_counts.items())
        },
        "selected_instruction_coverage_count": len(rows),
        "selected_instruction_coverage": (
            len(rows) / expected_task_count if expected_task_count else 1.0
        ),
        "same_seed_reproduction_count": sum(
            bool(row.get("same_seed_reproduced")) for row in rows
        ),
        "alternate_seed_changed_scene_count": len(
            {
                str(row["scene_spec_id"])
                for row in rows
                if row.get("alternate_seed_style_changed")
            }
        ),
        "scene_style_consistency_checked_count": len(scene_styles),
        "scene_style_consistency_failure_count": sum(
            len(styles) > 1 for styles in scene_styles.values()
        ),
        "scene_shared_style_count": len(
            {
                str(row["scene_spec_id"])
                for row in rows
                if row.get("shared_style_valid")
            }
        ),
        "failures": instruction_failures,
    }
    checker_failures = [
        canonical_data(row)
        for row in failures
        if str(row.get("stage"))
        in {"checker", "skill", "alignment", "source", "schema"}
    ]
    checker = {
        **common,
        "artifact_kind": "checker_alignment_report",
        "checker_family_counts": dict(
            sorted(Counter(str(row["checker_family"]) for row in rows).items())
        ),
        "skill_route_counts": dict(
            sorted(Counter(str(row["skill_route"]) for row in rows).items())
        ),
        "absolute_coordinate_leak_count": 0,
        "cross_component_alignment_count": len(rows),
        "failures": checker_failures,
    }
    legacy_failures = [
        canonical_data(row)
        for row in failures
        if str(row.get("stage")) in {"legacy", "source", "schema"}
    ]
    legacy = {
        **common,
        "artifact_kind": "legacy_export_report",
        "task_variant_roundtrip_count": len(rows),
        "setup_payload_count": len(rows),
        "failures": legacy_failures,
    }
    for report in (instruction, checker, legacy):
        report["report_hash"] = artifact_hash(report)
    return instruction, checker, legacy


_TASK_WORKER_REALIZATION_DIR = Path()
_TASK_WORKER_OUTPUT_DIR = Path()
_TASK_WORKER_INSTRUCTION_SEED = 0


def _init_task_worker(
    realization_dir: str,
    output_dir: str,
    instruction_seed: int,
) -> None:
    global _TASK_WORKER_REALIZATION_DIR
    global _TASK_WORKER_OUTPUT_DIR
    global _TASK_WORKER_INSTRUCTION_SEED

    _TASK_WORKER_REALIZATION_DIR = Path(realization_dir)
    _TASK_WORKER_OUTPUT_DIR = Path(output_dir)
    _TASK_WORKER_INSTRUCTION_SEED = int(instruction_seed)


def _validated_worker_count(workers: int) -> int:
    worker_count = int(workers)
    if worker_count < 1:
        raise ValueError("workers must be a positive integer")
    return worker_count


def _run_task_scene_workers(
    function: Callable[[str], dict[str, Any]],
    scene_paths: Sequence[Path],
    *,
    realization_dir: Path,
    output_dir: Path,
    instruction_seed: int,
    workers: int,
) -> Iterable[dict[str, Any]]:
    """Run independent SceneSpecs while preserving canonical source order."""

    worker_count = min(
        _validated_worker_count(workers), max(1, len(scene_paths))
    )
    initargs = (
        str(realization_dir),
        str(output_dir),
        instruction_seed,
    )
    tasks = [str(path) for path in scene_paths]
    if worker_count == 1:
        _init_task_worker(*initargs)
        for task in tasks:
            yield function(task)
        return

    try:
        process_context = multiprocessing.get_context("fork")
    except ValueError:  # pragma: no cover - non-POSIX compatibility fallback
        process_context = multiprocessing.get_context()
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=process_context,
        initializer=_init_task_worker,
        initargs=initargs,
    ) as executor:
        # Avoid ProcessPoolExecutor.map's eager submission of every SceneSpec.
        # Full M13 builds contain tens of thousands of inputs, so an unbounded
        # submission phase can fill the wakeup pipe before results are drained.
        task_iterator = iter(tasks)
        pending = deque()
        for _ in range(worker_count * 2):
            try:
                task = next(task_iterator)
            except StopIteration:
                break
            pending.append(executor.submit(function, task))

        while pending:
            future = pending.popleft()
            yield future.result()
            try:
                task = next(task_iterator)
            except StopIteration:
                continue
            pending.append(executor.submit(function, task))


def _freeze_task_scene(scene_path_text: str) -> dict[str, Any]:
    scene_path = Path(scene_path_text)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    expected_paths: list[str] = []
    delete_paths: list[str] = []
    expected_task_count = 0
    progress_id = scene_path.stem
    scene_artifact = _try_load_yaml(scene_path)
    if scene_artifact is None:
        failures.append(
            _error(
                "scene_artifact_invalid",
                f"cannot load {scene_path}",
                stage="source",
                path=str(scene_path),
            ).to_record()
        )
        return {
            "progress_id": progress_id,
            "rows": rows,
            "failures": failures,
            "expected_paths": expected_paths,
            "delete_paths": delete_paths,
            "expected_task_count": expected_task_count,
        }

    realized_id = str(
        scene_artifact.get("realized_family_id", scene_path.parent.name)
    )
    realized_path = _TASK_WORKER_REALIZATION_DIR / f"{realized_id}.yml"
    realized_artifact = _try_load_yaml(realized_path)
    scene = scene_artifact.get("scene_spec") or {}
    scene_id = str(scene.get("id", scene_path.stem))
    progress_id = scene_id
    variant_ids = [str(value) for value in scene.get("goal_variant_ids", [])]
    expected_task_count = len(variant_ids)
    if realized_artifact is None:
        failures.extend(
            _error(
                "realization_artifact_missing",
                f"cannot load {realized_path}",
                stage="source",
                scene_spec_id=scene_id,
                realized_family_id=realized_id,
                goal_variant_id=variant_id,
                path=str(realized_path),
            ).to_record()
            for variant_id in variant_ids
        )
        return {
            "progress_id": progress_id,
            "rows": rows,
            "failures": failures,
            "expected_paths": expected_paths,
            "delete_paths": delete_paths,
            "expected_task_count": expected_task_count,
        }

    realized = realized_artifact["realized_scene_family"]
    join_failures = _source_join_failures(scene_artifact, realized_artifact)
    if join_failures:
        first = join_failures[0]
        failures.extend(
            _error(
                first["code"],
                first["message"],
                stage="source",
                scene_spec_id=scene_id,
                realized_family_id=realized_id,
                goal_variant_id=variant_id,
            ).to_record()
            for variant_id in variant_ids
        )
        return {
            "progress_id": progress_id,
            "rows": rows,
            "failures": failures,
            "expected_paths": expected_paths,
            "delete_paths": delete_paths,
            "expected_task_count": expected_task_count,
        }

    variants = {
        str(row["id"]): row for row in realized.get("goal_variants", [])
    }
    try:
        pools, shared_styles = _common_styles(realized, scene)
    except TaskRenderError as exc:
        failures.extend(
            _attributed_failure(
                exc,
                scene_spec_id=scene_id,
                realized_family_id=realized_id,
                goal_variant_id=variant_id,
            )
            for variant_id in variant_ids
        )
        return {
            "progress_id": progress_id,
            "rows": rows,
            "failures": failures,
            "expected_paths": expected_paths,
            "delete_paths": delete_paths,
            "expected_task_count": expected_task_count,
        }

    for variant_id in variant_ids:
        output_path = _TASK_WORKER_OUTPUT_DIR / scene_id / f"{variant_id}.yml"
        expected_paths.append(str(output_path))
        try:
            variant = variants[variant_id]
            existing = _try_load_yaml(output_path)
            if existing is not None and _reusable(
                existing,
                scene_artifact=scene_artifact,
                realized_artifact=realized_artifact,
                goal_variant_id=variant_id,
                instruction_seed=_TASK_WORKER_INSTRUCTION_SEED,
            ):
                artifact = existing
            else:
                task_record = render_task_record(
                    scene_artifact,
                    realized_artifact,
                    variant,
                    instruction_seed=_TASK_WORKER_INSTRUCTION_SEED,
                    shared_style_ids=shared_styles,
                    instruction_candidates=pools[variant_id],
                    _skip_preflight=True,
                )
                preflight = preflight_task_record(
                    task_record,
                    scene_spec=scene,
                    realized_family=realized,
                    goal_variant=variant,
                )
                artifact = _artifact(
                    task_record,
                    preflight,
                    scene_artifact=scene_artifact,
                    realized_artifact=realized_artifact,
                    instruction_seed=_TASK_WORKER_INSTRUCTION_SEED,
                )
                _write_yaml(output_path, artifact)
            rows.append(
                _report_row(
                    artifact,
                    scene_spec_id=scene_id,
                    goal_variant_id=variant_id,
                    shared_style_ids=shared_styles,
                    instruction_seed=_TASK_WORKER_INSTRUCTION_SEED,
                )
            )
        except TaskRenderError as exc:
            failures.append(
                _attributed_failure(
                    exc,
                    scene_spec_id=scene_id,
                    realized_family_id=realized_id,
                    goal_variant_id=variant_id,
                )
            )
            delete_paths.append(str(output_path))
        except Exception as exc:
            failures.append(
                _error(
                    "task_internal_error",
                    str(exc),
                    stage="task",
                    scene_spec_id=scene_id,
                    realized_family_id=realized_id,
                    goal_variant_id=variant_id,
                    path=str(output_path),
                ).to_record()
            )
            delete_paths.append(str(output_path))
    return {
        "progress_id": progress_id,
        "rows": rows,
        "failures": failures,
        "expected_paths": expected_paths,
        "delete_paths": delete_paths,
        "expected_task_count": expected_task_count,
    }


def _check_task_scene(scene_path_text: str) -> dict[str, Any]:
    scene_path = Path(scene_path_text)
    stale_paths: list[str] = []
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    all_task_paths: list[str] = []
    expected_task_count = 0
    progress_id = scene_path.stem
    progress_ready = False
    scene_artifact = _try_load_yaml(scene_path)
    if scene_artifact is None:
        stale_paths.append(str(scene_path))
        return {
            "progress_id": progress_id,
            "progress_ready": progress_ready,
            "stale_paths": stale_paths,
            "rows": rows,
            "failures": failures,
            "all_task_paths": all_task_paths,
            "expected_task_count": expected_task_count,
        }

    scene = scene_artifact.get("scene_spec") or {}
    scene_id = str(scene.get("id", scene_path.stem))
    progress_id = scene_id
    realized_id = str(
        scene_artifact.get("realized_family_id", scene_path.parent.name)
    )
    variant_ids = [str(value) for value in scene.get("goal_variant_ids", [])]
    expected_task_count = len(variant_ids)
    paths = {
        variant_id: _TASK_WORKER_OUTPUT_DIR / scene_id / f"{variant_id}.yml"
        for variant_id in variant_ids
    }
    all_task_paths.extend(str(path) for path in paths.values())
    realized_path = _TASK_WORKER_REALIZATION_DIR / f"{realized_id}.yml"
    realized_artifact = _try_load_yaml(realized_path)
    if realized_artifact is None:
        stale_paths.append(str(realized_path))
        return {
            "progress_id": progress_id,
            "progress_ready": progress_ready,
            "stale_paths": stale_paths,
            "rows": rows,
            "failures": failures,
            "all_task_paths": all_task_paths,
            "expected_task_count": expected_task_count,
        }

    realized = realized_artifact.get("realized_scene_family") or {}
    join_failures = _source_join_failures(scene_artifact, realized_artifact)
    if join_failures:
        stale_paths.append(str(scene_path))
        return {
            "progress_id": progress_id,
            "progress_ready": progress_ready,
            "stale_paths": stale_paths,
            "rows": rows,
            "failures": failures,
            "all_task_paths": all_task_paths,
            "expected_task_count": expected_task_count,
        }

    variants = {
        str(row["id"]): row for row in realized.get("goal_variants", [])
    }
    try:
        pools, shared_styles = _common_styles(realized, scene)
    except TaskRenderError as exc:
        failures.extend(
            _attributed_failure(
                exc,
                scene_spec_id=scene_id,
                realized_family_id=realized_id,
                goal_variant_id=variant_id,
            )
            for variant_id in variant_ids
        )
        stale_paths.extend(
            str(path) for path in paths.values() if path.is_file()
        )
        progress_ready = True
        return {
            "progress_id": progress_id,
            "progress_ready": progress_ready,
            "stale_paths": stale_paths,
            "rows": rows,
            "failures": failures,
            "all_task_paths": all_task_paths,
            "expected_task_count": expected_task_count,
        }

    for variant_id in variant_ids:
        path = paths[variant_id]
        artifact = _try_load_yaml(path)
        if artifact is not None and _reusable(
            artifact,
            scene_artifact=scene_artifact,
            realized_artifact=realized_artifact,
            goal_variant_id=variant_id,
            instruction_seed=_TASK_WORKER_INSTRUCTION_SEED,
        ):
            rows.append(
                _report_row(
                    artifact,
                    scene_spec_id=scene_id,
                    goal_variant_id=variant_id,
                    shared_style_ids=shared_styles,
                    instruction_seed=_TASK_WORKER_INSTRUCTION_SEED,
                )
            )
            continue
        try:
            render_task_record(
                scene_artifact,
                realized_artifact,
                variants[variant_id],
                instruction_seed=_TASK_WORKER_INSTRUCTION_SEED,
                shared_style_ids=shared_styles,
                instruction_candidates=pools[variant_id],
                _skip_preflight=True,
            )
        except TaskRenderError as exc:
            failures.append(
                _attributed_failure(
                    exc,
                    scene_spec_id=scene_id,
                    realized_family_id=realized_id,
                    goal_variant_id=variant_id,
                )
            )
            if path.is_file():
                stale_paths.append(str(path))
        except Exception as exc:
            stale_paths.append(f"{path}: {exc}")
        else:
            stale_paths.append(str(path))
    progress_ready = True
    return {
        "progress_id": progress_id,
        "progress_ready": progress_ready,
        "stale_paths": stale_paths,
        "rows": rows,
        "failures": failures,
        "all_task_paths": all_task_paths,
        "expected_task_count": expected_task_count,
    }


def _validate_task_scene(scene_path_text: str) -> dict[str, Any]:
    scene_path = Path(scene_path_text)
    errors: list[str] = []
    count = 0
    failure_count = 0
    progress_id = scene_path.stem
    try:
        scene_artifact = _load_yaml(scene_path)
        scene = scene_artifact["scene_spec"]
        scene_id = str(scene["id"])
        progress_id = scene_id
        realized_id = str(scene_artifact["realized_family_id"])
        realized_artifact = _load_yaml(
            _TASK_WORKER_REALIZATION_DIR / f"{realized_id}.yml"
        )
        realized = realized_artifact["realized_scene_family"]
        join_failures = _source_join_failures(
            scene_artifact, realized_artifact
        )
        if join_failures:
            errors.append(
                f"{scene_path}: source join failed: {join_failures}"
            )
            return {
                "progress_id": progress_id,
                "errors": errors,
                "count": count,
                "failure_count": failure_count,
            }
        variants = {
            str(row["id"]): row for row in realized["goal_variants"]
        }
        variant_ids = [str(value) for value in scene["goal_variant_ids"]]
        try:
            pools, shared_styles = _common_styles(realized, scene)
        except TaskRenderError:
            failure_count += len(variant_ids)
            for variant_id in variant_ids:
                path = (
                    _TASK_WORKER_OUTPUT_DIR / scene_id / f"{variant_id}.yml"
                )
                if path.is_file():
                    errors.append(
                        f"failure task unexpectedly has artifact: {path}"
                    )
            return {
                "progress_id": progress_id,
                "errors": errors,
                "count": count,
                "failure_count": failure_count,
            }
        for variant_id in variant_ids:
            variant = variants[variant_id]
            path = _TASK_WORKER_OUTPUT_DIR / scene_id / f"{variant_id}.yml"
            try:
                rebuilt = render_task_record(
                    scene_artifact,
                    realized_artifact,
                    variant,
                    instruction_seed=_TASK_WORKER_INSTRUCTION_SEED,
                    shared_style_ids=shared_styles,
                    instruction_candidates=pools[variant_id],
                )
            except TaskRenderError:
                failure_count += 1
                if path.is_file():
                    errors.append(
                        f"failure task unexpectedly has artifact: {path}"
                    )
                continue
            artifact = _try_load_yaml(path)
            if artifact is None or not _reusable(
                artifact,
                scene_artifact=scene_artifact,
                realized_artifact=realized_artifact,
                goal_variant_id=variant_id,
                instruction_seed=_TASK_WORKER_INSTRUCTION_SEED,
            ):
                errors.append(f"missing, stale, or invalid envelope: {path}")
                continue
            if canonical_data(rebuilt) != canonical_data(
                artifact["task_record"]
            ):
                errors.append(f"task payload mismatch: {path}")
                continue
            preflight = preflight_task_record(
                artifact["task_record"],
                scene_spec=scene,
                realized_family=realized,
                goal_variant=variant,
            )
            if preflight["status"] != "passed":
                errors.append(
                    f"task preflight failed: {path}: "
                    f"{preflight['failures']}"
                )
                continue
            count += 1
    except Exception as exc:
        errors.append(f"{scene_path}: {exc}")
    return {
        "progress_id": progress_id,
        "errors": errors,
        "count": count,
        "failure_count": failure_count,
    }


def freeze_task_records(
    *,
    scene_spec_dir: Path = DEFAULT_SCENE_SPEC_DIR,
    realization_dir: Path = DEFAULT_REALIZATION_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    instruction_seed: int = 0,
    workers: int = 1,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Freeze all task records for solved M12 SceneSpecs."""

    if instruction_seed < 0:
        raise ValueError("instruction_seed must be non-negative")
    scene_paths = _scene_paths(scene_spec_dir)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    expected_paths: set[Path] = set()
    delete_paths: set[Path] = set()
    expected_task_count = 0
    results = _run_task_scene_workers(
        _freeze_task_scene,
        scene_paths,
        realization_dir=realization_dir,
        output_dir=output_dir,
        instruction_seed=instruction_seed,
        workers=workers,
    )
    for index, result in enumerate(results, start=1):
        rows.extend(result["rows"])
        failures.extend(result["failures"])
        expected_paths.update(Path(path) for path in result["expected_paths"])
        delete_paths.update(Path(path) for path in result["delete_paths"])
        expected_task_count += int(result["expected_task_count"])
        if progress is not None:
            progress(index, len(scene_paths), str(result["progress_id"]))
    for path in sorted(delete_paths):
        path.unlink(missing_ok=True)
    if output_dir.is_dir():
        for path in output_dir.glob("ss_*/*.yml"):
            if path not in expected_paths:
                path.unlink()
        for directory in output_dir.glob("ss_*"):
            if directory.is_dir() and not any(directory.iterdir()):
                directory.rmdir()
    reports = _reports(
        rows,
        failures,
        scene_count=len(scene_paths),
        expected_task_count=expected_task_count,
        instruction_seed=instruction_seed,
        upstream_publication_ready=_layout_publication_ready(scene_spec_dir),
    )
    for name, report in zip(
        (
            INSTRUCTION_REPORT_FILE,
            CHECKER_REPORT_FILE,
            LEGACY_REPORT_FILE,
        ),
        reports,
    ):
        _write_yaml(output_dir / name, report)
    return {
        "implementation_complete": reports[0]["implementation_complete"],
        "coverage_complete": reports[0]["coverage_complete"],
        "stage_acceptance_ready": reports[0]["stage_acceptance_ready"],
        "publication_ready": reports[0]["publication_ready"],
        "summary": reports[0]["summary"],
        INSTRUCTION_REPORT_FILE: reports[0],
        CHECKER_REPORT_FILE: reports[1],
        LEGACY_REPORT_FILE: reports[2],
    }


def check_task_records(
    *,
    scene_spec_dir: Path = DEFAULT_SCENE_SPEC_DIR,
    realization_dir: Path = DEFAULT_REALIZATION_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    instruction_seed: int = 0,
    workers: int = 1,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Check M13 source freshness and artifact envelopes."""

    scene_paths = _scene_paths(scene_spec_dir)
    stale_paths: list[str] = []
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    all_task_paths: set[Path] = set()
    expected_task_count = 0
    results = _run_task_scene_workers(
        _check_task_scene,
        scene_paths,
        realization_dir=realization_dir,
        output_dir=output_dir,
        instruction_seed=instruction_seed,
        workers=workers,
    )
    for index, result in enumerate(results, start=1):
        stale_paths.extend(result["stale_paths"])
        rows.extend(result["rows"])
        failures.extend(result["failures"])
        all_task_paths.update(Path(path) for path in result["all_task_paths"])
        expected_task_count += int(result["expected_task_count"])
        if progress is not None and result["progress_ready"]:
            progress(index, len(scene_paths), str(result["progress_id"]))
    if output_dir.is_dir():
        stale_paths.extend(
            str(path)
            for path in output_dir.glob("ss_*/*.yml")
            if path not in all_task_paths
        )
    reports = _reports(
        rows,
        failures,
        scene_count=len(scene_paths),
        expected_task_count=expected_task_count,
        instruction_seed=instruction_seed,
        upstream_publication_ready=_layout_publication_ready(scene_spec_dir),
    )
    for name, expected in zip(
        (
            INSTRUCTION_REPORT_FILE,
            CHECKER_REPORT_FILE,
            LEGACY_REPORT_FILE,
        ),
        reports,
    ):
        actual = _try_load_yaml(output_dir / name)
        if (
            actual is None
            or not _report_hash_valid(actual)
            or canonical_data(actual) != canonical_data(expected)
        ):
            stale_paths.append(str(output_dir / name))
    return {
        "stale": bool(stale_paths),
        "stale_paths": sorted(set(stale_paths)),
        "scene_spec_count": len(scene_paths),
        "task_record_count": len(rows),
        "failure_count": len(failures),
    }


def validate_task_records(
    *,
    scene_spec_dir: Path = DEFAULT_SCENE_SPEC_DIR,
    realization_dir: Path = DEFAULT_REALIZATION_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    instruction_seed: int = 0,
    workers: int = 1,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Rebuild and validate every frozen M13 task record."""

    errors: list[str] = []
    count = 0
    failure_count = 0
    scene_paths = _scene_paths(scene_spec_dir)
    results = _run_task_scene_workers(
        _validate_task_scene,
        scene_paths,
        realization_dir=realization_dir,
        output_dir=output_dir,
        instruction_seed=instruction_seed,
        workers=workers,
    )
    for index, result in enumerate(results, start=1):
        errors.extend(result["errors"])
        count += int(result["count"])
        failure_count += int(result["failure_count"])
        if progress is not None:
            progress(index, len(scene_paths), str(result["progress_id"]))
    check = check_task_records(
        scene_spec_dir=scene_spec_dir,
        realization_dir=realization_dir,
        output_dir=output_dir,
        instruction_seed=instruction_seed,
        workers=workers,
        progress=None,
    )
    errors.extend(f"stale: {path}" for path in check["stale_paths"])
    return {
        "valid": not errors,
        "errors": errors,
        "scene_spec_count": len(scene_paths),
        "task_record_count": count,
        "failure_count": failure_count,
    }
