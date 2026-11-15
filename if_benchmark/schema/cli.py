"""BenchmarkV1 Milestone 6 schema freezer and validator.

This module owns the canonical contract artifacts for the BenchmarkV1 object
model.  It deliberately avoids simulator imports and later milestone behavior:
the implementation validates record shape, reference boundaries, deterministic
serialization, and provenance affordances only.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import jsonschema

from .artifacts import (
    load_examples as load_artifact_examples,
    load_yaml as load_artifact_yaml,
    write_artifact_set,
    write_yaml as write_artifact_yaml,
)
from .contracts import (
    CONFIG,
    CONSTRAINT_ALLOWED_KEYS,
    CONSTRAINT_REQUIRED_KEYS,
    DEFAULT_AUDIT_PATH,
    DEFAULT_EXAMPLES_DIR,
    DEFAULT_SCHEMA_PATH,
    DOCUMENT_KINDS,
    GOAL_REF_KEYS,
    KIND_TO_PAYLOAD,
    KIND_TO_REF,
    ROLE_REF_PARAM_KEYS,
    SCHEMA_VERSION,
)
from .serialization import (
    artifact_hash as public_artifact_hash,
    canonical_json as public_canonical_json,
    deterministic_id as public_deterministic_id,
    migrate_document as public_migrate_document,
)
from .validation import validate_bundle


_DEFAULT_SCHEMA_VALIDATOR: jsonschema.Draft202012Validator | None = None


def config_hash() -> str:
    return _sha256_bytes(_canonical_json(CONFIG).encode("utf-8"))


def deterministic_id(prefix: str, payload: Any, *, length: int = 16) -> str:
    """Return a stable id derived from canonical JSON content."""

    return public_deterministic_id(prefix, payload, length=length)


def migrate_document(document: dict[str, Any], *, from_version: int | None = None) -> dict[str, Any]:
    """Milestone 6 migration stub.

    Version 1 is the first canonical schema.  The helper exists so later
    milestones have a stable place to add schema migrations without changing
    validator call sites.
    """

    return public_migrate_document(document, from_version=from_version)


def build_schema_artifact() -> dict[str, Any]:
    """Build the YAML-encoded JSON Schema artifact."""

    schema: dict[str, Any] = {
        "$schema": CONFIG["schema_dialect"],
        "$id": "https://robotwin.local/if_benchmark/benchmark_v1.schema.yml",
        "title": "BenchmarkV1 Milestone 6 Contract Schema",
        "description": "Canonical object model contracts for BenchmarkV1 pipeline artifacts.",
        "schema_version": SCHEMA_VERSION,
        "type": "object",
        "required": ["schema_version", "kind"],
        "properties": {
            "schema_version": {"const": SCHEMA_VERSION},
            "kind": {"enum": list(DOCUMENT_KINDS)},
        },
        "additionalProperties": False,
        "oneOf": [],
        "$defs": _schema_defs(),
    }
    for kind in DOCUMENT_KINDS:
        payload_key = KIND_TO_PAYLOAD[kind]
        schema["properties"][payload_key] = {"$ref": KIND_TO_REF[kind]}
        other_payloads = [
            KIND_TO_PAYLOAD[other]
            for other in DOCUMENT_KINDS
            if KIND_TO_PAYLOAD[other] != payload_key
        ]
        schema["oneOf"].append(
            {
                "properties": {"kind": {"const": kind}},
                "required": ["kind", payload_key],
                "not": {
                    "anyOf": [{"required": [other_payload]} for other_payload in other_payloads]
                },
            }
        )
    return schema


def build_examples() -> dict[str, dict[str, Any]]:
    """Return all canonical Milestone 6 example documents."""

    d1a_archetype = _d1a_attribute_archetype()
    d2b_role_graph = _d2b_relation_role_graph()
    assignment = _d1a_assignment_candidate()
    goal_variant = _d1a_goal_variant("pick_a", "subj_a", "obj_a", "red")
    realized = _d1a_realized_scene_family()
    scene_spec = _d1a_scene_spec()
    goal = goal_variant["goal"]
    semantic_frame = {
        "verb": "pick",
        "action_family": "A2O_state",
        "relation": None,
        "state": "acquired",
        "arm": "left",
        "entity_refs": {"subject": "obj_a"},
        "descriptors": goal["descriptors"],
        "params": {},
    }
    instruction_candidates = [
        {
            "template_id": f"tpl_pick_style_{index}",
            "style_id": f"style_{index}",
            "text": text,
            "semantic_frame": semantic_frame,
            "goal_signature": "goal-signature",
        }
        for index, text in enumerate(
            (
                "Use the left arm to pick up the red cube.",
                "With the left arm, lift the red cube from the table.",
                "Pick up the red cube using the left arm.",
                "Using your left arm, grasp and lift the red cube.",
            )
        )
    ]
    scene_integrity_policy = {
        "policy_version": (
            "m13_m15_scene_integrity_v1_explicit_goal_entities_exempt"
        ),
        "participant_source": (
            "concrete_goal_recursive_scene_entity_references_v1"
        ),
        "translation_limit_m": 0.004,
        "rotation_limit_deg": 3.0,
        "articulation_normalized_limit": 0.02,
        "limits_inclusive": True,
    }
    scene_integrity_policy["policy_hash"] = _artifact_hash(
        scene_integrity_policy, None
    )
    task_record = {
        "id": "task_d1a_pick_red",
        "scene_spec_id": scene_spec["id"],
        "realized_family_id": realized["id"],
        "goal_variant_id": goal_variant["id"],
        "goal": goal,
        "goal_provenance": {
            "abstract_family_id": "asf_d1a_color_subject_pair",
            "assignment_id": goal_variant["assignment_id"],
            "schema_id": goal_variant["schema_id"],
            "role_refs": goal_variant["role_refs"],
            "entity_refs": goal_variant["entity_refs"],
            "scene_entity_keys": sorted(scene_spec["entities"]),
            "intended_axes": goal_variant["intended_axes"],
            "goal_hash": "goal-hash",
            "source_scene_hash": "scene-hash",
            "source_realization_hash": "realization-hash",
        },
        "instruction": {
            "language": "en",
            "renderer_version": 3,
            "candidates": instruction_candidates,
            "selected": instruction_candidates[0],
            "selection": {
                "policy": "sha256_scene_shared_style_v1",
                "instruction_seed": 0,
                "scene_spec_id": scene_spec["id"],
                "shared_style_ids": [f"style_{index}" for index in range(4)],
                "selected_style_id": "style_0",
                "selection_digest": "selection-digest",
                "candidate_set_hash": "candidate-set-hash",
            },
        },
        "success_contract": {
            "contract_id": "benchmarkv1_shared_success_contract",
            "success_contract_version": "benchmarkv1_success_contract_v1",
            "success_contract_hash": "success-contract-hash",
        },
        "checker_config": {
            "version": 5,
            "family": "A2O_state",
            "goal_signature": "goal-signature",
            "entity_refs": {"subject": "obj_a"},
            "success_contract": {
                "contract_id": "benchmarkv1_shared_success_contract",
                "success_contract_version": "benchmarkv1_success_contract_v1",
                "success_contract_hash": "success-contract-hash",
            },
            "predicate": {
                "name": "pick_success",
                "arguments": {"subject": "obj_a", "requested_arm": "left"},
            },
            "scene_integrity": {
                "policy": scene_integrity_policy,
                "exempt_entities": ["obj_a"],
                "protected_entities": ["obj_b"],
                "protected_articulated_entities": [],
            },
        },
        "skill_config": {
            "version": 8,
            "operation_anchor_policy_version": "operation_anchor_projection_v2_link_fk",
            "goal_signature": "goal-signature",
            "semantic_arm": "left",
            "execution_arm": "left",
            "route": "direct",
            "steps": [{"op": "grasp", "subject": "obj_a", "arm": "left"}],
        },
        "legacy_export": {
            "version": 1,
            "task_name": "if_scene",
            "task_variant": goal_variant["export"]["variant"],
            "setup_payload": {
                "task_record_id": "task_d1a_pick_red",
                "scene_spec_id": scene_spec["id"],
                "realized_family_id": realized["id"],
                "layout_seed": 7,
                "embodiment": "aloha-agilex",
                "instruction": instruction_candidates[0]["text"],
                "task_variant": goal_variant["export"]["variant"],
            },
        },
        "provenance": {
            "generator": "if_benchmark.tasks",
            "config_hash": "task-config-hash",
            "sources": [
                {"kind": "scene_spec", "hash": "scene-hash", "record_id": scene_spec["id"]}
            ],
        },
    }
    minimal_pair = {
        "id": "mp_d1a_red_blue",
        "scene_spec_id": scene_spec["id"],
        "realized_family_id": realized["id"],
        "layout_seed": scene_spec["layout_seed"],
        "assignment_id": goal_variant["assignment_id"],
        "observation_id": scene_spec["initial_observation"]["observation_id"],
        "scene_state_hash": scene_spec["initial_observation"]["scene_state_hash"],
        "from_goal_variant_id": "pick_a",
        "to_goal_variant_id": "pick_b",
        "intended_axis": "D1a",
        "intended_subtype": "color",
        "intended_dcode": "D1a.color",
        "actual_axes": ["D1a"],
        "actual_dcodes": ["D1a.color"],
        "unmapped_semantic_deltas": [],
        "status": "valid",
        "source_scene_hash": "scene-hash",
        "source_realization_hash": "realization-hash",
        "from_endpoint": {
            "goal_variant_id": "pick_a",
            "task_record_id": "task_pick_a",
            "task_artifact_hash": "task-a-hash",
            "goal_hash": "goal-a-hash",
            "instruction_hash": "instruction-a-hash",
            "checker_hash": "checker-a-hash",
            "style_id": "style_0",
        },
        "to_endpoint": {
            "goal_variant_id": "pick_b",
            "task_record_id": "task_pick_b",
            "task_artifact_hash": "task-b-hash",
            "goal_hash": "goal-b-hash",
            "instruction_hash": "instruction-b-hash",
            "checker_hash": "checker-b-hash",
            "style_id": "style_0",
        },
    }
    contrast_members = [
        {
            "matrix_index": index,
            "goal_variant_id": variant,
            "axis_value": [
                {
                    "slot": "subject",
                    "entity": entity,
                    "descriptor": {
                        "subtype": "color",
                        "value": color,
                    },
                }
            ],
            "intent_goal_hash": f"goal-{suffix}-hash",
            "task_record_id": f"task_pick_{suffix}",
            "task_artifact_hash": f"task-{suffix}-hash",
            "goal_hash": f"goal-{suffix}-hash",
            "instruction_hash": f"instruction-{suffix}-hash",
            "checker_hash": f"checker-{suffix}-hash",
            "style_id": "style_0",
        }
        for index, (variant, entity, color, suffix) in enumerate(
            (
                ("pick_a", "obj_a", "red", "a"),
                ("pick_b", "obj_b", "blue", "b"),
            )
        )
    ]
    axis_contrast_set = {
        "id": "cs_d1a_red_blue",
        "scene_spec_id": scene_spec["id"],
        "realized_family_id": realized["id"],
        "layout_seed": scene_spec["layout_seed"],
        "observation_id": scene_spec["initial_observation"]["observation_id"],
        "scene_state_hash": scene_spec["initial_observation"]["scene_state_hash"],
        "axis": "D1a",
        "subtype": "color",
        "dcode": "D1a.color",
        "context_signature": "context-hash",
        "k": 2,
        "members": contrast_members,
        "pair_ids": [minimal_pair["id"]],
        "matrix_ready": True,
    }
    scene_validation_record = {
        "artifact_kind": "scene_validation_record",
        "schema_version": 1,
        "scene_validation_version": 9,
        "id": "svr_d1a_color",
        "scene_spec_id": scene_spec["id"],
        "realized_family_id": realized["id"],
        "source_scene_hash": "scene-hash",
        "source_realization_hash": "realization-hash",
        "source_unit_hash": "unit-hash",
        "plan_hash": "plan-hash",
        "config_hash": "validation-config-hash",
        "contract_id": "benchmarkv1_shared_success_contract",
        "success_contract_version": "benchmarkv1_success_contract_v1",
        "success_contract_hash": "success-contract-hash",
        "calibration_only": True,
        "publication_ready": False,
        "descriptor_gate": {
            "status": "passed",
            "passed": True,
            "render_count": 1,
            "frame_hash": "frame-hash",
            "entity_metrics": {},
            "visible_entities": {"obj_a": True, "obj_b": True},
            "checks": [],
            "failures": [],
        },
        "not_success_at_t0": {"passed": True, "task_count": 1},
        "infrastructure_failure": False,
        "source_failure": False,
        "internal_failure": False,
        "task_results": [
            {
                "task_record_id": task_record["id"],
                "goal_variant_id": task_record["goal_variant_id"],
                "source_task_hash": "task-hash",
                "validated": True,
                "excluded": False,
                "success_count": 2,
                "denominator": 3,
                "minimum_success_count": 2,
                "success_rate": 2.0 / 3.0,
                "classification_counts": {"failure": 1, "intended": 2},
                "t0_false": True,
                "contract_id": "benchmarkv1_shared_success_contract",
                "success_contract_version": "benchmarkv1_success_contract_v1",
                "success_contract_hash": "success-contract-hash",
                "success_snapshot_hashes": [
                    "success-snapshot-seed-0",
                    "success-snapshot-seed-1",
                    "success-snapshot-seed-2",
                ],
                "infrastructure_failure_count": 0,
                "source_failure_count": 0,
                "internal_failure_count": 0,
                "attempts": [],
            }
        ],
        "summary": {
            "task_count": 1,
            "validated_task_count": 1,
            "excluded_task_count": 0,
        },
        "provenance": {
            "generator": "if_benchmark.scene_validation.freeze",
            "config_hash": "validation-config-hash",
            "sources": [
                {
                    "kind": "scene_spec",
                    "hash": "scene-hash",
                    "record_id": scene_spec["id"],
                }
            ],
        },
        "artifact_hash": "record-hash",
    }

    return {
        "axis_plan_d4_arm.yml": {
            "schema_version": SCHEMA_VERSION,
            "kind": "axis_plan",
            "axis_plan": [
                {
                    "axis": "D4",
                    "subtype": "arm",
                    "role": "subject",
                    "values": ["left", "right"],
                }
            ],
        },
        "abstract_family_d1a_color.yml": {
            "schema_version": SCHEMA_VERSION,
            "kind": "abstract_family",
            "abstract_family": {
                "id": "asf_d1a_color_subject_pair",
                "axis_plan": [
                    {
                        "axis": "D1a",
                        "subtype": "color",
                        "role": "subject",
                        "values": "any_reviewed_pair",
                    }
                ],
                "archetypes": ["attribute_subject_pair"],
                "randomization": {
                    "assignment_seed_policy": "sample_assignment_candidate",
                    "layout_seed_policy": "sample_layout_per_realized_family",
                },
            },
        },
        "archetype_d1a_attribute_subject_pair.yml": {
            "schema_version": SCHEMA_VERSION,
            "kind": "archetype",
            "archetype": d1a_archetype,
        },
        "role_graph_d2b_relation_pair.yml": {
            "schema_version": SCHEMA_VERSION,
            "kind": "role_graph",
            "role_graph": d2b_role_graph,
        },
        "constraint_reachable_by_both.yml": {
            "schema_version": SCHEMA_VERSION,
            "kind": "constraint",
            "constraint": _constraint(
                "c_reachable_both",
                "reach_zone",
                "layout",
                "layout",
                ["subj"],
                {"allowed": ["both"]},
            ),
        },
        "variant_schema_d1b_spatial.yml": {
            "schema_version": SCHEMA_VERSION,
            "kind": "variant_schema",
            "variant_schema": {
                "id": "pick_left_of_landmark",
                "action": {"verb": "pick", "arm": "left"},
                "role_bindings": {"subject": "subj_left"},
                "descriptors": {
                    "subject": [
                        {
                            "kind": "spatial",
                            "predicates": ["left_of"],
                            "source_role": "subj_left",
                            "reference_role": "landmark",
                        }
                    ]
                },
                "params": {},
            },
        },
        "assignment_candidate_d1a_color.yml": {
            "schema_version": SCHEMA_VERSION,
            "kind": "assignment_candidate",
            "assignment_candidate": assignment,
        },
        "concrete_goal_typed_descriptors.yml": {
            "schema_version": SCHEMA_VERSION,
            "kind": "concrete_goal",
            "concrete_goal": {
                "entity_keys": ["obj_subject", "obj_landmark"],
                "goal": {
                    "verb": "pick",
                    "subject": "obj_subject",
                    "arm": "left",
                    "target": None,
                    "subject2": None,
                    "target2": None,
                    "descriptors": {
                        "subject": {
                            "attribute": {"subtype": "color", "value": "red"},
                            "spatial": {
                                "predicates": ["left_of"],
                                "reference": "obj_landmark",
                            },
                            "ordinal": {
                                "basis": "landmark",
                                "order": "nearest",
                                "reference": "obj_landmark",
                            },
                            "part": {"region": "handle"},
                        }
                    },
                    "params": {},
                },
            },
        },
        "goal_variant_d1a_pick_a.yml": {
            "schema_version": SCHEMA_VERSION,
            "kind": "goal_variant",
            "goal_variant": goal_variant,
        },
        "realized_scene_family_d1a_color.yml": {
            "schema_version": SCHEMA_VERSION,
            "kind": "realized_scene_family",
            "realized_scene_family": realized,
        },
        "realized_scene_family_unit_contracts.yml": {
            "schema_version": SCHEMA_VERSION,
            "kind": "realized_scene_family",
            "realized_scene_family": _unit_contract_realized_scene_family(),
        },
        "scene_spec_d1a_color.yml": {
            "schema_version": SCHEMA_VERSION,
            "kind": "scene_spec",
            "scene_spec": scene_spec,
        },
        "task_record_d1a_color.yml": {
            "schema_version": SCHEMA_VERSION,
            "kind": "task_record",
            "task_record": task_record,
        },
        "minimal_pair_d1a_color.yml": {
            "schema_version": SCHEMA_VERSION,
            "kind": "minimal_pair",
            "minimal_pair": minimal_pair,
        },
        "axis_contrast_set_d1a_color.yml": {
            "schema_version": SCHEMA_VERSION,
            "kind": "axis_contrast_set",
            "axis_contrast_set": axis_contrast_set,
        },
        "scene_validation_record_d1a_color.yml": {
            "schema_version": SCHEMA_VERSION,
            "kind": "scene_validation_record",
            "scene_validation_record": scene_validation_record,
        },
    }


def freeze_schema(
    schema_path: Path = DEFAULT_SCHEMA_PATH,
    examples_dir: Path = DEFAULT_EXAMPLES_DIR,
    audit_path: Path = DEFAULT_AUDIT_PATH,
) -> dict[str, Any]:
    """Write the canonical schema, examples, and audit report."""

    schema = build_schema_artifact()
    examples = build_examples()
    audit = build_schema_audit_report(schema, examples, schema_path, examples_dir, audit_path)
    if not audit["summary"]["acceptance_ready"]:
        raise ValueError(
            "refusing to freeze incomplete BenchmarkV1 schema artifacts: "
            f"{audit['summary']}"
        )

    write_artifact_set(
        schema_path,
        examples_dir,
        audit_path,
        schema=schema,
        examples=examples,
        audit=audit,
    )
    return {"schema": schema, "examples": examples, "audit": audit}


def check_schema(
    schema_path: Path = DEFAULT_SCHEMA_PATH,
    examples_dir: Path = DEFAULT_EXAMPLES_DIR,
    audit_path: Path = DEFAULT_AUDIT_PATH,
) -> dict[str, Any]:
    """Compare current Milestone 6 artifacts with generated canonical output."""

    generated_schema = build_schema_artifact()
    generated_examples = build_examples()
    generated_audit = build_schema_audit_report(
        generated_schema,
        generated_examples,
        schema_path,
        examples_dir,
        audit_path,
    )

    current_schema = _load_yaml(schema_path)
    current_audit = _load_yaml(audit_path)
    current_examples = _load_examples(examples_dir)

    missing_artifacts = []
    if current_schema is None:
        missing_artifacts.append(_path_string(schema_path))
    if current_audit is None:
        missing_artifacts.append(_path_string(audit_path))
    for name in generated_examples:
        if name not in current_examples:
            missing_artifacts.append(_path_string(examples_dir / name))

    extra_examples = sorted(set(current_examples) - set(generated_examples))
    stale_examples = [
        name
        for name in sorted(set(current_examples) & set(generated_examples))
        if _canonical_json(current_examples[name]) != _canonical_json(generated_examples[name])
    ]
    stale = bool(
        missing_artifacts
        or extra_examples
        or stale_examples
        or _canonical_json(current_schema) != _canonical_json(generated_schema)
        or _canonical_json(current_audit) != _canonical_json(generated_audit)
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "stale": stale,
        "missing_artifacts": missing_artifacts,
        "extra_examples": extra_examples,
        "stale_examples": stale_examples,
        "current_schema_hash": _artifact_hash(current_schema, None) if current_schema else None,
        "generated_schema_hash": _artifact_hash(generated_schema, None),
        "current_audit_hash": current_audit.get("audit_hash") if current_audit else None,
        "generated_audit_hash": generated_audit.get("audit_hash"),
    }


def validate_file(path: Path, kind: str | None = None) -> dict[str, Any]:
    document = _load_yaml_required(path)
    result = validate_document(document, kind=kind)
    result["path"] = _path_string(path)
    return result


def validate_document(
    document: Any,
    *,
    kind: str | None = None,
    schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate one BenchmarkV1 document and return a structured report."""

    errors: list[dict[str, Any]] = []
    try:
        normalized = _normalize_document(document, kind)
    except ValueError as exc:
        return {
            "schema_version": SCHEMA_VERSION,
            "valid": False,
            "kind": kind,
            "errors": [_error([], str(exc), "document_normalization")],
        }

    global _DEFAULT_SCHEMA_VALIDATOR
    if schema is None:
        if _DEFAULT_SCHEMA_VALIDATOR is None:
            _DEFAULT_SCHEMA_VALIDATOR = jsonschema.Draft202012Validator(
                build_schema_artifact()
            )
        validator = _DEFAULT_SCHEMA_VALIDATOR
    else:
        validator = jsonschema.Draft202012Validator(schema)
    for err in sorted(validator.iter_errors(normalized), key=lambda item: list(item.path)):
        errors.append(_error(list(err.path), err.message, "json_schema"))

    if not errors:
        doc_kind, payload = _document_kind_payload(normalized)
        errors.extend(_semantic_errors(doc_kind, payload))
    return {
        "schema_version": SCHEMA_VERSION,
        "valid": not errors,
        "kind": normalized.get("kind"),
        "errors": errors,
    }


def build_schema_audit_report(
    schema: dict[str, Any],
    examples: dict[str, dict[str, Any]],
    schema_path: Path = DEFAULT_SCHEMA_PATH,
    examples_dir: Path = DEFAULT_EXAMPLES_DIR,
    audit_path: Path = DEFAULT_AUDIT_PATH,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    valid_count = 0
    for name, example in sorted(examples.items()):
        result = validate_document(example, schema=schema)
        if result["valid"]:
            valid_count += 1
        rows.append(
            {
                "path": _path_string(examples_dir / name),
                "kind": result.get("kind"),
                "valid": result["valid"],
                "error_count": len(result["errors"]),
                "hash": _artifact_hash(example, None),
                "errors": result["errors"],
            }
        )

    constraint_count_by_scope: Counter[str] = Counter()
    constraint_count_by_phase: Counter[str] = Counter()
    document_count_by_kind: Counter[str] = Counter()
    descriptor_kinds: set[str] = set()
    unit_types: set[str] = set()
    for example in examples.values():
        kind, payload = _document_kind_payload(example)
        document_count_by_kind[kind] += 1
        for constraint in _collect_constraints(payload):
            constraint_count_by_scope[str(constraint.get("scope"))] += 1
            constraint_count_by_phase[str(constraint.get("phase"))] += 1
        descriptor_kinds.update(_collect_descriptor_kinds(payload))
        unit_types.update(_collect_unit_types(payload))

    bundle_names = [
        "abstract_family_d1a_color.yml",
        "archetype_d1a_attribute_subject_pair.yml",
        "assignment_candidate_d1a_color.yml",
        "realized_scene_family_d1a_color.yml",
        "scene_spec_d1a_color.yml",
    ]
    bundle_report = validate_bundle([examples[name] for name in bundle_names])
    missing_document_kinds = sorted(set(DOCUMENT_KINDS) - set(document_count_by_kind))
    missing_descriptor_kinds = sorted(
        {"attribute", "spatial", "ordinal", "part"} - descriptor_kinds
    )
    missing_unit_types = sorted(
        {"ObjectUnit", "LandmarkRelationUnit", "ReceiverReservationUnit"} - unit_types
    )
    acceptance_ready = (
        valid_count == len(examples)
        and bundle_report["valid"]
        and not missing_document_kinds
        and not missing_descriptor_kinds
        and not missing_unit_types
    )

    audit = {
        "schema_version": SCHEMA_VERSION,
        "config_hash": config_hash(),
        "source_paths": {
            "schema": _path_string(schema_path),
            "examples": _path_string(examples_dir),
            "audit": _path_string(audit_path),
        },
        "schema_hash": _artifact_hash(schema, None),
        "summary": {
            "acceptance_ready": acceptance_ready,
            "example_count": len(examples),
            "valid_example_count": valid_count,
            "invalid_example_count": len(examples) - valid_count,
            "document_count_by_kind": dict(sorted(document_count_by_kind.items())),
            "missing_document_kinds": missing_document_kinds,
            "descriptor_kinds": sorted(descriptor_kinds),
            "missing_descriptor_kinds": missing_descriptor_kinds,
            "unit_types": sorted(unit_types),
            "missing_unit_types": missing_unit_types,
            "representative_bundle_valid": bundle_report["valid"],
            "representative_bundle_error_count": len(bundle_report["errors"]),
            "constraint_count_by_scope": dict(sorted(constraint_count_by_scope.items())),
            "constraint_count_by_phase": dict(sorted(constraint_count_by_phase.items())),
        },
        "examples": rows,
        "representative_bundle": {
            "paths": [_path_string(examples_dir / name) for name in bundle_names],
            "valid": bundle_report["valid"],
            "errors": bundle_report["errors"],
        },
        "invariant_checks": [
            {
                "name": "flat_constraint_records",
                "status": "covered",
                "description": "All constraints require id/type/scope/phase/roles/params/hard.",
            },
            {
                "name": "derived_constraint_indexes",
                "status": "covered",
                "description": "Optional constraint_index must match derived indexes from constraints.",
            },
            {
                "name": "variant_schema_role_refs",
                "status": "covered",
                "description": "VariantSchema role references resolve against RoleGraph roles.",
            },
            {
                "name": "concrete_goal_entity_refs",
                "status": "covered",
                "description": "ConcreteGoal references resolve against entity keys in context.",
            },
            {
                "name": "legacy_variant_export_boundary",
                "status": "covered",
                "description": "Legacy variant strings are accepted only at export.variant.",
            },
            {
                "name": "single_kind_payload",
                "status": "covered",
                "description": "Each document contains exactly one payload matching kind.",
            },
            {
                "name": "typed_descriptor_contracts",
                "status": "covered",
                "description": "Attribute, spatial, ordinal, and part descriptors have typed fields.",
            },
            {
                "name": "bundle_reference_alignment",
                "status": "covered",
                "description": "Representative family-to-scene references validate as one bundle.",
            },
            {
                "name": "deterministic_serialization_and_provenance",
                "status": "covered",
                "description": "Public helpers provide stable serialization, ids, hashes, and source refs.",
            },
        ],
        "migration_helpers": {
            "supported_versions": [SCHEMA_VERSION],
            "helpers": ["migrate_document"],
            "default_policy": "identity_for_v1_raise_other_versions",
        },
        "public_helpers": [
            "artifact_hash",
            "build_provenance",
            "canonical_json",
            "canonical_yaml",
            "deterministic_id",
            "dump_document",
            "load_document",
            "migrate_document",
            "validate_bundle",
            "validate_document",
        ],
    }
    audit["audit_hash"] = _artifact_hash(audit, "audit_hash")
    return audit


def derive_constraint_index(constraints: list[dict[str, Any]]) -> dict[str, Any]:
    by_phase: dict[str, list[str]] = {}
    by_scope: dict[str, list[str]] = {}
    for constraint in constraints:
        constraint_id = str(constraint.get("id"))
        by_phase.setdefault(str(constraint.get("phase")), []).append(constraint_id)
        by_scope.setdefault(str(constraint.get("scope")), []).append(constraint_id)
    return {
        "by_phase": {key: by_phase[key] for key in sorted(by_phase)},
        "by_scope": {key: by_scope[key] for key in sorted(by_scope)},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze, check, or validate BenchmarkV1 schema artifacts.")
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)
    parser.add_argument("--examples-dir", type=Path, default=DEFAULT_EXAMPLES_DIR)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT_PATH)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("freeze")
    subparsers.add_parser("check")
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("path", type=Path)
    validate_parser.add_argument("--kind", choices=DOCUMENT_KINDS)
    bundle_parser = subparsers.add_parser("validate-bundle")
    bundle_parser.add_argument("paths", type=Path, nargs="+")

    args = parser.parse_args(argv)
    if args.command == "freeze":
        artifacts = freeze_schema(args.schema, args.examples_dir, args.audit)
        print(
            f"freeze: wrote BenchmarkV1 schema and {len(artifacts['examples'])} examples; "
            f"audit {args.audit}"
        )
        return 0
    if args.command == "check":
        report = check_schema(args.schema, args.examples_dir, args.audit)
        print(_check_status_line(report))
        return 1 if report.get("stale") else 0
    if args.command == "validate":
        result = validate_file(args.path, kind=args.kind)
        if result["valid"]:
            print(f"validate: {args.path} is valid ({result['kind']})")
            return 0
        print(f"validate: {args.path} is invalid ({len(result['errors'])} errors)", file=sys.stderr)
        for err in result["errors"]:
            print(f"- {err['path']}: {err['message']}", file=sys.stderr)
        return 1
    if args.command == "validate-bundle":
        result = validate_bundle(args.paths)
        if result["valid"]:
            print(
                f"validate-bundle: {result['document_count']} BenchmarkV1 documents are valid"
            )
            return 0
        print(
            f"validate-bundle: bundle is invalid ({len(result['errors'])} errors)",
            file=sys.stderr,
        )
        for err in result["errors"]:
            print(
                f"- document {err['document']} {err['path']}: {err['message']}",
                file=sys.stderr,
            )
        return 1
    parser.error(f"unknown command: {args.command}")
    return 2


def _schema_defs() -> dict[str, Any]:
    identifier = {"type": "string", "pattern": "^[A-Za-z][A-Za-z0-9_\\-]*$"}
    role_id = {"type": "string", "pattern": "^[A-Za-z][A-Za-z0-9_]*$"}
    entity_key = {"type": "string", "pattern": "^obj_[A-Za-z0-9_]+$"}
    model_id = {"oneOf": [{"type": "string"}, {"type": "integer"}]}
    axis_spec = {
        "type": "object",
        "required": ["axis"],
        "properties": {
            "axis": {"enum": ["D1a", "D1b", "D1c", "D2a", "D2b", "D4"]},
            "subtype": {"type": "string"},
            "role": {"type": "string"},
            "value": {},
            "values": {},
        },
        "additionalProperties": False,
    }
    descriptor_bundle = {
        "type": "object",
        "minProperties": 1,
        "properties": {
            "attribute": {"$ref": "#/$defs/AttributeDescriptor"},
            "spatial": {"$ref": "#/$defs/SpatialDescriptor"},
            "ordinal": {"$ref": "#/$defs/OrdinalDescriptor"},
            "part": {"$ref": "#/$defs/PartDescriptor"},
        },
        "additionalProperties": False,
    }
    # Optional frozen M11 resources. Their numerical geometry and content/source
    # hashes are additionally checked by the production manipulation contract.
    digest = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
    vector3 = {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "number"}}
    pieces = {"type": "array", "minItems": 1, "items": {
        "type": "array", "minItems": 4, "items": vector3}}
    resource = {
        "type": "object",
        "required": ["policy", "geometry", "geometry_identity", "geometry_key", "robots",
                     "robot_geometry_key", "support_plane_z_m", "subject_identity", "content_hash"],
        "properties": {
            "policy": {"type": "object"},
            "geometry": {"type": "object", "required": ["pieces", "sources", "contacts", "quat_wxyz"],
                         "properties": {"pieces": pieces, "sources": {"type": "array"},
                                        "contacts": {"type": "object"}, "quat_wxyz": {
                                            "type": "array", "minItems": 4, "maxItems": 4,
                                            "items": {"type": "number"}}},
                         "additionalProperties": False},
            "geometry_identity": {"type": "object"}, "geometry_key": digest,
            "robots": {"type": "object", "required": ["left", "right"],
                       "properties": {"left": {"type": "object"}, "right": {"type": "object"}},
                       "additionalProperties": False},
            "robot_geometry_key": digest, "support_plane_z_m": {"type": "number"},
            "subject_identity": {"type": "object"}, "content_hash": digest,
        },
        "additionalProperties": False,
    }
    operation_properties = {
        "operation_sources": {"type": "object", "additionalProperties": digest},
        "operation_spaces": {"type": "object", "propertyNames": entity_key,
                             "additionalProperties": resource},
        "operation_obstacles": {"type": "object", "propertyNames": entity_key,
            "additionalProperties": {
                "type": "object", "required": ["pieces", "geometry_key", "link_matrices"],
                "properties": {"pieces": pieces, "geometry_key": digest,
                               "link_matrices": {"type": "object", "additionalProperties": {
                                   "type": "array", "minItems": 4, "maxItems": 4,
                                   "items": {"type": "array", "minItems": 4, "maxItems": 4,
                                             "items": {"type": "number"}}}}},
                "additionalProperties": False}},
        "operation_spacing": {"type": "object",
            "required": ["policy", "original_gap_m", "gap_m", "proposal_branch",
                         "proposal_branches_by_subject", "branch_count", "not_a_grasp_certificate"],
            "properties": {"policy": {"type": "string"}, "original_gap_m": {"type": "number"},
                           "gap_m": {"type": "number"}, "proposal_branch": {"type": "object"},
                           "proposal_branches_by_subject": {"type": "object"},
                           "branch_count": {"type": "integer", "minimum": 1},
                           "not_a_grasp_certificate": {"const": True}},
            "additionalProperties": False},
    }
    return {
        "Identifier": identifier,
        "RoleId": role_id,
        "EntityKey": entity_key,
        "ModelId": model_id,
        "AxisSpec": axis_spec,
        "AxisPlan": {
            "type": "array",
            "minItems": 1,
            "items": {"$ref": "#/$defs/AxisSpec"},
        },
        "Role": {
            "type": "object",
            "required": ["id", "kind"],
            "properties": {
                "id": role_id,
                "kind": {"type": "string"},
                "allowed_asset_roles": {"type": "array", "items": {"type": "string"}},
                "descriptor_role": {"type": "string"},
                "cardinality": {"type": "integer", "minimum": 1},
                "semantic_tags": {"type": "array", "items": {"type": "string"}},
                "required_uses": {"type": "array", "items": {"type": "string"}},
                "layout_role": {"type": "string"},
            },
            "additionalProperties": False,
        },
        "Constraint": {
            "type": "object",
            "required": list(CONSTRAINT_REQUIRED_KEYS),
            "properties": {
                "id": identifier,
                "type": {"type": "string"},
                "scope": {"enum": ["semantic", "asset", "unit", "layout", "gate"]},
                "phase": {"enum": ["assignment", "unit_build", "layout", "gate", "manifest"]},
                "roles": {"type": "array", "items": role_id},
                "params": {"type": "object"},
                "hard": {"type": "boolean"},
                "related_constraint": identifier,
            },
            "additionalProperties": False,
        },
        "ConstraintIndex": {
            "type": "object",
            "required": ["by_phase", "by_scope"],
            "properties": {
                "by_phase": {
                    "type": "object",
                    "additionalProperties": {"type": "array", "items": identifier},
                },
                "by_scope": {
                    "type": "object",
                    "additionalProperties": {"type": "array", "items": identifier},
                },
            },
            "additionalProperties": False,
        },
        "Action": {
            "type": "object",
            "required": ["verb"],
            "properties": {
                "verb": {"type": "string", "minLength": 1},
                "family": {"type": "string", "minLength": 1},
                "arm": {"type": ["string", "null"]},
                "relation": {"type": ["string", "null"]},
                "state": {"type": ["string", "null"]},
            },
            "additionalProperties": False,
        },
        "DescriptorConstructor": {
            "type": "object",
            "required": ["kind"],
            "properties": {
                "kind": {"enum": ["attribute", "spatial", "ordinal", "part"]},
                "subtype": {"type": "string"},
                "basis": {"type": "string"},
                "order": {"type": "string"},
                "predicates": {"type": "array", "items": {"type": "string"}},
                "source_role": role_id,
                "reference_role": role_id,
                "params": {"type": "object"},
            },
            "additionalProperties": False,
            "allOf": [
                {
                    "if": {"properties": {"kind": {"const": "attribute"}}},
                    "then": {
                        "required": ["subtype", "source_role"],
                        "not": {
                            "anyOf": [
                                {"required": ["basis"]},
                                {"required": ["order"]},
                                {"required": ["predicates"]},
                                {"required": ["reference_role"]},
                            ]
                        },
                    },
                },
                {
                    "if": {"properties": {"kind": {"const": "spatial"}}},
                    "then": {
                        "required": ["predicates", "source_role", "reference_role"],
                        "properties": {
                            "predicates": {
                                "type": "array",
                                "minItems": 1,
                                "uniqueItems": True,
                                "items": {"type": "string", "minLength": 1},
                            }
                        },
                        "not": {
                            "anyOf": [
                                {"required": ["subtype"]},
                                {"required": ["basis"]},
                                {"required": ["order"]},
                            ]
                        },
                    },
                },
                {
                    "if": {"properties": {"kind": {"const": "ordinal"}}},
                    "then": {
                        "required": ["basis", "order", "source_role"],
                        "not": {
                            "anyOf": [
                                {"required": ["subtype"]},
                                {"required": ["predicates"]},
                            ]
                        },
                    },
                },
                {
                    "if": {"properties": {"kind": {"const": "part"}}},
                    "then": {
                        "required": ["source_role", "params"],
                        "properties": {
                            "params": {
                                "type": "object",
                                "required": ["region"],
                                "properties": {
                                    "region": {"type": "string", "minLength": 1},
                                    "metadata": {"type": "object"},
                                },
                                "additionalProperties": False,
                            }
                        },
                        "not": {
                            "anyOf": [
                                {"required": ["subtype"]},
                                {"required": ["basis"]},
                                {"required": ["order"]},
                                {"required": ["predicates"]},
                                {"required": ["reference_role"]},
                            ]
                        },
                    },
                },
            ],
        },
        "VariantSchema": {
            "type": "object",
            "required": ["id", "action", "role_bindings"],
            "properties": {
                "id": identifier,
                "action": {"$ref": "#/$defs/Action"},
                "role_bindings": {
                    "type": "object",
                    "minProperties": 1,
                    "propertyNames": identifier,
                    "additionalProperties": role_id,
                },
                "descriptors": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"$ref": "#/$defs/DescriptorConstructor"},
                    },
                },
                "params": {"type": "object"},
            },
            "additionalProperties": False,
        },
        "MinimalPairEdge": {
            "type": "object",
            "required": ["from", "to", "axis"],
            "properties": {
                "from": identifier,
                "to": identifier,
                "axis": {"enum": ["D1a", "D1b", "D1c", "D2a", "D2b", "D4"]},
                "subtype": {"type": "string"},
                "value": {},
            },
            "additionalProperties": False,
        },
        "Archetype": {
            "type": "object",
            "required": ["id", "roles", "constraints", "variant_schemas", "minimal_pair_edges"],
            "properties": {
                "id": identifier,
                "roles": {"type": "array", "items": {"$ref": "#/$defs/Role"}},
                "constraints": {"type": "array", "items": {"$ref": "#/$defs/Constraint"}},
                "constraint_index": {"$ref": "#/$defs/ConstraintIndex"},
                "variant_schemas": {"type": "array", "items": {"$ref": "#/$defs/VariantSchema"}},
                "minimal_pair_edges": {"type": "array", "items": {"$ref": "#/$defs/MinimalPairEdge"}},
                "merge_role": {"type": "boolean"},
                "provenance": {"$ref": "#/$defs/Provenance"},
            },
            "additionalProperties": False,
        },
        "RoleGraph": {
            "type": "object",
            "required": ["id", "roles", "constraints", "variant_schemas", "minimal_pair_edges"],
            "properties": {
                "id": identifier,
                "roles": {"type": "array", "items": {"$ref": "#/$defs/Role"}},
                "constraints": {"type": "array", "items": {"$ref": "#/$defs/Constraint"}},
                "constraint_index": {"$ref": "#/$defs/ConstraintIndex"},
                "variant_schemas": {"type": "array", "items": {"$ref": "#/$defs/VariantSchema"}},
                "minimal_pair_edges": {"type": "array", "items": {"$ref": "#/$defs/MinimalPairEdge"}},
                "provenance": {"$ref": "#/$defs/Provenance"},
            },
            "additionalProperties": False,
        },
        "AbstractSceneFamily": {
            "type": "object",
            "required": ["id", "axis_plan", "archetypes", "randomization"],
            "properties": {
                "id": identifier,
                "axis_plan": {"$ref": "#/$defs/AxisPlan"},
                "archetypes": {"type": "array", "items": identifier},
                "randomization": {"type": "object"},
                "role_graph_id": identifier,
                "provenance": {"$ref": "#/$defs/Provenance"},
            },
            "additionalProperties": False,
        },
        "AssetBinding": {
            "type": "object",
            "required": ["asset", "model_id", "category"],
            "properties": {
                "asset": {"type": "string"},
                "model_id": model_id,
                "category": {"type": "string"},
                "roles": {"type": "array", "items": {"type": "string"}},
            },
            "additionalProperties": False,
        },
        "ProvenFact": {
            "type": "object",
            "required": ["constraint", "fact"],
            "properties": {
                "constraint": identifier,
                "fact": {"type": "string", "minLength": 1},
                "stage": {"type": "string"},
                "status": {"type": "string"},
                "metadata": {"type": "object"},
            },
            "additionalProperties": False,
        },
        "EvidenceRef": {
            "type": "object",
            "required": ["kind"],
            "properties": {
                "kind": {"type": "string", "minLength": 1},
                "path": {"type": "string"},
                "record_id": {"type": "string"},
                "hash": {"type": "string"},
                "asset": {"type": "string"},
                "model_id": model_id,
                "field": {"type": "string"},
                "use": {"type": "string"},
                "relation": {"type": "string"},
                "stage": {"type": "string"},
                "status": {"type": "string"},
                "metadata": {"type": "object"},
            },
            "additionalProperties": False,
        },
        "ProvenanceSource": {
            "type": "object",
            "required": ["kind", "hash"],
            "properties": {
                "kind": {"type": "string", "minLength": 1},
                "hash": {"type": "string", "minLength": 1},
                "path": {"type": "string"},
                "record_id": {"type": "string"},
                "metadata": {"type": "object"},
            },
            "additionalProperties": False,
        },
        "Provenance": {
            "type": "object",
            "required": ["generator", "config_hash", "sources"],
            "properties": {
                "generator": {"type": "string", "minLength": 1},
                "config_hash": {"type": "string", "minLength": 1},
                "sources": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/ProvenanceSource"},
                },
                "metadata": {"type": "object"},
            },
            "additionalProperties": False,
        },
        "AssignmentCandidate": {
            "type": "object",
            "required": [
                "id",
                "abstract_family_id",
                "archetype_ids",
                "roles",
                "resolved_values",
                "proven_facts",
                "evidence_refs",
                "layout_requirements",
                "materialization_params",
            ],
            "properties": {
                "id": identifier,
                "abstract_family_id": identifier,
                "archetype_ids": {"type": "array", "items": identifier},
                "roles": {
                    "type": "object",
                    "minProperties": 1,
                    "propertyNames": role_id,
                    "additionalProperties": {"$ref": "#/$defs/AssetBinding"},
                },
                "resolved_values": {
                    "type": "object",
                    "propertyNames": role_id,
                },
                "proven_facts": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/ProvenFact"},
                },
                "evidence_refs": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/EvidenceRef"},
                },
                "layout_requirements": {"type": "array", "items": {"type": "object"}},
                "materialization_params": {"type": "object"},
                "provenance": {"$ref": "#/$defs/Provenance"},
            },
            "additionalProperties": False,
        },
        "AttributeDescriptor": {
            "type": "object",
            "required": ["subtype", "value"],
            "properties": {"subtype": {"type": "string"}, "value": {"type": "string"}},
            "additionalProperties": False,
        },
        "SpatialDescriptor": {
            "type": "object",
            "required": ["predicates", "reference"],
            "properties": {
                "predicates": {
                    "type": "array",
                    "minItems": 1,
                    "uniqueItems": True,
                    "items": {"type": "string", "minLength": 1},
                },
                "reference": entity_key,
            },
            "additionalProperties": False,
        },
        "OrdinalDescriptor": {
            "type": "object",
            "required": ["basis", "order"],
            "properties": {
                "basis": {"type": "string"},
                "order": {"type": "string"},
                "reference": entity_key,
            },
            "additionalProperties": False,
        },
        "PartDescriptor": {
            "type": "object",
            "required": ["region"],
            "properties": {"region": {"type": "string"}},
            "additionalProperties": False,
        },
        "DescriptorBundle": descriptor_bundle,
        "DescriptorSet": {
            "type": "object",
            "properties": {
                "subject": descriptor_bundle,
                "target": descriptor_bundle,
                "subject2": descriptor_bundle,
                "target2": descriptor_bundle,
            },
            "additionalProperties": False,
        },
        "OrientationAdjustmentParams": {
            "type": "object",
            "required": [
                "orientation_axis",
                "orientation_reference",
                "rotation_delta_deg",
            ],
            "properties": {
                "orientation_axis": {"const": "support_normal"},
                "orientation_reference": {"const": "initial_orientation"},
                "rotation_delta_deg": {
                    "type": "integer",
                    "enum": [-90, -60, -45, -30, 30, 45, 60, 90],
                },
            },
            "additionalProperties": False,
        },
        "ConcreteGoal": {
            "type": "object",
            "required": ["verb", "subject", "descriptors", "params"],
            "properties": {
                "verb": {"type": "string", "minLength": 1},
                "action_family": {"type": "string"},
                "relation": {"type": ["string", "null"]},
                "state": {"type": ["string", "null"]},
                "subject": entity_key,
                "arm": {"type": ["string", "null"]},
                "target": {"oneOf": [entity_key, {"type": "null"}]},
                "subject2": {"oneOf": [entity_key, {"type": "null"}]},
                "target2": {"oneOf": [entity_key, {"type": "null"}]},
                "descriptors": {"$ref": "#/$defs/DescriptorSet"},
                "params": {"type": "object"},
            },
            "additionalProperties": False,
            "allOf": [
                {
                    "if": {
                        "properties": {
                            "verb": {"const": "adjust_orientation"}
                        },
                        "required": ["verb"],
                    },
                    "then": {
                        "required": [
                            "action_family",
                            "relation",
                            "state",
                        ],
                        "properties": {
                            "action_family": {"const": "A2O_state"},
                            "relation": {"type": "null"},
                            "state": {"const": "orientation_adjusted"},
                            "params": {
                                "$ref": "#/$defs/OrientationAdjustmentParams"
                            },
                        },
                    },
                }
            ],
        },
        "ConcreteGoalDocument": {
            "type": "object",
            "required": ["entity_keys", "goal"],
            "properties": {
                "entity_keys": {"type": "array", "items": entity_key},
                "goal": {"$ref": "#/$defs/ConcreteGoal"},
            },
            "additionalProperties": False,
        },
        "GoalVariant": {
            "type": "object",
            "required": [
                "id",
                "schema_id",
                "assignment_id",
                "role_refs",
                "entity_refs",
                "intended_axes",
                "goal",
                "export",
            ],
            "properties": {
                "id": identifier,
                "schema_id": identifier,
                "assignment_id": identifier,
                "role_refs": {
                    "type": "object",
                    "minProperties": 1,
                    "propertyNames": identifier,
                    "additionalProperties": role_id,
                },
                "entity_refs": {
                    "type": "object",
                    "minProperties": 1,
                    "propertyNames": identifier,
                    "additionalProperties": entity_key,
                },
                "intended_axes": {"type": "array", "items": axis_spec},
                "goal": {"$ref": "#/$defs/ConcreteGoal"},
                "export": {
                    "type": "object",
                    "required": ["variant"],
                    "properties": {"variant": {"type": "string"}},
                    "additionalProperties": True,
                },
            },
            "additionalProperties": False,
        },
        "RoleBinding": {
            "type": "object",
            "required": ["entity_key", "asset", "model_id"],
            "properties": {
                "entity_key": entity_key,
                "asset": {"type": "string"},
                "model_id": model_id,
                "category": {"type": "string"},
            },
            "additionalProperties": False,
        },
        "SceneTemplate": {
            "type": "object",
            "required": ["key", "role_ref", "asset", "model_candidates", "roles"],
            "properties": {
                "key": entity_key,
                "role_ref": role_id,
                "asset": {"type": "string"},
                "model_candidates": {"type": "array", "items": model_id},
                "roles": {"type": "array", "items": {"type": "string"}},
            },
            "additionalProperties": False,
        },
        "RectFootprint": {
            "type": "object",
            "required": ["shape", "frame", "center_xy", "half_extents_xy", "source"],
            "properties": {
                "shape": {"const": "rect"},
                "frame": {"const": "unit_local"},
                "center_xy": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 2,
                    "items": {"type": "number"},
                },
                "half_extents_xy": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 2,
                    "items": {"type": "number", "exclusiveMinimum": 0},
                },
                "source": {"type": "string", "minLength": 1},
            },
            "additionalProperties": False,
        },
        "LocalPose": {
            "type": "object",
            "required": ["translation_xyz", "q0_wxyz", "relative_yaw_deg", "placement_mode"],
            "dependentRequired": {
                "initial_state_id": ["dynamic_mass_properties"]
            },
            "properties": {
                "translation_xyz": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 3,
                    "items": {"type": "number"},
                },
                "q0_wxyz": {
                    "type": "array",
                    "minItems": 4,
                    "maxItems": 4,
                    "items": {"type": "number"},
                },
                "relative_yaw_deg": {"type": "number"},
                "placement_mode": {"type": "string", "minLength": 1},
                "yaw_policy": {"$ref": "#/$defs/InitialPoseYawPolicy"},
                "initial_state_id": identifier,
                "initial_state_revision": {"type": "integer", "minimum": 0},
                "initial_state_revision_hash": {
                    "type": "string",
                    "minLength": 1,
                },
                "geometry_hash": {"type": "string", "minLength": 1},
                "canonical_state_hash": {
                    "type": "string",
                    "minLength": 1,
                },
                "assignment_state_revision": {
                    "$ref": "#/$defs/AssignmentStateRevision"
                },
                "scale_xyz": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 3,
                    "items": {"type": "number", "exclusiveMinimum": 0},
                },
                "origin_relative_bounds": {
                    "$ref": "#/$defs/OriginRelativeBounds"
                },
                "support": {"$ref": "#/$defs/InitialPoseSupport"},
                "articulated_joint_state": {
                    "$ref": "#/$defs/ArticulatedInitialJointState"
                },
                "dynamic_mass_properties": {
                    "$ref": "#/$defs/DynamicAssetMassProperties"
                },
            },
            "additionalProperties": False,
        },
        "Anchor": {
            "type": "array",
            "minItems": 3,
            "maxItems": 3,
            "items": {"type": "number"},
        },
        "OperationAnchor": {
            "type": "object",
            "required": [
                "id",
                "kind",
                "point_index",
                "position_xyz_m",
                "source_field",
                "source_frame",
                "operation_eligible",
                "moves_with_active_joint",
            ],
            "properties": {
                "id": {"type": "string", "minLength": 1},
                "kind": {
                    "enum": ["grasp_contact", "functional", "orientation"]
                },
                "point_index": {"type": "integer", "minimum": 0},
                "position_xyz_m": {"$ref": "#/$defs/Anchor"},
                "source_field": {"type": "string", "minLength": 1},
                "source_frame": {
                    "enum": ["actor_local", "link_local"]
                },
                "base_link": {"type": "string", "minLength": 1},
                "operation_eligible": {"type": "boolean"},
                "moves_with_active_joint": {"type": "boolean"},
                "approach_direction_xyz": {"$ref": "#/$defs/Anchor"},
                "approach_class": {
                    "enum": ["top_down", "oblique_down", "horizontal", "upward"]
                },
            },
            "additionalProperties": False,
        },
        "OperationAnchorContract": {
            "type": "object",
            "required": [
                "version",
                "projection_mode",
                "q0_wxyz",
                "scale_xyz",
                "operation_metadata_hash",
                "anchors",
                "contract_hash",
                "source",
            ],
            "properties": {
                "version": {"type": "string", "minLength": 1},
                "projection_mode": {
                    "enum": ["rigid_actor_local", "articulated_link_fk"]
                },
                "q0_wxyz": {
                    "type": "array",
                    "minItems": 4,
                    "maxItems": 4,
                    "items": {"type": "number"},
                },
                "scale_xyz": {"$ref": "#/$defs/Anchor"},
                "operation_metadata_hash": {
                    "type": "string",
                    "minLength": 1,
                },
                "contact_projection_hash": {
                    "type": "string",
                    "minLength": 1,
                },
                "canonical_state_hash": {
                    "type": "string",
                    "minLength": 1,
                },
                "articulated_joint_state_hash": {
                    "type": "string",
                    "minLength": 1,
                },
                "articulated_qpos": {
                    "type": "array",
                    "items": {"type": "number"},
                },
                "active_joint_index": {
                    "type": ["integer", "null"],
                    "minimum": 0,
                },
                "urdf_sha256": {"type": "string", "minLength": 1},
                "anchors": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"$ref": "#/$defs/OperationAnchor"},
                },
                "contract_hash": {"type": "string", "minLength": 1},
                "source": {
                    "type": "object",
                    "required": [
                        "asset",
                        "model_id",
                        "candidate_use",
                        "metadata_path",
                        "operation_metadata_hash",
                    ],
                    "properties": {
                        "asset": {"type": "string", "minLength": 1},
                        "model_id": model_id,
                        "candidate_use": {"type": "string", "minLength": 1},
                        "metadata_path": {"type": "string", "minLength": 1},
                        "metadata_sha256": {"type": "string", "minLength": 1},
                        "operation_metadata_hash": {
                            "type": "string",
                            "minLength": 1,
                        },
                        "urdf_path": {"type": "string", "minLength": 1},
                        "urdf_sha256": {"type": "string", "minLength": 1},
                        "canonical_state_hash": {
                            "type": "string",
                            "minLength": 1,
                        },
                        "assignment_state_revision": {
                            "$ref": "#/$defs/AssignmentStateRevision"
                        },
                        "contact_projection_hash": {
                            "type": "string",
                            "minLength": 1,
                        },
                        "validated_canonical_evidence": {
                            "type": "array",
                            "minItems": 1,
                            "items": {
                                "type": "object",
                                "required": [
                                    "use",
                                    "evidence_id",
                                    "report_path",
                                    "report_hash",
                                ],
                                "properties": {
                                    "use": {"type": "string", "minLength": 1},
                                    "evidence_id": {
                                        "type": "string",
                                        "minLength": 1,
                                    },
                                    "report_hash": {
                                        "type": "string",
                                        "minLength": 1,
                                    },
                                    "report_path": {
                                        "type": "string",
                                        "minLength": 1,
                                    },
                                },
                                "additionalProperties": False,
                            },
                        },
                        "validated_contact_contracts": {
                            "type": "array",
                            "minItems": 1,
                            "items": {
                                "type": "object",
                                "required": [
                                    "use",
                                    "evidence_id",
                                    "report_path",
                                    "report_hash",
                                    "binding",
                                ],
                                "properties": {
                                    "use": {"type": "string", "minLength": 1},
                                    "evidence_id": {
                                        "type": "string",
                                        "minLength": 1,
                                    },
                                    "report_hash": {
                                        "type": "string",
                                        "minLength": 1,
                                    },
                                    "report_path": {
                                        "type": "string",
                                        "minLength": 1,
                                    },
                                    "binding": {"type": "object"},
                                    "validated_contact_ids": {
                                        "type": "array",
                                        "minItems": 1,
                                        "uniqueItems": True,
                                        "items": {
                                            "type": "integer",
                                            "minimum": 0,
                                        },
                                    },
                                    "validated_contact_ids_by_arm": {
                                        "type": "object",
                                        "properties": {
                                            arm: {
                                                "type": "array",
                                                "minItems": 1,
                                                "uniqueItems": True,
                                                "items": {
                                                    "type": "integer",
                                                    "minimum": 0,
                                                },
                                            }
                                            for arm in ("left", "right")
                                        },
                                        "additionalProperties": False,
                                    },
                                },
                                "additionalProperties": False,
                            },
                        },
                        "validated_articulation_executions": {
                            "type": "array",
                            "minItems": 1,
                            "items": {
                                "type": "object",
                                "required": [
                                    "use",
                                    "evidence_id",
                                    "report_path",
                                    "report_hash",
                                    "binding",
                                ],
                                "properties": {
                                    "use": {"type": "string", "minLength": 1},
                                    "evidence_id": {
                                        "type": "string",
                                        "minLength": 1,
                                    },
                                    "report_hash": {
                                        "type": "string",
                                        "minLength": 1,
                                    },
                                    "report_path": {
                                        "type": "string",
                                        "minLength": 1,
                                    },
                                    "binding": {"type": "object"},
                                },
                                "additionalProperties": False,
                            },
                        },
                        "validated_touch_executions": {
                            "type": "array",
                            "minItems": 1,
                            "items": {
                                "type": "object",
                                "required": [
                                    "use",
                                    "evidence_id",
                                    "report_path",
                                    "report_hash",
                                    "binding",
                                ],
                                "properties": {
                                    "use": {"type": "string", "minLength": 1},
                                    "evidence_id": {
                                        "type": "string",
                                        "minLength": 1,
                                    },
                                    "report_hash": {
                                        "type": "string",
                                        "minLength": 1,
                                    },
                                    "report_path": {
                                        "type": "string",
                                        "minLength": 1,
                                    },
                                    "binding": {"type": "object"},
                                },
                                "additionalProperties": False,
                            },
                        },
                    },
                    "additionalProperties": False,
                },
            },
            "additionalProperties": False,
        },
        "LocalZBounds": {
            "type": "array",
            "minItems": 2,
            "maxItems": 2,
            "items": {"type": "number"},
        },
        "GuaranteedFact": {
            "type": "object",
            "required": ["predicate", "subject", "relation", "source_constraint_ids"],
            "properties": {
                "predicate": {"enum": ["spatial", "terminal_region"]},
                "subject": entity_key,
                "relation": {"type": "string", "minLength": 1},
                "reference": entity_key,
                "receivers": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 2,
                    "items": entity_key,
                },
                "region_id": identifier,
                "source_constraint_ids": {"type": "array", "items": identifier},
            },
            "additionalProperties": False,
            "allOf": [
                {
                    "if": {"properties": {"predicate": {"const": "spatial"}}},
                    "then": {"required": ["reference"]},
                },
                {
                    "if": {"properties": {"predicate": {"const": "terminal_region"}}},
                    "then": {"required": ["receivers", "region_id"]},
                },
            ],
        },
        "RigidM5TerminalPlacementRecipe": {
            "type": "object",
            "required": [
                "clearance_m",
                "yaw_convention",
                "settle_policy",
            ],
            "properties": {
                "clearance_m": {"type": "number", "minimum": 0},
                "yaw_convention": {
                    "const": "subject_yaw_minus_receiver_yaw"
                },
                "settle_policy": {
                    "const": (
                        "m5_terminal_settle_v11_proportional_asset_"
                        "mass_inertia"
                    )
                },
            },
            "additionalProperties": False,
        },
        "ArticulatedM5TerminalPlacementRecipe": {
            "type": "object",
            "required": [
                "recipe_id",
                "constructor",
                "clearance_m",
                "yaw_convention",
                "settle_policy",
                "receiver_state_profile_id",
                "receiver_state_profile_hash",
                "cavity_frame_link",
                "support_plane_z",
                "recipe_hash",
            ],
            "properties": {
                "recipe_id": {"type": "string", "minLength": 1},
                "constructor": {
                    "const": "articulated_place_in_cavity_link"
                },
                "clearance_m": {"type": "number", "minimum": 0},
                "yaw_convention": {
                    "const": "subject_yaw_minus_receiver_yaw"
                },
                "settle_policy": {
                    "const": (
                        "m5_ar_v3_proportional_asset_mass_inertia"
                    )
                },
                "receiver_state_profile_id": {
                    "type": "string",
                    "minLength": 1,
                },
                "receiver_state_profile_hash": {
                    "type": "string",
                    "minLength": 1,
                },
                "cavity_frame_link": {
                    "type": "string",
                    "minLength": 1,
                },
                "support_plane_z": {"type": "number"},
                "recipe_hash": {"type": "string", "minLength": 1},
            },
            "additionalProperties": False,
        },
        "M5TerminalPlacementRecipe": {
            "oneOf": [
                {"$ref": "#/$defs/RigidM5TerminalPlacementRecipe"},
                {
                    "$ref": (
                        "#/$defs/ArticulatedM5TerminalPlacementRecipe"
                    )
                },
            ]
        },
        "NonPairTerminalPlacementRecipe": {
            "type": "object",
            "required": ["clearance_m", "policy"],
            "properties": {
                "clearance_m": {"type": "number", "minimum": 0},
                "policy": {"const": "m11_non_pair_terminal_clearance_v1"},
            },
            "additionalProperties": False,
        },
        "TerminalPlacementRecipe": {
            "oneOf": [
                {"$ref": "#/$defs/M5TerminalPlacementRecipe"},
                {"$ref": "#/$defs/NonPairTerminalPlacementRecipe"},
            ]
        },
        "ReservedRegion": {
            "type": "object",
            "required": [
                "id",
                "relation",
                "subject_key",
                "receiver_keys",
                "footprint",
                "support_clearance_m",
                "terminal_clearance_m",
                "pair_evidence_ids",
                "relative_yaw_deg",
                "usable_region",
            ],
            "properties": {
                "id": identifier,
                "relation": {"type": "string", "minLength": 1},
                "subject_key": entity_key,
                "receiver_keys": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 2,
                    "items": entity_key,
                },
                "footprint": {"$ref": "#/$defs/RectFootprint"},
                "support_clearance_m": {"type": "number", "minimum": 0},
                "terminal_clearance_m": {"type": "number", "minimum": 0},
                "placement_recipe": {
                    "$ref": "#/$defs/TerminalPlacementRecipe"
                },
                "placement_recipe_source": {
                    "type": "string",
                    "minLength": 1,
                },
                "pair_evidence_ids": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                },
                "relative_yaw_deg": {"type": "number"},
                "relation_distance_band_m": {
                    "oneOf": [
                        {"type": "null"},
                        {
                            "type": "array",
                            "minItems": 2,
                            "maxItems": 2,
                            "items": {"type": "number", "minimum": 0},
                        },
                    ]
                },
                "relation_surface_gap_band_m": {
                    "oneOf": [
                        {"type": "null"},
                        {
                            "type": "array",
                            "minItems": 2,
                            "maxItems": 2,
                            "items": {"type": "number", "minimum": 0},
                        },
                    ]
                },
                "reserved_surface_gap_m": {
                    "oneOf": [
                        {"type": "null"},
                        {"type": "number", "minimum": 0},
                    ]
                },
                "reserved_center_distance_m": {
                    "type": "number",
                    "minimum": 0,
                },
                "usable_region": {
                    "oneOf": [
                        {"type": "null"},
                        {
                            "type": "object",
                            "required": [
                                "x",
                                "y",
                                "factor",
                                "edge_clearance_m",
                                "solver_policy_hash",
                            ],
                            "properties": {
                                "x": {"type": "number", "exclusiveMinimum": 0},
                                "y": {"type": "number", "exclusiveMinimum": 0},
                                "factor": {"type": ["number", "null"]},
                                "edge_clearance_m": {"type": ["number", "null"]},
                                "solver_policy_hash": {
                                    "type": ["string", "null"]
                                },
                            },
                            "additionalProperties": False,
                        },
                    ]
                },
            },
            "allOf": [
                {
                    "if": {
                        "properties": {
                            "relation": {"enum": ["in", "on", "stack"]}
                        }
                    },
                    "then": {
                        "required": [
                            "placement_recipe",
                            "placement_recipe_source",
                        ],
                        "properties": {
                            "placement_recipe": {
                                "$ref": "#/$defs/M5TerminalPlacementRecipe"
                            },
                            "placement_recipe_source": {
                                "enum": [
                                    "frozen_m5_projection",
                                    (
                                        "deterministic_m5_report_v3_"
                                        "recipe_rebind_v2"
                                    ),
                                ]
                            },
                        },
                    },
                    "else": {
                        "properties": {
                            "placement_recipe": {
                                "$ref": (
                                    "#/$defs/NonPairTerminalPlacementRecipe"
                                )
                            },
                            "placement_recipe_source": {
                                "const": "relation_default"
                            },
                        }
                    },
                }
            ],
            "additionalProperties": False,
        },
        "Unit": {
            "type": "object",
            "required": ["id", "type", "roles", "entity_keys"],
            "properties": {
                "id": identifier,
                "type": {"enum": ["ObjectUnit", "LandmarkRelationUnit", "ReceiverReservationUnit"]},
                "roles": {"type": "array", "items": role_id},
                "entity_keys": {"type": "array", "items": entity_key},
                "params": {"type": "object"},
            },
            "additionalProperties": False,
            "allOf": [
                {
                    "if": {"properties": {"type": {"const": "ObjectUnit"}}},
                    "then": {
                        "required": ["params"],
                        "properties": {
                            "roles": {"minItems": 1, "maxItems": 1},
                            "entity_keys": {"minItems": 1, "maxItems": 1},
                            "params": {
                                "type": "object",
                                "required": [
                                    "object_key",
                                    "footprint",
                                    "anchors",
                                    "local_pose",
                                    "local_z_bounds",
                                    "geometry_evidence",
                                ],
                                "properties": {
                                    "object_key": entity_key,
                                    "footprint": {"$ref": "#/$defs/RectFootprint"},
                                    "anchors": {
                                        "type": "object",
                                        "additionalProperties": {"$ref": "#/$defs/Anchor"},
                                    },
                                    "local_pose": {"$ref": "#/$defs/LocalPose"},
                                    "local_z_bounds": {
                                        "$ref": "#/$defs/LocalZBounds"
                                    },
                                    "geometry_evidence": {"type": "object"},
                                    **operation_properties,
                                    "operation_anchors": {
                                        "type": "object",
                                        "additionalProperties": {
                                            "$ref": "#/$defs/OperationAnchorContract"
                                        },
                                    },
                                },
                                "additionalProperties": False,
                            },
                        }
                    },
                },
                {
                    "if": {"properties": {"type": {"const": "LandmarkRelationUnit"}}},
                    "then": {
                        "required": ["params"],
                        "properties": {
                            "roles": {"minItems": 2},
                            "entity_keys": {"minItems": 2},
                            "params": {
                                "type": "object",
                                "required": [
                                    "landmark_key",
                                    "internal_layout",
                                    "member_footprints",
                                    "member_z_bounds",
                                    "footprint",
                                    "anchors",
                                    "guaranteed_facts",
                                    "geometry_evidence",
                                    "pair_evidence",
                                ],
                                "properties": {
                                    "landmark_key": entity_key,
                                    "internal_layout": {
                                        "type": "object",
                                        "additionalProperties": {"$ref": "#/$defs/LocalPose"},
                                    },
                                    "member_footprints": {
                                        "type": "object",
                                        "additionalProperties": {"$ref": "#/$defs/RectFootprint"},
                                    },
                                    "member_z_bounds": {
                                        "type": "object",
                                        "additionalProperties": {
                                            "$ref": "#/$defs/LocalZBounds"
                                        },
                                    },
                                    "footprint": {"$ref": "#/$defs/RectFootprint"},
                                    "anchors": {
                                        "type": "object",
                                        "additionalProperties": {"$ref": "#/$defs/Anchor"},
                                    },
                                    "guaranteed_facts": {
                                        "type": "array",
                                        "items": {"$ref": "#/$defs/GuaranteedFact"},
                                    },
                                    "geometry_evidence": {"type": "object"},
                                    "pair_evidence": {
                                        "type": "array",
                                        "items": {"type": "object"},
                                    },
                                    **operation_properties,
                                    "operation_anchors": {
                                        "type": "object",
                                        "additionalProperties": {
                                            "$ref": "#/$defs/OperationAnchorContract"
                                        },
                                    },
                                },
                                "additionalProperties": False,
                            },
                        },
                    },
                },
                {
                    "if": {"properties": {"type": {"const": "ReceiverReservationUnit"}}},
                    "then": {
                        "required": ["params"],
                        "properties": {
                            "roles": {"minItems": 1},
                            "entity_keys": {"minItems": 1},
                            "params": {
                                "type": "object",
                                "required": [
                                    "receiver_keys",
                                    "subject_keys",
                                    "relations",
                                    "receiver_layout",
                                    "receiver_footprints",
                                    "receiver_z_bounds",
                                    "reserved_regions",
                                    "footprint",
                                    "anchors",
                                    "guaranteed_facts",
                                    "geometry_evidence",
                                    "subject_geometry_evidence",
                                    "subject_z_bounds",
                                    "pair_evidence",
                                ],
                                "properties": {
                                    "receiver_keys": {
                                        "type": "array",
                                        "minItems": 1,
                                        "maxItems": 2,
                                        "items": entity_key,
                                    },
                                    "subject_keys": {
                                        "type": "array",
                                        "minItems": 1,
                                        "items": entity_key,
                                    },
                                    "relations": {
                                        "type": "array",
                                        "minItems": 1,
                                        "items": {"type": "string", "minLength": 1},
                                    },
                                    "receiver_layout": {
                                        "type": "object",
                                        "additionalProperties": {"$ref": "#/$defs/LocalPose"},
                                    },
                                    "receiver_footprints": {
                                        "type": "object",
                                        "additionalProperties": {"$ref": "#/$defs/RectFootprint"},
                                    },
                                    "receiver_z_bounds": {
                                        "type": "object",
                                        "additionalProperties": {
                                            "$ref": "#/$defs/LocalZBounds"
                                        },
                                    },
                                    "reserved_regions": {
                                        "type": "array",
                                        "minItems": 1,
                                        "items": {"$ref": "#/$defs/ReservedRegion"},
                                    },
                                    "footprint": {"$ref": "#/$defs/RectFootprint"},
                                    "anchors": {
                                        "type": "object",
                                        "additionalProperties": {"$ref": "#/$defs/Anchor"},
                                    },
                                    "guaranteed_facts": {
                                        "type": "array",
                                        "items": {"$ref": "#/$defs/GuaranteedFact"},
                                    },
                                    "geometry_evidence": {"type": "object"},
                                    "subject_geometry_evidence": {"type": "object"},
                                    "subject_z_bounds": {
                                        "type": "object",
                                        "additionalProperties": {
                                            "$ref": "#/$defs/LocalZBounds"
                                        },
                                    },
                                    "pair_evidence": {
                                        "type": "array",
                                        "items": {"type": "object"},
                                    },
                                    **operation_properties,
                                    "operation_anchors": {
                                        "type": "object",
                                        "additionalProperties": {
                                            "$ref": "#/$defs/OperationAnchorContract"
                                        },
                                    },
                                },
                                "additionalProperties": False,
                            },
                        },
                    },
                },
            ],
        },
        "UnitInput": {
            "type": "object",
            "required": [
                "id",
                "type",
                "roles",
                "entity_keys",
                "source_constraints",
                "geometry_refs",
                "pair_preconditions",
                "params",
            ],
            "properties": {
                "id": identifier,
                "type": {
                    "enum": [
                        "ObjectUnit",
                        "LandmarkRelationUnit",
                        "ReceiverReservationUnit",
                    ]
                },
                "roles": {
                    "type": "array",
                    "minItems": 1,
                    "items": role_id,
                },
                "entity_keys": {
                    "type": "array",
                    "minItems": 1,
                    "items": entity_key,
                },
                "source_constraints": {
                    "type": "array",
                    "items": identifier,
                },
                "geometry_refs": {
                    "type": "object",
                    "propertyNames": role_id,
                    "additionalProperties": {"type": "string", "minLength": 1},
                },
                "initial_state_refs": {
                    "type": "object",
                    "propertyNames": role_id,
                    "additionalProperties": {
                        "$ref": "#/$defs/CanonicalInitialState"
                    },
                },
                "pair_preconditions": {"type": "object"},
                "params": {"type": "object"},
            },
            "additionalProperties": False,
        },
        "RealizedSceneFamily": {
            "type": "object",
            "required": [
                "id",
                "abstract_family_id",
                "assignment_id",
                "archetype_ids",
                "role_bindings",
                "templates",
                "goal_variants",
                "unit_inputs",
                "units",
                "layout_constraints",
                "minimal_pair_edges",
                "evidence",
                "export",
            ],
            "properties": {
                "id": identifier,
                "abstract_family_id": identifier,
                "assignment_id": identifier,
                "archetype_ids": {"type": "array", "items": identifier},
                "role_bindings": {
                    "type": "object",
                    "minProperties": 1,
                    "propertyNames": role_id,
                    "additionalProperties": {"$ref": "#/$defs/RoleBinding"},
                },
                "templates": {"type": "array", "items": {"$ref": "#/$defs/SceneTemplate"}},
                "goal_variants": {"type": "array", "items": {"$ref": "#/$defs/GoalVariant"}},
                "unit_inputs": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"$ref": "#/$defs/UnitInput"},
                },
                "units": {"type": "array", "items": {"$ref": "#/$defs/Unit"}},
                "layout_constraints": {"type": "array", "items": {"$ref": "#/$defs/Constraint"}},
                "minimal_pair_edges": {"type": "array", "items": {"$ref": "#/$defs/MinimalPairEdge"}},
                "evidence": {"type": "object"},
                "export": {"type": "object"},
                "provenance": {"$ref": "#/$defs/Provenance"},
            },
            "additionalProperties": False,
        },
        "ScenePose": {
            "type": "object",
            "required": ["position", "quat_wxyz"],
            "properties": {
                "position": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 3,
                    "items": {"type": "number"},
                },
                "quat_wxyz": {
                    "type": "array",
                    "minItems": 4,
                    "maxItems": 4,
                    "items": {"type": "number"},
                },
            },
            "additionalProperties": False,
        },
        "SceneUnitPose": {
            "type": "object",
            "required": ["unit_id", "position", "yaw_deg"],
            "properties": {
                "unit_id": identifier,
                "position": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 3,
                    "items": {"type": "number"},
                },
                "yaw_deg": {"type": "number"},
            },
            "additionalProperties": False,
        },
        "SceneActor": {
            "type": "object",
            "required": ["embodiment", "base_pose"],
            "properties": {
                "embodiment": {"type": "string", "minLength": 1},
                "base_pose": {"$ref": "#/$defs/ScenePose"},
            },
            "additionalProperties": False,
        },
        "SceneCamera": {
            "type": "object",
            "required": [
                "name",
                "type",
                "width",
                "height",
                "fovy_deg",
                "near",
                "far",
                "pose",
                "table_visible_polygon_xy",
            ],
            "properties": {
                "name": {"type": "string", "minLength": 1},
                "type": {"type": "string", "minLength": 1},
                "width": {"type": "integer", "minimum": 1},
                "height": {"type": "integer", "minimum": 1},
                "fovy_deg": {"type": "number", "exclusiveMinimum": 0},
                "near": {"type": "number", "exclusiveMinimum": 0},
                "far": {"type": "number", "exclusiveMinimum": 0},
                "pose": {"$ref": "#/$defs/ScenePose"},
                "table_visible_polygon_xy": {
                    "type": "array",
                    "minItems": 3,
                    "items": {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": 2,
                        "items": {"type": "number"},
                    },
                },
            },
            "additionalProperties": False,
        },
        "ReachRecord": {
            "type": "object",
            "required": ["reachable_arms", "zone", "evidence"],
            "properties": {
                "reachable_arms": {
                    "type": "array",
                    "uniqueItems": True,
                    "items": {"enum": ["left", "right"]},
                },
                "zone": {
                    "enum": ["left_only", "both", "right_only", "none"]
                },
                "evidence": {"type": "string", "minLength": 1},
                "articulation_capability": {
                    "type": "object",
                    "required": [
                        "capability_version",
                        "uses",
                        "evidence",
                    ],
                    "properties": {
                        "capability_version": {
                            "const": "articulation_arm_contact_candidate_v2_physical_acquisition"
                        },
                        "uses": {
                            "type": "object",
                            "minProperties": 1,
                            "additionalProperties": {
                                "type": "object",
                                "required": [
                                    "capable_arms",
                                    "arm_capabilities",
                                ],
                                "properties": {
                                    "capable_arms": {
                                        "type": "array",
                                        "minItems": 1,
                                        "maxItems": 2,
                                        "uniqueItems": True,
                                        "items": {
                                            "enum": ["left", "right"]
                                        },
                                    },
                                    "arm_capabilities": {
                                        "type": "array",
                                        "minItems": 1,
                                        "items": {"type": "object"},
                                    },
                                },
                                "additionalProperties": False,
                            },
                        },
                        "evidence": {
                            "const": "m11_exact_m4_articulation_capability_v1"
                        },
                    },
                    "additionalProperties": False,
                },
                "pick_planning_requirements": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "required": [
                            "goal_variant_id",
                            "selected_arm",
                            "required_arms",
                            "requirement_mode",
                            "route",
                            "selection",
                            "scope",
                            "planning_witness_required",
                            "final_feasibility_proven",
                        ],
                        "properties": {
                            "goal_variant_id": identifier,
                            "selected_arm": {"enum": ["left", "right"]},
                            "required_arms": {
                                "type": "array",
                                "minItems": 1,
                                "maxItems": 1,
                                "uniqueItems": True,
                                "items": {"enum": ["left", "right"]},
                            },
                            "requirement_mode": {"const": "all"},
                            "route": {"const": "direct"},
                            "selection": {
                                "enum": [
                                    "semantic_arm",
                                    "concrete_scene_reach",
                                    "direct_terminal_reach",
                                    "exact_planning_fallback",
                                ]
                            },
                            "scope": {
                                "const": "exact_scene_planning_required"
                            },
                            "planning_witness_required": {"const": True},
                            "final_feasibility_proven": {"const": False},
                        },
                        "additionalProperties": False,
                    },
                },
            },
            "additionalProperties": False,
        },
        "TerminalReachRecord": {
            "type": "object",
            "required": [
                "subject",
                "relation",
                "required_arm",
                "position_xy",
                "reachable_arms",
                "zone",
                "proxy_passed",
                "evidence",
            ],
            "properties": {
                "subject": entity_key,
                "relation": {"type": "string", "minLength": 1},
                "required_arm": {
                    "enum": ["left", "right", "both", "any"]
                },
                "position_xy": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 2,
                    "items": {"type": "number"},
                },
                "reachable_arms": {
                    "type": "array",
                    "uniqueItems": True,
                    "items": {"enum": ["left", "right"]},
                },
                "zone": {
                    "enum": ["left_only", "both", "right_only", "none"]
                },
                "proxy_passed": {"const": True},
                "evidence": {"type": "string", "minLength": 1},
            },
            "additionalProperties": False,
        },
        "InitialObservation": {
            "type": "object",
            "required": [
                "observation_id",
                "scene_state_hash",
                "variant_invariant",
                "artifact_path",
            ],
            "properties": {
                "observation_id": identifier,
                "scene_state_hash": {"type": "string", "minLength": 1},
                "variant_invariant": {"const": True},
                "artifact_path": {"type": ["string", "null"]},
            },
            "additionalProperties": False,
        },
        "SceneEntity": {
            "type": "object",
            "required": ["role_ref", "asset", "model_id", "unit_id", "pose"],
            "properties": {
                "role_ref": role_id,
                "asset": {"type": "string"},
                "model_id": model_id,
                "unit_id": identifier,
                "pose": {"$ref": "#/$defs/ScenePose"},
                "initial_pose": {
                    "$ref": "#/$defs/CanonicalInitialStatePlacement"
                },
            },
            "additionalProperties": False,
        },
        "SceneSpec": {
            "type": "object",
            "required": [
                "id",
                "realized_family_id",
                "layout_seed",
                "entities",
                "unit_poses",
                "actor",
                "camera",
                "reach",
                "terminal_reach",
                "initial_observation",
                "goal_variant_ids",
                "provenance",
            ],
            "properties": {
                "id": identifier,
                "realized_family_id": identifier,
                "layout_seed": {"type": "integer"},
                "entities": {
                    "type": "object",
                    "minProperties": 1,
                    "propertyNames": entity_key,
                    "additionalProperties": {"$ref": "#/$defs/SceneEntity"},
                },
                "unit_poses": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"$ref": "#/$defs/SceneUnitPose"},
                },
                "actor": {"$ref": "#/$defs/SceneActor"},
                "camera": {"$ref": "#/$defs/SceneCamera"},
                "reach": {
                    "type": "object",
                    "minProperties": 1,
                    "propertyNames": entity_key,
                    "additionalProperties": {"$ref": "#/$defs/ReachRecord"},
                },
                "terminal_reach": {
                    "type": "object",
                    "additionalProperties": {
                        "$ref": "#/$defs/TerminalReachRecord"
                    },
                },
                "initial_observation": {
                    "$ref": "#/$defs/InitialObservation"
                },
                "goal_variant_ids": {"type": "array", "items": identifier},
                "provenance": {"$ref": "#/$defs/Provenance"},
            },
            "additionalProperties": False,
        },
        "InstructionCandidate": {
            "type": "object",
            "required": [
                "template_id",
                "style_id",
                "text",
                "semantic_frame",
                "goal_signature",
            ],
            "properties": {
                "template_id": identifier,
                "style_id": identifier,
                "text": {"type": "string", "minLength": 1},
                "semantic_frame": {"type": "object"},
                "goal_signature": {"type": "string", "minLength": 1},
            },
            "additionalProperties": False,
        },
        "RenderedInstruction": {
            "type": "object",
            "required": [
                "language",
                "renderer_version",
                "candidates",
                "selected",
                "selection",
            ],
            "properties": {
                "language": {"const": "en"},
                "renderer_version": {"type": "integer", "minimum": 1},
                "candidates": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"$ref": "#/$defs/InstructionCandidate"},
                },
                "selected": {"$ref": "#/$defs/InstructionCandidate"},
                "selection": {
                    "type": "object",
                    "required": [
                        "policy",
                        "instruction_seed",
                        "scene_spec_id",
                        "shared_style_ids",
                        "selected_style_id",
                        "selection_digest",
                        "candidate_set_hash",
                    ],
                    "properties": {
                        "policy": {"type": "string", "minLength": 1},
                        "instruction_seed": {"type": "integer", "minimum": 0},
                        "scene_spec_id": identifier,
                        "shared_style_ids": {
                            "type": "array",
                            "minItems": 1,
                            "uniqueItems": True,
                            "items": identifier,
                        },
                        "selected_style_id": identifier,
                        "selection_digest": {"type": "string", "minLength": 1},
                        "candidate_set_hash": {"type": "string", "minLength": 1},
                    },
                    "additionalProperties": False,
                },
            },
            "additionalProperties": False,
        },
        "GoalProvenance": {
            "type": "object",
            "required": [
                "abstract_family_id",
                "assignment_id",
                "schema_id",
                "role_refs",
                "entity_refs",
                "scene_entity_keys",
                "intended_axes",
                "goal_hash",
                "source_scene_hash",
                "source_realization_hash",
            ],
            "properties": {
                "abstract_family_id": identifier,
                "assignment_id": identifier,
                "schema_id": identifier,
                "role_refs": {"type": "object"},
                "entity_refs": {"type": "object"},
                "scene_entity_keys": {
                    "type": "array",
                    "minItems": 1,
                    "uniqueItems": True,
                    "items": entity_key,
                },
                "intended_axes": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/AxisSpec"},
                },
                "goal_hash": {"type": "string", "minLength": 1},
                "source_scene_hash": {"type": "string", "minLength": 1},
                "source_realization_hash": {"type": "string", "minLength": 1},
            },
            "additionalProperties": False,
        },
        "CheckerConfig": {
            "type": "object",
            "required": [
                "version",
                "family",
                "goal_signature",
                "entity_refs",
                "success_contract",
                "predicate",
                "scene_integrity",
            ],
            "properties": {
                "version": {"const": 5},
                "family": {
                    "enum": ["A2O_state", "A2O_event", "O2O_relation"]
                },
                "goal_signature": {"type": "string", "minLength": 1},
                "entity_refs": {"type": "object"},
                "success_contract": {"$ref": "#/$defs/SuccessContractIdentity"},
                "relation_contract": {"$ref": "#/$defs/RelationContractIdentity"},
                "predicate": {
                    "type": "object",
                    "required": ["name", "arguments"],
                    "properties": {
                        "name": {
                            "enum": [
                                "pick_success",
                                "articulation_state_success",
                                "click_press_success",
                                "orientation_success",
                                "placement_completed"
                            ]
                        },
                        "arguments": {"type": "object"}
                    },
                    "additionalProperties": False,
                },
                "scene_integrity": {
                    "type": "object",
                    "required": [
                        "policy",
                        "exempt_entities",
                        "protected_entities",
                        "protected_articulated_entities",
                    ],
                    "properties": {
                        "policy": {
                            "type": "object",
                            "required": [
                                "policy_version",
                                "participant_source",
                                "translation_limit_m",
                                "rotation_limit_deg",
                                "articulation_normalized_limit",
                                "limits_inclusive",
                                "policy_hash",
                            ],
                            "properties": {
                                "policy_version": {
                                    "const": (
                                        "m13_m15_scene_integrity_v1_"
                                        "explicit_goal_entities_exempt"
                                    )
                                },
                                "participant_source": {
                                    "const": (
                                        "concrete_goal_recursive_"
                                        "scene_entity_references_v1"
                                    )
                                },
                                "translation_limit_m": {
                                    "type": "number",
                                    "minimum": 0.0,
                                },
                                "rotation_limit_deg": {
                                    "type": "number",
                                    "minimum": 0.0,
                                },
                                "articulation_normalized_limit": {
                                    "type": "number",
                                    "minimum": 0.0,
                                },
                                "limits_inclusive": {"const": True},
                                "policy_hash": {
                                    "type": "string",
                                    "minLength": 1,
                                },
                            },
                            "additionalProperties": False,
                        },
                        "exempt_entities": {
                            "type": "array",
                            "uniqueItems": True,
                            "items": entity_key,
                        },
                        "protected_entities": {
                            "type": "array",
                            "uniqueItems": True,
                            "items": entity_key,
                        },
                        "protected_articulated_entities": {
                            "type": "array",
                            "uniqueItems": True,
                            "items": entity_key,
                        },
                    },
                    "additionalProperties": False,
                },
            },
            "additionalProperties": False,
        },
        "SuccessContractIdentity": {
            "type": "object",
            "required": [
                "contract_id",
                "success_contract_version",
                "success_contract_hash"
            ],
            "properties": {
                "contract_id": {"type": "string", "minLength": 1},
                "success_contract_version": {"type": "string", "minLength": 1},
                "success_contract_hash": {"type": "string", "minLength": 1}
            },
            "additionalProperties": False,
        },
        "RelationContractIdentity": {
            "type": "object",
            "required": ["relation_contract_version", "relation_contract_hash"],
            "properties": {
                "relation_contract_version": {"type": "string", "minLength": 1},
                "relation_contract_hash": {"type": "string", "minLength": 1}
            },
            "additionalProperties": False,
        },
        "SkillConfig": {
            "type": "object",
            "required": [
                "version",
                "operation_anchor_policy_version",
                "goal_signature",
                "semantic_arm",
                "execution_arm",
                "route",
                "steps",
            ],
            "properties": {
                "version": {"type": "integer", "minimum": 1},
                "operation_anchor_policy_version": {
                    "type": "string",
                    "minLength": 1,
                },
                "goal_signature": {"type": "string", "minLength": 1},
                "semantic_arm": {"type": ["string", "null"]},
                "execution_arm": {"enum": ["left", "right"]},
                "route": {"const": "direct"},
                "steps": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "object"},
                },
            },
            "additionalProperties": False,
        },
        "LegacyTaskExport": {
            "type": "object",
            "required": [
                "version",
                "task_name",
                "task_variant",
                "setup_payload",
            ],
            "properties": {
                "version": {"type": "integer", "minimum": 1},
                "task_name": {"const": "if_scene"},
                "task_variant": {"type": "string", "minLength": 1},
                "setup_payload": {"type": "object"},
            },
            "additionalProperties": False,
        },
        "TaskRecord": {
            "type": "object",
            "required": [
                "id",
                "scene_spec_id",
                "realized_family_id",
                "goal_variant_id",
                "goal",
                "goal_provenance",
                "instruction",
                "success_contract",
                "checker_config",
                "skill_config",
                "legacy_export",
                "provenance",
            ],
            "properties": {
                "id": identifier,
                "scene_spec_id": identifier,
                "realized_family_id": identifier,
                "goal_variant_id": identifier,
                "goal": {"$ref": "#/$defs/ConcreteGoal"},
                "goal_provenance": {"$ref": "#/$defs/GoalProvenance"},
                "instruction": {"$ref": "#/$defs/RenderedInstruction"},
                "success_contract": {"$ref": "#/$defs/SuccessContractIdentity"},
                "relation_contract": {"$ref": "#/$defs/RelationContractIdentity"},
                "checker_config": {"$ref": "#/$defs/CheckerConfig"},
                "skill_config": {"$ref": "#/$defs/SkillConfig"},
                "legacy_export": {"$ref": "#/$defs/LegacyTaskExport"},
                "provenance": {"$ref": "#/$defs/Provenance"},
            },
            "additionalProperties": False,
        },
        "MinimalPairEndpoint": {
            "type": "object",
            "required": [
                "goal_variant_id",
                "task_record_id",
                "task_artifact_hash",
                "goal_hash",
                "instruction_hash",
                "checker_hash",
                "style_id",
            ],
            "properties": {
                "goal_variant_id": identifier,
                "task_record_id": identifier,
                "task_artifact_hash": {"type": "string", "minLength": 1},
                "goal_hash": {"type": "string", "minLength": 1},
                "instruction_hash": {"type": "string", "minLength": 1},
                "checker_hash": {"type": "string", "minLength": 1},
                "style_id": identifier,
            },
            "additionalProperties": False,
        },
        "ValidatedMinimalPair": {
            "type": "object",
            "required": [
                "id",
                "scene_spec_id",
                "realized_family_id",
                "layout_seed",
                "assignment_id",
                "observation_id",
                "scene_state_hash",
                "from_goal_variant_id",
                "to_goal_variant_id",
                "intended_axis",
                "intended_subtype",
                "intended_dcode",
                "actual_axes",
                "actual_dcodes",
                "unmapped_semantic_deltas",
                "status",
                "source_scene_hash",
                "source_realization_hash",
                "from_endpoint",
                "to_endpoint",
            ],
            "properties": {
                "id": identifier,
                "scene_spec_id": identifier,
                "realized_family_id": identifier,
                "layout_seed": {"type": "integer", "minimum": 0},
                "assignment_id": identifier,
                "observation_id": identifier,
                "scene_state_hash": {"type": "string", "minLength": 1},
                "from_goal_variant_id": identifier,
                "to_goal_variant_id": identifier,
                "intended_axis": {
                    "enum": ["D1a", "D1b", "D1c", "D2a", "D2b", "D4"]
                },
                "intended_subtype": {"type": "string", "minLength": 1},
                "intended_dcode": {"type": "string", "minLength": 1},
                "actual_axes": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "enum": ["D1a", "D1b", "D1c", "D2a", "D2b", "D4"]
                    },
                },
                "actual_dcodes": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "string", "minLength": 1},
                },
                "unmapped_semantic_deltas": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                },
                "status": {"const": "valid"},
                "source_scene_hash": {"type": "string", "minLength": 1},
                "source_realization_hash": {"type": "string", "minLength": 1},
                "from_endpoint": {"$ref": "#/$defs/MinimalPairEndpoint"},
                "to_endpoint": {"$ref": "#/$defs/MinimalPairEndpoint"},
            },
            "additionalProperties": False,
        },
        "AxisContrastMember": {
            "type": "object",
            "required": [
                "matrix_index",
                "goal_variant_id",
                "axis_value",
                "intent_goal_hash",
            ],
            "properties": {
                "matrix_index": {"type": "integer", "minimum": 0},
                "goal_variant_id": identifier,
                "axis_value": {},
                "intent_goal_hash": {"type": "string", "minLength": 1},
                "task_record_id": identifier,
                "task_artifact_hash": {"type": "string", "minLength": 1},
                "goal_hash": {"type": "string", "minLength": 1},
                "instruction_hash": {"type": "string", "minLength": 1},
                "checker_hash": {"type": "string", "minLength": 1},
                "style_id": identifier,
            },
            "additionalProperties": False,
        },
        "AxisContrastSet": {
            "type": "object",
            "required": [
                "id",
                "scene_spec_id",
                "realized_family_id",
                "layout_seed",
                "observation_id",
                "scene_state_hash",
                "axis",
                "subtype",
                "dcode",
                "context_signature",
                "k",
                "members",
                "pair_ids",
                "matrix_ready",
            ],
            "properties": {
                "id": identifier,
                "scene_spec_id": identifier,
                "realized_family_id": identifier,
                "layout_seed": {"type": "integer", "minimum": 0},
                "observation_id": identifier,
                "scene_state_hash": {"type": "string", "minLength": 1},
                "axis": {
                    "enum": ["D1a", "D1b", "D1c", "D2a", "D2b", "D4"]
                },
                "subtype": {"type": "string", "minLength": 1},
                "dcode": {"type": "string", "minLength": 1},
                "context_signature": {"type": "string", "minLength": 1},
                "k": {"type": "integer", "minimum": 2},
                "members": {
                    "type": "array",
                    "minItems": 2,
                    "items": {"$ref": "#/$defs/AxisContrastMember"},
                },
                "pair_ids": {
                    "type": "array",
                    "items": identifier,
                },
                "matrix_ready": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
        "OriginRelativeBounds": {
            "type": "object",
            "required": [
                "frame",
                "min_xyz",
                "max_xyz",
                "extents_xyz",
                "source",
            ],
            "properties": {
                "frame": {
                    "const": "actor_origin_after_q0_before_world_yaw"
                },
                "min_xyz": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 3,
                    "items": {"type": "number"},
                },
                "max_xyz": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 3,
                    "items": {"type": "number"},
                },
                "extents_xyz": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 3,
                    "items": {"type": "number"},
                },
                "source": {"type": "string", "minLength": 1},
                "error": {"type": "string"},
            },
            "additionalProperties": False,
        },
        "InitialPoseSupport": {
            "type": "object",
            "required": ["plane_z", "clearance_m", "origin_z", "formula"],
            "properties": {
                "plane_z": {"type": "number"},
                "clearance_m": {"type": "number", "minimum": 0},
                "origin_z": {"type": "number"},
                "formula": {
                    "const": "support_z-local_z_min+support_clearance"
                },
            },
            "additionalProperties": False,
        },
        "InitialPoseYawPolicy": {
            "oneOf": [
                {
                    "type": "object",
                    "required": ["mode", "period_deg"],
                    "properties": {
                        "mode": {"const": "free"},
                        "period_deg": {
                            "type": "number",
                            "exclusiveMinimum": 0,
                        },
                    },
                    "additionalProperties": False,
                },
                {
                    "type": "object",
                    "required": ["mode", "yaw_deg"],
                    "properties": {
                        "mode": {"const": "fixed"},
                        "yaw_deg": {"type": "number"},
                    },
                    "additionalProperties": False,
                },
            ]
        },
        "ArticulatedInitialJointState": {
            "type": "object",
            "required": ["mode", "qpos"],
            "properties": {
                "mode": {
                    "enum": [
                        "not_articulated",
                        "exact_model_data_default",
                        "exact_reviewed_profile",
                    ]
                },
                "profile_id": {"type": "string"},
                "qpos": {"type": "array", "items": {"type": "number"}},
                "joint_index": {"type": ["integer", "null"], "minimum": 0},
                "joints": {"type": "array", "items": {"type": "object"}},
                "joint_targets": {
                    "type": "array",
                    "items": {"type": "object"},
                },
            },
            "additionalProperties": False,
        },
        "AssignmentStateRevision": {
            "type": "object",
            "required": [
                "artifact_kind",
                "schema_version",
                "revision_hash",
                "source_canonical_state_hash",
                "coordinate_semantics",
                "review_authority",
                "runtime_geometry_hash",
            ],
            "properties": {
                "artifact_kind": {
                    "const": "AssignmentJointCoordinateRevision"
                },
                "schema_version": {"const": 1},
                "revision_hash": {"type": "string", "minLength": 1},
                "source_canonical_state_hash": {
                    "type": "string",
                    "minLength": 1,
                },
                "coordinate_semantics": {
                    "type": "string",
                    "minLength": 1,
                },
                "review_authority": {"type": "string", "minLength": 1},
                "runtime_geometry_hash": {
                    "type": "string",
                    "minLength": 1,
                },
            },
            "additionalProperties": False,
        },
        "DynamicAssetMassProperties": {
            "oneOf": [
                {
                    "type": "object",
                    "required": [
                        "mode",
                        "target_mass_kg",
                        "inertia_mode",
                        "policy_version",
                        "source_fields",
                    ],
                    "properties": {
                        "mode": {"const": "collision_geometry_auto"},
                        "target_mass_kg": {"type": "null"},
                        "inertia_mode": {
                            "const": "collision_geometry_auto"
                        },
                        "policy_version": {"type": "string", "minLength": 1},
                        "source_fields": {
                            "type": "array",
                            "maxItems": 0,
                        },
                    },
                    "additionalProperties": False,
                },
                {
                    "type": "object",
                    "required": [
                        "mode",
                        "target_mass_kg",
                        "inertia_mode",
                        "policy_version",
                        "source_fields",
                    ],
                    "properties": {
                        "mode": {"const": "explicit_target"},
                        "target_mass_kg": {
                            "type": "number",
                            "exclusiveMinimum": 0,
                        },
                        "inertia_mode": {"const": "density_proportional"},
                        "policy_version": {"type": "string", "minLength": 1},
                        "source_fields": {
                            "type": "array",
                            "minItems": 1,
                            "items": {"type": "string", "minLength": 1},
                            "uniqueItems": True,
                        },
                    },
                    "additionalProperties": False,
                },
            ]
        },
        "CanonicalInitialState": {
            "type": "object",
            "required": [
                "artifact_kind",
                "schema_version",
                "initial_state_id",
                "asset",
                "model_id",
                "initial_state_profile_id",
                "revision",
                "revision_hash",
                "q0_wxyz",
                "articulated_joint_state",
                "yaw_policy",
                "scale_xyz",
                "origin_relative_bounds",
                "support",
                "geometry_hash",
                "status",
                "canonical_state_hash",
            ],
            "properties": {
                "artifact_kind": {"const": "CanonicalInitialState"},
                "schema_version": {"const": 1},
                "initial_state_id": identifier,
                "asset": {"type": "string", "minLength": 1},
                "model_id": model_id,
                "initial_state_profile_id": identifier,
                "revision": {"type": "integer", "minimum": 0},
                "revision_hash": {"type": "string", "minLength": 1},
                "q0_wxyz": {
                    "type": "array",
                    "minItems": 4,
                    "maxItems": 4,
                    "items": {"type": "number"},
                },
                "articulated_joint_state": {
                    "$ref": "#/$defs/ArticulatedInitialJointState"
                },
                "yaw_policy": {"$ref": "#/$defs/InitialPoseYawPolicy"},
                "scale_xyz": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 3,
                    "items": {"type": "number", "exclusiveMinimum": 0},
                },
                "origin_relative_bounds": {
                    "$ref": "#/$defs/OriginRelativeBounds"
                },
                "support": {"$ref": "#/$defs/InitialPoseSupport"},
                "geometry_hash": {"type": "string", "minLength": 1},
                "candidate_roles": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "candidate_uses": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "dynamic_mass_properties": {
                    "$ref": "#/$defs/DynamicAssetMassProperties"
                },
                "assignment_eligible": {"type": "boolean"},
                "status": {"enum": ["approved", "unsupported"]},
                "support_validation": {"type": "object"},
                "support_validation_hash": {"type": "string"},
                "assignment_state_revision": {
                    "$ref": "#/$defs/AssignmentStateRevision"
                },
                "canonical_state_hash": {"type": "string", "minLength": 1},
            },
            "additionalProperties": False,
        },
        "InitialPoseApproval": {
            "type": "object",
            "required": [
                "artifact_kind",
                "schema_version",
                "initial_state_id",
                "revision",
                "revision_hash",
                "geometry_hash",
                "render_manifest_hash",
                "review_hash",
                "reviewer",
                "decision",
                "approval_hash",
            ],
            "properties": {
                "artifact_kind": {"const": "InitialPoseApproval"},
                "schema_version": {"const": 1},
                "initial_state_id": identifier,
                "revision": {"type": "integer", "minimum": 0},
                "revision_hash": {"type": "string", "minLength": 1},
                "geometry_hash": {"type": "string", "minLength": 1},
                "render_manifest_hash": {"type": "string", "minLength": 1},
                "review_hash": {"type": "string", "minLength": 1},
                "reviewer": {"type": "string", "minLength": 1},
                "decision": {"enum": ["approved", "unsupported"]},
                "notes": {"type": "string"},
                "approval_hash": {"type": "string", "minLength": 1},
            },
            "additionalProperties": False,
        },
        "CanonicalInitialStatePlacement": {
            "type": "object",
            "required": [
                "initial_state_id",
                "revision",
                "revision_hash",
                "geometry_hash",
                "canonical_state_hash",
                "q0_wxyz",
                "world_yaw_deg",
                "yaw_policy",
                "scale_xyz",
                "origin_relative_bounds",
                "support",
                "articulated_joint_state",
                "dynamic_mass_properties",
            ],
            "properties": {
                "initial_state_id": identifier,
                "revision": {"type": "integer", "minimum": 0},
                "revision_hash": {"type": "string", "minLength": 1},
                "geometry_hash": {"type": "string", "minLength": 1},
                "canonical_state_hash": {
                    "type": "string",
                    "minLength": 1,
                },
                "q0_wxyz": {
                    "type": "array",
                    "minItems": 4,
                    "maxItems": 4,
                    "items": {"type": "number"},
                },
                "world_yaw_deg": {"type": "number"},
                "yaw_policy": {"$ref": "#/$defs/InitialPoseYawPolicy"},
                "scale_xyz": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 3,
                    "items": {"type": "number"},
                },
                "origin_relative_bounds": {
                    "$ref": "#/$defs/OriginRelativeBounds"
                },
                "support": {"$ref": "#/$defs/InitialPoseSupport"},
                "articulated_joint_state": {
                    "$ref": "#/$defs/ArticulatedInitialJointState"
                },
                "dynamic_mass_properties": {
                    "$ref": "#/$defs/DynamicAssetMassProperties"
                },
                "assignment_state_revision": {
                    "$ref": "#/$defs/AssignmentStateRevision"
                },
            },
            "additionalProperties": False,
        },
        "SceneValidationRecord": {
            "type": "object",
            "required": [
                "artifact_kind",
                "schema_version",
                "scene_validation_version",
                "id",
                "scene_spec_id",
                "realized_family_id",
                "source_scene_hash",
                "source_realization_hash",
                "source_unit_hash",
                "plan_hash",
                "config_hash",
                "contract_id",
                "success_contract_version",
                "success_contract_hash",
                "calibration_only",
                "publication_ready",
                "descriptor_gate",
                "not_success_at_t0",
                "infrastructure_failure",
                "source_failure",
                "internal_failure",
                "task_results",
                "summary",
                "provenance",
                "artifact_hash",
            ],
            "properties": {
                "artifact_kind": {"const": "scene_validation_record"},
                "schema_version": {"const": 1},
                "scene_validation_version": {"const": 9},
                "id": identifier,
                "scene_spec_id": identifier,
                "realized_family_id": identifier,
                "source_scene_hash": {"type": "string", "minLength": 1},
                "source_realization_hash": {"type": "string", "minLength": 1},
                "source_unit_hash": {"type": "string", "minLength": 1},
                "plan_hash": {"type": "string", "minLength": 1},
                "config_hash": {"type": "string", "minLength": 1},
                "contract_id": {"type": "string", "minLength": 1},
                "success_contract_version": {"type": "string", "minLength": 1},
                "success_contract_hash": {"type": "string", "minLength": 1},
                "calibration_only": {"type": "boolean"},
                "publication_ready": {"type": "boolean"},
                "descriptor_gate": {"type": "object"},
                "not_success_at_t0": {
                    "type": "object",
                    "required": ["passed", "task_count"],
                    "properties": {
                        "passed": {"type": "boolean"},
                        "task_count": {"type": "integer", "minimum": 0},
                    },
                    "additionalProperties": False,
                },
                "infrastructure_failure": {"type": "boolean"},
                "source_failure": {"type": "boolean"},
                "internal_failure": {"type": "boolean"},
                "task_results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": [
                            "task_record_id",
                            "goal_variant_id",
                            "source_task_hash",
                            "validated",
                            "excluded",
                            "success_count",
                            "denominator",
                            "minimum_success_count",
                            "success_rate",
                            "classification_counts",
                            "t0_false",
                            "contract_id",
                            "success_contract_version",
                            "success_contract_hash",
                            "success_snapshot_hashes",
                            "infrastructure_failure_count",
                            "source_failure_count",
                            "internal_failure_count",
                            "attempts",
                        ],
                        "properties": {
                            "task_record_id": identifier,
                            "goal_variant_id": identifier,
                            "source_task_hash": {
                                "type": "string",
                                "minLength": 1,
                            },
                            "validated": {"type": "boolean"},
                            "excluded": {"type": "boolean"},
                            "success_count": {
                                "type": "integer",
                                "minimum": 0,
                            },
                            "denominator": {
                                "type": "integer",
                                "minimum": 0,
                            },
                            "minimum_success_count": {
                                "type": "integer",
                                "minimum": 1,
                            },
                            "success_rate": {
                                "type": ["number", "null"],
                                "minimum": 0,
                                "maximum": 1,
                            },
                            "classification_counts": {
                                "type": "object",
                                "additionalProperties": {
                                    "type": "integer",
                                    "minimum": 0,
                                },
                            },
                            "t0_false": {"type": "boolean"},
                            "contract_id": {"type": "string", "minLength": 1},
                            "success_contract_version": {"type": "string", "minLength": 1},
                            "success_contract_hash": {"type": "string", "minLength": 1},
                            "success_snapshot_hashes": {
                                "type": "array",
                                "uniqueItems": True,
                                "items": {"type": "string", "minLength": 1},
                            },
                            "infrastructure_failure_count": {
                                "type": "integer",
                                "minimum": 0,
                            },
                            "source_failure_count": {
                                "type": "integer",
                                "minimum": 0,
                            },
                            "internal_failure_count": {
                                "type": "integer",
                                "minimum": 0,
                            },
                            "attempts": {
                                "type": "array",
                                "items": {"type": "object"},
                            },
                        },
                        "additionalProperties": False,
                    },
                },
                "summary": {
                    "type": "object",
                    "required": [
                        "task_count",
                        "validated_task_count",
                        "excluded_task_count",
                    ],
                    "properties": {
                        "task_count": {"type": "integer", "minimum": 0},
                        "validated_task_count": {
                            "type": "integer",
                            "minimum": 0,
                        },
                        "excluded_task_count": {
                            "type": "integer",
                            "minimum": 0,
                        },
                    },
                    "additionalProperties": False,
                },
                "provenance": {"$ref": "#/$defs/Provenance"},
                "artifact_hash": {"type": "string", "minLength": 1},
            },
            "additionalProperties": False,
        },
    }


def _d1a_attribute_archetype() -> dict[str, Any]:
    return {
        "id": "attribute_subject_pair",
        "roles": [
            {
                "id": "subj_a",
                "kind": "manipuland",
                "allowed_asset_roles": ["manipuland"],
                "descriptor_role": "subject",
                "cardinality": 1,
            },
            {
                "id": "subj_b",
                "kind": "manipuland",
                "allowed_asset_roles": ["manipuland"],
                "descriptor_role": "subject",
                "cardinality": 1,
            },
        ],
        "constraints": [
            _constraint("c_same_category", "same_category", "semantic", "assignment", ["subj_a", "subj_b"]),
            _constraint("c_distinct_subjects", "distinct_entity", "semantic", "assignment", ["subj_a", "subj_b"]),
            _constraint("c_attr_a", "reviewed_attribute", "asset", "assignment", ["subj_a"], {"kind": "color"}),
            _constraint("c_attr_b", "reviewed_attribute", "asset", "assignment", ["subj_b"], {"kind": "color"}),
            _constraint(
                "c_unique_a",
                "unique_descriptor",
                "semantic",
                "assignment",
                ["subj_a"],
                {"kind": "color", "peers": ["subj_a", "subj_b"]},
            ),
            _constraint(
                "c_unique_b",
                "unique_descriptor",
                "semantic",
                "assignment",
                ["subj_b"],
                {"kind": "color", "peers": ["subj_a", "subj_b"]},
            ),
            _constraint("c_pick_a", "use_validated", "asset", "assignment", ["subj_a"], {"use": "pick_left"}),
            _constraint("c_pick_b", "use_validated", "asset", "assignment", ["subj_b"], {"use": "pick_left"}),
        ],
        "variant_schemas": [
            _variant_schema("pick_a", "subj_a"),
            _variant_schema("pick_b", "subj_b"),
        ],
        "minimal_pair_edges": [{"from": "pick_a", "to": "pick_b", "axis": "D1a", "subtype": "color"}],
    }


def _d2b_relation_role_graph() -> dict[str, Any]:
    return {
        "id": "rg_d2b_directional_relation_pair",
        "roles": [
            {
                "id": "subj",
                "kind": "manipuland",
                "allowed_asset_roles": ["manipuland"],
                "descriptor_role": "subject",
                "cardinality": 1,
            },
            {
                "id": "anchor",
                "kind": "reference_anchor",
                "allowed_asset_roles": ["manipuland", "reference", "surface", "container"],
                "descriptor_role": "target",
                "cardinality": 1,
            },
        ],
        "constraints": [
            _constraint("c_pick_subject", "use_validated", "asset", "assignment", ["subj"], {"use": "pick_left"}),
            _constraint(
                "c_anchor_valid",
                "use_validated",
                "asset",
                "assignment",
                ["anchor"],
                {"use": "relation_anchor"},
            ),
            _constraint(
                "c_terminal_relations",
                "terminal_relation_supported",
                "semantic",
                "unit_build",
                ["subj", "anchor"],
                {"relations": ["left_of", "right_of"]},
            ),
            _constraint(
                "c_not_success_at_t0",
                "not_success_at_t0",
                "layout",
                "layout",
                ["subj", "anchor"],
                {"relations": ["left_of", "right_of"]},
            ),
        ],
        "variant_schemas": [
            {
                "id": "place_left",
                "action": {"verb": "place_left_of", "arm": "left", "relation": "left_of"},
                "role_bindings": {"subject": "subj", "target": "anchor"},
                "params": {},
            },
            {
                "id": "place_right",
                "action": {"verb": "place_right_of", "arm": "left", "relation": "right_of"},
                "role_bindings": {"subject": "subj", "target": "anchor"},
                "params": {},
            },
        ],
        "minimal_pair_edges": [
            {"from": "place_left", "to": "place_right", "axis": "D2b", "subtype": "relation"}
        ],
    }


def _d1a_assignment_candidate() -> dict[str, Any]:
    return {
        "id": "ac_color_cube_red_blue_000",
        "abstract_family_id": "asf_d1a_color_subject_pair",
        "archetype_ids": ["attribute_subject_pair"],
        "roles": {
            "subj_a": {
                "asset": "073_rubikscube",
                "model_id": "0",
                "category": "cube",
                "roles": ["manipuland"],
            },
            "subj_b": {
                "asset": "073_rubikscube",
                "model_id": "1",
                "category": "cube",
                "roles": ["manipuland"],
            },
        },
        "resolved_values": {
            "subj_a": {"noun": "cube", "attributes": {"color": "red"}},
            "subj_b": {"noun": "cube", "attributes": {"color": "blue"}},
        },
        "proven_facts": [
            {"constraint": "c_same_category", "fact": "same_category(subj_a, subj_b, cube)"},
            {"constraint": "c_unique_a", "fact": "unique_descriptor(subj_a, color:red)"},
            {"constraint": "c_unique_b", "fact": "unique_descriptor(subj_b, color:blue)"},
            {"constraint": "c_pick_a", "fact": "validated_use(subj_a, pick_left)"},
            {"constraint": "c_pick_b", "fact": "validated_use(subj_b, pick_left)"},
        ],
        "evidence_refs": [
            {"kind": "attribute_review", "asset": "073_rubikscube", "model_id": "0", "field": "color"},
            {"kind": "attribute_review", "asset": "073_rubikscube", "model_id": "1", "field": "color"},
            {"kind": "asset_validation", "asset": "073_rubikscube", "model_id": "0", "use": "pick_left"},
            {"kind": "asset_validation", "asset": "073_rubikscube", "model_id": "1", "use": "pick_left"},
        ],
        "layout_requirements": [
            {"type": "object_unit", "roles": ["subj_a"]},
            {"type": "object_unit", "roles": ["subj_b"]},
            {"type": "reach_zone", "role": "subj_a", "allowed": ["left_only", "both"]},
            {"type": "reach_zone", "role": "subj_b", "allowed": ["left_only", "both"]},
        ],
        "materialization_params": {
            "variant_substitutions": {
                "subj_a": "obj_a",
                "subj_b": "obj_b",
                "subj_a.attribute.color": "color:red",
                "subj_b.attribute.color": "color:blue",
            }
        },
        "provenance": {
            "generator": "if_benchmark.schema.examples",
            "config_hash": "m6_contract_v1",
            "sources": [
                {
                    "kind": "reviewed_attribute_catalog",
                    "path": "if_benchmark/data/attributes/reviewed_attribute_catalog.yml",
                    "hash": "example_attribute_catalog_hash",
                },
                {
                    "kind": "asset_validation",
                    "path": "if_benchmark/assets/asset_validation.yml",
                    "hash": "example_asset_validation_hash",
                },
            ],
        },
    }


def _d1a_goal_variant(schema_id: str, role_ref: str, entity_ref: str, color: str) -> dict[str, Any]:
    return {
        "id": schema_id,
        "schema_id": schema_id,
        "assignment_id": "ac_color_cube_red_blue_000",
        "role_refs": {"subject": role_ref},
        "entity_refs": {"subject": entity_ref},
        "intended_axes": [{"axis": "D1a", "subtype": "color"}],
        "goal": {
            "verb": "pick",
            "subject": entity_ref,
            "arm": "left",
            "target": None,
            "subject2": None,
            "target2": None,
            "descriptors": {
                "subject": {
                    "attribute": {
                        "subtype": "color",
                        "value": color,
                    }
                }
            },
            "params": {},
        },
        "export": {
            "variant": f"left_arm_pick({entity_ref})|desc=subject.attribute=color:{color}",
        },
    }


def _example_rect(
    half_extents: list[float], *, center: list[float] | None = None, source: str = "example"
) -> dict[str, Any]:
    return {
        "shape": "rect",
        "frame": "unit_local",
        "center_xy": list(center or [0.0, 0.0]),
        "half_extents_xy": list(half_extents),
        "source": source,
    }


def _example_pose(translation: list[float] | None = None) -> dict[str, Any]:
    return {
        "translation_xyz": list(translation or [0.0, 0.0, 0.0]),
        "q0_wxyz": [1.0, 0.0, 0.0, 0.0],
        "relative_yaw_deg": 0.0,
        "placement_mode": "table_resting",
    }


def _example_object_params(entity_key: str) -> dict[str, Any]:
    return {
        "object_key": entity_key,
        "footprint": _example_rect([0.04, 0.04]),
        "anchors": {
            "origin": [0.0, 0.0, 0.04],
            "footprint_center": [0.0, 0.0, 0.04],
            "support": [0.0, 0.0, 0.0],
        },
        "local_pose": _example_pose([0.0, 0.0, 0.04]),
        "local_z_bounds": [-0.04, 0.04],
        "geometry_evidence": {"source": "example"},
    }


def _d1a_realized_scene_family() -> dict[str, Any]:
    return {
        "id": "rsf_d1a_color_cube_red_blue",
        "abstract_family_id": "asf_d1a_color_subject_pair",
        "assignment_id": "ac_color_cube_red_blue_000",
        "archetype_ids": ["attribute_subject_pair"],
        "role_bindings": {
            "subj_a": {
                "entity_key": "obj_a",
                "asset": "073_rubikscube",
                "model_id": "0",
                "category": "cube",
            },
            "subj_b": {
                "entity_key": "obj_b",
                "asset": "073_rubikscube",
                "model_id": "1",
                "category": "cube",
            },
        },
        "templates": [
            {
                "key": "obj_a",
                "role_ref": "subj_a",
                "asset": "073_rubikscube",
                "model_candidates": ["0"],
                "roles": ["manipuland"],
            },
            {
                "key": "obj_b",
                "role_ref": "subj_b",
                "asset": "073_rubikscube",
                "model_candidates": ["1"],
                "roles": ["manipuland"],
            },
        ],
        "goal_variants": [
            _d1a_goal_variant("pick_a", "subj_a", "obj_a", "red"),
            _d1a_goal_variant("pick_b", "subj_b", "obj_b", "blue"),
        ],
        "unit_inputs": [
            {
                "id": "unit_input_obj_a",
                "type": "ObjectUnit",
                "roles": ["subj_a"],
                "entity_keys": ["obj_a"],
                "source_constraints": ["c_layout_obj_a_reach"],
                "geometry_refs": {"subj_a": "073_rubikscube|0"},
                "pair_preconditions": {},
                "params": {},
            },
            {
                "id": "unit_input_obj_b",
                "type": "ObjectUnit",
                "roles": ["subj_b"],
                "entity_keys": ["obj_b"],
                "source_constraints": ["c_layout_obj_b_reach"],
                "geometry_refs": {"subj_b": "073_rubikscube|1"},
                "pair_preconditions": {},
                "params": {},
            },
        ],
        "units": [
            {
                "id": "unit_obj_a",
                "type": "ObjectUnit",
                "roles": ["subj_a"],
                "entity_keys": ["obj_a"],
                "params": _example_object_params("obj_a"),
            },
            {
                "id": "unit_obj_b",
                "type": "ObjectUnit",
                "roles": ["subj_b"],
                "entity_keys": ["obj_b"],
                "params": _example_object_params("obj_b"),
            },
        ],
        "layout_constraints": [
            _constraint(
                "c_layout_obj_a_reach",
                "reach_zone",
                "layout",
                "layout",
                ["subj_a"],
                {"allowed": ["left_only", "both"]},
            ),
            _constraint(
                "c_layout_obj_b_reach",
                "reach_zone",
                "layout",
                "layout",
                ["subj_b"],
                {"allowed": ["left_only", "both"]},
            ),
        ],
        "minimal_pair_edges": [{"from": "pick_a", "to": "pick_b", "axis": "D1a", "subtype": "color"}],
        "evidence": {
            "assignment_candidate": "ac_color_cube_red_blue_000",
            "constraints_satisfied": ["c_same_category", "c_unique_a", "c_unique_b", "c_pick_a", "c_pick_b"],
        },
        "export": {
            "legacy_scene_family": {
                "variants_field": [
                    "left_arm_pick(obj_a)|desc=subject.attribute=color:red",
                    "left_arm_pick(obj_b)|desc=subject.attribute=color:blue",
                ]
            }
        },
    }


def _d1a_scene_spec() -> dict[str, Any]:
    return {
        "id": "ss_rsf_d1a_color_cube_red_blue_seed7",
        "realized_family_id": "rsf_d1a_color_cube_red_blue",
        "layout_seed": 7,
        "entities": {
            "obj_a": {
                "role_ref": "subj_a",
                "asset": "073_rubikscube",
                "model_id": "0",
                "unit_id": "unit_obj_a",
                "pose": {
                    "position": [-0.12, 0.12, 0.79],
                    "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                },
            },
            "obj_b": {
                "role_ref": "subj_b",
                "asset": "073_rubikscube",
                "model_id": "1",
                "unit_id": "unit_obj_b",
                "pose": {
                    "position": [0.14, 0.1, 0.79],
                    "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                },
            },
        },
        "unit_poses": [
            {
                "unit_id": "unit_obj_a",
                "position": [-0.12, 0.12, 0.74],
                "yaw_deg": 0.0,
            },
            {
                "unit_id": "unit_obj_b",
                "position": [0.14, 0.1, 0.74],
                "yaw_deg": 0.0,
            },
        ],
        "actor": {
            "embodiment": "aloha-agilex",
            "base_pose": {
                "position": [0.0, -0.65, 0.0],
                "quat_wxyz": [0.707106781187, 0.0, 0.0, 0.707106781187],
            },
        },
        "camera": {
            "name": "head_camera",
            "type": "D435",
            "width": 320,
            "height": 240,
            "fovy_deg": 37.0,
            "near": 0.1,
            "far": 100.0,
            "pose": {
                "position": [-0.032, -0.45, 1.35],
                "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
            },
            "table_visible_polygon_xy": [
                [-0.303932, -0.247436],
                [0.239932, -0.247436],
                [0.422136, 0.433252],
                [-0.486136, 0.433252],
            ],
        },
        "reach": {
            "obj_a": {
                "reachable_arms": ["left"],
                "zone": "left_only",
                "evidence": "example_static_proxy",
            },
            "obj_b": {
                "reachable_arms": ["left", "right"],
                "zone": "both",
                "evidence": "example_static_proxy",
            },
        },
        "terminal_reach": {},
        "initial_observation": {
            "observation_id": "obs_seed7",
            "scene_state_hash": "example_state_hash",
            "variant_invariant": True,
            "artifact_path": None,
        },
        "goal_variant_ids": ["pick_a", "pick_b"],
        "provenance": {
            "generator": "if_benchmark.layout",
            "config_hash": "example_layout_config_hash",
            "sources": [
                {
                    "kind": "realized_scene_family",
                    "hash": "example_realization_hash",
                    "record_id": "rsf_d1a_color_cube_red_blue",
                },
                {
                    "kind": "unit_build",
                    "hash": "example_unit_hash",
                    "record_id": "rsf_d1a_color_cube_red_blue",
                },
            ],
        },
    }


def _unit_contract_realized_scene_family() -> dict[str, Any]:
    return {
        "id": "rsf_unit_contracts",
        "abstract_family_id": "asf_unit_contracts",
        "assignment_id": "ac_unit_contracts",
        "archetype_ids": ["unit_contracts"],
        "role_bindings": {
            "subj": {
                "entity_key": "obj_subject",
                "asset": "073_rubikscube",
                "model_id": "0",
                "category": "cube",
            },
            "landmark": {
                "entity_key": "obj_landmark",
                "asset": "043_book",
                "model_id": "0",
                "category": "book",
            },
            "receiver": {
                "entity_key": "obj_receiver",
                "asset": "105_bowl",
                "model_id": "0",
                "category": "bowl",
            },
        },
        "templates": [
            {
                "key": "obj_subject",
                "role_ref": "subj",
                "asset": "073_rubikscube",
                "model_candidates": ["0"],
                "roles": ["manipuland"],
            },
            {
                "key": "obj_landmark",
                "role_ref": "landmark",
                "asset": "043_book",
                "model_candidates": ["0"],
                "roles": ["reference"],
            },
            {
                "key": "obj_receiver",
                "role_ref": "receiver",
                "asset": "105_bowl",
                "model_candidates": ["0"],
                "roles": ["container"],
            },
        ],
        "goal_variants": [
            {
                "id": "place_left",
                "schema_id": "place_left",
                "assignment_id": "ac_unit_contracts",
                "role_refs": {"subject": "subj", "target": "landmark"},
                "entity_refs": {
                    "subject": "obj_subject",
                    "target": "obj_landmark",
                },
                "intended_axes": [{"axis": "D2b", "subtype": "relation"}],
                "goal": {
                    "verb": "place_left_of",
                    "action_family": "O2O",
                    "relation": "left_of",
                    "subject": "obj_subject",
                    "arm": "left",
                    "target": "obj_landmark",
                    "subject2": None,
                    "target2": None,
                    "descriptors": {},
                    "params": {},
                },
                "export": {
                    "variant": "left_arm_place_left_of(obj_subject,obj_landmark)",
                },
            }
        ],
        "unit_inputs": [
            {
                "id": "unit_input_subject",
                "type": "ObjectUnit",
                "roles": ["subj"],
                "entity_keys": ["obj_subject"],
                "source_constraints": [],
                "geometry_refs": {"subj": "073_rubikscube|0"},
                "pair_preconditions": {},
                "params": {},
            },
            {
                "id": "unit_input_landmark_relation",
                "type": "LandmarkRelationUnit",
                "roles": ["subj", "landmark"],
                "entity_keys": ["obj_subject", "obj_landmark"],
                "source_constraints": [],
                "geometry_refs": {
                    "subj": "073_rubikscube|0",
                    "landmark": "043_book|0",
                },
                "pair_preconditions": {},
                "params": {"landmark_role": "landmark", "predicates": ["left_of"]},
            },
            {
                "id": "unit_input_receiver_reservation",
                "type": "ReceiverReservationUnit",
                "roles": ["receiver"],
                "entity_keys": ["obj_receiver"],
                "source_constraints": [],
                "geometry_refs": {"receiver": "105_bowl|0"},
                "pair_preconditions": {},
                "params": {"receiver_roles": ["receiver"], "relation": "place_in"},
            },
        ],
        "units": [
            {
                "id": "unit_subject",
                "type": "ObjectUnit",
                "roles": ["subj"],
                "entity_keys": ["obj_subject"],
                "params": _example_object_params("obj_subject"),
            },
            {
                "id": "unit_landmark_relation",
                "type": "LandmarkRelationUnit",
                "roles": ["subj", "landmark"],
                "entity_keys": ["obj_subject", "obj_landmark"],
                "params": {
                    "landmark_key": "obj_landmark",
                    "internal_layout": {
                        "obj_subject": _example_pose([-0.1, 0.0, 0.04]),
                        "obj_landmark": _example_pose([0.0, 0.0, 0.04]),
                    },
                    "member_footprints": {
                        "obj_subject": _example_rect(
                            [0.04, 0.04], center=[-0.1, 0.0]
                        ),
                        "obj_landmark": _example_rect([0.04, 0.04]),
                    },
                    "member_z_bounds": {
                        "obj_subject": [-0.04, 0.04],
                        "obj_landmark": [-0.04, 0.04],
                    },
                    "footprint": _example_rect(
                        [0.09, 0.04], center=[-0.05, 0.0]
                    ),
                    "anchors": {
                        "obj_subject:origin": [-0.1, 0.0, 0.04],
                        "obj_subject:support": [-0.1, 0.0, 0.0],
                        "obj_landmark:origin": [0.0, 0.0, 0.04],
                        "obj_landmark:support": [0.0, 0.0, 0.0],
                    },
                    "guaranteed_facts": [
                        {
                            "predicate": "spatial",
                            "subject": "obj_subject",
                            "relation": "left_of",
                            "reference": "obj_landmark",
                            "source_constraint_ids": [],
                        }
                    ],
                    "geometry_evidence": {
                        "obj_subject": {"source": "example"},
                        "obj_landmark": {"source": "example"},
                    },
                    "pair_evidence": [],
                },
            },
            {
                "id": "unit_receiver_reservation",
                "type": "ReceiverReservationUnit",
                "roles": ["receiver"],
                "entity_keys": ["obj_receiver"],
                "params": {
                    "receiver_keys": ["obj_receiver"],
                    "subject_keys": ["obj_subject"],
                    "relations": ["in"],
                    "receiver_layout": {
                        "obj_receiver": _example_pose([0.0, 0.0, 0.04]),
                    },
                    "receiver_footprints": {
                        "obj_receiver": _example_rect([0.09, 0.09]),
                    },
                    "receiver_z_bounds": {
                        "obj_receiver": [-0.04, 0.04],
                    },
                    "reserved_regions": [
                        {
                            "id": "region_subject_in_receiver",
                            "relation": "in",
                            "subject_key": "obj_subject",
                            "receiver_keys": ["obj_receiver"],
                            "footprint": _example_rect([0.04, 0.04]),
                            "support_clearance_m": 0.002,
                            "terminal_clearance_m": 0.03,
                            "placement_recipe": {
                                "clearance_m": 0.03,
                                "yaw_convention": (
                                    "subject_yaw_minus_receiver_yaw"
                                ),
                                "settle_policy": (
                                    "m5_terminal_settle_v11_proportional_"
                                    "asset_mass_inertia"
                                ),
                            },
                            "placement_recipe_source": (
                                "frozen_m5_projection"
                            ),
                            "pair_evidence_ids": ["pair_example"],
                            "relative_yaw_deg": 0.0,
                            "usable_region": {
                                "x": 0.12,
                                "y": 0.12,
                                "factor": 0.7,
                                "edge_clearance_m": 0.015,
                                "solver_policy_hash": "example",
                            },
                        }
                    ],
                    "footprint": _example_rect([0.09, 0.09]),
                    "anchors": {
                        "obj_receiver:origin": [0.0, 0.0, 0.04],
                        "obj_receiver:support": [0.0, 0.0, 0.0],
                    },
                    "guaranteed_facts": [
                        {
                            "predicate": "terminal_region",
                            "subject": "obj_subject",
                            "relation": "in",
                            "receivers": ["obj_receiver"],
                            "region_id": "region_subject_in_receiver",
                            "source_constraint_ids": [],
                        }
                    ],
                    "geometry_evidence": {
                        "obj_receiver": {"source": "example"}
                    },
                    "subject_geometry_evidence": {
                        "obj_subject": {"source": "example"}
                    },
                    "subject_z_bounds": {
                        "obj_subject": [-0.04, 0.04],
                    },
                    "pair_evidence": [
                        {"evidence_id": "pair_example", "pair_key": "example"}
                    ],
                },
            },
        ],
        "layout_constraints": [],
        "minimal_pair_edges": [],
        "evidence": {"purpose": "Milestone 6 unit contract coverage"},
        "export": {},
    }


def _variant_schema(schema_id: str, role_id: str) -> dict[str, Any]:
    return {
        "id": schema_id,
        "action": {"verb": "pick", "arm": "left"},
        "role_bindings": {"subject": role_id},
        "descriptors": {
            "subject": [
                {
                    "kind": "attribute",
                    "subtype": "color",
                    "source_role": role_id,
                }
            ]
        },
        "params": {},
    }


def _constraint(
    constraint_id: str,
    constraint_type: str,
    scope: str,
    phase: str,
    roles: list[str],
    params: dict[str, Any] | None = None,
    *,
    hard: bool = True,
) -> dict[str, Any]:
    return {
        "id": constraint_id,
        "type": constraint_type,
        "scope": scope,
        "phase": phase,
        "roles": roles,
        "params": params or {},
        "hard": hard,
    }


def _semantic_errors(kind: str, payload: Any) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    errors.extend(_legacy_variant_boundary_errors(payload))

    if kind in {"archetype", "role_graph"}:
        errors.extend(_role_graph_like_errors(payload))
    elif kind == "axis_plan":
        errors.extend(_axis_plan_errors(payload))
    elif kind == "constraint":
        errors.extend(_constraint_errors(payload, ["constraint"]))
    elif kind == "assignment_candidate":
        errors.extend(_assignment_candidate_errors(payload))
    elif kind == "concrete_goal":
        entity_keys = set(payload.get("entity_keys", []))
        errors.extend(_concrete_goal_errors(payload.get("goal", {}), entity_keys, ["concrete_goal", "goal"]))
    elif kind == "goal_variant":
        errors.extend(_goal_variant_errors(payload, None, ["goal_variant"]))
    elif kind == "realized_scene_family":
        errors.extend(_realized_scene_family_errors(payload))
    elif kind == "scene_spec":
        errors.extend(_scene_spec_errors(payload))
    elif kind == "task_record":
        errors.extend(_task_record_errors(payload))
    elif kind == "minimal_pair":
        errors.extend(_minimal_pair_errors(payload))
    elif kind == "axis_contrast_set":
        errors.extend(_axis_contrast_set_errors(payload))
    return errors


def _axis_plan_errors(payload: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = [
        (
            item.get("axis"),
            item.get("subtype"),
            item.get("role"),
            _canonical_json(item.get("value")),
            _canonical_json(item.get("values")),
        )
        for item in payload
        if isinstance(item, dict)
    ]
    return _duplicate_errors(keys, ["axis_plan"], "duplicate_axis_spec")


def _role_graph_like_errors(payload: dict[str, Any]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    role_ids = [role.get("id") for role in payload.get("roles", []) if isinstance(role, dict)]
    variant_ids = [
        variant.get("id")
        for variant in payload.get("variant_schemas", [])
        if isinstance(variant, dict)
    ]
    constraint_ids = [
        constraint.get("id")
        for constraint in payload.get("constraints", [])
        if isinstance(constraint, dict)
    ]
    errors.extend(_duplicate_errors(role_ids, ["roles"], "duplicate_role_id"))
    errors.extend(_duplicate_errors(variant_ids, ["variant_schemas"], "duplicate_variant_schema_id"))
    errors.extend(_duplicate_errors(constraint_ids, ["constraints"], "duplicate_constraint_id"))

    role_set = {str(role_id) for role_id in role_ids}
    variant_set = {str(variant_id) for variant_id in variant_ids}
    constraints = payload.get("constraints", [])
    if not isinstance(constraints, list):
        errors.append(_error(["constraints"], "constraints must be a flat list of records", "flat_constraints"))
        constraints = []
    for index, constraint in enumerate(constraints):
        if isinstance(constraint, dict):
            errors.extend(_constraint_errors(constraint, ["constraints", index]))
            for role in constraint.get("roles", []):
                if role not in role_set:
                    errors.append(
                        _error(["constraints", index, "roles"], f"unknown role reference: {role}", "role_refs")
                    )
            errors.extend(_param_role_ref_errors(constraint.get("params", {}), role_set, ["constraints", index, "params"]))
            related = constraint.get("related_constraint")
            if related is not None and related not in set(constraint_ids):
                errors.append(
                    _error(
                        ["constraints", index, "related_constraint"],
                        f"unknown related constraint id: {related}",
                        "constraint_refs",
                    )
                )

    for index, variant in enumerate(payload.get("variant_schemas", [])):
        if not isinstance(variant, dict):
            continue
        errors.extend(_variant_schema_role_errors(variant, role_set, ["variant_schemas", index]))
    for index, edge in enumerate(payload.get("minimal_pair_edges", [])):
        if not isinstance(edge, dict):
            continue
        if edge.get("from") == edge.get("to"):
            errors.append(
                _error(
                    ["minimal_pair_edges", index],
                    "minimal-pair edge endpoints must differ",
                    "variant_refs",
                )
            )
        for key in ("from", "to"):
            value = edge.get(key)
            if value not in variant_set:
                errors.append(_error(["minimal_pair_edges", index, key], f"unknown variant schema id: {value}", "variant_refs"))

    if "constraint_index" in payload:
        expected = derive_constraint_index(constraints)
        if payload.get("constraint_index") != expected:
            errors.append(
                _error(
                    ["constraint_index"],
                    "constraint_index must match indexes derived from constraints",
                    "derived_constraint_index",
                )
            )
    return errors


def _constraint_errors(constraint: dict[str, Any], path: list[Any]) -> list[dict[str, Any]]:
    errors = []
    keys = set(constraint)
    missing = [key for key in CONSTRAINT_REQUIRED_KEYS if key not in constraint]
    extra = sorted(keys - CONSTRAINT_ALLOWED_KEYS)
    if missing:
        errors.append(_error(path, f"constraint is missing required keys: {missing}", "flat_constraints"))
    if extra:
        errors.append(_error(path, f"constraint has non-canonical keys: {extra}", "flat_constraints"))
    return errors


def _assignment_candidate_errors(payload: dict[str, Any]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    role_ids = set(payload.get("roles", {}))
    resolved_ids = set(payload.get("resolved_values", {}))
    unknown_resolved = sorted(resolved_ids - role_ids)
    if unknown_resolved:
        errors.append(
            _error(
                ["resolved_values"],
                f"resolved_values contains unknown roles: {unknown_resolved}",
                "assignment_role_refs",
            )
        )
    for index, requirement in enumerate(payload.get("layout_requirements", [])):
        if not isinstance(requirement, dict):
            continue
        references: list[Any] = []
        if "role" in requirement:
            references.append(requirement["role"])
        if isinstance(requirement.get("roles"), list):
            references.extend(requirement["roles"])
        for role in references:
            if role not in role_ids:
                errors.append(
                    _error(
                        ["layout_requirements", index],
                        f"layout requirement references unknown role: {role}",
                        "assignment_role_refs",
                    )
                )
    return errors


def _variant_schema_role_errors(
    variant: dict[str, Any],
    role_set: set[str],
    path: list[Any],
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for slot, role in (variant.get("role_bindings") or {}).items():
        if role not in role_set:
            errors.append(_error(path + ["role_bindings", slot], f"unknown role reference: {role}", "variant_role_refs"))
    for slot, descriptors in (variant.get("descriptors") or {}).items():
        if not isinstance(descriptors, list):
            continue
        for index, descriptor in enumerate(descriptors):
            if not isinstance(descriptor, dict):
                continue
            for key in ("source_role", "reference_role"):
                role = descriptor.get(key)
                if role is not None and role not in role_set:
                    errors.append(
                        _error(
                            path + ["descriptors", slot, index, key],
                            f"unknown role reference: {role}",
                            "variant_role_refs",
                        )
                    )
    return errors


def _param_role_ref_errors(params: Any, role_set: set[str], path: list[Any]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    if not isinstance(params, dict):
        return errors
    for key in ROLE_REF_PARAM_KEYS:
        if key not in params:
            continue
        value = params[key]
        values = value if isinstance(value, list) else [value]
        for index, role in enumerate(values):
            if role not in role_set:
                suffix = [key, index] if isinstance(value, list) else [key]
                errors.append(_error(path + suffix, f"unknown role reference: {role}", "constraint_role_refs"))
    for key, value in params.items():
        if isinstance(value, dict):
            errors.extend(
                _param_role_ref_errors(value, role_set, path + [key])
            )
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, dict):
                    errors.extend(
                        _param_role_ref_errors(
                            item, role_set, path + [key, index]
                        )
                    )
    return errors


def _realized_scene_family_errors(payload: dict[str, Any]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    role_ids = set(payload.get("role_bindings", {}))
    entity_keys = {
        binding.get("entity_key")
        for binding in payload.get("role_bindings", {}).values()
        if isinstance(binding, dict)
    }
    entity_keys.update(template.get("key") for template in payload.get("templates", []) if isinstance(template, dict))
    entity_keys = {str(key) for key in entity_keys if key is not None}
    binding_entity_keys = [
        binding.get("entity_key")
        for binding in payload.get("role_bindings", {}).values()
        if isinstance(binding, dict)
    ]
    template_keys = [
        template.get("key")
        for template in payload.get("templates", [])
        if isinstance(template, dict)
    ]
    unit_ids = [
        unit.get("id")
        for unit in payload.get("units", [])
        if isinstance(unit, dict)
    ]
    unit_input_ids = [
        unit_input.get("id")
        for unit_input in payload.get("unit_inputs", [])
        if isinstance(unit_input, dict)
    ]
    errors.extend(
        _duplicate_errors(binding_entity_keys, ["role_bindings"], "duplicate_entity_key")
    )
    errors.extend(_duplicate_errors(template_keys, ["templates"], "duplicate_template_key"))
    errors.extend(_duplicate_errors(unit_ids, ["units"], "duplicate_unit_id"))
    errors.extend(
        _duplicate_errors(
            unit_input_ids, ["unit_inputs"], "duplicate_unit_input_id"
        )
    )

    for role_id, binding in payload.get("role_bindings", {}).items():
        if not isinstance(binding, dict):
            continue
        entity_key = binding.get("entity_key")
        templates = [
            template
            for template in payload.get("templates", [])
            if isinstance(template, dict) and template.get("key") == entity_key
        ]
        if len(templates) != 1:
            errors.append(
                _error(
                    ["role_bindings", role_id, "entity_key"],
                    f"role binding entity must have exactly one template: {entity_key}",
                    "template_alignment",
                )
            )
        elif templates[0].get("role_ref") != role_id:
            errors.append(
                _error(
                    ["templates"],
                    f"template {entity_key} role_ref does not match binding role {role_id}",
                    "template_alignment",
                )
            )
        else:
            template = templates[0]
            if template.get("asset") != binding.get("asset"):
                errors.append(
                    _error(
                        ["templates"],
                        f"template {entity_key} asset does not match role binding",
                        "template_alignment",
                    )
                )
            if binding.get("model_id") not in template.get("model_candidates", []):
                errors.append(
                    _error(
                        ["templates"],
                        f"template {entity_key} does not contain bound model_id",
                        "template_alignment",
                    )
                )

    goal_variant_ids = [
        variant.get("id")
        for variant in payload.get("goal_variants", [])
        if isinstance(variant, dict)
    ]
    errors.extend(_duplicate_errors(goal_variant_ids, ["goal_variants"], "duplicate_goal_variant_id"))
    variant_set = {str(item) for item in goal_variant_ids}
    for index, constraint in enumerate(payload.get("layout_constraints", [])):
        if isinstance(constraint, dict):
            errors.extend(_constraint_errors(constraint, ["layout_constraints", index]))
            for role in constraint.get("roles", []):
                if role not in role_ids:
                    errors.append(
                        _error(["layout_constraints", index, "roles"], f"unknown role reference: {role}", "role_refs")
                    )
    for index, unit in enumerate(payload.get("units", [])):
        if not isinstance(unit, dict):
            continue
        for role in unit.get("roles", []):
            if role not in role_ids:
                errors.append(_error(["units", index, "roles"], f"unknown role reference: {role}", "role_refs"))
        for entity_key in unit.get("entity_keys", []):
            if entity_key not in entity_keys:
                errors.append(_error(["units", index, "entity_keys"], f"unknown entity key: {entity_key}", "entity_refs"))
        params = unit.get("params") or {}
        for key in ("landmark_key",):
            value = params.get(key)
            if value is not None and value not in entity_keys:
                errors.append(
                    _error(
                        ["units", index, "params", key],
                        f"unknown entity key: {value}",
                        "entity_refs",
                    )
                )
        for key in ("receiver_keys",):
            for value in params.get(key, []):
                if value not in entity_keys:
                    errors.append(
                        _error(
                            ["units", index, "params", key],
                            f"unknown entity key: {value}",
                            "entity_refs",
                        )
                    )
        for key in ("subject_keys",):
            for value in params.get(key, []):
                if value not in entity_keys:
                    errors.append(
                        _error(
                            ["units", index, "params", key],
                            f"unknown entity key: {value}",
                            "entity_refs",
                        )
                    )
        for key in (
            "internal_layout",
            "member_footprints",
            "receiver_layout",
            "receiver_footprints",
        ):
            for value in (params.get(key) or {}):
                if value not in entity_keys:
                    errors.append(
                        _error(
                            ["units", index, "params", key],
                            f"unknown entity key: {value}",
                            "entity_refs",
                        )
                    )
        region_ids = {
            region.get("id")
            for region in params.get("reserved_regions", [])
            if isinstance(region, dict)
        }
        for region_index, region in enumerate(params.get("reserved_regions", [])):
            if not isinstance(region, dict):
                continue
            for key in ("subject_key",):
                value = region.get(key)
                if value not in entity_keys:
                    errors.append(
                        _error(
                            ["units", index, "params", "reserved_regions", region_index, key],
                            f"unknown entity key: {value}",
                            "entity_refs",
                        )
                    )
            for value in region.get("receiver_keys", []):
                if value not in entity_keys:
                    errors.append(
                        _error(
                            ["units", index, "params", "reserved_regions", region_index, "receiver_keys"],
                            f"unknown entity key: {value}",
                            "entity_refs",
                        )
                    )
        for fact_index, fact in enumerate(params.get("guaranteed_facts", [])):
            if not isinstance(fact, dict):
                continue
            for key in ("subject", "reference"):
                value = fact.get(key)
                if value is not None and value not in entity_keys:
                    errors.append(
                        _error(
                            ["units", index, "params", "guaranteed_facts", fact_index, key],
                            f"unknown entity key: {value}",
                            "entity_refs",
                        )
                    )
            for value in fact.get("receivers", []):
                if value not in entity_keys:
                    errors.append(
                        _error(
                            ["units", index, "params", "guaranteed_facts", fact_index, "receivers"],
                            f"unknown entity key: {value}",
                            "entity_refs",
                        )
                    )
            region_id = fact.get("region_id")
            if region_id is not None and region_id not in region_ids:
                errors.append(
                    _error(
                        ["units", index, "params", "guaranteed_facts", fact_index, "region_id"],
                        f"unknown reserved region id: {region_id}",
                        "region_refs",
                    )
                )
    for index, unit_input in enumerate(payload.get("unit_inputs", [])):
        if not isinstance(unit_input, dict):
            continue
        errors.extend(
            _param_role_ref_errors(
                unit_input.get("params") or {},
                role_ids,
                ["unit_inputs", index, "params"],
            )
        )
        for role in unit_input.get("roles", []):
            if role not in role_ids:
                errors.append(
                    _error(
                        ["unit_inputs", index, "roles"],
                        f"unknown role reference: {role}",
                        "role_refs",
                    )
                )
        for role in (unit_input.get("geometry_refs") or {}):
            if role not in role_ids:
                errors.append(
                    _error(
                        ["unit_inputs", index, "geometry_refs"],
                        f"unknown role reference: {role}",
                        "role_refs",
                    )
                )
        for entity_key in unit_input.get("entity_keys", []):
            if entity_key not in entity_keys:
                errors.append(
                    _error(
                        ["unit_inputs", index, "entity_keys"],
                        f"unknown entity key: {entity_key}",
                        "entity_refs",
                    )
                )
    for index, variant in enumerate(payload.get("goal_variants", [])):
        if isinstance(variant, dict):
            if variant.get("assignment_id") != payload.get("assignment_id"):
                errors.append(
                    _error(
                        ["goal_variants", index, "assignment_id"],
                        "goal variant assignment_id does not match realized family",
                        "assignment_alignment",
                    )
                )
            errors.extend(
                _goal_variant_errors(
                    variant,
                    entity_keys,
                    ["goal_variants", index],
                    role_context=role_ids,
                )
            )
    for index, edge in enumerate(payload.get("minimal_pair_edges", [])):
        if not isinstance(edge, dict):
            continue
        for key in ("from", "to"):
            value = edge.get(key)
            if value not in variant_set:
                errors.append(_error(["minimal_pair_edges", index, key], f"unknown goal variant id: {value}", "variant_refs"))
    return errors


def _goal_variant_errors(
    variant: dict[str, Any],
    entity_context: set[str] | None,
    path: list[Any],
    *,
    role_context: set[str] | None = None,
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    role_refs = variant.get("role_refs") or {}
    entity_refs = variant.get("entity_refs") or {}
    if set(role_refs) != set(entity_refs):
        errors.append(
            _error(
                path + ["entity_refs"],
                "role_refs and entity_refs must expose the same slots",
                "goal_variant_slot_alignment",
            )
        )
    if role_context is not None:
        for slot, role in role_refs.items():
            if role not in role_context:
                errors.append(
                    _error(
                        path + ["role_refs", slot],
                        f"unknown role reference: {role}",
                        "variant_role_refs",
                    )
                )
    entity_keys = set(entity_context) if entity_context is not None else set(entity_refs.values())
    if entity_context is not None:
        for slot, entity in entity_refs.items():
            if entity not in entity_context:
                errors.append(_error(path + ["entity_refs", slot], f"unknown entity key: {entity}", "entity_refs"))
    goal = variant.get("goal", {})
    for slot in GOAL_REF_KEYS:
        goal_value = goal.get(slot)
        ref_value = entity_refs.get(slot)
        if goal_value is not None and ref_value != goal_value:
            errors.append(
                _error(
                    path + ["goal", slot],
                    f"goal {slot} must match entity_refs.{slot}",
                    "goal_variant_slot_alignment",
                )
            )
        if ref_value is not None and goal_value is None and slot in goal:
            errors.append(
                _error(
                    path + ["entity_refs", slot],
                    f"entity_refs.{slot} is present but goal {slot} is null",
                    "goal_variant_slot_alignment",
                )
            )
    errors.extend(_concrete_goal_errors(goal, entity_keys, path + ["goal"]))
    return errors


def _scene_spec_errors(payload: dict[str, Any]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    entity_keys = set(payload.get("entities", {}))
    unit_ids = [
        row.get("unit_id")
        for row in payload.get("unit_poses", [])
        if isinstance(row, dict)
    ]
    errors.extend(_duplicate_errors(unit_ids, ["unit_poses"], "duplicate_unit_pose"))
    errors.extend(
        _duplicate_errors(
            list(payload.get("goal_variant_ids", [])),
            ["goal_variant_ids"],
            "duplicate_goal_variant_id",
        )
    )
    for entity_key, entity in payload.get("entities", {}).items():
        if not isinstance(entity, dict):
            continue
        if entity_key not in entity_keys:
            errors.append(
                _error(
                    ["entities", entity_key],
                    f"invalid scene entity key: {entity_key}",
                    "entity_refs",
                )
            )
    return errors


def _task_record_errors(payload: dict[str, Any]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    provenance = payload.get("goal_provenance") or {}
    entity_keys = {
        str(value) for value in provenance.get("scene_entity_keys") or []
    }
    errors.extend(
        _concrete_goal_errors(
            payload.get("goal") or {},
            entity_keys,
            ["task_record", "goal"],
        )
    )
    candidates = (payload.get("instruction") or {}).get("candidates") or []
    template_ids = [
        row.get("template_id") for row in candidates if isinstance(row, dict)
    ]
    style_ids = [
        row.get("style_id") for row in candidates if isinstance(row, dict)
    ]
    texts = [row.get("text") for row in candidates if isinstance(row, dict)]
    errors.extend(
        _duplicate_errors(
            template_ids,
            ["task_record", "instruction", "candidates"],
            "duplicate_instruction_template_id",
        )
    )
    errors.extend(
        _duplicate_errors(
            style_ids,
            ["task_record", "instruction", "candidates"],
            "duplicate_instruction_style_id",
        )
    )
    errors.extend(
        _duplicate_errors(
            texts,
            ["task_record", "instruction", "candidates"],
            "duplicate_instruction_text",
        )
    )
    selected = (payload.get("instruction") or {}).get("selected")
    if selected not in candidates:
        errors.append(
            _error(
                ["task_record", "instruction", "selected"],
                "selected instruction must be a member of candidates",
                "instruction_selection",
            )
        )
    selection = (payload.get("instruction") or {}).get("selection") or {}
    if isinstance(selected, dict) and selected.get("style_id") != selection.get(
        "selected_style_id"
    ):
        errors.append(
            _error(
                ["task_record", "instruction", "selection", "selected_style_id"],
                "selected style id must match selected instruction",
                "instruction_selection",
            )
        )
    return errors


def _minimal_pair_errors(payload: dict[str, Any]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    left = payload.get("from_goal_variant_id")
    right = payload.get("to_goal_variant_id")
    if left == right:
        errors.append(
            _error(
                ["minimal_pair", "to_goal_variant_id"],
                "minimal-pair endpoints must differ",
                "minimal_pair_endpoints",
            )
        )
    expected_dcode = (
        f"{payload.get('intended_axis')}.{payload.get('intended_subtype')}"
    )
    if payload.get("intended_dcode") != expected_dcode:
        errors.append(
            _error(
                ["minimal_pair", "intended_dcode"],
                "intended D-code must equal axis.subtype",
                "minimal_pair_dcode",
            )
        )
    if payload.get("actual_dcodes") != [payload.get("intended_dcode")]:
        errors.append(
            _error(
                ["minimal_pair", "actual_dcodes"],
                "valid minimal pair must have exactly its intended D-code",
                "minimal_pair_dcode",
            )
        )
    if payload.get("actual_axes") != [payload.get("intended_axis")]:
        errors.append(
            _error(
                ["minimal_pair", "actual_axes"],
                "valid minimal pair must have exactly its intended axis",
                "minimal_pair_axis",
            )
        )
    if payload.get("unmapped_semantic_deltas"):
        errors.append(
            _error(
                ["minimal_pair", "unmapped_semantic_deltas"],
                "valid minimal pair cannot contain unmapped semantic deltas",
                "minimal_pair_semantics",
            )
        )
    for endpoint_key, variant_key in (
        ("from_endpoint", "from_goal_variant_id"),
        ("to_endpoint", "to_goal_variant_id"),
    ):
        endpoint = payload.get(endpoint_key) or {}
        if endpoint.get("goal_variant_id") != payload.get(variant_key):
            errors.append(
                _error(
                    ["minimal_pair", endpoint_key, "goal_variant_id"],
                    "endpoint goal variant does not match pair endpoint",
                    "minimal_pair_endpoints",
                )
            )
    left_style = (payload.get("from_endpoint") or {}).get("style_id")
    right_style = (payload.get("to_endpoint") or {}).get("style_id")
    if left_style != right_style:
        errors.append(
            _error(
                ["minimal_pair", "to_endpoint", "style_id"],
                "minimal-pair endpoints must share instruction style",
                "minimal_pair_instruction_style",
            )
        )
    return errors


def _axis_contrast_set_errors(payload: dict[str, Any]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    expected_dcode = f"{payload.get('axis')}.{payload.get('subtype')}"
    if payload.get("dcode") != expected_dcode:
        errors.append(
            _error(
                ["axis_contrast_set", "dcode"],
                "contrast-set D-code must equal axis.subtype",
                "contrast_set_dcode",
            )
        )
    members = payload.get("members") or []
    k = payload.get("k")
    if len(members) != k:
        errors.append(
            _error(
                ["axis_contrast_set", "members"],
                "contrast-set member count must equal K",
                "contrast_set_size",
            )
        )
    indices = [
        row.get("matrix_index") for row in members if isinstance(row, dict)
    ]
    if indices != list(range(len(members))):
        errors.append(
            _error(
                ["axis_contrast_set", "members"],
                "matrix indices must be contiguous and ordered from zero",
                "contrast_set_order",
            )
        )
    variant_ids = [
        row.get("goal_variant_id") for row in members if isinstance(row, dict)
    ]
    errors.extend(
        _duplicate_errors(
            variant_ids,
            ["axis_contrast_set", "members"],
            "duplicate_contrast_set_member",
        )
    )
    pair_ids = list(payload.get("pair_ids") or [])
    errors.extend(
        _duplicate_errors(
            pair_ids,
            ["axis_contrast_set", "pair_ids"],
            "duplicate_contrast_set_pair",
        )
    )
    if payload.get("matrix_ready") and isinstance(k, int):
        expected_pair_count = k * (k - 1) // 2
        if len(pair_ids) != expected_pair_count:
            errors.append(
                _error(
                    ["axis_contrast_set", "pair_ids"],
                    "matrix-ready contrast set must reference a complete clique",
                    "contrast_set_clique",
                )
            )
        required_endpoint_keys = {
            "task_record_id",
            "task_artifact_hash",
            "goal_hash",
            "instruction_hash",
            "checker_hash",
            "style_id",
        }
        for index, member in enumerate(members):
            if isinstance(member, dict) and not required_endpoint_keys <= set(
                member
            ):
                errors.append(
                    _error(
                        ["axis_contrast_set", "members", index],
                        "matrix-ready member is missing task endpoint evidence",
                        "contrast_set_endpoint",
                    )
                )
        styles = {
            row.get("style_id") for row in members if isinstance(row, dict)
        }
        if len(styles) != 1:
            errors.append(
                _error(
                    ["axis_contrast_set", "members"],
                    "matrix-ready members must share instruction style",
                    "contrast_set_instruction_style",
                )
            )
    return errors


def _concrete_goal_errors(goal: dict[str, Any], entity_keys: set[str], path: list[Any]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for key in GOAL_REF_KEYS:
        value = goal.get(key)
        if value is not None and value not in entity_keys:
            errors.append(_error(path + [key], f"goal reference is not a concrete entity key in context: {value}", "goal_entity_refs"))
    descriptors = goal.get("descriptors") or {}
    for slot, bundle in descriptors.items():
        if not isinstance(bundle, dict):
            continue
        spatial = bundle.get("spatial")
        if isinstance(spatial, dict):
            reference = spatial.get("reference")
            if reference is not None and reference not in entity_keys:
                errors.append(
                    _error(path + ["descriptors", slot, "spatial", "reference"], f"unknown entity key: {reference}", "goal_entity_refs")
                )
        ordinal = bundle.get("ordinal")
        if isinstance(ordinal, dict):
            reference = ordinal.get("reference")
            if reference is not None and reference not in entity_keys:
                errors.append(
                    _error(path + ["descriptors", slot, "ordinal", "reference"], f"unknown entity key: {reference}", "goal_entity_refs")
                )
    return errors


def _legacy_variant_boundary_errors(payload: Any, path: list[Any] | None = None) -> list[dict[str, Any]]:
    path = [] if path is None else path
    errors: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            next_path = path + [key]
            if key == "variant" and (len(path) == 0 or path[-1] != "export"):
                errors.append(_error(next_path, "legacy variant string is only allowed at export.variant", "legacy_variant_boundary"))
            errors.extend(_legacy_variant_boundary_errors(value, next_path))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            errors.extend(_legacy_variant_boundary_errors(value, path + [index]))
    return errors


def _collect_constraints(payload: Any) -> list[dict[str, Any]]:
    constraints: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        for key in ("constraints", "layout_constraints"):
            value = payload.get(key)
            if isinstance(value, list):
                constraints.extend(item for item in value if isinstance(item, dict))
        for value in payload.values():
            if isinstance(value, (dict, list)):
                constraints.extend(_collect_constraints(value))
    elif isinstance(payload, list):
        for item in payload:
            constraints.extend(_collect_constraints(item))
    return constraints


def _collect_descriptor_kinds(payload: Any) -> set[str]:
    kinds: set[str] = set()
    if isinstance(payload, dict):
        constructor_kind = payload.get("kind")
        if constructor_kind in {"attribute", "spatial", "ordinal", "part"}:
            kinds.add(str(constructor_kind))
        for key in ("attribute", "spatial", "ordinal", "part"):
            if key in payload and isinstance(payload[key], dict):
                kinds.add(key)
        for value in payload.values():
            kinds.update(_collect_descriptor_kinds(value))
    elif isinstance(payload, list):
        for item in payload:
            kinds.update(_collect_descriptor_kinds(item))
    return kinds


def _collect_unit_types(payload: Any) -> set[str]:
    unit_types: set[str] = set()
    if isinstance(payload, dict):
        if payload.get("type") in {
            "ObjectUnit",
            "LandmarkRelationUnit",
            "ReceiverReservationUnit",
        }:
            unit_types.add(str(payload["type"]))
        for value in payload.values():
            unit_types.update(_collect_unit_types(value))
    elif isinstance(payload, list):
        for item in payload:
            unit_types.update(_collect_unit_types(item))
    return unit_types


def _duplicate_errors(values: list[Any], path: list[Any], rule: str) -> list[dict[str, Any]]:
    counts = Counter(value for value in values if value is not None)
    return [
        _error(path, f"duplicate id: {value}", rule)
        for value, count in sorted(counts.items())
        if count > 1
    ]


def _normalize_document(document: Any, kind: str | None = None) -> dict[str, Any]:
    if kind is not None and kind not in DOCUMENT_KINDS:
        raise ValueError(f"unknown BenchmarkV1 document kind: {kind}")
    if not isinstance(document, dict) and kind != "axis_plan":
        raise ValueError("BenchmarkV1 document must be a mapping")
    if kind is None:
        if not isinstance(document, dict):
            raise ValueError("BenchmarkV1 document kind could not be inferred")
        doc_kind = document.get("kind")
        if doc_kind not in DOCUMENT_KINDS:
            raise ValueError("BenchmarkV1 document must include a valid kind")
        payload_key = KIND_TO_PAYLOAD[str(doc_kind)]
        if payload_key not in document:
            raise ValueError(f"BenchmarkV1 document is missing payload key: {payload_key}")
        return document

    payload_key = KIND_TO_PAYLOAD[kind]
    if isinstance(document, dict) and document.get("kind") == kind and payload_key in document:
        return document
    if isinstance(document, dict) and "kind" in document and document.get("kind") != kind:
        raise ValueError(f"document kind {document.get('kind')} does not match requested kind {kind}")
    return {"schema_version": SCHEMA_VERSION, "kind": kind, payload_key: document}


def _document_kind_payload(document: dict[str, Any]) -> tuple[str, Any]:
    kind = str(document["kind"])
    return kind, document[KIND_TO_PAYLOAD[kind]]


def _load_examples(examples_dir: Path) -> dict[str, dict[str, Any]]:
    return load_artifact_examples(examples_dir)


def _load_yaml_required(path: Path) -> dict[str, Any]:
    data = _load_yaml(path)
    if data is None:
        raise FileNotFoundError(path)
    return data


def _load_yaml(path: Path) -> dict[str, Any] | None:
    return load_artifact_yaml(path)


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    write_artifact_yaml(path, data)


def _artifact_hash(data: Any, hash_key: str | None) -> str:
    omitted = (hash_key,) if hash_key is not None else ()
    return public_artifact_hash(data, omit_keys=omitted)


def _canonical_json(data: Any) -> str:
    return public_canonical_json(data)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _error(path: list[Any], message: str, rule: str) -> dict[str, Any]:
    return {
        "path": _path_string_from_parts(path),
        "message": message,
        "rule": rule,
    }


def _path_string_from_parts(parts: list[Any]) -> str:
    if not parts:
        return "$"
    return "$." + ".".join(str(part) for part in parts)


def _path_string(path: Path) -> str:
    return str(path).replace("\\", "/")


def _check_status_line(report: dict[str, Any]) -> str:
    if report.get("stale"):
        missing = report.get("missing_artifacts", [])
        extras = report.get("extra_examples", [])
        stale = report.get("stale_examples", [])
        parts = []
        if missing:
            parts.append(f"missing {len(missing)}")
        if extras:
            parts.append(f"extra examples {len(extras)}")
        if stale:
            parts.append(f"stale examples {len(stale)}")
        suffix = f" ({', '.join(parts)})" if parts else ""
        return f"check: BenchmarkV1 schema artifacts are stale{suffix}"
    return "check: BenchmarkV1 schema artifacts are current"


if __name__ == "__main__":
    raise SystemExit(main())
