"""Canonical BenchmarkV1 document registry and contract constants."""

from __future__ import annotations

from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
DEFAULT_SCHEMA_PATH = Path("if_benchmark/schema/benchmark_v1.schema.yml")
DEFAULT_EXAMPLES_DIR = Path("if_benchmark/schema/examples")
DEFAULT_AUDIT_PATH = Path("if_benchmark/data/schema/schema_audit_report.yml")

DOCUMENT_KINDS = (
    "abstract_family",
    "axis_plan",
    "archetype",
    "role_graph",
    "constraint",
    "variant_schema",
    "assignment_candidate",
    "realized_scene_family",
    "scene_spec",
    "concrete_goal",
    "goal_variant",
    "task_record",
    "minimal_pair",
    "axis_contrast_set",
    "scene_validation_record",
)

KIND_TO_PAYLOAD: dict[str, str] = {kind: kind for kind in DOCUMENT_KINDS}

KIND_TO_REF: dict[str, str] = {
    "abstract_family": "#/$defs/AbstractSceneFamily",
    "axis_plan": "#/$defs/AxisPlan",
    "archetype": "#/$defs/Archetype",
    "role_graph": "#/$defs/RoleGraph",
    "constraint": "#/$defs/Constraint",
    "variant_schema": "#/$defs/VariantSchema",
    "assignment_candidate": "#/$defs/AssignmentCandidate",
    "realized_scene_family": "#/$defs/RealizedSceneFamily",
    "scene_spec": "#/$defs/SceneSpec",
    "concrete_goal": "#/$defs/ConcreteGoalDocument",
    "goal_variant": "#/$defs/GoalVariant",
    "task_record": "#/$defs/TaskRecord",
    "minimal_pair": "#/$defs/ValidatedMinimalPair",
    "axis_contrast_set": "#/$defs/AxisContrastSet",
    "scene_validation_record": "#/$defs/SceneValidationRecord",
}

CONSTRAINT_REQUIRED_KEYS = ("id", "type", "scope", "phase", "roles", "params", "hard")
CONSTRAINT_ALLOWED_KEYS = set(CONSTRAINT_REQUIRED_KEYS) | {"related_constraint"}
ROLE_REF_PARAM_KEYS = {
    "role",
    "roles",
    "source_role",
    "reference_role",
    "subject_roles",
    "receiver_roles",
    "peers",
}
GOAL_REF_KEYS = ("subject", "target", "subject2", "target2")

CONFIG: dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "document_kinds": list(DOCUMENT_KINDS),
    "constraint_required_keys": list(CONSTRAINT_REQUIRED_KEYS),
    "constraint_scopes": ["semantic", "asset", "unit", "layout", "gate"],
    "constraint_phases": ["assignment", "unit_build", "layout", "gate", "manifest"],
    "schema_dialect": "https://json-schema.org/draft/2020-12/schema",
    "legacy_variant_boundary": "export.variant only",
    "migration_helpers": ["migrate_document"],
}
