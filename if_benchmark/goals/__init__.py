"""Milestone 9 goal realization, legacy codec, and frozen artifacts."""

from __future__ import annotations

from .core import (
    GoalMaterializationError,
    canonical_action_family,
    canonical_relation_value,
    canonical_state_value,
    check_goal_artifacts,
    decode_legacy_variant,
    encode_legacy_variant,
    freeze_goal_artifacts,
    materialize_assignment_goals,
    materialize_goal_variant,
    validate_goal_artifacts,
)

__all__ = [
    "GoalMaterializationError",
    "canonical_action_family",
    "canonical_relation_value",
    "canonical_state_value",
    "check_goal_artifacts",
    "decode_legacy_variant",
    "encode_legacy_variant",
    "freeze_goal_artifacts",
    "materialize_assignment_goals",
    "materialize_goal_variant",
    "validate_goal_artifacts",
]
