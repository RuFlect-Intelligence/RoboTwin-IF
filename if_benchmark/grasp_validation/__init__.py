"""R02 authored-Contact-constrained grasp validation and runtime policy.

Exports are resolved lazily so CPU-only artifact builders can import the pure
Contact identity module without importing runtime quaternion dependencies.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "APPROACH_CLASS_ORDER",
    "CONTACT_TO_GRASP_FRAME",
    "EXACT_GRASP_POLICY",
    "EXACT_GRASP_POLICY_VERSION",
    "GraspContractError",
    "build_contact_contract",
    "classify_approach",
    "contact_groups_for_ids",
    "exact_grasp_poses",
    "grasp_candidate_sort_key",
    "joint_path_metrics",
    "scaled_contact_world_matrix",
    "select_best_retreat_per_contact",
    "support_safe_grasp_poses",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)
    from . import core

    return getattr(core, name)
