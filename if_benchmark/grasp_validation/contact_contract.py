"""Pure R02 Contact contract identity.

This module deliberately has no simulator, motion-planner, or quaternion
conversion dependency.  Milestones 4 and 11 both import it so the Contact
projection that was physically validated is exactly the projection embedded
in a Unit operation-anchor contract.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from if_benchmark.schema import artifact_hash, deterministic_id


EXACT_GRASP_POLICY_VERSION = "contact_constrained_support_projection_v1"

CONTACT_TO_GRASP_FRAME = np.asarray(
    [
        [0.0, 0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=float,
)

EXACT_GRASP_POLICY: dict[str, Any] = {
    "version": EXACT_GRASP_POLICY_VERSION,
    "candidate_semantics": (
        "authored_orientation_with_minimum_support_safe_translation"
    ),
    "contact_to_grasp_frame": CONTACT_TO_GRASP_FRAME.tolist(),
    "gripper_tcp_offset_m": 0.12,
    "retreat_distances_m": [0.08, 0.06, 0.04, 0.02],
    "retreat_variant_selection": "largest_reachable_clearance_first",
    "motion_stages": ["high_alignment", "fixed_orientation_approach", "close"],
    "terminal_execution": {
        "position_tolerance_m": 0.008,
        "orientation_tolerance_deg": 5.0,
        "maximum_actor_translation_before_close_m": 0.005,
        "maximum_actor_rotation_before_close_deg": 3.0,
        "settle_steps": 125,
    },
    "minimum_normalized_joint_margin": 0.10,
    "maximum_physical_contact_attempts": 12,
    "table_approach_thresholds": {
        "top_down_max_z": -0.85,
        "oblique_down_max_z": -0.30,
        "horizontal_max_abs_z": 0.30,
        "reject_upward_min_z": 0.30,
    },
    "table_direction_order": ["top_down", "oblique_down", "horizontal"],
    "single_arm_pick_contact_scope": "all_authored_contacts",
    "grouped_use_contact_scope": "explicit_contact_group",
    "synthetic_pose_generation": ["minimum_world_z_support_projection"],
    "table_gripper_geometry": {
        "embodiment": "aloha-agilex",
        "source": (
            "reviewed union of link6/link7/link8 collision-mesh bounds in "
            "the configured move-group ee_link frame"
        ),
        "collision_envelope_min_xyz_m": [-0.001, -0.087, -0.030396484],
        "collision_envelope_max_xyz_m": [0.155570003, 0.087, 0.060804807],
        "inner_pad_x_interval_m": [0.069570001, 0.155570003],
        "inner_pad_z_interval_m": [-0.030396484, 0.030603519],
        "support_buffer_m": 0.0,
    },
    "ranking": [
        "approach_class",
        "normalized_total_travel",
        "maximum_normalized_joint_step",
        "negative_minimum_joint_limit_margin",
        "cartesian_path_length_m",
        "contact_id",
        "contact_group_indices",
        "retreat_distance_m",
    ],
}


class GraspContractError(ValueError):
    """Raised when an authored Contact or R02 evidence violates the contract."""


def _clean_float(value: float) -> float:
    rounded = round(float(value), 12)
    return 0.0 if rounded == 0.0 else rounded


def clean_vector(values: Sequence[float]) -> list[float]:
    return [_clean_float(value) for value in values]


def clean_matrix(values: Sequence[Sequence[float]]) -> list[list[float]]:
    matrix = np.asarray(values, dtype=float)
    return [clean_vector(row) for row in matrix]


def validate_contact_matrix(
    matrix: Sequence[Sequence[float]],
    *,
    rotation_tolerance: float = 5e-3,
) -> np.ndarray:
    """Validate, but never reinterpret or orthonormalise, one Contact matrix."""

    value = np.asarray(matrix, dtype=float)
    if value.shape != (4, 4):
        raise GraspContractError(
            f"Contact matrix must have shape (4, 4), got {value.shape}"
        )
    if not np.isfinite(value).all():
        raise GraspContractError("Contact matrix contains non-finite values")
    if not np.allclose(
        value[3], np.asarray([0.0, 0.0, 0.0, 1.0]), atol=1e-8
    ):
        raise GraspContractError("Contact matrix has an invalid homogeneous row")
    rotation = value[:3, :3]
    if not np.allclose(
        rotation.T @ rotation,
        np.eye(3),
        atol=float(rotation_tolerance),
    ):
        raise GraspContractError("Contact rotation is not orthonormal")
    determinant = float(np.linalg.det(rotation))
    if not math.isclose(
        determinant, 1.0, abs_tol=float(rotation_tolerance)
    ):
        raise GraspContractError(
            f"Contact rotation determinant must be +1, got {determinant}"
        )
    return value.copy()


def contact_groups_for_ids(
    contact_count: int,
    groups: Sequence[Sequence[int]] = (),
    *,
    strict: bool = True,
) -> dict[int, list[int]]:
    """Return group provenance for every Contact without inventing membership."""

    count = int(contact_count)
    if count < 0:
        raise GraspContractError("contact_count must be non-negative")
    memberships: dict[int, list[int]] = {index: [] for index in range(count)}
    for group_index, group in enumerate(groups):
        seen: set[int] = set()
        for raw_contact_id in group:
            contact_id = int(raw_contact_id)
            if contact_id < 0 or contact_id >= count:
                if strict:
                    raise GraspContractError(
                        f"Contact Group {group_index} contains invalid id {contact_id}"
                    )
                continue
            if contact_id in seen:
                if strict:
                    raise GraspContractError(
                        f"Contact Group {group_index} repeats id {contact_id}"
                    )
                continue
            seen.add(contact_id)
            memberships[contact_id].append(group_index)
    return memberships


def allowed_contact_ids(
    contact_count: int,
    *,
    use_profile: str,
    groups: Sequence[Sequence[int]] = (),
    group_index: int | None = None,
) -> list[int]:
    """Resolve the explicit single-arm or grouped-use Contact scope."""

    count = max(int(contact_count), 0)
    if use_profile == "single_arm_pick":
        return list(range(count))
    if use_profile not in {"handover", "bimanual", "grouped_regrasp"}:
        raise GraspContractError(f"unsupported grasp use profile: {use_profile}")
    if group_index is None or group_index < 0 or group_index >= len(groups):
        raise GraspContractError(
            f"{use_profile} requires an existing Contact Group index"
        )
    result = []
    for raw_id in groups[group_index]:
        contact_id = int(raw_id)
        if contact_id < 0 or contact_id >= count:
            raise GraspContractError(
                f"Contact Group {group_index} contains invalid id {contact_id}"
            )
        if contact_id not in result:
            result.append(contact_id)
    if not result:
        raise GraspContractError(
            f"{use_profile} Contact Group {group_index} is empty"
        )
    return result


def build_contact_projection(
    *,
    contact_matrices: Sequence[Sequence[Sequence[float]]],
    contact_groups: Sequence[Sequence[int]] = (),
) -> dict[str, Any]:
    """Return the Contact-only source projection shared by M4 and M11."""

    matrices = [
        clean_matrix(validate_contact_matrix(matrix))
        for matrix in contact_matrices
    ]
    normalized_groups = [
        [int(value) for value in group] for group in contact_groups
    ]
    memberships = contact_groups_for_ids(
        len(matrices), normalized_groups, strict=False
    )
    group_issues = []
    for group_index, group in enumerate(normalized_groups):
        seen: set[int] = set()
        for contact_id in group:
            if contact_id < 0 or contact_id >= len(matrices):
                group_issues.append(
                    {
                        "group_index": group_index,
                        "contact_id": contact_id,
                        "reason": "contact_id_out_of_range",
                    }
                )
            elif contact_id in seen:
                group_issues.append(
                    {
                        "group_index": group_index,
                        "contact_id": contact_id,
                        "reason": "duplicate_contact_id",
                    }
                )
            seen.add(contact_id)
    payload = {
        "contact_matrices": matrices,
        "contact_groups": normalized_groups,
        "contact_group_issues": group_issues,
        "contact_group_memberships": {
            str(contact_id): indices
            for contact_id, indices in sorted(memberships.items())
        },
    }
    return {
        **payload,
        "contact_projection_hash": artifact_hash(payload),
    }


def build_contact_contract(
    *,
    asset: str,
    model_id: str,
    use: str,
    contact_matrices: Sequence[Sequence[Sequence[float]]],
    contact_groups: Sequence[Sequence[int]] = (),
    use_profile: str = "single_arm_pick",
) -> dict[str, Any]:
    """Freeze the Contact-only identity independently from R01 geometry."""

    projection = build_contact_projection(
        contact_matrices=contact_matrices,
        contact_groups=contact_groups,
    )
    matrices = projection["contact_matrices"]
    normalized_groups = projection["contact_groups"]
    strict_groups = use_profile != "single_arm_pick"
    if strict_groups:
        contact_groups_for_ids(
            len(matrices), normalized_groups, strict=True
        )
    payload = {
        "policy_version": EXACT_GRASP_POLICY_VERSION,
        "asset": str(asset),
        "model_id": str(model_id),
        "use": str(use),
        "use_profile": str(use_profile),
        "contact_matrices": matrices,
        "contact_groups": normalized_groups,
        "contact_group_issues": projection["contact_group_issues"],
        "contact_group_memberships": projection["contact_group_memberships"],
        "policy": EXACT_GRASP_POLICY,
    }
    contract_hash = artifact_hash(payload)
    return {
        **payload,
        "contact_contract_id": deterministic_id("gcc", payload, length=20),
        "contact_contract_hash": contract_hash,
        "contact_projection_hash": projection["contact_projection_hash"],
    }


def resolved_contact_preflight_policy_version(
    observed: Mapping[str, Any],
    binding: Mapping[str, Any],
) -> str | None:
    """Resolve the explicit policy or its exact contract-hash commitment.

    Early schema-v3 M4 workers validated policy equality before emitting the
    runtime preflight but accidentally omitted only ``policy_version`` from
    that diagnostic.  The exact contract hash commits policy version, asset,
    model, use, matrices, groups, profile, and policy.  Therefore the missing
    redundant field is recoverable only when every independent Contact
    identity agrees with the source binding.
    """

    explicit = observed.get("policy_version")
    if explicit is not None:
        return str(explicit)
    expected = binding.get("policy_version")
    if expected != EXACT_GRASP_POLICY_VERSION:
        return None
    identity_fields = (
        "contact_contract_id",
        "contact_contract_hash",
        "contact_projection_hash",
        "candidate_use",
        "use_profile",
        "contact_count",
        "contact_groups",
    )
    if any(observed.get(field) != binding.get(field) for field in identity_fields):
        return None
    return str(expected)
