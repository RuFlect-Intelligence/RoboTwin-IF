"""Dependency-light identities for shared M4/M15 physical execution."""

from __future__ import annotations

from typing import Any, Mapping

from if_benchmark.asset_validation_contract import M4_EXECUTABLE_USES
from if_benchmark.schema import artifact_hash


DYNAMIC_ASSET_MASS_INERTIA_POLICY_VERSION = (
    "dynamic_asset_mass_inertia_v2_explicit_override_or_geometry_auto"
)
UNIFIED_PICK_EXECUTION_VERSION = "m4_m15_compiled_pick_acquisition_v8"
UNIFIED_HELD_LIFT_VERSION = "m4_m15_single_motion_held_lift_v4"
UNIFIED_HELD_POSE_VERSION = (
    "m4_m15_verified_held_actor_pose_v2_dual_ee_frame"
)
UNIFIED_ARTICULATION_EXECUTION_VERSION = (
    "m4_m15_continuous_articulation_motion_v6_contact_component_frame"
)
HOLD_EE_FRAME_ARM = "arm_ee"
HOLD_PLANNING_EE_FRAME_ARM = "planner_arm_ee"
# Shared collection/inference actuator and manipuland settings.
HELD_MOTION_ARM_DRIVE_STIFFNESS = 5000.0
HELD_MOTION_ARM_DRIVE_DAMPING = 150.0
PICK_MANIPULAND_LINEAR_DAMPING = 0.1
PICK_MANIPULAND_ANGULAR_DAMPING = 0.5
PICK_REPLAY_ARM_DRIVE_STIFFNESS = 5000.0
PICK_REPLAY_ARM_DRIVE_DAMPING = 150.0
GRIPPER_PLAN_STEPS = 200
GRIPPER_TERMINAL_HOLD_RATIO = 0.5
M4_WORKER_EXECUTION_POLICY_MANIFEST_VERSION = (
    "m4_worker_execution_policy_manifest_v3"
)
def m4_worker_execution_policy_manifest() -> dict[str, str]:
    """Return the exact policy identity emitted by the physical worker."""

    manifest = {
        "manifest_version": M4_WORKER_EXECUTION_POLICY_MANIFEST_VERSION,
        "dynamic_asset_mass_inertia_policy_version": (
            DYNAMIC_ASSET_MASS_INERTIA_POLICY_VERSION
        ),
        "unified_pick_execution_version": UNIFIED_PICK_EXECUTION_VERSION,
        "unified_held_lift_version": UNIFIED_HELD_LIFT_VERSION,
        "unified_held_pose_version": UNIFIED_HELD_POSE_VERSION,
        "unified_articulation_execution_version": (
            UNIFIED_ARTICULATION_EXECUTION_VERSION
        ),
        "physical_hold_ee_frame": HOLD_EE_FRAME_ARM,
        "planning_ee_frame": HOLD_PLANNING_EE_FRAME_ARM,
    }
    manifest["manifest_hash"] = artifact_hash(manifest)
    return manifest


def m4_accepted_worker_execution_policy_manifests(
    *, use: str | None = None
) -> tuple[dict[str, str], ...]:
    """Return the current versionless worker contract.

    ``use`` remains accepted for API compatibility.  Historical global
    v18/v19 labels are deliberately outside the admission contract.
    """

    del use
    return (m4_worker_execution_policy_manifest(),)


def m4_worker_execution_policy_manifest_errors(
    value: Any,
    *,
    use: str | None = None,
) -> list[str]:
    """Validate worker-owned policy evidence without repairing it."""

    if not isinstance(value, Mapping):
        return ["worker_execution_policy_manifest_missing"]
    manifest = dict(value)
    expected = m4_worker_execution_policy_manifest()
    failures: list[str] = []
    if manifest.get("manifest_hash") != artifact_hash(
        manifest, omit_keys=("manifest_hash",)
    ):
        failures.append("worker_execution_policy_manifest_hash_mismatch")
    # Global skill-policy labels were bookkeeping only and are no longer
    # recorded or used for admission.  The short-lived v19 label also changed
    # the shared Pick component identity even for non-bimanual rows; ignore
    # that historical field for every currently executable non-bimanual use.
    ignored = {"manifest_hash", "skill_execution_policy_version"}
    if use in M4_EXECUTABLE_USES:
        ignored.add("unified_pick_execution_version")
    observed_contract = {
        key: item for key, item in manifest.items() if key not in ignored
    }
    expected_contract = {
        key: item for key, item in expected.items() if key not in ignored
    }
    if observed_contract == expected_contract:
        return sorted(set(failures))
    if set(observed_contract) != set(expected_contract):
        failures.append("worker_execution_policy_manifest_fields_mismatch")
    for field, expected_value in expected_contract.items():
        if observed_contract.get(field) != expected_value:
            failures.append(f"worker_execution_policy_{field}_mismatch")
    return sorted(set(failures))


__all__ = [
    "DYNAMIC_ASSET_MASS_INERTIA_POLICY_VERSION",
    "HOLD_EE_FRAME_ARM",
    "HOLD_PLANNING_EE_FRAME_ARM",
    "M4_WORKER_EXECUTION_POLICY_MANIFEST_VERSION",
    "UNIFIED_HELD_LIFT_VERSION",
    "UNIFIED_HELD_POSE_VERSION",
    "UNIFIED_PICK_EXECUTION_VERSION",
    "UNIFIED_ARTICULATION_EXECUTION_VERSION",
    "m4_accepted_worker_execution_policy_manifests",
    "m4_worker_execution_policy_manifest",
    "m4_worker_execution_policy_manifest_errors",
]
