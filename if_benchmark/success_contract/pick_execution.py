"""Pure R03 decisions shared by the M4 and M15 simulator adapters."""

from __future__ import annotations

import math
from typing import Any, Mapping

from .contract import SuccessContract


PICK_ACQUISITION_POLICY_VERSION = (
    "continuous_world_positive_z_post_close_hard_d6_tight_clearance_v7"
)
PICK_ACQUISITION_TRACKING_GUARD_M = 0.01
PICK_ACQUISITION_REQUIRED_CLEARANCE_M = float(
    SuccessContract.load().section("hold")["pick_support_clearance_m"]
)


def hold_drift_policy(*, phase: str, contract: SuccessContract) -> dict[str, Any]:
    """Resolve phase-specific hold drift semantics from the shared contract."""

    phase_name = str(phase)
    thresholds = contract.section("hold")
    if phase_name == "pick":
        return {
            "phase": "pick",
            "translation_limit_m": float(
                thresholds["pick_relative_translation_drift_exclusive_max_m"]
            ),
            "translation_limit_inclusive": False,
            "rotation_constraint_enabled": bool(
                thresholds["pick_relative_rotation_constraint_enabled"]
            ),
            "rotation_limit_deg": None,
        }
    if phase_name == "carry":
        return {
            "phase": "carry",
            "translation_limit_m": float(thresholds["relative_translation_drift_m"]),
            "translation_limit_inclusive": True,
            "rotation_constraint_enabled": True,
            "rotation_limit_deg": float(thresholds["relative_rotation_drift_deg"]),
        }
    raise ValueError(f"unsupported hold phase: {phase_name}")


def pick_attachment_policy(*, contract: SuccessContract) -> dict[str, Any]:
    """Resolve the physical post-close attachment policy for Pick."""

    thresholds = contract.section("hold")
    return {
        "mode": str(thresholds["pick_post_close_attachment"]),
        "drive_stiffness": float(thresholds["pick_attachment_drive_stiffness"]),
        "drive_damping": float(thresholds["pick_attachment_drive_damping"]),
        "residual_translation_m": float(
            thresholds["pick_attachment_residual_translation_exclusive_max_m"]
        ),
        "translation_limit_inclusive": False,
        "rotation_constraint_enabled": bool(
            thresholds["pick_attachment_rotation_constraint_enabled"]
        ),
        "residual_rotation_deg": None,
    }


def evaluate_pick_attachment(
    metrics: Mapping[str, Any], *, contract: SuccessContract
) -> dict[str, Any]:
    """Fail closed when the live fixed constraint is absent or drifting."""

    policy = pick_attachment_policy(contract=contract)
    active = bool(metrics.get("attachment_active"))
    translation = float(metrics.get("constraint_translation_residual_m", math.inf))
    checks = {
        "attachment_active": active,
        "translation_residual_within_limit": (
            translation < float(policy["residual_translation_m"])
        ),
        "rotation_constraint_disabled": not bool(
            policy["rotation_constraint_enabled"]
        ),
    }
    failure_code = None
    if not checks["attachment_active"]:
        failure_code = "pick_attachment_lost"
    elif not all(checks.values()):
        failure_code = "pick_attachment_residual_exceeded"
    return {
        "passed": failure_code is None,
        "failure_code": failure_code,
        "checks": checks,
        "metrics": {
            **dict(metrics),
            "policy": policy,
        },
    }


def continuous_acquisition_lift_delta(
    *,
    clearance_before_m: float,
    required_clearance_m: float,
    tracking_guard_m: float = PICK_ACQUISITION_TRACKING_GUARD_M,
) -> float:
    """Return one bounded-headroom world +Z acquisition displacement.

    The guard belongs only to motion execution.  It never changes the live
    support-clearance threshold used by the Pick success predicate.
    """

    clearance = float(clearance_before_m)
    required = float(required_clearance_m)
    guard = float(tracking_guard_m)
    if not all(math.isfinite(value) for value in (clearance, required, guard)):
        raise ValueError("continuous acquisition lift inputs must be finite")
    if required < 0.0:
        raise ValueError("required acquisition clearance must be non-negative")
    if guard < 0.0:
        raise ValueError("acquisition tracking guard must be non-negative")
    if guard > PICK_ACQUISITION_TRACKING_GUARD_M:
        raise ValueError("acquisition tracking guard exceeds policy maximum")
    required_gap = max(0.0, required - clearance)
    return 0.0 if required_gap == 0.0 else required_gap + guard


def remaining_lift_delta(*, total_delta_m: float, achieved_delta_m: float) -> float:
    return max(0.0, float(total_delta_m) - float(achieved_delta_m))


def hold_failure_code(
    summary: Mapping[str, Any],
    *,
    phase: str,
    contract: SuccessContract,
) -> str | None:
    """Classify the first complete physical hold failure; never proposes recovery."""

    gripper = summary["gripper"]
    closure = gripper["closure_evidence"]
    hold = gripper["hold_consistency"]
    subject = str(closure["subject"])
    arm = str(gripper["arm"])
    has_inner = any(
        {str(row.get("entity_a") or ""), str(row.get("entity_b") or "")}
        == {subject, f"gripper:{arm}"}
        and str(row.get("region") or "") == "inner_pad"
        for row in summary["active_contacts"]
    )
    attachment = gripper.get("post_close_attachment")
    attachment_holds = bool(
        isinstance(attachment, Mapping)
        and attachment.get("passed") is True
        and attachment.get("subject") == subject
        and attachment.get("certified_close") is True
        and (attachment.get("checks") or {}).get("attachment_active") is True
        and (attachment.get("metrics") or {}).get("policy", {}).get("mode")
        == "fixed_constraint"
    )
    policy = hold_drift_policy(phase=phase, contract=contract)
    prefix = str(policy["phase"])
    # ``engaged`` is a coarse normalized-opening classifier.  Wide objects can
    # produce a real persistent bilateral pinch while remaining above that
    # profile threshold, so direct closure evidence (or its fixed-attachment
    # continuation) is the stronger physical signal.
    physically_engaged = bool(
        gripper.get("engaged") or closure.get("valid") or attachment_holds
    )
    if not physically_engaged:
        return "carry_gripper_disengaged" if prefix == "carry" else "pick_contact_lost"
    if not (has_inner or attachment_holds):
        return f"{prefix}_contact_lost"
    if not bool(closure.get("valid")):
        return "carry_closure_lost" if prefix == "carry" else "pick_contact_lost"
    translation = float(hold["translation_drift_m"])
    translation_limit = float(policy["translation_limit_m"])
    translation_failed = (
        translation > translation_limit
        if bool(policy["translation_limit_inclusive"])
        else translation >= translation_limit
    )
    if translation_failed:
        return f"{prefix}_translation_slip"
    if bool(policy["rotation_constraint_enabled"]) and float(
        hold["rotation_drift_deg"]
    ) > float(policy["rotation_limit_deg"]):
        return f"{prefix}_rotation_slip"
    return None


__all__ = [
    "PICK_ACQUISITION_POLICY_VERSION",
    "PICK_ACQUISITION_TRACKING_GUARD_M",
    "PICK_ACQUISITION_REQUIRED_CLEARANCE_M",
    "continuous_acquisition_lift_delta",
    "evaluate_pick_attachment",
    "hold_drift_policy",
    "hold_failure_code",
    "pick_attachment_policy",
    "remaining_lift_delta",
]
