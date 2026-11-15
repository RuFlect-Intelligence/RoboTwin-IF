"""Pure contract helpers for frozen articulated-operation capability evidence.

This module deliberately has no dependency on :mod:`if_benchmark.scene_validation`
or any milestone implementation.  M4, M8, M11, M12, M13, and M15 can therefore
validate the same exact-use/arm/Contact-candidate evidence without creating a
milestone import cycle.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from if_benchmark.schema.serialization import artifact_hash, canonical_data


ARTICULATION_CAPABILITY_VERSION = (
    "articulation_arm_contact_candidate_v5_contact_component_orientation"
)
LEGACY_ARTICULATION_CAPABILITY_VERSION = (
    "articulation_arm_contact_candidate_v3_requested_arm_denominator"
)
ARTICULATION_ARM_SAMPLING_VERSION = "m4_articulation_requested_arm_v1"
ARTICULATION_REQUESTED_ARMS = ("left", "right")


def _candidate_values(
    candidate: Mapping[str, Any],
) -> tuple[int, float, float, str] | None:
    try:
        raw_contact_id = candidate["contact_id"]
        contact_id = int(raw_contact_id)
        pre_grasp_m = float(candidate["pre_grasp_m"])
        roll_rad = float(candidate["roll_rad"])
        orientation_variant = str(candidate["orientation_variant"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if (
        isinstance(raw_contact_id, bool)
        or float(raw_contact_id) != float(contact_id)
        or contact_id < 0
        or not math.isfinite(pre_grasp_m)
        or not 0.0 <= pre_grasp_m <= 0.5
        or not math.isfinite(roll_rad)
        or abs(roll_rad) > 2.0 * math.pi + 1e-12
        or orientation_variant
        not in {
            "authored_closing_sign",
            "opposite_closing_sign",
            "authored_contact_frame",
        }
    ):
        return None
    return contact_id, pre_grasp_m, roll_rad, orientation_variant


def _same_contact_id(value: Any, expected: int) -> bool:
    try:
        return not isinstance(value, bool) and int(value) == expected
    except (TypeError, ValueError, OverflowError):
        return False


def articulation_success_capability(
    action: Mapping[str, Any], *, use: str
) -> dict[str, Any] | None:
    """Project one successful runtime action to its exact arm/Contact candidate.

    This is capability evidence, not a Cartesian or joint trace.  Downstream
    scenes still lower the authored Contact in their own live frame.
    """

    selected_arm = str(action.get("selected_arm") or "")
    if selected_arm not in {"left", "right"}:
        return None
    result = action.get("articulation_result")
    if not isinstance(result, Mapping):
        result = next(
            (
                row.get("result")
                for row in action.get("arm_attempts") or []
                if isinstance(row, Mapping)
                and str(row.get("arm") or "") == selected_arm
                and isinstance(row.get("result"), Mapping)
                and row["result"].get("success") is True
            ),
            None,
        )
    if not isinstance(result, Mapping) or result.get("success") is not True:
        return None
    for acquire in result.get("trace") or []:
        if (
            not isinstance(acquire, Mapping)
            or acquire.get("phase") != "acquire"
            or acquire.get("passed") is not True
            or acquire.get("contact_id") is None
        ):
            continue
        try:
            contact_id = int(acquire["contact_id"])
        except (TypeError, ValueError, OverflowError):
            continue
        attempts = list(acquire.get("attempts") or [])
        successful = next(
            (
                row
                for row in reversed(attempts)
                if isinstance(row, Mapping)
                and _same_contact_id(row.get("contact_id"), contact_id)
                and row.get("planned") is True
                and (
                    (row.get("observation") or {}).get("passed") is True
                    or (row.get("attachment") or {}).get("passed") is True
                )
            ),
            None,
        )
        if not isinstance(successful, Mapping):
            continue
        values = _candidate_values(
            {
                "contact_id": contact_id,
                "pre_grasp_m": successful.get("pre_grasp_m"),
                "roll_rad": successful.get("roll_rad"),
                "orientation_variant": successful.get("orientation_variant"),
            }
        )
        if values is None:
            continue
        contact_id, pre_grasp_m, roll_rad, orientation_variant = values
        return canonical_data(
            {
                "version": ARTICULATION_CAPABILITY_VERSION,
                "use": str(use),
                "arm": selected_arm,
                "candidate": {
                    "contact_id": contact_id,
                    "pre_grasp_m": pre_grasp_m,
                    "roll_rad": roll_rad,
                    "orientation_variant": orientation_variant,
                },
                "moving_link": str(acquire.get("moving_link") or ""),
                "engagement_mode": str(acquire.get("engagement_mode") or ""),
            }
        )
    return None


def aggregate_articulation_capabilities(
    attempts: Sequence[Mapping[str, Any]],
    *,
    use: str,
    denominator: int,
    threshold: float,
    require_explicit_runtime_capability: bool = False,
) -> dict[str, Any]:
    """Aggregate exact candidates with an independent denominator per arm.

    New M4 evidence binds every attempt to ``requested_arm``. A successful
    fallback by the other arm is rejected instead of being credited to the
    requested arm. The shared-denominator path remains readable only for old
    evidence callers that do not require the explicit runtime contract.
    """

    failures: set[str] = set()
    by_arm: dict[str, dict[str, Any]] = {}
    explicit_requested_arms = {
        str(attempt.get("requested_arm") or "")
        for attempt in attempts
        if str(attempt.get("requested_arm") or "")
        in ARTICULATION_REQUESTED_ARMS
    }
    explicit_sampling = bool(explicit_requested_arms)
    if require_explicit_runtime_capability and not explicit_requested_arms:
        failures.add("articulation_requested_arm_coverage_missing")
    denominators = {
        arm: sum(
            1
            for attempt in attempts
            if str(attempt.get("requested_arm") or "") == arm
        )
        for arm in ARTICULATION_REQUESTED_ARMS
    }
    for attempt in attempts:
        requested_arm = str(attempt.get("requested_arm") or "")
        if explicit_sampling and requested_arm not in ARTICULATION_REQUESTED_ARMS:
            failures.add("articulation_requested_arm_missing")
            continue
        if attempt.get("success") is not True:
            continue
        action = (attempt.get("diagnostics") or {}).get("articulation_action")
        if not isinstance(action, Mapping):
            failures.add("articulation_success_capability_missing")
            continue
        derived = articulation_success_capability(action, use=str(use))
        explicit = action.get("successful_capability")
        if require_explicit_runtime_capability and not isinstance(explicit, Mapping):
            failures.add("articulation_runtime_success_capability_missing")
            continue
        if isinstance(explicit, Mapping) and canonical_data(dict(explicit)) != derived:
            failures.add("articulation_runtime_success_capability_mismatch")
            continue
        capability = canonical_data(dict(explicit)) if isinstance(explicit, Mapping) else derived
        if not isinstance(capability, Mapping):
            failures.add("articulation_success_capability_missing")
            continue
        arm = str(capability.get("arm") or "")
        selected_arm = str(action.get("selected_arm") or "")
        action_requested_arm = str(action.get("requested_arm") or "")
        if explicit_sampling and (
            arm != requested_arm
            or selected_arm != requested_arm
            or action_requested_arm != requested_arm
        ):
            failures.add("articulation_requested_arm_mismatch")
            continue
        candidate = capability.get("candidate") or {}
        candidate_values = (
            _candidate_values(candidate)
            if isinstance(candidate, Mapping)
            else None
        )
        if (
            capability.get("version") != ARTICULATION_CAPABILITY_VERSION
            or capability.get("use") != str(use)
            or arm not in {"left", "right"}
            or candidate_values is None
            or not capability.get("moving_link")
            or not capability.get("engagement_mode")
        ):
            failures.add("articulation_success_capability_invalid")
            continue
        row = by_arm.setdefault(
            arm,
            {
                "version": ARTICULATION_CAPABILITY_VERSION,
                "use": str(use),
                "arm": arm,
                "success": 0,
                "denominator": int(
                    denominators[arm] if explicit_sampling else denominator
                ),
                "threshold": float(threshold),
                "candidate_counts": {},
            },
        )
        row["success"] += 1
        candidate_key = artifact_hash(
            {
                "candidate": candidate,
                "moving_link": capability["moving_link"],
                "engagement_mode": capability["engagement_mode"],
            }
        )
        counted = row["candidate_counts"].setdefault(
            candidate_key,
            {
                "candidate": canonical_data(dict(candidate)),
                "moving_link": str(capability["moving_link"]),
                "engagement_mode": str(capability["engagement_mode"]),
                "success": 0,
            },
        )
        counted["success"] += 1
    arm_capabilities: list[dict[str, Any]] = []
    for arm, raw in sorted(by_arm.items()):
        arm_denominator = int(raw["denominator"])
        rate = raw["success"] / arm_denominator if arm_denominator else 0.0
        if rate + 1e-12 < float(threshold):
            continue
        arm_capabilities.append(
            canonical_data(
                {
                    "version": ARTICULATION_CAPABILITY_VERSION,
                    "use": str(use),
                    "arm": arm,
                    "success": int(raw["success"]),
                    "denominator": arm_denominator,
                    "min_rate": float(rate),
                    "threshold": float(threshold),
                    "successful_candidates": sorted(
                        raw["candidate_counts"].values(),
                        key=lambda row: artifact_hash(row),
                    ),
                }
            )
        )
    overall_success = sum(
        1 for attempt in attempts if attempt.get("success") is True
    )
    qualification_rates = (
        [
            sum(
                1
                for attempt in attempts
                if str(attempt.get("requested_arm") or "") == arm
                and attempt.get("success") is True
            )
            / denominators[arm]
            for arm in ARTICULATION_REQUESTED_ARMS
            if denominators[arm]
        ]
        if explicit_sampling
        else [overall_success / denominator]
        if denominator
        else []
    )
    if (
        qualification_rates
        and max(qualification_rates) + 1e-12 >= float(threshold)
        and not arm_capabilities
    ):
        failures.add("articulation_arm_capability_threshold_not_met")
    return canonical_data(
        {
            "capability_version": ARTICULATION_CAPABILITY_VERSION,
            "sampling_version": (
                ARTICULATION_ARM_SAMPLING_VERSION if explicit_sampling else None
            ),
            "arm_capabilities": arm_capabilities,
            "failures": sorted(failures),
        }
    )


def articulation_capability_errors(
    binding: Mapping[str, Any], *, use: str
) -> list[str]:
    """Validate the exact use/arm/Contact-candidate capability projection."""

    errors: list[str] = []
    if binding.get("capability_version") != ARTICULATION_CAPABILITY_VERSION:
        errors.append("capability_version")
    if binding.get("sampling_version") != ARTICULATION_ARM_SAMPLING_VERSION:
        errors.append("sampling_version")
    if binding.get("requested_arms") != list(ARTICULATION_REQUESTED_ARMS):
        errors.append("requested_arms")
    arm_results = binding.get("arm_results")
    if not isinstance(arm_results, Mapping) or set(arm_results) != set(
        ARTICULATION_REQUESTED_ARMS
    ):
        errors.append("arm_results")
    failures = binding.get("failures", [])
    if (
        not isinstance(failures, Sequence)
        or isinstance(failures, (str, bytes))
        or any(not isinstance(failure, str) or not failure for failure in failures)
        or bool(failures)
    ):
        errors.append("failures")
    rows = binding.get("arm_capabilities")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        return [*errors, "arm_capabilities"]
    seen_arms: set[str] = set()
    for index, row in enumerate(rows):
        prefix = f"arm_capabilities[{index}]"
        if not isinstance(row, Mapping):
            errors.append(prefix)
            continue
        arm = str(row.get("arm") or "")
        if (
            row.get("version") != ARTICULATION_CAPABILITY_VERSION
            or row.get("use") != str(use)
            or arm not in {"left", "right"}
        ):
            errors.append(prefix)
            continue
        if arm in seen_arms:
            errors.append(f"{prefix}.duplicate_arm")
        seen_arms.add(arm)
        try:
            success = int(row.get("success") or 0)
            denominator = int(row.get("denominator") or 0)
            threshold = float(row.get("threshold") or 0.0)
            rate = float(row.get("min_rate") or 0.0)
        except (TypeError, ValueError, OverflowError):
            errors.append(f"{prefix}.rate")
            continue
        if (
            success <= 0
            or denominator <= 0
            or success > denominator
            or threshold <= 0.0
            or threshold > 1.0
            or not math.isfinite(threshold)
            or not math.isfinite(rate)
            or abs(rate - success / denominator) > 1e-12
            or rate + 1e-12 < threshold
        ):
            errors.append(f"{prefix}.rate")
        candidates = row.get("successful_candidates")
        if (
            not isinstance(candidates, Sequence)
            or isinstance(candidates, (str, bytes))
            or not candidates
        ):
            errors.append(f"{prefix}.successful_candidates")
            continue
        seen_candidates: set[tuple[int, float, float]] = set()
        candidate_success_total = 0
        for candidate_index, candidate_row in enumerate(candidates):
            candidate_prefix = f"{prefix}.successful_candidates[{candidate_index}]"
            if not isinstance(candidate_row, Mapping):
                errors.append(candidate_prefix)
                continue
            candidate = candidate_row.get("candidate")
            values = (
                _candidate_values(candidate)
                if isinstance(candidate, Mapping)
                else None
            )
            if values is None:
                errors.append(f"{candidate_prefix}.candidate")
                continue
            key = values
            if key in seen_candidates:
                errors.append(f"{candidate_prefix}.duplicate")
            seen_candidates.add(key)
            try:
                candidate_success = int(candidate_row.get("success") or 0)
            except (TypeError, ValueError, OverflowError):
                candidate_success = 0
            candidate_success_total += max(candidate_success, 0)
            if (
                candidate_success <= 0
                or not candidate_row.get("moving_link")
                or not candidate_row.get("engagement_mode")
            ):
                errors.append(candidate_prefix)
        if candidate_success_total != success:
            errors.append(f"{prefix}.candidate_success_total")
    if isinstance(arm_results, Mapping):
        result_pass_arms = {
            str(arm)
            for arm, result in arm_results.items()
            if isinstance(result, Mapping) and result.get("pass") is True
        }
        if seen_arms != result_pass_arms:
            errors.append("arm_results.capability_arms")
    return errors


def validated_articulation_arm_capability(
    binding: Mapping[str, Any], *, use: str, arm: str
) -> dict[str, Any] | None:
    """Return one exact per-arm capability only after full binding validation."""

    if articulation_capability_errors(binding, use=str(use)):
        return None
    matches = [
        row
        for row in binding.get("arm_capabilities") or []
        if str(row.get("use") or "") == str(use)
        and str(row.get("arm") or "") == str(arm)
    ]
    return canonical_data(dict(matches[0])) if len(matches) == 1 else None
