"""Pure M4 capability contract for short authored-contact actions.

Click and Press are not articulated-state actions and do not own an
articulation acquisition candidate.  Their capability is the much smaller
fact proved by the physical worker: one requested arm made an actual PhysX
contact at one authored Contact id and the shared success predicate passed.

The helpers in this module intentionally have no simulator or milestone
dependencies so every later consumer can validate the exact same binding.
"""

from __future__ import annotations

import math
import re
from typing import Any, Mapping, Sequence

from if_benchmark.operation_placement import (
    M4_TOUCH_PLACEMENT_POLICY_HASH,
    M4_TOUCH_PLACEMENT_POLICY_VERSION,
)
from if_benchmark.schema.serialization import canonical_data


TOUCH_EXECUTABLE_USES = frozenset(("click_subject", "press_subject"))
TOUCH_REQUESTED_ARMS = ("left", "right")
TOUCH_ARM_SAMPLING_VERSION = "m4_touch_requested_arm_v1"
TOUCH_CAPABILITY_VERSION = "touch_arm_authored_contact_v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _canonical_float(value: float) -> float:
    """Use the same numeric representation as normalized M4 report rows."""

    return round(float(value), 12)


def _exact_nonnegative_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if (
        isinstance(value, bool)
        or parsed < 0
        or float(value) != float(parsed)
    ):
        return None
    return parsed


def _same_event_field(
    predicate_event: Mapping[str, Any],
    runtime_event: Mapping[str, Any],
    field: str,
) -> bool:
    left = predicate_event.get(field)
    right = runtime_event.get(field)
    if field in {
        "distance_to_authored_center_m",
        "target_local_extent_m",
        "summed_impulse_norm",
        "subject_displacement_m",
    }:
        try:
            left_value = float(left)
            right_value = float(right)
        except (TypeError, ValueError, OverflowError):
            return False
        return (
            math.isfinite(left_value)
            and math.isfinite(right_value)
            and abs(left_value - right_value) <= 1e-12
        )
    return left == right


def _successful_touch_capability(
    attempt: Mapping[str, Any], *, use: str
) -> tuple[dict[str, Any] | None, str | None]:
    """Derive one capability only from matched predicate/runtime evidence."""

    requested_arm = str(attempt.get("requested_arm") or "")
    if requested_arm not in TOUCH_REQUESTED_ARMS:
        return None, "touch_requested_arm_missing"
    predicate = attempt.get("predicate_result")
    if (
        not isinstance(predicate, Mapping)
        or predicate.get("predicate") != "click_press_success"
        or predicate.get("status") != "passed"
        or predicate.get("passed") is not True
    ):
        return None, "touch_success_predicate_evidence_missing"
    predicate_event = predicate.get("metrics")
    diagnostics = attempt.get("diagnostics")
    action = (
        diagnostics.get("touch_action")
        if isinstance(diagnostics, Mapping)
        else None
    )
    runtime_event = (
        action.get("contact_event") if isinstance(action, Mapping) else None
    )
    if not isinstance(predicate_event, Mapping) or not isinstance(
        runtime_event, Mapping
    ):
        return None, "touch_runtime_contact_event_missing"
    contact_id = _exact_nonnegative_int(runtime_event.get("contact_id"))
    if (
        action.get("use") != str(use)
        or action.get("requested_arm") != requested_arm
        or _exact_nonnegative_int(action.get("contact_id")) != contact_id
        or str(runtime_event.get("arm") or "") != requested_arm
        or str(runtime_event.get("subject") or "") != "subject"
        or contact_id is None
    ):
        return None, "touch_runtime_contact_event_invalid"
    if any(
        not _same_event_field(predicate_event, runtime_event, field)
        for field in (
            "arm",
            "subject",
            "contact_id",
            "distance_to_authored_center_m",
            "target_local_extent_m",
            "contact_point_count",
            "summed_impulse_norm",
            "subject_displacement_m",
            "subject_displacement_failed",
        )
    ):
        return None, "touch_predicate_runtime_event_mismatch"
    placement = attempt.get("canonical_placement_witness")
    selected = (
        placement.get("selected_contact_ids_by_arm")
        if isinstance(placement, Mapping)
        else None
    )
    if (
        not isinstance(selected, Mapping)
        or _exact_nonnegative_int(selected.get(requested_arm)) != contact_id
    ):
        return None, "touch_placement_contact_mismatch"
    return (
        canonical_data(
            {
                "version": TOUCH_CAPABILITY_VERSION,
                "use": str(use),
                "arm": requested_arm,
                "contact_id": contact_id,
            }
        ),
        None,
    )


def aggregate_touch_capabilities(
    attempts: Sequence[Mapping[str, Any]],
    *,
    use: str,
    threshold: float,
) -> dict[str, Any]:
    """Aggregate Click/Press evidence with one denominator per requested arm."""

    use_name = str(use)
    if use_name not in TOUCH_EXECUTABLE_USES:
        raise ValueError(f"unsupported touch use: {use_name!r}")
    threshold_value = _canonical_float(threshold)
    if not math.isfinite(threshold_value) or not 0.0 < threshold_value <= 1.0:
        raise ValueError("touch threshold must be finite and in (0, 1]")

    failures: set[str] = set()
    placement_witness_hashes: set[str] = set()
    denominator_by_arm = {arm: 0 for arm in TOUCH_REQUESTED_ARMS}
    success_by_arm = {arm: 0 for arm in TOUCH_REQUESTED_ARMS}
    contact_ids_by_arm: dict[str, set[int]] = {
        arm: set() for arm in TOUCH_REQUESTED_ARMS
    }
    for attempt in attempts:
        requested_arm = str(attempt.get("requested_arm") or "")
        if requested_arm not in TOUCH_REQUESTED_ARMS:
            failures.add("touch_requested_arm_missing")
            continue
        denominator_by_arm[requested_arm] += 1
        placement = attempt.get("canonical_placement_witness")
        witness_hash = str(
            placement.get("witness_hash")
            if isinstance(placement, Mapping)
            else ""
        )
        if _SHA256.fullmatch(witness_hash) is None:
            failures.add("touch_placement_witness_hash_missing")
        else:
            placement_witness_hashes.add(witness_hash)
        if attempt.get("success") is not True:
            continue
        success_by_arm[requested_arm] += 1
        capability, failure = _successful_touch_capability(
            attempt, use=use_name
        )
        if failure is not None:
            failures.add(failure)
            continue
        assert capability is not None
        contact_ids_by_arm[requested_arm].add(int(capability["contact_id"]))

    arm_results: dict[str, dict[str, Any]] = {}
    arm_capabilities: list[dict[str, Any]] = []
    for arm in TOUCH_REQUESTED_ARMS:
        denominator = denominator_by_arm[arm]
        success = success_by_arm[arm]
        rate = _canonical_float(
            success / denominator if denominator else 0.0
        )
        contacts = contact_ids_by_arm[arm]
        if len(contacts) > 1:
            failures.add("touch_contact_id_inconsistent_within_arm")
        passed = bool(
            denominator
            and rate + 1e-12 >= threshold_value
            and len(contacts) == 1
        )
        arm_results[arm] = {
            "denominator": int(denominator),
            "success": int(success),
            "min_rate": rate,
            "threshold": threshold_value,
            "pass": passed,
        }
        if passed:
            arm_capabilities.append(
                canonical_data(
                    {
                        "version": TOUCH_CAPABILITY_VERSION,
                        "use": use_name,
                        "arm": arm,
                        "contact_id": next(iter(contacts)),
                        "success": int(success),
                        "denominator": int(denominator),
                        "min_rate": rate,
                        "threshold": threshold_value,
                    }
                )
            )
    frozen_contact_ids = {
        int(row["contact_id"]) for row in arm_capabilities
    }
    if len(placement_witness_hashes) != 1:
        failures.add("touch_placement_witness_inconsistent_across_arms")
        arm_capabilities = []
        for result in arm_results.values():
            result["pass"] = False
    if len(frozen_contact_ids) > 1:
        failures.add("touch_contact_id_inconsistent_across_arms")
        arm_capabilities = []
        for result in arm_results.values():
            result["pass"] = False
    return canonical_data(
        {
            "capability_version": TOUCH_CAPABILITY_VERSION,
            "sampling_version": TOUCH_ARM_SAMPLING_VERSION,
            "placement_policy_version": M4_TOUCH_PLACEMENT_POLICY_VERSION,
            "placement_policy_hash": M4_TOUCH_PLACEMENT_POLICY_HASH,
            "requested_arms": list(TOUCH_REQUESTED_ARMS),
            "canonical_placement_witness_hash": (
                next(iter(placement_witness_hashes))
                if len(placement_witness_hashes) == 1
                else None
            ),
            "arm_results": arm_results,
            "arm_capabilities": arm_capabilities,
            "failures": sorted(failures),
        }
    )


def touch_capability_errors(
    binding: Mapping[str, Any], *, use: str
) -> list[str]:
    """Validate one complete frozen Click/Press per-arm binding."""

    errors: list[str] = []
    use_name = str(use)
    if use_name not in TOUCH_EXECUTABLE_USES:
        return ["use"]
    if binding.get("capability_version") != TOUCH_CAPABILITY_VERSION:
        errors.append("capability_version")
    if binding.get("sampling_version") != TOUCH_ARM_SAMPLING_VERSION:
        errors.append("sampling_version")
    if binding.get("placement_policy_version") != (
        M4_TOUCH_PLACEMENT_POLICY_VERSION
    ):
        errors.append("placement_policy_version")
    if binding.get("placement_policy_hash") != M4_TOUCH_PLACEMENT_POLICY_HASH:
        errors.append("placement_policy_hash")
    if binding.get("requested_arms") != list(TOUCH_REQUESTED_ARMS):
        errors.append("requested_arms")
    if _SHA256.fullmatch(
        str(binding.get("canonical_placement_witness_hash") or "")
    ) is None:
        errors.append("canonical_placement_witness_hash")
    failures = binding.get("failures")
    if (
        not isinstance(failures, Sequence)
        or isinstance(failures, (str, bytes))
        or any(not isinstance(value, str) or not value for value in failures)
        or bool(failures)
    ):
        errors.append("failures")
    arm_results = binding.get("arm_results")
    if not isinstance(arm_results, Mapping) or set(arm_results) != set(
        TOUCH_REQUESTED_ARMS
    ):
        errors.append("arm_results")
        arm_results = {}
    passed_arms: set[str] = set()
    for arm in TOUCH_REQUESTED_ARMS:
        result = arm_results.get(arm)
        prefix = f"arm_results.{arm}"
        if not isinstance(result, Mapping):
            continue
        try:
            denominator = int(result.get("denominator") or 0)
            success = int(result.get("success") or 0)
            rate = float(result.get("min_rate") or 0.0)
            threshold = float(result.get("threshold") or 0.0)
            excluded_blocking = int(
                result.get("excluded_blocking_attempts") or 0
            )
        except (TypeError, ValueError, OverflowError):
            errors.append(prefix)
            continue
        expected_rate = success / denominator if denominator else 0.0
        expected_pass = bool(
            denominator
            and not excluded_blocking
            and expected_rate + 1e-12 >= threshold
        )
        if (
            denominator < 0
            or success < 0
            or success > denominator
            or excluded_blocking < 0
            or not math.isfinite(rate)
            or not math.isfinite(threshold)
            or not 0.0 < threshold <= 1.0
            or abs(rate - expected_rate) > 1e-12
            or bool(result.get("pass")) != expected_pass
        ):
            errors.append(prefix)
        expected_outcome = (
            "blocked"
            if excluded_blocking
            else "pass"
            if expected_pass
            else "fail"
        )
        if result.get("outcome", expected_outcome) != expected_outcome:
            errors.append(f"{prefix}.outcome")
        if result.get("pass") is True:
            passed_arms.add(arm)

    rows = binding.get("arm_capabilities")
    if (
        not isinstance(rows, Sequence)
        or isinstance(rows, (str, bytes))
        or not rows
    ):
        return sorted(set([*errors, "arm_capabilities"]))
    seen_arms: set[str] = set()
    contact_ids: set[int] = set()
    for index, row in enumerate(rows):
        prefix = f"arm_capabilities[{index}]"
        if not isinstance(row, Mapping):
            errors.append(prefix)
            continue
        arm = str(row.get("arm") or "")
        contact_id = _exact_nonnegative_int(row.get("contact_id"))
        if (
            row.get("version") != TOUCH_CAPABILITY_VERSION
            or row.get("use") != use_name
            or arm not in TOUCH_REQUESTED_ARMS
            or contact_id is None
        ):
            errors.append(prefix)
            continue
        if arm in seen_arms:
            errors.append(f"{prefix}.duplicate_arm")
        seen_arms.add(arm)
        contact_ids.add(contact_id)
        result = arm_results.get(arm) if isinstance(arm_results, Mapping) else None
        if not isinstance(result, Mapping) or any(
            row.get(key) != result.get(key)
            for key in (
                "success",
                "denominator",
                "min_rate",
                "threshold",
            )
        ):
            errors.append(f"{prefix}.arm_result")
    if seen_arms != passed_arms:
        errors.append("arm_results.capability_arms")
    if len(contact_ids) != 1:
        errors.append("arm_capabilities.contact_id")
    return sorted(set(errors))


def validated_touch_arm_capability(
    binding: Mapping[str, Any], *, use: str, arm: str
) -> dict[str, Any] | None:
    """Return one per-arm touch capability only after full validation."""

    if touch_capability_errors(binding, use=str(use)):
        return None
    matches = [
        row
        for row in binding.get("arm_capabilities") or []
        if str(row.get("use") or "") == str(use)
        and str(row.get("arm") or "") == str(arm)
    ]
    return canonical_data(dict(matches[0])) if len(matches) == 1 else None


__all__ = [
    "TOUCH_ARM_SAMPLING_VERSION",
    "TOUCH_CAPABILITY_VERSION",
    "TOUCH_EXECUTABLE_USES",
    "TOUCH_REQUESTED_ARMS",
    "aggregate_touch_capabilities",
    "touch_capability_errors",
    "validated_touch_arm_capability",
]
