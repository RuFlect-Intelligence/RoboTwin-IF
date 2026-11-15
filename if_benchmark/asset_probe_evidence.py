"""Dependency-light accounting rules for canonical M4 probe evidence.

The physical worker owns each attempt.  A report row is only a projection of
those attempts, so consumers must recompute its denominator and verdict-facing
counts instead of trusting fields written by a runner or repair utility.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


M4_BLOCKING_FAILURE_PREFIXES = (
    "source_",
    "internal_",
    "worker_",
    "setup_",
    "missing_candidate_",
)
M4_SOURCE_OR_INTERNAL_FAILURE_PREFIXES = ("source_", "internal_")

M4_RETRYABLE_FAILURE_REASONS = frozenset(
    {
        "worker_import_failed",
        "worker_process_failed",
        "worker_timeout",
        "worker_no_json_result",
        "worker_physical_collision_validation_failed",
    }
)
M4_RETRYABLE_DIAGNOSTIC_MARKERS = (
    "cuda kernel build failed",
    "warp nvrtc compilation error",
    "nvrtc_error_compilation",
    "'robot' object has no attribute 'left_planner'",
    "'robot' object has no attribute 'right_planner'",
    '"robot" object has no attribute "left_planner"',
    '"robot" object has no attribute "right_planner"',
)


def m4_attempt_is_blocking(attempt: Mapping[str, Any]) -> bool:
    """Return whether an attempt stopped before terminal checker evidence."""

    reason = str(attempt.get("failure_reason") or "")
    return reason.startswith(M4_BLOCKING_FAILURE_PREFIXES)


def m4_attempt_is_retryable(attempt: Mapping[str, Any]) -> bool:
    """Return whether resume should replace one transient infrastructure row.

    Physical/setup blockers remain durable evidence. Only explicit worker
    transport failures and the known Warp/partially-initialized-Robot cascade
    are discarded from a resume checkpoint and scheduled again.
    """

    reason = str(attempt.get("failure_reason") or "").strip().lower()
    if reason in M4_RETRYABLE_FAILURE_REASONS:
        return True
    diagnostics = "\n".join(
        str(attempt.get(field) or "")
        for field in (
            "exception",
            "exception_type",
            "stderr_tail",
            "stdout_tail",
            "traceback_tail",
        )
    ).lower()
    return any(marker in diagnostics for marker in M4_RETRYABLE_DIAGNOSTIC_MARKERS)


def m4_row_attempt_accounting_errors(
    row: Mapping[str, Any],
    attempts: Sequence[Mapping[str, Any]],
    *,
    require_seed_coverage: bool = True,
) -> list[str]:
    """Audit a canonical M4 row against all of its worker attempts.

    Blocking attempts are deliberately part of ``attempts`` but never part of
    the statistical denominator.  Their presence must still block promotion;
    a caller may not discard them and then present the remaining attempts as a
    passing row.
    """

    if (
        row.get("articulation_arm_sampling_version") is not None
        or row.get("touch_arm_sampling_version") is not None
    ):
        return _m4_requested_arm_attempt_accounting_errors(
            row,
            attempts,
            require_seed_coverage=require_seed_coverage,
        )

    failures: set[str] = set()
    attempt_rows = list(attempts)
    row_identity = (
        str(row.get("asset")),
        str(row.get("model_id")),
        str(row.get("use")),
    )
    for attempt in attempt_rows:
        attempt_identity = (
            str(attempt.get("asset")),
            str(attempt.get("model_id")),
            str(attempt.get("use")),
        )
        if attempt_identity != row_identity:
            failures.add("attempt_identity_mismatch")

    expected_attempts = _nonnegative_int(row, "attempts", failures)
    if expected_attempts is not None and expected_attempts != len(attempt_rows):
        failures.add("attempt_count_mismatch")

    blocking_attempts = [
        attempt for attempt in attempt_rows if m4_attempt_is_blocking(attempt)
    ]
    denominator_attempts = [
        attempt for attempt in attempt_rows if not m4_attempt_is_blocking(attempt)
    ]
    source_or_internal_count = sum(
        1
        for attempt in blocking_attempts
        if str(attempt.get("failure_reason") or "").startswith(
            M4_SOURCE_OR_INTERNAL_FAILURE_PREFIXES
        )
    )

    expected_blocking = _nonnegative_int(
        row, "excluded_blocking_attempts", failures
    )
    if expected_blocking is not None and expected_blocking != len(blocking_attempts):
        failures.add("blocking_attempt_count_mismatch")
    expected_source_or_internal = _nonnegative_int(
        row, "excluded_source_or_internal_attempts", failures
    )
    if (
        expected_source_or_internal is not None
        and expected_source_or_internal != source_or_internal_count
    ):
        failures.add("source_or_internal_attempt_count_mismatch")

    expected_denominator = _nonnegative_int(row, "denominator", failures)
    if (
        expected_denominator is not None
        and expected_denominator != len(denominator_attempts)
    ):
        failures.add("denominator_mismatch")

    plan_success = sum(
        1 for attempt in denominator_attempts if bool(attempt.get("plan_success"))
    )
    success = sum(
        1 for attempt in denominator_attempts if bool(attempt.get("success"))
    )
    if any(
        bool(attempt.get("success")) and not bool(attempt.get("plan_success"))
        for attempt in denominator_attempts
    ):
        failures.add("attempt_success_without_plan_success")
    expected_plan_success = _nonnegative_int(row, "plan_success", failures)
    if expected_plan_success is not None and expected_plan_success != plan_success:
        failures.add("plan_success_count_mismatch")
    expected_success = _nonnegative_int(row, "success", failures)
    if expected_success is not None and expected_success != success:
        failures.add("success_count_mismatch")

    rate = success / len(denominator_attempts) if denominator_attempts else 0.0
    min_rate = _finite_float(row, "min_rate", failures)
    if min_rate is not None and round(min_rate, 12) != round(rate, 12):
        failures.add("min_rate_mismatch")
    threshold = _finite_float(row, "threshold", failures)
    if threshold is not None and not 0.0 <= threshold <= 1.0:
        failures.add("threshold_out_of_range")

    _audit_seeds(
        row,
        attempt_rows,
        failures,
        require_seed_coverage=require_seed_coverage,
    )

    evidence_blocked = _row_evidence_blocked(row, failures)
    expected_outcome = (
        "blocked"
        if blocking_attempts or evidence_blocked
        else "pass"
        if threshold is not None and rate >= threshold
        else "fail"
    )
    outcome = row.get("outcome")
    passed = row.get("pass")
    if outcome not in {"pass", "fail", "blocked"}:
        failures.add("outcome_invalid")
    if not isinstance(passed, bool):
        failures.add("pass_not_boolean")
    if blocking_attempts and (outcome != "blocked" or passed is not False):
        failures.add("blocking_attempt_did_not_block_row")
    if outcome == "pass":
        if passed is not True:
            failures.add("pass_outcome_flag_mismatch")
        if blocking_attempts:
            failures.add("passing_row_contains_blocking_attempt")
        if threshold is not None and rate < threshold:
            failures.add("passing_row_below_threshold")
    elif outcome == "fail":
        if passed is not False:
            failures.add("fail_outcome_flag_mismatch")
        if blocking_attempts:
            failures.add("failed_row_contains_blocking_attempt")
        if threshold is not None and rate >= threshold:
            failures.add("failed_row_meets_threshold")
    elif outcome == "blocked" and passed is not False:
        failures.add("blocked_outcome_flag_mismatch")
    if passed is True and outcome != "pass":
        failures.add("pass_flag_outcome_mismatch")
    if outcome in {"pass", "fail", "blocked"} and outcome != expected_outcome:
        failures.add("outcome_does_not_match_attempt_evidence")

    return sorted(failures)


def _m4_requested_arm_attempt_accounting_errors(
    row: Mapping[str, Any],
    attempts: Sequence[Mapping[str, Any]],
    *,
    require_seed_coverage: bool,
) -> list[str]:
    """Audit an explicit requested-arm matrix without pooling denominators."""

    failures: set[str] = set()
    attempt_rows = list(attempts)
    row_identity = _identity(row)
    if any(_identity(attempt) != row_identity for attempt in attempt_rows):
        failures.add("attempt_identity_mismatch")

    field_prefix = (
        "touch"
        if row.get("touch_arm_sampling_version") is not None
        else "articulation"
    )
    requested_arms = row.get(f"{field_prefix}_requested_arms")
    if requested_arms != ["left", "right"]:
        failures.add(f"{field_prefix}_requested_arms_invalid")
        requested_arms = ["left", "right"]
    if any(
        str(attempt.get("requested_arm") or "") not in requested_arms
        for attempt in attempt_rows
    ):
        failures.add("attempt_requested_arm_invalid")

    expected_attempts = _nonnegative_int(row, "attempts", failures)
    if expected_attempts is not None and expected_attempts != len(attempt_rows):
        failures.add("attempt_count_mismatch")
    blocking_attempts = [
        attempt for attempt in attempt_rows if m4_attempt_is_blocking(attempt)
    ]
    denominator_attempts = [
        attempt for attempt in attempt_rows if not m4_attempt_is_blocking(attempt)
    ]
    source_or_internal_count = sum(
        1
        for attempt in blocking_attempts
        if str(attempt.get("failure_reason") or "").startswith(
            M4_SOURCE_OR_INTERNAL_FAILURE_PREFIXES
        )
    )
    for field, observed in (
        ("excluded_blocking_attempts", len(blocking_attempts)),
        ("excluded_source_or_internal_attempts", source_or_internal_count),
        ("denominator", len(denominator_attempts)),
        (
            "plan_success",
            sum(1 for attempt in denominator_attempts if attempt.get("plan_success")),
        ),
        (
            "success",
            sum(1 for attempt in denominator_attempts if attempt.get("success")),
        ),
    ):
        expected = _nonnegative_int(row, field, failures)
        if expected is not None and expected != observed:
            failures.add(f"{field}_mismatch")

    row_seeds = row.get("seeds")
    if not isinstance(row_seeds, list) or any(
        not _is_int(seed) for seed in row_seeds
    ):
        failures.add("row_seeds_invalid")
        row_seeds = []
    arm_results = row.get(f"{field_prefix}_arm_results")
    if not isinstance(arm_results, Mapping) or set(arm_results) != set(
        requested_arms
    ):
        failures.add(f"{field_prefix}_arm_results_invalid")
        arm_results = {}

    qualification_rates: list[float] = []
    qualified = False
    for arm in requested_arms:
        arm_attempts = [
            attempt
            for attempt in attempt_rows
            if str(attempt.get("requested_arm") or "") == arm
        ]
        if require_seed_coverage:
            attempt_seeds = [attempt.get("seed") for attempt in arm_attempts]
            if (
                any(not _is_int(seed) for seed in attempt_seeds)
                or sorted(int(seed) for seed in attempt_seeds)
                != sorted(int(seed) for seed in row_seeds)
            ):
                failures.add(f"{arm}_seed_coverage_mismatch")
        result = arm_results.get(arm)
        if not isinstance(result, Mapping):
            continue
        blocked = [
            attempt for attempt in arm_attempts if m4_attempt_is_blocking(attempt)
        ]
        denominator_rows = [
            attempt
            for attempt in arm_attempts
            if not m4_attempt_is_blocking(attempt)
        ]
        success = sum(1 for attempt in denominator_rows if attempt.get("success"))
        plan_success = sum(
            1 for attempt in denominator_rows if attempt.get("plan_success")
        )
        expected_values = {
            "attempts": len(arm_attempts),
            "denominator": len(denominator_rows),
            "excluded_blocking_attempts": len(blocked),
            "plan_success": plan_success,
            "success": success,
        }
        for field, observed in expected_values.items():
            if result.get(field) != observed:
                failures.add(f"{arm}_{field}_mismatch")
        rate = success / len(denominator_rows) if denominator_rows else 0.0
        try:
            observed_rate = float(result.get("rate"))
            threshold = float(result.get("threshold"))
        except (TypeError, ValueError, OverflowError):
            failures.add(f"{arm}_rate_invalid")
            continue
        if not math.isfinite(observed_rate) or abs(observed_rate - rate) > 1e-12:
            failures.add(f"{arm}_rate_mismatch")
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            failures.add(f"{arm}_threshold_invalid")
        if not blocked:
            qualification_rates.append(rate)
        arm_pass = result.get("pass")
        if not isinstance(arm_pass, bool):
            failures.add(f"{arm}_pass_invalid")
        elif arm_pass:
            qualified = True
            if blocked or rate + 1e-12 < threshold or result.get("outcome") != "pass":
                failures.add(f"{arm}_passing_result_invalid")
        elif blocked and result.get("outcome") != "blocked":
            failures.add(f"{arm}_blocking_result_invalid")

    min_rate = _finite_float(row, "min_rate", failures)
    expected_rate = max(qualification_rates, default=0.0)
    if min_rate is not None and abs(min_rate - expected_rate) > 1e-12:
        failures.add("min_rate_mismatch")
    threshold = _finite_float(row, "threshold", failures)
    evidence_blocked = _row_evidence_blocked(row, failures)
    expected_outcome = (
        "blocked"
        if evidence_blocked or (blocking_attempts and not qualified)
        else "pass"
        if qualified
        else "fail"
    )
    if row.get("outcome") != expected_outcome:
        failures.add("outcome_does_not_match_attempt_evidence")
    if row.get("pass") is not (expected_outcome == "pass"):
        failures.add("pass_flag_outcome_mismatch")
    if threshold is not None and qualified and expected_rate + 1e-12 < threshold:
        failures.add("passing_row_below_threshold")
    return sorted(failures)


def m4_report_attempt_accounting_errors(
    records: Any,
    attempts: Any,
    *,
    require_seed_coverage: bool = True,
) -> list[str]:
    """Audit the one-to-one record/attempt-group accounting of a report."""

    failures: set[str] = set()
    if not isinstance(records, list) or not all(
        isinstance(row, Mapping) for row in records
    ):
        return ["report_records_invalid"]
    if not isinstance(attempts, list) or not all(
        isinstance(attempt, Mapping) for attempt in attempts
    ):
        return ["report_attempts_invalid"]

    record_by_key: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in records:
        key = _identity(row)
        if key in record_by_key:
            failures.add("duplicate_record_identity")
        else:
            record_by_key[key] = row

    attempts_by_key: dict[
        tuple[str, str, str], list[Mapping[str, Any]]
    ] = {}
    for attempt in attempts:
        attempts_by_key.setdefault(_identity(attempt), []).append(attempt)

    if set(record_by_key) != set(attempts_by_key):
        failures.add("orphan_or_missing_attempt_group")
    for key, row in record_by_key.items():
        key_errors = m4_row_attempt_accounting_errors(
            row,
            attempts_by_key.get(key, []),
            require_seed_coverage=require_seed_coverage,
        )
        failures.update(f"{'|'.join(key)}:{error}" for error in key_errors)
    return sorted(failures)


def _audit_seeds(
    row: Mapping[str, Any],
    attempts: Sequence[Mapping[str, Any]],
    failures: set[str],
    *,
    require_seed_coverage: bool,
) -> None:
    row_seed_value = row.get("seeds")
    if not isinstance(row_seed_value, list) or any(
        not _is_int(seed) for seed in row_seed_value
    ):
        failures.add("row_seeds_invalid")
        return
    row_seeds = [int(seed) for seed in row_seed_value]
    if len(set(row_seeds)) != len(row_seeds):
        failures.add("row_seeds_not_unique")

    present = [attempt.get("seed") is not None for attempt in attempts]
    if not attempts:
        attempt_seeds: list[int] = []
    elif all(present):
        if any(not _is_int(attempt.get("seed")) for attempt in attempts):
            failures.add("attempt_seed_invalid")
            return
        attempt_seeds = [int(attempt["seed"]) for attempt in attempts]
        if len(set(attempt_seeds)) != len(attempt_seeds):
            failures.add("attempt_seeds_not_unique")
    elif any(present):
        failures.add("attempt_seed_coverage_partial")
        return
    elif require_seed_coverage:
        failures.add("attempt_seed_missing")
        return
    else:
        return

    if sorted(row_seeds) != sorted(attempt_seeds):
        failures.add("row_attempt_seed_mismatch")


def _row_evidence_blocked(
    row: Mapping[str, Any], failures: set[str]
) -> bool:
    blocked = False
    for field in (
        "success_evidence_failures",
        "execution_policy_evidence_failures",
        "contact_evidence_failures",
        "articulation_evidence_failures",
        "touch_evidence_failures",
    ):
        value = row.get(field, [])
        if not isinstance(value, list):
            failures.add(f"{field}_invalid")
            continue
        blocked = blocked or bool(value)
    predicate_blocked = row.get("predicate_blocked_attempts", 0)
    if not _is_int(predicate_blocked) or int(predicate_blocked) < 0:
        failures.add("predicate_blocked_attempts_invalid")
    else:
        blocked = blocked or int(predicate_blocked) > 0
    return blocked


def _identity(value: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(value.get("asset")),
        str(value.get("model_id")),
        str(value.get("use")),
    )


def _nonnegative_int(
    row: Mapping[str, Any], field: str, failures: set[str]
) -> int | None:
    value = row.get(field)
    if not _is_int(value) or int(value) < 0:
        failures.add(f"{field}_invalid")
        return None
    return int(value)


def _finite_float(
    row: Mapping[str, Any], field: str, failures: set[str]
) -> float | None:
    value = row.get(field)
    if isinstance(value, bool):
        failures.add(f"{field}_invalid")
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        failures.add(f"{field}_invalid")
        return None
    if not math.isfinite(number):
        failures.add(f"{field}_invalid")
        return None
    return number


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


__all__ = [
    "M4_BLOCKING_FAILURE_PREFIXES",
    "M4_SOURCE_OR_INTERNAL_FAILURE_PREFIXES",
    "m4_attempt_is_blocking",
    "m4_report_attempt_accounting_errors",
    "m4_row_attempt_accounting_errors",
]
