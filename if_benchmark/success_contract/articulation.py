"""Shared articulation state semantics for success checks and execution."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from if_benchmark.schema.serialization import canonical_data

from .contract import SuccessContract


def articulation_direction(state: str, profile: Mapping[str, Any]) -> int | None:
    """Resolve the signed active-joint direction from the reviewed profile."""

    state = str(state)
    positive_use = str(profile.get("positive_use") or "")
    positive_state = {
        "open": "open",
        "close": "close",
        "switch_on": "on",
        "on": "on",
        "switch_off": "off",
        "off": "off",
    }.get(positive_use)
    opposite = {"open": "close", "close": "open", "on": "off", "off": "on"}
    if positive_state is not None:
        if state == positive_state:
            return 1
        if state == opposite[positive_state]:
            return -1
        return None
    if state == "open":
        return 1
    if state == "close":
        return -1
    return None


def articulation_target_interval(
    state: str,
    profile: Mapping[str, Any],
    contract: SuccessContract,
) -> list[float] | None:
    """Return the normalized target interval used by the global checker."""

    interval = (profile.get("target_intervals") or {}).get(str(state))
    if interval is None:
        direction = articulation_direction(state, profile)
        interval_key = {
            "open": "open",
            "close": "close",
            "on": "binary_positive" if direction == 1 else "binary_negative",
            "off": "binary_positive" if direction == 1 else "binary_negative",
        }.get(str(state))
        interval = (
            contract.section("articulation").get("target_intervals") or {}
        ).get(interval_key)
    if not isinstance(interval, Sequence) or isinstance(interval, (str, bytes)):
        return None
    if len(interval) != 2:
        return None
    lower, upper = float(interval[0]), float(interval[1])
    if not (0.0 <= lower <= upper <= 1.0):
        return None
    return [lower, upper]


def resolve_articulation_goal(
    *,
    state: str,
    initial_qpos: float,
    lower: float,
    upper: float,
    state_profile: Mapping[str, Any],
    contract: SuccessContract,
) -> dict[str, Any]:
    """Resolve the feasible state target shared by Checker and Expert.

    The feasible band is the intersection of the reviewed target interval,
    the minimum signed-progress requirement, joint limits, and an optional
    normalized safety interval supplied by the state profile.  The selected
    target is the midpoint of that band, which maximizes terminal margin
    without inventing a second acceptance threshold.
    """

    lower = float(lower)
    upper = float(upper)
    initial = float(initial_qpos)
    span = upper - lower
    base = {
        "state": str(state),
        "initial_qpos": initial,
        "lower": lower,
        "upper": upper,
        "span": span,
    }
    if span <= 0.0:
        return {**base, "status": "blocked", "failure_code": "invalid_joint_limits"}
    direction = articulation_direction(state, state_profile)
    if direction is None:
        return {
            **base,
            "status": "blocked",
            "failure_code": "articulation_direction_profile_missing",
        }
    interval = articulation_target_interval(state, state_profile, contract)
    if interval is None:
        return {
            **base,
            "direction_sign": direction,
            "status": "blocked",
            "failure_code": "articulation_target_interval_missing",
        }
    initial_ratio = (initial - lower) / span
    minimum = float(
        contract.section("articulation")["minimum_signed_normalized_progress"]
    )
    feasible_lower, feasible_upper = interval
    if direction > 0:
        feasible_lower = max(feasible_lower, initial_ratio + minimum)
    else:
        feasible_upper = min(feasible_upper, initial_ratio - minimum)
    safety = state_profile.get("safety_interval")
    if safety is not None:
        if (
            not isinstance(safety, Sequence)
            or isinstance(safety, (str, bytes))
            or len(safety) != 2
        ):
            return {
                **base,
                "direction_sign": direction,
                "target_interval": interval,
                "status": "blocked",
                "failure_code": "articulation_safety_interval_invalid",
            }
        safety_lower, safety_upper = float(safety[0]), float(safety[1])
        feasible_lower = max(feasible_lower, safety_lower)
        feasible_upper = min(feasible_upper, safety_upper)
    if feasible_lower > feasible_upper:
        return canonical_data(
            {
                **base,
                "direction_sign": direction,
                "initial_ratio": initial_ratio,
                "minimum_signed_normalized_progress": minimum,
                "target_interval": interval,
                "feasible_target_interval": [feasible_lower, feasible_upper],
                "status": "blocked",
                "failure_code": "articulation_target_infeasible",
            }
        )
    target_ratio = (feasible_lower + feasible_upper) / 2.0
    target_qpos = lower + span * target_ratio
    return canonical_data(
        {
            **base,
            "direction_sign": direction,
            "initial_ratio": initial_ratio,
            "minimum_signed_normalized_progress": minimum,
            "target_interval": interval,
            "feasible_target_interval": [feasible_lower, feasible_upper],
            "target_ratio": target_ratio,
            "target_qpos": target_qpos,
            "target_interval_margin": min(
                target_ratio - interval[0], interval[1] - target_ratio
            ),
            "status": "resolved",
            "failure_code": None,
        }
    )


__all__ = [
    "articulation_direction",
    "articulation_target_interval",
    "resolve_articulation_goal",
]
