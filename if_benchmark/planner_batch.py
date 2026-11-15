"""Shared operational batch shape for the CuRobo planning adapter."""

from collections.abc import Mapping
import math
from numbers import Integral, Real
from typing import Any

CUROBO_FIXED_BATCH_SIZE = 16


def normalize_candidate_search_budget(
    value: Mapping[str, Any] | None,
) -> dict[str, int | float] | None:
    """Validate an explicitly bounded candidate search, never a success gate.

    The ratio counts CuRobo pose rows (including padding), not distinct
    Contacts or complete Pick paths. A returned candidate still needs all
    physical/path checks. ``None`` retains the existing planner defaults.
    """

    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {"max_attempts", "success_ratio"}:
        raise ValueError("candidate search budget requires only max_attempts and success_ratio")
    attempts, ratio = value["max_attempts"], value["success_ratio"]
    if isinstance(attempts, bool) or not isinstance(attempts, Integral) or not 1 <= attempts <= 10:
        raise ValueError("candidate max_attempts must be an integer from 1 to 10")
    if (isinstance(ratio, bool) or not isinstance(ratio, Real)
            or not math.isfinite(float(ratio)) or not 0 < ratio <= 1):
        raise ValueError("candidate success_ratio must be finite and in (0, 1]")
    return {"max_attempts": int(attempts), "success_ratio": float(ratio)}


__all__ = ["CUROBO_FIXED_BATCH_SIZE", "normalize_candidate_search_budget"]
