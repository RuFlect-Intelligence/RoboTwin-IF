"""Bound new M12 searches without narrowing frozen evidence eligibility."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from if_benchmark.planner_batch import normalize_candidate_search_budget


LIVE_SEED_BUDGET_EXHAUSTED = "layout_live_seed_budget_exhausted"
SEARCH_BUDGET_CODES = frozenset({
    LIVE_SEED_BUDGET_EXHAUSTED,
    "pick_planning_search_budget_exhausted",
    "search_budget_exhausted",
})


def normalize_search_budget(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    keys = {"new_live_seed_limit", "new_planning_layout_limit"}
    if not isinstance(value, Mapping) or not keys <= set(value) or set(value) - keys - {"candidate_search_budget"}:
        raise ValueError("search_budget requires exactly " + ", ".join(sorted(keys)))
    result = {}
    for key in sorted(keys):
        raw = value[key]
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
            raise ValueError(f"search_budget.{key} must be a positive integer")
        result[key] = raw
    if "candidate_search_budget" in value:
        candidate = normalize_candidate_search_budget(value["candidate_search_budget"])
        if candidate is None:
            raise ValueError("candidate_search_budget must be explicit when present")
        result["candidate_search_budget"] = candidate
    return result


@dataclass
class FamilyLiveSearchBudget:
    policy: dict[str, Any]
    live_seeds: list[int] = field(default_factory=list)
    skipped_seeds: list[int] = field(default_factory=list)

    def start_live_seed(self, seed: int) -> bool:
        if seed in self.live_seeds or seed in self.skipped_seeds:
            raise ValueError("one seed cannot consume the live budget twice")
        if len(self.live_seeds) >= self.policy["new_live_seed_limit"]:
            self.skipped_seeds.append(seed)
            return False
        self.live_seeds.append(seed)
        return True

    def exhausted_metrics(self) -> dict[str, Any]:
        return {
            "search_budget": dict(self.policy),
            "new_live_seed_count": len(self.live_seeds),
            "live_seeds": list(self.live_seeds),
            "physical_feasibility": "not_evaluated",
        }


def contains_limited_planning_failure(value: Any) -> bool:
    """Find an actually evaluated failed target, not a pruned/successful one."""
    if isinstance(value, Mapping):
        if value.get("status") == "failed" and value.get("candidate_search_budget_applied") is True:
            return True
        planning = value.get("last_planning")
        if (
            value.get("plannable") is False
            and value.get("status") in (None, "failed")
            and isinstance(planning, Mapping)
            and planning.get("candidate_search_budget_applied") is True
        ):
            return True
        return any(contains_limited_planning_failure(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return any(contains_limited_planning_failure(child) for child in value)
    return False


def failure_summary_counts(failures: Any, *, budgeted: bool) -> tuple[int, int, int]:
    """Return geometry/seed-unsolved, budget, and input-error counts.

    Legacy reports retain their exact historical summary representation.
    Budget-aware reports separate search exhaustion from proven infeasibility.
    """
    from .core import _layout_seed_failure_disposition

    unsolved = sum(_layout_seed_failure_disposition(row) == "seed_unsolved" for row in failures)
    budget = sum(str(row.get("code")) in SEARCH_BUDGET_CODES for row in failures) if budgeted else 0
    return unsolved - budget, budget, len(failures) - unsolved
