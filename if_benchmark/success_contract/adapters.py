"""Stage-labelled entrypoints that preserve one global checker implementation."""

from __future__ import annotations

from typing import Any, Mapping

from .contract import SuccessContract
from .predicates import evaluate_success


def _evaluate(predicate, arguments, snapshot, contract):
    return evaluate_success(predicate, arguments, snapshot, contract)


evaluate_m4_success = _evaluate
evaluate_m5_terminal_success = _evaluate
evaluate_m5_skill_success = _evaluate
evaluate_m13_success = _evaluate
evaluate_m15_success = _evaluate


__all__ = [
    "evaluate_m4_success",
    "evaluate_m5_terminal_success",
    "evaluate_m5_skill_success",
    "evaluate_m13_success",
    "evaluate_m15_success",
]
