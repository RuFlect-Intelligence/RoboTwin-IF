"""Milestone 8 assignment indexes, solver, artifacts, and selection API."""

from __future__ import annotations

from .core import (
    AssetIndex,
    PairIndex,
    build_asset_index,
    build_assignment_artifacts,
    build_pair_index,
    check_assignment_artifacts,
    freeze_assignment_artifacts,
    select_assignment_candidate,
    solve_assignment_domain,
    validate_assignment_artifacts,
)

__all__ = [
    "AssetIndex",
    "PairIndex",
    "build_asset_index",
    "build_assignment_artifacts",
    "build_pair_index",
    "check_assignment_artifacts",
    "freeze_assignment_artifacts",
    "select_assignment_candidate",
    "solve_assignment_domain",
    "validate_assignment_artifacts",
]
