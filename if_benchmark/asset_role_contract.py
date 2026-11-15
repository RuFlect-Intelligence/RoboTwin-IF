"""Shared mapping from catalog candidate uses to BenchmarkV1 asset roles."""

from __future__ import annotations


CANDIDATE_TO_ASSET_ROLE = {
    "pick_subject": "manipuland",
    "place_in_receiver": "container",
    "place_on_receiver": "surface",
    "stack_base": "stack_base",
    "relation_anchor": "reference",
    "articulated_open_close": "articulated",
    "activate_subject": "articulated",
}
