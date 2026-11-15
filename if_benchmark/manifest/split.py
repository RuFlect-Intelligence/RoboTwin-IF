"""Deterministic constrained, realized-family-atomic Train/Test allocation."""

from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
import hashlib
import math
import warnings
from typing import Any, Mapping, Sequence

from .contracts import ManifestConfig
from .io import seal


def coverage_tokens(graph: Mapping[str, Any]) -> set[str]:
    return {*(f"dcode:{e['dcode']}" for e in graph["edges"]),
            *(f"skill:{k['verb']}|{k['asset']}" for n in graph["nodes"] if n["published"] for k in n["skill_keys"])}


def group_graphs(graphs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for graph in sorted(graphs, key=lambda x: x["scene_spec_id"]):
        if not graph["published"]:
            continue
        gid = graph["scene_group_id"]
        group = groups.setdefault(gid, {"scene_group_id": gid, "scene_spec_ids": [],
                                       "tokens": set(), "scene_count": 0, "node_count": 0, "edge_count": 0})
        group["scene_spec_ids"].append(graph["scene_spec_id"])
        group["tokens"].update(coverage_tokens(graph))
        group["scene_count"] += 1
        group["node_count"] += sum(n["published"] for n in graph["nodes"])
        group["edge_count"] += len(graph["edges"])
    return [{**g, "tokens": sorted(g["tokens"])} for _, g in sorted(groups.items())]


def solve_split(graphs: Sequence[Mapping[str, Any]], config: ManifestConfig) -> dict[str, Any]:
    groups = group_graphs(graphs)
    eligible: dict[str, list[str]] = defaultdict(list)
    for g in groups:
        for token in g["tokens"]:
            eligible[token].append(g["scene_group_id"])
    rare = [{"token": t, "eligible_scene_group_ids": ids} for t, ids in sorted(eligible.items()) if len(ids) < 2]
    base = {"artifact_kind": "m16_split_solution", "config_hash": config.hash,
            "scene_group_count": len(groups), "eligible_coverage": dict(sorted(eligible.items())),
            "target_test_fraction": config.test_fraction, "split_seed": config.split_seed,
            "optimization_priority": ["scene_count", "node_count", "edge_count", "seeded_group_rank"],
            "assignments": [], "infeasible_cells": rare}
    if rare or not groups:
        return seal({**base, "status": "split_infeasible", "reason": "insufficient_eligible_groups" if rare else "empty_publication_graph"})
    import numpy as np
    import scipy
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_array
    if scipy.__version__ != config.solver_version:
        return seal({**base, "status": "split_incomplete", "reason": "solver_version_mismatch",
                     "solver_version": scipy.__version__})
    count, nvars = len(groups), len(groups) + 3
    rows, cols, vals, lo, hi = [], [], [], [], []
    index = {g["scene_group_id"]: i for i, g in enumerate(groups)}

    def constraint(items, lower, upper):
        row = len(lo)
        for column, value in items:
            rows.append(row); cols.append(column); vals.append(float(value))
        lo.append(lower); hi.append(upper)

    for ids in eligible.values():
        constraint(((index[g], 1) for g in ids), 1, len(ids) - 1)
    fraction = Fraction(str(config.test_fraction))
    numerator, denominator = fraction.numerator, fraction.denominator
    dimensions = ("scene_count", "node_count", "edge_count")
    for j, name in enumerate(dimensions):
        values = [g[name] * denominator for g in groups]
        target = numerator * sum(g[name] for g in groups)
        constraint([*enumerate(values), (count + j, -1)], -np.inf, target)
        constraint([*enumerate(values), (count + j, 1)], target, np.inf)
    lower = np.zeros(nvars)
    # Every selected size lies on an integer lattice. Without this bound the
    # LP relaxation promises zero deviation even when, e.g., 10% of 23,892
    # scenes is nonintegral, leaving HiGHS to prove that impossibility by search.
    for j, name in enumerate(dimensions):
        lattice = math.gcd(*(g[name] * denominator for g in groups))
        remainder = numerator * sum(g[name] for g in groups) % lattice
        lower[count + j] = min(remainder, lattice - remainder)
    upper = np.array([1.] * count + [np.inf] * 3)
    objectives = []
    for j in range(3):
        c = np.zeros(nvars); c[count + j] = 1
        objectives.append(c)
    ranks = sorted(range(count), key=lambda i: (
        hashlib.sha256(f"{config.split_seed}:{groups[i]['scene_group_id']}".encode()).hexdigest(), groups[i]["scene_group_id"]))
    c = np.zeros(nvars)
    for rank, i in enumerate(ranks, start=1):
        c[i] = rank
    objectives.append(c)
    diagnostics, result = [], None
    for j, objective in enumerate(objectives):
        matrix = coo_array((np.asarray(vals), (np.asarray(rows), np.asarray(cols))), shape=(len(lo), nvars)).tocsc()
        # HiGHS consumes these native options; scipy warns that they are passed
        # through. Pin threads/seed rather than depending on host CPU count.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Unrecognized options detected")
            result = milp(objective, integrality=np.ones(nvars), bounds=Bounds(lower, upper),
                constraints=LinearConstraint(matrix, np.asarray(lo), np.asarray(hi)),
                options={"presolve": True, "mip_rel_gap": 0.0, "time_limit": config.split_time_limit_seconds,
                         "node_limit": config.split_node_limit, "threads": 1, "random_seed": config.split_seed})
        diagnostics.append({"phase": base["optimization_priority"][j], "status": int(result.status),
                            "objective": float(result.fun) if result.fun is not None else None})
        if result.status != 0:
            return seal({**base, "status": "split_infeasible" if result.status == 2 else "split_incomplete",
                         "reason": "joint_coverage_constraints" if result.status == 2 else "solver_did_not_complete",
                         "solver_version": scipy.__version__, "solver_phases": diagnostics})
        if j < 3:
            optimum = round(float(result.x[count + j]))
            lower[count + j] = upper[count + j] = optimum
    assignments = [{**g, "split": "test" if result.x[i] > .5 else "train"} for i, g in enumerate(groups)]
    # Check the integer solution independently, never trust a rounded incumbent.
    for token, ids in eligible.items():
        eligible_ids = set(ids)
        sides = {a["split"] for a in assignments if a["scene_group_id"] in eligible_ids}
        if sides != {"train", "test"}:
            return seal({**base, "status": "split_incomplete", "reason": "invalid_solver_solution", "token": token})
    actual = {}
    for name in dimensions:
        total = sum(g[name] for g in assignments)
        test = sum(g[name] for g in assignments if g["split"] == "test")
        actual[name] = {"train": total - test, "test": test, "test_fraction": test / total,
                        "deviation": test / total - config.test_fraction}
    return seal({**base, "status": "complete", "solver_version": scipy.__version__,
                 "solver_phases": diagnostics, "assignments": assignments, "actual_ratios": actual})


def audit_split(graphs: Sequence[Mapping[str, Any]], split: Mapping[str, Any]) -> dict[str, Any]:
    issues, seen = [], {}
    for group in split["assignments"]:
        gid = group["scene_group_id"]
        if gid in seen:
            issues.append(f"duplicate group: {gid}")
        seen[gid] = group["split"]
    identities = {side: {name: set() for name in ("groups", "scenes", "observations", "nodes", "edges", "sets")}
                  for side in ("train", "test")}
    coverage = {side: set() for side in ("train", "test")}
    for graph in graphs:
        if not graph["published"]:
            continue
        side = seen.get(graph["scene_group_id"])
        if side not in identities:
            issues.append(f"unassigned group: {graph['scene_group_id']}")
            continue
        values = identities[side]
        values["groups"].add(graph["scene_group_id"])
        values["scenes"].add(graph["scene_spec_id"])
        values["observations"].add(graph["initial_observation"]["observation_id"])
        values["nodes"].update(n["id"] for n in graph["nodes"] if n["published"])
        values["edges"].update(e["id"] for e in graph["edges"])
        values["sets"].update(s["id"] for s in graph["derived_sets"] if s["publication_eligible"])
        coverage[side].update(coverage_tokens(graph))
    leaks = {name: sorted(identities["train"][name] & identities["test"][name]) for name in identities["train"]}
    all_tokens = set(split["eligible_coverage"])
    missing = {side: sorted(all_tokens - values) for side, values in coverage.items()}
    return seal({"artifact_kind": "m16_split_leakage_report", "complete": split["status"] == "complete",
                 "passed": split["status"] == "complete" and not issues and not any(leaks.values()) and not any(missing.values()),
                 "issues": sorted(set(issues)), "intersections": leaks, "missing_coverage": missing,
                 "coverage": {s: sorted(v) for s, v in coverage.items()},
                 "counts": {s: {k: len(v) for k, v in d.items()} for s, d in identities.items()}})


def build_bundles(graphs: Sequence[Mapping[str, Any]], split: Mapping[str, Any], manifest_hash: str) -> dict[str, Any]:
    assignments = {g["scene_group_id"]: g["split"] for g in split["assignments"]}
    bundles, errors = [], []
    for graph in sorted(graphs, key=lambda x: x["scene_spec_id"]):
        if not graph["published"]:
            continue
        nodes = [n for n in graph["nodes"] if n["published"]]
        preflight = graph.get("bundle_preflight", {})
        if not preflight.get("passed"):
            errors.append({"scene_spec_id": graph["scene_spec_id"], "reason": "shared_initial_state_preflight_failed"})
        bundles.append(seal({"id": f"bundle_{graph['scene_spec_id']}", "scene_spec_id": graph["scene_spec_id"],
            "scene_group_id": graph["scene_group_id"], "split": assignments.get(graph["scene_group_id"]),
            "node_ids": sorted(n["id"] for n in nodes), "initial_observation": graph["initial_observation"],
            "preflight": preflight, "scene_graph_hash": graph["artifact_hash"]}))
    return seal({"artifact_kind": "m16_scene_counterfactual_bundles", "manifest_hash": manifest_hash,
                 "split_hash": split["artifact_hash"], "bundles": bundles, "errors": errors,
                 "complete": not errors and split["status"] == "complete"})
