"""Evidence-backed coverage and explicitly denominated survivor diagnostics."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
from typing import Any, Mapping, Sequence

from if_benchmark.orientation_adjustment import orientation_delta_id
from if_benchmark.schema.serialization import artifact_hash, canonical_json
from .io import seal
from .split import coverage_tokens


def goal_values(goal: Mapping[str, Any], dcode: str) -> list[Any]:
    axis, subtype = dcode.split(".", 1)
    if axis == "D4":
        return [goal.get("arm")]
    if axis == "D2b":
        return [goal.get("relation")]
    if axis == "D2a":
        return [orientation_delta_id(goal["params"]["rotation_delta_deg"])
                if subtype == "orientation_adjustment" else goal.get("state")]
    kind = {"D1a": "attribute", "D1b": "spatial", "D1c": "ordinal"}[axis]
    result = []
    for bundle in goal.get("descriptors", {}).values():
        value = bundle.get(kind)
        if not value:
            continue
        if axis == "D1a" and value.get("subtype") == subtype:
            result.append(value["value"])
        elif axis == "D1b":
            result.append("+".join(value.get("predicates", [])))
        elif axis == "D1c" and value.get("basis", subtype) == subtype:
            result.append(value.get("value", value.get("order", value.get("rank"))))
    return result


def coverage_report(graphs: Sequence[Mapping[str, Any]], vocabulary: Mapping[str, Any],
                    split: Mapping[str, Any], parents: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    assignments = {g["scene_group_id"]: g["split"] for g in split["assignments"]}
    cells = {c["id"]: {**c, "candidate_contexts": set(), "published_contexts": set(),
                              "train_contexts": set(), "test_contexts": set(),
                              "source_blocked_contexts": set()} for c in vocabulary.get("declared_cells", [])}
    lookup = {(c["axis"] + "." + c["subtype"], canonical_json(c["value"])): c["id"] for c in vocabulary.get("declared_cells", [])}
    marginals = {side: {name: Counter() for name in ("dcodes", "skills", "assets", "categories")}
                 for side in ("all", "train", "test")}
    unregistered = set()
    processed = {c["parent_set_id"] for g in graphs for c in g["derived_sets"]}
    missing_parents = sorted(p["id"] for p in parents if p["id"] not in processed)
    for parent in parents:
        axis = parent["dcode"].split(".")[0]
        for member in parent["members"]:
            projection = member["axis_value"]
            if isinstance(projection, list):
                kind = {"D1a": "attribute", "D1b": "spatial", "D1c": "ordinal"}[axis]
                goal = {"descriptors": {r["slot"]: {kind: r["descriptor"]} for r in projection}}
            elif isinstance(projection, Mapping):
                goal = projection
            else:
                # Small API fixtures may use scalar axis labels. Source M14
                # records are schema-checked structured projections.
                continue
            for value in goal_values(goal, parent["dcode"]):
                key = (parent["dcode"], canonical_json(value))
                cell_id = lookup.get(key)
                if cell_id is None:
                    unregistered.add(key)
                    continue
                cells[cell_id]["candidate_contexts"].add(parent["id"])
                if parent["id"] not in processed:
                    cells[cell_id]["source_blocked_contexts"].add(parent["id"])
    for graph in graphs:
        nodes = {n["id"]: n for n in graph["nodes"]}
        aliases = {a["task_record_id"]: n for n in graph["nodes"] for a in n["aliases"]}
        for context in graph["derived_sets"]:
            endpoints = [m for member in context["members"] for m in member["parent_members"]]
            endpoints.extend({"task_record_id": m["task_record_id"]} for m in context["removed_endpoints"])
            for member in endpoints:
                node = aliases[member["task_record_id"]]
                for value in goal_values(node["identity"]["goal"], context["dcode"]):
                    key = (context["dcode"], canonical_json(value))
                    cell_id = lookup.get(key)
                    if cell_id is None:
                        unregistered.add(key)
                        continue
                    cell = cells[cell_id]
                    cell["candidate_contexts"].add(context["parent_set_id"])
                    if context["publication_eligible"] and node["published"] and any(m["node_id"] == node["id"] for m in context["members"]):
                        cell["published_contexts"].add(context["id"])
                        side = assignments.get(graph["scene_group_id"])
                        if side:
                            cell[f"{side}_contexts"].add(context["id"])
        if not graph["published"]:
            continue
        sides = ["all"] + ([assignments[graph["scene_group_id"]]] if graph["scene_group_id"] in assignments else [])
        for side in sides:
            m = marginals[side]
            for context in graph["derived_sets"]:
                if context["publication_eligible"]:
                    m["dcodes"][context["dcode"]] += 1
            for node in nodes.values():
                if node["published"]:
                    m["skills"].update(f"{k['verb']}|{k['asset']}" for k in node["skill_keys"])
            m["assets"].update(v["asset"] for v in graph["role_bindings"].values())
            m["categories"].update(v["category"] for v in graph["role_bindings"].values())
    records = []
    for _, cell in sorted(cells.items()):
        original, published = len(cell["candidate_contexts"]), len(cell["published_contexts"])
        cell["status"] = ("complete" if original and published == original else "degraded" if published
                          else "source_blocked" if cell["source_blocked_contexts"] else "dead" if original else "uncovered_upstream")
        cell["claimed"] = published > 0
        records.append({k: sorted(v) if isinstance(v, set) else v for k, v in cell.items()})
    return seal({"artifact_kind": "m16_coverage_report", "cells": records,
        "coverage_complete": bool(records) and all(c["status"] == "complete" for c in records),
        "source_failed_parent_set_ids": missing_parents,
        "unregistered_values": [{"dcode": d, "value": v} for d, v in sorted(unregistered)],
        "marginals": {s: {k: dict(sorted(c.items())) for k, c in m.items()} for s, m in marginals.items()},
        "denominators": {"dcodes": "live_contexts", "skills": "unique_canonical_nodes_per_skill",
                         "assets": "scene_role_bindings", "categories": "scene_role_bindings"}})


def _features(graph: Mapping[str, Any], node: Mapping[str, Any]) -> dict[str, list[str]]:
    task = node["identity"]
    entities = {v["entity_key"]: v for v in graph["role_bindings"].values()}
    subject = entities[task["goal"]["subject"]]
    axes = sorted({a["axis"] for alias in node["aliases"] for a in alias["goal_provenance"]["intended_axes"]})
    metrics = graph.get("descriptor_diagnostics", {}).get("entity_metrics", {}).get(task["goal"]["subject"], {})
    area = metrics.get("area_px")
    layout = graph.get("layout_diagnostics", {})
    positions = layout.get("positions", {})
    entity = task["goal"]["subject"]
    ranks = {axis: str(1 + sum(p[column] < positions[entity][column] for p in positions.values()))
             if entity in positions else "missing" for column, axis in enumerate(("x", "y"))}
    reach = layout.get("reach", {}).get(entity, {})
    fill = metrics.get("bbox_fill")
    return {"asset": [subject["asset"]], "model": [subject["asset"] + "|" + str(subject["model_id"])],
        "category": [subject["category"]], "fusion": ["+".join(axes)], "axis": axes,
        "layout_seed": [str(graph["layout_seed"])], "verb": [task["goal"]["verb"]],
        "arm": [str(task["skill_config"]["execution_arm"])], "route": [task["skill_config"]["route"]],
        "checker": [task["checker_config"]["predicate"]["name"]],
        "world_x_rank": [ranks["x"]], "world_y_rank": [ranks["y"]],
        "reach_zone": [reach.get("zone", "missing")],
        "reachable_arms": ["+".join(sorted(reach.get("reachable_arms", []))) or "missing"],
        "descriptor_bbox_fill": ["missing" if fill is None else "below_half" if fill < .5 else "half_or_more"],
        "descriptor_area": ["missing" if area is None else "below_60" if area < 60 else "60_to_599" if area < 600 else "600_plus"],
        "style": sorted({a["instruction"]["selected"]["style_id"] for a in node["aliases"]})}


def _mutual_information(pairs: Counter) -> float:
    total = sum(pairs.values())
    if not total:
        return 0.0
    left, right = Counter(), Counter()
    for (a, b), n in pairs.items():
        left[a] += n; right[b] += n
    return sum(n / total * math.log2(n * total / (left[a] * right[b])) for (a, b), n in sorted(pairs.items()))


def survivor_report(graphs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    features = defaultdict(Counter)
    contexts = defaultdict(Counter)
    context_features = defaultdict(Counter)
    axis_asset = {"candidate": Counter(), "published": Counter()}
    totals = Counter()
    for graph in graphs:
        totals.update(graph["counts"])
        totals["orphan_nodes"] += len(graph["orphan_node_ids"])
        for node in graph["nodes"]:
            states = {a["admission"]["status"] for a in node["aliases"]}
            evidence = "observed" if node["validated"] or states <= {"validated", "excluded"} else "blocked" if "blocked" in states else "not_executed"
            f = _features(graph, node)
            for axis in f["axis"]:
                axis_asset["candidate"][(axis, f["asset"][0])] += 1
                if node["published"]:
                    axis_asset["published"][(axis, f["asset"][0])] += 1
            for dimension, values in f.items():
                for value in values:
                    c = features[(dimension, value)]
                    c["original"] += 1; c[evidence] += 1
                    c["validated"] += node["validated"]; c["published"] += node["published"]
        for context in graph["derived_sets"]:
            c = contexts[context["dcode"]]
            c["original"] += 1; c[context["status"]] += 1
            c["published"] += context["publication_eligible"]
            c[f"K{context['k_original']}_to_K{context['k_surviving']}"] += 1
            members = {m["node_id"] for m in context["members"]} | {m["node_id"] for m in context["removed_endpoints"]}
            features_in_context = {("dcode", context["dcode"])}
            features_in_context.update((dimension, value) for n in graph["nodes"] if n["id"] in members
                                       for dimension, values in _features(graph, n).items() for value in values)
            for key in features_in_context:
                c = context_features[key]
                c["original"] += 1
                c[context["status"]] += 1
                c["published"] += context["publication_eligible"]
    rows = []
    for (dimension, value), c in sorted(features.items()):
        rows.append({"dimension": dimension, "value": value, **dict(sorted(c.items())),
            "p_pass_given_observed_feature": c["validated"] / c["observed"] if c["observed"] else None,
            "p_published_given_feature": c["published"] / c["original"]})
    return seal({"artifact_kind": "m16_survivor_bias_report", "totals": dict(sorted(totals.items())),
        "scope": "source_joined_scene_graphs; source failures are counted separately in publication_graph_report",
        "features": rows, "contexts_by_dcode": {d: dict(sorted(c.items())) for d, c in sorted(contexts.items())},
        "context_features": [{"dimension": d, "value": v, **dict(sorted(c.items())),
            "p_survives_given_feature": c["published"] / c["original"],
            "p_degrades_given_feature": c["degraded_but_live"] / c["original"]}
            for (d, v), c in sorted(context_features.items())],
        "mutual_information_bits_axis_asset": {stage: _mutual_information(c) for stage, c in axis_asset.items()},
        "denominator_policy": "one canonical node per feature; missing/blocked execution is not a semantic failure",
        "context_denominator_policy": "one parent context per distinct member feature value; survival means publication-eligible",
        "mi_denominator": "one canonical node-axis membership; diagnostic only, not sampling weights"})
