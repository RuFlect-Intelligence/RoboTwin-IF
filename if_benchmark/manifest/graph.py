"""Pure canonical-node admission and complete induced M14 graph projection."""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from typing import Any, Mapping, Sequence

from if_benchmark.schema.serialization import artifact_hash, canonical_data, deterministic_id
from .contracts import issue
from .io import IntegrityError, seal


def node_identity(task: Mapping[str, Any], scene: Mapping[str, Any],
                  source_hashes: Mapping[str, str]) -> dict[str, Any]:
    """Keep complete semantics and execution requirements, omit language aliases."""
    return canonical_data({
        "scene_spec_id": scene["id"],
        "observation_id": scene["initial_observation"]["observation_id"],
        "scene_state_hash": scene["initial_observation"]["scene_state_hash"],
        "goal": task["goal"], "checker_config": task["checker_config"],
        "skill_config": task["skill_config"], "source_hashes": dict(source_hashes),
    })


def skill_keys(task: Mapping[str, Any], bindings: Mapping[str, Any]) -> list[dict[str, str]]:
    entities = {v["entity_key"]: v for v in bindings.values()}
    goal = task["goal"]
    subjects = {goal[k] for k in ("subject", "subject2") if goal.get(k)}
    return [{"verb": goal["verb"], "asset": asset} for asset in
            sorted({entities[entity]["asset"] for entity in subjects})]


def admit_task(task_id: str, attempts: Sequence[Mapping[str, Any]], *,
               attempts_path: str = "", missing_reason: str = "not_executed",
               blocked: bool = False) -> dict[str, Any]:
    """The 2/3 test is local to one original TaskRecord; never pool aliases."""
    ordered = sorted(attempts, key=lambda r: (r.get("seed", -1), r.get("attempt_id", "")))
    reasons: list[str] = []
    complete = len(ordered) == 3 and [r.get("seed") for r in ordered] == [0, 1, 2]
    if not complete:
        reasons.append("incomplete_seed_group" if ordered else missing_reason)
        blocked |= bool(ordered)
    refs, successes = [], 0
    for row in ordered:
        gate = (row.get("status") == "semantic_result"
                and (row.get("descriptor_gate") or {}).get("passed") is True
                and (row.get("t0_checker") or {}).get("status") == "failed")
        success = (gate and row.get("classification") == "intended"
                   and row.get("skill_completed") is True and row.get("stable") is True
                   and not row.get("failure_code")
                   and ((row.get("terminal_checkers") or {}).get(task_id) or {}).get("passed") is True)
        successes += int(success)
        if not gate:
            reasons.append(str(row.get("failure_code") or "task_gate_failed"))
        if row.get("status") != "semantic_result":
            blocked = True
        refs.append({"attempt_id": row["attempt_id"], "attempt_hash": row["attempt_hash"],
                     "task_record_id": task_id, "seed": row["seed"], "path": attempts_path,
                     "classification": row.get("classification"), "qualified_success": success,
                     "failure_code": row.get("failure_code"),
                     "capture": canonical_data((row.get("evidence") or {}).get("expert_capture"))})
    all_gates = complete and all(r.get("status") == "semantic_result"
        and (r.get("descriptor_gate") or {}).get("passed") is True
        and (r.get("t0_checker") or {}).get("status") == "failed" for r in ordered)
    validated = all_gates and successes >= 2 and not blocked
    if ordered and successes < 2:
        reasons.append("fewer_than_two_intended_successes")
    return {"task_record_id": task_id, "validated": validated,
            "status": "blocked" if blocked else "validated" if validated else "excluded" if ordered else "not_executed",
            "success_count": successes,
            "denominator": sum(r.get("status") == "semantic_result" for r in ordered),
            "complete_seed_group": complete, "all_gates_passed": all_gates,
            "classification_counts": dict(sorted(Counter(str(r.get("classification")) for r in ordered).items())),
            "attempts": refs, "reasons": sorted(set(reasons))}


def build_scene_graph(*, scene: Mapping[str, Any], realized: Mapping[str, Any],
                      tasks: Sequence[Mapping[str, Any]], admissions: Sequence[Mapping[str, Any]],
                      parent_sets: Sequence[Mapping[str, Any]], pairs: Sequence[Mapping[str, Any]],
                      attempts: Sequence[Mapping[str, Any]], source_hashes: Mapping[str, str],
                      integrity_blocked: bool = False) -> dict[str, Any]:
    scene_id, family_id = scene["id"], realized["id"]
    by_task = {t["id"]: t for t in tasks}
    by_admission = {a["task_record_id"]: a for a in admissions}
    if len(by_task) != len(tasks) or set(by_task) != set(by_admission):
        raise IntegrityError("task/admission identity mismatch")
    nodes, aliases = {}, {}
    for task_id, task in sorted(by_task.items()):
        identity = node_identity(task, scene, source_hashes)
        node_id = deterministic_id("node", identity)
        aliases[task_id] = node_id
        node = nodes.setdefault(node_id, {"id": node_id, "identity": identity,
            "skill_keys": skill_keys(task, realized["role_bindings"]), "aliases": [],
            "validated": False, "successful_attempts": [], "published": False})
        admission = by_admission[task_id]
        node["aliases"].append({"task_record_id": task_id,
            "goal_variant_id": task["goal_variant_id"],
            "goal_provenance": task["goal_provenance"],
            "instruction": task["instruction"], "admission": admission})
        node["validated"] |= bool(admission["validated"])
    for node in nodes.values():
        node["validated"] &= not integrity_blocked
        success_refs = {}
        if node["validated"]:
            for alias in node["aliases"]:
                a = alias["admission"]
                # A failed alias may contribute successes, but corrupt/incomplete
                # evidence and failed descriptor/t0 gates cannot supply experts.
                if a["all_gates_passed"] and a["status"] != "blocked":
                    for ref in a["attempts"]:
                        if ref["qualified_success"]:
                            old = success_refs.get(ref["attempt_id"])
                            if old is not None and old != ref:
                                raise IntegrityError("conflicting duplicate attempt reference")
                            success_refs[ref["attempt_id"]] = ref
        node["successful_attempts"] = [success_refs[k] for k in sorted(success_refs)]
    overlaps, incomplete_checks = [], []
    for attempt in sorted(attempts, key=lambda r: r["attempt_id"]):
        matched = sorted({aliases[t] for t, result in attempt.get("terminal_checkers", {}).items()
                          if result.get("passed") is True and t in aliases})
        if len(matched) > 1:
            overlaps.append({"attempt_id": attempt["attempt_id"], "attempt_hash": attempt["attempt_hash"],
                             "matched_node_ids": matched, "classification": attempt.get("classification")})
        blocked_nodes = sorted({aliases[t] for t, result in attempt.get("terminal_checkers", {}).items()
                                if result.get("status") == "blocked" and t in aliases})
        if (blocked_nodes and attempt.get("classification") == "intended" and attempt.get("skill_completed") is True
                and attempt.get("stable") is True and not attempt.get("failure_code")):
            incomplete_checks.append({"attempt_id": attempt["attempt_id"], "attempt_hash": attempt["attempt_hash"],
                                      "matched_node_ids": matched, "blocked_node_ids": blocked_nodes})
    by_pair = {p["id"]: p for p in pairs}
    if len(by_pair) != len(pairs):
        raise IntegrityError("duplicate M14 pair id")
    derived, live_edges, failures = [], {}, []
    for parent in sorted(parent_sets, key=lambda s: s["id"]):
        if (not parent.get("matrix_ready") or parent["scene_spec_id"] != scene_id
                or parent["realized_family_id"] != family_id
                or any(parent[k] != scene["initial_observation"][k]
                       for k in ("observation_id", "scene_state_hash"))):
            raise IntegrityError("parent contrast set scene/observation mismatch")
        members_by_node, removed = {}, []
        for member in parent["members"]:
            task_id = member["task_record_id"]
            if task_id not in aliases:
                raise IntegrityError("M14 member has no M13 endpoint")
            node_id = aliases[task_id]
            if nodes[node_id]["validated"]:
                members_by_node.setdefault(node_id, []).append(member)
            else:
                removed.append({"task_record_id": task_id, "node_id": node_id,
                                "evidence": by_admission[task_id]})
        ordered = sorted(members_by_node, key=lambda n: (
            artifact_hash(members_by_node[n][0]["axis_value"]), n))
        k = len(ordered)
        retention = "fully_preserved" if k == parent["k"] else "degraded_but_live" if k >= 2 else "dead_context"
        induced: dict[tuple[str, str], list[str]] = defaultdict(list)
        for pair_id in parent["pair_ids"]:
            if pair_id not in by_pair:
                raise IntegrityError("parent set references missing M14 pair")
            p = by_pair[pair_id]
            left = aliases[p["from_endpoint"]["task_record_id"]]
            right = aliases[p["to_endpoint"]["task_record_id"]]
            if (p.get("status") != "valid" or p.get("intended_dcode") != parent["dcode"]
                    or p.get("actual_dcodes") != [parent["dcode"]]
                    or p.get("scene_spec_id") != scene_id):
                raise IntegrityError("invalid M14 induced edge")
            if left != right and left in members_by_node and right in members_by_node:
                induced[tuple(sorted((left, right)))].append(pair_id)
        expected = {tuple(sorted(p)) for p in combinations(ordered, 2)}
        if set(induced) != expected:
            raise IntegrityError("surviving contrast members do not form an M14 clique")
        conflicts = [o for o in overlaps if len(set(o["matched_node_ids"]) & set(ordered)) >= 2]
        incomplete = [o for o in incomplete_checks if set(o["matched_node_ids"]) & set(ordered)
                      and (set(o["blocked_node_ids"]) - set(o["matched_node_ids"])) & set(ordered)]
        eligible = k >= 2 and not conflicts and not incomplete and not integrity_blocked
        if conflicts:
            failures.append(issue("checker_overlap", owner="checker", blocking=False,
                                  scene_spec_id=scene_id, parent_set_id=parent["id"], evidence=conflicts))
        if incomplete:
            failures.append(issue("checker_crossmatch_incomplete", owner="checker", blocking=False,
                                  scene_spec_id=scene_id, parent_set_id=parent["id"], evidence=incomplete))
        row = {"parent_set_id": parent["id"], "parent_set_hash": artifact_hash(parent),
               "scene_spec_id": scene_id, "scene_group_id": family_id,
               "observation_id": parent["observation_id"], "scene_state_hash": parent["scene_state_hash"],
               "dcode": parent["dcode"], "context_signature": parent["context_signature"],
               "k_original": parent["k"], "k_surviving": k, "status": retention,
               "publication_eligible": eligible, "overlap_evidence": conflicts,
               "incomplete_crossmatch_evidence": incomplete,
               "members": [{"node_id": n, "matrix_index": i,
                            "axis_value": members_by_node[n][0]["axis_value"],
                            "parent_members": members_by_node[n]} for i, n in enumerate(ordered)],
               "removed_endpoints": removed,
               "pair_ids": sorted(p for values in induced.values() for p in values)}
        row["id"] = deterministic_id("dcs", row)
        derived.append(seal(row))
        if eligible:
            for (left, right), pair_ids in sorted(induced.items()):
                identity = {"scene_spec_id": scene_id, "left": left, "right": right, "dcode": parent["dcode"]}
                edge_id = deterministic_id("edge", identity)
                edge = live_edges.setdefault(edge_id, {"id": edge_id, **identity, "pair_ids": [], "derived_set_ids": []})
                edge["pair_ids"] = sorted(set(edge["pair_ids"]) | set(pair_ids))
                edge["derived_set_ids"].append(row["id"])
    incident = {e[key] for e in live_edges.values() for key in ("left", "right")}
    for node in nodes.values():
        node["published"] = node["id"] in incident
    original_edges = {(min(aliases[p["from_endpoint"]["task_record_id"]], aliases[p["to_endpoint"]["task_record_id"]]),
                       max(aliases[p["from_endpoint"]["task_record_id"]], aliases[p["to_endpoint"]["task_record_id"]]),
                       p["intended_dcode"]) for p in pairs
                      if aliases[p["from_endpoint"]["task_record_id"]] != aliases[p["to_endpoint"]["task_record_id"]]}
    return seal({"artifact_kind": "m16_scene_graph", "scene_spec_id": scene_id,
        "scene_group_id": family_id, "abstract_family_id": realized["abstract_family_id"],
        "assignment_id": realized["assignment_id"], "layout_seed": scene["layout_seed"],
        "initial_observation": scene["initial_observation"], "source_hashes": dict(source_hashes),
        "role_bindings": realized["role_bindings"],
        "nodes": [nodes[n] for n in sorted(nodes)], "edges": [live_edges[e] for e in sorted(live_edges)],
        "derived_sets": derived, "published": bool(incident), "failures": failures,
        "orphan_node_ids": sorted(n for n, v in nodes.items() if v["validated"] and n not in incident),
        "counts": {"m13_tasks": len(tasks), "m14_pairs": len(pairs),
                   "original_nodes": len(nodes), "original_edges": len(original_edges),
                   "validated_edges": sum(nodes[a]["validated"] and nodes[b]["validated"] for a, b, _ in original_edges),
                   "validated_nodes": sum(n["validated"] for n in nodes.values()),
                   "published_nodes": len(incident), "published_edges": len(live_edges)}})
