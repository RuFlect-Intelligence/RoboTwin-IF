"""Strict raw M15 ingestion; scene acceptance and expert export are not inputs."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from if_benchmark.minimal_pairs import core as m14
from if_benchmark.scene_validation import core as m15
from if_benchmark.scene_integrity import build_scene_integrity_binding
from if_benchmark.tasks import core as m13
from if_benchmark.initial_pose.runtime import validate_loaded_initial_pose
from if_benchmark.schema.serialization import artifact_hash
from .contracts import issue
from .graph import admit_task, build_scene_graph
from .io import InputLedger, IntegrityError, relative_reference, verify


def require(condition: bool, message: str) -> None:
    if not condition:
        raise IntegrityError(message)


def load_dispatches(collection: Path, release: Path, ledger: InputLedger, *,
                    config_path: Path | None = None) -> tuple[dict, list[dict]]:
    """Consume frozen results; historical producer code hashes are provenance.

    Never locate, import, or hash code in a producer checkout or /tmp snapshot.
    The dispatch self-hash still binds those historical strings unchanged.
    """
    dispatches, failures = {}, []
    config_path = config_path or Path(__file__).parents[1] / "data/scene_validation/scene_validation_config.yml"
    for path in sorted(collection.glob("fused_dispatch*.json")):
        try:
            d = ledger.load(path)
            require(d.get("artifact_kind") == "scene_validation_fused_dispatch", "invalid dispatch kind")
            require(d.get("scope") == "full" and d.get("runtime") == "real", "diagnostic dispatch is not M16 input")
            require(len(d["scene_paths"]) == len(set(d["scene_paths"])), "duplicate dispatch scene")
            dispatches[d["artifact_hash"]] = {"data": d, "path": str(path), "scene_paths": set(d["scene_paths"])}
            # Artifact roots are declared by the source, not a machine prefix map.
            original_release = Path(d["source_roots"]["scene_spec_dir"]).parent
            for original, digest in d.get("source_file_hashes", {}).items():
                p = Path(original)
                if not p.is_relative_to(original_release) or p.suffix not in {".json", ".yml", ".yaml", ".jsonl"}:
                    continue
                try:
                    target = relative_reference(original, str(original_release), release)
                    ledger.pin(target, digest)
                except (OSError, ValueError) as exc:
                    failures.append(issue("dispatch_frozen_artifact_unavailable", dispatch_hash=d["artifact_hash"],
                                          original_path=original, detail=str(exc)))
            try:
                ledger.pin(config_path)
                config = m15.ValidationConfig.load(config_path)
                require(config.hash == d["config_hash"], "dispatch validation policy hash mismatch")
            except (OSError, ValueError) as exc:
                failures.append(issue("dispatch_local_config_mismatch", dispatch_hash=d["artifact_hash"], detail=str(exc)))
        except (OSError, ValueError, KeyError) as exc:
            failures.append(issue("invalid_dispatch", path=str(path), detail=str(exc)))
    if not dispatches:
        failures.append(issue("missing_full_dispatch"))
    return dispatches, failures


def source_join(scene_path: Path, release: Path, ledger: InputLedger,
                exclusions: set[tuple[str, str]]) -> dict[str, Any]:
    s = ledger.load(scene_path)
    scene, family_id = s["scene_spec"], s["realized_family_id"]
    r = ledger.load(release / "m10_realized_families" / f"{family_id}.yml")
    u = ledger.load(release / "m11_units" / f"{family_id}.yml")
    errors = m14._source_failures(s, r) + m14._initial_observation_failures(s)
    require(not errors, str(errors))
    require(s.get("layout_solver_version") == m15.LAYOUT_SOLVER_VERSION, "stale M12 version")
    require(r.get("realization_version") == m15.REALIZATION_VERSION, "stale M10 version")
    require(u.get("unit_build_version") == m15.UNIT_BUILD_VERSION, "stale M11 version")
    require(u.get("artifact_kind") == "unit_build" and u.get("preflight", {}).get("status") == "passed", "invalid M11")
    require(u["realized_family_id"] == family_id and u["source_realization_hash"] == r["artifact_hash"]
            and s["source_unit_hash"] == u["artifact_hash"], "stale M11/M12 source join")
    pick = m15.validated_scene_pick_planning_contract(s, realized_artifact=r, unit_artifact=u)
    variants = {v["id"]: v for v in r["realized_scene_family"]["goal_variants"]}
    tasks, ids, seen_variants = [], set(), set()
    for path in sorted((release / "m13_tasks" / scene["id"]).glob("*.yml")):
        t = ledger.load(path)
        task, variant_id = t["task_record"], t["goal_variant_id"]
        require(task["id"] not in ids and variant_id in variants, "duplicate task or unknown variant")
        require(not m14._task_failures(t, scene_artifact=s, realized_artifact=r, variant=variants[variant_id]),
                "M13 Goal/source alignment failure")
        require(t.get("task_render_version") == m15.TASK_RENDER_VERSION, "stale M13 version")
        checker, skill = task["checker_config"], task["skill_config"]
        require(checker.get("version") == m15.CHECKER_CONFIG_VERSION
                and checker.get("success_contract") == m15.SUCCESS_CONTRACT.identity
                and checker.get("scene_integrity") == build_scene_integrity_binding(task["goal"], scene["entities"], m15.SUCCESS_CONTRACT),
                "stale checker/integrity contract")
        require(skill.get("route") == "direct" and not any(x.get("op") == "handover" for x in skill["steps"]),
                "disabled execution route")
        if task["goal"].get("action_family") == "O2O_relation":
            require(checker.get("relation_contract") == m15.RELATION_CONTRACT.identity, "stale relation contract")
        # Reuse the source renderer's pure alignment validation, including the
        # chosen instruction and checker/skill semantic frame.
        alignment = m13.preflight_task_record(task, scene_spec=scene,
            realized_family=r["realized_scene_family"], goal_variant=variants[variant_id])
        require(alignment["status"] == "passed", f"task rendering alignment failure: {alignment['failures']}")
        m15.preflight_skill_operation_anchors(skill, u, r, scene)
        ids.add(task["id"])
        seen_variants.add(variant_id)
        tasks.append(t)
    missing = set(scene["goal_variant_ids"]) - seen_variants
    require(all((scene["id"], v) in exclusions for v in missing), "unaccounted missing M13 endpoints")
    require(not seen_variants - set(scene["goal_variant_ids"]), "unexpected M13 endpoints")
    return {"scene": s, "realized": r, "unit": u, "tasks": tasks, "pick_planning_contract": pick,
            "routes": {}, "missing_goal_variant_ids": sorted(missing)}


def validate_m14_scene(joined: Mapping[str, Any], sets: list, pairs: list) -> None:
    s, r = joined["scene"], joined["realized"]
    tasks = {t["task_record"]["id"]: t for t in joined["tasks"]}
    by_variant = {t["goal_variant_id"]: t for t in joined["tasks"]}
    for p in pairs:
        require(not m14._validate_pair_record(p), "invalid M14 pair record")
        require(p["source_scene_hash"] == s["artifact_hash"] and p["source_realization_hash"] == r["artifact_hash"], "stale M14 source join")
        for side in ("from_endpoint", "to_endpoint"):
            endpoint = p[side]
            require(endpoint["task_record_id"] in tasks and endpoint == m14._task_endpoint(tasks[endpoint["task_record_id"]]), "M14 endpoint mismatch")
        left = tasks[p["from_endpoint"]["task_record_id"]]["task_record"]
        right = tasks[p["to_endpoint"]["task_record_id"]]["task_record"]
        delta = m14.edge_axis(left["goal"], right["goal"],
            left_intended_axes=left["goal_provenance"]["intended_axes"],
            right_intended_axes=right["goal_provenance"]["intended_axes"])
        require(list(delta.actual_dcodes) == [p["intended_dcode"]] and not delta.unmapped_semantic_deltas, "M14 multi-axis or semantic delta")
    rebuilt, _ = m14.build_contrast_sets(scene_spec=s["scene_spec"],
        realized_family=r["realized_scene_family"], task_artifacts=by_variant, pair_records=pairs)
    require(sorted((x for x in rebuilt if x["matrix_ready"]), key=lambda x: x["id"]) == sorted(sets, key=lambda x: x["id"]), "M14 parent contrast context mismatch")


def _load_routes(result: Mapping[str, Any], joined: dict, collection: Path, ledger: InputLedger) -> list[dict]:
    failures = []
    seen = set()
    by_task = {t["task_record"]["id"]: t for t in joined["tasks"]}
    for ref in result.get("route_reports", []):
        task_id = ref["task_record_id"]
        require(task_id not in seen, "duplicate route reference")
        seen.add(task_id)
        path = collection / "routes" / joined["scene"]["scene_spec"]["id"] / f"{task_id}.json"
        route = ledger.load(path, "report_hash")
        require(route["report_hash"] == ref["report_hash"] and route.get("status") == ref["status"], "route result binding mismatch")
        require(task_id in by_task, "route for unknown task")
        expected_sources = {k: joined[k]["artifact_hash"] for k in ("scene", "realized", "unit")}
        expected_sources["task"] = by_task[task_id]["artifact_hash"]
        require(route.get("artifact_kind") == "exact_scene_route_witness"
                and route.get("scene_spec_id") == result["scene_spec_id"]
                and route.get("task_record_id") == task_id and route.get("source_hashes") == expected_sources,
                "route source identity mismatch")
        if route.get("status") == "witness_found":
            task = by_task[task_id]
            route = m15.validate_route_witness_report(route, scene_spec_id=result["scene_spec_id"], task_record_id=task_id,
                source_hashes={"scene": joined["scene"]["artifact_hash"], "realized": joined["realized"]["artifact_hash"],
                               "unit": joined["unit"]["artifact_hash"], "task": task["artifact_hash"]},
                skill_config=task["task_record"]["skill_config"], unit_artifact=joined["unit"], scene_spec=joined["scene"]["scene_spec"])
            pick = joined["pick_planning_contract"]
            targets = [t for t in pick["request"]["targets"]
                       if t["entity_key"] == route["subject"] and t["arm"] == route["arm"]]
            witnesses = [w for w in pick["witness"]["results"] if w.get("entity_key") == route["subject"]
                         and w.get("arm") == route["arm"] and w.get("status") == "passed"]
            require(len(targets) == len(witnesses) == 1, "route has no unique M12 Pick target/witness")
            target, selected = targets[0], route["selected_witness"]
            request = target["planning_request"]
            require(route["m12_pick_planning_request_hash"] == artifact_hash(request), "route M12 Pick request mismatch")
            require(selected["contact_id"] in request["contact_ids"]
                    and selected["contact_id"] == witnesses[0]["selected_candidate"]["contact_id"]
                    and selected["contact_contract_hash"] == target["contact_contract_binding"]["contact_contract_hash"],
                    "route Contact differs from M12 frozen Pick witness")
            joined["routes"][task_id] = route
        elif route.get("status") != "no_witness_in_search_domain":
            failures.append(issue("route_nonsemantic_failure", task_record_id=task_id, evidence=ref))
    return failures


def _verify_attempt(row: dict, joined: Mapping[str, Any], task: Mapping[str, Any],
                    plan: Mapping[str, Any], scene_folder: Path, original_folder: str,
                    ledger: InputLedger) -> None:
    verify(row, "attempt_hash")
    tid, sid = task["id"], joined["scene"]["scene_spec"]["id"]
    require(row.get("attempt_id") == m15.stable_attempt_id(plan, sid, tid, row["seed"]), "attempt ID mismatch")
    require(row.get("plan_hash") == plan["artifact_hash"] and row.get("config_hash") == plan["config_hash"], "attempt plan/config mismatch")
    require(row.get("scene_spec_id") == sid and row.get("goal_variant_id") == task["goal_variant_id"]
            and row.get("source_scene_hash") == joined["scene"]["artifact_hash"], "attempt scene/goal mismatch")
    source_task = next(t for t in joined["tasks"] if t["task_record"]["id"] == tid)
    require(row.get("source_task_hash") == source_task["artifact_hash"], "attempt task source mismatch")
    if row.get("status") != "semantic_result":
        return
    for key, value in m15.SUCCESS_CONTRACT.identity.items():
        require(row.get(key) == value, "attempt SuccessContract mismatch")
    ids = {t["task_record"]["id"] for t in joined["tasks"]}
    checks = row.get("terminal_checkers") or {}
    initial = row.get("t0_all_checkers") or {}
    require(set(checks) == ids and set(initial) == ids, "incomplete scene cross-match")
    require(row.get("t0_checker") == initial[tid], "intended t0 checker mismatch")
    require(row.get("success_snapshot_hash") and {c.get("success_snapshot_hash") for c in checks.values()} == {row["success_snapshot_hash"]}, "terminal snapshot mismatch")
    require(len({c.get("success_snapshot_hash") for c in initial.values()}) == 1
            and all(c.get("success_snapshot_hash") for c in initial.values()), "initial snapshot mismatch")
    # A failed grasp can leave closure/hold observations unavailable. M15
    # records that checker as blocked/None inside a semantic failure; it is not
    # an infrastructure attempt. Keep the original sentinel and classification.
    require(all((c.get("status") == "blocked" and c.get("passed") is None)
                or (c.get("status") in {"passed", "failed"} and type(c.get("passed")) is bool
                    and (c["status"] == "passed") == c["passed"])
                for c in [*checks.values(), *initial.values()]), "malformed checker result")
    classification = m15.classify_checker_matches(tid, checks)
    require(all(row.get(k) == v for k, v in classification.items()), "terminal classification mismatch")
    evidence = row.get("evidence") or {}
    require(all(name in evidence for name in ("rgb", "depth", "actor_segmentation", "canonical_validation")), "missing initial observation evidence")
    for name in ("rgb", "depth", "actor_segmentation", "expert_capture"):
        if name == "expert_capture":
            capture = evidence.get(name) or {}
            require(capture.get("status") != "complete" or bool(capture.get("path")), "complete expert capture lacks reference")
            if not capture.get("path"):
                continue
        ref = evidence[name]
        path = relative_reference(ref["path"], original_folder, scene_folder)
        pin = ledger.pin(path, ref["sha256"])
        require(pin["size_bytes"] == ref["size_bytes"], "observation evidence size mismatch")
    canonical = evidence["canonical_validation"]
    require(set(canonical) == set(joined["scene"]["scene_spec"]["entities"]), "canonical state evidence entity mismatch")
    require(all(v.get("passed") is True for v in canonical.values()), "canonical initial state failed")
    for entity, value in canonical.items():
        frozen = joined["scene"]["scene_spec"]["entities"][entity]
        initial_pose = frozen["initial_pose"]
        require(value.get("canonical_state_hash") == initial_pose["canonical_state_hash"], "initial canonical-state hash mismatch")
        require(value.get("geometry_hash") == initial_pose["geometry_hash"], "initial geometry hash mismatch")
        expected = value["expected"]
        require(expected["position"] == frozen["pose"]["position"]
                and expected["quat_wxyz"] == frozen["pose"]["quat_wxyz"]
                and expected["scale_xyz"] == initial_pose["scale_xyz"], "variant-specific initial pose")
        joint_state = initial_pose.get("articulated_joint_state") or {}
        require(expected["joint_qpos"] == joint_state.get("qpos", [])
                and expected["joints"] == joint_state.get("joints", [])
                and value.get("assignment_state_revision") == initial_pose.get("assignment_state_revision"),
                "variant-specific initial articulation state")
        for key in ("initial_state_id", "revision", "revision_hash"):
            require(value[key] == initial_pose[key], "initial state revision mismatch")
        actual = value["actual"]
        require(m15._pose_within_tolerance(expected["position"], expected["quat_wxyz"],
                    actual["position"], actual["quat_wxyz"]), "actual initial pose differs from SceneSpec")
        validate_loaded_initial_pose(canonical_state=initial_pose, actual_q_wxyz=actual["quat_wxyz"],
            actual_joint_qpos=actual["joint_qpos"], actual_scale_xyz=actual["scale_xyz"],
            actual_origin_z=actual["origin_z"], expected_world_yaw_deg=initial_pose.get("world_yaw_deg", 0.0))


def bundle_preflight(graph: Mapping[str, Any], rows: list[dict]) -> dict:
    semantic = [r for r in rows if r.get("status") == "semantic_result"]
    tasks = {a["task_record_id"] for n in graph["nodes"] if n["published"] for a in n["aliases"]}
    t0_false = bool(semantic) and all(
        (r.get("t0_all_checkers", {}).get(tid) or {}).get("status") == "failed"
        for r in semantic for tid in tasks)
    states = {artifact_hash({entity: {"expected": value.get("expected"),
        "canonical_state_hash": value.get("canonical_state_hash"),
        "assignment_state_revision": value.get("assignment_state_revision")}
        for entity, value in r["evidence"]["canonical_validation"].items()}) for r in semantic}
    return {"passed": t0_false and len(states) == 1,
        "all_published_checkers_false_at_t0": t0_false,
        "shared_expected_state_hashes": sorted(states),
        "d2a_present": any(e["dcode"].startswith("D2a.") for e in graph["edges"]),
        "alternative_reachability": "each_published_node_has_original_two_of_three_success"}


def process_scene(job: tuple, context: Mapping[str, Any]) -> dict[str, Any]:
    """Bounded scene unit: raw attempts never escape into global aggregation."""
    original_path, relative_path = job
    release, collection = Path(context["release"]), Path(context["collection"])
    ledger, failures = InputLedger(), []
    scene_folder = collection / "scenes" / hashlib.sha256(original_path.encode()).hexdigest()[:24]
    result_path = scene_folder / "result.json"
    joined, graph = None, None
    try:
        joined = source_join(release / "m12_scene_specs" / relative_path, release, ledger, context["exclusions"])
        inventory = context["m12_inventory"][relative_path]
        require(joined["scene"]["artifact_hash"] == inventory["artifact_hash"], "SceneSpec differs from M12 publication inventory")
        ledger.pin(release / "m12_scene_specs" / relative_path, inventory["byte_sha256"])
        family = joined["realized"]["realized_scene_family"]
        assignment = family["evidence"]["assignment"]
        require(assignment["candidate_hash"] == context["candidate_hashes"].get((family["abstract_family_id"], family["assignment_id"])), "M10 assignment candidate lineage mismatch")
        goal_relative = Path("m09_goal_variants") / family["abstract_family_id"] / f"{family['assignment_id']}.yml"
        goals = ledger.load(release / goal_relative)
        require(goals["goal_variants"] == family["goal_variants"], "M9/M10 GoalVariant mismatch")
        expected_sources = {**context["lineage_hashes"], str(goal_relative): goals["artifact_hash"]}
        for original, digest in joined["realized"]["source_hashes"].items():
            relative = str(Path(original).relative_to(context["declared_release_root"]))
            require(expected_sources.get(relative) == digest, "M10 upstream source lineage mismatch")
        for ref in assignment["evidence_refs"]:
            if ref.get("kind") == "asset_validation":
                if ref.get("stage") == "static_geometry":
                    require(ref.get("status") == "validated" and bool(ref.get("record_id")), "invalid inline static geometry evidence")
                else:
                    require(ref.get("hash") in context["evidence_hashes"], "asset validation source is not the frozen release")
        sid = joined["scene"]["scene_spec"]["id"]
        sets, pairs = context["sets"].get(sid, []), context["pairs"].get(sid, [])
        validate_m14_scene(joined, sets, pairs)
        result = ledger.load(result_path)
        require(result.get("scene_path") == original_path and result.get("scene_spec_id") in {None, sid}, "scene result identity mismatch")
        parent = context["dispatches"].get(result.get("dispatch_hash"))
        require(parent is not None and original_path in parent["scene_paths"], "result parent dispatch mismatch")
        require(result.get("status") != "no_tasks" or not joined["tasks"], "no_tasks result has M13 tasks")
        failures.extend(_load_routes(result, joined, collection, ledger))
        rows, grouped, attempts_path = [], defaultdict(list), scene_folder / "attempts.jsonl"
        if result.get("status") not in {"completed", "route_failed", "no_tasks"}:
            failures.append(issue("m15_nonsemantic_scene_failure", owner="infrastructure", scene_spec_id=sid,
                                  status=result.get("status"), evidence=result.get("errors", [])))
        if result.get("attempts_path") is not None:
            require(Path(result["attempts_path"]).name == "attempts.jsonl", "unexpected attempts filename")
            plan = ledger.load(scene_folder / "validation_plan.yml")
            d = parent["data"]
            require(plan.get("execution_only") is True and plan.get("publication_ready") is False
                    and plan.get("artifact_kind") == "scene_validation_execution_plan", "invalid M15 execution child")
            require(plan.get("parent_dispatch_hash") == d["artifact_hash"] and plan.get("config_hash") == d["config_hash"]
                    and plan.get("source_roots") == d["source_roots"] and plan.get("scope") == d["scope"], "child parent contract mismatch")
            require(Path(plan["parent_dispatch_path"]).name == Path(parent["path"]).name, "child parent path mismatch")
            require(plan["artifact_hash"] == result.get("child_plan_hash") and len(plan["scenes"]) == 1, "child plan hash mismatch")
            child = plan["scenes"][0]
            require(child["scene_path"] == original_path and child["scene_spec_id"] == sid, "child scene mismatch")
            require(all(child[k] == joined[j]["artifact_hash"] for k, j in (
                ("source_scene_hash", "scene"), ("source_realization_hash", "realized"), ("source_unit_hash", "unit"))), "child source join mismatch")
            expected_tasks = {t["task_record"]["id"]: t["artifact_hash"] for t in joined["tasks"]}
            require(len(child["task_records"]) == len(expected_tasks)
                    and {t["task_record_id"]: t["source_task_hash"] for t in child["task_records"]} == expected_tasks, "child task set mismatch")
            ledger.pin(attempts_path)
            original_folder = str(Path(result["attempts_path"]).parent)
            by_task = {t["task_record"]["id"]: t["task_record"] for t in joined["tasks"]}
            seen = set()
            with attempts_path.open() as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    require(row.get("task_record_id") in by_task and row.get("seed") in [0, 1, 2], "unexpected attempt task/seed")
                    require(row.get("attempt_id") not in seen, "duplicate attempt")
                    _verify_attempt(row, joined, by_task[row["task_record_id"]], plan, scene_folder, original_folder, ledger)
                    seen.add(row["attempt_id"])
                    grouped[row["task_record_id"]].append(row)
                    rows.append(row)
            require(len(rows) == result.get("attempt_count"), "scene attempt count mismatch")
            if any(len(grouped[t]) != 3 for t in by_task):
                failures.append(issue("incomplete_task_attempts", scene_spec_id=sid))
            descriptors = {artifact_hash(row["descriptor_gate"]) for row in rows if row.get("descriptor_gate")}
            require(len(descriptors) <= 1, "non-invariant descriptor observation")
            for row in rows:
                if row.get("status") != "semantic_result":
                    failures.append(issue("m15_nonsemantic_attempt", scene_spec_id=sid, attempt_id=row["attempt_id"], status=row.get("status")))
        elif result.get("status") == "completed" and joined["tasks"]:
            raise IntegrityError("completed scene has no attempt evidence")
        admissions = []
        for t in joined["tasks"]:
            tid = t["task_record"]["id"]
            needs_route = any(s.get("op") == "carry_to_relation" for s in t["task_record"]["skill_config"]["steps"])
            if grouped[tid] and needs_route:
                require(tid in joined["routes"], "executed task lacks exact route witness")
            reason = "task_route_failed" if needs_route and tid not in joined["routes"] else "scene_route_gate_not_executed"
            admissions.append(admit_task(tid, grouped[tid], attempts_path=str(attempts_path), missing_reason=reason))
        graph = build_scene_graph(scene=joined["scene"]["scene_spec"], realized=joined["realized"]["realized_scene_family"],
            tasks=[t["task_record"] for t in joined["tasks"]], admissions=admissions,
            parent_sets=sets, pairs=pairs, attempts=rows,
            source_hashes={k: joined[k]["artifact_hash"] for k in ("scene", "realized", "unit")},
            integrity_blocked=False)
        graph["assignment_evidence"] = joined["realized"]["realized_scene_family"]["evidence"]["assignment"]
        graph["m15_status"] = result.get("status")
        graph["source_result_hash"] = result["artifact_hash"]
        graph["source_paths"] = {"scene": str(release / "m12_scene_specs" / relative_path),
            "realized": str(release / "m10_realized_families" / f"{graph['scene_group_id']}.yml"),
            "unit": str(release / "m11_units" / f"{graph['scene_group_id']}.yml"),
            "tasks": [str(p) for p in sorted((release / "m13_tasks" / sid).glob("*.yml"))],
            "result": str(result_path), "attempts": str(attempts_path) if rows else None}
        graph["descriptor_diagnostics"] = next((r["descriptor_gate"] for r in rows if r.get("descriptor_gate")), {})
        graph["layout_diagnostics"] = {"reach": joined["scene"]["scene_spec"].get("reach", {}),
            "positions": {k: v["pose"]["position"] for k, v in joined["scene"]["scene_spec"]["entities"].items()}}
        graph["bundle_preflight"] = bundle_preflight(graph, rows)
        from .io import seal
        graph = seal(graph)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        graph = None
        failures.append(issue("scene_source_integrity_failure", path=str(result_path), detail=str(exc),
                              scene_spec_id=joined["scene"]["scene_spec"]["id"] if joined else None))
    except Exception as exc:
        graph = None
        failures.append(issue("m16_internal_failure", owner="internal", path=str(result_path), detail=f"{type(exc).__name__}: {exc}"))
    try:
        ledger.stable()
    except OSError as exc:
        failures.append(issue("input_disappeared", detail=str(exc)))
    except IntegrityError as exc:
        failures.append(issue("input_changed", detail=str(exc)))
    return {"graph": graph, "failures": failures, "input_files": ledger.records(), "input_stats": ledger.files}
