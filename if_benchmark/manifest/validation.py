"""Independent M16 structural and source-evidence acceptance checks."""

from __future__ import annotations

from itertools import combinations
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Mapping

import jsonschema

from if_benchmark.schema.serialization import artifact_hash, deterministic_id
from .contracts import ARTIFACTS
from .graph import admit_task, build_scene_graph, node_identity
from .io import IntegrityError, parallel_map, read, verify
from .split import audit_split, build_bundles, group_graphs
from .sources import bundle_preflight
from .reports import coverage_report, survivor_report


def graph_errors(graph: Mapping[str, Any], parents: Mapping[str, Any] | None = None) -> list[str]:
    errors = []

    def expect(value, message):
        if not value:
            errors.append(f"{graph['scene_spec_id']}: {message}")

    nodes = {n["id"]: n for n in graph["nodes"]}
    edges = {e["id"]: e for e in graph["edges"]}
    expect(len(nodes) == len(graph["nodes"]) and len(edges) == len(graph["edges"]), "duplicate node/edge")
    published = {n["id"] for n in nodes.values() if n["published"]}
    incident = {e[k] for e in edges.values() for k in ("left", "right")}
    expect(published == incident, "published nodes differ from edge incidence")
    expect(graph["published"] == bool(edges), "scene retention mismatch")
    orphan = {n["id"] for n in nodes.values() if n["validated"] and not n["published"]}
    expect(orphan == set(graph["orphan_node_ids"]), "orphan partition mismatch")
    all_aliases = set()
    for node in nodes.values():
        expect(node["id"] == deterministic_id("node", node["identity"]), "canonical node ID mismatch")
        expect(node["identity"]["scene_spec_id"] == graph["scene_spec_id"]
               and node["identity"]["scene_state_hash"] == graph["initial_observation"]["scene_state_hash"], "cross-scene node")
        expect(node["validated"] == any(a["admission"]["validated"] for a in node["aliases"]), "canonical alias admission mismatch")
        expect(not node["published"] or node["validated"], "unvalidated published node")
        success_refs = {}
        for alias in node["aliases"]:
            tid, a = alias["task_record_id"], alias["admission"]
            expect(tid not in all_aliases, "duplicate task alias")
            all_aliases.add(tid)
            refs = a["attempts"]
            expect(len({r["attempt_id"] for r in refs}) == len(refs), "duplicate alias attempt")
            expect(a["success_count"] == sum(r["qualified_success"] for r in refs), "alias success count mismatch")
            if a["validated"]:
                expect(a["all_gates_passed"] and a["denominator"] == 3
                       and [r["seed"] for r in refs] == [0, 1, 2] and a["success_count"] >= 2,
                       "alias did not meet original 2/3 admission")
            if node["validated"] and a["all_gates_passed"] and a["status"] != "blocked":
                for ref in refs:
                    if ref["qualified_success"]:
                        success_refs[ref["attempt_id"]] = ref
        expect(node["successful_attempts"] == [success_refs[k] for k in sorted(success_refs)], "successful reference union mismatch")
    live_edge_keys, live_ids = set(), set()
    for context in graph["derived_sets"]:
        try:
            verify(context)
        except IntegrityError as exc:
            errors.append(str(exc))
        members = [m["node_id"] for m in context["members"]]
        k = len(set(members))
        expect(k == len(members) == context["k_surviving"], "derived K or member uniqueness mismatch")
        expect([m["matrix_index"] for m in context["members"]] == list(range(k)), "derived matrix indices mismatch")
        expect(all(n in nodes and nodes[n]["validated"] for n in members), "invalid surviving member")
        expect(context["scene_spec_id"] == graph["scene_spec_id"]
               and context["scene_group_id"] == graph["scene_group_id"]
               and context["observation_id"] == graph["initial_observation"]["observation_id"]
               and context["scene_state_hash"] == graph["initial_observation"]["scene_state_hash"], "derived context crossed scene/observation")
        status = "fully_preserved" if k == context["k_original"] else "degraded_but_live" if k >= 2 else "dead_context"
        expect(context["status"] == status, "derived retention classification mismatch")
        if parents is not None:
            parent = parents.get(context["parent_set_id"])
            expect(parent is not None, "unknown M14 parent")
            if parent is not None:
                expect(artifact_hash(parent) == context["parent_set_hash"], "M14 parent hash mismatch")
                expect(parent["k"] == context["k_original"] and parent["context_signature"] == context["context_signature"], "M14 parent context mismatch")
                expect(set(context["pair_ids"]) <= set(parent["pair_ids"]), "invented derived pair")
        if context["publication_eligible"]:
            expect(k >= 2 and not context["overlap_evidence"] and not context["incomplete_crossmatch_evidence"],
                   "dead/ambiguous/incomplete context published")
            expect(len(context["pair_ids"]) >= k * (k - 1) // 2, "missing induced clique pairs")
            live_ids.add(context["id"])
            live_edge_keys.update((min(a, b), max(a, b), context["dcode"]) for a, b in combinations(members, 2))
    edge_keys = set()
    for edge in edges.values():
        expect(edge["left"] != edge["right"], "self edge")
        expect(edge["scene_spec_id"] == graph["scene_spec_id"], "cross-scene edge")
        expect(set(edge["derived_set_ids"]) <= live_ids and edge["derived_set_ids"], "edge without live context")
        edge_keys.add((min(edge["left"], edge["right"]), max(edge["left"], edge["right"]), edge["dcode"]))
    expect(edge_keys == live_edge_keys, "publication edges differ from live clique union")
    return errors


def _source_admission_errors(graph: Mapping[str, Any], parents: list | None = None,
                             pairs: list | None = None) -> list[str]:
    errors, raw = [], []
    source_paths = graph["source_paths"]
    tasks = {}
    for path in source_paths["tasks"]:
        value = read(Path(path)); verify(value)
        tasks[value["task_record"]["id"]] = value["task_record"]
    if source_paths["attempts"]:
        with Path(source_paths["attempts"]).open() as stream:
            raw = [json.loads(line) for line in stream if line.strip()]
        for row in raw:
            verify(row, "attempt_hash")
    for node in graph["nodes"]:
        for alias in node["aliases"]:
            tid = alias["task_record_id"]
            if tid not in tasks:
                errors.append(f"unknown source task {tid}")
                continue
            task = tasks[tid]
            scene = {"id": graph["scene_spec_id"], "initial_observation": graph["initial_observation"]}
            if node_identity(task, scene, graph["source_hashes"]) != node["identity"]:
                errors.append(f"canonical identity differs from source: {tid}")
            rows = [r for r in raw if r["task_record_id"] == tid]
            saved = alias["admission"]
            if rows:
                rebuilt = admit_task(tid, rows, attempts_path=source_paths["attempts"])
                if rebuilt != saved:
                    errors.append(f"admission differs from raw attempts: {tid}")
            elif saved["validated"] or saved["success_count"]:
                errors.append(f"unexecuted alias promoted: {tid}")
    if {a["task_record_id"] for n in graph["nodes"] for a in n["aliases"]} != set(tasks):
        errors.append("source task omitted from canonical graph")
    if parents is not None and pairs is not None:
        source_scene = read(Path(source_paths["scene"])); verify(source_scene)
        source_realized = read(Path(source_paths["realized"])); verify(source_realized)
        admissions = [a["admission"] for n in graph["nodes"] for a in n["aliases"]]
        rebuilt = build_scene_graph(scene=source_scene["scene_spec"], realized=source_realized["realized_scene_family"],
            tasks=list(tasks.values()), admissions=admissions, parent_sets=parents, pairs=pairs,
            attempts=raw, source_hashes=graph["source_hashes"])
        for key, value in rebuilt.items():
            if key != "artifact_hash" and graph[key] != value:
                errors.append(f"graph differs from source projection: {graph['scene_spec_id']} {key}")
        if bundle_preflight(graph, raw) != graph["bundle_preflight"]:
            errors.append(f"bundle initial state differs from raw evidence: {graph['scene_spec_id']}")
    return errors


def _validate_scene(job: tuple) -> tuple[dict, list[str]]:
    root, ref, parents, pairs = job
    graph = read(root / ref["path"])
    verify(graph)
    errors = graph_errors(graph, {p["id"]: p for p in parents})
    if graph["artifact_hash"] != ref["scene_graph_hash"]:
        errors.append(f"scene graph changed during validation: {ref['path']}")
    try:
        errors.extend(_source_admission_errors(graph, parents, pairs))
    except (OSError, ValueError, KeyError) as exc:
        errors.append(f"{graph['scene_spec_id']}: {exc}")
    return graph, errors


def validate_publication(root: Path, *, workers: int = 8) -> dict[str, Any]:
    errors = []
    schema = read(Path(__file__).with_name("manifest.schema.json"))
    validator = jsonschema.Draft202012Validator(schema)
    docs = {name: read(root / name) for name in ARTIFACTS}
    for name, doc in docs.items():
        errors.extend(f"{name}: {e.message}" for e in validator.iter_errors(doc))
    manifest = docs["benchmark_manifest.yml"]
    inputs = docs["input_evidence_index.yml"]["files"]
    if manifest["source_input_hash"] != artifact_hash(inputs):
        errors.append("manifest input index binding mismatch")
    m14_paths = [Path(row["path"]) for row in inputs if row["path"].endswith("/m14_minimal_pairs/minimal_pairs.yml")]
    parents = None
    parents_by_scene, pairs_by_scene = defaultdict(list), defaultdict(list)
    if len(m14_paths) == 1:
        m14 = read(m14_paths[0]); verify(m14)
        if m14["artifact_hash"] != manifest["source_m14_hash"]:
            errors.append("manifest M14 source mismatch")
        parents = {s["id"]: s for s in m14["contrast_sets"]}
        for s in m14["contrast_sets"]:
            parents_by_scene[s["scene_spec_id"]].append(s)
        for p in m14["pairs"]:
            pairs_by_scene[p["scene_spec_id"]].append(p)
    else:
        errors.append("missing unique M14 source")
    graphs = []
    jobs = ((root, ref, parents_by_scene[ref["scene_spec_id"]], pairs_by_scene[ref["scene_spec_id"]])
            for ref in manifest["scenes"])
    for graph, scene_errors in parallel_map(_validate_scene, jobs, workers=workers, processes=True):
        errors.extend(scene_errors)
        graphs.append(graph)
    from .core import _scene_summary
    if [_scene_summary(g) for g in graphs] != manifest["scenes"]:
        errors.append("manifest scene summaries differ from graph records")
    split = docs["train_test_split.yml"]
    if split["manifest_hash"] != manifest["artifact_hash"]:
        errors.append("split manifest binding mismatch")
    groups = group_graphs(graphs)
    if split["status"] == "complete":
        actual = [{k: v for k, v in g.items() if k != "split"} for g in split["assignments"]]
        if actual != groups:
            errors.append("split group graph weights or membership mismatch")
    if audit_split(graphs, split) != docs["split_leakage_report.yml"]:
        errors.append("split leakage report differs from recomputed audit")
    if build_bundles(graphs, split, manifest["artifact_hash"]) != docs["scene_counterfactual_bundles.yml"]:
        errors.append("bundle source is not the complete scene V16")
    if docs["derived_contrast_sets.yml"]["records"] != [s for g in graphs for s in g["derived_sets"]]:
        errors.append("global derived sets differ from scene records")
    vocab_paths = [Path(r["path"]) for r in inputs if r["path"].endswith("/m07_families/axis_vocabulary.yml")]
    if len(vocab_paths) != 1 or coverage_report(graphs, read(vocab_paths[0]), split, list((parents or {}).values())) != docs["coverage_report.yml"]:
        errors.append("coverage report differs from source cell registry and graph")
    if survivor_report(graphs) != docs["survivor_bias_report.yml"]:
        errors.append("survivor report differs from graph")
    expected_admissions = [{"task_record_id": a["task_record_id"], "node_id": n["id"], "scene_spec_id": g["scene_spec_id"],
        "scene_graph_path": f"scenes/{g['scene_spec_id']}.json", "status": a["admission"]["status"],
        "success_count": a["admission"]["success_count"], "denominator": a["admission"]["denominator"],
        "reasons": a["admission"]["reasons"]} for g in graphs for n in g["nodes"] for a in n["aliases"]]
    if docs["task_admission_index.yml"]["records"] != expected_admissions:
        errors.append("task admission index differs from graph")
    report = docs["publish_block_report.yml"]
    if dict(Counter(f["code"] for f in report["failures"])) != report["failure_code_counts"]:
        errors.append("failure summary differs from records")
    counts = dict(sum((Counter(g["counts"]) for g in graphs), Counter()))
    # Counter addition drops zero entries; the serialized count keys remain explicit.
    counts = {k: counts.get(k, 0) for g in graphs for k in g["counts"]}
    counts.update(scene_count=len(graphs), retained_scene_count=sum(g["published"] for g in graphs),
                  source_failed_scene_count=manifest["expected_scene_count"] - len(graphs))
    if report["counts"] != counts or docs["publication_graph_report.yml"]["counts"] != counts:
        errors.append("publication counts differ from graph")
    if report["publication_ready"]:
        if any(f["blocking"] for f in report["failures"]) or split["status"] != "complete" or not docs["scene_counterfactual_bundles.yml"]["complete"]:
            errors.append("publication-ready flag overrides an unresolved blocker")
    return {"valid": not errors, "errors": errors, "publication_ready": report["publication_ready"],
            "scene_count": len(graphs)}
