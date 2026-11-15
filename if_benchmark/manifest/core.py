"""Artifact-driven M16 build/check/validate/freeze orchestration."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
from typing import Any, Callable, Mapping

from if_benchmark.schema.serialization import artifact_hash
from if_benchmark.scene_validation import core as m15
from .contracts import ARTIFACTS, DEFAULT_COLLECTION, DEFAULT_OUTPUT, DEFAULT_RELEASE, VERSION, ManifestConfig, issue
from .io import (InputLedger, IntegrityError, atomic_write, file_hash, parallel_map, read,
                 relative_reference, seal, source_file_error, source_stamps, stamp, verify, verify_stamps)
from .reports import coverage_report, survivor_report
from .sources import load_dispatches, process_scene, require
from .split import audit_split, build_bundles, solve_split


_CONTEXT: dict = {}


def _code_hash(code_root: Path | None = None) -> str:
    """Bind local interpretation code, including shared semantic validators.

    Data mounts and historical producer checkouts are never traversed.
    """
    from .reader_compatibility import code_files
    return artifact_hash(code_files(code_root or Path(__file__).parents[1]))


def _worker(job: tuple) -> dict:
    cache = Path(_CONTEXT["cache"]) / (hashlib.sha256(job[0].encode()).hexdigest() + ".json")
    expected = {"job": list(job), "build_key": _CONTEXT["build_key"]}
    if cache.is_file():
        try:
            previous = read(cache)
            verify(previous)
            if previous["identity"] == expected and all(
                    stamp(Path(row["path"])) == tuple(row["stat"]) for row in previous["result"]["input_stats"].values()):
                return previous["result"]
        except (OSError, ValueError, KeyError):
            pass
    result = process_scene(job, _CONTEXT)
    # Never reuse absent-source/infrastructure failures: a later repair must be
    # observed, including files that did not yet exist in the old input ledger.
    if not result["failures"]:
        atomic_write(cache, seal({"identity": expected, "result": result}))
    return result


def _bounded_results(jobs, workers: int):
    if workers == 1:
        for job in jobs:
            yield _worker(job)
        return
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("fork")) as executor:
        iterator, pending = iter(jobs), deque()
        for _ in range(workers * 2):
            job = next(iterator, None)
            if job is not None:
                pending.append(executor.submit(_worker, job))
        while pending:
            yield pending.popleft().result()
            job = next(iterator, None)
            if job is not None:
                pending.append(executor.submit(_worker, job))


def _prepare(release: Path, collection: Path, ledger: InputLedger, progress=None) -> tuple[dict, list[tuple], list[dict], dict]:
    failures, upstream = [], {}
    run = ledger.load(release / "run_manifest.yml", None)
    for stage in ("m4_release", "m5_release", "m6_release", "m7_release", "m8_release", "m9_release", "m10_release", "m11_release", "m12_release", "m13_release", "m14_release"):
        require(stage in run, f"missing release stage: {stage}")
        for key, value in run[stage].items():
            if key.endswith("_sha256"):
                source_key = key[:-7]
                if source_key in run[stage]:
                    ledger.pin(release / run[stage][source_key], value)
        upstream[stage] = run[stage]
    documents = {}
    specs = (
        ("m14", "m14_minimal_pairs/minimal_pairs.yml", "artifact_hash"),
        ("m14_audit", "m14_minimal_pairs/minimal_pair_audit.yml", "audit_hash"),
        ("m14_exclusions", "m14_minimal_pairs/invalid_pair_report.yml", "report_hash"),
        ("m12", "m12_scene_specs/layout_report.yml", "report_hash"),
        ("m13", "m13_tasks/instruction_report.yml", "report_hash"),
        ("m13_checker", "m13_tasks/checker_alignment_report.yml", "report_hash"),
        ("vocabulary", "m07_families/axis_vocabulary.yml", None),
    )
    for name, relative, key in specs:
        if progress:
            progress(f"manifest: verifying {relative}")
        documents[name] = ledger.load(release / relative, key)
    # M10's assignment and M4/M5 lineage is verified once at the global boundary;
    # scene workers consume the compact exact-candidate hash index.
    lineage_hashes, candidate_hashes = {}, {}
    for name in ("abstract_families.yml", "archetypes.yml"):
        relative = "m07_families/" + name
        lineage_hashes[relative] = artifact_hash(ledger.load(release / relative, None))
    for path in sorted((release / "m08_assignment_domains").glob("*.yml")):
        if path.name in {"domain_summary.yml", "empty_domain_report.yml"}:
            continue
        if progress:
            progress(f"manifest: indexing {path.name}")
        domain = ledger.load(path, "domain_hash")
        require(domain.get("artifact_kind") == "assignment_domain", "invalid M8 domain kind")
        lineage_hashes[str(path.relative_to(release))] = domain["domain_hash"]
        for candidate in domain["candidates"]:
            key = (domain["abstract_family_id"], candidate["id"])
            require(key not in candidate_hashes, "duplicate assignment candidate")
            candidate_hashes[key] = artifact_hash(candidate)
    evidence_hashes = set()
    for stage, keys in (("m4_release", ("report",)), ("m5_release", ("terminal_report", "canonical_skill_report"))):
        for key in keys:
            value = ledger.load(release / run[stage][key], "report_hash")
            evidence_hashes.add(value["report_hash"])
            if stage == "m4_release":
                require(value["report_hash"] == run[stage]["report_hash"], "M4 release report identity mismatch")
    m14 = documents["m14"]
    require(m14["artifact_hash"] == documents["m14_audit"]["artifact_set_hash"], "M14 audit source mismatch")
    require(m14["source_artifact_set_hashes"]["scene_specs"] == documents["m12"]["artifact_set_hash"]
            and m14["source_artifact_set_hashes"]["tasks"] == documents["m13"]["artifact_set_hash"], "M14 M12/M13 artifact set mismatch")
    require(documents["m13"]["artifact_set_hash"] == documents["m13_checker"]["artifact_set_hash"]
            and documents["m13"]["config_hash"] == documents["m13_checker"]["config_hash"], "M13 exclusion report mismatch")
    for name in ("m14", "m14_audit", "m12", "m13", "m13_checker"):
        require(documents[name].get("publication_ready") is True, f"upstream not publication-ready: {name}")
    manifest_ref = documents["m12"].get("artifact_manifest")
    require(manifest_ref is not None, "M16 requires explicit M12 published artifact inventory")
    scenes = ledger.load(release / "m12_scene_specs" / manifest_ref["path"], "report_hash")
    require(scenes["report_hash"] == manifest_ref["report_hash"], "M12 inventory mismatch")
    published = {r["path"]: r for r in scenes["records"]}
    require(len(published) == len(scenes["records"]), "duplicate M12 published path")
    dispatches, dispatch_failures = load_dispatches(collection, release, ledger)
    failures.extend(dispatch_failures)
    paths = {}
    for parent in dispatches.values():
        d = parent["data"]
        for original in d["scene_paths"]:
            relative = str(Path(original).relative_to(Path(d["source_roots"]["scene_spec_dir"])))
            require(".." not in Path(relative).parts and relative in published, "dispatch scene outside M12 publication")
            require(relative not in paths or paths[relative] == original, "multiple source spellings for one scene")
            paths[relative] = original
    for relative in sorted(set(published) - set(paths)):
        failures.append(issue("scene_missing_from_dispatch", relative_path=relative))
    pairs, sets = defaultdict(list), defaultdict(list)
    for kind, dest in (("pairs", pairs), ("contrast_sets", sets)):
        seen = set()
        for row in m14[kind]:
            require(row["id"] not in seen, f"duplicate M14 {kind} id")
            seen.add(row["id"])
            dest[row["scene_spec_id"]].append(row)
    exclusions = {(f["scene_spec_id"], f["goal_variant_id"]) for name in ("m13", "m13_checker")
                  for f in documents[name].get("failures", []) if f.get("scene_spec_id") and f.get("goal_variant_id")}
    context = {"release": str(release), "collection": str(collection), "dispatches": dispatches,
               "pairs": dict(pairs), "sets": dict(sets), "exclusions": exclusions,
               "m12_inventory": published, "declared_release_root": run["root"],
               "lineage_hashes": lineage_hashes, "candidate_hashes": candidate_hashes,
               "evidence_hashes": evidence_hashes}
    audit = {"upstream": upstream, "vocabulary": documents["vocabulary"], "m14_parents": m14["contrast_sets"],
             "m14_hash": m14["artifact_hash"], "expected_scene_count": len(published),
             "m14_counts": documents["m14_audit"]["summary"],
             "upstream_exclusion_reports": {name: {"hash": documents[name].get("report_hash"),
                "summary": documents[name].get("summary"), "failure_code_counts": documents[name].get("failure_code_counts")}
                for name in ("m13", "m13_checker", "m14_exclusions")}}
    return context, [(paths[p], p) for p in sorted(paths)], failures, audit


def _scene_summary(graph: Mapping[str, Any]) -> dict:
    return {"scene_spec_id": graph["scene_spec_id"], "scene_group_id": graph["scene_group_id"],
        "observation_id": graph["initial_observation"]["observation_id"],
        "scene_state_hash": graph["initial_observation"]["scene_state_hash"],
        "scene_graph_hash": graph["artifact_hash"], "path": f"scenes/{graph['scene_spec_id']}.json",
        "node_ids": sorted(n["id"] for n in graph["nodes"] if n["published"]),
        "edge_ids": [e["id"] for e in graph["edges"]], "published": graph["published"], "counts": graph["counts"]}


def _assemble(graphs: list[dict], failures: list[dict], inputs: list[dict], audit: dict,
              config: ManifestConfig, build_key: str) -> dict[str, dict]:
    graph_failures = [f for g in graphs for f in g["failures"]]
    all_failures = failures + graph_failures
    manifest = seal({"artifact_kind": "m16_benchmark_manifest", "version": VERSION,
        "config_hash": config.hash, "build_key": build_key, "source_m14_hash": audit["m14_hash"],
        "implementation_hash": _code_hash(),
        "scenes": [_scene_summary(g) for g in graphs], "source_input_hash": artifact_hash(inputs),
        "expected_scene_count": audit["expected_scene_count"],
        "graph_scope": "candidate_until_publish_block_report_and_freeze_receipt_pass",
        **({"phase_scope": audit["phase_scope"]} if "phase_scope" in audit else {})})
    split = solve_split(graphs, config)
    if "fixed_assignments" in audit and split.get("assignments") != audit["fixed_assignments"]:
        all_failures.append(issue("phase_split_membership_changed", owner="split"))
    split = seal({**split, "manifest_hash": manifest["artifact_hash"]})
    leakage = audit_split(graphs, split)
    bundles = build_bundles(graphs, split, manifest["artifact_hash"])
    coverage = coverage_report(graphs, audit["vocabulary"], split, audit.get("m14_parents", ()))
    if coverage["unregistered_values"]:
        all_failures.append(issue("unregistered_coverage_values", evidence=coverage["unregistered_values"]))
    if split["status"] != "complete":
        all_failures.append(issue(split["status"], owner="split", reason=split.get("reason"), evidence=split["infeasible_cells"]))
    if not leakage["passed"]:
        all_failures.append(issue("split_leakage_audit_incomplete", owner="split"))
    for error in bundles["errors"]:
        all_failures.append(issue(error["reason"], owner="scene_state", scene_spec_id=error["scene_spec_id"]))
    ready = bool(any(g["published"] for g in graphs)) and not any(f["blocking"] for f in all_failures)
    summaries = [g["counts"] for g in graphs]
    counts = {k: sum(s[k] for s in summaries) for k in sorted({k for s in summaries for k in s})}
    counts.update(scene_count=len(graphs), retained_scene_count=sum(g["published"] for g in graphs),
                  source_failed_scene_count=audit["expected_scene_count"] - len(graphs))
    documents = {
        "benchmark_manifest.yml": manifest,
        "derived_contrast_sets.yml": seal({"artifact_kind": "m16_derived_contrast_sets", "manifest_hash": manifest["artifact_hash"],
            "records": [s for g in graphs for s in g["derived_sets"]]}),
        "publication_graph_report.yml": seal({"artifact_kind": "m16_publication_graph_report", "counts": counts,
            "m14_counts": audit["m14_counts"], "orphans": [{"scene_spec_id": g["scene_spec_id"], "node_ids": g["orphan_node_ids"]}
                for g in graphs if g["orphan_node_ids"]], "upstream_exclusions": audit["upstream_exclusion_reports"]}),
        "coverage_report.yml": coverage,
        "survivor_bias_report.yml": survivor_report(graphs),
        "train_test_split.yml": split,
        "scene_counterfactual_bundles.yml": bundles,
        "split_feasibility_report.yml": seal({"artifact_kind": "m16_split_feasibility_report",
            **{k: v for k, v in split.items() if k not in {"artifact_hash", "artifact_kind", "assignments"}}}),
        "split_leakage_report.yml": leakage,
        "input_evidence_index.yml": seal({"artifact_kind": "m16_input_evidence_index", "files": inputs}),
        "task_admission_index.yml": seal({"artifact_kind": "m16_task_admission_index", "records": [
            {"task_record_id": a["task_record_id"], "node_id": n["id"], "scene_spec_id": g["scene_spec_id"],
             "scene_graph_path": f"scenes/{g['scene_spec_id']}.json", "status": a["admission"]["status"],
             "success_count": a["admission"]["success_count"], "denominator": a["admission"]["denominator"],
             "reasons": a["admission"]["reasons"]} for g in graphs for n in g["nodes"] for a in n["aliases"]]}),
    }
    documents["publish_block_report.yml"] = seal({"artifact_kind": "m16_publish_block_report", "version": VERSION,
        "publication_ready": ready, "coverage_complete": coverage["coverage_complete"], "counts": counts,
        "failures": sorted(all_failures, key=lambda f: json.dumps(f, sort_keys=True)),
        "failure_code_counts": dict(sorted(Counter(f["code"] for f in all_failures).items())),
        "artifact_hashes": {name: value["artifact_hash"] for name, value in sorted(documents.items())}})
    return documents


def build_manifest(*, release_root: Path = DEFAULT_RELEASE, m15_root: Path = DEFAULT_COLLECTION,
                   output_dir: Path = DEFAULT_OUTPUT, config: ManifestConfig | None = None,
                   workers: int = 8, progress: Callable[[str], None] = print,
                   scope_from: Path | None = None) -> dict:
    config = config or ManifestConfig()
    config.validate()
    if workers < 1:
        raise ValueError("workers must be positive")
    release_root, m15_root, output_dir = Path(release_root), Path(m15_root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / ".build.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _build(release_root, m15_root, output_dir, config, workers, progress, scope_from)


def _build(release: Path, collection: Path, output: Path, config: ManifestConfig, workers: int, progress,
           scope_from: Path | None = None) -> dict:
    global _CONTEXT
    code_hash = _code_hash()
    ledger = InputLedger()
    candidate = output / "candidate"
    candidate.mkdir(exist_ok=True)
    progress("manifest: checking frozen M4-M14 controls and raw M15 dispatches")
    try:
        context, jobs, failures, audit = _prepare(release, collection, ledger, progress)
    except (OSError, ValueError, KeyError) as exc:
        report = seal({"artifact_kind": "m16_publish_block_report", "publication_ready": False,
                       "failures": [issue("input_preflight_failed", detail=str(exc))], "artifact_hashes": {}})
        atomic_write(candidate / "publish_block_report.yml", report)
        return report
    require(_code_hash() == code_hash, "M16 implementation changed during source preparation; restart build")
    selected_hashes = None
    if scope_from is not None:
        from .scope import prepare_scope
        jobs, failures, audit, selected_hashes = prepare_scope(
            Path(scope_from), release, ledger, jobs, failures, audit, config)
    build_key = artifact_hash({"config": config.hash, "implementation": code_hash,
                              "controls": ledger.records(), "jobs": jobs})
    _CONTEXT = {**context, "cache": str(output / ".cache" / "scenes"), "build_key": build_key}
    graphs, input_files = [], {r["path"]: r for r in ledger.records()}
    input_stats = {k: tuple(v["stat"]) for k, v in ledger.files.items()}
    progress(f"manifest: processing {len(jobs)} scenes with {workers} CPU workers")
    for i, result in enumerate(_bounded_results(jobs, workers), start=1):
        failures.extend(result["failures"])
        for row in result["input_files"]:
            if row["path"] in input_files and input_files[row["path"]] != row:
                failures.append(issue("input_changed_between_scenes", path=row["path"]))
            input_files[row["path"]] = row
        for path, row in result["input_stats"].items():
            current = tuple(row["stat"])
            if path in input_stats and input_stats[path] != current:
                failures.append(issue("input_metadata_changed_between_scenes", path=path))
            input_stats[path] = current
        graph = result["graph"]
        if graph is not None:
            if selected_hashes is not None and selected_hashes.get(graph["scene_spec_id"]) != graph["artifact_hash"]:
                failures.append(issue("phase_scene_graph_changed", scene_spec_id=graph["scene_spec_id"]))
            graphs.append(graph)
            atomic_write(candidate / "scenes" / f"{graph['scene_spec_id']}.json", graph)
        if i == 1 or i % 250 == 0 or i == len(jobs):
            progress(f"manifest: scenes {i}/{len(jobs)}, graphs {len(graphs)}, blocking records {sum(f['blocking'] for f in failures)}")
            atomic_write(output / "status.json", {"phase": "scenes", "done": i, "total": len(jobs), "build_key": build_key})
    graphs.sort(key=lambda g: g["scene_spec_id"])
    if selected_hashes is not None and set(selected_hashes) != {g["scene_spec_id"] for g in graphs}:
        failures.append(issue("phase_scene_membership_changed"))
    require(len({g["scene_spec_id"] for g in graphs}) == len(graphs), "duplicate scene graph")
    try:
        verify_stamps(input_stats, workers=workers)
    except (OSError, ValueError) as exc:
        failures.append(issue("input_controls_changed", detail=str(exc)))
    if _code_hash() != code_hash:
        failures.append(issue("builder_changed_during_run", owner="internal"))
    progress("manifest: solving constrained split and assembling coverage/bundle audits")
    documents = _assemble(graphs, failures, [input_files[k] for k in sorted(input_files)], audit, config, build_key)
    atomic_write(candidate / "manifest_config.yml", asdict(config))
    for name in ARTIFACTS:
        atomic_write(candidate / name, documents[name])
    atomic_write(output / "status.json", {"phase": "complete", "build_key": build_key,
        "publication_ready": documents["publish_block_report.yml"]["publication_ready"]})
    return documents["publish_block_report.yml"]


def _artifact_root(output: Path) -> Path:
    return output if (output / "benchmark_manifest.yml").is_file() else output / "candidate"


def check_manifest(*, output_dir: Path = DEFAULT_OUTPUT, verify_sources: bool = True, workers: int = 8,
                   source_binding: Path | None = None, live_sources: bool = False) -> dict:
    from .reader_compatibility import implementation_matches
    from .source_snapshots import resolve_sources
    root, errors = _artifact_root(Path(output_dir)), []
    resolution = {"mode": "not_checked"}
    source_count = 0
    try:
        code_hash = _code_hash()
        report = read(root / "publish_block_report.yml"); verify(report)
        require(set(report["artifact_hashes"]) == set(ARTIFACTS) - {"publish_block_report.yml"}, "incomplete M16 artifact set")
        inputs = None
        for name, expected in report["artifact_hashes"].items():
            value = read(root / name); verify(value)
            require(value["artifact_hash"] == expected, f"stale M16 artifact: {name}")
            if name == "input_evidence_index.yml":
                inputs = value
        manifest = read(root / "benchmark_manifest.yml")
        require(implementation_matches(manifest["implementation_hash"]), "M16 implementation changed; rebuild required")
        files, binding_ledger, resolution = resolve_sources(root, inputs, binding_path=source_binding, live_sources=live_sources)
        config = ManifestConfig.load(root / "manifest_config.yml")
        require(config.hash == manifest["config_hash"], "M16 config hash mismatch")
        for ref in manifest["scenes"]:
            require(ref["path"] == f"scenes/{ref['scene_spec_id']}.json"
                    and "/" not in ref["scene_spec_id"] and ".." not in ref["scene_spec_id"], "invalid scene graph path")
            graph = read(root / ref["path"]); verify(graph)
            require(graph["artifact_hash"] == ref["scene_graph_hash"], f"stale scene graph: {ref['path']}")
        if verify_sources:
            for error in parallel_map(source_file_error, files, workers=workers):
                source_count += 1
                if error:
                    errors.append(error)
        binding_ledger.stable()
        require(_code_hash() == code_hash, "local implementation changed during check")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append(str(exc))
    return {"valid": not errors, "stale": bool(errors), "errors": errors,
            "source_resolution": resolution, "sources_verified": verify_sources and not errors,
            "source_file_count": source_count,
            "verifier_implementation_hash": _code_hash()}


def validate_manifest(*, output_dir: Path = DEFAULT_OUTPUT, verify_sources: bool = True, workers: int = 8,
                      source_binding: Path | None = None, live_sources: bool = False) -> dict:
    from .validation import validate_publication
    from .reader_compatibility import implementation_matches
    from .source_snapshots import resolve_sources
    root = _artifact_root(Path(output_dir))
    try:
        code_hash = _code_hash()
        manifest = read(root / "benchmark_manifest.yml")
        verify(manifest)
        require(implementation_matches(manifest["implementation_hash"]), "M16 implementation changed; rebuild required")
        inputs = read(root / "input_evidence_index.yml"); verify(inputs)
        files, binding_ledger, resolution = resolve_sources(root, inputs, binding_path=source_binding, live_sources=live_sources)
        source_stats = source_stamps(files, workers=workers)
        checked = check_manifest(output_dir=root, verify_sources=verify_sources, workers=workers,
                                 source_binding=source_binding, live_sources=live_sources)
        if not checked["valid"]:
            return checked
        result = validate_publication(root, workers=workers)
        verify_stamps(source_stats, workers=workers)
        binding_ledger.stable()
        require(_code_hash() == code_hash, "local implementation changed during validation")
        return {**result, "source_resolution": resolution, "sources_verified": verify_sources,
                "source_file_count": checked["source_file_count"],
                "verifier_implementation_hash": code_hash}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"valid": False, "errors": [f"invalid publication artifact: {exc}"]}


def freeze_manifest(*, output_dir: Path = DEFAULT_OUTPUT, workers: int = 8) -> dict:
    output = Path(output_dir)
    candidate = output / "candidate"
    with (output / ".build.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        checked = validate_manifest(output_dir=candidate, workers=workers)
        if not checked["valid"]:
            return {"frozen": False, "errors": checked["errors"]}
        report = read(candidate / "publish_block_report.yml")
        if not report["publication_ready"]:
            return {"frozen": False, "errors": ["publication blocked"], "failure_code_counts": report["failure_code_counts"]}
        revision = output / ".frozen" / report["artifact_hash"]
        if not revision.exists():
            revision.parent.mkdir(exist_ok=True)
            staging = revision.with_name("." + revision.name + ".tmp")
            require(not staging.exists(), "incomplete freeze staging exists; inspect before retry")
            staging.mkdir()
            for name in (*ARTIFACTS, "manifest_config.yml"):
                shutil.copy2(candidate / name, staging / name)
            (staging / "scenes").mkdir()
            for ref in read(candidate / "benchmark_manifest.yml")["scenes"]:
                shutil.copy2(candidate / ref["path"], staging / ref["path"])
            checked = check_manifest(output_dir=staging, workers=workers)
            require(checked["valid"], f"freeze copy/source verification failed: {checked['errors']}")
            os.replace(staging, revision)
        else:
            checked = check_manifest(output_dir=revision, verify_sources=False)
            if not checked["valid"]:
                return {"frozen": False, "errors": ["existing frozen revision is corrupt", *checked["errors"]]}
        for name in (*ARTIFACTS, "manifest_config.yml", "scenes"):
            link = output / name
            require((link.is_symlink() and link.readlink() == Path("current") / name)
                    or (not link.exists() and not link.is_symlink()), f"unowned publication path: {link}")
            if not link.is_symlink():
                link.symlink_to(Path("current") / name)
        temporary = output / ".current.tmp"
        require(not temporary.exists() and not temporary.is_symlink(), "freeze pointer staging already exists")
        temporary.symlink_to(Path(".frozen") / report["artifact_hash"])
        os.replace(temporary, output / "current")
        receipt = seal({"artifact_kind": "m16_freeze_receipt", "publication_ready": True,
                        "publish_block_report_hash": report["artifact_hash"], "revision": str(revision)})
        atomic_write(output / "M16_FROZEN.json", receipt)
        return {"frozen": True, "receipt": receipt}
