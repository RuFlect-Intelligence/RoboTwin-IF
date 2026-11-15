"""Explicit phase publication scope; never relabel excluded source failures."""
from pathlib import Path

from .contracts import ManifestConfig
from .io import read, verify
from .sources import require


def prepare_scope(path, release, ledger, jobs, failures, audit, config):
    scope = ledger.load(path)
    require(scope.get("artifact_kind") == "m16_phase1_split_preflight", "unsupported phase selection")
    require(scope["config"] == vars(config), "phase config differs from original split")
    require(scope["split"]["status"] == "complete", "incomplete phase split")
    verify(scope["split"])
    selected = scope["selected_graph_hashes"]
    require(bool(selected), "empty phase scope")
    old_root = path.parent / "candidate"
    original = ledger.load(old_root / "benchmark_manifest.yml")
    require(original["artifact_hash"] == scope["source_manifest_hash"], "phase source manifest changed")
    relative_ids = {}
    for sid, digest in selected.items():
        graph = ledger.load(old_root / "scenes" / f"{sid}.json")
        require(graph["artifact_hash"] == digest and graph["published"], "phase graph changed")
        require(not graph["failures"] and graph["bundle_preflight"]["passed"], "blocked phase scene")
        require(all(e["dcode"].split(".")[0] in {"D1a", "D1b", "D1c", "D4"} for e in graph["edges"]),
                "unsupported phase axis")
        relative = str(Path(graph["source_paths"]["scene"]).relative_to(release / "m12_scene_specs"))
        require(relative not in relative_ids, "duplicate scoped scene path")
        relative_ids[relative] = sid
    selected_jobs = [j for j in jobs if j[1] in relative_ids]
    require(len(selected_jobs) == len(selected), "phase scene absent from dispatch")
    outside = [f for f in failures if f.get("relative_path") and f["relative_path"] not in relative_ids]
    failures = [f for f in failures if f not in outside]
    return selected_jobs, failures, {**audit,
        "expected_scene_count": len(selected),
        "fixed_assignments": scope["split"]["assignments"],
        "phase_scope": {
            "version": "m16_fixed_phase1_pick_scope_v1",
            "selection_path": str(path), "selection_hash": scope["artifact_hash"],
            "source_manifest_hash": original["artifact_hash"],
            "source_split_hash": scope["split"]["artifact_hash"],
            "full_input_scene_count": audit["expected_scene_count"],
            "selection_policy": scope["selection_policy"],
            "selection_counts": scope["selection_counts"],
            "excluded_scene_reasons": scope["excluded_scene_reasons"],
            "outside_scope_source_failures": outside,
            "claim": "fixed_selected_four_axis_phase_only",
        }}, selected
