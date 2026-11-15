"""Immutable Mixed 50 panel and runtime input verification."""

from collections import Counter, defaultdict
from hashlib import file_digest
from pathlib import Path
import json
import re

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ID = Path(__file__).with_name("bundle-id.txt")
VERSION = "mixed50-v1"


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe(root, ref):
    rel = Path(ref)
    require(not rel.is_absolute() and ".." not in rel.parts and rel.parts, f"Invalid relative path {ref}")
    path = root / rel
    require(path.resolve().is_relative_to(root.resolve()), f"Path escapes bundle {ref}")
    return path


def load(root, verify_files=True):
    root = root.resolve()
    sums = root / "SHA256SUMS"
    identity = digest(sums)
    require(identity == BUNDLE_ID.read_text().strip(), "Wrong Mixed 50 bundle or altered checksum list")
    listed = {}
    for line in sums.read_text().splitlines():
        expected, ref = line.split("  ", 1)
        require(ref not in listed and re.fullmatch(r"[a-f0-9]{64}", expected), "Invalid checksum list")
        path = safe(root, ref)
        require(path.is_file(), f"Missing bundle file {ref}")
        if verify_files or ref == "panel/manifest.json":
            require(digest(path) == expected, f"Checksum mismatch for {ref}")
        listed[ref] = expected
    panel = json.loads((root / "panel/manifest.json").read_text())
    require(panel["version"] == VERSION and panel["seeds"] == list(range(10)), "Unexpected benchmark protocol")
    scenes = {s["scene_spec_id"]: s for s in panel["scenes"]}
    require(len(scenes) == len(panel["scenes"]) == 50, "Expected 50 unique scenes")
    for scene in scenes.values():
        paths = scene["source_paths"]
        for ref in [paths["scene"], paths["realized"], paths["unit"], *paths["tasks"]]:
            require(ref in listed, f"Unbound runtime source {ref}")
        require(paths["tasks"] or scene.get("runtime_tasks"), "Scene is missing complete task context")
    pairs = defaultdict(list)
    ids = set()
    for node in panel["nodes"]:
        nid = node["node_id"]
        require(nid not in ids and re.fullmatch(r"[a-zA-Z0-9_-]+", nid), "Invalid or duplicate task node")
        ids.add(nid)
        require(node["scene_spec_id"] in scenes, "Task references an unknown scene")
        require(node["instruction"] == node["task"]["instruction"]["selected"]["text"], "Instruction mismatch")
        pairs[node["scene_spec_id"]].append(node)
    require(pairs.keys() == scenes.keys(), "Missing scene in evaluation roster")
    pools = Counter()
    numbers = set()
    for pair in pairs.values():
        require(len(pair) == 2 and pair[0]["pool"] == pair[1]["pool"]
                and pair[0]["number"] == pair[1]["number"] and pair[0]["instruction"] != pair[1]["instruction"]
                and sum(n["pretraining_instruction_selected"] for n in pair) == 1, "Invalid instruction pair")
        pools[pair[0]["pool"]] += 1
        numbers.add(pair[0]["number"])
    require(pools == {"pick": 25, "d2b": 25} and numbers == set(range(1, 51)), "Unexpected scene roster")
    panel["identity"] = identity
    panel["file_count"] = len(listed)
    return panel


def episodes(panel, smoke=False):
    scene_numbers = {1, 26} if smoke else set(range(1, 51))
    seeds = [0] if smoke else panel["seeds"]
    return [
        {"id": f"{node['node_id']}_seed{seed}", "node_id": node["node_id"],
         "scene_spec_id": node["scene_spec_id"], "seed": seed}
        for number in sorted(scene_numbers) for seed in seeds
        for node in panel["nodes"] if node["number"] == number
    ]
