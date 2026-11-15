import argparse
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys

from .grounding import score_trace
from .panel import ROOT, digest, episodes, load, require, safe
from if_benchmark.manifest.io import atomic_write, read, seal, verify


class PolicyBridge:
    __name__ = "robotwin_if.custom_policy"

    def __init__(self, state_reader):
        self.state_reader = state_reader
        self.state_readers = {}

    def reset_model(self, model):
        model.reset(self.seed)

    def eval(self, env, model, observation):
        action = model.act(observation, env.get_instruction())
        env.take_sampled_joint_action(action, period_s=.04)

    def observe(self, env, obs):
        cameras = obs["observation"]
        reader = self.state_readers.get(id(env.robot))
        if reader is None:
            reader = self.state_readers[id(env.robot)] = self.state_reader(env.robot)
        return {
            "rgb": {name: cameras[name]["rgb"] for name in
                    ("head_camera", "left_camera", "right_camera")},
            "joint_position": reader.read()["observation.state"],
            "endpose": obs["endpose"],
        }


def verify_result(directory, identity, episode):
    path = directory / "result.json"
    record = read(path)
    verify(record)
    require(record["episode"] == episode and record["run_id"] == identity, "Attempt binding changed")
    refs = {}
    for entry in record["evidence"]:
        filename = entry["path"]
        require(filename not in refs, "Duplicate physical evidence file")
        refs[filename] = entry["sha256"]
        require(digest(safe(directory, filename)) == refs[filename], f"Changed attempt evidence {filename}")
    if record["status"] == "semantic_result":
        require("physics.jsonl.gz" in refs and type(record["success"]) is bool
                and type(record["grounding"]["passed"]) is bool, "Missing semantic evidence")
    return record


def run(args, panel):
    import random
    import numpy as np
    import torch
    from if_benchmark.evaluation.contracts import EvaluationPolicy
    from if_benchmark.evaluation.runner import execute_episode
    from if_benchmark.scene_validation.expert_capture import RobotStateReader
    from .observer import AnyArmPhysicalObserver

    require((ROOT / "assets").is_dir() and (ROOT / "assets").samefile(args.bundle / "assets"),
            "Extract the data archive into the checkout, or link checkout/assets to the bundle assets")
    require(torch.cuda.is_available(), "Mixed 50 simulation requires CUDA and a GPU with Vulkan support")
    spec = importlib.util.find_spec(args.policy)
    require(spec is not None and spec.origin and Path(spec.origin).is_file(), "Policy module not found")
    config = json.loads(args.config.read_text()) if args.config else {}
    require(isinstance(config, dict), "Policy config must be a JSON object")
    roster = episodes(panel, args.smoke)
    protocol = EvaluationPolicy(max_actions=1000, success_seconds=40., success_check_steps=20)
    runtime_folders = ("envs", "if_benchmark", "robotwin_if", "task_config")
    implementation = {p.relative_to(ROOT).as_posix(): digest(p)
                      for folder in runtime_folders for p in sorted((ROOT / folder).rglob("*"))
                      if p.is_file() and p.suffix in (".py", ".yml", ".yaml")}
    binding = seal({
        "bundle": panel["identity"], "scope": "smoke" if args.smoke else "mixed50",
        "policy": args.policy, "policy_module_sha256": digest(Path(spec.origin)),
        "config_sha256": seal({"config": config})["artifact_hash"], "action": "joint14",
        "policy_revision": args.policy_revision,
        "implementation_sha256": seal({"files": implementation})["artifact_hash"],
    })
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    scenes = {s["scene_spec_id"]: deepcopy(s) for s in panel["scenes"]}
    for scene in scenes.values():
        scene["source_paths"] = {
            key: [str(safe(args.bundle, ref)) for ref in value] if isinstance(value, list)
            else str(safe(args.bundle, value)) for key, value in scene["source_paths"].items()
        }
    nodes = {node["node_id"]: node for node in panel["nodes"]}
    with (output / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        job = output / "run.json"
        if job.exists():
            require(read(job) == binding, "Run settings changed. Choose a new output directory.")
        else:
            atomic_write(job, binding)
        factory = getattr(importlib.import_module(args.policy), "create_policy", None)
        require(callable(factory), "Policy module must export create_policy(config)")
        model = factory(config)
        try:
            require(all(callable(getattr(model, name, None)) for name in ("reset", "act")),
                    "Policy must implement reset(seed) and act(observation, instruction)")
            bridge = PolicyBridge(RobotStateReader)
            for episode in roster:
                dest = output / "attempts" / episode["id"]
                if dest.exists():
                    file = dest / "result.json"
                    if file.exists():
                        saved = verify_result(dest, binding["artifact_hash"], episode)
                        if saved["status"] == "semantic_result":
                            continue
                    archive = output / "interrupted"
                    archive.mkdir(exist_ok=True)
                    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                    dest.rename(archive / (dest.name + "_" + stamp))
                dest.mkdir(parents=True)
                seed = bridge.seed = episode["seed"]
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                bridge.state_readers = {}
                node = nodes[episode["node_id"]]
                scene = scenes[episode["scene_spec_id"]]
                result = execute_episode(
                    episode, node, scene, {"contexts": [], "raster_camera": True},
                    protocol, bridge, model, dest, observation_adapter=bridge.observe,
                    observer_factory=AnyArmPhysicalObserver if node["pool"] == "pick" else None,
                )
                if result["status"] == "semantic_result":
                    try:
                        result["grounding"] = score_trace(dest / "physics.jsonl.gz", node, result)
                    except (ValueError, OSError, EOFError, KeyError, StopIteration) as exc:
                        result.update(status="infrastructure_or_internal_error", error=f"Grounding evidence: {exc}")
                atomic_write(dest / "result.json", seal(
                    {**result, "episode": episode, "run_id": binding["artifact_hash"]}))
                print(f"{episode['id']} {result['status']}", flush=True)
                require(result["status"] == "semantic_result", result.get("error", "Evaluation failed"))
        finally:
            close = getattr(model, "close", None)
            if callable(close):
                close()


def report(args, panel):
    binding = read(args.output / "run.json")
    verify(binding)
    require(binding["bundle"] == panel["identity"] and binding["scope"] in ("smoke", "mixed50"),
            "Run was created against a different test protocol")
    roster = episodes(panel, binding["scope"] == "smoke")
    nodes = {n["node_id"]: n for n in panel["nodes"]}
    rows = defaultdict(list)
    identities = {}
    for episode in roster:
        node = nodes[episode["node_id"]]
        dest = args.output / "attempts" / episode["id"]
        record = None
        if (dest / "result.json").is_file():
            record = verify_result(dest, binding["artifact_hash"], episode)
            if record["status"] == "semantic_result":
                key = (episode["scene_spec_id"], episode["seed"])
                initial = (record["initial_state_hash"], record["initial_observation_hash"])
                require(identities.setdefault(key, initial) == initial, f"Unpaired initial evidence {key}")
        exposure = "trajectory_supervised" if node["pretraining_instruction_selected"] else "counterpart"
        for group in ("all", node["pool"], exposure, f"{node['pool']}_{exposure}"):
            rows[group].append(record)
    metrics = {}
    for group, records in sorted(rows.items()):
        done = [r for r in records if r and r["status"] == "semantic_result"]
        expected, completed = len(records), len(done)
        successes = sum(r["success"] for r in done)
        groundings = sum(r["grounding"]["passed"] for r in done)
        metrics[group] = {
            "expected": expected, "completed": completed, "missing": records.count(None),
            "errors": expected - completed - records.count(None),
            "successes": successes, "groundings": groundings,
            "success_rate": successes / expected if completed == expected else None,
            "grounding_rate": groundings / expected if completed == expected else None,
        }
    result = {"run_id": binding["artifact_hash"], "scope": binding["scope"],
              "complete": metrics["all"]["expected"] == metrics["all"]["completed"], "metrics": metrics}
    atomic_write(args.output / "report.json", result)
    print(json.dumps(result, indent=2))
    return 0 if result["complete"] else 1


def validate_scene_inputs(panel, bundle):
    from if_benchmark.evaluation.runner import runtime_scene_inputs
    for source in panel["scenes"]:
        scene = deepcopy(source)
        scene["source_paths"] = {
            key: [str(safe(bundle, ref)) for ref in value] if isinstance(value, list)
            else str(safe(bundle, value)) for key, value in scene["source_paths"].items()
        }
        joined = runtime_scene_inputs(scene)
        for key, expected in scene["source_hashes"].items():
            require(joined[key]["artifact_hash"] == expected, f"Unmatched frozen {key} artifact")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m robotwin_if")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "run", "report"):
        cmd = commands.add_parser(command)
        cmd.add_argument("--bundle", type=Path, default=ROOT)
        if command != "validate":
            cmd.add_argument("--output", type=Path, required=True)
        if command == "run":
            cmd.add_argument("--policy", required=True, help="Python module exposing create_policy(config)")
            cmd.add_argument("--config", type=Path, help="Policy JSON configuration")
            cmd.add_argument("--policy-revision", default="", help="Identifier for external policy dependencies/weights")
            cmd.add_argument("--smoke", action="store_true", help="One paired Pick and one paired Placement scene")
    args = parser.parse_args(argv)
    try:
        args.bundle = args.bundle.resolve()
        if hasattr(args, "output"):
            args.output = args.output.resolve()
        panel = load(args.bundle, verify_files=args.command != "report")
        if args.command == "validate":
            validate_scene_inputs(panel, args.bundle)
            print(json.dumps({"panel": panel["version"], "scenes": len(panel["scenes"]),
                              "instructions": len(panel["nodes"]), "episodes": len(episodes(panel)),
                              "verified_files": panel["file_count"], "bundle_id": panel["identity"]}))
        elif args.command == "run":
            run(args, panel)
        else:
            return report(args, panel)
        return 0
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        print(f"RoboTwin-IF: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
