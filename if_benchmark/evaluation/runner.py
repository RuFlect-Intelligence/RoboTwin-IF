"""Fixed episode execution; the two metrics share one immutable evidence trace."""
from dataclasses import fields
import fcntl
import gzip
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np

from if_benchmark.manifest.io import atomic_write, file_hash, read, seal, verify
from if_benchmark.schema.serialization import artifact_hash
from .contracts import EvaluationPolicy, load_plan
from .intent import PickIntent, ArmIntent, shared_done


class EpisodeDone(BaseException):
    """Exact-step stop; a policy's generic exception retry must not swallow it."""


class PolicyEnvironment:
    """Public RoboTwin policy surface; evaluator goals and traces stay private."""
    def __init__(self, env):
        self.__env = env

    def get_obs(self):
        return self.__env.get_obs()

    def get_instruction(self):
        return self.__env.get_instruction()

    def take_action(self, action, action_type="qpos"):
        if action_type not in {"qpos", "ee"}:
            raise ValueError("M17 supports qpos and ee actions")
        values = np.asarray(action)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError("policy action must be a finite vector")
        expected = (len(self.__env.robot.left_arm_joints)+len(self.__env.robot.right_arm_joints)+2
                    if action_type == "qpos" else 16)
        if len(values) != expected:
            raise ValueError("policy action dimension mismatch")
        return self.__env.take_action(values, action_type=action_type)

    def take_sampled_joint_action(self, action, *, period_s):
        from if_benchmark.native_wam.robotwin_sampled_joint_control import take_joint_action
        return take_joint_action(self.__env, action, period_s=period_s)

    @property
    def take_action_cnt(self):
        return self.__env.take_action_cnt

    @property
    def step_lim(self):
        return self.__env.step_lim

    @property
    def eval_success(self):
        # Early termination belongs to the shared scheduler, not the policy.
        return False


def install_observation_rendering(env):
    """Sync unchanged camera state when read, avoiding unused per-tick GPU work."""
    if env.crazy_random_light or env.render_freq or env.eval_video_path is not None:
        raise ValueError("observation-boundary rendering requires fixed lights and headless evaluation")
    update, get_obs = env._update_render, env.get_obs
    def observed():
        update()
        return get_obs()
    env._update_render = lambda: None
    env.get_obs = observed
    return update, get_obs


def observation_hash(value):
    def normalize(x):
        if isinstance(x, np.ndarray):
            return {"dtype": str(x.dtype), "shape": list(x.shape),
                    "sha256": hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()}
        if isinstance(x, dict):
            return {str(k):normalize(v) for k,v in x.items()}
        if isinstance(x, (tuple, list)):
            return [normalize(v) for v in x]
        if isinstance(x, np.generic):
            return x.item()
        return x
    return artifact_hash(normalize(value))


def next_policy_observation(module, model, previous, read_observation):
    requires = getattr(module, "requires_observation", None)
    if callable(requires) and not requires(model):
        return previous
    return read_observation()


def policy_identity(name, config):
    module_name = "if_benchmark.evaluation.policies" if name in {"noop", "reference"} else name
    module = importlib.import_module(module_name)
    for symbol in ("get_model", "eval", "reset_model"):
        if not callable(getattr(module, symbol, None)):
            raise ValueError(f"policy missing {symbol}")
    identity = seal({"artifact_kind": "m17_policy_binding", "name": name, "module": module_name,
                     "module_hash": file_hash(Path(module.__file__)), "config": config,
                     "debug_reference": name in {"noop", "reference"}})
    return module, identity


def check_record(record, episode, plan, binding, directory):
    verify(record)
    if (record["episode"] != episode or record["plan_hash"] != plan["artifact_hash"]
            or record["policy_hash"] != binding["artifact_hash"]):
        raise ValueError("attempt identity mismatch")
    for ref in record.get("evidence", []):
        p = directory / ref["path"]
        if not p.resolve().is_relative_to(directory.resolve()) or file_hash(p) != ref["sha256"]:
            raise ValueError("attempt evidence changed")


def runtime_environment():
    packages = {}
    for name in ("numpy", "scipy", "torch", "sapien", "mplib"):
        packages[name] = importlib.metadata.version(name)
    return {"python": sys.executable, "python_version": sys.version, "packages": packages}


def runtime_scene_inputs(scene):
    """Preserve all task-driven physics, independently of the selected prompt."""
    from if_benchmark.scene_validation.core import resolve_entity_dynamics
    joined = {k:read(Path(scene["source_paths"][k])) for k in ("scene", "realized", "unit")}
    for value in joined.values():
        verify(value)
    tasks = [read(Path(p)) for p in scene["source_paths"].get("tasks", [])]
    embedded = scene.get("runtime_tasks", [])
    if not isinstance(embedded, list) or any(not isinstance(t, dict) for t in embedded):
        raise ValueError("Invalid embedded scene task context")
    tasks.extend(embedded)
    if not tasks:
        raise ValueError("Scene task context is empty; movable actors would become static. Rebuild the evaluation plan.")
    joined.update(tasks=tasks, routes={})
    dynamics = resolve_entity_dynamics(joined)
    if scene.get("runtime_dynamics") is not None and scene["runtime_dynamics"] != dynamics:
        raise ValueError("Scene dynamics differ from the frozen task context")
    return joined


def run(plan_path, output, *, policy_name="noop", policy_config=None, worker_index=0,
        worker_count=1, limit=None):
    plan = load_plan(plan_path)
    if worker_count < 1 or not 0 <= worker_index < worker_count:
        raise ValueError("invalid worker partition")
    config = dict(policy_config or {})
    if policy_name in {"noop", "reference"}:
        config["kind"] = policy_name
        config["observation_mode"] = "initial_only"
    module, binding = policy_identity(policy_name, config)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    # A short initialization lock also makes concurrent worker identity checks atomic.
    with (output / ".run.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        run_doc = seal({"artifact_kind": "m17_run", "plan_hash": plan["artifact_hash"],
                        "policy": binding, "environment": runtime_environment()})
        if (output / "run.json").exists():
            if read(output / "run.json") != run_doc:
                raise ValueError("run configuration/environment changed; use a new output directory")
        else:
            atomic_write(output / "run.json", run_doc)
    for path,digest in plan["source_files"].items():
        if file_hash(Path(path)) != digest:
            raise ValueError(f"frozen source changed: {path}")
    nodes = {n["node_id"]:n for n in plan["nodes"]}
    scenes = {s["scene_spec_id"]:s for s in plan["scenes"]}
    import torch
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    model = module.get_model(config)
    policy = EvaluationPolicy(**{f.name:plan["policy"][f.name] for f in fields(EvaluationPolicy)})
    policy.validate()
    completed = 0
    runtime_cache = {}
    for i,episode in enumerate(plan["episodes"]):
        if i % worker_count != worker_index:
            continue
        if limit is not None and completed >= limit:
            break
        directory = output / "attempts" / episode["id"]
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / ".lock").open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            record_path = directory / "result.json"
            if record_path.exists():
                previous = read(record_path)
                check_record(previous, episode, plan, binding, directory)
                if previous["status"] != "semantic_result":
                    raise RuntimeError(f"blocked attempt requires diagnosis/new run: {episode['id']}")
                completed += 1
                continue
            partial = directory / "physics.jsonl.gz"
            if partial.exists():
                # Preserve an interrupted trace before repeating its fixed episode.
                digest = file_hash(partial)
                partial.rename(directory / f"interrupted_{digest}.jsonl.gz")
            node,scene = nodes[episode["node_id"]],scenes[episode["scene_spec_id"]]
            result = execute_episode(episode, node, scene, plan, policy, module, model, directory,
                                     runtime_cache=runtime_cache)
            record = seal({"artifact_kind": "m17_shared_rollout_attempt", "episode": episode,
                           "plan_hash": plan["artifact_hash"], "policy_hash": binding["artifact_hash"], **result})
            atomic_write(record_path, record)
            print(json.dumps({"episode": episode["id"], "status": record["status"],
                              "success": record.get("success", False), "worker": worker_index}, ensure_ascii=False), flush=True)
            if record["status"] != "semantic_result":
                raise RuntimeError(f"evaluation blocked: {record['error']}")
            completed += 1
    return {"worker": worker_index, "completed_or_resumed": completed, "plan_episode_count": len(plan["episodes"])}


def execute_episode(episode, node, scene, plan, policy, module, model, directory, *, runtime_cache=None,
                    observation_adapter=None, observer_factory=None):
    # Simulation dependencies are confined to the real execution branch.
    from envs.if_scene_validation import IFSceneValidationRuntime
    from if_benchmark.scene_validation.core import ValidationConfig
    from if_benchmark.scene_validation.expert_capture import RobotStateReader
    from .observation import PhysicalObserver

    runtime, old_step, env = None, None, None
    trace = None
    render_methods = None
    recorder = None
    collection_physics = None
    evidence = []
    started = time.monotonic()
    try:
        random.seed(episode["seed"])
        joined = runtime_scene_inputs(scene)
        from if_benchmark.scene_validation.core import resolve_entity_dynamics
        dynamics = resolve_entity_dynamics(joined)
        if node.get("verb", node["task"]["goal"].get("verb")) == "pick" and dynamics.get(node["task"]["goal"]["subject"]) != "dynamic":
            raise ValueError("Pick subject is not a dynamic rigid body; refusing an invalid model evaluation")
        config = ValidationConfig.load(Path("if_benchmark/data/scene_validation/scene_validation_config.yml"))
        runtime = IFSceneValidationRuntime(joined, evidence_dir=directory, config=config.data,
                                          reusable_env_cache=runtime_cache)
        runtime.reset(episode["seed"], diagnostic_video_render=plan.get("raster_camera", False))
        env = runtime.env
        env.eval_mode = True
        env.step_lim = policy.max_actions
        env.set_instruction(node["task"]["instruction"]["selected"]["text"])
        env.check_success = lambda: False  # Success is latched by the passive observer.
        from .collection_physics import install as install_physics, physics_contract
        expected_physics = physics_contract()
        if plan.get('physics_contract', expected_physics) != expected_physics:
            raise ValueError('Collection physics contract changed; rebuild the evaluation plan')
        reference_policy = module.__name__ == "if_benchmark.evaluation.policies" and model.get('kind') == 'reference'
        if not reference_policy:
            collection_physics = install_physics(env)
        observer = (observer_factory or PhysicalObserver)(env, node["task"], scene["source_hashes"])
        reader = RobotStateReader(env.robot)
        required = {c["event_kind"] for c in plan["contexts"] if any(m["node_id"] == node["node_id"] for m in c["members"])}
        initial_state_hash = runtime._physical_state_hash()
        initial_obs = env.get_obs()
        if observation_adapter is not None:
            initial_obs = observation_adapter(env, initial_obs)
        initial_obs_hash = observation_hash(initial_obs)
        render_methods = install_observation_rendering(env)
        module.reset_model(model)
        if module.__name__ == "if_benchmark.evaluation.policies":
            module.bind_reference(model, node, scene, episode["seed"], policy.waypoint_hz)
        baseline = reader.read()["action"].astype(float).tolist()
        sizes = [len(row["joints"])+1 for row in reader.arms]
        cut = sizes[0]
        split_command = lambda values: {"left": list(map(float,values[:cut])), "right": list(map(float,values[cut:]))}
        scales = {}
        for row in reader.arms:
            scales[row["arm"]] = [float(j.get_limits()[0][1]-j.get_limits()[0][0]) for j in row["joints"]] + [1.]
        pick = PickIntent(scene["candidate_entities"], observer.links,
                          impulse=policy.contact_impulse, persistence=policy.contact_steps)
        arm = ArmIntent(split_command(baseline), scales, window=policy.arm_seconds,
                        activity=policy.arm_activity, dominance=policy.arm_dominance)
        dt = float(env.scene.get_timestep())
        trace_path = directory / "physics.jsonl.gz"
        trace = gzip.open(trace_path, "wt", encoding="utf-8", compresslevel=1)
        recording = plan.get('recording')
        if recording is not None:
            from .recording import EpisodeRecorder
            recorder = EpisodeRecorder(directory, env, model,
                {'episode': episode, 'instruction': node['task']['instruction']['selected']['text'],
                 'policy': plan['policy'], 'dt': dt,
                 'action_representation': getattr(getattr(model, 'server', model), '_if_action_representation', 'abs_ee')},
                lambda: observer.step, render_methods[0], initial_obs, recording)
        def emit(row):
            trace.write(json.dumps(row, separators=(",", ":"), allow_nan=False)+"\n")
        emit({"kind": "initial", "physical_state_hash": initial_state_hash,
              "observation_hash": initial_obs_hash, "dt": dt, "action_baseline": split_command(baseline),
              "action_scales": scales, "success_initial": observer.initial,
              "entity_dynamics": dynamics, 'physics_contract':expected_physics})
        old_step = env.scene.step
        def step(*args, **kwargs):
            command = split_command(reader.read()["action"])
            value = old_step(*args, **kwargs)
            contacts = observer.observe()
            t = observer.step*dt
            if "pick" in required and t <= policy.pick_seconds + 1e-8:
                pick.observe(contacts, observer.step, t)
            if "arm" in required:
                arm.observe(command, dt)
            emit({"kind": "physics", "step": observer.step, "command": command,
                  "contacts": contacts, "actual_grippers": observer.grippers,
                  "tracked_poses": observer.poses[-1]})
            if recorder is not None:
                recorder.video.capture(observer.step)
            if observer.step % policy.success_check_steps == 0:
                snapshot,result = observer.evaluate()
                emit({"kind": "success_check", "step": observer.step, "snapshot": snapshot, "result": result})
            if shared_done(success=observer.success_result is not None, time_seconds=t,
                pick_required="pick" in required, pick_result=pick.result, arm_required="arm" in required,
                policy=policy, action_count=0):
                raise EpisodeDone()
            return value
        env.scene.step = step
        policy_env = PolicyEnvironment(env)
        initial_only = (module.__name__ == "if_benchmark.evaluation.policies"
                        and model.get("observation_mode") == "initial_only")
        obs = initial_obs
        observation_count = 1
        def read_observation():
            nonlocal observation_count
            value = env.get_obs()
            observation_count += 1
            return observation_adapter(env, value) if observation_adapter is not None else value
        try:
            while env.take_action_cnt < policy.max_actions:
                before = observer.step
                # Buffered synchronous policies consume sensors only at replan;
                # other policies keep the existing fresh-per-action contract.
                if observer.step != 0 and not initial_only:
                    obs = next_policy_observation(module, model, obs, read_observation)
                module.eval(policy_env, model, obs)
                if observer.step == before:
                    raise RuntimeError("policy did not advance the simulation through take_action")
        except EpisodeDone:
            pass
        if observer.last_result is None or observer.step % policy.success_check_steps:
            snapshot,result = observer.evaluate()
            emit({"kind": "success_check", "step": observer.step, "snapshot": snapshot, "result": result})
        success = observer.success_result is not None
        result = {"status": "semantic_result", "success": success,
                  "success_result": observer.success_result or observer.last_result,
                  "success_first_step": observer.success_step,
                  "success_first_seconds": observer.success_step*dt if success else None,
                  "intent": {k:v for k,v in {"pick":pick.finish(), "arm":arm.finish() if "arm" in required else None}.items() if k in required},
                  "initial_state_hash": initial_state_hash, "initial_observation_hash": initial_obs_hash,
                  "physics_steps": observer.step, "simulation_seconds": observer.step*dt,
                  "action_count": env.take_action_cnt,
                  "policy_observation_count": observation_count,
                  "entity_dynamics": dynamics,
                  "physics_contract": expected_physics,
                  "termination_reason": ("success_horizon" if observer.step*dt+1e-8 >= policy.success_seconds
                      else "action_budget" if env.take_action_cnt >= policy.max_actions else "success_and_intent_complete")}
        if hasattr(observer, "last_per_arm"):
            result.update(per_arm_success=observer.last_per_arm, success_arm=observer.success_arm)
        execution = getattr(env, '_if_sampled_execution', None)
        if execution is not None:
            path = directory / 'sampled_execution.json'
            atomic_write(path, dict(commands=execution, object_dependent_control=False))
            evidence.append(dict(path=path.name, sha256=file_hash(path)))
        if module.__name__ == "if_benchmark.evaluation.policies" and model["kind"] == "reference":
            result["reference_identity"] = model["reference_identity"]
    except Exception as exc:
        result = {"status": "infrastructure_or_internal_error", "error": str(exc), "traceback": traceback.format_exc()}
    finally:
        if recorder is not None:
            try:
                result['recording'] = recorder.close()
            except Exception as exc:
                result = {"status": "infrastructure_or_internal_error", "error": str(exc),
                          "traceback": traceback.format_exc()}
        if plan.get('recording') is not None:
            from .recording import recording_evidence
            evidence.extend(recording_evidence(directory))
        if old_step is not None and env is not None:
            env.scene.step = old_step
        if render_methods is not None:
            env._update_render,env.get_obs = render_methods
        if trace is not None:
            trace.close()
            evidence.append({"path": "physics.jsonl.gz", "sha256": file_hash(directory / "physics.jsonl.gz")})
        if collection_physics is not None:
            physical_record = collection_physics.report()
            try:
                collection_physics.close()
                physical_record.update(cleanup_complete=True, events=collection_physics.events)
            except Exception as exc:
                physical_record.update(cleanup_complete=False, cleanup_error=str(exc))
                result = {"status":"infrastructure_or_internal_error", "error":f"collection physics cleanup: {exc}"}
            path = directory / 'collection_physics.json'
            atomic_write(path, physical_record)
            evidence.append(dict(path=path.name,sha256=file_hash(path)))
        if runtime is not None:
            try:
                runtime.close()
            except Exception as exc:
                result = {"status": "infrastructure_or_internal_error", "error": f"runtime cleanup: {exc}"}
    return {**result, "evidence": evidence, "wall_seconds": time.monotonic()-started}
