"""Measured physical observations, including collection-compatible constraints."""
from collections import defaultdict, deque
from copy import deepcopy

import numpy as np

from if_benchmark.collision_geometry import collision_aabb_tight
from if_benchmark.execution_policy import HOLD_EE_FRAME_ARM
from if_benchmark.pair_terminal_settle_worker import stability_from_snapshots
from if_benchmark.scene_validation.core import evaluate_task_checker
from if_benchmark.success_contract import SuccessContract
from if_benchmark.success_contract.snapshot import build_success_snapshot
from if_benchmark.success_contract.observation import (
    DEFAULT_ALOHA_GRIPPER_PROFILE, load_embodiment_profile,
    observe_actual_grippers, observe_sapien_active_contacts,
    hold_consistency_from_relative_history,
)
from if_benchmark.schema.serialization import artifact_hash
from .success_duration import SuccessDuration


def observed_articulations(env, initial):
    """Preserve frozen profiles, but measure current joints for scene integrity."""
    rows = deepcopy(initial)
    for key, row in rows.items():
        actor = env.entities[key]
        index = env.articulation_joint_indices[key]
        limits = np.asarray(actor.get_qlimits(), dtype=float)[index]
        row.update(qpos=float(np.asarray(actor.get_qpos())[index]),
                   lower=float(limits[0]), upper=float(limits[1]))
    return rows


class PhysicalObserver:
    def __init__(self, env, task, source_hashes):
        self.env, self.task, self.source_hashes = env, task, source_hashes
        self.contract = SuccessContract.load()
        self.profile = load_embodiment_profile(DEFAULT_ALOHA_GRIPPER_PROFILE)
        self.policy = self.contract.section("stability")
        self.window = self.policy["observation_window_steps"]
        self.tracked = env._stability_components()
        self.poses = deque(maxlen=self.window)
        self.holds = {}
        self.contact_counts = {}
        self.step = 0
        self.success_duration = SuccessDuration(task, self.policy)
        self.initial = env.snapshot(settle=False)
        self.endpoints = {str(w.get_name()): key for key,w in env.entities.items()}
        for key,w in env.entities.items():
            if env.dynamics[key] == "articulated_fixed":
                self.endpoints.update({str(link.get_name()):key for link in w.actor.get_links()})
        table = getattr(env.table, "actor", env.table)
        self.endpoints[str(table.get_name())] = "support:table"
        self.links = []
        for arm,p in self.profile["arms"].items():
            links = [f["link"] for f in p["fingers"]] + p["palm_wrist_links"]
            self.links.extend(links)
            self.endpoints.update({name:f"gripper:{arm}" for name in links})
        self.grippers, self.contacts = {}, []
        self.last_snapshot = self.last_result = None
        self.success_snapshot = self.success_result = None
        self.success_step = None

    def observe(self):
        env = self.env
        self.step += 1
        self.grippers = observe_actual_grippers(env.robot, embodiment_profile=self.profile)
        self.contacts = observe_sapien_active_contacts(env.scene,
            endpoint_by_body_name=self.endpoints, embodiment_profile=self.profile,
            grippers=self.grippers, contract=self.contract)
        controller = getattr(env, '_if_collection_physics', None)
        if controller is not None:
            controller.after_step(self.grippers, self.contacts, self.step)
        self.poses.append([{"p": actor.get_pose().p.tolist(), "q": actor.get_pose().q.tolist()}
                           for _,_,actor in self.tracked])
        fingers = defaultdict(set)
        for c in self.contacts:
            if c.get("arm") and c.get("region") == "inner_pad":
                entities = {c["entity_a"], c["entity_b"]} & set(env.entities)
                for subject in entities:
                    fingers[(c["arm"],subject)].add(c.get("finger"))
        bilateral = {k for k,v in fingers.items() if len(v-{None}) >= 2}
        self.contact_counts = {k:self.contact_counts.get(k,0)+1 for k in bilateral}
        for arm in ("left", "right"):
            touched = sorted(subject for a,subject in fingers if a == arm)
            held = self.holds.get(arm)
            attached = controller.observation(arm) if controller is not None else None
            if attached is not None:
                if held is None or held['subject'] != attached['subject'] or held['step'] != attached['reference_step']:
                    held = {"subject": attached['subject'], "ee_frame": HOLD_EE_FRAME_ARM,
                        "ee_to_actor_transform": attached['reference']['ee_to_actor_transform'],
                        "step": attached['reference_step'], "relative": deque(maxlen=self.window)}
                    self.holds[arm] = held
            elif held and (touched != [held["subject"]]):
                self.holds.pop(arm)
                held = None
            qualified = [subject for a,subject in bilateral if a == arm and self.contact_counts[(a,subject)] >= 2]
            if held is None and len(qualified) == 1 and touched == qualified:
                subject = qualified[0]
                relative = np.linalg.inv(env._hold_ee_matrix(env._arm(arm))) @ env._pose_matrix(subject)
                held = {"subject": subject, "ee_frame": HOLD_EE_FRAME_ARM,
                        "ee_to_actor_transform": relative.tolist(), "step": self.step,
                        "relative": deque(maxlen=self.window)}
                self.holds[arm] = held
            state = self.grippers[arm]
            state["closure_evidence"] = {"subject": held["subject"] if held else None,
                "valid": bool(held), "path": "observed_bilateral_inner_pad",
                "reference_step": held["step"] if held else None,
                "actual_width_m": state["actual_width_m"], "continuous_steps":
                    max((n for (a,_),n in self.contact_counts.items() if a == arm), default=0)}
            if attached is not None:
                state['closure_evidence'] = {**deepcopy(attached['closure']),
                    'path':'post_close_fixed_attachment', 'actual_width_m':state['actual_width_m']}
                state['post_close_attachment'] = deepcopy(attached['attachment'])
            if held:
                relative = np.linalg.inv(env._hold_ee_matrix(env._arm(arm))) @ env._pose_matrix(held["subject"])
                held["relative"].append(relative.tolist())
            state["hold_consistency"] = {"subject": None, "translation_drift_m": 0.,
                                         "rotation_drift_deg": 0., "ee_frame": HOLD_EE_FRAME_ARM}
        self.success_duration.observe(self.grippers, self.step)
        return self.contacts

    def evaluate(self):
        env = self.env
        history = list(self.poses)
        raw = stability_from_snapshots(history) if len(history) >= 2 else {"stable": False, "per_entity": []}
        if len(history) < self.window:
            raw = {"stable": False, "per_entity": [{"stable": False} for _ in self.tracked]}
        else:
            raw = env._apply_axisymmetric_stability(raw, history, self.tracked, contract=self.contract)
        stability = env._snapshot_stability_contract(raw,
            tracked_components=[(key,component) for key,component,_ in self.tracked])
        entities = {}
        for key,actor in env.entities.items():
            pose = actor.get_pose()
            entities[key] = {"pose": {"position": pose.p.tolist(), "quat_wxyz": pose.q.tolist()},
                             "aabb": collision_aabb_tight(actor),
                             "support_plane_z": self.initial["entities"][key]["support_plane_z"]}
        grippers = deepcopy(self.grippers)
        for arm,hold in self.holds.items():
            grippers[arm]["hold_consistency"] = hold_consistency_from_relative_history(
                list(hold["relative"]), subject=hold["subject"], ee_frame=HOLD_EE_FRAME_ARM,
                reference=hold, symmetry_axis=env._hold_symmetry_axis(hold["subject"]))
        observed_arms = sorted(a for a,h in self.holds.items() if h["subject"] == self.task["goal"]["subject"])
        provenance = {"primary_arm": observed_arms[0] if len(observed_arms) == 1 else None,
                      "primary_op": "observed_pick", "participating_arms": sorted(self.holds),
                      "source": "physical_bilateral_contact_and_hold_history"}
        provenance["primary_event_hash"] = artifact_hash(provenance)
        terminal = {"entities": entities, "grippers": grippers, "active_contacts": self.contacts,
                    "held_by": {a:h["subject"] for a,h in self.holds.items()},
                    "action_provenance": provenance,
                    "articulations": observed_articulations(env, self.initial["articulations"]),
                    "poses": {k:v["pose"] for k,v in entities.items()}, "stable": stability["stable"]}
        snapshot = build_success_snapshot(initial=self.initial, terminal=terminal, stability=stability,
            execution_events={}, source_hashes=self.source_hashes, contract=self.contract,
            profiles={"embodiment": self.profile})
        result = evaluate_task_checker(self.task, snapshot)
        if result["status"] == "blocked":
            raise RuntimeError(f"physical_success_observation_incomplete:{result['failed_predicates']}")
        # The M15 settle contract requires 600 steps before a 200-step window.
        # Here the same duration is observed passively while the policy runs.
        result = self.success_duration.evaluate(result, self.step)
        self.last_snapshot, self.last_result = snapshot, result
        if result["passed"] and self.success_result is None:
            self.success_snapshot, self.success_result = snapshot, result
            self.success_step = self.step
        return snapshot, result
