"""Robot-aware, support-only Initial Pose physics probe.

The actor is loaded once at the exact approved state.  No task action, pose
candidate, or runtime fallback exists in this environment.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
import sapien

from if_benchmark.collision_geometry import collision_aabb_tight
from if_benchmark.initial_pose.runtime import (
    apply_dynamic_mass_properties,
    bind_loaded_actor_scale,
    validate_loaded_initial_pose,
)
from ._base_task import Base_Task
from .robot import Robot
from .utils import create_actor, create_sapien_urdf_obj


SETTLE_STEPS = 1600
TAIL_STEPS = 200
CONTACT_EPS = 1e-6
ROBOT_HOME_SETTLE_STEPS = 240


class if_initial_pose_support(Base_Task):
    def setup_demo(self, **kwargs):
        self.support_job = kwargs.get("support_job") or {}
        self.support_metrics: dict[str, Any] = {}
        self.support_trajectory: list[dict[str, Any]] = []
        super()._init_task_env_(**kwargs)

    def load_camera(self, **kwargs):
        # Support evidence is numeric; avoiding camera creation keeps this
        # materially lighter than M4/M5 while retaining the real SAPIEN scene.
        self.cameras = None

    def load_robot(self, **kwargs):
        # Load the real benchmark embodiment at home without constructing a
        # motion planner.  It is present solely for initial-contact evidence.
        self.robot = Robot(self.scene, need_topp=False, **kwargs)
        self.robot.init_joints()

    def together_open_gripper(self, save_freq=None, left_pos=1, right_pos=1):
        """Place the support-only robot at its open-gripper home state.

        ``Base_Task`` normally obtains a gripper trajectory from the motion
        planner here.  The support probe deliberately does not construct a
        planner, so drive the real articulation joints directly and let them
        settle before the subject actor is spawned.  This keeps the robot
        collision geometry authoritative without introducing a planner-only
        dependency into a physics-only gate.
        """

        del save_freq
        for _ in range(ROBOT_HOME_SETTLE_STEPS):
            self.robot.set_gripper(float(left_pos), "left", gripper_eps=0.0)
            self.robot.set_gripper(float(right_pos), "right", gripper_eps=0.0)
            self.scene.step()

    def load_actors(self):
        job = self.support_job
        q = _yaw_adjusted_q(job["q0_wxyz"], float(job["yaw_deg"]))
        origin_z = float(job["support"]["origin_z"]) + float(self.table_z_bias)
        pose = sapien.Pose([0.0, 0.0, origin_z], q)
        joint_state = job.get("articulated_joint_state") or {}
        if joint_state.get("mode") != "not_articulated":
            model_id = _model_id(job.get("model_id"))
            self.subject = create_sapien_urdf_obj(
                scene=self,
                pose=pose,
                modelname=str(job["asset"]),
                modelid=model_id,
                fix_root_link=True,
                mass=None,
            )
            qpos = np.asarray(joint_state.get("qpos") or [], dtype=float)
            if qpos.shape != np.asarray(self.subject.get_qpos()).shape:
                raise RuntimeError("source_initial_pose_joint_count_mismatch")
            self.subject.set_qpos(qpos)
            self.subject.set_qvel(np.zeros_like(self.subject.get_qvel()))
        else:
            self.subject = create_actor(
                scene=self,
                pose=pose,
                modelname=str(job["asset"]),
                scale=job["scale_xyz"],
                model_id=_model_id(job.get("model_id")),
                convex=True,
                mass=None,
            )
        if self.subject is None:
            raise RuntimeError("actor_load_failed")
        dynamic_mass_properties = apply_dynamic_mass_properties(
            self.subject, job
        )
        if joint_state.get("mode") == "not_articulated":
            bind_loaded_actor_scale(self.subject, job["scale_xyz"])
        # create_sapien_urdf_obj applies model transform metadata and both
        # helpers preprocess task-relative Z.  Support evidence must start at
        # the exact approved world root pose, with no helper-owned fallback.
        self.subject.actor.set_pose(pose)
        self.subject.set_name(f"initial_pose_support:{job['initial_state_id']}")
        scale = _scale3((self.subject.config or {}).get("scale"))
        runtime_state = {
            **job,
            "support": {
                **job["support"],
                "origin_z": origin_z,
            },
        }
        actual_joint = (
            np.asarray(self.subject.get_qpos(), dtype=float).tolist()
            if joint_state.get("mode") != "not_articulated"
            else []
        )
        self.preflight = validate_loaded_initial_pose(
            canonical_state=runtime_state,
            actual_q_wxyz=np.asarray(self.subject.get_pose().q, dtype=float).tolist(),
            actual_joint_qpos=actual_joint,
            actual_scale_xyz=scale,
            actual_origin_z=float(self.subject.get_pose().p[2]),
            expected_world_yaw_deg=float(job["yaw_deg"]),
        )
        self.preflight["dynamic_mass_properties"] = dict(
            dynamic_mass_properties
        )
        self.preflight["actual_mass_properties"] = _mass_properties_rows(
            self.subject
        )

    def check_stable(self):
        spawn = _pose_row(self.subject)
        states = [spawn]
        subject_entities = _actor_entities(self.subject)
        robot_entities = _robot_entities(self.robot)
        support_impulse = 0.0
        initial_robot_contact = False
        support_contact_steps = 0
        for step in range(SETTLE_STEPS):
            self.scene.step()
            contact = _contact_observation(
                self.scene,
                subject_entities=subject_entities,
                table_entity=self.table,
                robot_entities=robot_entities,
            )
            support_impulse += contact["support_impulse_ns"]
            support_contact_steps += int(contact["support_impulse_ns"] > CONTACT_EPS)
            if step < 5 and contact["robot_impulse_ns"] > CONTACT_EPS:
                initial_robot_contact = True
            row = _pose_row(self.subject)
            states.append(row)
            if step % 20 == 0 or step >= SETTLE_STEPS - TAIL_STEPS:
                self.support_trajectory.append(
                    {"step": step + 1, **row, **contact}
                )
        settled = states[-1]
        tail = states[-(TAIL_STEPS + 1) :]
        # Measure total path length over the settle tail.  Both a single-step
        # maximum and a start/end delta can hide a body that oscillates and
        # returns near its starting pose.
        tail_translation = sum(
            _translation_distance(tail[index], tail[index + 1])
            for index in range(len(tail) - 1)
        )
        tail_rotation = sum(
            _rotation_distance(tail[index], tail[index + 1])
            for index in range(len(tail) - 1)
        )
        aabb = collision_aabb_tight(self.subject)
        bottom = float(aabb["min"][2])
        support_z = float(self.support_job["support"]["plane_z"]) + float(
            self.table_z_bias
        )
        bottom_delta = bottom - support_z
        edge_margin = min(
            0.6 - abs(float(aabb["min"][0])),
            0.6 - abs(float(aabb["max"][0])),
            0.35 - abs(float(aabb["min"][1])),
            0.35 - abs(float(aabb["max"][1])),
        )
        self.support_metrics = {
            "spawn_to_settled_translation_m": _translation_distance(spawn, settled),
            "spawn_to_settled_rotation_deg": _rotation_distance(spawn, settled),
            "tail_translation_drift_m": float(tail_translation),
            "tail_rotation_drift_deg": float(tail_rotation),
            "bottom_gap_m": max(0.0, float(bottom_delta)),
            "bottom_penetration_m": max(0.0, -float(bottom_delta)),
            "support_impulse_ns": float(support_impulse),
            "table_edge_margin_m": float(edge_margin),
            "initial_robot_contact": bool(initial_robot_contact),
            "settled_collision_aabb": aabb,
            "collision_aabb_source": "sapien_collision_compute_global_aabb_tight",
            "support_mode": str(self.support_job["support_mode"]),
            "support_contact_steps": int(support_contact_steps),
            "settle_steps": SETTLE_STEPS,
            "tail_steps": TAIL_STEPS,
        }
        checks = {
            "spawn_translation_within_limit": self.support_metrics[
                "spawn_to_settled_translation_m"
            ]
            <= 0.02,
            "spawn_rotation_within_limit": self.support_metrics[
                "spawn_to_settled_rotation_deg"
            ]
            <= 10.0,
            "tail_translation_within_limit": tail_translation <= 0.003,
            "tail_rotation_within_limit": tail_rotation <= 2.0,
            "bottom_gap_within_limit": self.support_metrics["bottom_gap_m"] <= 0.005,
            "bottom_penetration_within_limit": self.support_metrics[
                "bottom_penetration_m"
            ]
            <= 0.003,
            "support_contact_observed": (
                support_contact_steps > 0
                if self.support_job["support_mode"] == "dynamic_table_support"
                else True
            ),
            "table_edge_margin_positive": edge_margin >= 0.02,
            "no_initial_robot_contact": not initial_robot_contact,
        }
        self.support_metrics["checks"] = checks
        self.support_metrics["failed_checks"] = sorted(
            name for name, passed in checks.items() if not passed
        )
        return all(checks.values()), self.support_metrics["failed_checks"]

def _model_id(value: Any) -> int | str:
    text = str(value)
    return int(text) if text.isdigit() else text


def _scale3(value: Any) -> list[float]:
    if isinstance(value, (int, float)):
        return [float(value)] * 3
    result = [float(item) for item in value or [1.0, 1.0, 1.0]]
    return result * 3 if len(result) == 1 else result


def _yaw_adjusted_q(q_wxyz: Iterable[float], yaw_deg: float) -> list[float]:
    theta = math.radians(yaw_deg) / 2.0
    left = [math.cos(theta), 0.0, 0.0, math.sin(theta)]
    right = [float(value) for value in q_wxyz]
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    result = np.asarray(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=float,
    )
    return (result / np.linalg.norm(result)).tolist()


def _pose_row(actor: Any) -> dict[str, Any]:
    pose = actor.get_pose()
    return {
        "p": np.asarray(pose.p, dtype=float).tolist(),
        "q_wxyz": np.asarray(pose.q, dtype=float).tolist(),
    }


def _translation_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    return float(
        np.linalg.norm(np.asarray(left["p"], dtype=float) - np.asarray(right["p"], dtype=float))
    )


def _rotation_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    a = np.asarray(left["q_wxyz"], dtype=float)
    b = np.asarray(right["q_wxyz"], dtype=float)
    a /= max(float(np.linalg.norm(a)), 1e-12)
    b /= max(float(np.linalg.norm(b)), 1e-12)
    dot = float(np.clip(abs(float(np.dot(a, b))), 0.0, 1.0))
    return float(math.degrees(2.0 * math.acos(dot)))


def _mass_properties_rows(actor: Any) -> list[dict[str, Any]]:
    root = getattr(actor, "actor", actor)
    get_links = getattr(root, "get_links", None)
    if callable(get_links):
        bodies = list(get_links())
    else:
        bodies = [
            component
            for component in root.get_components()
            if isinstance(
                component, sapien.physx.PhysxRigidDynamicComponent
            )
        ]
    rows = []
    for index, body in enumerate(bodies):
        get_name = getattr(body, "get_name", None)
        rows.append(
            {
                "body_index": index,
                "body_name": str(get_name()) if callable(get_name) else None,
                "mass_kg": float(body.mass),
                "principal_inertia_kg_m2": np.asarray(
                    body.inertia, dtype=float
                ).tolist(),
            }
        )
    return rows


def _actor_entities(actor: Any) -> set[Any]:
    root = getattr(actor, "actor", actor)
    entities = _component_entities(root)
    get_links = getattr(root, "get_links", None)
    if callable(get_links):
        for link in get_links():
            entities.update(_component_entities(link))
    return entities


def _robot_entities(robot: Any) -> set[Any]:
    entities: set[Any] = set()
    for root in (robot.left_entity, robot.right_entity):
        entities.update(_component_entities(root))
        for link in root.get_links():
            entities.update(_component_entities(link))
    return entities


def _component_entities(value: Any) -> set[Any]:
    entities = {value}
    entity = getattr(value, "entity", None)
    if entity is not None:
        entities.add(entity)
    return entities


def _contact_observation(
    scene: Any,
    *,
    subject_entities: set[Any],
    table_entity: Any,
    robot_entities: set[Any],
) -> dict[str, float]:
    support = 0.0
    robot = 0.0
    for contact in scene.get_contacts():
        first = contact.bodies[0].entity
        second = contact.bodies[1].entity
        if first in subject_entities:
            other = second
        elif second in subject_entities:
            other = first
        else:
            continue
        impulse = sum(
            float(np.linalg.norm(np.asarray(point.impulse, dtype=float)))
            for point in contact.points
        )
        if other == table_entity:
            support += impulse
        if other in robot_entities:
            robot += impulse
    return {"support_impulse_ns": support, "robot_impulse_ns": robot}
