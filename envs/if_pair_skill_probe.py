"""Canonical clean-table skill probe for Milestone 5 pair validation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import sapien

from ._base_task import Base_Task
from ._GLOBAL_CONFIGS import GRASP_DIRECTION_DIC
from .utils import (
    Action,
    ArmTag,
    create_actor,
    create_box,
    create_sapien_urdf_obj,
    create_sphere,
    create_table,
)
from if_benchmark.articulated_terminal_contract import (
    articulation_state_metrics,
    articulated_place_in_success_metrics,
    cavity_witness_world_pose,
)
from if_benchmark.pair_skill_contract import (
    HANG_ENTRY_POLICY_ID,
    aabb_centered_root_position_xy,
    articulated_entry_waypoints,
    canonical_scene_fixture,
    hang_skill_phase_contract,
    setup_stability_decision,
)
from if_benchmark.pair_skill_runtime import hold_grippers_and_step
from if_benchmark.pair_terminal_settle_worker import (
    aligned_functional_root_pose,
    actor_aabb,
    contact_metrics,
    dump_into_success_metrics,
    hang_on_success_metrics,
    has_contact,
    place_in_success_metrics,
    place_on_success_metrics,
    stack_success_metrics,
    stability_from_snapshots,
    stamp_on_success_metrics,
)
from if_benchmark.initial_pose.runtime import (
    apply_dynamic_mass_properties,
    apply_world_yaw,
    approved_world_yaw_deg,
    bind_loaded_actor_scale,
    validate_loaded_initial_pose,
)
from if_benchmark.schema import artifact_hash
from if_benchmark.success_contract import (
    DEFAULT_ALOHA_GRIPPER_PROFILE,
    SuccessContract,
    build_success_snapshot,
    evaluate_m5_skill_success,
    load_embodiment_profile,
    observe_actual_grippers,
    observe_sapien_active_contacts,
)


TABLE_TOP_Z = 0.741
PAIR_QPOS_HINTS = {
    ("hang_on", "subject"): [0.707, 0.707, 0.0, 0.0],
    ("hang_on", "receiver"): [-0.22, -0.22, 0.67, 0.67],
    ("dump_into", "subject"): [0.651892, 0.651428, 0.274378, 0.274584],
    ("dump_into", "receiver"): [0.5, 0.5, 0.5, 0.5],
    ("stamp_on", "subject"): [0.5, 0.5, 0.5, 0.5],
    ("stamp_on", "receiver"): [1.0, 0.0, 0.0, 0.0],
}


class if_pair_skill_probe(Base_Task):
    """One robot rollout; no world-yaw or receiver-pose randomization."""

    def setup_demo(self, *, probe_job: dict[str, Any], seed: int, **kwargs: Any) -> None:
        render_dir = kwargs.pop("skill_render_dir", None)
        self.probe_job = probe_job
        self.probe_seed = int(seed)
        self.fixture = canonical_scene_fixture(probe_job, seed)
        self.relation = str(probe_job["relation"])
        self.selected_arm = ArmTag(str(probe_job["selected_arm"]))
        self.proxy_contents: list[Any] = []
        self.proxy_content_radius_m = 0.005
        self.released = False
        self.settled = False
        self.checker_success = False
        self.probe_metrics: dict[str, Any] = {}
        self.skill_render_dir = Path(render_dir) if render_dir else None
        # Articulated receivers still use the established visual URDF loader.
        # Rigid canonical jobs can run on an explicit PhysX-only scene.
        self.physics_only = bool(
            self.skill_render_dir is None
            and not probe_job.get("receiver_state_profile")
        )
        self.hang_stepped_entry = (
            self.relation == "hang_on"
            and (
                self.skill_render_dir is not None
                or str(
                    (probe_job.get("probe_spec") or {}).get(
                        "hang_entry_policy",
                        "",
                    )
                )
                in {
                    "stepped_diagnostic_v1",
                    HANG_ENTRY_POLICY_ID,
                }
            )
        )
        self.skill_debug_camera = None
        self.skill_render_count = 0
        kwargs["curobo_planner_arms"] = (str(self.selected_arm),)
        super()._init_task_env_(**kwargs)
        if self.relation == "dump_into":
            self._load_proxy_contents()
            self._hold_grippers_and_step(40)
        if self.skill_render_dir is not None:
            if self.skill_debug_camera is None:
                self._setup_skill_debug_camera()
            self._capture_skill_frame("setup")

    def setup_scene(self, **kwargs: Any) -> None:
        if not self.physics_only:
            super().setup_scene(**kwargs)
            return
        if self.render_freq:
            raise ValueError("physics-only canonical skill probe requires render_freq=0")
        scene_config = sapien.SceneConfig()
        set_scene_config = getattr(sapien.physx, "set_scene_config", None)
        cpu_system_type = getattr(sapien.physx, "PhysxCpuSystem", None)
        if set_scene_config is None or cpu_system_type is None:
            raise RuntimeError("physics_only_sapien_runtime_unavailable")
        set_scene_config(scene_config)
        self.engine = None
        self.renderer = None
        self.scene = sapien.Scene([cpu_system_type()])
        self.scene.set_timestep(kwargs.get("timestep", 1 / 250))
        try:
            self.scene.add_ground(kwargs.get("ground_height", 0), render=False)
        except TypeError:
            self.scene.add_ground(kwargs.get("ground_height", 0))
        self.scene.default_physical_material = self.scene.create_physical_material(
            kwargs.get("static_friction", 0.5),
            kwargs.get("dynamic_friction", 0.5),
            kwargs.get("restitution", 0),
        )
        self.direction_light_lst = []
        self.point_light_lst = []

    def check_stable(self):
        """Capture the canonical fixture before setup stability can reject it."""
        if self.skill_render_dir is not None and self.skill_debug_camera is None:
            self._setup_skill_debug_camera()
            self._capture_skill_frame("pre_stability")
        is_stable, unstable_names = super().check_stable()
        decision = setup_stability_decision(
            unstable_actor_names=list(unstable_names),
            subject_actor_name="m5_skill_subject",
            has_validated_m4_pick_pose=self.subject_initial_m4_pose is not None,
        )
        if decision["delegated_m4_subject_stability"]:
            self.subject.actor.set_pose(self.subject_initial_m4_pose)
            _zero_rigid_velocity(self.subject.actor)
            self.probe_metrics["setup_stability"] = decision
        if decision["setup_stable"]:
            return True, []
        return is_stable, decision["remaining_unstable_actor_names"]

    def load_actors(self) -> None:
        if self.probe_job.get("receiver_state_profile"):
            self._load_articulated_receiver()
        else:
            self._load_rigid_receiver()
        self._load_subject()
        self.add_prohibit_area(self.subject, padding=0.02)
        self.add_prohibit_area(self.receiver, padding=0.02)
        self._validate_initial_pose_preflight()

    def load_camera(self, **kwargs: Any) -> None:
        """Canonical probes are physics-only unless an audit render is asked for."""

        if self.skill_render_dir is not None:
            super().load_camera(**kwargs)
        else:
            self.cameras = None
            # Preserve the one setup physics step performed by Base_Task's
            # camera path; only the render-only work is omitted.
            self.scene.step()

    def create_table_and_wall(
        self,
        table_xy_bias: list[float] = [0, 0],
        table_height: float = 0.74,
    ) -> None:
        if self.skill_render_dir is not None or self.random_background:
            super().create_table_and_wall(
                table_xy_bias=table_xy_bias,
                table_height=table_height,
            )
            return
        self.table_xy_bias = table_xy_bias
        table_height += self.table_z_bias
        self.wall_texture = None
        self.table_texture = None
        self.wall = create_box(
            self.scene,
            sapien.Pose(p=[0, 1, 1.5]),
            half_size=[3, 0.6, 1.5],
            color=(1, 0.9, 0.9),
            name="wall",
            is_static=True,
            load_visual=False,
        )
        self.table = create_table(
            self.scene,
            sapien.Pose(p=[table_xy_bias[0], table_xy_bias[1], table_height]),
            length=1.2,
            width=0.7,
            height=table_height,
            thickness=0.05,
            is_static=True,
            load_visual=False,
        )

    def _load_subject(self) -> None:
        endpoint = self.probe_job["subject"]
        xy = self.fixture["subject"]["xy"]
        terminal_q_wxyz = _yaw_adjusted_q(self._role_q("subject"), self._witness_yaw_deg())
        approved = (self.probe_job.get("canonical_initial_states") or {}).get("subject")
        if not isinstance(approved, dict):
            raise RuntimeError("source_initial_pose_binding_missing:subject")
        initial_yaw_deg = approved_world_yaw_deg(approved)
        initial_q_wxyz = apply_world_yaw(
            approved["q0_wxyz"], initial_yaw_deg
        )
        initial_z = float(approved["support"]["origin_z"]) + float(self.table_z_bias)
        pose = sapien.Pose(
            [xy[0], xy[1], initial_z],
            initial_q_wxyz,
        )
        self.subject = create_actor(
            self,
            pose=pose,
            modelname=str(endpoint["asset"]),
            scale=approved["scale_xyz"],
            model_id=_model_id(endpoint),
            convex=True,
            mass=None,
            load_visual=self.skill_render_dir is not None,
        )
        if self.subject is None:
            raise RuntimeError("actor_load_failed:subject")
        bind_loaded_actor_scale(self.subject, approved["scale_xyz"])
        apply_dynamic_mass_properties(self.subject, approved)
        self.subject.set_name("m5_skill_subject")
        initial_pose = self.subject.get_pose()
        initial_aabb = actor_aabb(self.subject)
        aligned_root = aabb_centered_root_position_xy(
            root_xyz=initial_pose.p.tolist(),
            aabb_center_xyz=initial_aabb["center"],
            target_xy=xy,
        )
        aligned_root[2] = initial_z
        self.subject.actor.set_pose(sapien.Pose(aligned_root, initial_q_wxyz))
        self.subject_initial_m4_pose = sapien.Pose(aligned_root, initial_q_wxyz)

        # Record the terminal witness geometry separately; initial pickup
        # remains the exact approved q0 and may never switch pose at runtime.
        self.subject.actor.set_pose(
            sapien.Pose([xy[0], xy[1], TABLE_TOP_Z + 0.4], terminal_q_wxyz)
        )
        terminal_pose = self.subject.get_pose()
        terminal_aabb = actor_aabb(self.subject)
        terminal_rotation = terminal_pose.to_transformation_matrix()[:3, :3]
        self.subject_terminal_q_wxyz = terminal_pose.q.tolist()
        self.subject_terminal_local_bottom_z = (
            float(terminal_aabb["min"][2]) - float(terminal_pose.p[2])
        )
        self.subject_terminal_local_center_offset = terminal_rotation.T @ (
            np.asarray(terminal_aabb["center"], dtype=float)
            - np.asarray(terminal_pose.p, dtype=float)
        )
        self.subject.actor.set_pose(sapien.Pose(aligned_root, initial_q_wxyz))

    def _load_rigid_receiver(self) -> None:
        endpoint = self.probe_job["receiver"]
        fixture = self.fixture["receiver"]
        approved = (self.probe_job.get("canonical_initial_states") or {}).get("receiver")
        if not isinstance(approved, dict):
            raise RuntimeError("source_initial_pose_binding_missing:receiver")
        self.receiver = create_actor(
            self,
            pose=sapien.Pose(
                [
                    fixture["xy"][0],
                    fixture["xy"][1],
                    float(approved["support"]["origin_z"]) + float(self.table_z_bias),
                ],
                self._role_q("receiver"),
            ),
            modelname=str(endpoint["asset"]),
            scale=approved["scale_xyz"],
            model_id=_model_id(endpoint),
            convex=True,
            is_static=True,
            mass=None,
            load_visual=self.skill_render_dir is not None,
        )
        if self.receiver is None:
            raise RuntimeError("actor_load_failed:receiver")
        bind_loaded_actor_scale(self.receiver, approved["scale_xyz"])
        receiver_yaw_deg = approved_world_yaw_deg(approved)
        receiver_q_wxyz = apply_world_yaw(
            approved["q0_wxyz"], receiver_yaw_deg
        )
        self.receiver.actor.set_pose(
            sapien.Pose(
                [
                    fixture["xy"][0],
                    fixture["xy"][1],
                    float(approved["support"]["origin_z"])
                    + float(self.table_z_bias),
                ],
                receiver_q_wxyz,
            )
        )
        self.receiver.set_name("m5_skill_receiver")

    def _load_articulated_receiver(self) -> None:
        endpoint = self.probe_job["receiver"]
        fixture = self.fixture["receiver"]
        approved = (self.probe_job.get("canonical_initial_states") or {}).get(
            "receiver"
        )
        if not isinstance(approved, dict):
            raise RuntimeError("source_initial_pose_binding_missing:receiver")
        self.receiver = create_sapien_urdf_obj(
            self,
            pose=sapien.Pose(
                [
                    fixture["xy"][0],
                    fixture["xy"][1],
                    float(approved["support"]["origin_z"])
                    + float(self.table_z_bias),
                ],
                approved["q0_wxyz"],
            ),
            modelname=str(endpoint["asset"]),
            modelid=int(endpoint["model_id"]),
            fix_root_link=True,
            mass=None,
        )
        if self.receiver is None:
            raise RuntimeError("actor_load_failed:receiver")
        apply_dynamic_mass_properties(self.receiver, approved)
        receiver_yaw_deg = approved_world_yaw_deg(approved)
        receiver_q_wxyz = apply_world_yaw(
            approved["q0_wxyz"], receiver_yaw_deg
        )
        self.receiver.actor.set_pose(
            sapien.Pose(
                [
                    fixture["xy"][0],
                    fixture["xy"][1],
                    float(approved["support"]["origin_z"])
                    + float(self.table_z_bias),
                ],
                receiver_q_wxyz,
            )
        )
        profile = self.probe_job["receiver_state_profile"]
        joint_state = approved.get("articulated_joint_state") or {}
        qpos = np.asarray(joint_state.get("qpos") or [], dtype=float)
        resolved = list(joint_state.get("joint_targets") or [])
        if qpos.shape != np.asarray(self.receiver.get_qpos()).shape:
            raise RuntimeError(
                "source_initial_pose_joint_count_mismatch:receiver"
            )
        self.receiver.set_qpos(qpos)
        self.receiver.set_qvel(np.zeros_like(self.receiver.get_qvel()))
        self.initial_receiver_qpos = np.asarray(qpos, dtype=float)
        self.resolved_joint_targets = resolved
        drive = profile.get("passive_joint_properties") or {}
        if drive:
            _hold_reviewed_joint_targets(self.receiver, resolved, drive)
        self.receiver.set_name("m5_skill_articulated_receiver")

    def _load_proxy_contents(self) -> None:
        center = np.asarray(actor_aabb(self.subject)["center"], dtype=float)
        offsets = [
            [-0.012, 0.000, 0.012],
            [0.000, 0.000, 0.012],
            [0.012, 0.000, 0.012],
            [-0.006, 0.012, 0.012],
            [0.006, 0.012, 0.012],
        ]
        for index, offset_xyz in enumerate(offsets):
            offset = np.asarray(offset_xyz, dtype=float)
            proxy = create_sphere(
                self.scene,
                pose=sapien.Pose(center + offset, [1.0, 0.0, 0.0, 0.0]),
                radius=self.proxy_content_radius_m,
                color=[1.0, 0.1, 0.1],
                name=f"m5_proxy_{index}",
                load_visual=self.skill_render_dir is not None,
            )
            self.proxy_contents.append(proxy)

    def play_once(self) -> dict[str, Any]:
        probe_spec = self.probe_job.get("probe_spec") or {}
        contact_ids = (
            [4, 5, 6, 7]
            if self.relation == "stamp_on"
            else (1 if self.relation == "dump_into" else self._m4_pick_contact_ids())
        )
        gripper_pos = float(probe_spec.get("skill_gripper_pos") or 0.0)
        self.probe_metrics["pick_contact_ids"] = contact_ids
        self.probe_metrics["gripper_target_position"] = gripper_pos
        self._move_phase(
            "grasp",
            self.grasp_actor(
                self.subject,
                arm_tag=self.selected_arm,
                pre_grasp_dis=0.08,
                gripper_pos=gripper_pos,
                contact_point_id=contact_ids,
            )
        )
        if not self.plan_success:
            return self._finish_failed_rollout()
        if self.relation == "dump_into":
            lift_action = self.move_by_displacement(
                self.selected_arm,
                z=0.08,
                move_axis="arm",
            )
        else:
            lift_action = self.move_by_displacement(self.selected_arm, z=0.10)
        self._move_phase("lift", lift_action)
        if not self.plan_success:
            return self._finish_failed_rollout()
        # M5 intentionally evaluates terminal placement with lightweight
        # subject dynamics after the R01/M4 pickup phase has completed.
        if self.plan_success:
            self.subject.set_mass(0.01)
        if self.relation in {"place_in", "place_on", "stack"}:
            if self.probe_job.get("receiver_state_profile"):
                self._articulated_place_in()
            else:
                self._top_down_place()
        elif self.relation == "hang_on":
            self._hang_on()
        elif self.relation == "dump_into":
            self._dump_into()
        elif self.relation == "stamp_on":
            self._stamp_on()
        else:
            self.plan_success = False
            self.probe_metrics["failure_reason"] = "unsupported_relation"
        if not self.plan_success:
            return self._finish_failed_rollout()
        # Let released objects settle before measuring a separate stability
        # window.  ``Base_Task.delay`` counts controller iterations, not
        # seconds; four iterations was too short for a hanging mug to damp.
        self._hold_grippers_and_step(60)
        before = np.asarray(self.subject.get_pose().p, dtype=float)
        self._hold_grippers_and_step(25)
        after = np.asarray(self.subject.get_pose().p, dtype=float)
        self.settled = bool(np.linalg.norm(after - before) <= 0.01)
        self.released = self._selected_gripper_open()
        self.info["info"] = {
            "pair_key": self.probe_job["pair_key"],
            "relation": self.relation,
            "selected_arm": str(self.selected_arm),
        }
        self._capture_skill_frame("terminal")
        return self.info

    def _finish_failed_rollout(self) -> dict[str, Any]:
        """Return immediately once planning has irreversibly failed.

        Aggregation already treats ``plan_success=False`` as the semantic
        result and never invokes the checker.  Running the old terminal hold
        after that point only consumed physics steps; it could not recover the
        attempt.
        """

        self.settled = False
        self.released = self._selected_gripper_open()
        self.probe_metrics["settle_skipped_due_to_plan_failure"] = True
        self.info["info"] = {
            "pair_key": self.probe_job["pair_key"],
            "relation": self.relation,
            "selected_arm": str(self.selected_arm),
        }
        self._capture_skill_frame("plan_failed")
        return self.info

    def _hold_grippers_and_step(self, controller_iterations: int) -> None:
        """Fast equivalent of ``delay`` for an unchanged gripper target.

        ``Base_Task.delay`` rebuilds two 200-sample constant trajectories plus
        100 padded samples on every controller iteration.  This executes the
        same left/right drive writes and the same 300 PhysX steps per
        iteration, but constructs no redundant arrays or control dictionaries.
        """

        hold_grippers_and_step(
            self.robot,
            self.scene,
            controller_iterations,
        )

    def _top_down_place(self) -> None:
        anchor = self._receiver_target(top=True)
        if self.relation == "place_on" and self.subject.get_functional_point(0) is not None:
            self._functional_support_place(anchor)
            return
        receiver_aabb = actor_aabb(self.receiver)
        support_top_z = (
            float(receiver_aabb["max"][2])
            if self.relation in {"place_on", "stack", "stamp_on"}
            else float(anchor[2])
        )
        target_rotation = sapien.Pose(
            [0.0, 0.0, 0.0],
            self.subject_terminal_q_wxyz,
        ).to_transformation_matrix()[:3, :3]
        target_center_offset = target_rotation @ self.subject_terminal_local_center_offset
        target_root_xyz = [
            float(anchor[0]) - float(target_center_offset[0]),
            float(anchor[1]) - float(target_center_offset[1]),
            support_top_z
            + self._terminal_clearance_m()
            - self.subject_terminal_local_bottom_z,
        ]
        target_actor_pose = target_root_xyz + self.subject_terminal_q_wxyz
        approach_height = 0.05 if self.relation == "stack" else 0.10
        approach_pose, final_pose = self._ee_poses_for_actor_targets(
            target_actor_pose,
            approach_actor_pose=(
                (
                    np.asarray(target_root_xyz, dtype=float)
                    + np.array([0.0, 0.0, approach_height])
                ).tolist()
                + self.subject_terminal_q_wxyz
            ),
        )
        self.probe_metrics["top_down_waypoints"] = {
            "anchor_pose": [float(value) for value in anchor],
            "target_actor_pose": [float(value) for value in target_actor_pose],
            "entry_direction_world": [0.0, 0.0, -1.0],
            "approach_height_m": approach_height,
            "current_actor_pose": self.subject.get_pose().p.tolist()
            + self.subject.get_pose().q.tolist(),
            "current_ee_pose": self._selected_ee_pose(),
            "approach_ee_pose": [float(value) for value in approach_pose],
            "final_ee_pose": [float(value) for value in final_pose],
        }
        self._move_phase(
            "top_down_approach",
            self.move_to_pose(self.selected_arm, approach_pose),
        )
        self._move_phase(
            "top_down_descend",
            self.move_to_pose(self.selected_arm, final_pose),
        )
        self._move_phase("top_down_release", self.open_gripper(self.selected_arm))
        self._move_phase(
            "top_down_retract",
            self.move_to_pose(self.selected_arm, approach_pose),
        )

    def _functional_support_place(self, anchor: list[float]) -> None:
        """Reuse the reviewed RoboTwin place-on functional-point primitive."""

        approach_distance, final_distance, pre_axis = 0.12, 0.03, "grasp"
        args = {
            "actor": self.subject,
            "arm_tag": self.selected_arm,
            "target_pose": anchor,
            "functional_point_id": 0,
            "constrain": "free",
            "pre_dis_axis": pre_axis,
        }
        approach_pose = self.get_place_pose(pre_dis=approach_distance, **args)
        final_pose = self.get_place_pose(pre_dis=final_distance, **args)
        self.probe_metrics["top_down_waypoints"] = {
            "adapter": f"reviewed_{self.relation}_functional_point_skill",
            "anchor_pose": [float(value) for value in anchor],
            "subject_functional_point_id": 0,
            "approach_ee_pose": [float(value) for value in approach_pose],
            "final_ee_pose": [float(value) for value in final_pose],
        }
        self._move_phase("top_down_approach", self.move_to_pose(self.selected_arm, approach_pose))
        self._move_phase("top_down_descend", self.move_to_pose(self.selected_arm, final_pose))
        self._move_phase("top_down_release", self.open_gripper(self.selected_arm))
        self._move_phase("top_down_retract", self.move_by_displacement(self.selected_arm, z=0.10))

    def _articulated_place_in(self) -> None:
        profile = self.probe_job["receiver_state_profile"]
        cavity = profile["cavity"]
        link = self.receiver.link_dict[str(cavity["frame_link"])]
        link_pose = link.get_pose()
        cavity_world_matrix = link_pose.to_transformation_matrix()
        if cavity.get("frame_transform") is not None:
            cavity_world_matrix = cavity_world_matrix @ np.asarray(cavity["frame_transform"], dtype=float)
        witness = cavity_witness_world_pose(
            cavity_world_matrix=cavity_world_matrix,
            cavity=cavity,
            subject_q0_wxyz=self._role_q("subject"),
            witness_delta_deg=self._witness_yaw_deg(),
            subject_local_bottom_z=self.subject_terminal_local_bottom_z,
            clearance_m=self._terminal_clearance_m(),
        )
        target_actor_pose = witness["p"] + witness["q_wxyz"]
        target_rotation = sapien.Pose(witness["p"], witness["q_wxyz"]).to_transformation_matrix()[:3, :3]
        target_center_offset = target_rotation @ self.subject_terminal_local_center_offset
        target_center_world = np.asarray(witness["p"], dtype=float) + target_center_offset
        waypoints = articulated_entry_waypoints(
            profile,
            cavity_world_xyz=cavity_world_matrix[:3, 3].tolist(),
            cavity_world_q_wxyz=_matrix_q_wxyz(cavity_world_matrix[:3, :3]),
            final_world_xyz=target_center_world.tolist(),
        )
        center_to_root = -target_center_offset
        approach_root = np.asarray(waypoints["approach_world_xyz"], dtype=float) + center_to_root
        approach_actor_pose = approach_root.tolist() + witness["q_wxyz"]
        approach_ee_pose, final_ee_pose = self._ee_poses_for_actor_targets(
            target_actor_pose,
            approach_actor_pose=approach_actor_pose,
        )
        self.probe_metrics["articulated_waypoints"] = {
            **waypoints,
            "approach_actor_pose": approach_actor_pose,
            "target_actor_pose": target_actor_pose,
            "approach_ee_pose": approach_ee_pose,
            "final_ee_pose": final_ee_pose,
        }
        self._move_phase(
            "articulated_approach",
            self.move_to_pose(self.selected_arm, approach_ee_pose),
        )
        self._move_phase(
            "articulated_insert",
            self.move_to_pose(self.selected_arm, final_ee_pose),
        )
        self._move_phase("articulated_release", self.open_gripper(self.selected_arm))
        retreat = np.asarray(waypoints["retreat_world_xyz"]) - np.asarray(waypoints["final_world_xyz"])
        self._move_phase(
            "articulated_retreat",
            self.move_by_displacement(
                self.selected_arm,
                x=float(retreat[0]),
                y=float(retreat[1]),
                z=float(retreat[2]),
            )
        )

    def _ee_poses_for_actor_targets(
        self,
        final_actor_pose: list[float],
        *,
        approach_actor_pose: list[float],
    ) -> tuple[list[float], list[float]]:
        current_actor_matrix = self.subject.get_pose().to_transformation_matrix()
        current_ee = self._selected_ee_pose()
        current_ee_matrix = sapien.Pose(current_ee[:3], current_ee[3:]).to_transformation_matrix()
        actor_to_ee = np.linalg.inv(current_actor_matrix) @ current_ee_matrix

        def transfer(actor_pose: list[float]) -> list[float]:
            matrix = sapien.Pose(actor_pose[:3], actor_pose[3:]).to_transformation_matrix() @ actor_to_ee
            return matrix[:3, 3].tolist() + _matrix_q_wxyz(matrix[:3, :3])

        return transfer(approach_actor_pose), transfer(final_actor_pose)

    def _selected_ee_pose(self) -> list[float]:
        pose = (
            self.robot.get_left_ee_pose()
            if str(self.selected_arm) == "left"
            else self.robot.get_right_ee_pose()
        )
        return np.asarray(pose, dtype=float).tolist()

    def _m4_pick_contact_ids(self) -> int | list[int]:
        groups = self.subject.config.get("contact_points_group") or []
        if not groups:
            return 0
        first = [int(value) for value in groups[0]]
        return first[0] if len(first) == 1 else first

    def _terminal_clearance_m(self) -> float:
        recipe = self.probe_job.get("placement_recipe") or {}
        default = 0.03 if self.relation in {"place_in", "dump_into"} else 0.008
        return float(recipe.get("clearance_m") if recipe.get("clearance_m") is not None else default)

    def _hang_on(self) -> None:
        target = self.receiver.get_functional_point(0)
        if target is None:
            self.plan_success = False
            return
        contract = hang_skill_phase_contract(
            self.probe_job.get("probe_spec") or {}
        )
        if not self.hang_stepped_entry:
            self._move_phase(
                contract["legacy_phases"][0],
                self.place_actor(
                    self.subject,
                    arm_tag=self.selected_arm,
                    target_pose=target,
                    functional_point_id=0,
                    constrain="align",
                    pre_dis=contract["legacy_approach_distance_m"],
                    dis=contract["legacy_insert_distance_m"],
                    pre_dis_axis="fp",
                ),
            )
            self._move_phase(
                contract["legacy_phases"][1],
                self.move_by_displacement(
                    self.selected_arm,
                    z=0.10,
                    move_axis="arm",
                    quat=GRASP_DIRECTION_DIC["front"],
                ),
            )
            return
        place_args = {
            "actor": self.subject,
            "arm_tag": self.selected_arm,
            "target_pose": target,
            "functional_point_id": 0,
            "constrain": "align",
            "pre_dis_axis": "fp",
        }
        approach_pose = self.get_place_pose(
            pre_dis=contract["approach_distance_m"],
            **place_args,
        )
        self.probe_metrics["hang_waypoints"] = {
            "receiver_functional_pose": [float(value) for value in target],
            "approach_ee_pose": [float(value) for value in approach_pose],
            "legacy_approach_distance_m": contract[
                "legacy_approach_distance_m"
            ],
            "approach_distance_m": contract[
                "approach_distance_m"
            ],
            "insert_distances_m": list(
                contract["insert_distances_m"]
            ),
            "phases": list(contract["phases"]),
            "insert_steps": [],
        }
        self._move_phase(
            contract["phases"][0],
            self.move_to_pose(self.selected_arm, approach_pose),
        )
        insert_distances = list(contract["insert_distances_m"])
        for index, distance in enumerate(insert_distances):
            insert_pose = self.get_place_pose(
                pre_dis=float(distance),
                **place_args,
            )
            phase = (
                contract["phases"][1]
                if index == len(insert_distances) - 1
                else f"hang_insert_step_{index + 1}"
            )
            self.probe_metrics["hang_waypoints"]["insert_steps"].append(
                {
                    "phase": phase,
                    "pre_distance_m": float(distance),
                    "ee_pose": [float(value) for value in insert_pose],
                }
            )
            self._move_phase(
                phase,
                self.move_to_pose(self.selected_arm, insert_pose),
            )
        readiness = self._hang_pre_release_readiness(contract)
        for correction_index in range(
            int(contract["max_corrections"])
        ):
            if readiness["ready"] or not self.plan_success:
                break
            correction_pose = self.get_place_pose(
                pre_dis=contract["terminal_insert_distance_m"],
                **place_args,
            )
            self._move_phase(
                f"hang_corrective_insert_{correction_index + 1}",
                self.move_to_pose(self.selected_arm, correction_pose),
            )
            readiness = self._hang_pre_release_readiness(contract)
        self.probe_metrics["hang_pre_release_readiness"] = readiness
        if not self.plan_success:
            return
        if (
            contract["require_readiness_before_release"]
            and not readiness["ready"]
        ):
            self.probe_metrics[
                "failure_reason"
            ] = "hang_pre_release_alignment_failed"
            self.plan_success = False
            return
        self._move_phase(
            contract["phases"][2],
            self.open_gripper(self.selected_arm),
        )
        self._move_phase(
            contract["phases"][3],
            self.move_by_displacement(
                self.selected_arm,
                z=0.10,
                move_axis="arm",
                quat=GRASP_DIRECTION_DIC["front"],
            )
        )

    def _hang_pre_release_readiness(
        self,
        contract: dict[str, Any],
    ) -> dict[str, Any]:
        state = self._hang_phase_state()
        translation_error = float(
            state["functional_translation_error_m"]
        )
        rotation_error = float(state["functional_rotation_error_deg"])
        translation_tolerance = float(
            contract["translation_tolerance_m"]
        )
        rotation_tolerance = float(
            contract["rotation_tolerance_deg"]
        )
        return {
            "ready": (
                translation_error <= translation_tolerance
                and rotation_error <= rotation_tolerance
            ),
            "functional_translation_error_m": translation_error,
            "functional_rotation_error_deg": rotation_error,
            "translation_tolerance_m": translation_tolerance,
            "rotation_tolerance_deg": rotation_tolerance,
        }

    def _dump_into(self) -> None:
        target = np.asarray(self._receiver_target(top=True)[:3], dtype=float)
        pour_pose = np.array([target[0], target[1] - 0.05, 1.05], dtype=float)
        actions = [
            Action(
                self.selected_arm,
                "move",
                target_pose=pour_pose.tolist() + [-0.694654, -0.178228, 0.165979, -0.676862],
            )
        ]
        self._move_phase("dump_pour_pose", (self.selected_arm, actions))
        for delta in (-0.015, 0.015, -0.01, 0.01):
            self._move_phase(
                "dump_shake",
                self.move_by_displacement(self.selected_arm, y=delta, z=-abs(delta) / 2.0),
            )
        self._move_phase("dump_release", self.open_gripper(self.selected_arm))
        self._hold_grippers_and_step(6)

    def _stamp_on(self) -> None:
        target = self._receiver_target(top=True)
        args = {
            "actor": self.subject,
            "arm_tag": self.selected_arm,
            "target_pose": target,
            "constrain": "auto",
        }
        approach_pose = self.get_place_pose(pre_dis=0.05, **args)
        contact_pose = self.get_place_pose(pre_dis=0.02, **args)
        self.probe_metrics["stamp_waypoints"] = {
            "target_pose": [float(value) for value in target],
            "approach_ee_pose": [float(value) for value in approach_pose],
            "contact_ee_pose": [float(value) for value in contact_pose],
        }
        self._move_phase(
            "stamp_approach",
            self.move_to_pose(self.selected_arm, approach_pose),
        )
        self._move_phase("stamp_contact", self.move_to_pose(self.selected_arm, contact_pose))
        self._move_phase("stamp_release", self.open_gripper(self.selected_arm))
        self._move_phase("stamp_retract", self.move_to_pose(self.selected_arm, approach_pose))

    def _move_phase(self, phase: str, actions: Any) -> bool:
        before = bool(self.plan_success)
        result = bool(self.move(actions))
        after = bool(self.plan_success)
        trace = self.probe_metrics.setdefault("motion_phase_trace", [])
        phase_row = {
            "phase": phase,
            "entered": before,
            "move_returned": result,
            "plan_success_after": after,
        }
        if self.probe_job.get("receiver_state_profile"):
            phase_row["receiver_qpos_after"] = np.asarray(
                self.receiver.get_qpos(),
                dtype=float,
            ).tolist()
            subject_pose = self.subject.get_pose()
            phase_row["subject_root_pose_after"] = (
                np.asarray(subject_pose.p, dtype=float).tolist()
                + np.asarray(subject_pose.q, dtype=float).tolist()
            )
        if self.hang_stepped_entry:
            self.probe_metrics.setdefault("hang_phase_states", {})[phase] = (
                self._hang_phase_state()
            )
        trace.append(phase_row)
        if before and not after and "motion_failure_phase" not in self.probe_metrics:
            self.probe_metrics["motion_failure_phase"] = phase
        self._capture_skill_frame(phase)
        return result

    def _hang_phase_state(self) -> dict[str, Any]:
        """Capture target, contact, and dynamics evidence for each Hang phase."""

        subject_pose = self.subject.get_pose()
        subject_matrix = subject_pose.to_transformation_matrix()
        subject_fp_matrix = np.asarray(
            self.subject.get_functional_point(0, "matrix"),
            dtype=float,
        )
        receiver_fp_matrix = np.asarray(
            self.receiver.get_functional_point(0, "matrix"),
            dtype=float,
        )
        terminal_root_matrix = aligned_functional_root_pose(
            subject_matrix,
            subject_fp_matrix,
            receiver_fp_matrix,
            offset_along_receiver_axis_m=0.05,
        )
        desired_fp_matrix = receiver_fp_matrix.copy()
        desired_fp_matrix[:3, 3] += desired_fp_matrix[:3, 2] * 0.05
        velocity = _rigid_velocity_state(self.subject.actor)
        gripper_contacts = [
            np.asarray(position, dtype=float).tolist()
            for position in self.get_gripper_actor_contact_position(
                self.subject.get_name()
            )
        ]
        return {
            "subject_root_pose": (
                np.asarray(subject_pose.p, dtype=float).tolist()
                + np.asarray(subject_pose.q, dtype=float).tolist()
            ),
            "subject_functional_pose": _matrix_pose_list(subject_fp_matrix),
            "receiver_functional_pose": _matrix_pose_list(receiver_fp_matrix),
            "desired_terminal_functional_pose": _matrix_pose_list(
                desired_fp_matrix
            ),
            "desired_terminal_root_pose": _matrix_pose_list(
                terminal_root_matrix
            ),
            "functional_translation_error_m": float(
                np.linalg.norm(
                    subject_fp_matrix[:3, 3] - desired_fp_matrix[:3, 3]
                )
            ),
            "functional_rotation_error_deg": _rotation_error_deg(
                subject_fp_matrix[:3, :3],
                desired_fp_matrix[:3, :3],
            ),
            "root_translation_error_m": float(
                np.linalg.norm(
                    subject_matrix[:3, 3] - terminal_root_matrix[:3, 3]
                )
            ),
            "root_rotation_error_deg": _rotation_error_deg(
                subject_matrix[:3, :3],
                terminal_root_matrix[:3, :3],
            ),
            "linear_velocity_mps": velocity["linear"],
            "angular_velocity_radps": velocity["angular"],
            "subject_rack_contact": contact_metrics(
                self.scene,
                self.subject.actor,
                self.receiver.actor,
            ),
            "subject_table_contact": contact_metrics(
                self.scene,
                self.subject.actor,
                self.table,
            ),
            "subject_gripper_contact_count": len(gripper_contacts),
            "subject_gripper_contact_positions": gripper_contacts,
        }

    def _setup_skill_debug_camera(self) -> None:
        self.skill_render_dir.mkdir(parents=True, exist_ok=True)
        camera = self.scene.add_camera(
            name="m5_skill_debug_camera",
            width=800,
            height=600,
            fovy=np.deg2rad(62.0),
            near=0.04,
            far=5.0,
        )
        if self.probe_job.get("receiver_state_profile"):
            # An articulated receiver can fully occlude a front camera (the
            # reviewed cabinet is taller than the insertion target).  An
            # oblique robot-side view keeps the drawer/door, gripper, and
            # subject visible in the same frame.
            position = np.array([-0.46, -0.30, 1.08], dtype=float)
            target = np.array([0.0, 0.07, 0.84], dtype=float)
            forward = target - position
            forward /= np.linalg.norm(forward)
            left = np.cross(np.array([0.0, 0.0, 1.0], dtype=float), forward)
            left /= np.linalg.norm(left)
        else:
            position = np.array([0.0, 0.42, 1.30], dtype=float)
            target = np.array([0.0, 0.05, 0.82], dtype=float)
            left = np.array([1.0, 0.0, 0.0], dtype=float)
            forward = target - position
            forward /= np.linalg.norm(forward)
        up = np.cross(forward, left)
        matrix = np.eye(4)
        matrix[:3, :3] = np.stack([forward, left, up], axis=1)
        matrix[:3, 3] = position
        camera.entity.set_pose(sapien.Pose(matrix))
        self.skill_debug_camera = camera

    def _capture_skill_frame(self, phase: str) -> None:
        if self.skill_debug_camera is None or self.skill_render_dir is None:
            return
        from PIL import Image, ImageDraw

        self.scene.update_render()
        self.skill_debug_camera.take_picture()
        rgba = np.asarray(self.skill_debug_camera.get_picture("Color"))
        rgb = (rgba[..., :3] * 255).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(image)
        draw.rectangle((8, 8, 790, 54), fill=(0, 0, 0))
        draw.text(
            (14, 14),
            f"{self.probe_job['pair_key']}  seed={self.probe_seed}",
            fill=(255, 255, 255),
        )
        draw.text(
            (14, 32),
            f"phase={phase}  plan_success={self.plan_success}",
            fill=(255, 255, 255),
        )
        path = self.skill_render_dir / f"{self.skill_render_count:03d}_{phase}.png"
        self.skill_render_count += 1
        image.save(path)

    def check_success(self) -> bool:
        if self.relation in {"place_in", "place_on", "stack"}:
            contract = SuccessContract.load()
            profile = load_embodiment_profile(DEFAULT_ALOHA_GRIPPER_PROFILE)
            endpoint_by_name = {
                str(self.subject.get_name()): "subject",
                str(self.receiver.get_name()): "receiver",
                str(self.table.get_name()): "support:table",
            }
            raw_receiver = getattr(self.receiver, "actor", self.receiver)
            if hasattr(raw_receiver, "get_links"):
                for link in raw_receiver.get_links():
                    endpoint_by_name[str(link.get_name())] = "receiver"
            for arm, arm_profile in profile["arms"].items():
                for row in arm_profile.get("fingers") or []:
                    endpoint_by_name[str(row["link"])] = f"gripper:{arm}"
                for name in arm_profile.get("palm_wrist_links") or []:
                    endpoint_by_name[str(name)] = f"gripper:{arm}"
            minimum_steps = int(contract.section("stability")["minimum_settle_steps"])
            window_steps = int(contract.section("stability")["observation_window_steps"])
            for _ in range(minimum_steps):
                self.scene.step()
            history = []
            for _ in range(window_steps):
                self.scene.step()
                history.append(
                    [
                        {"p": actor.get_pose().p.tolist(), "q": actor.get_pose().q.tolist()}
                        for actor in (self.subject, self.receiver)
                    ]
                )
            stability_metrics = stability_from_snapshots(history)
            grippers = observe_actual_grippers(
                self.robot, embodiment_profile=profile
            )
            active_contacts = observe_sapien_active_contacts(
                self.scene,
                endpoint_by_body_name=endpoint_by_name,
                embodiment_profile=profile,
                grippers=grippers,
                contract=contract,
            )
            per_entity = stability_metrics.get("per_entity") or []
            entity_stability = {
                name: (
                    dict(per_entity[index])
                    if index < len(per_entity)
                    else {"stable": False}
                )
                for index, name in enumerate(("subject", "receiver"))
            }
            subject_state = self._actor_state(self.subject)
            receiver_state = self._actor_state(self.receiver)
            terminal = {
                "entities": {
                    "subject": {"pose": {"position": subject_state["p"], "quat_wxyz": subject_state["q"]}, "aabb": subject_state["aabb"]},
                    "receiver": {"pose": {"position": receiver_state["p"], "quat_wxyz": receiver_state["q"]}, "aabb": receiver_state["aabb"]},
                },
                "grippers": grippers,
                "active_contacts": active_contacts,
            }
            snapshot = build_success_snapshot(
                initial=terminal,
                terminal=terminal,
                stability={
                    "entities": entity_stability,
                    "whole_scene": {"stable": bool(stability_metrics.get("stable"))},
                },
                execution_events={"released": self.released},
                source_hashes={"pair_probe_job": artifact_hash(self.probe_job)},
                contract=contract,
                profiles={"embodiment": profile},
            )
            result = evaluate_m5_skill_success(
                "placement_completed",
                {
                    "subject": "subject",
                    "target": "receiver",
                    "relation": {"place_in": "in", "place_on": "on", "stack": "stack"}[self.relation],
                },
                snapshot,
                contract,
            )
            self.checker_success = result["status"] == "passed"
            self.probe_metrics.update(
                {
                    **result["metrics"],
                    "predicate_result": result,
                    **contract.identity,
                    "relation_contract_hash": result.get("relation_contract_hash"),
                    "success_snapshot_hash": snapshot["success_snapshot_hash"],
                    "released": self.released,
                    "settled": self.settled,
                    "canonical_fixture": self.fixture,
                }
            )
            return self.checker_success
        stability = {"stable": self.settled}
        subject = self._actor_state(self.subject)
        receiver = self._actor_state(self.receiver)
        contacts = {
            "subject_receiver": self._has_actor_contact(self.subject, self.receiver),
            "subject_table": self._has_actor_contact(self.subject, self.table),
        }
        if self.relation == "place_in":
            if self.probe_job.get("receiver_state_profile"):
                metrics = self._articulated_place_in_metrics(subject, stability)
            else:
                metrics = place_in_success_metrics(subject, receiver, contacts, stability)
        elif self.relation == "place_on":
            metrics = place_on_success_metrics(subject, receiver, contacts, stability)
        elif self.relation == "stack":
            metrics = stack_success_metrics(subject, receiver, contacts, stability)
        elif self.relation == "hang_on":
            metrics = hang_on_success_metrics(subject, receiver, stability)
        elif self.relation == "dump_into":
            sphere_states = [
                {
                    "aabb": {
                        "center": list(proxy.get_pose().p),
                        "min": (
                            np.asarray(proxy.get_pose().p) - self.proxy_content_radius_m
                        ).tolist(),
                        "max": (
                            np.asarray(proxy.get_pose().p) + self.proxy_content_radius_m
                        ).tolist(),
                    }
                }
                for proxy in self.proxy_contents
            ]
            metrics = dump_into_success_metrics(receiver["aabb"], sphere_states, TABLE_TOP_Z, stability)
        else:
            metrics = stamp_on_success_metrics(
                subject,
                receiver,
                contacts,
                stability,
                self.probe_job.get("probe_spec") or {},
            )
        self.checker_success = bool(metrics.get("success"))
        self.probe_metrics.update(
            {
                **metrics,
                "released": self.released,
                "settled": self.settled,
                "canonical_fixture": self.fixture,
            }
        )
        return self.checker_success

    def _articulated_place_in_metrics(self, subject: dict[str, Any], stability: dict[str, Any]) -> dict[str, Any]:
        profile = self.probe_job["receiver_state_profile"]
        cavity = profile["cavity"]
        link = self.receiver.link_dict[str(cavity["frame_link"])]
        cavity_world_matrix = link.get_pose().to_transformation_matrix()
        if cavity.get("frame_transform") is not None:
            cavity_world_matrix = cavity_world_matrix @ np.asarray(cavity["frame_transform"], dtype=float)
        contact_by_link = {}
        for name in sorted(set(cavity["allowed_support_links"]) | set(cavity["forbidden_support_links"])):
            receiver_link = self.receiver.link_dict.get(str(name))
            contact_by_link[str(name)] = (
                contact_metrics(self.scene, self.subject.actor, _link_entity(receiver_link))
                if receiver_link is not None
                else {"active": False, "missing_link": True}
            )
        articulation_state = articulation_state_metrics(
            final_qpos=np.asarray(self.receiver.get_qpos(), dtype=float).tolist(),
            resolved_joint_targets=self.resolved_joint_targets,
        )
        metrics = articulated_place_in_success_metrics(
            subject_state=subject,
            cavity_world_matrix=cavity_world_matrix,
            cavity=cavity,
            contact_by_link=contact_by_link,
            subject_table_contact=self._has_actor_contact(self.subject, self.table),
            subject_world_contact=False,
            terminal_stability=stability,
            articulation_state=articulation_state,
        )
        metrics["contact_by_link"] = contact_by_link
        return metrics

    def _actor_state(self, actor: Any) -> dict[str, Any]:
        state = {
            "p": actor.get_pose().p.tolist(),
            "q": actor.get_pose().q.tolist(),
            "aabb": actor_aabb(actor),
        }
        fp = actor.get_functional_point(0)
        if fp is not None:
            state["functional_point_0"] = list(fp[:3])
        if actor is self.subject and self.relation == "stamp_on":
            points = [actor.get_contact_point(index) for index in [4, 5, 6, 7]]
            if all(point is not None for point in points):
                state["contact_face_4_5_6_7"] = np.mean(np.asarray(points)[:, :3], axis=0).tolist()
        return state

    def _receiver_target(self, *, top: bool) -> list[float]:
        probe_spec = self.probe_job.get("probe_spec") or {}
        anchor = probe_spec.get("receiver_target") or probe_spec.get("terminal_anchor") or {}
        if anchor.get("type") == "functional_point":
            point = self.receiver.get_functional_point(int(anchor.get("id") or 0))
            if point is not None:
                return list(point)
        if anchor.get("type") == "target_pose":
            point = self.receiver.get_target_point(int(anchor.get("id") or 0))
            if point is not None:
                return list(point)
        aabb = actor_aabb(self.receiver)
        center = list(aabb["center"])
        if top:
            center[2] = aabb["max"][2]
        return center + self.receiver.get_pose().q.tolist()

    def _subject_half_height(self) -> float:
        aabb = actor_aabb(self.subject)
        return max(0.005, (float(aabb["max"][2]) - float(aabb["min"][2])) / 2.0)

    def _selected_gripper_open(self) -> bool:
        observed = (
            self.is_left_gripper_open()
            if str(self.selected_arm) == "left"
            else self.is_right_gripper_open()
        )
        return bool(observed)

    def _has_actor_contact(self, left: Any, right: Any) -> bool:
        left_entity = getattr(left, "actor", left)
        right_entity = getattr(right, "actor", right)
        return has_contact(self.scene, left_entity, right_entity)

    def _role_q(self, role: str) -> list[float]:
        poses = (self.probe_job.get("probe_spec") or {}).get("canonical_poses") or {}
        pose = poses.get(role) or {}
        if not pose:
            hint = PAIR_QPOS_HINTS.get((self.relation, role))
            if hint is not None:
                return list(hint)
        return list(
            pose.get("q_wxyz")
            or pose.get("q0_wxyz")
            or pose.get("q")
            or [1.0, 0.0, 0.0, 0.0]
        )

    def _validate_initial_pose_preflight(self) -> None:
        states = self.probe_job.get("canonical_initial_states") or {}
        for role, actor in (("subject", self.subject), ("receiver", self.receiver)):
            state = states.get(role)
            if not isinstance(state, dict):
                raise RuntimeError(f"source_initial_pose_binding_missing:{role}")
            # Articulated receiver state profiles own their exact qpos.  Rigid
            # endpoints naturally report an empty joint vector.
            actual_joint = []
            if (state.get("articulated_joint_state") or {}).get("mode") != "not_articulated":
                actual_joint = np.asarray(actor.get_qpos(), dtype=float).tolist()
            scale = (actor.config or {}).get("scale") or [1.0, 1.0, 1.0]
            if isinstance(scale, (int, float)):
                scale = [float(scale)] * 3
            elif len(scale) == 1:
                scale = list(scale) * 3
            runtime_state = dict(state)
            runtime_state["support"] = {
                **state["support"],
                "origin_z": float(state["support"]["origin_z"]) + float(self.table_z_bias),
            }
            validate_loaded_initial_pose(
                canonical_state=runtime_state,
                actual_q_wxyz=np.asarray(actor.get_pose().q, dtype=float).tolist(),
                actual_joint_qpos=actual_joint,
                actual_scale_xyz=list(scale),
                actual_origin_z=float(actor.get_pose().p[2]),
                expected_world_yaw_deg=approved_world_yaw_deg(state),
            )

    def _witness_yaw_deg(self) -> float:
        witness = self.probe_job.get("yaw_witness") or {}
        return float(witness.get("witness_delta_deg") or 0.0)

    def _m4_pick_pose(self) -> dict[str, Any] | None:
        selected_use = f"pick_{self.selected_arm}"
        for ref in self.probe_job.get("m4_evidence_refs") or []:
            if str(ref.get("use")) != selected_use:
                continue
            pose = ref.get("validated_pose") or {}
            q_wxyz = pose.get("q_wxyz")
            if isinstance(q_wxyz, list) and len(q_wxyz) == 4:
                try:
                    return {
                        **pose,
                        "q_wxyz": [float(value) for value in q_wxyz],
                        "z": float(pose["z"]),
                    }
                except (KeyError, TypeError, ValueError):
                    return None
        return None


def _model_id(endpoint: dict[str, Any]) -> int | str:
    value = str(endpoint["model_id"])
    return int(value) if value.isdigit() else value


def _yaw_adjusted_q(q_wxyz: list[float], yaw_deg: float) -> list[float]:
    theta = math.radians(float(yaw_deg)) / 2.0
    zq = np.asarray([math.cos(theta), 0.0, 0.0, math.sin(theta)])
    q = np.asarray(q_wxyz, dtype=float)
    w1, x1, y1, z1 = zq
    w2, x2, y2, z2 = q
    out = np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )
    return (out / np.linalg.norm(out)).tolist()


def _matrix_q_wxyz(matrix: np.ndarray) -> list[float]:
    import transforms3d as t3d

    return t3d.quaternions.mat2quat(np.asarray(matrix, dtype=float)).tolist()


def _matrix_pose_list(matrix: np.ndarray) -> list[float]:
    matrix = np.asarray(matrix, dtype=float)
    return matrix[:3, 3].tolist() + _matrix_q_wxyz(matrix[:3, :3])


def _rotation_error_deg(left: np.ndarray, right: np.ndarray) -> float:
    relative = np.asarray(left, dtype=float) @ np.asarray(
        right,
        dtype=float,
    ).T
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _rigid_velocity_state(actor: Any) -> dict[str, list[float] | None]:
    try:
        components = actor.get_components()
    except Exception:
        components = []
    for component in components:
        if not isinstance(component, sapien.physx.PhysxRigidDynamicComponent):
            continue
        return {
            "linear": np.asarray(
                component.get_linear_velocity(),
                dtype=float,
            ).tolist(),
            "angular": np.asarray(
                component.get_angular_velocity(),
                dtype=float,
            ).tolist(),
        }
    return {"linear": None, "angular": None}


def _link_entity(link: Any) -> Any:
    entity = getattr(link, "entity", None)
    if entity is not None:
        return entity
    getter = getattr(link, "get_entity", None)
    return getter() if callable(getter) else link


def _zero_rigid_velocity(actor: Any) -> None:
    try:
        components = actor.get_components()
    except Exception:
        return
    for component in components:
        if not isinstance(component, sapien.physx.PhysxRigidDynamicComponent):
            continue
        for name in ("set_linear_velocity", "set_angular_velocity"):
            setter = getattr(component, name, None)
            if callable(setter):
                setter([0.0, 0.0, 0.0])


def _hold_reviewed_joint_targets(
    receiver: Any,
    targets: list[dict[str, Any]],
    drive: dict[str, Any],
) -> None:
    joints = list(receiver.actor.get_active_joints())
    for target in targets:
        joint = joints[int(target["joint_index"])]
        joint.set_drive_properties(
            stiffness=float(drive.get("stiffness") or 200.0),
            damping=float(drive.get("damping") or 20.0),
            force_limit=float(drive.get("force_limit") or 100.0),
        )
        if drive.get("friction") is not None:
            joint.set_friction(float(drive["friction"]))
        joint.set_drive_target(float(target["target_qpos"]))
