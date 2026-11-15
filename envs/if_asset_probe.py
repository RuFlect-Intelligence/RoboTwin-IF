from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import sapien
import transforms3d as t3d

from if_benchmark.grasp_validation.held_collision import (
    bind_held_collision, execute_held_lift,
    HELD_COLLISION_POLICY,
)
from if_benchmark.asset_probe_runtime import (
    articulated_task_hint,
    articulation_contact_sequence,
    as_contact_id_list,
    contact_groups_from_candidate,
    canonical_probe_placement,
    functional_point_count,
    switch_contact_matrix_for_use,
)
from if_benchmark.operation_placement import m4_runtime_refinement_origins
from if_benchmark.collision_geometry import (
    collision_aabb_tight,
    collision_shape_local_meshes,
    collision_support_clearance,
    frozen_support_plane_z,
)
from if_benchmark.diagnostic_video import DiagnosticVideoWriter
from if_benchmark.initial_pose.runtime import (
    apply_dynamic_mass_properties,
    apply_world_yaw,
    approved_world_yaw_deg,
    bind_loaded_actor_scale,
    validate_loaded_initial_pose,
)
from if_benchmark.initial_pose.core import file_sha256
from if_benchmark.pair_terminal_settle_worker import actor_aabb, stability_from_snapshots
from if_benchmark.schema import artifact_hash
from if_benchmark.schema.serialization import canonical_data
from if_benchmark.success_contract import (
    PICK_ACQUISITION_REQUIRED_CLEARANCE_M,
    DEFAULT_ALOHA_GRIPPER_PROFILE,
    HOLD_EE_FRAME_ARM,
    PICK_ACQUISITION_POLICY_VERSION,
    PICK_ACQUISITION_TRACKING_GUARD_M,
    SuccessContract,
    SuccessObservationError,
    axisymmetric_pose_stability,
    build_hold_reference,
    build_success_snapshot,
    ee_to_actor_transform,
    evaluate_pick_attachment,
    evaluate_m4_success,
    hold_consistency_from_relative_history,
    hold_failure_code,
    held_actor_target_to_planning_ee_pose,
    load_embodiment_profile,
    observe_actual_grippers,
    observe_sapien_active_contacts,
    summarize_hold_observation,
)
from if_benchmark.articulation_capability import articulation_success_capability
from if_benchmark.touch_capability import TOUCH_EXECUTABLE_USES
from if_benchmark.scene_validation.articulation_execution import (
    ArticulationExecutionError,
    ArticulationRuntimeCallbacks,
    CONTINUOUS_ARTICULATION_WAYPOINT_COUNT,
    ContinuousArticulationPlanCallbacks,
    ContinuousArticulationReplayCallbacks,
    articulation_acquisition_candidates,
    articulation_arm_joint_limits,
    articulation_grasp_matrix,
    classify_articulation_engagement,
    compile_continuous_articulation_path,
    continuous_articulation_qpos_targets,
    continuous_articulation_target_candidates,
    derive_articulation_bilateral_grasp_frame,
    execute_continuous_articulation_path,
    future_contact_matrix,
    guarded_approach_matrices,
    preengagement_arm_fallback_allowed,
    resolve_articulation_execution_profile,
    run_articulation_closed_loop,
    select_articulation_acquisition_plan,
)
from if_benchmark.scene_validation.articulation_attachment import (
    HeldArmDriveLeaseManager,
    SapienFixedAttachment,
    articulation_attachment_status,
    boost_arm_drive,
    create_evaluated_pick_attachment,
    physical_ee_link,
    physical_ee_world_matrix,
    restore_arm_drive,
)
from if_benchmark.grasp_validation.core import (
    EXACT_GRASP_POLICY,
    build_contact_contract,
    grasp_candidate_sort_key,
)
from if_benchmark.grasp_validation.runtime import (
    exact_grasp_actions,
    plan_exact_contact_candidates,
    refresh_collision_world,
    strip_runtime_arrays,
)
from if_benchmark.grasp_validation.execution import (
    BimanualAcquisitionRequest,
    HandoverExecutionError,
    HandoverRequest,
    HeldPoseRequest,
    PickAcquisitionRequest,
    execute_bimanual_acquisition,
    execute_held_actor_pose,
    execute_handover,
    execute_pick_acquisition,
)

from ._base_task import Base_Task
from .utils import ArmTag, create_actor, create_sapien_urdf_obj, get_face_prod


ADDITIONAL_A2O_USES = {
    "click_subject",
    "press_subject",
    "rotate_to_orientation_subject",
    "adjust_orientation_subject",
    "bimanual_grab_subject",
}
ORIENTATION_USES = {"rotate_to_orientation_subject", "adjust_orientation_subject"}
SWITCH_STATE_USES = {"switch_on", "switch_off"}
ARTICULATION_USES = {"open", "close", *SWITCH_STATE_USES}


class if_asset_probe(Base_Task):
    """Minimal canonical asset-skill validation scene.

    Milestone 4 uses this task only for asset-scope executable probes.  The
    scene intentionally contains one subject asset and no distractors.
    """

    def setup_demo(self, **kwags):
        self.probe_job = kwags.get("probe_job") or {}
        self.probe_metrics = {}
        self.probe_timings = {}
        self.probe_diagnostics = {}
        self.hold_reference: dict[str, Any] | None = None
        self.pick_attachment: SapienFixedAttachment | None = None
        self.pick_certification: dict[str, Any] | None = None
        self.execution_events: dict[str, list[dict[str, Any]]] = {
            "contacts": []
        }
        self.requested_operation_arm: ArmTag | None = None
        self.placement_selected_arm: ArmTag | None = None
        self.placement_selected_contact_id: int | None = None
        self.placement_selected_contact_ids_by_arm: dict[str, int] = {}
        self._handover_pending_reference: dict[str, Any] | None = None
        self._handover_pending_attachment: SapienFixedAttachment | None = None
        self._handover_pending_certification: dict[str, Any] | None = None
        self._diagnostic_video_writer: DiagnosticVideoWriter | None = None
        self._diagnostic_video_stage = "setup"
        self._diagnostic_previous_save_freq = None
        self._if_probe_timings = self.probe_timings
        super()._init_task_env_(**kwags)
        self.held_arm_drive_leases = HeldArmDriveLeaseManager(self.robot)

    def _diagnostic_frame(self) -> np.ndarray:
        self._update_render()
        self.cameras.update_picture()
        rgb = np.asarray(
            self.cameras.get_rgb()["head_camera"]["rgb"], dtype=np.uint8
        )
        from PIL import Image, ImageDraw

        image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(image, "RGBA")
        draw.rectangle((0, 0, image.width, 24), fill=(0, 0, 0, 190))
        draw.text(
            (5, 4),
            f"{self.asset_name}:{self.model_id}:{self.use} | {self._diagnostic_video_stage}"[:100],
            fill=(255, 230, 96, 255),
        )
        return np.asarray(image, dtype=np.uint8)

    def _capture_diagnostic_video(self, *, repeat: int = 1) -> None:
        if self._diagnostic_video_writer is None:
            return
        frame = self._diagnostic_frame()
        for _ in range(max(1, int(repeat))):
            self._diagnostic_video_writer.write(frame)

    def _take_picture(self) -> None:
        if self._diagnostic_video_writer is not None:
            self._capture_diagnostic_video()
            return
        super()._take_picture()

    def begin_diagnostic_video(self, path: Path, *, fps: int = 8) -> None:
        if self._diagnostic_video_writer is not None:
            raise RuntimeError("diagnostic_video_already_active")
        self._update_render()
        self.cameras.update_picture()
        rgb = np.asarray(self.cameras.get_rgb()["head_camera"]["rgb"])
        self._diagnostic_previous_save_freq = self.save_freq
        self.save_freq = 50
        self._diagnostic_video_stage = "initial"
        self._diagnostic_video_writer = DiagnosticVideoWriter(
            Path(path), width=int(rgb.shape[1]), height=int(rgb.shape[0]), fps=fps
        )
        self._capture_diagnostic_video(repeat=max(1, fps // 2))

    def finish_diagnostic_video(self, *, outcome: str) -> dict[str, Any]:
        writer = self._diagnostic_video_writer
        if writer is None:
            raise RuntimeError("diagnostic_video_not_active")
        self._diagnostic_video_stage = str(outcome)
        self._capture_diagnostic_video(repeat=max(1, writer.fps))
        try:
            return dict(writer.close())
        finally:
            self._diagnostic_video_writer = None
            self.save_freq = self._diagnostic_previous_save_freq
            self._diagnostic_previous_save_freq = None

    def load_actors(self):
        candidate = self.probe_job.get("candidate") or {}
        self.asset_name = str(self.probe_job.get("asset"))
        self.model_id = _model_id(self.probe_job.get("model_id"))
        self.use = str(self.probe_job.get("use"))
        requested_arm = self.probe_job.get("requested_arm")
        if self.use in ARTICULATION_USES or self.use in TOUCH_EXECUTABLE_USES:
            if requested_arm not in {"left", "right"}:
                raise RuntimeError("source_requested_arm_binding_missing")
            self.requested_operation_arm = ArmTag(str(requested_arm))
        if self.use in ARTICULATION_USES:
            assert self.requested_operation_arm is not None
            self.requested_articulation_arm = self.requested_operation_arm
        self.candidate_use = str(
            self.probe_job.get("candidate_use") or self.use
        )
        self.candidate = candidate
        self.contact_groups = _contact_groups(candidate)
        self.drive = candidate.get("articulation_drive") or {}
        self.canonical_initial_state = self.probe_job.get("canonical_initial_state")
        if not isinstance(self.canonical_initial_state, dict):
            raise RuntimeError("source_initial_pose_binding_missing")
        self.approved_pose = {
            "qpos": apply_world_yaw(
                self.canonical_initial_state["q0_wxyz"],
                approved_world_yaw_deg(self.canonical_initial_state),
            ),
            "z": float(self.canonical_initial_state["support"]["origin_z"]),
            "source": "human_approved_initial_pose",
        }
        pose = self._subject_pose()
        if self.use in ARTICULATION_USES:
            self.subject = create_sapien_urdf_obj(
                scene=self,
                pose=pose,
                modelname=self.asset_name,
                modelid=self.model_id,
                fix_root_link=True,
                mass=None,
            )
            if self.subject is None:
                raise RuntimeError("actor_load_failed")
            self._configure_switch_contact_point()
            self._apply_articulation_drive_properties()
            self._configure_articulation_initial_state()
        else:
            self.subject = create_actor(
                scene=self,
                pose=pose,
                modelname=self.asset_name,
                scale=self.canonical_initial_state["scale_xyz"],
                model_id=self.model_id,
                convex=True,
                mass=None,
            )
            if self.subject is None:
                raise RuntimeError("actor_load_failed")
            bind_loaded_actor_scale(
                self.subject, self.canonical_initial_state["scale_xyz"]
            )
        # Actor helpers may apply model_data.transform_matrix and, for task
        # environments, table_z_bias.  The approved root pose owns both q0
        # and support height, so restore the exact runtime pose explicitly.
        self.subject.actor.set_pose(
            sapien.Pose(
                [float(pose.p[0]), float(pose.p[1]), float(pose.p[2]) + float(self.table_z_bias)],
                np.asarray(pose.q, dtype=float).tolist(),
            )
        )
        candidate_mass = (
            (candidate.get("physical_defaults") or {}).get("mass")
            if (candidate.get("physical_defaults") or {}).get("mass")
            is not None
            else self.drive.get("mass")
        )
        frozen_mass = self.canonical_initial_state.get(
            "dynamic_mass_properties"
        ) or {}
        frozen_target = frozen_mass.get("target_mass_kg")
        if (candidate_mass is None) != (frozen_target is None) or (
            candidate_mass is not None
            and abs(float(candidate_mass) - float(frozen_target)) > 1e-12
        ):
            raise RuntimeError("source_dynamic_mass_binding_mismatch")
        self.probe_diagnostics["dynamic_mass_properties"] = (
            apply_dynamic_mass_properties(
                self.subject, self.canonical_initial_state
            )
        )
        self.subject.set_name(self.asset_name)
        self._refine_laptop_articulation_placement_with_ik()
        self.initial_subject_z = float(self.subject.get_pose().p[2])
        self.initial_subject_x = float(self.subject.get_pose().p[0])
        self.initial_subject_pose = deepcopy(self.subject.get_pose())
        self.initial_subject_aabb = actor_aabb(self.subject)
        self.initial_subject_collision_aabb = collision_aabb_tight(self.subject)
        self.subject_support_plane_z = frozen_support_plane_z(
            self.canonical_initial_state,
            z_bias=float(self.table_z_bias),
        )
        if self.use in ARTICULATION_USES:
            self.initial_qpos = np.array(self.subject.get_qpos(), dtype=float)
        self._validate_initial_pose_preflight()
        self._validate_contact_contract_binding()
        padding = float((candidate.get("physical_defaults") or {}).get("padding") or self.drive.get("padding") or 0.08)
        self.add_prohibit_area(self.subject, padding=padding)

    def play_once(self):
        self.probe_diagnostics["planner_pose_attempts"] = []
        self.plan_success = True
        info = self._play_once_current_pose()
        self._record_planner_pose_attempt()
        return info

    def _play_once_current_pose(self):
        if self.use == "pick_left":
            return self._pick(ArmTag("left"))
        if self.use == "pick_right":
            return self._pick(ArmTag("right"))
        if self.use == "handover":
            return self._handover()
        if self.use in ARTICULATION_USES:
            return self._articulate()
        if self.use == "click_subject":
            return self._click_subject()
        if self.use == "press_subject":
            return self._press_subject()
        if self.use in ORIENTATION_USES:
            return self._orientation_subject()
        if self.use == "bimanual_grab_subject":
            return self._bimanual_grab_subject()
        self.plan_success = False
        self.info["info"] = {"failure_reason": "unsupported_use"}
        return self.info

    def check_success(self):
        snapshot = self._build_success_snapshot()
        if self.use in {"pick_left", "pick_right"}:
            name = "pick_success"
            arguments = {
                "subject": "subject",
                "requested_arm": self.use.removeprefix("pick_"),
            }
        elif self.use == "handover":
            name = "air_handover_success"
            arguments = {
                "subject": "subject",
                "source_arm": str(getattr(self, "handover_source", ArmTag("left"))),
                "target_arm": str(getattr(self, "handover_target", ArmTag("right"))),
            }
        elif self.use in ARTICULATION_USES:
            name = "articulation_state_success"
            arguments = {
                "subject": "subject",
                "state": {
                    "switch_on": "on",
                    "switch_off": "off",
                }.get(self.use, self.use),
            }
        elif self.use in {"click_subject", "press_subject"}:
            name = "click_press_success"
            arguments = {
                "subject": "subject",
                "arm": str(
                    getattr(
                        self,
                        "extended_arm",
                        self.requested_operation_arm,
                    )
                ),
                "contact_id": int(
                    getattr(
                        self,
                        "extended_touch_contact_id",
                        self._single_contact_id(default=0),
                    )
                ),
            }
        elif self.use in ORIENTATION_USES:
            name = "orientation_success"
            arguments = {
                "subject": "subject",
                "target_quat_wxyz": list(
                    getattr(
                        self,
                        "extended_target_qpos",
                        self._orientation_target_qpos(),
                    )
                ),
            }
        elif self.use == "bimanual_grab_subject":
            name = "bimanual_success"
            arguments = {"subject": "subject"}
        else:
            return False
        result = evaluate_m4_success(
            name, arguments, snapshot, SuccessContract.load()
        )
        self.success_checker_result = result
        self.probe_metrics = {
            **result["metrics"],
            "predicate_result": result,
            "success_snapshot_hash": snapshot["success_snapshot_hash"],
            **SuccessContract.load().identity,
        }
        return result["status"] == "passed"

    def _build_success_snapshot(
        self, *, pick_certification: bool = False
    ) -> dict[str, Any]:
        """Collect the M4 terminal SuccessSnapshot without issuing an action."""

        contract = SuccessContract.load()
        profile = load_embodiment_profile(DEFAULT_ALOHA_GRIPPER_PROFILE)
        endpoint_by_name = {
            str(self.subject.get_name()): "subject",
            str(self.table.get_name()): "support:table",
        }
        if self.use in ARTICULATION_USES:
            for link in self.subject.actor.get_links():
                endpoint_by_name[str(link.get_name())] = "subject"
        for arm, arm_profile in profile["arms"].items():
            for row in arm_profile.get("fingers") or []:
                endpoint_by_name[str(row["link"])] = f"gripper:{arm}"
            for name in arm_profile.get("palm_wrist_links") or []:
                endpoint_by_name[str(name)] = f"gripper:{arm}"
        if self.use in ARTICULATION_USES:
            for link in self.subject.actor.get_links():
                endpoint_by_name[str(link.get_name())] = "subject"

        stability_config = contract.section("stability")
        is_pick = bool(
            pick_certification
            or self.use in {"pick_left", "pick_right"}
        )
        settle_steps = (
            0 if is_pick else int(stability_config["minimum_settle_steps"])
        )
        observation_steps = (
            int(contract.section("hold")["pick_certification_observation_steps"])
            if is_pick
            else int(stability_config["observation_window_steps"])
        )
        for _ in range(settle_steps):
            self.scene.step()
        history: list[list[dict[str, Any]]] = []
        relative_history: dict[str, list[np.ndarray]] = {
            "left": [],
            "right": [],
        }
        consecutive_inner = {"left": 0, "right": 0}
        maximum_inner = {"left": 0, "right": 0}
        last_contacts: list[dict[str, Any]] = []
        last_grippers: dict[str, Any] = {}
        for _ in range(observation_steps):
            self.scene.step()
            tracked_pose = self.subject.get_pose()
            if self.use in ARTICULATION_USES:
                joint_index = int(self.drive.get("joint_index", 0))
                joints = list(self.subject.actor.get_active_joints())
                if joint_index < len(joints):
                    tracked_pose = joints[joint_index].child_link.get_pose()
            history.append(
                [{"p": tracked_pose.p.tolist(), "q": tracked_pose.q.tolist()}]
            )
            last_grippers = observe_actual_grippers(
                self.robot, embodiment_profile=profile
            )
            last_contacts = observe_sapien_active_contacts(
                self.scene,
                endpoint_by_body_name=endpoint_by_name,
                embodiment_profile=profile,
                grippers=last_grippers,
                contract=contract,
            )
            for arm in ("left", "right"):
                inner = {
                    row.get("finger")
                    for row in last_contacts
                    if {row.get("entity_a"), row.get("entity_b")}
                    == {"subject", f"gripper:{arm}"}
                    and row.get("region") == "inner_pad"
                }
                consecutive_inner[arm] = (
                    consecutive_inner[arm] + 1 if len(inner) >= 2 else 0
                )
                maximum_inner[arm] = max(
                    maximum_inner[arm], consecutive_inner[arm]
                )
                # Must match the immutable hold reference: the rigid physical
                # EE link used by the D6, not the TCP or terminal-joint planning
                # frame.  The latter changes relative to the child link when
                # wrist joint 6 rotates and invents false carry drift.
                relative_history[arm].append(
                    ee_to_actor_transform(
                        self._probe_hold_ee_matrix(ArmTag(arm)),
                        self.subject.get_pose().to_transformation_matrix(),
                    )
                )

        stability = stability_from_snapshots(
            history, rotation_invariant=is_pick
        )
        symmetry_axis = self._probe_symmetry_axis() if is_pick else None
        continuity_minimum = int(
            profile["classification"]["continuous_contact_min_steps"]
        )
        for arm in ("left", "right"):
            state = last_grippers[arm]
            if maximum_inner[arm] >= continuity_minimum:
                state["closure_evidence"] = {
                    "subject": "subject",
                    "valid": True,
                    "path": "persistent_inner_pad",
                    "actual_width_m": state["actual_width_m"],
                    "continuous_steps": maximum_inner[arm],
                }
            relative = relative_history[arm]
            reference = (
                self.hold_reference
                if is_pick
                and isinstance(self.hold_reference, Mapping)
                and self.hold_reference.get("arm") == arm
                else None
            )
            if not relative:
                continue
            state["hold_consistency"] = hold_consistency_from_relative_history(
                relative,
                subject="subject",
                ee_frame=HOLD_EE_FRAME_ARM,
                reference=reference,
                symmetry_axis=symmetry_axis if is_pick else None,
            )
            if (
                is_pick
                and reference is not None
                and arm == str(reference.get("arm") or "")
                and isinstance(self.pick_attachment, SapienFixedAttachment)
            ):
                attachment = self._observe_probe_pick_attachment()
                certified = reference.get("certified_closure_evidence")
                if attachment["passed"] and isinstance(certified, Mapping):
                    state["closure_evidence"] = {
                        **deepcopy(dict(certified)),
                        "path": "post_close_fixed_attachment",
                        "actual_width_m": float(state["actual_width_m"]),
                    }
                    state["post_close_attachment"] = {
                        **deepcopy(attachment),
                        "subject": "subject",
                        "certified_close": True,
                    }

        terminal_pose = self.subject.get_pose()
        terminal_entity = {
            "pose": {
                "position": terminal_pose.p.tolist(),
                "quat_wxyz": terminal_pose.q.tolist(),
            },
            "aabb": collision_aabb_tight(self.subject),
            "support_plane_z": float(self.subject_support_plane_z),
        }
        initial_entity = {
            "pose": {
                "position": self.initial_subject_pose.p.tolist(),
                "quat_wxyz": self.initial_subject_pose.q.tolist(),
            },
            "aabb": deepcopy(self.initial_subject_collision_aabb),
            "support_plane_z": float(self.subject_support_plane_z),
        }
        initial: dict[str, Any] = {"entities": {"subject": initial_entity}}
        terminal: dict[str, Any] = {
            "entities": {"subject": terminal_entity},
            "grippers": last_grippers,
            "active_contacts": last_contacts,
        }
        profile_hashes = {"embodiment": profile["profile_hash"]}
        if self.use in ARTICULATION_USES:
            joint_index = int(self.drive.get("joint_index", 0))
            limits = np.asarray(self.subject.get_qlimits(), dtype=float)[joint_index]
            drive_profile = deepcopy(
                getattr(self, "articulation_state_profile", self.drive)
            )
            drive_profile.setdefault(
                "positive_use",
                (
                    contract.section("articulation")["default_positive_use"]
                    if self.use in SWITCH_STATE_USES
                    else "open"
                ),
            )
            drive_hash = artifact_hash(drive_profile)
            initial["articulations"] = {
                "subject": {"qpos": float(self.initial_qpos[joint_index])}
            }
            terminal["articulations"] = {
                "subject": {
                    "qpos": float(self.subject.get_qpos()[joint_index]),
                    "lower": float(limits[0]),
                    "upper": float(limits[1]),
                    "state_profile": drive_profile,
                    "state_profile_hash": drive_hash,
                }
            }
            profile_hashes["articulation"] = drive_hash
        source_hashes = {"probe_job": artifact_hash(self.probe_job)}
        if is_pick:
            source_hashes["pick_acquisition_policy"] = artifact_hash(
                {
                    "version": PICK_ACQUISITION_POLICY_VERSION,
                    "tracking_guard_m": PICK_ACQUISITION_TRACKING_GUARD_M,
                }
            )
        return build_success_snapshot(
            initial=initial,
            terminal=terminal,
            stability={
                "entities": {"subject": stability},
                "whole_scene": {"stable": bool(stability.get("stable"))},
            },
            execution_events=deepcopy(self.execution_events),
            source_hashes=source_hashes,
            contract=contract,
            profiles={"embodiment": profile},
            profile_hashes=profile_hashes,
        )

    def _subject_pose(self):
        pose_defaults = self.approved_pose
        qpos = pose_defaults["qpos"]
        z = pose_defaults["z"]
        if self.use == "handover":
            raise RuntimeError("m4_capability_explicitly_not_evaluated")
        placement = canonical_probe_placement(
            use=self.use,
            candidate=self.candidate,
            model_data=_load_model_data(self.candidate),
            canonical_initial_state=self.canonical_initial_state,
            relative_yaw_deg=approved_world_yaw_deg(
                self.canonical_initial_state
            ),
        )
        if placement.get("status") != "passed":
            raise RuntimeError(
                str(placement.get("failure_code") or "m4_placement_unsat")
            )
        origin_xy = [float(value) for value in placement["origin_xy_m"]]
        self.canonical_placement_witness = deepcopy(placement)
        self.probe_diagnostics["canonical_placement_witness"] = deepcopy(
            placement
        )
        selected_arm = placement.get("selected_arm")
        selected_contact_ids = placement.get("selected_contact_ids_by_arm") or {}
        self.placement_selected_contact_ids_by_arm = {
            str(arm): int(contact_id)
            for arm, contact_id in selected_contact_ids.items()
        }
        if self.use in TOUCH_EXECUTABLE_USES:
            if self.requested_operation_arm is None:
                raise RuntimeError("source_requested_arm_binding_missing")
            requested_arm = str(self.requested_operation_arm)
            if requested_arm not in self.placement_selected_contact_ids_by_arm:
                raise RuntimeError("source_requested_arm_contact_missing")
            self.placement_selected_arm = self.requested_operation_arm
            self.placement_selected_contact_id = int(
                self.placement_selected_contact_ids_by_arm[requested_arm]
            )
        if selected_arm in {"left", "right"}:
            self.placement_selected_arm = ArmTag(str(selected_arm))
            self.placement_selected_contact_id = int(
                self.placement_selected_contact_ids_by_arm[str(selected_arm)]
            )
        source = (
            f"{pose_defaults['source']}; canonical_placement:"
            f"{placement['policy_version']}"
        )
        self.probe_diagnostics["root_pose"] = {
            "qpos": qpos,
            "z": z,
            "source": source,
            "approved_pose_count": 1,
            "xlim": [origin_xy[0], origin_xy[0]],
            "ylim": [origin_xy[1], origin_xy[1]],
            "rotate_rand": False,
            "selected_arm": selected_arm,
            "canonical_placement_witness_hash": placement["witness_hash"],
            "operation_anchor_placement": deepcopy(placement),
        }
        return sapien.Pose(
            [origin_xy[0], origin_xy[1], float(z)],
            list(qpos),
        )

    def _laptop_articulation_pregrasp_targets(
        self,
        execution_profile: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        """Build the exact acquisition Pre-Grasp targets at the current root."""

        if execution_profile.get("engagement_mode") != "grasp_follow":
            return []
        points = (self.subject.config or {}).get("contact_points", [])
        moving_link = self._probe_articulation_joint_motion()["child"]
        moving_link_matrix = np.asarray(
            moving_link.get_pose().to_transformation_matrix(), dtype=float
        )
        moving_link_meshes = collision_shape_local_meshes(moving_link)
        segments = list(execution_profile.get("segments") or [])
        if not segments:
            return []
        # The closed-loop executor acquires segment zero from the initial
        # state.  A Contact belonging only to a later segment must not make
        # this initial Pre-Grasp certificate pass.
        contact_ids = list(
            dict.fromkeys(
                int(value)
                for value in segments[0].get("contact_ids") or []
            )
        )
        geometry_frames: dict[tuple[int, str], dict[str, Any]] = {}
        targets: list[dict[str, Any]] = []
        for raw_contact_id, pre_grasp_m, roll_rad, orientation_variant in (
            articulation_acquisition_candidates(
                execution_profile, contact_ids
            )
        ):
            contact_id = int(raw_contact_id)
            if contact_id < 0 or contact_id >= len(points):
                continue
            expected_base = str(
                execution_profile.get("contact_frame_base")
                or execution_profile["moving_link"]
            )
            if str(points[contact_id].get("base") or "") != expected_base:
                continue
            matrix = self._probe_articulation_contact_matrix(
                contact_id, execution_profile
            )
            geometry_key = (contact_id, str(orientation_variant))
            try:
                if geometry_key not in geometry_frames:
                    geometry_frames[geometry_key] = (
                        derive_articulation_bilateral_grasp_frame(
                            matrix,
                            moving_link_matrix,
                            moving_link_meshes,
                            minimum_closing_width_m=float(
                                execution_profile["minimum_closing_width_m"]
                            ),
                            maximum_closing_width_m=float(
                                execution_profile["maximum_closing_width_m"]
                            ),
                            maximum_contact_shape_distance_m=float(
                                execution_profile[
                                    "maximum_contact_shape_distance_m"
                                ]
                            ),
                            local_surface_patch_radius_m=float(
                                execution_profile[
                                    "local_surface_patch_radius_m"
                                ]
                            ),
                            minimum_opposed_normal_cosine=float(
                                execution_profile[
                                    "minimum_opposed_normal_cosine"
                                ]
                            ),
                            orientation_variant=str(orientation_variant),
                        )
                    )
            except ArticulationExecutionError:
                continue
            pose = self._probe_articulation_grasp_pose(
                matrix,
                pre_grasp_m=float(pre_grasp_m),
                roll_rad=float(roll_rad),
                grasp_rotation_world=geometry_frames[geometry_key][
                    "rotation_world"
                ],
            )
            targets.append(
                {
                    "pose": pose,
                    "contact_id": contact_id,
                    "pre_grasp_m": float(pre_grasp_m),
                    "roll_rad": float(roll_rad),
                    "orientation_variant": str(orientation_variant),
                }
            )
        return targets

    def _refine_laptop_articulation_placement_with_ik(self) -> None:
        """Choose one shared laptop root with real left/right Pre-Grasp IK."""

        if self.asset_name != "015_laptop" or self.use not in {"open", "close"}:
            return
        origins = m4_runtime_refinement_origins(
            self.canonical_placement_witness
        )
        if not origins:
            return
        try:
            execution_profile = resolve_articulation_execution_profile(
                self.asset_name,
                self.model_id,
                self.use,
                contract=SuccessContract.load(),
            )
        except ArticulationExecutionError as exc:
            self.probe_diagnostics["laptop_pre_grasp_ik_placement"] = {
                "status": "skipped",
                "failure_code": str(exc),
            }
            return

        initial_pose = deepcopy(self.subject.actor.get_pose())
        evaluations: list[dict[str, Any]] = []
        selected: dict[str, Any] | None = None
        selected_arm_rows: dict[str, Any] = {}
        for origin in origins:
            xy = [float(value) for value in origin["origin_xy_m"]]
            self.subject.actor.set_pose(
                sapien.Pose(
                    [xy[0], xy[1], float(initial_pose.p[2])],
                    np.asarray(initial_pose.q, dtype=float).tolist(),
                )
            )
            try:
                targets = self._laptop_articulation_pregrasp_targets(
                    execution_profile
                )
            except (ArticulationExecutionError, RuntimeError, ValueError) as exc:
                evaluations.append(
                    {
                        "origin_xy_m": xy,
                        "translation_from_static_origin_m": float(
                            origin["translation_from_static_origin_m"]
                        ),
                        "target_count": 0,
                        "arms": {},
                        "status": "target_geometry_unavailable",
                        "failure_code": str(exc),
                    }
                )
                continue
            evaluation = {
                "origin_xy_m": xy,
                "translation_from_static_origin_m": float(
                    origin["translation_from_static_origin_m"]
                ),
                "target_count": len(targets),
                "arms": {},
            }
            both_arms_passed = bool(targets)
            arm_rows: dict[str, Any] = {}
            poses = [row["pose"] for row in targets]
            for arm in ("left", "right"):
                try:
                    ik = self.robot.solve_ik_batch(arm, poses) if poses else {}
                except RuntimeError as exc:
                    evaluation["arms"][arm] = {
                        "success_count": 0,
                        "planner_error": str(exc),
                    }
                    both_arms_passed = False
                    continue
                raw_statuses = ik.get("status")
                statuses = (
                    list(raw_statuses) if raw_statuses is not None else []
                )
                passed_indices = [
                    index
                    for index, status in enumerate(statuses)
                    if str(status) == "Success"
                ]
                raw_position_errors = ik.get("position_error")
                position_errors = [
                    float(value)
                    for value in (
                        raw_position_errors
                        if raw_position_errors is not None
                        else []
                    )
                ]
                raw_rotation_errors = ik.get("rotation_error")
                rotation_errors = [
                    float(value)
                    for value in (
                        raw_rotation_errors
                        if raw_rotation_errors is not None
                        else []
                    )
                ]
                arm_row: dict[str, Any] = {
                    "success_count": len(passed_indices),
                    "minimum_position_error_m": (
                        min(position_errors) if position_errors else None
                    ),
                    "minimum_rotation_error_rad": (
                        min(rotation_errors) if rotation_errors else None
                    ),
                }
                if passed_indices:
                    target = targets[passed_indices[0]]
                    arm_row["selected_target"] = {
                        key: target[key]
                        for key in (
                            "contact_id",
                            "pre_grasp_m",
                            "roll_rad",
                            "orientation_variant",
                        )
                    }
                else:
                    both_arms_passed = False
                evaluation["arms"][arm] = arm_row
                arm_rows[arm] = arm_row
            evaluation["status"] = (
                "passed" if both_arms_passed else "ik_unreachable"
            )
            evaluations.append(evaluation)
            if both_arms_passed:
                selected = origin
                selected_arm_rows = arm_rows
                break

        diagnostic = canonical_data(
            {
                "status": "passed" if selected is not None else "unsat",
                "method": "shared_laptop_pre_grasp_6dof_ik_v1",
                "candidate_count": len(origins),
                "evaluated_candidate_count": len(evaluations),
                "required_arms": ["left", "right"],
                "evaluations": evaluations,
            }
        )
        if selected is None:
            self.subject.actor.set_pose(initial_pose)
            self.probe_diagnostics[
                "laptop_pre_grasp_ik_placement"
            ] = diagnostic
            return

        selected_xy = [float(value) for value in selected["origin_xy_m"]]
        selected_arm_targets = {
            arm: deepcopy(row["selected_target"])
            for arm, row in selected_arm_rows.items()
        }
        diagnostic["selected_origin_xy_m"] = selected_xy
        diagnostic["selected_arm_targets"] = selected_arm_targets
        placement_payload = {
            key: deepcopy(value)
            for key, value in self.canonical_placement_witness.items()
            if key != "witness_hash"
        }
        placement_payload["origin_xy_m"] = selected_xy
        placement_payload["candidate_score"] = list(
            selected["static_score"]
        )
        placement_payload["runtime_ik_refinement"] = {
            "method": diagnostic["method"],
            "required_arms": diagnostic["required_arms"],
            "selected_origin_xy_m": selected_xy,
            # Keep the publication witness independent of floating-point IK
            # residuals and solver batching.  The selected authored candidate
            # is the durable evidence; complete residuals remain diagnostic.
            "selected_arm_targets": selected_arm_targets,
        }
        placement_payload = canonical_data(placement_payload)
        placement = canonical_data(
            {
                **placement_payload,
                "witness_hash": artifact_hash(placement_payload),
            }
        )
        self.canonical_placement_witness = placement
        self.probe_diagnostics["canonical_placement_witness"] = deepcopy(
            placement
        )
        root = self.probe_diagnostics.get("root_pose") or {}
        root.update(
            {
                "xlim": [selected_xy[0], selected_xy[0]],
                "ylim": [selected_xy[1], selected_xy[1]],
                "source": (
                    f"{root.get('source')}; "
                    "runtime_refinement:shared_laptop_pre_grasp_6dof_ik_v1"
                ),
                "canonical_placement_witness_hash": placement[
                    "witness_hash"
                ],
                "operation_anchor_placement": deepcopy(placement),
            }
        )
        self.probe_diagnostics["root_pose"] = root
        self.probe_diagnostics[
            "laptop_pre_grasp_ik_placement"
        ] = diagnostic

    def _validate_initial_pose_preflight(self) -> None:
        actual_pose = self.subject.get_pose()
        scale = (self.subject.config or {}).get("scale") or [1.0, 1.0, 1.0]
        if isinstance(scale, (int, float)):
            scale = [float(scale)] * 3
        elif len(scale) == 1:
            scale = list(scale) * 3
        joint_qpos = (
            np.array(self.subject.get_qpos(), dtype=float).tolist()
            if self.use in ARTICULATION_USES
            else []
        )
        expected_origin_z = float(self.canonical_initial_state["support"]["origin_z"])
        # Base_Task adds table_z_bias when actor helpers preprocess the pose.
        canonical_for_runtime = deepcopy(self.canonical_initial_state)
        canonical_for_runtime["support"] = {
            **canonical_for_runtime["support"],
            "origin_z": expected_origin_z + float(self.table_z_bias),
        }
        diagnostics = validate_loaded_initial_pose(
            canonical_state=canonical_for_runtime,
            actual_q_wxyz=np.asarray(actual_pose.q, dtype=float).tolist(),
            actual_joint_qpos=joint_qpos,
            actual_scale_xyz=list(scale),
            actual_origin_z=float(actual_pose.p[2]),
            expected_world_yaw_deg=approved_world_yaw_deg(
                self.canonical_initial_state
            ),
        )
        diagnostics["initial_pose_binding"] = self.probe_job.get("initial_pose_binding")
        self.probe_diagnostics["initial_pose_preflight"] = diagnostics

    def _validate_contact_contract_binding(self) -> None:
        profiles = {
            "pick_left": ("pick_subject", "single_arm_pick"),
            "pick_right": ("pick_subject", "single_arm_pick"),
            "handover": ("handover_subject", "handover"),
            "bimanual_grab_subject": (
                "bimanual_grab_subject",
                "bimanual",
            ),
        }
        expected = profiles.get(self.use)
        if expected is None:
            return
        binding = self.probe_job.get("contact_contract_binding")
        if not isinstance(binding, dict):
            raise RuntimeError("source_contact_contract_binding_missing")
        candidate_use, use_profile = expected
        if self.candidate_use != candidate_use:
            raise RuntimeError(
                "source_contact_contract_candidate_use_mismatch:"
                f"{self.candidate_use}!={candidate_use}"
            )
        metadata_path_value = (
            ((self.candidate.get("evidence") or {}).get("metadata") or {}).get(
                "path"
            )
        )
        if not metadata_path_value:
            raise RuntimeError("source_contact_metadata_path_missing")
        metadata_path = Path(str(metadata_path_value))
        metadata_sha256 = file_sha256(metadata_path)
        contact_key = self.subject.POINTS["contact"]
        matrices = (self.subject.config or {}).get(contact_key)
        if not isinstance(matrices, list) or not matrices:
            raise RuntimeError("source_contact_matrices_missing")
        contract = build_contact_contract(
            asset=self.asset_name,
            model_id=str(self.model_id),
            use=candidate_use,
            contact_matrices=matrices,
            contact_groups=self.contact_groups,
            use_profile=use_profile,
        )
        if contract["policy_version"] != binding.get("policy_version"):
            raise RuntimeError("source_contact_policy_version_mismatch")
        if contract["contact_contract_hash"] != binding.get(
            "contact_contract_hash"
        ):
            raise RuntimeError("source_contact_contract_hash_mismatch")
        if contract["contact_projection_hash"] != binding.get(
            "contact_projection_hash"
        ):
            raise RuntimeError("source_contact_projection_hash_mismatch")
        if len(matrices) != int(binding.get("contact_count", -1)):
            raise RuntimeError("source_contact_contract_count_mismatch")
        self.probe_diagnostics["contact_contract_preflight"] = {
            "policy_version": contract["policy_version"],
            "contact_contract_id": contract["contact_contract_id"],
            "contact_contract_hash": contract["contact_contract_hash"],
            "contact_projection_hash": contract[
                "contact_projection_hash"
            ],
            "candidate_use": candidate_use,
            "use_profile": use_profile,
            "contact_count": len(matrices),
            "contact_groups": deepcopy(self.contact_groups),
            "contact_metadata_path": str(metadata_path),
            "contact_metadata_sha256": metadata_sha256,
        }

    def check_stable(self):
        is_stable, unstable_list = super().check_stable()
        attempts = [
            {
                **self.probe_diagnostics.get("root_pose", {}),
                "stable": bool(is_stable),
                "unstable_objects": list(unstable_list),
            }
        ]
        self.probe_diagnostics["root_pose_attempts"] = attempts
        if is_stable:
            settled_pose = deepcopy(self.subject.get_pose())
            self.initial_subject_z = float(settled_pose.p[2])
            self.initial_subject_x = float(settled_pose.p[0])
            self.initial_subject_pose = settled_pose
            self.initial_subject_aabb = actor_aabb(self.subject)
            self.initial_subject_collision_aabb = collision_aabb_tight(
                self.subject
            )
        return is_stable, unstable_list

    def _record_planner_pose_attempt(self) -> None:
        self.probe_diagnostics.setdefault("planner_pose_attempts", []).append(
            {
                **self.probe_diagnostics.get("root_pose", {}),
                "plan_success": bool(getattr(self, "plan_success", False)),
            }
        )

    def _probe_ee_matrix(self, arm: ArmTag) -> np.ndarray:
        pose = np.asarray(self.get_arm_pose(arm), dtype=float)
        return np.asarray(
            sapien.Pose(pose[:3], pose[3:]).to_transformation_matrix(),
            dtype=float,
        )

    def _probe_hold_ee_matrix(self, arm: ArmTag) -> np.ndarray:
        return physical_ee_world_matrix(self.robot, str(arm))

    def _probe_active_contacts(
        self,
        profile: Mapping[str, Any],
        grippers: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        endpoint_by_name = {
            str(self.subject.get_name()): "subject",
            str(self.table.get_name()): "support:table",
        }
        raw_subject = getattr(self.subject, "actor", None)
        get_links = getattr(raw_subject, "get_links", None)
        if callable(get_links):
            for link in get_links():
                endpoint_by_name[str(link.get_name())] = "subject"
        for arm, arm_profile in profile["arms"].items():
            for row in arm_profile.get("fingers") or []:
                endpoint_by_name[str(row["link"])] = f"gripper:{arm}"
            for name in arm_profile.get("palm_wrist_links") or []:
                endpoint_by_name[str(name)] = f"gripper:{arm}"
        return observe_sapien_active_contacts(
            self.scene,
            endpoint_by_body_name=endpoint_by_name,
            embodiment_profile=profile,
            grippers=grippers,
            contract=SuccessContract.load(),
        )

    def _freeze_probe_hold_reference(self, arm: ArmTag) -> None:
        profile = load_embodiment_profile(DEFAULT_ALOHA_GRIPPER_PROFILE)
        gripper = observe_actual_grippers(
            self.robot, embodiment_profile=profile
        )[str(arm)]
        self.hold_reference = build_hold_reference(
            subject="subject",
            arm=str(arm),
            actor_matrix=self.subject.get_pose().to_transformation_matrix(),
            ee_matrix=self._probe_hold_ee_matrix(arm),
            planning_ee_matrix=self._probe_ee_matrix(arm),
            actual_width_m=float(gripper["actual_width_m"]),
            ee_frame=HOLD_EE_FRAME_ARM,
        )

    def _prepare_probe_collision(self, arm: ArmTag) -> None:
        if not isinstance(self.hold_reference, dict):
            raise RuntimeError("immutable_hold_reference_missing")
        arm_name = str(arm)
        binding = bind_held_collision(self, self.subject, arm_name, name="held:subject")
        self.hold_reference.update(
            collision_world_hash=binding["world"][arm_name]["world_hash"],
            attachment_hash=binding["attachment"]["attachment_hash"],
            carry_collision_binding=deepcopy(binding["attachment"]),
            held_collision_policy=HELD_COLLISION_POLICY,
        )

    def _create_probe_pick_attachment(self, arm: ArmTag) -> dict[str, Any]:
        """Weld a verified closed grasp to the physical EE without snapping."""

        self._remove_probe_pick_attachment()
        attachment, result = create_evaluated_pick_attachment(
            self,
            self.subject,
            arm=str(arm),
        )
        if attachment is not None:
            self.pick_attachment = attachment
        return result

    def _observe_probe_pick_attachment(self) -> dict[str, Any]:
        if not isinstance(self.pick_attachment, SapienFixedAttachment):
            return evaluate_pick_attachment({}, contract=SuccessContract.load())
        return evaluate_pick_attachment(
            self.pick_attachment.metrics(), contract=SuccessContract.load()
        )

    def _remove_probe_pick_attachment(self) -> None:
        if isinstance(self.pick_attachment, SapienFixedAttachment):
            self.pick_attachment.remove()
        self.pick_attachment = None

    def _move_probe_held(
        self, arm: ArmTag, motion: tuple[Any, list[Any]]
    ) -> None:
        if not isinstance(self.hold_reference, dict):
            raise RuntimeError("immutable_hold_reference_missing")
        arm_name = str(arm)
        world = self.robot.update_world_obstacles([])
        world_hash = str((world.get(arm_name) or {}).get("world_hash") or "")
        self.hold_reference["collision_world_hash"] = world_hash
        for action in motion[1]:
            action.args.update(
                {
                    "expected_world_hash": world_hash,
                    "expected_attachment_hash": self.hold_reference[
                        "attachment_hash"
                    ],
                    "m15_exact_table_validation": True,
                }
            )
        self.move(motion)

    def _probe_hold_observation(
        self,
        arm: ArmTag,
        *,
        steps: int,
        reference: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        resolved_reference = reference or self.hold_reference
        if not isinstance(resolved_reference, Mapping):
            raise RuntimeError("immutable_hold_reference_missing")
        profile = load_embodiment_profile(DEFAULT_ALOHA_GRIPPER_PROFILE)
        samples = []
        history = []
        for _ in range(int(steps)):
            self.scene.step()
            grippers = observe_actual_grippers(
                self.robot, embodiment_profile=profile
            )
            contacts = self._probe_active_contacts(profile, grippers)
            pose = self.subject.get_pose()
            samples.append(
                {
                    "gripper": grippers[str(arm)],
                    "active_contacts": contacts,
                    "actor_matrix": pose.to_transformation_matrix(),
                    "ee_matrix": self._probe_hold_ee_matrix(arm),
                    "ee_frame": HOLD_EE_FRAME_ARM,
                }
            )
            history.append([{"p": pose.p.tolist(), "q": pose.q.tolist()}])
        symmetry_axis = self._probe_symmetry_axis()
        value = summarize_hold_observation(
            samples,
            reference=resolved_reference,
            subject="subject",
            arm=str(arm),
            embodiment_profile=profile,
            symmetry_axis=symmetry_axis,
        )
        stability = stability_from_snapshots(history)
        if symmetry_axis is not None:
            policy = SuccessContract.load().section("stability")
            stability = axisymmetric_pose_stability(
                [frame[0] for frame in history],
                symmetry_axis=symmetry_axis,
                translation_limit_m=float(policy["translation_drift_m"]),
                rotation_limit_deg=float(policy["rotation_drift_deg"]),
            )
        value["stability"] = stability
        return value

    def _probe_symmetry_axis(self) -> list[float] | None:
        configured_axes = SuccessContract.load().section("hold").get(
            "reviewed_symmetry_axes", {}
        )
        axis_name = (
            str(configured_axes.get(self.asset_name))
            if isinstance(configured_axes, Mapping)
            and self.asset_name in configured_axes
            else None
        )
        return {
            "x": [1.0, 0.0, 0.0],
            "y": [0.0, 1.0, 0.0],
            "z": [0.0, 0.0, 1.0],
        }.get(axis_name)

    def _probe_hold_failure(self, summary: Mapping[str, Any]) -> str | None:
        return hold_failure_code(
            summary, phase="pick", contract=SuccessContract.load()
        )

    def _unified_subject_actor(self, subject_key: str) -> Any:
        if subject_key != "subject":
            raise KeyError(subject_key)
        return self.subject

    def _unified_contact_count(self, subject_key: str) -> int:
        actor = self._unified_subject_actor(subject_key)
        return len(
            (actor.config or {}).get(actor.POINTS["contact"], [])
        )

    def _unified_configure_pick_dynamics(
        self,
        subject_key: str,
        *,
        linear_damping: float,
        angular_damping: float,
    ) -> dict[str, Any]:
        actor = self._unified_subject_actor(subject_key)
        count = 0
        for component in actor.actor.get_components():
            if isinstance(
                component, sapien.physx.PhysxRigidDynamicComponent
            ):
                component.set_linear_damping(float(linear_damping))
                component.set_angular_damping(float(angular_damping))
                count += 1
        result = {
            "linear_damping": float(linear_damping),
            "angular_damping": float(angular_damping),
            "dynamic_component_count": count,
        }
        self.probe_diagnostics["pick_dynamics"] = deepcopy(result)
        return result

    def _unified_reset_planning(self) -> None:
        self.plan_success = True

    def _unified_record_grasp_plan(self, planning: Mapping[str, Any]) -> None:
        self.probe_diagnostics.setdefault("exact_grasp_plans", []).append(
            deepcopy(dict(planning))
        )

    def _unified_record_grasp_attempt(
        self, attempt: Mapping[str, Any]
    ) -> None:
        self.probe_diagnostics.setdefault(
            "exact_pick_physical_attempts", []
        ).append(deepcopy(dict(attempt)))

    def _unified_freeze_hold_reference(
        self, subject_key: str, arm: str
    ) -> None:
        del subject_key
        self.hold_reference = None
        self._freeze_probe_hold_reference(ArmTag(arm))
        if self.use == "bimanual_grab_subject":
            references = getattr(self, "bimanual_hold_references", None)
            if not isinstance(references, dict):
                references = {}
                self.bimanual_hold_references = references
            references[str(arm)] = deepcopy(self.hold_reference)

    def _unified_observe_hold(
        self, subject_key: str, arm: str, *, phase: str
    ) -> dict[str, Any]:
        del subject_key, phase
        continuity = int(
            load_embodiment_profile(DEFAULT_ALOHA_GRIPPER_PROFILE)[
                "classification"
            ]["continuous_contact_min_steps"]
        )
        reference = None
        if self.use == "bimanual_grab_subject":
            reference = (
                getattr(self, "bimanual_hold_references", {}) or {}
            ).get(str(arm))
        return self._probe_hold_observation(
            ArmTag(arm), steps=continuity, reference=reference
        )

    def _unified_hold_failure(
        self, summary: Mapping[str, Any], *, phase: str
    ) -> str | None:
        if isinstance(self.pick_attachment, SapienFixedAttachment):
            attachment = self._observe_probe_pick_attachment()
            if not attachment["passed"]:
                return str(attachment["failure_code"])
        return hold_failure_code(
            summary, phase=phase, contract=SuccessContract.load()
        )

    def _unified_store_certified_closure(
        self, arm: str, evidence: Mapping[str, Any]
    ) -> None:
        reference = self.hold_reference
        if self.use == "bimanual_grab_subject":
            reference = (
                getattr(self, "bimanual_hold_references", {}) or {}
            ).get(str(arm))
        if not isinstance(reference, dict):
            raise RuntimeError("immutable_hold_reference_missing")
        if reference.get("arm") != str(arm):
            raise RuntimeError("immutable_hold_reference_arm_mismatch")
        reference["certified_closure_evidence"] = deepcopy(
            dict(evidence)
        )

    def _unified_create_attachment(
        self, subject_key: str, arm: str
    ) -> dict[str, Any]:
        del subject_key
        return self._create_probe_pick_attachment(ArmTag(arm))

    def _unified_prepare_held_collision(
        self, subject_key: str, arm: str
    ) -> None:
        del subject_key
        self._prepare_probe_collision(ArmTag(arm))

    def _unified_support_clearance(
        self,
        subject_key: str,
        *,
        support_plane_z_m: float | None = None,
    ) -> float:
        del subject_key
        frozen_support = float(self.subject_support_plane_z)
        if (
            support_plane_z_m is not None
            and not math.isclose(
                float(support_plane_z_m),
                frozen_support,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            raise RuntimeError("pick_support_plane_contract_mismatch")
        return collision_support_clearance(self.subject, frozen_support)

    def _unified_move_held_world_z(
        self, subject_key: str, arm: str, delta_m: float, *,
        terminal_settle_max_steps: int, terminal_joint_tolerance_rad: float,
    ) -> bool:
        self.plan_success = True
        return execute_held_lift(self, self.subject, str(arm), delta_m, self.hold_reference,
            support_plane_z_m=float(self.subject_support_plane_z),
            required_clearance_m=PICK_ACQUISITION_REQUIRED_CLEARANCE_M,
            terminal_settle_max_steps=terminal_settle_max_steps,
            terminal_joint_tolerance_rad=terminal_joint_tolerance_rad)

    def _unified_observe_attachment(self, arm: str) -> dict[str, Any]:
        del arm
        return self._observe_probe_pick_attachment()

    def _unified_total_lift_delta(self, subject_key: str) -> float:
        del subject_key
        return float(
            self.subject.get_pose().p[2] - self.initial_subject_pose.p[2]
        )

    def _unified_current_actor_pose(self, subject_key: str) -> list[float]:
        del subject_key
        pose = self.subject.get_pose()
        return [*pose.p.tolist(), *pose.q.tolist()]

    def _unified_acquire_held_drive(
        self,
        subject_key: str,
        arm: str,
        *,
        stiffness: float,
        damping: float,
    ) -> dict[str, Any]:
        return self.held_arm_drive_leases.acquire(
            subject_key,
            arm,
            stiffness=float(stiffness),
            damping=float(damping),
        )

    def _unified_require_held_drive(
        self, subject_key: str, arm: str
    ) -> dict[str, Any]:
        return self.held_arm_drive_leases.require(subject_key, arm)

    def _unified_release_held_drive(
        self, subject_key: str, arm: str
    ) -> bool:
        return self.held_arm_drive_leases.release(subject_key, arm)

    def _unified_release_all_held_drives(self) -> None:
        manager = getattr(self, "held_arm_drive_leases", None)
        if manager is not None:
            manager.release_all()

    def _unified_held_actor_target_ee_pose(
        self,
        subject_key: str,
        arm: str,
        actor_target_pose: Sequence[float],
    ) -> list[float]:
        return self._unified_handover_actor_target_ee_pose(
            subject_key, arm, actor_target_pose
        )

    def _unified_move_held_to_ee_pose(
        self,
        subject_key: str,
        arm: str,
        target_ee_pose: Sequence[float],
        *,
        terminal_settle_max_steps: int,
        terminal_joint_tolerance_rad: float,
    ) -> bool:
        del subject_key
        arm_tag = ArmTag(arm)
        self.plan_success = True
        motion = self.move_to_pose(arm_tag, list(target_ee_pose))
        for action in motion[1]:
            action.args.update(
                {
                    "terminal_settle_max_steps": int(
                        terminal_settle_max_steps
                    ),
                    "terminal_joint_tolerance_rad": float(
                        terminal_joint_tolerance_rad
                    ),
                }
            )
        self._move_probe_held(arm_tag, motion)
        return bool(self.plan_success)

    def _unified_certify_pick(
        self, subject_key: str, arm: str
    ) -> dict[str, Any]:
        del subject_key
        snapshot = self._build_success_snapshot(
            pick_certification=True
        )
        predicate = evaluate_m4_success(
            "pick_success",
            {"subject": "subject", "requested_arm": str(arm)},
            snapshot,
            SuccessContract.load(),
        )
        failure_code = None
        if predicate["status"] != "passed":
            failure_code = (
                "pick_vertical_lift_unreachable"
                if "support_clearance"
                in predicate.get("failed_predicates", [])
                else "pick_success_failed"
            )
        return {
            "passed": predicate["status"] == "passed",
            "failure_code": failure_code,
            "predicate_result": predicate,
            "success_snapshot_hash": snapshot["success_snapshot_hash"],
        }

    def _unified_commit_pick(
        self,
        subject_key: str,
        arm: str,
        candidate: Mapping[str, Any],
        certification: Mapping[str, Any],
    ) -> None:
        del subject_key
        self.pick_certification = {
            "subject": "subject",
            "arm": str(arm),
            "contact_id": int(candidate["contact_id"]),
            "approach_vector": list(
                candidate["approach_direction_world"]
            ),
            "pick_acquisition_policy_version": (
                PICK_ACQUISITION_POLICY_VERSION
            ),
            "predicate_result": deepcopy(
                certification.get("predicate_result")
            ),
            "success_snapshot_hash": certification.get(
                "success_snapshot_hash"
            ),
        }

    def _unified_release_for_retry(
        self, subject_key: str, arm: str
    ) -> None:
        del subject_key
        arm_tag = ArmTag(arm)
        try:
            self._remove_probe_pick_attachment()
            self.robot.detach_collision_object(str(arm_tag))
            self.hold_reference = None
            self.pick_certification = None
            self.plan_success = True
            self.move(self.open_gripper(arm_tag))
            for _ in range(12):
                self.scene.step()
            self.plan_success = True
            self.move(self.move_by_displacement(arm_tag, z=0.05))
            self.plan_success = True
        finally:
            self.held_arm_drive_leases.release_arm(str(arm_tag))

    def close_env(self, clear_cache: bool = False) -> None:
        self._unified_release_all_held_drives()
        super().close_env(clear_cache=clear_cache)

    def _pick(
        self,
        arm_tag: ArmTag,
        *,
        contact_ids: Sequence[int] | None = None,
        use_profile: str = "single_arm_pick",
        group_index: int | None = None,
    ):
        self.probe_diagnostics["exact_pick_physical_attempts"] = []
        result = execute_pick_acquisition(
            self,
            PickAcquisitionRequest(
                subject="subject",
                arm=str(arm_tag),
                asset=self.asset_name,
                model_id=str(self.model_id),
                use=self.candidate_use,
                use_profile=use_profile,
                requested_pre_grasp_m=0.08,
                table_acquisition=True,
                support_plane_z_m=float(self.subject_support_plane_z),
                contact_groups=self.contact_groups,
                group_index=group_index,
                contact_ids=contact_ids,
                maximum_attempts=int(
                    EXACT_GRASP_POLICY[
                        "maximum_physical_contact_attempts"
                    ]
                ),
                minimum_normalized_joint_margin=float(
                    EXACT_GRASP_POLICY[
                        "minimum_normalized_joint_margin"
                    ]
                ),
                world_yaw_deg=0.0,
                reach_zone=f"m4_{self.use}",
            ),
        )
        physical_attempts = result["attempts"]
        self.plan_success = bool(
            result["verified"]
            or any(
                row.get("execution_started")
                for row in physical_attempts
            )
        )
        self.info["info"] = {
            "{A}": f"{self.asset_name}/base{self.model_id}",
            "{arm}": str(arm_tag),
            "canonical_probe": self.use,
            "root_pose": self.probe_diagnostics.get("root_pose", {}),
            "exact_pick_physical_attempts": deepcopy(physical_attempts),
            "pick_certification": deepcopy(self.pick_certification),
            "pick_acquisition_policy_version": (
                PICK_ACQUISITION_POLICY_VERSION
            ),
            "unified_pick_execution": result["policy_version"],
            "failure_reason": result.get("failure_code"),
        }
        return self.info

    def _unified_handover_ensure_source(
        self,
        subject_key: str,
        source_arm: str,
        contact_groups: Sequence[Sequence[int]],
    ) -> bool:
        del contact_groups
        if self._unified_handover_source_ready(
            subject_key, source_arm
        ):
            return True
        self._pick(
            ArmTag(source_arm),
            contact_ids=None,
            use_profile="single_arm_pick",
        )
        return self._unified_handover_source_ready(
            subject_key, source_arm
        )

    def _unified_handover_source_ready(
        self, subject_key: str, source_arm: str
    ) -> bool:
        certification = self.pick_certification or {}
        return bool(
            subject_key == "subject"
            and certification.get("subject") == "subject"
            and certification.get("arm") == source_arm
            and certification.get("contact_id") is not None
        )

    def _unified_handover_target_empty(self, target_arm: str) -> bool:
        del target_arm
        return (
            self._handover_pending_attachment is None
            and self._handover_pending_reference is None
        )

    def _unified_handover_source_attachment_active(
        self, source_arm: str
    ) -> bool:
        del source_arm
        return isinstance(
            self.pick_attachment, SapienFixedAttachment
        )

    def _unified_handover_source_contact_id(
        self, source_arm: str
    ) -> int:
        certification = self.pick_certification or {}
        if (
            certification.get("arm") != source_arm
            or certification.get("contact_id") is None
        ):
            raise HandoverExecutionError(
                "handover_source_contact_missing"
            )
        return int(certification["contact_id"])

    def _unified_handover_actor_target_ee_pose(
        self,
        subject_key: str,
        arm: str,
        actor_target_pose: Sequence[float],
    ) -> list[float]:
        del subject_key
        reference = self._unified_handover_source_reference(arm)
        try:
            return held_actor_target_to_planning_ee_pose(
                reference, actor_target_pose
            )
        except SuccessObservationError as exc:
            raise HandoverExecutionError(
                "verified_grasp_transform_missing"
            ) from exc

    def _unified_handover_source_reference(
        self, source_arm: str
    ) -> dict[str, Any]:
        if (
            not isinstance(self.hold_reference, dict)
            or self.hold_reference.get("arm") != source_arm
        ):
            raise HandoverExecutionError(
                "handover_source_reference_missing"
            )
        return self.hold_reference

    def _unified_handover_move_source(
        self,
        subject_key: str,
        source_arm: str,
        target_pose: Sequence[float],
        *,
        collision_obstacles: Sequence[Mapping[str, Any]] = (),
    ) -> bool:
        del subject_key
        arm = ArmTag(source_arm)
        self.plan_success = True
        world = refresh_collision_world(
            self,
            target_actor=self.subject,
            extra_obstacles=collision_obstacles,
        )
        world_hash = str(
            (world.get(source_arm) or {}).get("world_hash") or ""
        )
        self.hold_reference["collision_world_hash"] = world_hash
        motion = self.move_to_pose(arm, list(target_pose))
        for action in motion[1]:
            action.args.update(
                {
                    "expected_world_hash": world_hash,
                    "expected_attachment_hash": self.hold_reference[
                        "attachment_hash"
                    ],
                }
            )
        self.move(motion)
        return bool(self.plan_success)

    def _unified_handover_preposition_target(
        self,
        subject_key: str,
        target_arm: str,
        target_pose: Sequence[float],
        *,
        collision_obstacles: Sequence[Mapping[str, Any]] = (),
    ) -> bool:
        del subject_key
        arm = ArmTag(target_arm)
        self.plan_success = True
        self._diagnostic_video_stage = "handover:target_preposition"
        world = refresh_collision_world(
            self,
            target_actor=self.subject,
            extra_obstacles=collision_obstacles,
        )
        world_hash = str(
            (world.get(target_arm) or {}).get("world_hash") or ""
        )
        motion = self.move_to_pose(arm, list(target_pose))
        for action in motion[1]:
            action.args["expected_world_hash"] = world_hash
        self.move(motion)
        passed = bool(self.plan_success)
        self.plan_success = True
        return passed

    def _unified_handover_freeze_target_reference(
        self, subject_key: str, target_arm: str
    ) -> None:
        del subject_key
        profile = load_embodiment_profile(
            DEFAULT_ALOHA_GRIPPER_PROFILE
        )
        gripper = observe_actual_grippers(
            self.robot, embodiment_profile=profile
        )[target_arm]
        self._handover_pending_reference = build_hold_reference(
            subject="subject",
            arm=target_arm,
            actor_matrix=self.subject.get_pose().to_transformation_matrix(),
            ee_matrix=self._probe_hold_ee_matrix(ArmTag(target_arm)),
            planning_ee_matrix=self._probe_ee_matrix(ArmTag(target_arm)),
            actual_width_m=float(gripper["actual_width_m"]),
            ee_frame=HOLD_EE_FRAME_ARM,
        )

    def _unified_handover_active_contacts(self) -> list[dict[str, Any]]:
        profile = load_embodiment_profile(
            DEFAULT_ALOHA_GRIPPER_PROFILE
        )
        grippers = observe_actual_grippers(
            self.robot, embodiment_profile=profile
        )
        return self._probe_active_contacts(profile, grippers)

    def _unified_handover_observe_target_close(
        self, subject_key: str, target_arm: str
    ) -> dict[str, Any]:
        del subject_key
        continuity = int(
            load_embodiment_profile(DEFAULT_ALOHA_GRIPPER_PROFILE)[
                "classification"
            ]["continuous_contact_min_steps"]
        )
        return self._probe_hold_observation(
            ArmTag(target_arm),
            steps=continuity,
            reference=self._handover_pending_reference,
        )

    def _unified_handover_target_hold_failure(
        self, summary: Mapping[str, Any], *, phase: str
    ) -> str | None:
        if phase == "carry":
            return self._unified_hold_failure(
                summary, phase=phase
            )
        return hold_failure_code(
            summary, phase=phase, contract=SuccessContract.load()
        )

    def _unified_handover_store_target_closure(
        self, target_arm: str, evidence: Mapping[str, Any]
    ) -> None:
        if (
            not isinstance(self._handover_pending_reference, dict)
            or self._handover_pending_reference.get("arm")
            != target_arm
        ):
            raise HandoverExecutionError(
                "handover_target_reference_missing"
            )
        self._handover_pending_reference[
            "certified_closure_evidence"
        ] = deepcopy(dict(evidence))

    def _unified_handover_create_target_attachment(
        self, subject_key: str, target_arm: str
    ) -> dict[str, Any]:
        del subject_key
        attachment, result = create_evaluated_pick_attachment(
            self,
            self.subject,
            arm=target_arm,
        )
        self._handover_pending_attachment = attachment
        return result

    def _unified_handover_finalize_target_reference(
        self,
        subject_key: str,
        target_arm: str,
        closure_evidence: Mapping[str, Any],
    ) -> None:
        """Freeze carry drift at the target D6 zero-residual relation."""

        del subject_key
        old_reference = (
            self._handover_pending_reference
            if isinstance(self._handover_pending_reference, dict)
            else self.hold_reference
        )
        profile = load_embodiment_profile(
            DEFAULT_ALOHA_GRIPPER_PROFILE
        )
        gripper = observe_actual_grippers(
            self.robot, embodiment_profile=profile
        )[target_arm]
        reference = build_hold_reference(
            subject="subject",
            arm=target_arm,
            actor_matrix=self.subject.get_pose().to_transformation_matrix(),
            ee_matrix=self._probe_hold_ee_matrix(ArmTag(target_arm)),
            planning_ee_matrix=self._probe_ee_matrix(ArmTag(target_arm)),
            actual_width_m=float(gripper["actual_width_m"]),
            ee_frame=HOLD_EE_FRAME_ARM,
        )
        reference["certified_closure_evidence"] = deepcopy(
            dict(closure_evidence)
        )
        reference["reference_phase"] = "target_attachment_zero_residual"
        if isinstance(old_reference, Mapping):
            for key in ("collision_world_hash", "attachment_hash"):
                if key in old_reference:
                    reference[key] = deepcopy(old_reference[key])
        if isinstance(self._handover_pending_reference, dict):
            self._handover_pending_reference = reference
        else:
            self.hold_reference = reference

    def _unified_handover_abort_target(self, target_arm: str) -> None:
        try:
            if isinstance(
                self._handover_pending_attachment,
                SapienFixedAttachment,
            ):
                self._handover_pending_attachment.remove()
            self._handover_pending_attachment = None
            self._handover_pending_reference = None
            self._handover_pending_certification = None
            self.plan_success = True
            self.move(self.open_gripper(ArmTag(target_arm)))
            self.plan_success = True
        finally:
            self.held_arm_drive_leases.release_arm(target_arm)

    def _unified_handover_stage_target(
        self,
        subject_key: str,
        target_arm: str,
        candidate: Mapping[str, Any],
        plan: Mapping[str, Any],
    ) -> None:
        del subject_key
        self._handover_pending_certification = {
            "subject": "subject",
            "arm": target_arm,
            "contact_id": int(candidate["contact_id"]),
            "approach_vector": list(
                candidate["approach_direction_world"]
            ),
            "handover_plan_hash": plan["plan_hash"],
            "handover_policy_version": plan["policy_version"],
        }

    def _unified_handover_swap_ownership(
        self, subject_key: str, source_arm: str, target_arm: str
    ) -> None:
        del subject_key
        if not isinstance(
            self._handover_pending_attachment,
            SapienFixedAttachment,
        ):
            raise HandoverExecutionError(
                "handover_target_attachment_missing"
            )
        self._remove_probe_pick_attachment()
        self.robot.detach_collision_object(source_arm)
        self.hold_reference = self._handover_pending_reference
        self.pick_attachment = self._handover_pending_attachment
        self.pick_certification = (
            self._handover_pending_certification
        )
        self._handover_pending_reference = None
        self._handover_pending_attachment = None
        self._handover_pending_certification = None
        self._prepare_probe_collision(ArmTag(target_arm))

    def _unified_handover_release_source(
        self, source_arm: str
    ) -> bool:
        try:
            self.plan_success = True
            self.move(self.open_gripper(ArmTag(source_arm)))
            passed = bool(self.plan_success)
            self.plan_success = True
            return passed
        finally:
            self.held_arm_drive_leases.release_arm(source_arm)

    def _unified_handover_retreat_source(
        self, source_arm: str, target_pose: Sequence[float]
    ) -> bool:
        arm = ArmTag(source_arm)
        self.plan_success = True
        self.move(self.move_to_pose(arm, list(target_pose)))
        passed = bool(self.plan_success)
        self.plan_success = True
        return passed

    def _unified_handover_move_target(
        self,
        subject_key: str,
        target_arm: str,
        target_pose: Sequence[float],
    ) -> bool:
        del subject_key
        arm = ArmTag(target_arm)
        self.plan_success = True
        self._move_probe_held(
            arm, self.move_to_pose(arm, list(target_pose))
        )
        passed = bool(self.plan_success)
        self.plan_success = True
        return passed

    def _unified_handover_observe_target_carry(
        self, subject_key: str, target_arm: str
    ) -> dict[str, Any]:
        del subject_key
        continuity = int(
            load_embodiment_profile(DEFAULT_ALOHA_GRIPPER_PROFILE)[
                "classification"
            ]["continuous_contact_min_steps"]
        )
        return self._probe_hold_observation(
            ArmTag(target_arm), steps=continuity
        )

    def _unified_handover_record(
        self, op: str, payload: Mapping[str, Any]
    ) -> None:
        self.probe_diagnostics.setdefault(
            "unified_handover_events", []
        ).append(
            {
                "op": op,
                **{
                    key: deepcopy(value)
                    for key, value in payload.items()
                    if not str(key).startswith("_")
                },
            }
        )

    def _unified_handover_complete(
        self,
        subject_key: str,
        source_arm: str,
        target_arm: str,
        result: Mapping[str, Any],
    ) -> None:
        del subject_key, source_arm, target_arm, result
        self.plan_success = True

    def _handover(self):
        if len(self.contact_groups) < 2:
            self.plan_success = False
            self.info["info"] = {
                "{A}": f"{self.asset_name}/base{self.model_id}",
                "canonical_probe": self.use,
                "root_pose": self.probe_diagnostics.get(
                    "root_pose", {}
                ),
                "failure_reason": "missing_handover_contact_groups",
                "contact_group_count": len(self.contact_groups),
            }
            return self.info
        source = ArmTag(
            "right" if self.subject.get_pose().p[0] > 0 else "left"
        )
        target = source.opposite
        self.handover_source = source
        self.handover_target = target
        self.probe_diagnostics["handover_contact_groups"] = {
            "contact_groups": deepcopy(self.contact_groups),
            "functional_point_count": functional_point_count(
                self.candidate,
                model_data=_load_model_data(self.candidate),
            ),
        }
        try:
            result = execute_handover(
                self,
                HandoverRequest(
                    subject="subject",
                    source_arm=str(source),
                    target_arm=str(target),
                    asset=self.asset_name,
                    model_id=str(self.model_id),
                    contact_groups=self.contact_groups,
                    table_top_m=float(
                        self.initial_subject_aabb["min"][2]
                    ),
                    world_yaw_deg=0.0,
                    reach_zone=f"m4_{self.use}",
                    minimum_normalized_joint_margin=float(
                        EXACT_GRASP_POLICY[
                            "minimum_normalized_joint_margin"
                        ]
                    ),
                ),
            )
            failure_reason = None
            self.plan_success = True
        except HandoverExecutionError as exc:
            result = None
            failure_reason = exc.code
            self.plan_success = False
        self.info["info"] = {
            "{A}": f"{self.asset_name}/base{self.model_id}",
            "{source_arm}": str(source),
            "{target_arm}": str(target),
            "canonical_probe": self.use,
            "root_pose": self.probe_diagnostics.get("root_pose", {}),
            "handover_contact_groups": self.probe_diagnostics.get(
                "handover_contact_groups", {}
            ),
            "unified_handover_result": deepcopy(result),
            "failure_reason": failure_reason,
        }
        return self.info

    def _articulate(self):
        requested_arm = self._articulation_arm()
        contract = SuccessContract.load()
        state = {"switch_on": "on", "switch_off": "off"}.get(
            self.use, self.use
        )
        state_profile = deepcopy(self.drive)
        if state in {"on", "off"}:
            state_profile.setdefault(
                "positive_use",
                contract.section("articulation")["default_positive_use"],
            )
        try:
            execution_profile = resolve_articulation_execution_profile(
                self.asset_name,
                self.model_id,
                self.use,
                contract=contract,
            )
        except ArticulationExecutionError as exc:
            self.plan_success = False
            self.probe_diagnostics["articulation_action"] = {
                "failure_code": str(exc),
            }
            return self.info
        profile_provenance = {
            "execution_profile_version": execution_profile["profile_version"],
            "execution_profile_hash": execution_profile["profile_hash"],
            "execution_profile_source_hash": execution_profile[
                "source_asset_catalog_sha256"
            ],
            "motion_policy_version": execution_profile[
                "motion_policy_version"
            ],
            "motion_policy_hash": execution_profile["motion_policy_hash"],
        }
        # Bind provenance before any geometry lookup or planning.  Every later
        # failure path must remain attributable to the exact reviewed profile
        # selected for this attempt.
        self.probe_diagnostics["articulation_action"] = deepcopy(
            profile_provenance
        )
        state_profile.setdefault(
            "positive_use", execution_profile["positive_use"]
        )
        self.articulation_state_profile = deepcopy(state_profile)
        actual_link = self._probe_active_joint_child_name()
        if actual_link != execution_profile["moving_link"]:
            self.plan_success = False
            self.probe_diagnostics["articulation_action"] = {
                **profile_provenance,
                "failure_code": "articulation_contact_wrong_link",
                "actual_moving_link": actual_link,
                "expected_moving_link": execution_profile["moving_link"],
            }
            return self.info
        selected_arm = requested_arm
        result: dict[str, Any] | None = None
        arm_attempts: list[dict[str, Any]] = []
        # M4 v3 capability evidence is arm-requested evidence. Never let a
        # successful opposite-arm fallback count for the requested arm.
        arm_sequence = (
            (requested_arm,)
            if hasattr(self, "requested_articulation_arm")
            else (requested_arm, requested_arm.opposite)
        )
        for arm in arm_sequence:
            selected_arm = arm
            result = self._run_probe_articulation_with_arm(
                arm=arm,
                state=state,
                state_profile=state_profile,
                execution_profile=execution_profile,
            )
            arm_attempts.append({"arm": str(arm), "result": result})
            if result["success"]:
                break
            if not preengagement_arm_fallback_allowed(result):
                break
        self.articulation_arm = selected_arm
        self.plan_success = bool(result and result["success"])
        action_diagnostic = {
            **profile_provenance,
            "requested_arm": str(requested_arm),
            "selected_arm": str(selected_arm),
            "drive_mode": self.drive.get("drive_mode"),
            "articulation_result": result,
            "arm_attempts": arm_attempts,
            "continuous_motion": deepcopy(
                self.probe_diagnostics.get(
                    "continuous_articulation_motion", {}
                )
            ),
            "source": articulated_task_hint(self.asset_name, self.model_id).get("source")
            if articulated_task_hint(self.asset_name, self.model_id)
            else "generic_articulation_probe",
        }
        successful_capability = articulation_success_capability(
            action_diagnostic, use=self.use
        )
        if successful_capability is not None:
            action_diagnostic["successful_capability"] = successful_capability
        self.probe_diagnostics["articulation_action"] = action_diagnostic
        self.info["info"] = {
            "{A}": f"{self.asset_name}/{self.model_id}",
            "{arm}": str(selected_arm),
            "canonical_probe": self.use,
            "root_pose": self.probe_diagnostics.get("root_pose", {}),
            "articulation_action": self.probe_diagnostics.get("articulation_action", {}),
        }
        return self.info

    def _probe_active_joint_child_name(self) -> str | None:
        joint_index = int(self.drive.get("joint_index", 0))
        joints = list(self.subject.actor.get_active_joints())
        if joint_index >= len(joints):
            return None
        child = joints[joint_index].child_link
        return str(child.get_name())

    def _probe_articulation_joint_motion(self) -> dict[str, Any]:
        joint_index = int(self.drive.get("joint_index", 0))
        joint = list(self.subject.actor.get_active_joints())[joint_index]
        child = joint.child_link
        joint_pose = getattr(joint, "pose_in_child", None)
        if callable(joint_pose):
            joint_pose = joint_pose()
        if joint_pose is None:
            getter = getattr(joint, "get_pose_in_child", None)
            joint_pose = getter() if callable(getter) else None
        if joint_pose is None:
            raise ArticulationExecutionError("articulation_joint_frame_invalid")
        frame = (
            np.asarray(child.get_pose().to_transformation_matrix(), dtype=float)
            @ np.asarray(joint_pose.to_transformation_matrix(), dtype=float)
        )
        joint_type = getattr(joint, "type", None)
        if callable(joint_type):
            joint_type = joint_type()
        if joint_type is None:
            getter = getattr(joint, "get_type", None)
            joint_type = getter() if callable(getter) else ""
        return {
            "joint_type": str(joint_type),
            "axis_world": frame[:3, 0],
            "pivot_world": frame[:3, 3],
            "child": child,
        }

    def _probe_articulation_contact_matrix(
        self,
        contact_id: int,
        execution_profile: Mapping[str, Any],
    ) -> np.ndarray:
        if execution_profile.get("engagement_mode") != "switch_press":
            return np.asarray(
                self.subject.get_contact_point(int(contact_id), "matrix"),
                dtype=float,
            )
        point = deepcopy(
            (self.subject.config or {}).get("contact_points", [])[contact_id]
        )
        local = np.asarray(point["matrix"], dtype=float)
        local[:3, 3] *= float(
            (self.subject.config or {}).get("scale") or 1.0
        )
        if execution_profile.get("mirror_contact_local_y"):
            local[1, 3] *= -1.0
        base = str(
            execution_profile.get("contact_frame_base") or point.get("base")
        )
        return (
            np.asarray(
                self.subject.link_dict[base]
                .get_pose()
                .to_transformation_matrix(),
                dtype=float,
            )
            @ local
        )

    def _probe_articulation_grasp_pose(
        self,
        contact_matrix: np.ndarray,
        *,
        pre_grasp_m: float,
        roll_rad: float = 0.0,
        grasp_rotation_world: Sequence[Sequence[float]] | None = None,
    ) -> list[float]:
        pose = articulation_grasp_matrix(
            contact_matrix,
            pre_grasp_m=pre_grasp_m,
            roll_magnitude_rad=roll_rad,
            grasp_rotation_world=grasp_rotation_world,
        )
        return [
            *pose[:3, 3].tolist(),
            *t3d.quaternions.mat2quat(pose[:3, :3]).tolist(),
        ]

    def _run_probe_articulation_with_arm(
        self,
        *,
        arm: ArmTag,
        state: str,
        state_profile: Mapping[str, Any],
        execution_profile: Mapping[str, Any],
    ) -> dict[str, Any]:
        joint_index = int(self.drive.get("joint_index", 0))
        limits = np.asarray(self.subject.get_qlimits(), dtype=float)[joint_index]
        profile = load_embodiment_profile(DEFAULT_ALOHA_GRIPPER_PROFILE)
        required_steps = int(
            profile["classification"]["continuous_contact_min_steps"]
        )
        runtime_state: dict[str, Any] = {
            "contact_id": None,
            "reference": None,
            "attachment": None,
            "arm_drive_properties": None,
        }

        planner = (
            self.robot.left_plan_path
            if str(arm) == "left"
            else self.robot.right_plan_path
        )
        topp = (
            self.robot.left_mplib_planner
            if str(arm) == "left"
            else self.robot.right_mplib_planner
        )
        arm_qpos = (
            self.robot.get_left_arm_real_jointState
            if str(arm) == "left"
            else self.robot.get_right_arm_real_jointState
        )

        def retime_path(
            geometry: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray, float]:
            _, position, velocity, _, duration = topp.TOPP(
                geometry, 1 / 250, verbose=False
            )
            return (
                np.asarray(position, dtype=float),
                np.asarray(velocity, dtype=float),
                float(duration),
            )

        planning_runtime = ContinuousArticulationPlanCallbacks(
            plan_pose=lambda pose, last_qpos: planner(
                list(pose), last_qpos=last_qpos
            ),
            retime_path=retime_path,
            validate_path=lambda positions: (
                self.robot.validate_arm_path_physical_collisions(
                    str(arm),
                    positions,
                    sample_stride=1,
                    excluded_entities=(self.subject,),
                )
            ),
        )

        def compile_continuous_pose_chain(
            target_poses: Sequence[Sequence[float]],
            *,
            stage: str,
        ) -> dict[str, Any]:
            return compile_continuous_articulation_path(
                target_poses,
                runtime=planning_runtime,
                stage=stage,
            )

        def execute_continuous_plan(
            plan: Mapping[str, Any],
            *,
            require_attachment: bool,
        ) -> dict[str, Any]:
            def command_joint(
                position: np.ndarray,
                velocity: np.ndarray,
                index: int,
            ) -> None:
                self.robot.set_arm_joints(position, velocity, str(arm))
                self.scene.step()
                if self.render_freq and index % self.render_freq == 0:
                    self._update_render()
                    self.viewer.render()
                if (
                    self.save_freq is not None
                    and self.save_freq > 0
                    and index % self.save_freq == 0
                ):
                    self._update_render()
                    self._take_picture()

            def observe_checkpoint(
                index: int, count: int
            ) -> Mapping[str, Any] | None:
                del index, count
                if not require_attachment:
                    return None
                attachment = runtime_state.get("attachment")
                if not isinstance(attachment, SapienFixedAttachment):
                    return {
                        "passed": False,
                        "failure_code": "articulation_attachment_lost",
                        "attachment_active": False,
                    }
                metrics = attachment.metrics()
                return {
                    "passed": bool(metrics.get("attachment_active")),
                    "failure_code": (
                        None
                        if metrics.get("attachment_active")
                        else "articulation_attachment_lost"
                    ),
                    "attachment_active": bool(
                        metrics.get("attachment_active")
                    ),
                }

            return execute_continuous_articulation_path(
                plan,
                runtime=ContinuousArticulationReplayCallbacks(
                    command_joint=command_joint,
                    read_arm_qpos=arm_qpos,
                    observe_checkpoint=observe_checkpoint,
                ),
            )

        def read_qpos() -> float:
            return float(self.subject.get_qpos()[joint_index])

        def physical_observation(segment: Mapping[str, Any]) -> dict[str, Any]:
            del segment
            consecutive = 0
            ever_engaged = False
            maximum_translation = 0.0
            maximum_rotation = 0.0
            last_gripper: dict[str, Any] = {}
            last_contacts: list[dict[str, Any]] = []
            for _ in range(required_steps):
                self.scene.step()
                grippers = observe_actual_grippers(
                    self.robot, embodiment_profile=profile
                )
                contacts = self._probe_active_contacts(profile, grippers)
                last_gripper = grippers[str(arm)]
                last_contacts = contacts
                reference = runtime_state.get("reference")
                if reference is not None:
                    child_matrix = np.asarray(
                        self._probe_articulation_joint_motion()[
                            "child"
                        ].get_pose().to_transformation_matrix(),
                        dtype=float,
                    )
                    relative = np.linalg.inv(self._probe_ee_matrix(arm)) @ child_matrix
                    expected = np.asarray(reference, dtype=float)
                    maximum_translation = max(
                        maximum_translation,
                        float(np.linalg.norm(relative[:3, 3] - expected[:3, 3])),
                    )
                    cosine = max(
                        -1.0,
                        min(
                            1.0,
                            (float(np.trace(relative[:3, :3] @ expected[:3, :3].T)) - 1.0)
                            / 2.0,
                        ),
                    )
                    maximum_rotation = max(
                        maximum_rotation, math.degrees(math.acos(cosine))
                    )
                sample = classify_articulation_engagement(
                    mode=str(execution_profile["engagement_mode"]),
                    subject="subject",
                    arm=str(arm),
                    expected_link=str(execution_profile["moving_link"]),
                    gripper=last_gripper,
                    contacts=last_contacts,
                    continuous_steps=1,
                    required_steps=1,
                    translation_drift_m=maximum_translation,
                    rotation_drift_deg=maximum_rotation,
                    translation_limit_m=float(
                        SuccessContract.load().section("hold")[
                            "relative_translation_drift_m"
                        ]
                    ),
                    rotation_limit_deg=float(
                        SuccessContract.load().section("hold")[
                            "relative_rotation_drift_deg"
                        ]
                    ),
                    allowed_contact_regions=execution_profile.get(
                        "allowed_contact_regions", ("inner_pad", "fingertip")
                    ),
                )
                ever_engaged = ever_engaged or bool(sample["passed"])
                consecutive = consecutive + 1 if sample["passed"] else 0
            result = classify_articulation_engagement(
                mode=str(execution_profile["engagement_mode"]),
                subject="subject",
                arm=str(arm),
                expected_link=str(execution_profile["moving_link"]),
                gripper=last_gripper,
                contacts=last_contacts,
                continuous_steps=consecutive,
                required_steps=required_steps,
                translation_drift_m=maximum_translation,
                rotation_drift_deg=maximum_rotation,
                translation_limit_m=float(
                    SuccessContract.load().section("hold")[
                        "relative_translation_drift_m"
                    ]
                ),
                rotation_limit_deg=float(
                    SuccessContract.load().section("hold")[
                        "relative_rotation_drift_deg"
                    ]
                ),
                allowed_contact_regions=execution_profile.get(
                    "allowed_contact_regions", ("inner_pad", "fingertip")
                ),
            )
            result["ever_engaged"] = ever_engaged
            return result

        def attachment_observation(
            segment: Mapping[str, Any],
        ) -> dict[str, Any]:
            del segment
            attachment = runtime_state.get("attachment")
            if not isinstance(attachment, SapienFixedAttachment):
                return {
                    "passed": False,
                    "failure_code": "articulation_attachment_lost",
                    "attachment_active": False,
                }
            maximum_translation = 0.0
            maximum_rotation = 0.0
            metrics: dict[str, Any] = {}
            for _ in range(
                int(execution_profile.get("attachment_observation_steps", required_steps))
            ):
                self.scene.step()
                metrics = attachment.metrics()
                maximum_translation = max(
                    maximum_translation,
                    float(metrics["constraint_translation_residual_m"]),
                )
                maximum_rotation = max(
                    maximum_rotation,
                    float(metrics["constraint_rotation_residual_deg"]),
                )
            return articulation_attachment_status(
                metrics,
                maximum_translation_m=maximum_translation,
                maximum_rotation_deg=maximum_rotation,
            )

        def observation(segment: Mapping[str, Any]) -> dict[str, Any]:
            if runtime_state.get("attachment") is not None:
                return attachment_observation(segment)
            return physical_observation(segment)

        def create_attachment(contact_matrix: np.ndarray) -> dict[str, Any]:
            moving_link = self._probe_articulation_joint_motion()["child"]
            ee_link = physical_ee_link(self.robot, str(arm))
            before_qpos = read_qpos()
            runtime_state["arm_drive_properties"] = boost_arm_drive(
                self.robot,
                str(arm),
                stiffness=float(execution_profile["attached_arm_stiffness"]),
                damping=float(execution_profile["attached_arm_damping"]),
            )
            try:
                attachment = SapienFixedAttachment.create(
                    self.scene,
                    parent_body=moving_link,
                    child_body=ee_link,
                    anchor_world_matrix=contact_matrix,
                    stiffness=float(
                        execution_profile["attachment_drive_stiffness"]
                    ),
                    damping=float(
                        execution_profile["attachment_drive_damping"]
                    ),
                )
            except ArticulationExecutionError as exc:
                restore_arm_drive(runtime_state["arm_drive_properties"] or [])
                runtime_state["arm_drive_properties"] = None
                return {
                    "passed": False,
                    "failure_code": str(exc),
                    "attachment_active": False,
                }
            runtime_state["attachment"] = attachment
            for _ in range(
                int(execution_profile["attachment_creation_settle_steps"])
            ):
                self.scene.step()
            metrics = attachment.metrics()
            qpos_jump = abs(read_qpos() - before_qpos)
            span = float(limits[1] - limits[0])
            qpos_ok = qpos_jump <= span * float(
                execution_profile["attachment_creation_qpos_jump_ratio"]
            )
            if not (metrics["attachment_active"] and qpos_ok):
                attachment.remove()
                runtime_state["attachment"] = None
                restore_arm_drive(runtime_state["arm_drive_properties"] or [])
                runtime_state["arm_drive_properties"] = None
                return {
                    "passed": False,
                    "failure_code": "articulation_attachment_pose_jump",
                    "attachment_active": False,
                    "metrics": {**metrics, "creation_qpos_jump": qpos_jump},
                }
            return {
                "passed": True,
                "failure_code": None,
                "attachment_active": True,
                "metrics": {**metrics, "creation_qpos_jump": qpos_jump},
            }

        def remove_attachment() -> None:
            attachment = runtime_state.get("attachment")
            if isinstance(attachment, SapienFixedAttachment):
                attachment.remove()
            runtime_state["attachment"] = None
            restore_arm_drive(runtime_state.get("arm_drive_properties") or [])
            runtime_state["arm_drive_properties"] = None

        def release_candidate() -> None:
            self.plan_success = True
            self.move(self.open_gripper(arm))
            for _ in range(12):
                self.scene.step()
            self.plan_success = True
            self.move(self.move_by_displacement(arm, z=0.05, move_axis="world"))
            self.plan_success = True

        def acquire(segment: Mapping[str, Any]) -> dict[str, Any]:
            self._diagnostic_video_stage = f"acquire:{state}"
            points = (self.subject.config or {}).get("contact_points", [])
            attempts: list[dict[str, Any]] = []
            geometry_frames: dict[tuple[int, str], dict[str, Any]] = {}
            moving_link_matrix: np.ndarray | None = None
            moving_link_meshes: list[dict[str, np.ndarray]] | None = None
            if execution_profile["engagement_mode"] == "grasp_follow":
                try:
                    moving_link = self._probe_articulation_joint_motion()["child"]
                    moving_link_matrix = np.asarray(
                        moving_link.get_pose().to_transformation_matrix(),
                        dtype=float,
                    )
                    moving_link_meshes = collision_shape_local_meshes(
                        moving_link
                    )
                except (RuntimeError, ValueError) as exc:
                    return {
                        "passed": False,
                        "failure_code": str(exc),
                        "attempts": attempts,
                    }
            candidates = articulation_acquisition_candidates(
                execution_profile, segment["contact_ids"]
            )
            fixed_attachment = bool(
                execution_profile.get("post_acquisition_attachment")
                == "fixed_constraint"
                and execution_profile["engagement_mode"] == "grasp_follow"
            )
            prepared: list[dict[str, Any]] = []
            for raw_contact_id, pre_grasp_m, roll_rad, orientation_variant in candidates:
                contact_id = int(raw_contact_id)
                if contact_id < 0 or contact_id >= len(points):
                    attempts.append(
                        {
                            "contact_id": contact_id,
                            "pre_grasp_m": pre_grasp_m,
                            "roll_rad": roll_rad,
                            "orientation_variant": orientation_variant,
                            "phase": "candidate_validation",
                            "failure_code": "articulation_contact_missing",
                        }
                    )
                    continue
                expected_base = str(
                    execution_profile.get("contact_frame_base")
                    or execution_profile["moving_link"]
                )
                if str(points[contact_id].get("base") or "") != expected_base:
                    attempts.append(
                        {
                            "contact_id": contact_id,
                            "pre_grasp_m": pre_grasp_m,
                            "roll_rad": roll_rad,
                            "orientation_variant": orientation_variant,
                            "phase": "candidate_validation",
                            "failure_code": "articulation_contact_wrong_link",
                        }
                    )
                    continue
                matrix = self._probe_articulation_contact_matrix(
                    contact_id, execution_profile
                )
                grasp_rotation_world = None
                bilateral_geometry = None
                if execution_profile["engagement_mode"] == "grasp_follow":
                    try:
                        geometry_key = (contact_id, str(orientation_variant))
                        if geometry_key not in geometry_frames:
                            geometry_frames[geometry_key] = (
                                derive_articulation_bilateral_grasp_frame(
                                    matrix,
                                    moving_link_matrix,
                                    moving_link_meshes or [],
                                    minimum_closing_width_m=float(
                                        execution_profile[
                                            "minimum_closing_width_m"
                                        ]
                                    ),
                                    maximum_closing_width_m=float(
                                        execution_profile[
                                            "maximum_closing_width_m"
                                        ]
                                    ),
                                    maximum_contact_shape_distance_m=float(
                                        execution_profile[
                                            "maximum_contact_shape_distance_m"
                                        ]
                                    ),
                                    local_surface_patch_radius_m=float(
                                        execution_profile[
                                            "local_surface_patch_radius_m"
                                        ]
                                    ),
                                    minimum_opposed_normal_cosine=float(
                                        execution_profile[
                                            "minimum_opposed_normal_cosine"
                                        ]
                                    ),
                                    orientation_variant=str(orientation_variant),
                                )
                            )
                        bilateral_geometry = geometry_frames[geometry_key]
                        grasp_rotation_world = bilateral_geometry[
                            "rotation_world"
                        ]
                    except ArticulationExecutionError as exc:
                        attempts.append(
                            {
                                "contact_id": contact_id,
                                "pre_grasp_m": pre_grasp_m,
                                "roll_rad": roll_rad,
                                "orientation_variant": orientation_variant,
                                "phase": "bilateral_geometry",
                                "failure_code": str(exc),
                            }
                        )
                        continue
                pre_grasp_pose = self._probe_articulation_grasp_pose(
                    matrix,
                    pre_grasp_m=pre_grasp_m,
                    roll_rad=roll_rad,
                    grasp_rotation_world=grasp_rotation_world,
                )
                contact_pose = self._probe_articulation_grasp_pose(
                    matrix,
                    pre_grasp_m=0.0,
                    roll_rad=roll_rad,
                    grasp_rotation_world=grasp_rotation_world,
                )
                start_matrix = np.asarray(
                    sapien.Pose(
                        pre_grasp_pose[:3], pre_grasp_pose[3:]
                    ).to_transformation_matrix(),
                    dtype=float,
                )
                end_matrix = np.asarray(
                    sapien.Pose(
                        contact_pose[:3], contact_pose[3:]
                    ).to_transformation_matrix(),
                    dtype=float,
                )
                approach = guarded_approach_matrices(
                    start_matrix,
                    end_matrix,
                    steps=int(
                        execution_profile.get("guarded_approach_steps", 1)
                    ),
                )
                prepared.append(
                    {
                        "contact_id": contact_id,
                        "pre_grasp_m": float(pre_grasp_m),
                        "roll_rad": float(roll_rad),
                        "orientation_variant": str(orientation_variant),
                        "grasp_orientation_mode": execution_profile.get(
                            "grasp_orientation_mode"
                        ),
                        "bilateral_geometry": bilateral_geometry,
                        "contact_matrix": matrix,
                        "pre_grasp_pose": pre_grasp_pose,
                        "contact_pose": contact_pose,
                        "target_poses": [
                            pre_grasp_pose,
                            *[
                                [
                                    *approach_matrix[:3, 3].tolist(),
                                    *t3d.quaternions.mat2quat(
                                        approach_matrix[:3, :3]
                                    ).tolist(),
                                ]
                                for approach_matrix in approach
                            ],
                        ],
                    }
                )

            selection = select_articulation_acquisition_plan(
                prepared,
                compile_candidate=lambda candidate: (
                    compile_continuous_pose_chain(
                        candidate["target_poses"],
                        stage="continuous_articulation_acquisition",
                    )
                ),
                joint_limits=articulation_arm_joint_limits(self.robot, str(arm)),
                minimum_normalized_joint_margin=float(
                    execution_profile.get("minimum_normalized_joint_margin", 0.0)
                ),
                maximum_plans=int(
                    execution_profile["maximum_acquisition_attempts"]
                ),
            )
            attempts.extend(selection["planning_attempts"])
            self.probe_diagnostics.setdefault(
                "continuous_articulation_motion", {}
            )["acquisition"] = {
                "status": selection["status"],
                "failure_code": selection.get("failure_code"),
                "planning_attempts": selection["planning_attempts"],
            }
            if selection["status"] != "passed":
                return {
                    "passed": False,
                    "failure_code": selection["failure_code"],
                    "attempts": attempts,
                }

            selected = selection["candidate"]
            contact_id = int(selected["contact_id"])
            pre_grasp_m = float(selected["pre_grasp_m"])
            roll_rad = float(selected["roll_rad"])
            orientation_variant = str(selected["orientation_variant"])
            matrix = np.asarray(selected["contact_matrix"], dtype=float)
            candidate_initial_qpos = read_qpos()
            direction = int(segment["direction_sign"])
            negative_tolerance = float(
                segment["negative_progress_tolerance_qpos"]
            )

            def moved_negative() -> bool:
                return (
                    not fixed_attachment
                    and direction * (read_qpos() - candidate_initial_qpos)
                    < -negative_tolerance
                )

            def negative_failure(phase: str) -> dict[str, Any]:
                attempts.append(
                    {
                        "contact_id": contact_id,
                        "pre_grasp_m": pre_grasp_m,
                        "roll_rad": roll_rad,
                        "orientation_variant": orientation_variant,
                        "phase": phase,
                        "before_qpos": candidate_initial_qpos,
                        "after_qpos": read_qpos(),
                        "failure_code": "articulation_negative_progress",
                    }
                )
                return {
                    "passed": False,
                    "failure_code": "articulation_negative_progress",
                    "attempts": attempts,
                }

            self._diagnostic_video_stage = (
                f"acquire:c{contact_id}:pre={pre_grasp_m}:frame={orientation_variant}"
            )
            if execution_profile["engagement_mode"] in {
                "switch_press", "push_follow"
            }:
                self.plan_success = True
                self.move(self.close_gripper(arm_tag=arm, pos=0.0))
            approach_execution = execute_continuous_plan(
                selection["plan"], require_attachment=False
            )
            if str(approach_execution.get("status")) != "passed":
                attempts.append(
                    {
                        "contact_id": contact_id,
                        "pre_grasp_m": pre_grasp_m,
                        "roll_rad": roll_rad,
                        "orientation_variant": orientation_variant,
                        "phase": "acquisition_replay",
                        "planned": True,
                        "failure_code": approach_execution.get("failure_code"),
                        "execution": approach_execution,
                    }
                )
                return {
                    "passed": False,
                    "failure_code": approach_execution.get("failure_code")
                    or "articulation_continuous_tracking_invalid",
                    "attempts": attempts,
                }
            if moved_negative():
                return negative_failure("approach")
            self.plan_success = True
            if execution_profile["engagement_mode"] == "grasp_follow":
                self.move(self.close_gripper(arm_tag=arm, pos=0.0))
            if moved_negative():
                return negative_failure("close")
            displacement = execution_profile.get("engagement_displacement")
            if self.plan_success and isinstance(displacement, Mapping):
                xyz = [float(value) for value in displacement["xyz"]]
                self.move(
                    self.move_by_displacement(
                        arm,
                        x=xyz[0],
                        y=xyz[1],
                        z=xyz[2],
                        move_axis=str(displacement.get("frame") or "world"),
                    )
                )
            if moved_negative():
                return negative_failure("engagement_displacement")
            runtime_state["contact_id"] = contact_id
            child_matrix = np.asarray(
                self._probe_articulation_joint_motion()[
                    "child"
                ].get_pose().to_transformation_matrix(),
                dtype=float,
            )
            runtime_state["reference"] = (
                np.linalg.inv(self._probe_ee_matrix(arm)) @ child_matrix
            )
            result = physical_observation(segment)
            attempts.append({
                "contact_id": contact_id,
                "pre_grasp_m": pre_grasp_m,
                "roll_rad": roll_rad,
                "orientation_variant": orientation_variant,
                "planned": True,
                "before_qpos": candidate_initial_qpos,
                "after_qpos": read_qpos(),
                "pre_grasp_pose": selected["pre_grasp_pose"],
                "contact_pose": selected["contact_pose"],
                "execution": approach_execution,
                "observation": result,
            })
            if result["passed"]:
                attachment_result = (
                    create_attachment(matrix)
                    if fixed_attachment
                    else {
                        "passed": True,
                        "attachment_active": False,
                    }
                )
                attempts[-1]["attachment"] = attachment_result
                if not attachment_result["passed"]:
                    return {
                        "passed": False,
                        "failure_code": attachment_result["failure_code"],
                        "attachment_active": False,
                        "attempts": attempts,
                    }
                return {
                    "passed": True,
                    "failure_code": None,
                    "attachment_active": bool(
                        attachment_result["attachment_active"]
                    ),
                    "contact_id": contact_id,
                    "moving_link": execution_profile["moving_link"],
                    "engagement_mode": execution_profile["engagement_mode"],
                    "attempts": attempts,
                }
            if (
                direction * (read_qpos() - candidate_initial_qpos)
                > negative_tolerance
            ):
                return {
                "passed": False,
                "failure_code": "articulation_contact_not_acquired",
                "joint_moved_without_engagement": True,
                "attempts": attempts,
            }
            if result.get("ever_engaged") or result.get("failure_code") in {
                "articulation_contact_wrong_link",
                "articulation_translation_slip",
                "articulation_rotation_slip",
            }:
                return {
                    "passed": False,
                    "failure_code": result["failure_code"],
                    "attempts": attempts,
                }
            release_candidate()
            runtime_state["reference"] = None
            return {
                "passed": False,
                "failure_code": "articulation_contact_not_acquired",
                "attempts": attempts,
            }

        def command_segment(
            segment: Mapping[str, Any], target_qpos: float
        ) -> dict[str, Any]:
            self._diagnostic_video_stage = f"servo:q={target_qpos:.4f}"
            contact_id = int(runtime_state["contact_id"])
            current_qpos = read_qpos()
            current_contact = self._probe_articulation_contact_matrix(
                contact_id, execution_profile
            )
            motion = self._probe_articulation_joint_motion()
            initial_ee = self._probe_ee_matrix(arm)
            candidate_attempts: list[dict[str, Any]] = []
            compiled_motion: dict[str, Any] | None = None
            accepted_target_qpos: float | None = None
            for candidate_index, candidate_target in enumerate(
                continuous_articulation_target_candidates(
                    segment, segment["goal"]
                )
            ):
                target_poses: list[list[float]] = []
                for checkpoint_qpos in continuous_articulation_qpos_targets(
                    current_qpos,
                    candidate_target,
                    waypoint_count=CONTINUOUS_ARTICULATION_WAYPOINT_COUNT,
                ):
                    desired_contact = future_contact_matrix(
                        current_contact,
                        joint_type=motion["joint_type"],
                        joint_axis_world=motion["axis_world"],
                        joint_pivot_world=motion["pivot_world"],
                        q_delta=checkpoint_qpos - current_qpos,
                    )
                    target_ee = (
                        desired_contact
                        @ np.linalg.inv(current_contact)
                        @ initial_ee
                    )
                    target_poses.append(
                        [
                            *target_ee[:3, 3].tolist(),
                            *t3d.quaternions.mat2quat(
                                target_ee[:3, :3]
                            ).tolist(),
                        ]
                    )
                candidate_plan = compile_continuous_pose_chain(
                    target_poses,
                    stage=(
                        f"continuous_articulation_servo_candidate_"
                        f"{candidate_index}"
                    ),
                )
                candidate_attempts.append(
                    {
                        key: value
                        for key, value in {
                            **candidate_plan,
                            "target_qpos": candidate_target,
                        }.items()
                        if key not in {"position", "velocity"}
                    }
                )
                if candidate_plan.get("status") == "passed":
                    compiled_motion = candidate_plan
                    accepted_target_qpos = float(candidate_target)
                    break
            if compiled_motion is None or accepted_target_qpos is None:
                last = candidate_attempts[-1]
                return {
                    "status": "failed",
                    "failure_code": last.get("failure_code")
                    or "articulation_continuous_path_unreachable",
                    "candidate_attempts": candidate_attempts,
                }
            motion_execution = execute_continuous_plan(
                compiled_motion,
                require_attachment=(
                    runtime_state.get("attachment") is not None
                ),
            )
            self.plan_success = (
                str(motion_execution.get("status")) == "passed"
            )
            result = {
                key: value
                for key, value in {
                    **compiled_motion,
                    "execution": motion_execution,
                    "initial_qpos": current_qpos,
                    "target_qpos": float(target_qpos),
                    "accepted_target_qpos": accepted_target_qpos,
                    "candidate_attempts": candidate_attempts,
                    "measured_terminal_qpos": read_qpos(),
                    "state_projection_used": False,
                }.items()
                if key not in {"position", "velocity"}
            }
            self.probe_diagnostics.setdefault(
                "continuous_articulation_motion", {}
            )["servo"] = result
            return {
                "status": motion_execution.get("status"),
                "failure_code": motion_execution.get("failure_code"),
                **result,
            }

        def release_retreat(segment: Mapping[str, Any]) -> None:
            del segment
            self._diagnostic_video_stage = "release_retreat"
            remove_attachment()
            self.plan_success = True
            self.move(self.open_gripper(arm))
            retreat = execution_profile.get("retreat") or {}
            xyz = [float(value) for value in retreat.get("xyz", [0.0, 0.0, 0.05])]
            self.plan_success = True
            self.move(
                self.move_by_displacement(
                    arm,
                    x=xyz[0],
                    y=xyz[1],
                    z=xyz[2],
                    move_axis=str(retreat.get("frame") or "world"),
                )
            )
            if not self.plan_success:
                raise ArticulationExecutionError(
                    "articulation_retreat_unreachable"
                )
            runtime_state["reference"] = None

        callbacks = ArticulationRuntimeCallbacks(
            read_qpos=read_qpos,
            acquire=acquire,
            command_step=None,
            observe=observation,
            release_retreat=release_retreat,
            settle_steps=lambda steps: [self.scene.step() for _ in range(int(steps))],
            command_segment=command_segment,
        )
        try:
            return run_articulation_closed_loop(
                state=state,
                initial_qpos=float(self.initial_qpos[joint_index]),
                lower=float(limits[0]),
                upper=float(limits[1]),
                state_profile=state_profile,
                execution_profile=execution_profile,
                contract=SuccessContract.load(),
                runtime=callbacks,
            )
        except ArticulationExecutionError as exc:
            return {
                "success": False,
                "failure_code": str(exc),
                "trace": [],
            }

    def _click_subject(self):
        if self.requested_operation_arm is None:
            raise RuntimeError("source_requested_arm_binding_missing")
        arm = self.requested_operation_arm
        contact_id = self._single_contact_id(default=0)
        self.extended_arm = arm
        self.extended_touch_contact_id = contact_id
        self.probe_diagnostics["extended_action"] = {
            "source": "envs/click_bell.py + envs/click_alarmclock.py",
            "arm": str(arm),
            "contact_id": int(contact_id),
            "downward_displacement": -0.05,
        }
        self.move(
            self.grasp_actor(
                self.subject,
                arm_tag=arm,
                pre_grasp_dis=0.1,
                grasp_dis=0.1,
                contact_point_id=contact_id,
            )
        )
        self.move(self.move_by_displacement(arm, z=-0.05))
        self._record_touch_contact_event(arm, contact_id)
        self.info["info"] = self._extended_info(arm)
        return self.info

    def _press_subject(self):
        if self.requested_operation_arm is None:
            raise RuntimeError("source_requested_arm_binding_missing")
        arm = self.requested_operation_arm
        contact_id = self._single_contact_id(default=2)
        self.extended_arm = arm
        self.extended_touch_contact_id = contact_id
        self.probe_diagnostics["extended_action"] = {
            "source": "envs/press_stapler.py",
            "arm": str(arm),
            "contact_id": int(contact_id),
            "pre_grasp_dis": 0.1,
            "press_dis": 0.02,
        }
        self.move(
            self.grasp_actor(
                self.subject,
                arm_tag=arm,
                pre_grasp_dis=0.1,
                grasp_dis=0.1,
                contact_point_id=contact_id,
            )
        )
        self.move(self.close_gripper(arm_tag=arm))
        self.move(
            self.grasp_actor(
                self.subject,
                arm_tag=arm,
                pre_grasp_dis=0.02,
                grasp_dis=0.02,
                contact_point_id=contact_id,
            )
        )
        self._record_touch_contact_event(arm, contact_id)
        self.info["info"] = self._extended_info(arm)
        return self.info

    def _record_touch_contact_event(
        self, arm: ArmTag, contact_id: int
    ) -> None:
        """Freeze an actual PhysX contact against the authored operation point."""

        self.probe_diagnostics["touch_action"] = {
            "use": str(self.use),
            "requested_arm": str(self.requested_operation_arm or ""),
            "contact_id": int(contact_id),
            "contact_event": None,
        }

        target = np.asarray(
            self.subject.get_contact_point(int(contact_id))[:3], dtype=float
        )
        positions: list[np.ndarray] = []
        summed_impulse = 0.0
        subject_name = str(self.subject.get_name())
        gripper_names = set(self.robot.gripper_name)
        for contact in self.scene.get_contacts():
            names = [body.entity.name for body in contact.bodies]
            if subject_name not in names or not any(
                name in gripper_names for name in names
            ):
                continue
            for point in contact.points:
                positions.append(np.asarray(point.position, dtype=float))
                summed_impulse += float(
                    np.linalg.norm(np.asarray(point.impulse, dtype=float))
                )
        if not positions:
            return
        nearest = min(positions, key=lambda point: float(np.linalg.norm(point - target)))
        bounds = actor_aabb(self.subject)
        extent = float(
            np.max(
                np.asarray(bounds["max"], dtype=float)
                - np.asarray(bounds["min"], dtype=float)
            )
        )
        displacement = float(
            np.linalg.norm(
                np.asarray(self.subject.get_pose().p, dtype=float)
                - np.asarray(self.initial_subject_pose.p, dtype=float)
            )
        )
        maximum_displacement = float(
            SuccessContract.load().section("click_press")[
                "maximum_subject_displacement_m"
            ]
        )
        event = {
            "arm": str(arm),
            "subject": "subject",
            "contact_id": int(contact_id),
            "position_xyz_m": nearest.tolist(),
            "authored_center_xyz_m": target.tolist(),
            "distance_to_authored_center_m": float(
                np.linalg.norm(nearest - target)
            ),
            "target_local_extent_m": extent,
            # A PhysX contact manifold can remain geometrically active while
            # its terminal-step impulse is zero (a settled touch). Preserve
            # manifold evidence separately from impulse so Click/Press does
            # not turn a real resting contact into a false negative.
            "contact_point_count": len(positions),
            "summed_impulse_norm": float(summed_impulse),
            "subject_displacement_m": displacement,
            "subject_displacement_failed": bool(
                displacement > maximum_displacement
            ),
        }
        self.execution_events["contacts"].append(event)
        self.probe_diagnostics["touch_action"]["contact_event"] = deepcopy(
            event
        )

    def _orientation_subject(self):
        arm = self._side_arm()
        contact_ids = self._contact_ids(0)
        target_qpos = self._orientation_target_qpos()
        self.extended_arm = arm
        self.extended_target_qpos = np.array(target_qpos, dtype=float)
        self._pick(
            arm,
            contact_ids=as_contact_id_list(contact_ids),
            use_profile="single_arm_pick",
        )
        result = None
        placement_result = None
        if isinstance(self.pick_certification, Mapping):
            current = self._unified_current_actor_pose("subject")
            target_pose = [*current[:3], *target_qpos]
            self.extended_target_pose = target_pose
            result = execute_held_actor_pose(
                self,
                HeldPoseRequest(
                    subject="subject",
                    arm=str(arm),
                    target_actor_pose=target_pose,
                    position_tolerance_m=0.01,
                    orientation_tolerance_deg=3.0,
                ),
            )
            if result["passed"]:
                rotated = self._unified_current_actor_pose("subject")
                rotated_aabb = actor_aabb(self.subject)
                support_target_z = float(
                    rotated[2]
                    + self.initial_subject_aabb["min"][2]
                    - rotated_aabb["min"][2]
                    + SuccessContract.load().section("orientation")[
                        "release_clearance_m"
                    ]
                )
                support_pose = [
                    float(rotated[0]),
                    float(rotated[1]),
                    support_target_z,
                    *target_qpos,
                ]
                placement_result = execute_held_actor_pose(
                    self,
                    HeldPoseRequest(
                        subject="subject",
                        arm=str(arm),
                        target_actor_pose=support_pose,
                        position_tolerance_m=0.01,
                        orientation_tolerance_deg=3.0,
                    ),
                )
                if placement_result["passed"]:
                    # Physical open-and-retreat after the held actor reaches
                    # the same support plane; no pose teleport or state reset.
                    self._unified_release_for_retry("subject", str(arm))
            if not result["passed"] or not (
                isinstance(placement_result, Mapping)
                and placement_result.get("passed")
            ):
                self.plan_success = False
        self.probe_diagnostics["extended_action"] = {
            "source": "shared_verified_pick_then_held_actor_pose",
            "arm": str(arm),
            "contact_ids": as_contact_id_list(contact_ids),
            "target_quat_wxyz": list(target_qpos),
            "held_pose_result": deepcopy(result),
            "support_placement_result": deepcopy(placement_result),
        }
        self.info["info"] = self._extended_info(arm)
        return self.info

    def _bimanual_grab_subject(self):
        result = execute_bimanual_acquisition(
            self,
            BimanualAcquisitionRequest(
                subject="subject",
                asset=self.asset_name,
                model_id=str(self.model_id),
                contact_groups=self.contact_groups,
                support_plane_z_m=float(self.subject_support_plane_z),
                requested_pre_grasp_m=0.08,
                lift_delta_m=0.08,
                minimum_normalized_joint_margin=float(
                    EXACT_GRASP_POLICY["minimum_normalized_joint_margin"]
                ),
                world_yaw_deg=0.0,
                reach_zone=f"m4_{self.use}",
            ),
        )
        self.plan_success = bool(result["verified"])
        self.probe_diagnostics["bimanual_execution"] = deepcopy(result)
        self.probe_diagnostics["extended_action"] = {
            "source": "shared_simultaneous_bimanual_acquisition",
            "contact_group_count": len(self.contact_groups),
            "selected_group_indices": deepcopy(
                result.get("selected_group_indices")
            ),
            "selected_contact_ids": deepcopy(
                result.get("selected_contact_ids")
            ),
        }
        self.info["info"] = {
            "{A}": f"{self.asset_name}/base{self.model_id}",
            "{left_arm}": "left",
            "{right_arm}": "right",
            "canonical_probe": self.use,
            "root_pose": self.probe_diagnostics.get("root_pose", {}),
            "bimanual_contact_groups": self.probe_diagnostics.get(
                "extended_action", {}
            ),
            "failure_reason": result.get("failure_code"),
        }
        return self.info

    def _check_handover_success(self):
        target = getattr(self, "handover_target", ArmTag("right"))
        source = getattr(self, "handover_source", ArmTag("left"))
        target_closed = self.is_left_gripper_close() if target == "left" else self.is_right_gripper_close()
        source_open = self.is_left_gripper_open() if source == "left" else self.is_right_gripper_open()
        pose = self.subject.get_pose().p
        handed_side = pose[0] < -0.02 if target == "left" else pose[0] > 0.02
        contact = len(self.get_gripper_actor_contact_position(self.asset_name)) > 0
        self.probe_metrics = {
            "target_closed": target_closed,
            "source_open": source_open,
            "handed_side": bool(handed_side),
            "contact": contact,
            "z": float(pose[2]),
            "contact_group_count": len(self.contact_groups),
        }
        return bool(target_closed and source_open and handed_side and contact and pose[2] > 0.82)

    def _check_touch_success(self):
        arm = getattr(self, "extended_arm", self._side_arm())
        contact_id = int(getattr(self, "extended_touch_contact_id", self._single_contact_id(default=0)))
        arm_closed = self.is_left_gripper_close() if arm == "left" else self.is_right_gripper_close()
        target = np.array(self.subject.get_contact_point(contact_id)[:3], dtype=float)
        positions = self.get_gripper_actor_contact_position(self.asset_name)
        contact_match = False
        for position in positions:
            point = np.array(position[:3], dtype=float)
            if np.all(np.abs(point[:2] - target[:2]) < np.array([0.035, 0.035])) and abs(point[2] - target[2]) < 0.05:
                contact_match = True
                break
        self.probe_metrics = {
            "gripper_closed": arm_closed,
            "contact": bool(positions),
            "contact_match": contact_match,
            "contact_id": contact_id,
        }
        return bool(arm_closed and contact_match)

    def _check_orientation_success(self):
        arm = getattr(self, "extended_arm", self._side_arm())
        pose = self.subject.get_pose()
        current_q = np.array(pose.q, dtype=float)
        target_q = np.array(getattr(self, "extended_target_qpos", self._orientation_target_qpos()), dtype=float)
        if current_q[0] < 0:
            current_q = current_q * -1
        if target_q[0] < 0:
            target_q = target_q * -1
        if self.use == "adjust_orientation_subject":
            side = float(getattr(self, "extended_orientation_side", 1.0))
            try:
                functional_pose = np.array(self.subject.get_functional_point(0)[:3], dtype=float)
            except Exception:
                functional_pose = np.array(pose.p, dtype=float)
            side_ok = functional_pose[0] > 0.15 if side > 0 else functional_pose[0] < -0.15
            height_ok = functional_pose[2] > 0.88
            self.probe_metrics = {
                "side_ok": bool(side_ok),
                "height_ok": bool(height_ok),
                "functional_point": functional_pose.tolist(),
                "arm": str(arm),
            }
            return bool(side_ok and height_ok)
        q_ok = bool(np.all(np.abs(current_q - target_q) < np.array([0.08, 0.08, 0.08, 0.08])))
        left_open = self.is_left_gripper_open()
        right_open = self.is_right_gripper_open()
        self.probe_metrics = {
            "q_ok": q_ok,
            "current_qpos": current_q.tolist(),
            "target_qpos": target_q.tolist(),
            "left_open": bool(left_open),
            "right_open": bool(right_open),
            "z": float(pose.p[2]),
        }
        return bool(q_ok and left_open and right_open and float(pose.p[2]) < self.initial_subject_z + 0.08)

    def _check_bimanual_grab_success(self):
        left_closed = self.is_left_gripper_close()
        right_closed = self.is_right_gripper_close()
        lifted = float(self.subject.get_pose().p[2]) >= self.initial_subject_z + 0.05
        contact = len(self.get_gripper_actor_contact_position(self.asset_name)) > 0
        distinct_groups = len(self.contact_groups) >= 2 and set(self.contact_groups[0]).isdisjoint(set(self.contact_groups[1]))
        self.probe_metrics = {
            "left_closed": bool(left_closed),
            "right_closed": bool(right_closed),
            "lifted": bool(lifted),
            "contact": bool(contact),
            "distinct_contact_groups": bool(distinct_groups),
            "contact_group_count": len(self.contact_groups),
        }
        return bool(left_closed and right_closed and lifted and contact and distinct_groups)

    def _apply_articulation_drive_properties(self) -> None:
        drive_props = self.drive.get("drive_props")
        if not isinstance(drive_props, list) or len(drive_props) < 2:
            return
        try:
            damping = float(drive_props[0])
            stiffness = float(drive_props[1])
            self.subject.set_properties(damping, stiffness)
            self.probe_diagnostics["articulation_drive_props"] = {
                "damping": damping,
                "stiffness": stiffness,
            }
        except Exception:
            self.probe_diagnostics["articulation_drive_props"] = {"error": "set_properties_failed"}

    def _configure_articulation_initial_state(self):
        joint_index = int(self.drive.get("joint_index", 0))
        qlimits = np.array(self.subject.get_qlimits(), dtype=float)
        qpos = np.array(self.subject.get_qpos(), dtype=float)
        if joint_index >= len(qpos) or joint_index >= len(qlimits):
            raise RuntimeError("actor_load_failed")
        approved_qpos = list(
            (self.canonical_initial_state.get("articulated_joint_state") or {}).get("qpos") or []
        )
        if len(approved_qpos) != len(qpos):
            raise RuntimeError("source_initial_pose_joint_state_mismatch")
        qpos = np.asarray(approved_qpos, dtype=float)
        diagnostics = {
            "source": "human_approved_initial_pose",
            "joint_index": joint_index,
            "qpos": qpos.tolist(),
        }
        self.subject.set_qpos(qpos)
        try:
            self.subject.set_qvel(np.zeros_like(qpos))
        except Exception:
            pass
        self.probe_diagnostics["articulation_initial_state"] = diagnostics

    def _configure_switch_contact_point(self) -> None:
        if self.use not in SWITCH_STATE_USES:
            return
        contact_sequence = self._articulation_contact_sequence()
        contact_id = int(contact_sequence[0])
        points = (self.subject.config or {}).get("contact_points")
        if not isinstance(points, list) or contact_id >= len(points):
            raise RuntimeError("actor_load_failed")
        point = points[contact_id]
        if not isinstance(point, dict) or point.get("matrix") is None:
            raise RuntimeError("actor_load_failed")
        original_y = float(point["matrix"][1][3])
        resolved = switch_contact_matrix_for_use(point["matrix"], self.use)
        self.probe_diagnostics["switch_contact"] = {
            "contact_id": contact_id,
            "base": point.get("base"),
            "original_local_y": original_y,
            "resolved_local_y": float(resolved[1][3]),
            "source": "reviewed rocker endpoint mirror about local joint-axis plane",
        }

    def _articulation_arm(self) -> ArmTag:
        if hasattr(self, "requested_articulation_arm"):
            return self.requested_articulation_arm
        if self.asset_name == "044_microwave":
            return ArmTag("left")
        if self.asset_name == "015_laptop":
            face_prod = get_face_prod(self.subject.get_pose().q, [1, 0, 0], [1, 0, 0])
            self.probe_diagnostics["laptop_face_prod"] = float(face_prod)
            return ArmTag("left" if face_prod > 0 else "right")
        if self.use in SWITCH_STATE_USES:
            return ArmTag("left")
        arm = "right" if self.subject.get_pose().p[0] >= 0 else "left"
        return ArmTag(arm)

    def _articulation_contact_sequence(self) -> list[int]:
        return articulation_contact_sequence(self.drive, self._contact_ids(0), use=self.use)

    def _contact_ids(self, group_index: int):
        if group_index < len(self.contact_groups):
            ids = self.contact_groups[group_index]
            return ids if len(ids) != 1 else ids[0]
        return 0

    def _exact_grasp_actions(
        self,
        arm: ArmTag,
        *,
        use_profile: str,
        requested_pre_grasp_m: float,
        table_acquisition: bool,
        group_index: int | None = None,
        contact_ids: list[int] | None = None,
    ):
        """Plan one canonical probe grasp through the shared R02 engine."""

        requested = float(requested_pre_grasp_m)
        configured = [
            float(value)
            for value in EXACT_GRASP_POLICY["retreat_distances_m"]
        ]
        retreats = (
            [requested] + [value for value in configured if value != requested]
            if any(abs(requested - value) <= 1e-9 for value in configured)
            else configured
        )
        planning = plan_exact_contact_candidates(
            self,
            self.subject,
            arm=str(arm),
            asset=self.asset_name,
            model_id=str(self.model_id),
            use=self.candidate_use,
            use_profile=use_profile,
            contact_groups=self.contact_groups,
            group_index=group_index,
            contact_ids=contact_ids,
            table_acquisition=table_acquisition,
            retreat_distances_m=retreats,
            minimum_normalized_joint_margin=float(
                EXACT_GRASP_POLICY["minimum_normalized_joint_margin"]
            ),
            world_yaw_deg=0.0,
            reach_zone=f"m4_{self.use}",
            support_plane_z_m=(
                float(self.subject_support_plane_z)
                if table_acquisition
                else None
            ),
        )
        self.probe_diagnostics.setdefault("exact_grasp_plans", []).append(
            strip_runtime_arrays(planning)
        )
        candidate = planning.get("best_candidate")
        if not isinstance(candidate, dict):
            self.plan_success = False
            return ArmTag(str(arm)), []
        return exact_grasp_actions(candidate, arm=str(arm))

    def _single_contact_id(self, *, default: int) -> int:
        selected = getattr(self, "placement_selected_contact_id", None)
        if selected is not None:
            return int(selected)
        ids = as_contact_id_list(self._contact_ids(0))
        metadata = self.candidate.get("metadata") or {}
        count = int(metadata.get("contact_points_count") or len(ids) or 0)
        if count and default < count:
            return int(default)
        return int(ids[0]) if ids else 0

    def _side_arm(self) -> ArmTag:
        selected = getattr(self, "placement_selected_arm", None)
        if selected in {"left", "right"}:
            return ArmTag(str(selected))
        return ArmTag("right" if self.subject.get_pose().p[0] >= 0 else "left")

    def _orientation_target_qpos(self) -> list[float]:
        metadata = self.candidate.get("metadata") or {}
        for key in ("target_qpos", "orientation_target_qpos"):
            value = metadata.get(key)
            if isinstance(value, list) and len(value) == 4:
                return [float(item) for item in value]
        return [0.707, 0.707, 0.0, 0.0]

    def _extended_info(self, arm: ArmTag) -> dict[str, Any]:
        return {
            "{A}": f"{self.asset_name}/base{self.model_id}",
            "{arm}": str(arm),
            "canonical_probe": self.use,
            "root_pose": self.probe_diagnostics.get("root_pose", {}),
            "extended_action": self.probe_diagnostics.get("extended_action", {}),
        }


def _contact_groups(candidate: dict[str, Any]) -> list[list[int]]:
    return contact_groups_from_candidate(candidate, model_data=_load_model_data(candidate))


def _load_model_data(candidate: dict[str, Any]) -> dict[str, Any]:
    path = (((candidate.get("evidence") or {}).get("metadata") or {}).get("path"))
    if not path:
        return {}
    try:
        import json

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _model_id(value: Any):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return value
