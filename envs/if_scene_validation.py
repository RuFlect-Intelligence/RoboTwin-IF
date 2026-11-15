"""BenchmarkV1 Milestone 15 exact-scene simulator adapter.

The adapter consumes SceneSpec, Unit, RealizedSceneFamily, and TaskRecord
objects directly.  It never parses ``legacy_export.task_variant`` and never
loads legacy IF design snapshots.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import sapien
import transforms3d as t3d
import yaml

from ._base_task import Base_Task
from .utils import (
    ArmTag,
    UnStableError,
    create_actor,
    create_box,
    create_sapien_urdf_obj,
    create_table,
)
from if_benchmark.grasp_validation.carry_motion import carry_motion_mode
from if_benchmark.grasp_validation.held_collision import (
    plan_held_path,
    bind_held_collision, refresh_held_world, live_table_obstacles, execute_held_lift,
    HELD_COLLISION_POLICY, validate_held_handoff, observe_live_held_camera,
)
from if_benchmark.collision_geometry import (
    collision_aabb_tight,
    collision_shape_local_meshes,
    collision_support_clearance,
    frozen_support_plane_z,
)
from if_benchmark.pair_terminal_settle_worker import (
    actor_aabb,
    contact_metrics,
    place_in_success_metrics,
    place_on_success_metrics,
    stability_from_snapshots,
)
from if_benchmark.articulated_pair_validation import resolve_joint_targets
from if_benchmark.articulated_terminal_contract import articulation_state_metrics
from if_benchmark.initial_pose.core import InitialPoseContractError
from if_benchmark.initial_pose.runtime import (
    apply_dynamic_mass_properties,
    bind_loaded_actor_scale,
    validate_loaded_initial_pose,
)
from if_benchmark.grasp_validation.core import (
    EXACT_GRASP_POLICY,
    build_contact_contract,
)
from if_benchmark.grasp_validation.runtime import (
    refresh_collision_world,
    strip_runtime_arrays,
)
from if_benchmark.grasp_validation.convex_cover import live_collision_pieces
from if_benchmark.grasp_validation.execution import (
    HandoverExecutionError,
    HandoverRequest,
    HeldPoseRequest,
    PickAcquisitionRequest,
    certify_pick_acquisition_plans,
    execute_handover,
    execute_held_actor_pose,
    execute_pick_acquisition,
    pick_acquisition_request_from_record,
    plan_pick_acquisition,
)
from if_benchmark.orientation_adjustment import (
    ORIENTATION_ADJUSTMENT_VERSION,
    normalize_orientation_adjustment_params,
    target_quaternion_wxyz,
)
from if_benchmark.grasp_validation.planning_contract import (
    classify_pick_planning_failure,
    pick_planning_request_record,
    validated_scene_pick_planning_contract,
)
from if_benchmark.reach_validation.contract import (
    ROUTE_EXECUTION_CONTRACT,
    ROUTE_CARRY_PHASES,
    held_collision_descriptor,
    release_retreat_ee_matrix,
    route_collision_excluded_entity_keys,
    route_phase_actor_poses,
    route_phase_policy,
    route_release_retreat_policy,
    route_scene_collision_obstacles,
    route_terminal_admissibility,
    terminal_candidate_from_sources,
)
from if_benchmark.terminal_placement import (
    articulated_cavity_world_matrix,
    load_reviewed_receiver_profiles,
    terminal_observation_key,
    terminal_recipe_binding,
    validate_terminal_frame_binding,
)
from if_benchmark.schema import artifact_hash
from if_benchmark.success_contract import (
    PICK_ACQUISITION_REQUIRED_CLEARANCE_M,
    HOLD_EE_FRAME_ARM,
    PICK_ACQUISITION_POLICY_VERSION,
    PICK_ACQUISITION_TRACKING_GUARD_M,
    SuccessContract,
    SuccessObservationError,
    build_hold_reference,
    build_success_snapshot,
    ee_to_actor_transform,
    evaluate_pick_attachment,
    evaluate_m15_success,
    hold_consistency_from_relative_history,
    hold_failure_code,
    held_actor_target_to_planning_ee_pose,
    observe_actual_grippers,
    observe_sapien_active_contacts,
    require_hold_ee_frame,
    summarize_hold_observation,
)
from if_benchmark.scene_validation.core import (
    _pose_within_tolerance,
    SceneValidationError,
    dispatch_skill_config,
    preflight_skill_operation_anchors,
    resolve_entity_dynamics,
    resolve_reservation_anchor,
)
from if_benchmark.articulation_capability import (
    validated_articulation_arm_capability,
)
from if_benchmark.touch_capability import (
    TOUCH_CAPABILITY_VERSION,
    TOUCH_EXECUTABLE_USES,
)
from if_benchmark.scene_validation.skill_policy import (
    STABILITY_COMPONENT_POLICY_VERSION,
    articulation_use,
)
from if_benchmark.scene_validation.runtime_sources import EpisodeObservationSources
from if_benchmark.scene_validation.runtime_contacts import observe_current_sapien_contacts
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


_TOUCH_EVENT_BY_USE = {
    "click_subject": "click",
    "press_subject": "press",
}
_TOUCH_EVENT_VERSION = "m15_touch_contact_event_v1"


def _validated_touch_step(step: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the closed M13-to-M15 Click/Press execution binding."""

    subject = str(step.get("subject") or "")
    arm = str(step.get("arm") or "")
    use = str(step.get("use") or "")
    event = str(step.get("event") or "")
    anchor_kind = str(step.get("operation_anchor_kind") or "")
    if not subject:
        raise ValueError("touch_subject_missing")
    if arm not in {"left", "right"}:
        raise ValueError("touch_arm_invalid")
    if use not in TOUCH_EXECUTABLE_USES:
        raise ValueError("touch_use_invalid")
    if event != _TOUCH_EVENT_BY_USE[use]:
        raise ValueError("touch_event_use_mismatch")
    if anchor_kind != "functional_or_contact":
        raise ValueError("touch_operation_anchor_kind_invalid")

    raw_contact_id = step.get("contact_id")
    try:
        contact_id = int(raw_contact_id)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("touch_contact_id_invalid") from exc
    if (
        isinstance(raw_contact_id, bool)
        or contact_id < 0
        or float(raw_contact_id) != float(contact_id)
    ):
        raise ValueError("touch_contact_id_invalid")
    operation_anchor_id = str(step.get("operation_anchor_id") or "")
    if operation_anchor_id != f"grasp_contact:{contact_id}":
        raise ValueError("touch_operation_anchor_id_invalid")

    capability = step.get("touch_capability")
    if not isinstance(capability, Mapping):
        raise ValueError("touch_capability_missing")
    arm_capability = capability.get("arm_capability")
    if not isinstance(arm_capability, Mapping):
        raise ValueError("touch_arm_capability_missing")
    try:
        capability_contact_id = int(arm_capability.get("contact_id"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("touch_arm_capability_invalid") from exc
    if (
        capability.get("capability_version") != TOUCH_CAPABILITY_VERSION
        or capability.get("use") != use
        or capability.get("evidence")
        != "m11_exact_m4_touch_capability_v1"
        or arm_capability.get("version") != TOUCH_CAPABILITY_VERSION
        or arm_capability.get("use") != use
        or arm_capability.get("arm") != arm
        or capability_contact_id != contact_id
    ):
        raise ValueError("touch_arm_capability_invalid")
    return {
        "subject": subject,
        "arm": arm,
        "use": use,
        "event": event,
        "contact_id": contact_id,
        "operation_anchor_kind": anchor_kind,
        "operation_anchor_id": operation_anchor_id,
        "touch_capability": deepcopy(dict(capability)),
    }


def _rotation_error_deg(left: np.ndarray, right: np.ndarray) -> float:
    """Return the shortest SO(3) angle between two rotation matrices."""

    relative = np.asarray(left, dtype=float) @ np.asarray(
        right, dtype=float
    ).T
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _source_initial_pose_error(
    reason: str,
    entity_key: str,
    **context: Any,
) -> RuntimeError:
    """Build one machine-readable initial-pose source failure.

    The runtime wrapper classifies every message with this prefix as a source
    failure.  Keeping the reason and evidence in JSON preserves a stable
    failure code without making downstream tooling parse human prose.
    """

    detail = {
        "reason": str(reason),
        "entity_key": str(entity_key),
        **context,
    }
    return RuntimeError(
        "source_initial_pose_contract_error:"
        + json.dumps(
            detail,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


class _SkillPlanFailure(RuntimeError):
    def __init__(self, op: str) -> None:
        super().__init__(f"planning_failed:{op}")
        self.op = op


class _SkillExecutionFailure(RuntimeError):
    def __init__(self, op: str, code: str) -> None:
        super().__init__(f"{code}:{op}")
        self.op = op
        self.code = code


class _RolloutVideoWriter:
    """Small ffmpeg-backed writer for diagnostic head-camera rollouts."""

    def __init__(
        self,
        path: Path,
        *,
        width: int,
        height: int,
        fps: int,
    ) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary_path = path.with_name(f"{path.stem}.tmp.mp4")
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.frame_count = 0
        self.process = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                f"{self.width}x{self.height}",
                "-framerate",
                str(self.fps),
                "-i",
                "-",
                "-an",
                "-vcodec",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                str(self.temporary_path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray) -> None:
        value = np.ascontiguousarray(frame, dtype=np.uint8)
        if value.shape != (self.height, self.width, 3):
            raise RuntimeError(
                "rollout_video_frame_shape_mismatch:"
                f"{value.shape}:{self.height}x{self.width}"
            )
        if self.process.stdin is None:
            raise RuntimeError("rollout_video_stdin_missing")
        self.process.stdin.write(value.tobytes())
        self.frame_count += 1

    def close(self) -> int:
        if self.process.stdin is not None and not self.process.stdin.closed:
            self.process.stdin.close()
        return_code = self.process.wait(timeout=30)
        stderr = (
            self.process.stderr.read().decode("utf-8", errors="replace")
            if self.process.stderr is not None
            else ""
        )
        if return_code != 0:
            raise RuntimeError(
                f"rollout_video_ffmpeg_failed:{return_code}:{stderr[-1000:]}"
            )
        if self.frame_count <= 0 or not self.temporary_path.is_file():
            raise RuntimeError("rollout_video_empty")
        self.temporary_path.replace(self.path)
        return self.frame_count


def entity_dynamics(joined: Mapping[str, Any]) -> dict[str, str]:
    """Resolve variant-invariant dynamics across every task in one scene."""
    return resolve_entity_dynamics(joined)


def _canonical_validation_evidence(
    joined: Mapping[str, Any], entity_key: str, state: str | None = None
) -> dict[str, Any]:
    """Resolve only evidence explicitly frozen into M10 checker plans."""

    realized = joined["realized"]
    plans = (realized.get("preflight") or {}).get("checker_plans") or []
    entity_spec = joined["scene"]["scene_spec"]["entities"].get(entity_key)
    if not isinstance(entity_spec, Mapping):
        raise RuntimeError(f"unknown_articulated_entity:{entity_key}")
    candidates: list[Mapping[str, Any]] = []
    for plan in plans:
        refs = plan.get("entity_refs") or {}
        if refs.get("subject") != entity_key:
            continue
        if state is not None and plan.get("state") != state:
            continue
        candidates.extend(plan.get("required_evidence") or [])
    for ref in sorted(
        candidates,
        key=lambda item: (
            str(item.get("record_id")),
            str(item.get("asset")),
            str(item.get("model_id")),
            str(item.get("use")),
        ),
    ):
        path_value = ref.get("path")
        if (
            ref.get("kind") != "asset_validation"
            or ref.get("status") != "validated"
            or ref.get("stage") != "canonical_skill"
            or not path_value
            or ref.get("asset") != entity_spec.get("asset")
            or str(ref.get("model_id")) != str(entity_spec.get("model_id"))
        ):
            continue
        path = Path(str(path_value))
        if not path.is_file():
            raise RuntimeError(f"missing_canonical_validation_evidence:{path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise RuntimeError(f"invalid_canonical_validation_evidence:{path}")
        if (
            value.get("evidence_id") != ref.get("record_id")
            or value.get("report_hash") != ref.get("hash")
            or value.get("report_hash")
            != artifact_hash(value, omit_keys=("report_hash",))
        ):
            raise RuntimeError(
                f"stale_canonical_validation_evidence:{path}"
            )
        expected_use = str(ref.get("use") or "")
        matching_records = [
            item
            for item in value.get("records", [])
            if item.get("asset") == ref.get("asset")
            and str(item.get("model_id")) == str(ref.get("model_id"))
            and str(item.get("use")) == expected_use
            and bool(item.get("pass"))
        ]
        if len(matching_records) != 1:
            raise RuntimeError(
                "canonical_validation_evidence_record_not_unique_or_validated:"
                f"{entity_key}:{expected_use}"
            )
        return {
            "reference": deepcopy(dict(ref)),
            "record": value,
            "path": str(path),
            "file_hash": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    raise RuntimeError(
        f"missing_canonical_validation_evidence:{entity_key}:{state}"
    )


def _accepted_evidence_attempt(
    evidence: Mapping[str, Any], state: str | None
) -> Mapping[str, Any]:
    attempts = evidence["record"].get("attempts") or []
    reference = evidence["reference"]
    expected_use = str(reference.get("use") or "")
    accepted = [
        item
        for item in attempts
        if bool(item.get("success"))
        and item.get("asset") == reference.get("asset")
        and str(item.get("model_id")) == str(reference.get("model_id"))
        and str(item.get("use")) == expected_use
    ]
    if not accepted:
        raise RuntimeError(
            "canonical_validation_evidence_has_no_successful_attempt"
        )
    return sorted(
        accepted,
        key=lambda item: (
            int(item.get("seed", 0)),
            str(item.get("attempt_id", "")),
        ),
    )[0]


def _per_scene_ids(value: Any) -> list[int]:
    raw = getattr(value, "per_scene_id", None)
    if raw is None:
        getter = getattr(value, "get_per_scene_id", None)
        raw = getter() if callable(getter) else None
    return [int(raw)] if raw is not None else []


def _quat_multiply(
    left: Sequence[float], right: Sequence[float]
) -> list[float]:
    lw, lx, ly, lz = (float(item) for item in left)
    rw, rx, ry, rz = (float(item) for item in right)
    return [
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ]


def _z_rotation_quat(yaw_deg: float) -> list[float]:
    half = math.radians(float(yaw_deg)) / 2.0
    return [math.cos(half), 0.0, 0.0, math.sin(half)]


@lru_cache(maxsize=8)
def _runtime_camera_profile(camera_type: str) -> dict[str, Any]:
    """Read one immutable camera profile without creating render sensors."""

    path = Path(__file__).resolve().parents[1] / "task_config/_camera_config.yml"
    with path.open("r", encoding="utf-8") as stream:
        profiles = yaml.safe_load(stream)
    if not isinstance(profiles, Mapping):
        raise RuntimeError("scene_camera_profile_catalog_invalid")
    profile = profiles.get(str(camera_type))
    if not isinstance(profile, Mapping):
        raise RuntimeError(f"scene_camera_type_unsupported:{camera_type}")
    return deepcopy(dict(profile))


def _planning_head_camera_contract(
    runtime_args: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the authored head-camera source without allocating a sensor.

    M12 Pick planning never observes pixels.  It still validates the exact
    SceneSpec camera type/intrinsics and preserves the camera pose in the
    SceneSpec hash; only the render-only SAPIEN camera entity is omitted.
    """

    camera_options = runtime_args.get("camera")
    embodiment = runtime_args.get("left_embodiment_config")
    if not isinstance(camera_options, Mapping) or not isinstance(
        embodiment, Mapping
    ):
        raise RuntimeError("scene_camera_runtime_config_missing")
    if not bool(camera_options.get("collect_head_camera", True)):
        raise RuntimeError("head_camera_missing")
    static_cameras = embodiment.get("static_camera_list")
    if not isinstance(static_cameras, Sequence) or isinstance(
        static_cameras, (str, bytes)
    ):
        raise RuntimeError("head_camera_missing")
    if not any(
        isinstance(candidate, Mapping)
        and str(candidate.get("name") or "") == "head_camera"
        for candidate in static_cameras
    ):
        raise RuntimeError("head_camera_missing")
    camera_type = str(camera_options.get("head_camera_type") or "D435")
    profile = _runtime_camera_profile(camera_type)
    return {
        "type": camera_type,
        "width": int(profile["w"]),
        "height": int(profile["h"]),
        "fovy_deg": float(profile["fovy"]),
        "near": 0.1,
        "far": 100.0,
    }


def _m12_runtime_fast_path_decision(
    joined: Mapping[str, Any],
    dynamics: Mapping[str, str],
    runtime_args: Mapping[str, Any],
    *,
    diagnostic_video_render: bool,
) -> dict[str, bool]:
    """Return the operational render fast-path decision for one runtime."""

    randomization = runtime_args.get("domain_randomization") or {}
    render_neutral = (
        isinstance(randomization, Mapping)
        and not bool(randomization.get("random_background", False))
        and not bool(randomization.get("random_light", False))
        and math.isclose(
            float(randomization.get("random_head_camera_dis", 0.0)),
            0.0,
            abs_tol=1e-12,
        )
    )
    renderless = bool(
        joined.get("planning_only") is True
        and not diagnostic_video_render
        and os.environ.get("M12_DISABLE_RENDER_FAST_PATH") != "1"
        and int(runtime_args.get("render_freq", 0)) == 0
        and render_neutral
    )
    physics_only_scene = bool(
        renderless
        and all(value != "articulated_fixed" for value in dynamics.values())
    )
    return {
        "active": renderless,
        "camera_sensors_skipped": renderless,
        "entity_visuals_skipped": renderless,
        "renderer_skipped": physics_only_scene,
        "articulated_renderer_fallback": bool(
            renderless and not physics_only_scene
        ),
        "physics_setup_step_preserved": renderless,
    }


class if_scene_validation(Base_Task):
    """One exact BenchmarkV1 scene, with task execution supplied as Skill IR."""

    def setup_demo(
        self,
        *,
        joined: Mapping[str, Any],
        seed: int,
        skill_execution_config: Mapping[str, Any] | None = None,
        stability_config: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        # A reused Python environment still owns a fresh physical episode and
        # fresh validated source copies. Hot observation loops retain the same
        # thresholds and reject any source mutation until the next reset.
        self._observation_sources = EpisodeObservationSources()
        self._m15_single_pass_contacts = bool(
            kwargs.pop("m15_single_pass_contacts", False)
        )
        self._m15_strict_pregrasp_collision = bool(
            kwargs.pop("m15_strict_pregrasp_collision", False)
        )
        self._m15_corrected_carry_collision = bool(
            kwargs.pop("m15_corrected_carry_collision", False)
        )
        self.diagnostic_video_render = bool(
            kwargs.pop("diagnostic_video_render", False)
        )
        self.joined = deepcopy(dict(joined))
        self.validation_seed = int(seed)
        self.skill_execution_config = deepcopy(
            dict(skill_execution_config or {})
        )
        self.stability_config = deepcopy(dict(stability_config or {}))
        self.scene_artifact = self.joined["scene"]
        self.scene_spec = self.scene_artifact["scene_spec"]
        self.realized = (
            self.joined["realized"].get("realized_scene_family")
            or self.joined["realized"]
        )
        self.unit_artifact = self.joined["unit"]
        self.dynamics = entity_dynamics(self.joined)
        # M12 only asks for collision-aware motion witnesses and never reads a
        # pixel.  Keep the exact Engine/Scene, physics setup, robot, actors and
        # collision geometry, while omitting render-only resources.  The
        # conservative renderer fallback for articulated entities avoids
        # changing the established URDF loading path.  M15 and diagnostic
        # video resets never enter this branch.
        self.m12_runtime_fast_path = _m12_runtime_fast_path_decision(
            self.joined,
            self.dynamics,
            kwargs,
            diagnostic_video_render=self.diagnostic_video_render,
        )
        self._m12_renderless = self.m12_runtime_fast_path["active"]
        self._m12_physics_only_scene = self.m12_runtime_fast_path[
            "renderer_skipped"
        ]
        self._m12_head_camera_contract: dict[str, Any] | None = None
        self.entities: dict[str, Any] = {}
        self.initial_pose_state: dict[str, Any] = {}
        self.initial_articulations: dict[str, float] = {}
        self.articulation_joint_indices: dict[str, int] = {}
        self.released_arms: set[str] = set()
        self.held_entities: dict[str, str] = {}
        self.hold_references: dict[str, dict[str, Any]] = {}
        self.pick_attachments: dict[str, SapienFixedAttachment] = {}
        self.pick_certifications: dict[str, dict[str, Any]] = {}
        self.pick_planning_witnesses: dict[str, dict[str, Any]] = {}
        self.pick_planning_requests: dict[str, PickAcquisitionRequest] = {}
        self.pick_planning_certificate: dict[str, Any] = {}
        self.pick_operation_anchor_preflight: dict[str, Any] = {}
        self.current_route_report: dict[str, Any] | None = None
        self.release_retreat_world_hashes: dict[str, str] = {}
        self.initial_success_entities: dict[str, dict[str, Any]] = {}
        self.primary_execution_arm: str | None = None
        self.primary_execution_op: str | None = None
        self.primary_execution_event_hash: str | None = None
        self.participating_arms: set[str] = set()
        self.execution_events: dict[str, list[dict[str, Any]]] = {
            "contacts": []
        }
        self.skill_diagnostics: list[dict[str, Any]] = []
        self.canonical_evidence: dict[str, Any] = {}
        # Source-only R01 checks run before Base_Task creates or moves any
        # simulator entity.  Runtime articulation-name checks necessarily run
        # after loading the articulation, but still before any qpos mutation.
        self._preflight_frozen_initial_pose_contracts()
        self.current_checker_config: Mapping[str, Any] = {}
        self.current_skill_config: Mapping[str, Any] = {}
        self._rollout_video_writer: _RolloutVideoWriter | None = None
        self._rollout_video_label = ""
        self._rollout_video_stage = "setup"
        self._rollout_video_stride = 10
        self._rollout_video_previous_save_freq: int | None = None
        self._m15_expert_review_active = False
        self._m15_expert_review_pending = 0
        self._m15_expert_review_sources: list[dict[str, Any]] = []
        super()._init_task_env_(seed=int(seed), **kwargs)
        self.held_arm_drive_leases = HeldArmDriveLeaseManager(self.robot)
        expected_embodiment = str(
            self.scene_spec.get("actor", {}).get("embodiment") or ""
        )
        actual_embodiment = str(kwargs.get("embodiment_name") or "")
        if expected_embodiment and actual_embodiment != expected_embodiment:
            raise RuntimeError(
                f"scene_embodiment_mismatch:{expected_embodiment}:{actual_embodiment}"
            )
        self._validate_exact_actor_pose()
        self._apply_exact_camera_pose()
        self._capture_initial_state()

    def _stability_components(self) -> list[tuple[str, str, Any]]:
        """Return every physical component covered by entity stability."""

        tracked: list[tuple[str, str, Any]] = []
        for key, wrapper in sorted(self.entities.items()):
            dynamics = self.dynamics.get(key)
            if dynamics == "dynamic":
                tracked.append((key, "root", wrapper.actor))
                continue
            if dynamics != "articulated_fixed":
                continue
            articulation = wrapper.actor
            tracked.append((key, "root", articulation))
            for joint_index, joint in enumerate(
                articulation.get_active_joints()
            ):
                child = joint.child_link
                child_name = str(child.get_name() or f"link_{joint_index}")
                tracked.append(
                    (
                        key,
                        f"active_joint_{joint_index}:{child_name}",
                        child,
                    )
                )
        return tracked

    def _apply_axisymmetric_stability(
        self,
        stability: Mapping[str, Any],
        snapshots: Sequence[Sequence[Mapping[str, Any]]],
        tracked: Sequence[tuple[str, str, Any]],
        *,
        contract: SuccessContract | None = None,
    ) -> dict[str, Any]:
        result = deepcopy(dict(stability))
        contract = contract or self._runtime_success_contract()
        raw_axes = contract.section("hold").get(
            "reviewed_symmetry_axes",
            self.stability_config.get("axial_symmetry_assets", {}),
        )
        if isinstance(raw_axes, Mapping):
            axial_axes = {
                str(asset): str(axis)
                for asset, axis in raw_axes.items()
            }
        else:
            axial_axes = {str(asset): "z" for asset in raw_axes}
        axis_indices = {"x": 0, "y": 1, "z": 2}
        for component_index, (row, (key, component_key, actor)) in enumerate(
            zip(result.get("per_entity", []), tracked)
        ):
            row["entity_key"] = key
            row["component_key"] = component_key
            row["actor_name"] = str(actor.get_name())
            asset = str(
                self.scene_spec["entities"][key].get("asset") or ""
            )
            axis_name = axial_axes.get(asset)
            if axis_name not in axis_indices:
                row["rotation_mode"] = "full_quaternion"
                continue
            final_q = np.asarray(
                snapshots[-1][component_index]["q"], dtype=float
            )
            final_axis = t3d.quaternions.quat2mat(final_q)[
                :, axis_indices[axis_name]
            ]
            axis_angles = []
            for frame in snapshots:
                current_q = np.asarray(
                    frame[component_index]["q"], dtype=float
                )
                current_axis = t3d.quaternions.quat2mat(current_q)[
                    :, axis_indices[axis_name]
                ]
                cosine = float(
                    np.clip(
                        np.dot(current_axis, final_axis)
                        / max(
                            np.linalg.norm(current_axis)
                            * np.linalg.norm(final_axis),
                            1e-12,
                        ),
                        -1.0,
                        1.0,
                    )
                )
                axis_angles.append(math.degrees(math.acos(cosine)))
            metric = f"max_local_{axis_name}_axis_rotation_deg"
            row[metric] = max(axis_angles, default=0.0)
            row["rotation_mode"] = f"local_{axis_name}_axis"
            success_stability = contract.section("stability")
            row["stable"] = bool(
                float(row.get("max_translation_m", math.inf))
                <= float(success_stability["translation_drift_m"])
                and float(row.get(metric, math.inf))
                <= float(success_stability["rotation_drift_deg"])
            )
        result["unstable_indices"] = [
            int(row["index"])
            for row in result.get("per_entity", [])
            if not row.get("stable")
        ]
        result["stable"] = not result["unstable_indices"]
        return result

    def check_stable(self) -> tuple[bool, list[str]]:
        """Apply the frozen M5 settle/window semantics during exact setup."""

        tracked = self._stability_components()
        minimum_steps = int(
            self.stability_config.get("minimum_steps", 600)
        )
        window_steps = int(
            self.stability_config.get("window_steps", 200)
        )
        maximum_steps = max(
            int(self.stability_config.get("maximum_steps", 1600)),
            minimum_steps + window_steps,
        )
        spawn_snapshots = {
            (key, component_key): {
                "p": actor.get_pose().p.tolist(),
                "q": actor.get_pose().q.tolist(),
            }
            for key, component_key, actor in tracked
        }
        for _ in range(minimum_steps):
            self.scene.step()
        steps = minimum_steps
        last: dict[str, Any] = {
            "stable": True,
            "per_entity": [],
        }
        while tracked and steps + window_steps <= maximum_steps:
            snapshots: list[list[dict[str, Any]]] = []
            for _ in range(window_steps):
                self.scene.step()
                snapshots.append(
                    [
                        {
                            "p": actor.get_pose().p.tolist(),
                            "q": actor.get_pose().q.tolist(),
                        }
                        for _, _, actor in tracked
                    ]
                )
            steps += window_steps
            last = stability_from_snapshots(snapshots)
            last["settle_steps"] = steps
            last = self._apply_axisymmetric_stability(
                last, snapshots, tracked
            )
            if last.get("stable"):
                break
        last["policy"] = "m5_settle_window_v2_all_active_components"
        last["component_policy_version"] = STABILITY_COMPONENT_POLICY_VERSION
        spawn_translation_limit = float(
            self.stability_config.get("spawn_translation_m", 0.02)
        )
        spawn_rotation_limit = float(
            self.stability_config.get("spawn_rotation_deg", 10.0)
        )
        spawn_failures = []
        spawn_failure_components: list[str] = []
        spawn_failure_tokens: set[tuple[str, str]] = set()
        for row, (key, component_key, actor) in zip(
            last.get("per_entity", []), tracked
        ):
            spawn = spawn_snapshots[(key, component_key)]
            settled = actor.get_pose()
            translation = float(
                np.linalg.norm(
                    np.asarray(settled.p, dtype=float)
                    - np.asarray(spawn["p"], dtype=float)
                )
            )
            rotation = _rotation_error_deg(
                t3d.quaternions.quat2mat(
                    np.asarray(settled.q, dtype=float)
                ),
                t3d.quaternions.quat2mat(
                    np.asarray(spawn["q"], dtype=float)
                ),
            )
            row["spawn_to_settled_translation_m"] = translation
            row["spawn_to_settled_rotation_deg"] = rotation
            row["spawn_translation_limit_m"] = spawn_translation_limit
            row["spawn_rotation_limit_deg"] = spawn_rotation_limit
            if (
                translation > spawn_translation_limit
                or rotation > spawn_rotation_limit
            ):
                row["stable"] = False
                row["spawn_drift_failed"] = True
                spawn_failures.append(key)
                spawn_failure_tokens.add((key, component_key))
                spawn_failure_components.append(
                    f"{key}:{component_key}"
                )
        last["spawn_drift_failed_entities"] = sorted(set(spawn_failures))
        last["spawn_drift_failed_components"] = spawn_failure_components
        if spawn_failures:
            last["stable"] = False
            last["unstable_indices"] = sorted(
                {
                    *[int(value) for value in last.get("unstable_indices", [])],
                    *[
                        index
                        for index, (key, component_key, _) in enumerate(tracked)
                        if (key, component_key) in spawn_failure_tokens
                    ],
                }
            )
        last = self._snapshot_stability_contract(
            last,
            tracked_components=[
                (key, component_key)
                for key, component_key, _ in tracked
            ],
        )
        self.initial_stability = deepcopy(last)
        unstable = sorted(
            key
            for key, row in (last.get("entities") or {}).items()
            if not row.get("stable")
        )
        return bool(last.get("stable")), unstable

    def setup_scene(self, **kwargs: Any) -> None:
        if self._m12_physics_only_scene:
            # Match Base_Task's physical construction exactly, except that no
            # renderer, lights or viewer are attached to the fresh Engine.
            # A new Engine/Scene is still mandatory for every M12 request.
            self.engine = sapien.Engine()
            self.renderer = None
            scene_config = sapien.SceneConfig()
            self.scene = self.engine.create_scene(scene_config)
            self.scene.set_timestep(kwargs.get("timestep", 1 / 250))
            try:
                self.scene.add_ground(
                    kwargs.get("ground_height", 0), render=False
                )
            except TypeError:
                self.scene.add_ground(kwargs.get("ground_height", 0))
            self.scene.default_physical_material = (
                self.scene.create_physical_material(
                    kwargs.get("static_friction", 0.5),
                    kwargs.get("dynamic_friction", 0.5),
                    kwargs.get("restitution", 0),
                )
            )
            self.direction_light_lst = []
            self.point_light_lst = []
            return
        super().setup_scene(**kwargs)
        if self.diagnostic_video_render:
            # Base_Task creates the render system first and the cameras later.
            # Selecting the raster camera shader here affects only cameras
            # subsequently created for the diagnostic rollout reset.
            sapien.render.set_camera_shader_dir("default")

    def create_table_and_wall(
        self,
        table_xy_bias: list[float] = [0, 0],
        table_height: float = 0.74,
    ) -> None:
        if not self._m12_renderless:
            super().create_table_and_wall(
                table_xy_bias=table_xy_bias,
                table_height=table_height,
            )
            return
        # These calls retain the same static collision bodies and poses as the
        # normal runtime.  Only their RenderBody/visual shapes are omitted.
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
            sapien.Pose(
                p=[table_xy_bias[0], table_xy_bias[1], table_height]
            ),
            length=1.2,
            width=0.7,
            height=table_height,
            thickness=0.05,
            is_static=True,
            load_visual=False,
        )

    def load_camera(self, **kwargs: Any) -> None:
        if not self._m12_renderless:
            super().load_camera(**kwargs)
            return
        self._m12_head_camera_contract = _planning_head_camera_contract(
            kwargs
        )
        self.cameras = None
        # Base_Task's camera loader performs one physics step before syncing
        # render state.  Preserve that step so setup/stability semantics are
        # identical even though no render sensor exists.
        self.scene.step()

    def _initial_pose_contract(
        self,
        entity_key: str,
        spec: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        initial = spec.get("initial_pose")
        if not isinstance(initial, Mapping):
            raise _source_initial_pose_error(
                "binding_missing",
                entity_key,
                missing=["initial_pose"],
            )
        required = (
            "initial_state_id",
            "revision",
            "revision_hash",
            "geometry_hash",
            "canonical_state_hash",
            "q0_wxyz",
            "yaw_policy",
            "scale_xyz",
            "support",
            "articulated_joint_state",
        )
        missing = [key for key in required if initial.get(key) is None]
        if not str(initial.get("canonical_state_hash") or ""):
            missing.append("canonical_state_hash")
        if missing:
            raise _source_initial_pose_error(
                "binding_missing",
                entity_key,
                missing=sorted(set(missing)),
            )
        revision = initial.get("assignment_state_revision")
        if revision is not None:
            if not isinstance(revision, Mapping):
                raise _source_initial_pose_error(
                    "assignment_state_revision_invalid",
                    entity_key,
                    actual_type=type(revision).__name__,
                )
            runtime_geometry_hash = str(
                revision.get("runtime_geometry_hash") or ""
            )
            geometry_hash = str(initial.get("geometry_hash") or "")
            if not geometry_hash or runtime_geometry_hash != geometry_hash:
                raise _source_initial_pose_error(
                    "assignment_geometry_mismatch",
                    entity_key,
                    geometry_hash=geometry_hash,
                    runtime_geometry_hash=runtime_geometry_hash,
                    assignment_revision_hash=str(
                        revision.get("revision_hash") or ""
                    ),
                )
        return initial

    def _preflight_frozen_initial_pose_contracts(self) -> None:
        entities = self.scene_spec.get("entities")
        if not isinstance(entities, Mapping):
            raise _source_initial_pose_error(
                "scene_entities_missing",
                "<scene>",
            )
        for entity_key, spec in sorted(entities.items()):
            if not isinstance(spec, Mapping):
                raise _source_initial_pose_error(
                    "entity_spec_invalid",
                    str(entity_key),
                    actual_type=type(spec).__name__,
                )
            self._initial_pose_contract(str(entity_key), spec)

    def _validate_articulated_runtime_joint_contract(
        self,
        entity_key: str,
        actor: Any,
        initial_contract: Mapping[str, Any],
    ) -> tuple[np.ndarray, int]:
        initial = initial_contract.get("articulated_joint_state")
        if not isinstance(initial, Mapping):
            raise _source_initial_pose_error(
                "joint_state_missing",
                entity_key,
            )
        approved_qpos = initial.get("qpos")
        if not isinstance(approved_qpos, Sequence) or isinstance(
            approved_qpos, (str, bytes)
        ):
            raise _source_initial_pose_error(
                "joint_qpos_missing",
                entity_key,
            )
        try:
            target_qpos = np.asarray(approved_qpos, dtype=float)
        except (TypeError, ValueError) as exc:
            raise _source_initial_pose_error(
                "joint_qpos_invalid",
                entity_key,
                approved_qpos=list(approved_qpos),
            ) from exc
        if (
            target_qpos.ndim != 1
            or len(target_qpos) == 0
            or not np.all(np.isfinite(target_qpos))
        ):
            raise _source_initial_pose_error(
                "joint_qpos_invalid",
                entity_key,
                approved_qpos=target_qpos.tolist(),
            )

        runtime_qpos = np.asarray(actor.get_qpos(), dtype=float)
        active_joints = list(actor.actor.get_active_joints())
        runtime_names = [str(joint.get_name()) for joint in active_joints]
        declared_joints = initial.get("joints")
        if not isinstance(declared_joints, Sequence) or isinstance(
            declared_joints, (str, bytes)
        ):
            raise _source_initial_pose_error(
                "joint_descriptors_missing",
                entity_key,
            )
        if len(declared_joints) != len(active_joints):
            raise _source_initial_pose_error(
                "runtime_joint_count_mismatch",
                entity_key,
                declared_count=len(declared_joints),
                runtime_count=len(active_joints),
                runtime_joint_names=runtime_names,
            )
        if len(target_qpos) != len(runtime_qpos):
            raise _source_initial_pose_error(
                "runtime_qpos_count_mismatch",
                entity_key,
                approved_count=len(target_qpos),
                runtime_count=len(runtime_qpos),
            )

        declared_indices: set[int] = set()
        for row in declared_joints:
            if not isinstance(row, Mapping):
                raise _source_initial_pose_error(
                    "joint_descriptor_invalid",
                    entity_key,
                    actual_type=type(row).__name__,
                )
            raw_index = row.get("joint_index")
            if not isinstance(raw_index, int) or isinstance(raw_index, bool):
                raise _source_initial_pose_error(
                    "joint_index_invalid",
                    entity_key,
                    joint_index=raw_index,
                )
            joint_index = int(raw_index)
            if (
                joint_index < 0
                or joint_index >= len(active_joints)
                or joint_index in declared_indices
            ):
                raise _source_initial_pose_error(
                    "joint_index_invalid",
                    entity_key,
                    joint_index=joint_index,
                    runtime_joint_count=len(active_joints),
                )
            declared_indices.add(joint_index)
            declared_name = str(row.get("joint_name") or "")
            if not declared_name:
                raise _source_initial_pose_error(
                    "joint_name_missing",
                    entity_key,
                    joint_index=joint_index,
                )
            if runtime_names[joint_index] != declared_name:
                raise _source_initial_pose_error(
                    "runtime_joint_name_mismatch",
                    entity_key,
                    joint_index=joint_index,
                    declared_joint_name=declared_name,
                    runtime_joint_name=runtime_names[joint_index],
                )
        if declared_indices != set(range(len(active_joints))):
            raise _source_initial_pose_error(
                "joint_index_set_mismatch",
                entity_key,
                declared_joint_indices=sorted(declared_indices),
                runtime_joint_indices=list(range(len(active_joints))),
            )

        approved_joint_index = initial.get("joint_index")
        if approved_joint_index is None:
            if len(runtime_qpos) != 1:
                raise _source_initial_pose_error(
                    "active_joint_index_missing",
                    entity_key,
                    runtime_qpos_count=len(runtime_qpos),
                )
            approved_joint_index = 0
        if (
            not isinstance(approved_joint_index, int)
            or isinstance(approved_joint_index, bool)
            or int(approved_joint_index) < 0
            or int(approved_joint_index) >= len(runtime_qpos)
        ):
            raise _source_initial_pose_error(
                "active_joint_index_invalid",
                entity_key,
                joint_index=approved_joint_index,
                runtime_qpos_count=len(runtime_qpos),
            )
        return target_qpos, int(approved_joint_index)

    def load_actors(self) -> None:
        articulated_bindings: dict[str, tuple[Any, np.ndarray, int]] = {}
        for entity_key, spec in sorted(self.scene_spec["entities"].items()):
            pose = spec["pose"]
            initial_contract = self._initial_pose_contract(entity_key, spec)
            sapien_pose = sapien.Pose(
                [float(item) for item in pose["position"]],
                [float(item) for item in pose["quat_wxyz"]],
            )
            dynamic_kind = self.dynamics[entity_key]
            if dynamic_kind == "articulated_fixed":
                actor = create_sapien_urdf_obj(
                    self,
                    pose=sapien_pose,
                    modelname=str(spec["asset"]),
                    modelid=int(spec["model_id"]),
                    fix_root_link=True,
                    mass=None,
                )
            else:
                actor = create_actor(
                    self,
                    pose=sapien_pose,
                    modelname=str(spec["asset"]),
                    scale=initial_contract["scale_xyz"],
                    model_id=int(spec["model_id"]),
                    convex=True,
                    is_static=dynamic_kind == "static",
                    mass=None,
                    load_visual=not self._m12_renderless,
                )
            if actor is None:
                raise RuntimeError(f"actor_load_failed:{entity_key}")
            if dynamic_kind == "articulated_fixed":
                qpos, joint_index = (
                    self._validate_articulated_runtime_joint_contract(
                        entity_key,
                        actor,
                        initial_contract,
                    )
                )
                articulated_bindings[entity_key] = (
                    actor,
                    qpos,
                    joint_index,
                )
            if dynamic_kind != "articulated_fixed":
                bind_loaded_actor_scale(actor, initial_contract["scale_xyz"])
            # SceneSpec is the exact frozen placement.  Asset helpers may
            # otherwise add table bias or model transforms before the M15
            # preflight sees the actor.
            actor.actor.set_pose(sapien_pose)
            if dynamic_kind != "static":
                apply_dynamic_mass_properties(actor, initial_contract)
            actor.set_name(entity_key)
            self.entities[entity_key] = actor

        # Apply the frozen qpos only after every loaded articulation has passed
        # the R01 joint-index/name/count contract.  A bad later entity can
        # therefore never leave an earlier entity partially initialized.
        for entity_key, (
            actor,
            qpos,
            joint_index,
        ) in sorted(articulated_bindings.items()):
            actor.set_qpos(qpos)
            actor.set_qvel(np.zeros_like(qpos))
            self.articulation_joint_indices[entity_key] = joint_index
        self._validate_initial_pose_preflight()

    def _validate_initial_pose_preflight(self) -> None:
        for entity_key, actor in sorted(self.entities.items()):
            spec = self.scene_spec["entities"][entity_key]
            initial = self._initial_pose_contract(entity_key, spec)
            scale = (actor.config or {}).get("scale") or [1.0, 1.0, 1.0]
            if isinstance(scale, (int, float)):
                scale = [float(scale)] * 3
            elif len(scale) == 1:
                scale = list(scale) * 3
            joint_qpos = (
                np.asarray(actor.get_qpos(), dtype=float).tolist()
                if self.dynamics[entity_key] == "articulated_fixed"
                else []
            )
            expected_pose = spec["pose"]
            try:
                validate_loaded_initial_pose(
                    canonical_state=initial,
                    actual_q_wxyz=np.asarray(
                        actor.get_pose().q, dtype=float
                    ).tolist(),
                    actual_joint_qpos=joint_qpos,
                    actual_scale_xyz=list(scale),
                    actual_origin_z=float(actor.get_pose().p[2]),
                    expected_world_yaw_deg=float(
                        initial.get("world_yaw_deg") or 0.0
                    ),
                )
            except InitialPoseContractError as exc:
                raise _source_initial_pose_error(
                    "loaded_state_mismatch",
                    entity_key,
                    detail=str(exc),
                ) from exc
            if not _pose_within_tolerance(
                expected_pose["position"],
                expected_pose["quat_wxyz"],
                np.asarray(actor.get_pose().p, dtype=float),
                np.asarray(actor.get_pose().q, dtype=float),
            ):
                raise _source_initial_pose_error(
                    "scene_pose_mismatch",
                    entity_key,
                    expected_position=deepcopy(expected_pose["position"]),
                    expected_quat_wxyz=deepcopy(
                        expected_pose["quat_wxyz"]
                    ),
                    actual_position=np.asarray(
                        actor.get_pose().p, dtype=float
                    ).tolist(),
                    actual_quat_wxyz=np.asarray(
                        actor.get_pose().q, dtype=float
                    ).tolist(),
                )
            actual_pose = actor.get_pose()
            self.canonical_evidence[entity_key] = {
                "kind": "support_validated_human_approved_initial_pose",
                "passed": True,
                "initial_state_id": initial["initial_state_id"],
                "revision": initial["revision"],
                "revision_hash": initial["revision_hash"],
                "geometry_hash": initial["geometry_hash"],
                "canonical_state_hash": initial["canonical_state_hash"],
                "assignment_state_revision": deepcopy(
                    initial.get("assignment_state_revision")
                ),
                "expected": {
                    "position": deepcopy(expected_pose["position"]),
                    "quat_wxyz": deepcopy(expected_pose["quat_wxyz"]),
                    "scale_xyz": deepcopy(initial["scale_xyz"]),
                    "joint_qpos": deepcopy(
                        (initial.get("articulated_joint_state") or {}).get(
                            "qpos", []
                        )
                    ),
                    "joints": deepcopy(
                        (initial.get("articulated_joint_state") or {}).get(
                            "joints", []
                        )
                    ),
                    "origin_z": float(expected_pose["position"][2]),
                },
                "actual": {
                    "position": np.asarray(
                        actual_pose.p, dtype=float
                    ).tolist(),
                    "quat_wxyz": np.asarray(
                        actual_pose.q, dtype=float
                    ).tolist(),
                    "scale_xyz": [float(value) for value in scale],
                    "joint_qpos": joint_qpos,
                    "active_joint_names": (
                        [
                            str(joint.get_name())
                            for joint in actor.actor.get_active_joints()
                        ]
                        if self.dynamics[entity_key] == "articulated_fixed"
                        else []
                    ),
                    "origin_z": float(actual_pose.p[2]),
                },
            }

    def _apply_exact_camera_pose(self) -> None:
        camera = self.scene_spec["camera"]
        if camera.get("name") != "head_camera":
            raise RuntimeError("unsupported_scene_camera")
        expected_camera = {
            "width": int(camera["width"]),
            "height": int(camera["height"]),
            "fovy_deg": float(camera["fovy_deg"]),
            "near": float(camera["near"]),
            "far": float(camera["far"]),
        }
        head = None
        if self._m12_renderless:
            camera_contract = self._m12_head_camera_contract
            if not isinstance(camera_contract, Mapping):
                raise RuntimeError("head_camera_missing")
            actual_type = str(camera_contract["type"])
            actual_camera = {
                key: camera_contract[key] for key in expected_camera
            }
        else:
            head_config = None
            for candidate, name, candidate_config in zip(
                self.cameras.static_camera_list,
                self.cameras.static_camera_name,
                self.cameras.static_camera_config,
            ):
                if name == "head_camera":
                    head = candidate
                    head_config = candidate_config
                    break
            if head is None or head_config is None:
                raise RuntimeError("head_camera_missing")
            actual_type = str(self.cameras.head_camera_type)
            actual_camera = {
                "width": int(head_config["w"]),
                "height": int(head_config["h"]),
                "fovy_deg": float(head_config["fovy"]),
                "near": 0.1,
                "far": 100.0,
            }
        if str(camera.get("type")) != actual_type or any(
            not math.isclose(
                expected_camera[key], actual_camera[key], abs_tol=1e-9
            )
            for key in expected_camera
        ):
            raise RuntimeError(
                f"scene_camera_intrinsics_mismatch:{expected_camera}:{actual_camera}"
            )
        pose = camera["pose"]
        if self._m12_renderless:
            position = np.asarray(pose["position"], dtype=float)
            quaternion = np.asarray(pose["quat_wxyz"], dtype=float)
            if (
                position.shape != (3,)
                or quaternion.shape != (4,)
                or not np.all(np.isfinite(position))
                or not np.all(np.isfinite(quaternion))
                or not math.isclose(
                    float(np.linalg.norm(quaternion)), 1.0, abs_tol=1e-6
                )
            ):
                raise RuntimeError("scene_camera_pose_invalid")
            return
        assert head is not None
        head.entity.set_pose(
            sapien.Pose(
                [float(item) for item in pose["position"]],
                [float(item) for item in pose["quat_wxyz"]],
            )
        )

    def _validate_exact_actor_pose(self) -> None:
        expected = self.scene_spec["actor"]["base_pose"]
        actual = self.robot.left_entity.get_root_pose()
        actual_p = np.asarray(actual.p, dtype=float)
        actual_q = np.asarray(actual.q, dtype=float)
        if not _pose_within_tolerance(
            expected["position"],
            expected["quat_wxyz"],
            actual_p,
            actual_q,
        ):
            raise RuntimeError(
                "scene_actor_base_pose_mismatch:"
                f"expected={expected}:actual_p={actual_p.tolist()}:"
                f"actual_q={actual_q.tolist()}"
            )

    def _frozen_support_plane_z(self, entity_key: str) -> float:
        initial_pose = (
            self.scene_spec["entities"][str(entity_key)].get("initial_pose")
            or {}
        )
        try:
            return frozen_support_plane_z(initial_pose)
        except RuntimeError as exc:
            raise RuntimeError(
                f"scene_initial_pose_support_plane_missing:{entity_key}"
            ) from exc

    def _capture_initial_state(self) -> None:
        for key, actor in self.entities.items():
            pose = actor.get_pose()
            bounds = collision_aabb_tight(actor)
            support_plane_z = self._frozen_support_plane_z(key)
            self.initial_pose_state[key] = {
                "position": pose.p.tolist(),
                "quat_wxyz": pose.q.tolist(),
            }
            self.initial_success_entities[key] = {
                "pose": deepcopy(self.initial_pose_state[key]),
                "aabb": bounds,
                "support_plane_z": support_plane_z,
            }
            if self.dynamics[key] == "articulated_fixed":
                qpos = np.asarray(actor.get_qpos(), dtype=float)
                joint_index = self.articulation_joint_indices[key]
                self.initial_articulations[key] = float(qpos[joint_index])

    def descriptor_frame(self) -> dict[str, Any]:
        self._update_render()
        self.cameras.update_picture()
        rgb = self.cameras.get_rgb()["head_camera"]["rgb"]
        depth_mm = self.cameras.get_depth()["head_camera"]["depth"]
        segmentation = self.cameras.get_segmentation_ids(
            camera_name="head_camera", level="actor"
        )
        ids: dict[str, list[int]] = {}
        for key, wrapper in self.entities.items():
            raw = wrapper.actor
            if self.dynamics[key] == "articulated_fixed":
                values = [
                    item
                    for link in raw.get_links()
                    for item in _per_scene_ids(getattr(link, "entity", link))
                ]
            else:
                values = _per_scene_ids(raw)
            if not values:
                raise RuntimeError(f"actor_segmentation_id_missing:{key}")
            ids[key] = values
        return {
            "rgb": rgb,
            "depth": np.asarray(depth_mm, dtype=float) / 1000.0,
            "actor_segmentation": segmentation,
            "entity_actor_ids": ids,
            "world_relations": self._initial_world_relations(),
        }

    def _annotated_rollout_frame(self, rgb: np.ndarray | None = None) -> np.ndarray:
        if rgb is None:
            self._update_render()
            if os.environ.get("M15_SELECTIVE_CAMERA_CAPTURE", "1") != "0":
                from if_benchmark.scene_validation.expert_capture import capture_selected_rgb

                rgb = capture_selected_rgb(self.cameras, ["head_camera"])["head_camera"]
            else:
                self.cameras.update_picture()
                rgb = np.asarray(
                    self.cameras.get_rgb()["head_camera"]["rgb"],
                    dtype=np.uint8,
                )
        from PIL import Image, ImageDraw

        image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(image, "RGBA")
        width, height = image.size
        draw.rectangle((0, 0, width, 30), fill=(0, 0, 0, 185))
        draw.rectangle(
            (0, height - 20, width, height), fill=(0, 0, 0, 185)
        )
        draw.text(
            (5, 4),
            self._rollout_video_label[:68],
            fill=(255, 255, 255, 255),
        )
        draw.text(
            (5, height - 17),
            f"stage={self._rollout_video_stage}"[:68],
            fill=(255, 226, 96, 255),
        )
        return np.asarray(image, dtype=np.uint8)

    def _capture_rollout_video_frame(self, *, repeat: int = 1) -> None:
        if self._rollout_video_writer is None:
            return
        if getattr(self, "_m15_expert_review_active", False):
            # Fulfill review requests from the next real 25 Hz observation
            # (or genuine terminal frame), at most one sample interval later.
            # Coalesce requests between samples; never reuse a stale RGB frame.
            self._m15_expert_review_pending = max(self._m15_expert_review_pending, max(1, int(repeat)))
            return
        frame = self._annotated_rollout_frame()
        for _ in range(max(1, int(repeat))):
            self._rollout_video_writer.write(frame)

    def _record_expert_review_frame(self, images: Mapping[str, np.ndarray], step: int) -> None:
        writer = self._rollout_video_writer
        repeat = self._m15_expert_review_pending
        if writer is None or not repeat:
            return
        first = writer.frame_count
        frame = self._annotated_rollout_frame(rgb=images["head_camera"])
        for _ in range(repeat):
            writer.write(frame)
        self._m15_expert_review_sources.append({
            "first_review_frame": first, "frame_count": repeat,
            "expert_step_index": int(step), "stage": self._rollout_video_stage,
        })
        self._m15_expert_review_pending = 0

    def _take_picture(self) -> None:
        if self._rollout_video_writer is not None:
            self._capture_rollout_video_frame()
            return
        super()._take_picture()

    def begin_rollout_video(
        self,
        path: Path,
        *,
        label: str,
        fps: int = 8,
        sim_step_stride: int = 100,
    ) -> None:
        if self._rollout_video_writer is not None:
            raise RuntimeError("rollout_video_already_active")
        # The frozen descriptor sidecar is captured before video starts with
        # BenchmarkV1's normal 32-spp render.  Rollout video is diagnostic and
        # needs temporal coverage, so use a lightweight RT profile for frames
        # without changing physics, camera pose, or checker state.
        sapien.render.set_ray_tracing_samples_per_pixel(1)
        sapien.render.set_ray_tracing_path_depth(2)
        camera = self.scene_spec["camera"]
        self._rollout_video_label = str(label)
        self._rollout_video_stage = "initial"
        self._rollout_video_stride = max(1, int(sim_step_stride))
        self._rollout_video_previous_save_freq = self.save_freq
        self._m15_expert_review_sources = []
        self.save_freq = self._rollout_video_stride
        self._rollout_video_writer = _RolloutVideoWriter(
            path,
            width=int(camera["width"]),
            height=int(camera["height"]),
            fps=int(fps),
        )
        self._capture_rollout_video_frame(repeat=max(1, int(fps) // 2))

    def finish_rollout_video(self, *, outcome: str) -> dict[str, Any]:
        writer = self._rollout_video_writer
        if writer is None:
            raise RuntimeError("rollout_video_not_active")
        self._rollout_video_stage = str(outcome)
        try:
            try:
                self._capture_rollout_video_frame(repeat=max(1, writer.fps))
            except BaseException:
                # A failed batch renderer must remain fail-closed, but its
                # review encoder still needs teardown before reusing the lane.
                try:
                    writer.close()
                except BaseException:
                    pass  # Preserve the primary rendering/interrupt failure.
                raise
            else:
                frame_count = writer.close()
        finally:
            self._rollout_video_writer = None
            self.save_freq = self._rollout_video_previous_save_freq
            self._rollout_video_previous_save_freq = None
        value = _sidecar(writer.path)
        value.update(
            {
                "view": "head_camera",
                "fps": writer.fps,
                "frame_count": frame_count,
                "duration_sec": frame_count / float(writer.fps),
            }
        )
        if self._m15_expert_review_sources:
            value["expert_rgb_review"] = {
                "backend": "expert_rgb_requests_v1",
                "camera": "head_camera",
                "sample_alignment": "next captured expert observation; requests between observations coalesce",
                "frames": deepcopy(self._m15_expert_review_sources),
            }
        return value

    def _capture_settle_step(self, step_index: int) -> None:
        if (
            self._rollout_video_writer is not None
            and int(step_index) % self._rollout_video_stride == 0
        ):
            self._capture_rollout_video_frame()

    def _initial_world_relations(self) -> dict[str, list[list[str]]]:
        result: dict[str, list[list[str]]] = {"in": [], "on": []}
        table_value = getattr(self, "table", None)
        table_entity = getattr(table_value, "actor", table_value)
        keys = sorted(self.entities)
        for subject in keys:
            for target in keys:
                if subject == target:
                    continue
                subject_actor = self.entities[subject]
                target_actor = self.entities[target]
                pair_contact = contact_metrics(
                    self.scene, subject_actor.actor, target_actor.actor
                )
                table_contact = (
                    contact_metrics(
                        self.scene, subject_actor.actor, table_entity
                    )
                    if table_entity is not None
                    else {"active": False}
                )
                contacts = {
                    "subject_receiver": bool(pair_contact.get("active")),
                    "subject_table": bool(table_contact.get("active")),
                }
                stability = {"stable": True}
                subject_state = {
                    "aabb": collision_aabb_tight(subject_actor)
                }
                target_state = {"aabb": collision_aabb_tight(target_actor)}
                inside = place_in_success_metrics(
                    subject_state, target_state, contacts, stability
                )
                on = place_on_success_metrics(
                    subject_state, target_state, contacts, stability
                )
                if inside["success"]:
                    result["in"].append([subject, target])
                if on["success"]:
                    result["on"].append([subject, target])
        return result

    def execute_skill_config(
        self,
        skill_config: Mapping[str, Any],
        *,
        checker_config: Mapping[str, Any] | None = None,
        route_report: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.operation_anchor_preflight = preflight_skill_operation_anchors(
            skill_config,
            self.unit_artifact,
            self.joined["realized"],
            self.scene_spec,
        )
        self.current_checker_config = checker_config or {}
        self.current_skill_config = skill_config
        self.current_route_report = (
            None if route_report is None else deepcopy(dict(route_report))
        )
        self.pick_planning_witnesses = {}
        self.pick_planning_requests = {}
        try:
            self.pick_planning_certificate = self._preflight_pick_planning(
                skill_config
            )
        except TimeoutError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            self.pick_planning_certificate = {
                "passed": False,
                "required": True,
                "planning_only": True,
                "disposition": "source_failure",
                "failure_code": "planning_pick_frozen_contract_invalid:grasp",
                "results": {},
                "failures": [
                    {
                        "subject": None,
                        "arm": "all",
                        "failure_reason": "frozen_contract_invalid",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                ],
            }
        except Exception as exc:
            self.pick_planning_certificate = {
                "passed": False,
                "required": True,
                "planning_only": True,
                "disposition": "infrastructure_failure",
                "failure_code": "planning_pick_preflight_infrastructure_failed:grasp",
                "results": {},
                "failures": [
                    {
                        "subject": None,
                        "arm": "all",
                        "failure_reason": "planning_infrastructure_failure",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                ],
            }
        if not bool(self.pick_planning_certificate.get("passed", True)):
            failure = self.pick_planning_certificate["failures"][0]
            return {
                "completed": False,
                "failure_code": str(
                    self.pick_planning_certificate.get("failure_code")
                    or (
                        "planning_pick_unreachable_"
                        f"{failure['arm']}_{failure['failure_reason']}:grasp"
                    )
                ),
                "steps": deepcopy(self.skill_diagnostics),
                "pick_planning_certificate": strip_runtime_arrays(
                    self.pick_planning_certificate
                ),
            }

        def dispatch(op: str, step: Mapping[str, Any]) -> None:
            handler = getattr(self, f"_skill_{op}", None)
            if handler is None:
                raise RuntimeError(f"unsupported_skill_operator:{op}")
            self._rollout_video_stage = str(op)
            self._capture_rollout_video_frame()
            try:
                handler(step)
            finally:
                self._capture_rollout_video_frame()
            self.skill_diagnostics.append(
                {
                    "op": op,
                    "plan_success": bool(self.plan_success),
                    "step": deepcopy(dict(step)),
                }
            )
            if not self.plan_success:
                raise _SkillPlanFailure(op)

        try:
            dispatch_skill_config(skill_config, dispatch)
        except _SkillPlanFailure as exc:
            return {
                "completed": False,
                "failure_code": f"planning_failed:{exc.op}",
                "steps": deepcopy(self.skill_diagnostics),
                "pick_planning_certificate": strip_runtime_arrays(
                    self.pick_planning_certificate
                ),
            }
        except _SkillExecutionFailure as exc:
            return {
                "completed": False,
                "failure_code": f"{exc.code}:{exc.op}",
                "steps": deepcopy(self.skill_diagnostics),
                "pick_planning_certificate": strip_runtime_arrays(
                    self.pick_planning_certificate
                ),
            }
        return {
            "completed": bool(self.plan_success),
            "failure_code": None if self.plan_success else "planning_failed",
            "steps": deepcopy(self.skill_diagnostics),
            "pick_planning_certificate": strip_runtime_arrays(
                self.pick_planning_certificate
            ),
        }

    def _preflight_pick_planning(
        self,
        skill_config: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Re-prove this task's frozen M12 Pick target before physical action."""

        frozen = validated_scene_pick_planning_contract(
            self.scene_artifact,
            realized_artifact=self.joined["realized"],
            unit_artifact=self.unit_artifact,
        )

        grasp_steps: dict[str, Mapping[str, Any]] = {}
        for step in skill_config.get("steps", []):
            if str(step.get("op") or "") != "grasp":
                continue
            subject = str(step.get("subject") or "")
            arm = str(step.get("arm") or "")
            key = f"{subject}:{arm}"
            if not subject or arm not in {"left", "right"}:
                raise ValueError("Pick skill has an invalid grasp target")
            if key in grasp_steps:
                raise ValueError(f"Pick skill contains duplicate grasp {key}")
            grasp_steps[key] = step
        if not frozen["required"] or not grasp_steps:
            return {
                "passed": True,
                "required": False,
                "scene_contract_required": bool(frozen["required"]),
                "planning_only": True,
                "frozen_request_hash": (
                    None
                    if frozen["request"] is None
                    else frozen["request"]["request_hash"]
                ),
                "frozen_witness_hash": (
                    None
                    if frozen["witness"] is None
                    else frozen["witness"]["witness_hash"]
                ),
                "results": {},
                "failures": [],
            }

        frozen_request = frozen["request"]
        frozen_witness = frozen["witness"]
        assert isinstance(frozen_request, Mapping)
        assert isinstance(frozen_witness, Mapping)
        targets_by_key = {
            f"{target['entity_key']}:{target['arm']}": target
            for target in frozen_request["targets"]
        }
        required_target_keys = set(frozen["required_target_keys"])
        passed_target_keys = set(frozen["passed_target_keys"])

        # M12 records every required/candidate target-arm pair in the complete
        # SceneSpec.  M13 may select only a passing arm, and one M15 TaskRecord
        # consumes only the exact passing pair it will execute.
        exact_steps = []
        requests: list[PickAcquisitionRequest] = []
        frozen_requests_by_key: dict[str, PickAcquisitionRequest] = {}
        selected_route = (
            (self.current_route_report or {}).get("selected_witness") or {}
        )
        for key, source_step in sorted(grasp_steps.items()):
            if key not in required_target_keys:
                raise ValueError(f"frozen Pick contract does not require {key}")
            if key not in passed_target_keys:
                raise ValueError(f"frozen Pick witness did not pass {key}")
            target = targets_by_key.get(key)
            if not isinstance(target, Mapping):
                raise ValueError(f"frozen Pick request omits {key}")
            request = pick_acquisition_request_from_record(
                target["planning_request"]
            )
            if f"{request.subject}:{request.arm}" != key:
                raise ValueError(f"frozen Pick request identity mismatch for {key}")
            frozen_requests_by_key[key] = request
            if selected_route:
                route_subject = str(
                    (self.current_route_report or {}).get("subject") or ""
                )
                route_arm = str(
                    (self.current_route_report or {}).get("arm") or ""
                )
                if request.subject == route_subject and request.arm == route_arm:
                    contact_id = int(selected_route.get("contact_id", -1))
                    permitted = {
                        int(value) for value in (request.contact_ids or ())
                    }
                    if contact_id not in permitted:
                        raise ValueError(
                            "selected route Contact is not permitted by M12 Pick request"
                        )
                    if str(selected_route.get("contact_contract_hash") or "") != str(
                        target["contact_contract_binding"][
                            "contact_contract_hash"
                        ]
                    ):
                        raise ValueError(
                            "selected route Contact contract differs from M12"
                        )
                    request = replace(
                        request,
                        contact_ids=(contact_id,),
                        priority_contact_ids=tuple(
                            value for value in (request.priority_contact_ids or ())
                            if int(value) == contact_id
                        ),
                    )
            exact_steps.append(deepcopy(dict(source_step)))
            requests.append(request)
        self.pick_operation_anchor_preflight = (
            preflight_skill_operation_anchors(
                {**deepcopy(dict(skill_config)), "steps": exact_steps},
                self.unit_artifact,
                self.joined["realized"],
            )
        )
        check_by_key = {
            f"{row['subject']}:{'left' if row['contact_validation_use'] == 'pick_left' else 'right'}": row
            for row in self.pick_operation_anchor_preflight.get("checks", [])
            if row.get("op") == "grasp"
            and row.get("contact_validation_use")
            in {"pick_left", "pick_right"}
        }
        for request in requests:
            key = f"{request.subject}:{request.arm}"
            target = targets_by_key[key]
            check = check_by_key.get(key)
            if not isinstance(check, Mapping):
                raise ValueError(f"runtime lineage omits frozen Pick target {key}")
            if (
                target["validated_use"] != check.get("contact_validation_use")
                or target["operation_anchor_contract_hash"]
                != check.get("contract_hash")
                or target["contact_contract_binding"]
                != check.get("contact_contract_binding")
                or target["entity_pose"]
                != self.scene_spec["entities"][request.subject]["pose"]
            ):
                raise ValueError(f"runtime lineage differs from frozen Pick target {key}")

        certificate = certify_pick_acquisition_plans(self, requests)
        for request in requests:
            key = f"{request.subject}:{request.arm}"
            target = targets_by_key[key]
            result = certificate["results"][key]
            expected_request = pick_planning_request_record(request)
            route_constrained = request != frozen_requests_by_key[key]
            if (
                result.get("request") != expected_request
                or result.get("request_hash")
                != artifact_hash(expected_request)
            ):
                raise ValueError(
                    f"live Pick oracle request differs from selected target {key}"
                )
            if not route_constrained and (
                expected_request != target["planning_request"]
                or artifact_hash(expected_request)
                != target["planning_request_hash"]
            ):
                raise ValueError(
                    f"live Pick oracle request differs from frozen target {key}"
                )
            compiled = result.get("compiled_plan")
            if bool(result.get("plannable")) and (
                not isinstance(compiled, Mapping)
                or compiled.get("contact_contract_hash")
                != target["contact_contract_binding"]["contact_contract_hash"]
            ):
                raise ValueError(
                    f"live Pick oracle Contact contract differs for {key}"
                )
            if route_constrained and bool(result.get("plannable")) and int(
                compiled.get("contact_id", -1)
            ) != int(selected_route["contact_id"]):
                raise ValueError(
                    f"live Pick oracle changed selected route Contact for {key}"
                )
        certificate["required"] = True
        certificate["scene_contract_required"] = True
        certificate["frozen_request_hash"] = frozen_request["request_hash"]
        certificate["frozen_witness_hash"] = frozen_witness["witness_hash"]
        certificate["planned_target_keys"] = sorted(certificate["results"])
        certificate["operation_anchor_preflight"] = deepcopy(
            self.pick_operation_anchor_preflight
        )
        certificate["route_witness"] = (
            None
            if self.current_route_report is None
            else {
                "report_hash": self.current_route_report["report_hash"],
                "witness_hash": selected_route["witness_hash"],
                "contact_id": int(selected_route["contact_id"]),
            }
        )
        if not certificate["passed"]:
            classification = classify_pick_planning_failure(
                {
                    "results": [
                        {**row, "status": "failed"}
                        for row in certificate["failures"]
                    ]
                }
            )
            certificate["failure_classification"] = classification
            certificate["disposition"] = classification["disposition"]
            certificate["failure_code"] = {
                "infrastructure_failure": (
                    "planning_pick_preflight_infrastructure_failed:grasp"
                ),
                "source_failure": "planning_pick_preflight_source_invalid:grasp",
                "contract_failure": (
                    "planning_pick_preflight_contract_invalid:grasp"
                ),
            }.get(str(classification["disposition"]))
        self._record_skill_diagnostic(
            op="pick_planning_preflight",
            certificate=strip_runtime_arrays(certificate),
        )
        if certificate["passed"]:
            self.pick_planning_witnesses = {
                key: result
                for key, result in certificate["results"].items()
            }
            self.pick_planning_requests = {
                f"{request.subject}:{request.arm}": request
                for request in requests
            }
        return certificate

    def _arm(self, value: Any) -> ArmTag:
        return ArmTag(str(value))

    def _record_skill_diagnostic(self, **values: Any) -> None:
        self.skill_diagnostics.append(deepcopy(values))

    def _record_primary_action(self, arm: ArmTag, op: str) -> None:
        arm_name = str(arm)
        self.participating_arms.add(arm_name)
        if self.primary_execution_arm is None:
            self.primary_execution_arm = arm_name
            self.primary_execution_op = str(op)

    def _reset_planning(self) -> None:
        self.plan_success = True

    def _arm_ee_pose(self, arm: ArmTag) -> np.ndarray:
        return np.asarray(self.get_arm_pose(arm), dtype=float)

    def choose_best_pose(
        self,
        res_pose: Sequence[float],
        center_pose: Sequence[float],
        arm_tag: ArmTag | None = None,
    ) -> list[float] | None:
        """Preserve the exact authored Contact orientation in BenchmarkV1.

        Rigid table acquisition is planned by the R02 candidate engine.  This
        override also prevents the remaining articulated helpers from
        expanding an authored Contact into the legacy ten-pose arc.
        """

        del center_pose
        if not self.plan_success:
            return [-1.0] * 7
        if arm_tag is None:
            raise RuntimeError("grasp_pose_arm_missing")
        values = [float(value) for value in res_pose]
        if len(values) != 7:
            raise RuntimeError("exact_contact_pose_invalid")
        self._last_grasp_pose_selection = {
            "arm": str(arm_tag),
            "candidate_count": 1,
            "reachable_count": None,
            "selection_reason": "authored_exact_contact_no_synthetic_roll",
            "policy_version": EXACT_GRASP_POLICY["version"],
        }
        return values

    def _actor_position(self, subject_key: str) -> np.ndarray:
        return np.asarray(
            self.entities[subject_key].get_pose().p,
            dtype=float,
        )

    def _runtime_success_contract(self) -> SuccessContract:
        return self._observation_sources.success_contract()

    def _runtime_embodiment_profile(self) -> dict[str, Any]:
        return self._observation_sources.embodiment_profile()

    def _success_active_contacts(
        self,
        profile: Mapping[str, Any],
        grippers: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        """Read and classify current PhysX manifolds without changing state."""

        endpoint_by_name: dict[str, str] = {}
        for key, wrapper in self.entities.items():
            endpoint_by_name[str(wrapper.get_name())] = key
            raw = wrapper.actor
            if self.dynamics[key] == "articulated_fixed":
                for link in raw.get_links():
                    endpoint_by_name[str(link.get_name())] = key
        table_value = getattr(self, "table", None)
        table_entity = getattr(table_value, "actor", table_value)
        if table_entity is not None:
            endpoint_by_name[str(table_entity.get_name())] = "support:table"
        for arm, arm_profile in (profile.get("arms") or {}).items():
            for row in arm_profile.get("fingers") or []:
                endpoint_by_name[str(row["link"])] = f"gripper:{arm}"
            for name in arm_profile.get("palm_wrist_links") or []:
                endpoint_by_name[str(name)] = f"gripper:{arm}"
        observer = (
            observe_current_sapien_contacts
            if getattr(self, "_m15_single_pass_contacts", False)
            else observe_sapien_active_contacts
        )
        return observer(
            self.scene,
            endpoint_by_body_name=endpoint_by_name,
            embodiment_profile=profile,
            grippers=grippers,
            contract=self._runtime_success_contract(),
        )

    def _pose_matrix(self, subject_key: str) -> np.ndarray:
        pose = self.entities[subject_key].get_pose()
        return np.asarray(pose.to_transformation_matrix(), dtype=float)

    def _ee_matrix(self, arm: ArmTag) -> np.ndarray:
        pose = self._arm_ee_pose(arm)
        return np.asarray(
            sapien.Pose(pose[:3], pose[3:]).to_transformation_matrix(),
            dtype=float,
        )

    def _hold_ee_matrix(self, arm: ArmTag) -> np.ndarray:
        return physical_ee_world_matrix(self.robot, str(arm))

    def _hold_symmetry_axis(self, subject_key: str) -> list[float] | None:
        asset = str(self.scene_spec["entities"][subject_key].get("asset") or "")
        configured = self._runtime_success_contract().section("hold").get(
            "reviewed_symmetry_axes", {}
        )
        axis_name = (
            str(configured.get(asset))
            if isinstance(configured, Mapping) and asset in configured
            else None
        )
        return {
            "x": [1.0, 0.0, 0.0],
            "y": [0.0, 1.0, 0.0],
            "z": [0.0, 0.0, 1.0],
        }.get(axis_name)

    def _freeze_hold_reference(self, subject_key: str, arm: ArmTag) -> None:
        arm_name = str(arm)
        if arm_name in self.hold_references:
            raise RuntimeError("immutable_hold_reference_already_frozen")
        profile = self._runtime_embodiment_profile()
        gripper = observe_actual_grippers(
            self.robot, embodiment_profile=profile
        )[arm_name]
        self.hold_references[arm_name] = build_hold_reference(
            subject=subject_key,
            arm=arm_name,
            actor_matrix=self._pose_matrix(subject_key),
            ee_matrix=self._hold_ee_matrix(arm),
            planning_ee_matrix=self._ee_matrix(arm),
            actual_width_m=float(gripper["actual_width_m"]),
            ee_frame=HOLD_EE_FRAME_ARM,
        )

    def _prepare_held_collision(self, subject_key: str, arm: ArmTag) -> None:
        arm_name = str(arm)
        binding = bind_held_collision(self, self.entities[subject_key], arm_name,
                                      name=f"held:{subject_key}")
        self.hold_references[arm_name].update(
            collision_world_hash=binding["world"][arm_name]["world_hash"],
            attachment_hash=binding["attachment"]["attachment_hash"],
            carry_collision_binding=deepcopy(binding["attachment"]),
            held_collision_policy=HELD_COLLISION_POLICY,
        )

    def _create_pick_attachment(
        self, subject_key: str, arm: ArmTag
    ) -> dict[str, Any]:
        """Weld one verified closed grasp to the physical EE without snapping."""

        arm_name = str(arm)
        self._remove_pick_attachment(arm)
        actor = self.entities[subject_key]
        attachment, result = create_evaluated_pick_attachment(
            self,
            actor,
            arm=arm_name,
        )
        if attachment is not None:
            self.pick_attachments[arm_name] = attachment
        return result

    def _m15_live_table_collision_obstacles(self) -> list[dict[str, Any]]:
        return live_table_obstacles(self)

    def _prepare_m15_carry_collision(
        self, subject_key: str, arm: ArmTag
    ) -> None:
        """Admit measured Carry entry using the original physical closure cover."""

        if not getattr(self, "_m15_corrected_carry_collision", False):
            return
        arm_name = str(arm)
        reference = self.hold_references.get(arm_name)
        certification = self.pick_certifications.get(arm_name) or {}
        if (self.held_entities.get(arm_name) != subject_key
                or certification.get("subject") != subject_key
                or not isinstance(reference, dict)
                or reference.get("subject") != subject_key):
            raise RuntimeError("certified_carry_collision_subject_mismatch")
        previous = {
            "world_hash": reference.get("collision_world_hash"),
            "attachment_hash": reference.get("attachment_hash"),
        }
        if reference.get("held_collision_policy") != HELD_COLLISION_POLICY:
            raise RuntimeError("certified_carry_collision_policy_missing")
        attached = reference.get("carry_collision_binding") or {}
        if (attached.get("attachment_hash") != reference.get("attachment_hash")
                or attached.get("collision_geometry_mode") != "m15_carry_conservative_v1"
                or not attached.get("cover_version") or not attached.get("sphere_tensor_sha256")
                or not isinstance(attached.get("coverage_certificate"), Mapping)
                or not attached["coverage_certificate"]):
            raise RuntimeError("certified_carry_collision_binding_missing")
        # The physical acquisition already used this full cover. Keep the
        # original closure-to-EE relation rather than constructing a new one.
        world = refresh_held_world(self, self.entities[subject_key], arm_name,
                                   require_existing_binding=True)
        reference["collision_world_hash"] = world[arm_name]["world_hash"]
        handoff = validate_held_handoff(self, self.entities[subject_key], arm_name, reference)
        reference["held_handoff_validation"] = handoff
        if not handoff["collision_free"]:
            raise _SkillExecutionFailure("carry_to_relation", "held_handoff_collision")
        self._record_skill_diagnostic(
            op="m15_carry_collision_continued", subject=subject_key, arm=arm_name,
            previous_binding=previous, collision_world=deepcopy(world),
            attachment=deepcopy(attached), held_collision_policy=HELD_COLLISION_POLICY,
            held_handoff_validation=deepcopy(handoff),
            attachment_rebound=False,
        )

    def _observe_pick_attachment(self, arm: ArmTag) -> dict[str, Any]:
        attachment = self.pick_attachments.get(str(arm))
        if not isinstance(attachment, SapienFixedAttachment):
            return evaluate_pick_attachment({}, contract=self._runtime_success_contract())
        return evaluate_pick_attachment(
            attachment.metrics(), contract=self._runtime_success_contract()
        )

    def _remove_pick_attachment(self, arm: ArmTag) -> None:
        attachment = self.pick_attachments.pop(str(arm), None)
        if isinstance(attachment, SapienFixedAttachment):
            attachment.remove()

    def _move_with_held_collision(
        self,
        subject_key: str,
        arm: ArmTag,
        motion: tuple[Any, list[Any]],
        *,
        excluded_entity_keys: Sequence[str] = (),
        m15_exact_table_validation: bool = False,
        carry_target_pose=None,
        carry_motion=None,
    ) -> None:
        arm_name = str(arm)
        reference = self.hold_references[arm_name]
        excluded = {subject_key, *[str(value) for value in excluded_entity_keys]}
        obstacles = route_scene_collision_obstacles(
            self.scene_spec,
            self.entities,
            excluded_entity_keys=sorted(excluded),
            **({"complete_geometry": True,
                "max_obstacles": 64 - len(self._m15_live_table_collision_obstacles())}
               if reference.get("carry_collision_binding") else {}),
        )
        world = self.robot.update_world_obstacles(obstacles)
        if not isinstance(world, Mapping) or not bool(world.get("updated")):
            raise RuntimeError("held collision world update was not acknowledged")
        world_hash = str((world.get(arm_name) or {}).get("world_hash") or "")
        reference["collision_world_hash"] = world_hash
        for action in motion[1]:
            action.args.update(
                {
                    "expected_world_hash": world_hash,
                    "expected_attachment_hash": reference["attachment_hash"],
                }
            )
            if m15_exact_table_validation or reference.get("carry_collision_binding"):
                action.args["m15_exact_table_validation"] = True
        def check_actual_camera(stage):
            if reference.get("held_collision_policy") != HELD_COLLISION_POLICY:
                return
            pair = observe_live_held_camera(self, self.entities[subject_key], arm_name, reference)
            self._record_skill_diagnostic(op="held_camera_pose_check", subject=subject_key,
                arm=arm_name, stage=stage, fixed_camera_pair=pair)
            if not pair["passed"]:
                self.plan_success = False
                raise _SkillExecutionFailure("carry_to_relation", "held_actual_camera_collision")
        check_actual_camera("before_held_motion")
        if carry_target_pose is not None:
            q = np.asarray(getattr(self.robot, f"get_{arm_name}_arm_real_jointState")(), dtype=float)
            q = q[:len(getattr(self.robot, f"{arm_name}_arm_joints"))]
            result = plan_held_path(self, self.entities[subject_key], arm_name, q, carry_target_pose,
                world_hash=world_hash, attachment_hash=reference["attachment_hash"],
                carry_motion=carry_motion,
                excluded_entities=tuple(self.entities[key] for key in sorted(excluded - {subject_key})))
            getattr(self, f"{arm_name}_joint_path").append(result)
            self.plan_success = result.get("status") == "Success"
            if self.plan_success and not (result.get("carry_motion") or {}).get("motion_skipped", False):
                control = {"left_arm": None, "left_gripper": None, "right_arm": None, "right_gripper": None}
                control[f"{arm_name}_arm"] = result
                self.take_dense_action(control)
        else:
            self.move(motion)
        check_actual_camera("after_held_motion")

    def _observe_hold(
        self,
        subject_key: str,
        arm: ArmTag,
        *,
        steps: int,
        phase: str = "carry",
    ) -> dict[str, Any]:
        arm_name = str(arm)
        reference = self.hold_references.get(arm_name)
        if not isinstance(reference, Mapping):
            raise RuntimeError("immutable_hold_reference_missing")
        profile = self._runtime_embodiment_profile()
        samples: list[dict[str, Any]] = []
        pose_history: list[list[dict[str, Any]]] = []
        for _ in range(int(steps)):
            self.scene.step()
            grippers = observe_actual_grippers(
                self.robot, embodiment_profile=profile
            )
            contacts = self._success_active_contacts(profile, grippers)
            actor_pose = self.entities[subject_key].get_pose()
            samples.append(
                {
                    "gripper": grippers[arm_name],
                    "active_contacts": contacts,
                    "actor_matrix": self._pose_matrix(subject_key),
                    "ee_matrix": self._hold_ee_matrix(arm),
                    "ee_frame": HOLD_EE_FRAME_ARM,
                }
            )
            pose_history.append(
                [{"p": actor_pose.p.tolist(), "q": actor_pose.q.tolist()}]
            )
        summary = summarize_hold_observation(
            samples,
            reference=reference,
            subject=subject_key,
            arm=arm_name,
            embodiment_profile=profile,
            symmetry_axis=self._hold_symmetry_axis(subject_key),
        )
        attachment = self.pick_attachments.get(arm_name)
        certified = reference.get("certified_closure_evidence")
        if isinstance(attachment, SapienFixedAttachment):
            attachment_result = self._observe_pick_attachment(arm)
            if attachment_result["passed"] and isinstance(certified, Mapping):
                summary["gripper"]["closure_evidence"] = {
                    **deepcopy(dict(certified)),
                    "path": "post_close_fixed_attachment",
                    "actual_width_m": float(
                        summary["gripper"]["actual_width_m"]
                    ),
                }
                summary["gripper"]["post_close_attachment"] = {
                    **deepcopy(attachment_result),
                    "subject": subject_key,
                    "certified_close": True,
                }
        if phase == "pick":
            summary["stability"] = stability_from_snapshots(
                pose_history, rotation_invariant=True
            )
        else:
            summary["stability"] = self._apply_axisymmetric_stability(
                stability_from_snapshots(pose_history),
                pose_history,
                [
                    (
                        subject_key,
                        "root",
                        self.entities[subject_key].actor,
                    )
                ],
            )
        summary["profile"] = profile
        return summary

    def _hold_failure_code(
        self, summary: Mapping[str, Any], *, phase: str
    ) -> str | None:
        arm_name = str((summary.get("gripper") or {}).get("arm") or "")
        if arm_name in self.pick_attachments:
            attachment = self._observe_pick_attachment(self._arm(arm_name))
            if not attachment["passed"]:
                return str(attachment["failure_code"])
        return hold_failure_code(
            summary, phase=phase, contract=self._runtime_success_contract()
        )

    def _held_actor_target_ee_pose(
        self,
        subject_key: str,
        arm: ArmTag,
        actor_target_pose: Sequence[float],
    ) -> list[float]:
        """Apply the live verified grasp transform to one actor target pose."""

        reference = self.hold_references.get(str(arm)) or {}
        if reference.get("subject") != subject_key:
            raise RuntimeError("verified_grasp_transform_missing")
        try:
            return held_actor_target_to_planning_ee_pose(
                reference, actor_target_pose
            )
        except SuccessObservationError as exc:
            raise RuntimeError("verified_grasp_transform_missing") from exc

    def _contact_count(self, actor: Any) -> int:
        key = actor.POINTS["contact"]
        return len((actor.config or {}).get(key, []))

    def _release_for_retry(self, arm: ArmTag) -> None:
        try:
            self._remove_pick_attachment(arm)
            self.robot.detach_collision_object(str(arm))
            self.hold_references.pop(str(arm), None)
            self.pick_certifications.pop(str(arm), None)
            self.held_entities.pop(str(arm), None)
            self._reset_planning()
            self.move(self.open_gripper(arm))
            for _ in range(12):
                self.scene.step()
            self._reset_planning()
            self.move(self.move_by_displacement(arm, z=0.05))
            self._reset_planning()
        finally:
            self.held_arm_drive_leases.release_arm(str(arm))

    def _certify_pick(
        self,
        subject_key: str,
        arm: ArmTag,
        summary: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        contract = self._runtime_success_contract()
        profile = summary["profile"]
        grippers = observe_actual_grippers(
            self.robot, embodiment_profile=profile
        )
        grippers[str(arm)] = deepcopy(summary["gripper"])
        bounds = collision_aabb_tight(self.entities[subject_key])
        initial_entity = deepcopy(self.initial_success_entities[subject_key])
        terminal_entity = {
            "pose": {
                "position": self.entities[subject_key].get_pose().p.tolist(),
                "quat_wxyz": self.entities[subject_key].get_pose().q.tolist(),
            },
            "aabb": bounds,
            "support_plane_z": float(initial_entity["support_plane_z"]),
        }
        snapshot = build_success_snapshot(
            initial={"entities": {subject_key: initial_entity}},
            terminal={
                "entities": {subject_key: terminal_entity},
                "grippers": grippers,
                "active_contacts": deepcopy(summary["active_contacts"]),
                "action_provenance": {
                    "primary_arm": str(arm),
                    "participating_arms": [str(arm)],
                },
            },
            stability={
                "entities": {subject_key: deepcopy(summary["stability"])},
                "whole_scene": {
                    "stable": bool(summary["stability"].get("stable"))
                },
            },
            execution_events={},
            source_hashes={
                "joined_scene": artifact_hash(self.joined),
                "pick_acquisition_policy": artifact_hash(
                    {
                        "version": PICK_ACQUISITION_POLICY_VERSION,
                        "tracking_guard_m": (
                            PICK_ACQUISITION_TRACKING_GUARD_M
                        ),
                    }
                ),
            },
            contract=contract,
            profiles={"embodiment": profile},
        )
        result = evaluate_m15_success(
            "pick_success",
            {"subject": subject_key, "requested_arm": str(arm)},
            snapshot,
            contract,
        )
        return result, snapshot

    def _unified_subject_actor(self, subject_key: str) -> Any:
        return self.entities[subject_key]

    def _unified_contact_count(self, subject_key: str) -> int:
        return self._contact_count(self.entities[subject_key])

    def _unified_configure_pick_dynamics(
        self,
        subject_key: str,
        *,
        linear_damping: float,
        angular_damping: float,
    ) -> dict[str, Any]:
        actor = self.entities[subject_key]
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
        self._record_skill_diagnostic(
            op="configure_pick_dynamics",
            subject=subject_key,
            **result,
        )
        return result

    def _unified_reset_planning(self) -> None:
        self._reset_planning()

    def _unified_record_grasp_plan(self, planning: Mapping[str, Any]) -> None:
        self._record_skill_diagnostic(
            op="grasp_planning_pass", planning=deepcopy(dict(planning))
        )

    def _unified_record_grasp_attempt(self, attempt: Mapping[str, Any]) -> None:
        self._record_skill_diagnostic(
            op="grasp_candidate", **deepcopy(dict(attempt))
        )

    def _unified_freeze_hold_reference(
        self, subject_key: str, arm: str
    ) -> None:
        self.hold_references.pop(str(arm), None)
        self._freeze_hold_reference(subject_key, self._arm(arm))

    def _unified_observe_hold(
        self, subject_key: str, arm: str, *, phase: str
    ) -> dict[str, Any]:
        continuity = int(
            self._runtime_embodiment_profile()[
                "classification"
            ]["continuous_contact_min_steps"]
        )
        return self._observe_hold(
            subject_key,
            self._arm(arm),
            steps=continuity,
            phase=phase,
        )

    def _unified_hold_failure(
        self, summary: Mapping[str, Any], *, phase: str
    ) -> str | None:
        return self._hold_failure_code(summary, phase=phase)

    def _unified_store_certified_closure(
        self, arm: str, evidence: Mapping[str, Any]
    ) -> None:
        self.hold_references[str(arm)]["certified_closure_evidence"] = (
            deepcopy(dict(evidence))
        )

    def _unified_create_attachment(
        self, subject_key: str, arm: str
    ) -> dict[str, Any]:
        return self._create_pick_attachment(subject_key, self._arm(arm))

    def _unified_prepare_held_collision(
        self, subject_key: str, arm: str
    ) -> None:
        self._prepare_held_collision(subject_key, self._arm(arm))

    def _unified_support_clearance(
        self,
        subject_key: str,
        *,
        support_plane_z_m: float | None = None,
    ) -> float:
        frozen_support = float(
            self.initial_success_entities[subject_key]["support_plane_z"]
        )
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
        return collision_support_clearance(
            self.entities[subject_key], frozen_support
        )

    def _unified_move_held_world_z(
        self, subject_key: str, arm: str, delta_m: float, *,
        terminal_settle_max_steps: int, terminal_joint_tolerance_rad: float,
    ) -> bool:
        self._reset_planning()
        return execute_held_lift(self, self.entities[subject_key], str(arm), delta_m, self.hold_references[str(arm)],
            support_plane_z_m=float(self.initial_success_entities[subject_key]["support_plane_z"]),
            required_clearance_m=PICK_ACQUISITION_REQUIRED_CLEARANCE_M,
            terminal_settle_max_steps=terminal_settle_max_steps,
            terminal_joint_tolerance_rad=terminal_joint_tolerance_rad)

    def _unified_observe_attachment(self, arm: str) -> dict[str, Any]:
        return self._observe_pick_attachment(self._arm(arm))

    def _unified_total_lift_delta(self, subject_key: str) -> float:
        return float(
            self._actor_position(subject_key)[2]
            - np.asarray(
                self.initial_pose_state[subject_key]["position"], dtype=float
            )[2]
        )

    def _unified_current_actor_pose(self, subject_key: str) -> list[float]:
        pose = self.entities[subject_key].get_pose()
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
        return self._held_actor_target_ee_pose(
            subject_key, self._arm(arm), actor_target_pose
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
        arm_tag = self._arm(arm)
        self._reset_planning()
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
        self._move_with_held_collision(subject_key, arm_tag, motion)
        return bool(self.plan_success)

    def _unified_certify_pick(
        self, subject_key: str, arm: str
    ) -> dict[str, Any]:
        arm_tag = self._arm(arm)
        certification_steps = int(
            self._runtime_success_contract().section("hold")[
                "pick_certification_observation_steps"
            ]
        )
        summary = self._observe_hold(
            subject_key,
            arm_tag,
            steps=certification_steps,
            phase="pick",
        )
        predicate, snapshot = self._certify_pick(
            subject_key, arm_tag, summary
        )
        failure_code = self._hold_failure_code(summary, phase="pick")
        if (
            failure_code is None
            and "support_clearance"
            in predicate.get("failed_predicates", [])
        ):
            failure_code = "pick_vertical_lift_unreachable"
        if failure_code is None and predicate["status"] != "passed":
            failure_code = "pick_success_failed"
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
        self.held_entities[str(arm)] = subject_key
        self.pick_certifications[str(arm)] = {
            "subject": subject_key,
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
        self.released_arms.discard(str(arm))
        self._record_primary_action(self._arm(arm), "grasp")
        self._reset_planning()

    def _unified_release_for_retry(
        self, subject_key: str, arm: str
    ) -> None:
        del subject_key
        self._release_for_retry(self._arm(arm))

    def build_pick_acquisition_request(
        self,
        subject_key: str,
        arm: str,
        *,
        requested_pre_grasp_m: float,
        contact_ids: Sequence[int] | None = None,
        priority_contact_ids: Sequence[int] | None = None,
    ) -> PickAcquisitionRequest:
        """Lower one SceneSpec actor to the shared Pick-oracle request.

        This adapter depends only on the loaded SceneSpec, actor metadata, and
        stage-wide skill policy.  It deliberately does not inspect an M13
        TaskRecord, so the M12 Sim Gate and M15 runtime can use byte-identical
        request semantics.
        """

        if str(arm) not in {"left", "right"}:
            raise ValueError(f"unsupported Pick arm: {arm!r}")
        actor = self.entities[str(subject_key)]
        spec = self.scene_spec["entities"][str(subject_key)]
        config = getattr(actor, "config", None) or {}
        support_plane_z_m = self._frozen_support_plane_z(subject_key)
        return PickAcquisitionRequest(
            subject=str(subject_key),
            arm=str(arm),
            asset=str(spec["asset"]),
            model_id=str(spec["model_id"]),
            use="pick_subject",
            use_profile="single_arm_pick",
            requested_pre_grasp_m=float(requested_pre_grasp_m),
            table_acquisition=True,
            support_plane_z_m=float(support_plane_z_m),
            contact_groups=config.get("contact_points_group") or (),
            contact_ids=(
                None
                if contact_ids is None
                else [int(value) for value in contact_ids]
            ),
            priority_contact_ids=(
                None
                if priority_contact_ids is None
                else [int(value) for value in priority_contact_ids]
            ),
            maximum_attempts=int(
                self.skill_execution_config.get(
                    "maximum_grasp_attempts",
                    EXACT_GRASP_POLICY[
                        "maximum_physical_contact_attempts"
                    ],
                )
            ),
            minimum_normalized_joint_margin=float(
                self.skill_execution_config.get(
                    "minimum_normalized_joint_margin", 0.10
                )
            ),
            world_yaw_deg=float(
                (spec.get("initial_pose") or {}).get(
                    "world_yaw_deg", 0.0
                )
            ),
            reach_zone=str(
                (self.scene_spec.get("reach") or {})
                .get(str(subject_key), {})
                .get("zone", "runtime_scene")
            ),
        )

    def plan_pick_witness(
        self,
        subject_key: str,
        arm: str,
        *,
        requested_pre_grasp_m: float,
        contact_ids: Sequence[int] | None = None,
        priority_contact_ids: Sequence[int] | None = None,
        candidate_search_budget: Mapping[str, Any] | None = None,
        candidate_admission=None,
    ) -> dict[str, Any]:
        """Public, motion-only SceneSpec feasibility query for one arm."""

        started = time.monotonic()
        result = plan_pick_acquisition(
            self,
            self.build_pick_acquisition_request(
                subject_key,
                arm,
                requested_pre_grasp_m=float(requested_pre_grasp_m),
                contact_ids=contact_ids,
                priority_contact_ids=priority_contact_ids,
            ),
            **({"candidate_admission": candidate_admission} if candidate_admission is not None else {}),
            **({"candidate_search_budget": candidate_search_budget}
               if candidate_search_budget is not None else {}),
        )
        if os.environ.get("M12_PLANNING_TIMINGS") == "1":
            planning = result.get("last_planning") or {}
            print(
                "[m12-pick-timing] "
                + json.dumps(
                    {
                        "arm": str(arm),
                        "candidate_variant_count": int(
                            planning.get("candidate_variant_count") or 0
                        ),
                        "elapsed_sec": round(
                            time.monotonic() - started, 12
                        ),
                        "fallback_executed": bool(
                            (
                                planning.get("contact_priority_execution")
                                or {}
                            ).get("fallback_executed")
                        ),
                        "plannable": bool(result.get("plannable")),
                        "subject": str(subject_key),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                file=sys.stderr,
                flush=True,
            )
        return result

    def plan_pick_witness_record(
        self, planning_request: Mapping[str, Any], *,
        candidate_search_budget: Mapping[str, Any] | None = None,
        required_contact_id: int | None = None,
    ) -> dict[str, Any]:
        """Plan an exact frozen request, optionally limiting candidate search only."""

        return plan_pick_acquisition(
            self,
            pick_acquisition_request_from_record(planning_request),
            **({"candidate_search_budget": candidate_search_budget}
               if candidate_search_budget is not None else {}),
            **({"required_contact_id": required_contact_id}
               if required_contact_id is not None else {}),
        )

    def certify_pick_witnesses(
        self,
        subject_key: str,
        arms: Sequence[str],
        *,
        requested_pre_grasp_m: float,
    ) -> dict[str, Any]:
        """Build one all-required-arms certificate for a frozen SceneSpec."""

        return certify_pick_acquisition_plans(
            self,
            [
                self.build_pick_acquisition_request(
                    subject_key,
                    arm,
                    requested_pre_grasp_m=float(requested_pre_grasp_m),
                )
                for arm in arms
            ],
        )

    def _verified_grasp(
        self,
        subject_key: str,
        arm: ArmTag,
        *,
        requested_pre_grasp_m: float,
        required_total_lift_delta_m: float,
        preferred_contact_ids: Sequence[int] = (),
        micro_lift: bool = True,
    ) -> dict[str, Any]:
        del micro_lift
        witness_key = f"{subject_key}:{arm}"
        preplanned = self.pick_planning_witnesses.pop(witness_key, None)
        frozen_request = self.pick_planning_requests.pop(witness_key, None)
        if (preplanned is None) is not (frozen_request is None):
            raise RuntimeError("incomplete_frozen_pick_execution_pair")
        if frozen_request is not None:
            if (
                frozen_request.subject != subject_key
                or frozen_request.arm != str(arm)
            ):
                raise RuntimeError("frozen_pick_execution_identity_mismatch")
            request = frozen_request
        else:
            request = self.build_pick_acquisition_request(
                subject_key,
                str(arm),
                requested_pre_grasp_m=float(requested_pre_grasp_m),
                contact_ids=(
                    list(preferred_contact_ids)
                    if preferred_contact_ids
                    else None
                ),
            )
        result = execute_pick_acquisition(
            self,
            request,
            preplanned=preplanned,
            required_total_lift_delta_m=float(
                required_total_lift_delta_m
            ),
        )
        return result

    def _touch_operation_anchor_lineage(
        self,
        *,
        subject: str,
        arm: str,
        use: str,
        contact_id: int,
    ) -> dict[str, Any]:
        """Resolve the exact M11 anchor proof accepted by M15 preflight."""

        matches = [
            row
            for row in self.operation_anchor_preflight.get("checks", [])
            if row.get("op") == "touch_event"
            and str(row.get("subject") or "") == subject
        ]
        if len(matches) != 1:
            raise _SkillExecutionFailure(
                "touch_event", "touch_operation_anchor_lineage_missing"
            )
        lineage = matches[0]
        if any(
            observed != expected
            for observed, expected in (
                (lineage.get("touch_validation_use"), use),
                (lineage.get("touch_contact_id"), contact_id),
                (lineage.get("touch_arm"), arm),
            )
        ):
            raise _SkillExecutionFailure(
                "touch_event", "touch_operation_anchor_lineage_mismatch"
            )
        contract_hash = str(lineage.get("contract_hash") or "")
        if not contract_hash:
            raise _SkillExecutionFailure(
                "touch_event", "touch_operation_anchor_contract_missing"
            )
        return deepcopy(dict(lineage))

    def _record_touch_contact_event(
        self,
        *,
        binding: Mapping[str, Any],
        lineage: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Record one arm-specific PhysX manifold at the authored Contact."""

        subject = str(binding["subject"])
        arm = str(binding["arm"])
        contact_id = int(binding["contact_id"])
        actor = self.entities[subject]
        target = np.asarray(
            actor.get_contact_point(contact_id)[:3], dtype=float
        )

        profile = self._runtime_embodiment_profile()
        arm_profile = (profile.get("arms") or {}).get(arm) or {}
        gripper_names = {
            str(row.get("link") or "")
            for row in arm_profile.get("fingers") or []
        }
        gripper_names.update(
            str(value)
            for value in arm_profile.get("palm_wrist_links") or []
        )
        gripper_names.update(
            str(value)
            for value in getattr(
                self.robot, f"{arm}_fix_gripper_name", ()
            )
        )
        for joint_row in getattr(self.robot, f"{arm}_gripper", ()):
            joint = joint_row[0] if isinstance(joint_row, tuple) else joint_row
            child_link = getattr(joint, "child_link", None)
            if child_link is not None:
                gripper_names.add(str(child_link.get_name()))
        gripper_names.discard("")

        subject_names = {str(actor.get_name())}
        raw_actor = actor.actor
        if hasattr(raw_actor, "get_links"):
            subject_names.update(
                str(link.get_name()) for link in raw_actor.get_links()
            )
        positions: list[np.ndarray] = []
        summed_impulse = 0.0
        for contact in self.scene.get_contacts():
            names = [str(body.entity.name) for body in contact.bodies]
            if (
                len(names) != 2
                or not subject_names.intersection(names)
                or not gripper_names.intersection(names)
            ):
                continue
            for point in contact.points:
                positions.append(np.asarray(point.position, dtype=float))
                summed_impulse += float(
                    np.linalg.norm(np.asarray(point.impulse, dtype=float))
                )
        if not positions:
            self._record_skill_diagnostic(
                op="touch_event_observation",
                subject=subject,
                arm=arm,
                use=str(binding["use"]),
                event=str(binding["event"]),
                contact_id=contact_id,
                contact_point_count=0,
                passed=False,
                failure_code="touch_contact_not_acquired",
            )
            return None

        nearest = min(
            positions,
            key=lambda point: float(np.linalg.norm(point - target)),
        )
        bounds = actor_aabb(actor)
        extent = float(
            np.max(
                np.asarray(bounds["max"], dtype=float)
                - np.asarray(bounds["min"], dtype=float)
            )
        )
        displacement = float(
            np.linalg.norm(
                np.asarray(actor.get_pose().p, dtype=float)
                - np.asarray(
                    self.initial_pose_state[subject]["position"], dtype=float
                )
            )
        )
        maximum_displacement = float(
            self._runtime_success_contract().section("click_press")[
                "maximum_subject_displacement_m"
            ]
        )
        event = {
            "event_version": _TOUCH_EVENT_VERSION,
            "use": str(binding["use"]),
            "event": str(binding["event"]),
            "arm": arm,
            "subject": subject,
            "contact_id": contact_id,
            "operation_anchor_kind": "grasp_contact",
            "operation_anchor_id": str(binding["operation_anchor_id"]),
            "operation_anchor_contract_hash": str(
                lineage.get("contract_hash") or ""
            ),
            "operation_anchor_policy_version": str(
                self.operation_anchor_preflight.get("policy_version") or ""
            ),
            "position_xyz_m": nearest.tolist(),
            "authored_center_xyz_m": target.tolist(),
            "distance_to_authored_center_m": float(
                np.linalg.norm(nearest - target)
            ),
            "target_local_extent_m": extent,
            "contact_point_count": len(positions),
            "summed_impulse_norm": float(summed_impulse),
            "subject_displacement_m": displacement,
            "subject_displacement_failed": bool(
                displacement > maximum_displacement
            ),
        }
        event["event_hash"] = artifact_hash(event)
        self.execution_events["contacts"].append(deepcopy(event))
        self._record_skill_diagnostic(
            op="touch_event_observation",
            passed=True,
            event=event,
        )
        return event

    def _skill_touch_event(self, step: Mapping[str, Any]) -> None:
        """Execute one M4-certified Click/Press event in the live SceneSpec."""

        try:
            binding = _validated_touch_step(step)
        except (TypeError, ValueError) as exc:
            raise _SkillExecutionFailure(
                "touch_event", f"touch_skill_binding_invalid:{exc}"
            ) from exc
        subject = str(binding["subject"])
        if subject not in self.entities:
            raise _SkillExecutionFailure(
                "touch_event", "touch_subject_missing"
            )
        actor = self.entities[subject]
        contact_id = int(binding["contact_id"])
        if not 0 <= contact_id < self._contact_count(actor):
            raise _SkillExecutionFailure(
                "touch_event", "touch_contact_id_out_of_range"
            )
        arm = self._arm(binding["arm"])
        lineage = self._touch_operation_anchor_lineage(
            subject=subject,
            arm=str(arm),
            use=str(binding["use"]),
            contact_id=contact_id,
        )

        self._reset_planning()
        if binding["use"] == "click_subject":
            self.move(
                self.grasp_actor(
                    actor,
                    arm_tag=arm,
                    pre_grasp_dis=0.1,
                    grasp_dis=0.1,
                    contact_point_id=contact_id,
                )
            )
            if self.plan_success:
                self.move(self.move_by_displacement(arm, z=-0.05))
        else:
            self.move(
                self.grasp_actor(
                    actor,
                    arm_tag=arm,
                    pre_grasp_dis=0.1,
                    grasp_dis=0.1,
                    contact_point_id=contact_id,
                )
            )
            if self.plan_success:
                self.move(self.close_gripper(arm_tag=arm))
            if self.plan_success:
                self.move(
                    self.grasp_actor(
                        actor,
                        arm_tag=arm,
                        pre_grasp_dis=0.02,
                        grasp_dis=0.02,
                        contact_point_id=contact_id,
                    )
                )
        if not self.plan_success:
            return
        event = self._record_touch_contact_event(
            binding=binding,
            lineage=lineage,
        )
        if event is None:
            raise _SkillExecutionFailure(
                "touch_event", "touch_contact_not_acquired"
            )
        self._record_primary_action(arm, "touch_event")
        self.primary_execution_event_hash = str(event["event_hash"])

    def _skill_grasp(self, step: Mapping[str, Any]) -> None:
        subject_key = str(step["subject"])
        arm = self._arm(step["arm"])
        lineage = next(
            (
                row
                for row in self.operation_anchor_preflight.get("checks", [])
                if row.get("op") == "grasp"
                and row.get("subject") == subject_key
            ),
            {},
        )
        binding = lineage.get("contact_contract_binding")
        if isinstance(binding, Mapping):
            actor = self.entities[subject_key]
            config = getattr(actor, "config", None) or {}
            contact_key = actor.POINTS["contact"]
            matrices = config.get(contact_key)
            if not isinstance(matrices, list) or not matrices:
                raise _SkillExecutionFailure(
                    "grasp", "runtime_contact_contract_missing"
                )
            live_contract = build_contact_contract(
                asset=str(self.scene_spec["entities"][subject_key]["asset"]),
                model_id=str(
                    self.scene_spec["entities"][subject_key]["model_id"]
                ),
                use=str(binding["candidate_use"]),
                contact_matrices=matrices,
                contact_groups=config.get("contact_points_group") or (),
                use_profile=str(binding["use_profile"]),
            )
            if any(
                live_contract.get(field) != binding.get(field)
                for field in (
                    "policy_version",
                    "contact_contract_id",
                    "contact_contract_hash",
                    "contact_projection_hash",
                )
            ):
                raise _SkillExecutionFailure(
                    "grasp", "runtime_contact_contract_stale"
                )
        result = self._verified_grasp(
            subject_key,
            arm,
            requested_pre_grasp_m=float(
                step.get("pre_grasp_m", 0.06)
            ),
            required_total_lift_delta_m=float(
                step["total_lift_delta_m"]
            ),
        )
        self._record_skill_diagnostic(
            op="grasp",
            subject=subject_key,
            arm=str(arm),
            required_total_lift_delta_m=float(
                step["total_lift_delta_m"]
            ),
            result=deepcopy(result),
        )
        if not bool(result["verified"]):
            code = str(result.get("failure_code") or "grasp_not_verified")
            if code == "held_lift_planner_failed":
                code = "lift_pose_unreachable"
            elif code == "held_lift_target_not_reached":
                code = "lift_target_not_reached"
            raise _SkillExecutionFailure("grasp", code)

    def _skill_adjust_orientation(self, step: Mapping[str, Any]) -> None:
        """Execute one initial-relative rigid orientation adjustment.

        The object remains attached through the rotation and support-plane
        placement.  The following authored ``release`` step performs the only
        detach/open operation.
        """

        subject_key = str(step["subject"])
        arm = self._arm(step["arm"])
        arm_name = str(arm)
        if step.get("orientation_adjustment_version") != (
            ORIENTATION_ADJUSTMENT_VERSION
        ):
            raise _SkillExecutionFailure(
                "adjust_orientation", "orientation_adjustment_version_stale"
            )
        try:
            params = normalize_orientation_adjustment_params(
                {
                    key: step.get(key)
                    for key in (
                        "orientation_axis",
                        "orientation_reference",
                        "rotation_delta_deg",
                    )
                }
            )
        except ValueError as exc:
            raise _SkillExecutionFailure(
                "adjust_orientation", "orientation_adjustment_params_invalid"
            ) from exc
        if self.held_entities.get(arm_name) != subject_key:
            raise _SkillExecutionFailure(
                "adjust_orientation", "orientation_subject_not_held"
            )
        initial_pose = (
            self.initial_success_entities.get(subject_key, {}).get("pose") or {}
        )
        initial_quaternion = initial_pose.get("quat_wxyz")
        if not isinstance(initial_quaternion, Sequence):
            raise _SkillExecutionFailure(
                "adjust_orientation", "initial_orientation_missing"
            )
        target_quaternion = target_quaternion_wxyz(initial_quaternion, params)
        current = self._unified_current_actor_pose(subject_key)
        rotation_target = [*current[:3], *target_quaternion]
        rotation_result = execute_held_actor_pose(
            self,
            HeldPoseRequest(
                subject=subject_key,
                arm=arm_name,
                target_actor_pose=rotation_target,
                position_tolerance_m=0.01,
                orientation_tolerance_deg=3.0,
            ),
        )
        placement_result: dict[str, Any] | None = None
        if rotation_result["passed"]:
            rotated = self._unified_current_actor_pose(subject_key)
            bounds = collision_aabb_tight(self.entities[subject_key])
            support_plane = float(
                self.initial_success_entities[subject_key]["support_plane_z"]
            )
            release_clearance = float(
                self._runtime_success_contract().section("orientation")[
                    "release_clearance_m"
                ]
            )
            support_target = [
                float(rotated[0]),
                float(rotated[1]),
                float(
                    rotated[2]
                    + support_plane
                    + release_clearance
                    - float(bounds["min"][2])
                ),
                *target_quaternion,
            ]
            placement_result = execute_held_actor_pose(
                self,
                HeldPoseRequest(
                    subject=subject_key,
                    arm=arm_name,
                    target_actor_pose=support_target,
                    position_tolerance_m=0.01,
                    orientation_tolerance_deg=3.0,
                ),
            )
        self._record_skill_diagnostic(
            op="adjust_orientation",
            subject=subject_key,
            arm=arm_name,
            params=params,
            target_quat_wxyz=target_quaternion,
            held_rotation=deepcopy(rotation_result),
            support_placement=deepcopy(placement_result),
        )
        if not rotation_result["passed"]:
            raise _SkillExecutionFailure(
                "adjust_orientation",
                str(
                    rotation_result.get("failure_code")
                    or "orientation_rotation_failed"
                ),
            )
        if not isinstance(placement_result, Mapping) or not placement_result.get(
            "passed"
        ):
            raise _SkillExecutionFailure(
                "adjust_orientation",
                str(
                    (placement_result or {}).get("failure_code")
                    or "orientation_support_placement_failed"
                ),
            )
        self._record_primary_action(arm, "adjust_orientation")

    def _unified_handover_ensure_source(
        self,
        subject_key: str,
        source_arm: str,
        contact_groups: Sequence[Sequence[int]],
    ) -> bool:
        del contact_groups
        return self._unified_handover_source_ready(
            subject_key, source_arm
        )

    def _unified_handover_source_ready(
        self, subject_key: str, source_arm: str
    ) -> bool:
        certification = self.pick_certifications.get(source_arm) or {}
        return bool(
            self.held_entities.get(source_arm) == subject_key
            and certification.get("subject") == subject_key
            and certification.get("contact_id") is not None
        )

    def _unified_handover_target_empty(self, target_arm: str) -> bool:
        return self.held_entities.get(target_arm) is None

    def _unified_handover_source_attachment_active(
        self, source_arm: str
    ) -> bool:
        return source_arm in self.pick_attachments

    def _unified_handover_source_contact_id(
        self, source_arm: str
    ) -> int:
        certification = self.pick_certifications.get(source_arm) or {}
        if certification.get("contact_id") is None:
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
        return self._held_actor_target_ee_pose(
            subject_key, self._arm(arm), actor_target_pose
        )

    def _unified_handover_source_reference(
        self, source_arm: str
    ) -> dict[str, Any]:
        reference = self.hold_references.get(source_arm)
        if not isinstance(reference, dict):
            raise HandoverExecutionError(
                "handover_source_reference_missing"
            )
        return reference

    def _unified_handover_move_source(
        self,
        subject_key: str,
        source_arm: str,
        target_pose: Sequence[float],
        *,
        collision_obstacles: Sequence[Mapping[str, Any]] = (),
    ) -> bool:
        arm = self._arm(source_arm)
        self._reset_planning()
        actor = self.entities[subject_key]
        world = refresh_collision_world(
            self,
            target_actor=actor,
            extra_obstacles=collision_obstacles,
        )
        world_hash = str(
            (world.get(source_arm) or {}).get("world_hash") or ""
        )
        reference = self.hold_references[source_arm]
        reference["collision_world_hash"] = world_hash
        motion = self.move_to_pose(arm, list(target_pose))
        for action in motion[1]:
            action.args.update(
                {
                    "expected_world_hash": world_hash,
                    "expected_attachment_hash": reference[
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
        arm = self._arm(target_arm)
        self._reset_planning()
        world = refresh_collision_world(
            self,
            target_actor=self.entities[subject_key],
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
        self._reset_planning()
        return passed

    def _unified_handover_freeze_target_reference(
        self, subject_key: str, target_arm: str
    ) -> None:
        self.hold_references.pop(target_arm, None)
        self._freeze_hold_reference(
            subject_key, self._arm(target_arm)
        )

    def _unified_handover_active_contacts(self) -> list[dict[str, Any]]:
        profile = self._runtime_embodiment_profile()
        grippers = observe_actual_grippers(
            self.robot, embodiment_profile=profile
        )
        return self._success_active_contacts(profile, grippers)

    def _unified_handover_observe_target_close(
        self, subject_key: str, target_arm: str
    ) -> dict[str, Any]:
        return self._unified_observe_hold(
            subject_key, target_arm, phase="pick"
        )

    def _unified_handover_target_hold_failure(
        self, summary: Mapping[str, Any], *, phase: str
    ) -> str | None:
        return self._hold_failure_code(summary, phase=phase)

    def _unified_handover_store_target_closure(
        self, target_arm: str, evidence: Mapping[str, Any]
    ) -> None:
        self._unified_store_certified_closure(
            target_arm, evidence
        )

    def _unified_handover_create_target_attachment(
        self, subject_key: str, target_arm: str
    ) -> dict[str, Any]:
        return self._create_pick_attachment(
            subject_key, self._arm(target_arm)
        )

    def _unified_handover_finalize_target_reference(
        self,
        subject_key: str,
        target_arm: str,
        closure_evidence: Mapping[str, Any],
    ) -> None:
        """Freeze carry drift at the target D6 zero-residual relation."""

        old_reference = deepcopy(
            self.hold_references.get(target_arm) or {}
        )
        self.hold_references.pop(target_arm, None)
        self._freeze_hold_reference(
            subject_key, self._arm(target_arm)
        )
        self.hold_references[target_arm][
            "certified_closure_evidence"
        ] = deepcopy(dict(closure_evidence))
        self.hold_references[target_arm][
            "reference_phase"
        ] = "target_attachment_zero_residual"
        for key in ("collision_world_hash", "attachment_hash"):
            if key in old_reference:
                self.hold_references[target_arm][key] = deepcopy(
                    old_reference[key]
                )

    def _unified_handover_abort_target(self, target_arm: str) -> None:
        try:
            self._remove_pick_attachment(self._arm(target_arm))
            self.hold_references.pop(target_arm, None)
            self.held_entities.pop(target_arm, None)
            self.pick_certifications.pop(target_arm, None)
            self._reset_planning()
            self.move(self.open_gripper(self._arm(target_arm)))
            self._reset_planning()
        finally:
            self.held_arm_drive_leases.release_arm(target_arm)

    def _unified_handover_stage_target(
        self,
        subject_key: str,
        target_arm: str,
        candidate: Mapping[str, Any],
        plan: Mapping[str, Any],
    ) -> None:
        self.held_entities[target_arm] = subject_key
        self.pick_certifications[target_arm] = {
            "subject": subject_key,
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
        self._remove_pick_attachment(self._arm(source_arm))
        self.robot.detach_collision_object(source_arm)
        self.held_entities.pop(source_arm, None)
        self.hold_references.pop(source_arm, None)
        self.pick_certifications.pop(source_arm, None)
        self._prepare_held_collision(
            subject_key, self._arm(target_arm)
        )

    def _unified_handover_release_source(
        self, source_arm: str
    ) -> bool:
        try:
            self._reset_planning()
            self.move(self.open_gripper(self._arm(source_arm)))
            passed = bool(self.plan_success)
            if passed:
                self.released_arms.add(source_arm)
            self._reset_planning()
            return passed
        finally:
            self.held_arm_drive_leases.release_arm(source_arm)

    def _unified_handover_retreat_source(
        self, source_arm: str, target_pose: Sequence[float]
    ) -> bool:
        arm = self._arm(source_arm)
        self._reset_planning()
        self.move(self.move_to_pose(arm, list(target_pose)))
        passed = bool(self.plan_success)
        self._reset_planning()
        return passed

    def _unified_handover_move_target(
        self,
        subject_key: str,
        target_arm: str,
        target_pose: Sequence[float],
    ) -> bool:
        arm = self._arm(target_arm)
        self._reset_planning()
        self._move_with_held_collision(
            subject_key, arm, self.move_to_pose(arm, list(target_pose))
        )
        passed = bool(self.plan_success)
        self._reset_planning()
        return passed

    def _unified_handover_observe_target_carry(
        self, subject_key: str, target_arm: str
    ) -> dict[str, Any]:
        return self._unified_observe_hold(
            subject_key, target_arm, phase="carry"
        )

    def _unified_handover_record(
        self, op: str, payload: Mapping[str, Any]
    ) -> None:
        self._record_skill_diagnostic(
            op=op,
            **{
                key: deepcopy(value)
                for key, value in payload.items()
                if not str(key).startswith("_")
            },
        )

    def _unified_handover_complete(
        self,
        subject_key: str,
        source_arm: str,
        target_arm: str,
        result: Mapping[str, Any],
    ) -> None:
        del subject_key, source_arm, result
        self.released_arms.discard(target_arm)
        self._record_primary_action(
            self._arm(target_arm), "handover"
        )
        self._reset_planning()

    def _skill_handover(self, step: Mapping[str, Any]) -> None:
        subject_key = str(step["subject"])
        source = str(self._arm(step["source_arm"]))
        target = str(self._arm(step["target_arm"]))
        actor = self.entities[subject_key]
        spec = self.scene_spec["entities"][subject_key]
        config = getattr(actor, "config", None) or {}
        table_support = [
            float((item.get("position") or [0.0, 0.0, 0.74])[2])
            for item in self.scene_spec.get("unit_poses", [])
        ]
        try:
            execute_handover(
                self,
                HandoverRequest(
                    subject=subject_key,
                    source_arm=source,
                    target_arm=target,
                    asset=str(spec["asset"]),
                    model_id=str(spec["model_id"]),
                    contact_groups=(
                        config.get("contact_points_group") or ()
                    ),
                    table_top_m=(
                        min(table_support) if table_support else 0.74
                    ),
                    world_yaw_deg=float(
                        (spec.get("initial_pose") or {}).get(
                            "world_yaw_deg", 0.0
                        )
                    ),
                    reach_zone="m15_live_handover",
                    minimum_normalized_joint_margin=float(
                        self.skill_execution_config.get(
                            "minimum_normalized_joint_margin", 0.10
                        )
                    ),
                ),
            )
        except HandoverExecutionError as exc:
            raise _SkillExecutionFailure(
                "handover", exc.code
            ) from exc

    def _reservation_target(self, step: Mapping[str, Any]) -> list[float]:
        relation = str(step["relation"])
        subject = str(step["subject"])
        targets = [
            str(step[key])
            for key in ("target", "target2")
            if step.get(key)
        ]
        target_poses = [
            np.asarray(self.entities[key].get_pose().p, dtype=float)
            for key in targets
        ]
        if not target_poses:
            raise RuntimeError("receiver_reservation_anchor_missing")
        receiver_top = max(
            float(actor_aabb(self.entities[key])["max"][2])
            for key in targets
        )
        # M5's actor_aabb consumes RoboTwin actor wrappers, while the base task
        # table is a raw SAPIEN entity.  M12 already freezes every Unit frame on
        # the table support plane, so use that canonical Z instead of
        # reconstructing table geometry from the simulator entity.
        table_support_z = [
            float((item.get("position") or [0.0, 0.0, 0.74])[2])
            for item in self.scene_spec.get("unit_poses", [])
        ]
        table_top = min(table_support_z) if table_support_z else 0.74
        anchor = resolve_reservation_anchor(
            self.unit_artifact,
            self.scene_spec,
            relation=relation,
            subject=subject,
            receiver_top_z=receiver_top,
            table_top_z=table_top,
        )
        if anchor is None:
            raise RuntimeError(
                f"receiver_reservation_anchor_missing:{subject}:{relation}"
            )
        return anchor + self._reservation_orientation(
            relation=relation,
            subject=subject,
        )

    def _reservation_orientation(
        self, *, relation: str, subject: str
    ) -> list[float]:
        unit_poses = {
            str(item["unit_id"]): item
            for item in self.scene_spec.get("unit_poses", [])
        }
        subject_unit = next(
            (
                unit
                for unit in self.unit_artifact.get("units", [])
                if subject in unit.get("entity_keys", [])
            ),
            None,
        )
        if subject_unit is None or str(subject_unit.get("id")) not in unit_poses:
            raise RuntimeError(
                f"subject_unit_pose_missing:{subject}:{relation}"
            )
        reservation_unit = None
        reservation = None
        for unit in self.unit_artifact.get("units", []):
            for region in (unit.get("params") or {}).get(
                "reserved_regions", []
            ):
                if (
                    str(region.get("relation")) == relation
                    and str(
                        region.get("subject_key")
                        or region.get("subject")
                        or subject
                    )
                    == subject
                ):
                    reservation_unit = unit
                    reservation = region
                    break
            if reservation is not None:
                break
        if (
            reservation_unit is None
            or reservation is None
            or str(reservation_unit.get("id")) not in unit_poses
        ):
            raise RuntimeError(
                f"reservation_orientation_missing:{subject}:{relation}"
            )
        subject_unit_yaw = float(
            unit_poses[str(subject_unit["id"])].get("yaw_deg", 0.0)
        )
        receiver_unit_yaw = float(
            unit_poses[str(reservation_unit["id"])].get("yaw_deg", 0.0)
        )
        initial_q = self.scene_spec["entities"][subject]["pose"][
            "quat_wxyz"
        ]
        canonical_q = _quat_multiply(
            _z_rotation_quat(-subject_unit_yaw),
            initial_q,
        )
        world_yaw = receiver_unit_yaw + float(
            reservation.get("relative_yaw_deg", 0.0)
        )
        return _quat_multiply(_z_rotation_quat(world_yaw), canonical_q)

    def _carry_planner_feedback(
        self, arm: ArmTag, *, previous_result_count: int
    ) -> dict[str, Any]:
        """Record only the current phase's small planner receipt, never its path."""

        results = getattr(self, f"{str(arm)}_joint_path", ())
        if len(results) <= previous_result_count or not isinstance(results[-1], Mapping):
            return {"available": False}
        result = results[-1]
        return {
            "available": True,
            **{key: (None if result.get(key) is None else str(result[key]))
               for key in ("status", "planner_status", "world_hash", "attachment_hash")},
            **({"table_validation": deepcopy(result["table_validation"])}
               if isinstance(result.get("table_validation"), Mapping) else {}),
            **({"carry_motion": deepcopy(result["carry_motion"])}
               if isinstance(result.get("carry_motion"), Mapping) else {}),
        }

    def _skill_carry_to_relation(self, step: Mapping[str, Any]) -> None:
        subject_key = str(step["subject"])
        arm = self._arm(step["arm"])
        certification = self.pick_certifications.get(str(arm)) or {}
        if (
            self.held_entities.get(str(arm)) != subject_key
            or certification.get("subject") != subject_key
        ):
            raise _SkillExecutionFailure(
                "carry_to_relation", "subject_not_held"
            )
        continuity = int(
            self._runtime_embodiment_profile()[
                "classification"
            ]["continuous_contact_min_steps"]
        )
        route_report = self.current_route_report
        selected_route = (
            (route_report or {}).get("selected_witness") or {}
        )
        terminal_candidate = selected_route.get("terminal_candidate") or {}
        if (
            not route_report
            or str(route_report.get("subject") or "") != subject_key
            or str(route_report.get("arm") or "") != str(arm)
            or str(route_report.get("relation") or "")
            != str(step["relation"])
            or not terminal_candidate.get("actor_pose")
        ):
            raise _SkillExecutionFailure(
                "carry_to_relation", "route_witness_missing_or_stale"
            )
        route_terminal_candidate = deepcopy(dict(terminal_candidate))
        recipe_binding = route_terminal_candidate.get(
            "placement_recipe_binding"
        )
        if not isinstance(recipe_binding, Mapping):
            raise _SkillExecutionFailure(
                "carry_to_relation", "route_terminal_recipe_missing"
            )
        recipe = recipe_binding.get("placement_recipe") or {}
        profiles = load_reviewed_receiver_profiles()
        cavity_matrix = None
        articulated_terminal = (
            isinstance(recipe, Mapping)
            and recipe.get("constructor")
            == "articulated_place_in_cavity_link"
        )
        receiver_keys = [
            str(step[key])
            for key in ("target", "target2")
            if step.get(key)
        ]
        if articulated_terminal:
            profile_id = str(recipe.get("receiver_state_profile_id") or "")
            profile = profiles.get(profile_id)
            if not isinstance(profile, Mapping) or len(receiver_keys) != 1:
                raise _SkillExecutionFailure(
                    "carry_to_relation", "route_terminal_profile_missing"
                )
            cavity_matrix = articulated_cavity_world_matrix(
                self.entities[receiver_keys[0]], profile
            )
            route_frame = route_terminal_candidate.get(
                "terminal_frame_binding"
            )
            if not isinstance(route_frame, Mapping):
                raise _SkillExecutionFailure(
                    "carry_to_relation", "route_terminal_frame_missing"
                )
            frozen_frame = validate_terminal_frame_binding(route_frame)
            frame_translation_error = float(
                np.linalg.norm(
                    cavity_matrix[:3, 3] - frozen_frame[:3, 3]
                )
            )
            frame_orientation_error = _rotation_error_deg(
                cavity_matrix[:3, :3], frozen_frame[:3, :3]
            )
            frame_policy = ROUTE_EXECUTION_CONTRACT[
                "articulated_terminal_frame_tracking"
            ]
            if (
                frame_translation_error
                > float(frame_policy["translation_tolerance_m"])
                or frame_orientation_error
                > float(frame_policy["orientation_tolerance_deg"])
            ):
                raise _SkillExecutionFailure(
                    "carry_to_relation", "route_terminal_frame_changed"
                )
        live_terminal_candidate = terminal_candidate_from_sources(
            self.unit_artifact,
            self.scene_spec,
            step,
            offset_xy_m=route_terminal_candidate["offset_xy_m"],
            yaw_delta_deg=float(
                route_terminal_candidate["yaw_delta_deg"]
            ),
            cavity_world_matrix=cavity_matrix,
            receiver_profiles=profiles,
        )
        if live_terminal_candidate.get("placement_recipe_binding") != (
            recipe_binding
        ):
            raise _SkillExecutionFailure(
                "carry_to_relation", "route_terminal_recipe_stale"
            )
        if not articulated_terminal and (
            live_terminal_candidate != route_terminal_candidate
        ):
            raise _SkillExecutionFailure(
                "carry_to_relation", "route_terminal_lineage_stale"
            )
        terminal_candidate = live_terminal_candidate
        target = [
            float(value) for value in terminal_candidate["actor_pose"]
        ]
        self._record_skill_diagnostic(
            op="route_witness_consumed",
            subject=subject_key,
            arm=str(arm),
            relation=str(step["relation"]),
            route_report_hash=str(route_report["report_hash"]),
            route_witness_hash=str(selected_route["witness_hash"]),
            contact_id=int(selected_route["contact_id"]),
            terminal_offset_xy_m=list(
                terminal_candidate["offset_xy_m"]
            ),
            terminal_yaw_delta_deg=float(
                terminal_candidate["yaw_delta_deg"]
            ),
            placement_recipe_hash=str(
                recipe_binding["placement_recipe_hash"]
            ),
            placement_recipe_source=str(
                recipe_binding["placement_recipe_source"]
            ),
            placement_settle_policy=str(recipe.get("settle_policy") or ""),
            receiver_state_profile_id=recipe_binding.get(
                "receiver_state_profile_id"
            ),
            receiver_state_profile_hash=recipe_binding.get(
                "receiver_state_profile_hash"
            ),
            live_cavity_frame=bool(articulated_terminal),
        )
        target_position = np.asarray(target[:3], dtype=float)
        phase_actor_poses = selected_route.get("phase_actor_poses") or {}
        expected_phase_policy = route_phase_policy(str(step["relation"]))
        if (
            set(phase_actor_poses) != {"lift", *ROUTE_CARRY_PHASES}
            or selected_route.get("phase_policy") != expected_phase_policy
        ):
            raise _SkillExecutionFailure(
                "carry_to_relation", "route_phase_policy_missing_or_stale"
            )
        live_terminal_matrix = np.asarray(
            sapien.Pose(
                terminal_candidate["actor_pose"][:3],
                terminal_candidate["actor_pose"][3:],
            ).to_transformation_matrix(),
            dtype=float,
        )
        initial_scene_pose = self.scene_spec["entities"][subject_key]["pose"]
        initial_actor_matrix = np.asarray(
            sapien.Pose(
                initial_scene_pose["position"],
                initial_scene_pose["quat_wxyz"],
            ).to_transformation_matrix(),
            dtype=float,
        )
        live_phase_actor_matrices = route_phase_actor_poses(
            initial_actor_matrix,
            live_terminal_matrix,
            relation=str(step["relation"]),
        )
        self._prepare_m15_carry_collision(subject_key, arm)
        for stage_index, phase in enumerate(ROUTE_CARRY_PHASES):
            frozen_desired_pose = [
                float(value) for value in phase_actor_poses[phase]
            ]
            if len(frozen_desired_pose) != 7:
                raise _SkillExecutionFailure(
                    "carry_to_relation", "route_phase_target_invalid"
                )
            desired_matrix = live_phase_actor_matrices[phase]
            desired_pose = [
                *desired_matrix[:3, 3].tolist(),
                *t3d.quaternions.mat2quat(
                    desired_matrix[:3, :3]
                ).tolist(),
            ]
            before = self._actor_position(subject_key)
            before_error = float(
                np.linalg.norm(before - target_position)
            )
            excluded = route_collision_excluded_entity_keys(
                subject=subject_key,
                receiver_keys=receiver_keys,
                relation=str(step["relation"]),
                phase=phase,
            ) - {subject_key}
            waypoint = self._held_actor_target_ee_pose(
                subject_key, arm, desired_pose
            )
            self._reset_planning()
            previous_plan_result_count = len(
                getattr(self, f"{str(arm)}_joint_path", ())
            )
            self._move_with_held_collision(
                subject_key,
                arm,
                self.move_to_pose(
                    arm_tag=arm,
                    target_pose=waypoint,
                ),
                excluded_entity_keys=sorted(excluded),
                m15_exact_table_validation=(phase == "descent" and bool(
                    getattr(self, "_m15_corrected_carry_collision", False)
                )),
                **({"carry_target_pose": waypoint,
                    "carry_motion": carry_motion_mode(phase, live_phase_actor_matrices)}
                   if getattr(self, "_m15_corrected_carry_collision", False) else {}),
            )
            planned = bool(self.plan_success)
            actual = self._actor_position(subject_key)
            position_error = float(
                np.linalg.norm(actual - target_position)
            )
            planar_error = float(
                np.linalg.norm(actual[:2] - target_position[:2])
            )
            self._record_skill_diagnostic(
                op="carry_feedback_step",
                subject=subject_key,
                arm=str(arm),
                relation=str(step["relation"]),
                stage_index=stage_index,
                route_phase=phase,
                route_phase_actor_pose=desired_pose,
                receiver_collision=expected_phase_policy[
                    "receiver_collision"
                ][phase],
                excluded_entity_keys=sorted(excluded),
                planned=planned,
                planner_feedback=self._carry_planner_feedback(
                    arm, previous_result_count=previous_plan_result_count
                ),
                grasp_transform_preserved=True,
                target_position=target_position.tolist(),
                actual_position=actual.tolist(),
                position_error_before_m=before_error,
                position_error_m=position_error,
                planar_error_m=planar_error,
            )
            hold_summary = self._observe_hold(
                subject_key, arm, steps=continuity
            )
            hold_failure = self._hold_failure_code(
                hold_summary, phase="carry"
            )
            self._record_skill_diagnostic(
                op="carry_hold_observation",
                subject=subject_key,
                arm=str(arm),
                stage_index=stage_index,
                route_phase=phase,
                contact_point_id=certification.get("contact_id"),
                success_snapshot_hash=certification.get(
                    "success_snapshot_hash"
                ),
                first_failure_step=stage_index if hold_failure else None,
                hold_metrics=hold_summary["metrics"],
                failure_code=hold_failure,
            )
            if hold_failure is not None:
                raise _SkillExecutionFailure(
                    "carry_to_relation", hold_failure
                )
            if not planned:
                self._reset_planning()
                raise _SkillExecutionFailure(
                    "carry_to_relation",
                    f"route_{phase}_execution_failed",
                )

        actual_actor_pose = self.entities[subject_key].get_pose()
        actual_pose = [
            *[float(value) for value in actual_actor_pose.p],
            *[float(value) for value in actual_actor_pose.q],
        ]
        actual = np.asarray(actual_pose[:3], dtype=float)
        planar_error = float(
            np.linalg.norm(actual[:2] - target_position[:2])
        )
        position_error = float(np.linalg.norm(actual - target_position))
        admissibility = route_terminal_admissibility(
            actual_pose,
            terminal_candidate,
            actual_subject_aabb=collision_aabb_tight(
                self.entities[subject_key]
            ),
            actual_receiver_aabbs={
                key: collision_aabb_tight(self.entities[key])
                for key in receiver_keys
            },
        )
        self._record_skill_diagnostic(
            op="carry_target_feedback",
            subject=subject_key,
            relation=str(step["relation"]),
            route_phase="descent",
            target_position=target_position.tolist(),
            actual_position=actual.tolist(),
            planar_error_m=planar_error,
            position_error_m=position_error,
            terminal_admissibility=admissibility,
        )
        if not admissibility["passed"]:
            raise _SkillExecutionFailure(
                "carry_to_relation", "carry_terminal_not_admissible"
            )
        self._reset_planning()
        return

    def _skill_settle_stack(self, step: Mapping[str, Any]) -> None:
        del step
        for _ in range(40):
            self.scene.step()

    def _refresh_release_retreat_collision_world(
        self, subject_key: str, arm: ArmTag
    ) -> str:
        """Restore the receiver after the descent-only exclusion expires."""

        route = self.current_route_report or {}
        excluded = route_collision_excluded_entity_keys(
            subject=subject_key,
            receiver_keys=[str(value) for value in route.get("receivers") or []],
            relation=str(route.get("relation") or ""),
            phase="release_retreat",
        ) - {subject_key}
        obstacles = route_scene_collision_obstacles(
            self.scene_spec,
            self.entities,
            excluded_entity_keys=[subject_key, *sorted(excluded)],
            **({"complete_geometry": True,
                "max_obstacles": 64 - len(self._m15_live_table_collision_obstacles())}
               if (getattr(self, "hold_references", {}).get(str(arm)) or {}).get("carry_collision_binding")
               else {}),
        )
        world = self.robot.update_world_obstacles(obstacles)
        if not isinstance(world, Mapping) or not bool(world.get("updated")):
            raise RuntimeError(
                "release-retreat collision world update was not acknowledged"
            )
        world_hash = str((world.get(str(arm)) or {}).get("world_hash") or "")
        if not world_hash:
            raise RuntimeError("release-retreat collision world hash missing")
        self._record_skill_diagnostic(
            op="release_retreat_collision_world_restored",
            subject=subject_key,
            arm=str(arm),
            world_hash=world_hash,
            excluded_entity_keys=sorted(excluded),
        )
        return world_hash

    def _skill_release(self, step: Mapping[str, Any]) -> None:
        arm = self._arm(step["arm"])
        arm_name = str(arm)
        subject_key = self.held_entities.get(arm_name)
        self.release_retreat_world_hashes.pop(arm_name, None)
        try:
            self._remove_pick_attachment(arm)
            self.move(self.open_gripper(arm))
            self.robot.detach_collision_object(arm_name)
            for _ in range(20):
                self.scene.step()
            if subject_key is not None:
                self.release_retreat_world_hashes[arm_name] = (
                    self._refresh_release_retreat_collision_world(
                        subject_key, arm
                    )
                )
            self.released_arms.add(arm_name)
            self.held_entities.pop(arm_name, None)
            self.hold_references.pop(arm_name, None)
            self.pick_certifications.pop(arm_name, None)
        finally:
            self.held_arm_drive_leases.release_arm(arm_name)

    def close_env(self, clear_cache: bool = False) -> None:
        try:
            self._unified_release_all_held_drives()
        finally:
            batch = getattr(self, "_m15_scene_camera_batch", None)
            try:
                if batch is not None:
                    batch.close()
            finally:
                self._m15_scene_camera_batch = None
                super().close_env(clear_cache=clear_cache)

    def _skill_retreat_relative(self, step: Mapping[str, Any]) -> None:
        arm = self._arm(step["arm"])
        arm_name = str(arm)
        policy = route_release_retreat_policy()
        delta = float(step.get("delta_m", -1.0))
        if (
            str(step.get("frame") or "") != policy["frame"]
            or str(step.get("axis") or "") != policy["axis"]
            or not math.isclose(
                delta, float(policy["delta_m"]), abs_tol=1e-12
            )
        ):
            raise _SkillExecutionFailure(
                "retreat_relative", "route_retreat_policy_mismatch"
            )
        expected_world_hash = self.release_retreat_world_hashes.get(arm_name)
        if not expected_world_hash:
            raise _SkillExecutionFailure(
                "retreat_relative", "release_retreat_world_missing"
            )
        # Construct the target with the same pure helper used by the M14
        # certificate.  This is exactly -delta along EE-local X; it avoids
        # depending on Base_Task's historical `z`/`move_axis="arm"` alias.
        target_ee = release_retreat_ee_matrix(self._ee_matrix(arm))
        motion = self.move_to_pose(
            arm,
            self._matrix_pose(target_ee),
        )
        for action in motion[1]:
            action.args["expected_world_hash"] = expected_world_hash

        try:
            self.move(motion)
            if not self.plan_success:
                self._reset_planning()
                raise _SkillExecutionFailure(
                    "retreat_relative", "route_release_retreat_execution_failed"
                )
        finally:
            self.release_retreat_world_hashes.pop(arm_name, None)

    def _skill_actuate_state(self, step: Mapping[str, Any]) -> None:
        subject_key = str(step["subject"])
        state = str(step["state"])
        actor = self.entities[subject_key]
        requested_use = articulation_use(state)
        evidence = _canonical_validation_evidence(
            self.joined, subject_key, state
        )
        canonical_use = str(evidence["reference"].get("use") or "")
        if canonical_use != requested_use:
            raise RuntimeError(
                "canonical_articulation_use_mismatch:"
                f"{subject_key}:{requested_use}:{canonical_use}"
            )
        requested_arm = self._arm(step["arm"])
        joint_index = self.articulation_joint_indices[subject_key]
        qpos = np.asarray(actor.get_qpos(), dtype=float)
        qlimits = np.asarray(actor.get_qlimits(), dtype=float)
        if joint_index >= len(qpos) or joint_index >= len(qlimits):
            raise RuntimeError(
                f"articulation_joint_out_of_range:{subject_key}"
            )
        contract = self._runtime_success_contract()
        state_profile = deepcopy(
            evidence["reference"].get("state_profile") or {}
        )
        if state in {"on", "off"}:
            state_profile.setdefault(
                "positive_use",
                contract.section("articulation")["default_positive_use"],
            )
        entity_spec = self.scene_spec["entities"][subject_key]
        try:
            execution_profile = resolve_articulation_execution_profile(
                str(entity_spec["asset"]),
                str(entity_spec["model_id"]),
                canonical_use,
                contract=contract,
            )
        except ArticulationExecutionError as exc:
            self._record_skill_diagnostic(
                op="actuate_state_feedback",
                entity_key=subject_key,
                state=state,
                use=canonical_use,
                requested_arm=str(requested_arm),
                success=False,
                failure_code=str(exc),
            )
            raise _SkillExecutionFailure("actuate_state", str(exc)) from exc
        lineage = next(
            (
                row
                for row in self.operation_anchor_preflight.get("checks", [])
                if row.get("op") == "actuate_state"
                and row.get("subject") == subject_key
            ),
            {},
        )
        expected_execution = lineage.get(
            "articulation_execution_binding"
        )
        if isinstance(expected_execution, Mapping):
            observed_execution = {
                "profile_version": execution_profile["profile_version"],
                "profile_hash": execution_profile["profile_hash"],
                "profile_source_hash": execution_profile[
                    "source_asset_catalog_sha256"
                ],
                "motion_policy_version": execution_profile[
                    "motion_policy_version"
                ],
                "motion_policy_hash": execution_profile[
                    "motion_policy_hash"
                ],
            }
            expected_profile_execution = {
                key: expected_execution.get(key)
                for key in observed_execution
            }
            if expected_profile_execution != observed_execution:
                raise _SkillExecutionFailure(
                    "actuate_state",
                    "runtime_articulation_execution_profile_stale",
                )
        if not isinstance(expected_execution, Mapping):
            raise _SkillExecutionFailure(
                "actuate_state",
                "runtime_articulation_arm_capability_missing",
            )
        requested_capability = validated_articulation_arm_capability(
            expected_execution,
            use=canonical_use,
            arm=str(requested_arm),
        )
        if requested_capability is None:
            raise _SkillExecutionFailure(
                "actuate_state",
                "runtime_articulation_arm_capability_missing",
            )
        arm_candidates: list[tuple[ArmTag, Mapping[str, Any]]] = [
            (requested_arm, requested_capability)
        ]
        for candidate_row in requested_capability.get(
            "successful_candidates", []
        ):
            if (
                not isinstance(candidate_row, Mapping)
                or str(candidate_row.get("moving_link") or "")
                != str(execution_profile["moving_link"])
                or str(candidate_row.get("engagement_mode") or "")
                != str(execution_profile["engagement_mode"])
            ):
                raise _SkillExecutionFailure(
                    "actuate_state",
                    "articulation_candidate_execution_profile_stale",
                )
        state_profile.setdefault(
            "positive_use", execution_profile["positive_use"]
        )
        actual_link = self._active_joint_child_name(subject_key)
        if actual_link != execution_profile["moving_link"]:
            self._record_skill_diagnostic(
                op="actuate_state_feedback",
                entity_key=subject_key,
                state=state,
                use=canonical_use,
                requested_arm=str(requested_arm),
                actual_moving_link=actual_link,
                expected_moving_link=execution_profile["moving_link"],
                execution_profile_version=execution_profile["profile_version"],
                execution_profile_hash=execution_profile["profile_hash"],
                execution_profile_source_hash=execution_profile[
                    "source_asset_catalog_sha256"
                ],
                motion_policy_version=execution_profile[
                    "motion_policy_version"
                ],
                motion_policy_hash=execution_profile["motion_policy_hash"],
                success=False,
                failure_code="articulation_contact_wrong_link",
            )
            raise _SkillExecutionFailure(
                "actuate_state", "articulation_contact_wrong_link"
            )
        result: dict[str, Any] | None = None
        selected_arm = requested_arm
        arm_attempts: list[dict[str, Any]] = []
        for arm, arm_capability in arm_candidates:
            selected_arm = arm
            result = self._run_articulation_with_arm(
                subject_key=subject_key,
                arm=arm,
                state=state,
                state_profile=state_profile,
                execution_profile=execution_profile,
                validated_candidates=[
                    deepcopy(dict(row))
                    for row in arm_capability.get(
                        "successful_candidates", []
                    )
                    if isinstance(row, Mapping)
                ],
            )
            arm_attempts.append({"arm": str(arm), "result": result})
            if result["success"]:
                break
        final_qpos = float(actor.get_qpos()[joint_index])
        self._record_skill_diagnostic(
            op="actuate_state_feedback",
            entity_key=subject_key,
            state=state,
            use=canonical_use,
            requested_arm=str(requested_arm),
            selected_arm=str(selected_arm),
            initial_qpos=float(self.initial_articulations[subject_key]),
            final_qpos=final_qpos,
            execution_profile_version=execution_profile["profile_version"],
            execution_profile_hash=execution_profile["profile_hash"],
            execution_profile_source_hash=execution_profile[
                "source_asset_catalog_sha256"
            ],
            motion_policy_version=execution_profile[
                "motion_policy_version"
            ],
            motion_policy_hash=execution_profile["motion_policy_hash"],
            articulation_result=result,
            articulation_arm_attempts=arm_attempts,
            articulation_arm_capability=(
                lineage.get("articulation_arm_capability")
            ),
            success=bool(result and result["success"]),
        )
        prior_evidence = self.canonical_evidence.get(subject_key) or {}
        self.canonical_evidence[subject_key] = {
            **deepcopy(prior_evidence),
            "path": evidence["path"],
            "file_hash": evidence["file_hash"],
            "report_hash": evidence["record"].get("report_hash"),
            "record_id": evidence["reference"].get("record_id"),
            "state": state,
            "canonical_use": canonical_use,
            "validation_only": True,
            "action_trace_consumed": False,
            "articulation_execution_profile_hash": execution_profile[
                "profile_hash"
            ],
            "articulation_motion_policy_version": execution_profile[
                "motion_policy_version"
            ],
            "articulation_motion_policy_hash": execution_profile[
                "motion_policy_hash"
            ],
        }
        if not result or not result["success"]:
            raise _SkillExecutionFailure(
                "actuate_state",
                str(
                    (result or {}).get("failure_code")
                    or "articulation_target_interval_not_reached"
                ),
            )
        self._record_primary_action(selected_arm, "actuate_state")

    def _active_joint_child_name(self, subject_key: str) -> str | None:
        actor = self.entities[subject_key]
        joint_index = self.articulation_joint_indices[subject_key]
        joints = list(actor.actor.get_active_joints())
        if joint_index >= len(joints):
            return None
        child = getattr(joints[joint_index], "child_link", None)
        if child is None:
            return None
        getter = getattr(child, "get_name", None)
        return str(getter() if callable(getter) else "")

    def _matrix_pose(self, matrix: np.ndarray) -> list[float]:
        return (
            np.asarray(matrix[:3, 3], dtype=float).tolist()
            + t3d.quaternions.mat2quat(matrix[:3, :3]).tolist()
        )

    def _articulation_contact_matrix(
        self,
        subject_key: str,
        contact_id: int,
        execution_profile: Mapping[str, Any],
    ) -> np.ndarray:
        actor = self.entities[subject_key]
        if execution_profile.get("engagement_mode") != "switch_press":
            return np.asarray(
                actor.get_contact_point(int(contact_id), "matrix"), dtype=float
            )
        point = deepcopy((actor.config or {}).get("contact_points", [])[contact_id])
        local = np.asarray(point["matrix"], dtype=float)
        local[:3, 3] *= float((actor.config or {}).get("scale") or 1.0)
        if execution_profile.get("mirror_contact_local_y"):
            local[1, 3] *= -1.0
        base = str(execution_profile.get("contact_frame_base") or point.get("base"))
        link = actor.link_dict[base]
        return np.asarray(link.get_pose().to_transformation_matrix(), dtype=float) @ local

    def _articulation_grasp_pose(
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

    def _articulation_joint_motion(self, subject_key: str) -> dict[str, Any]:
        actor = self.entities[subject_key]
        joint_index = self.articulation_joint_indices[subject_key]
        joint = list(actor.actor.get_active_joints())[joint_index]
        child = joint.child_link
        joint_pose = getattr(joint, "pose_in_child", None)
        if callable(joint_pose):
            joint_pose = joint_pose()
        if joint_pose is None:
            getter = getattr(joint, "get_pose_in_child", None)
            joint_pose = getter() if callable(getter) else None
        if joint_pose is None:
            raise _SkillExecutionFailure(
                "actuate_state", "articulation_joint_frame_invalid"
            )
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

    def _run_articulation_with_arm(
        self,
        *,
        subject_key: str,
        arm: ArmTag,
        state: str,
        state_profile: Mapping[str, Any],
        execution_profile: Mapping[str, Any],
        validated_candidates: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        actor = self.entities[subject_key]
        joint_index = self.articulation_joint_indices[subject_key]
        limits = np.asarray(actor.get_qlimits(), dtype=float)[joint_index]
        profile = self._runtime_embodiment_profile()
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
                    excluded_entities=(actor,),
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
            return float(actor.get_qpos()[joint_index])

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
                contacts = self._success_active_contacts(profile, grippers)
                last_gripper = grippers[str(arm)]
                last_contacts = contacts
                reference = runtime_state.get("reference")
                if reference is not None:
                    child_matrix = np.asarray(
                        self._articulation_joint_motion(subject_key)[
                            "child"
                        ].get_pose().to_transformation_matrix(),
                        dtype=float,
                    )
                    relative = np.linalg.inv(self._ee_matrix(arm)) @ child_matrix
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
                    subject=subject_key,
                    arm=str(arm),
                    expected_link=str(execution_profile["moving_link"]),
                    gripper=last_gripper,
                    contacts=last_contacts,
                    continuous_steps=1,
                    required_steps=1,
                    translation_drift_m=maximum_translation,
                    rotation_drift_deg=maximum_rotation,
                    translation_limit_m=float(
                        self._runtime_success_contract().section("hold")[
                            "relative_translation_drift_m"
                        ]
                    ),
                    rotation_limit_deg=float(
                        self._runtime_success_contract().section("hold")[
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
                subject=subject_key,
                arm=str(arm),
                expected_link=str(execution_profile["moving_link"]),
                gripper=last_gripper,
                contacts=last_contacts,
                continuous_steps=consecutive,
                required_steps=required_steps,
                translation_drift_m=maximum_translation,
                rotation_drift_deg=maximum_rotation,
                translation_limit_m=float(
                    self._runtime_success_contract().section("hold")[
                        "relative_translation_drift_m"
                    ]
                ),
                rotation_limit_deg=float(
                    self._runtime_success_contract().section("hold")[
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

        def observe(segment: Mapping[str, Any]) -> dict[str, Any]:
            if runtime_state.get("attachment") is not None:
                return attachment_observation(segment)
            return physical_observation(segment)

        def create_attachment(contact_matrix: np.ndarray) -> dict[str, Any]:
            moving_link = self._articulation_joint_motion(subject_key)["child"]
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

        def acquire(segment: Mapping[str, Any]) -> dict[str, Any]:
            points = (actor.config or {}).get("contact_points", [])
            attempts: list[dict[str, Any]] = []
            geometry_frames: dict[tuple[int, str], dict[str, Any]] = {}
            moving_link_matrix: np.ndarray | None = None
            moving_link_meshes: list[dict[str, np.ndarray]] | None = None
            if execution_profile["engagement_mode"] == "grasp_follow":
                try:
                    moving_link = self._articulation_joint_motion(subject_key)[
                        "child"
                    ]
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
                execution_profile,
                segment["contact_ids"],
                validated_candidates=validated_candidates,
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
                matrix = self._articulation_contact_matrix(
                    subject_key, contact_id, execution_profile
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
                pre_grasp_pose = self._articulation_grasp_pose(
                    matrix,
                    pre_grasp_m=pre_grasp_m,
                    roll_rad=roll_rad,
                    grasp_rotation_world=grasp_rotation_world,
                )
                contact_pose = self._articulation_grasp_pose(
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
                            *[self._matrix_pose(value) for value in approach],
                        ],
                    }
                )

            self._reset_planning()
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

            if execution_profile["engagement_mode"] in {
                "switch_press", "push_follow"
            }:
                self._reset_planning()
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
                return {
                    "passed": False,
                    "failure_code": "articulation_negative_progress",
                    "attempts": attempts,
                }
            self._reset_planning()
            if execution_profile["engagement_mode"] == "grasp_follow":
                self.move(self.close_gripper(arm_tag=arm, pos=0.0))
            if moved_negative():
                return {
                    "passed": False,
                    "failure_code": "articulation_negative_progress",
                    "attempts": attempts,
                }
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
                return {
                    "passed": False,
                    "failure_code": "articulation_negative_progress",
                    "attempts": attempts,
                }
            runtime_state["contact_id"] = contact_id
            child_matrix = np.asarray(
                self._articulation_joint_motion(subject_key)[
                    "child"
                ].get_pose().to_transformation_matrix(),
                dtype=float,
            )
            runtime_state["reference"] = (
                np.linalg.inv(self._ee_matrix(arm)) @ child_matrix
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
            runtime_state["reference"] = None
            self._release_for_retry(arm)
            return {
                "passed": False,
                "failure_code": "articulation_contact_not_acquired",
                "attempts": attempts,
            }

        def command_segment(
            segment: Mapping[str, Any], target_qpos: float
        ) -> dict[str, Any]:
            contact_id = int(runtime_state["contact_id"])
            current_qpos = read_qpos()
            current_contact = self._articulation_contact_matrix(
                subject_key, contact_id, execution_profile
            )
            motion = self._articulation_joint_motion(subject_key)
            initial_ee = self._ee_matrix(arm)
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
                    target_poses.append(self._matrix_pose(target_ee))
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
            return {
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

        def release_retreat(segment: Mapping[str, Any]) -> None:
            del segment
            remove_attachment()
            self._reset_planning()
            self.move(self.open_gripper(arm))
            if not self.plan_success:
                raise _SkillExecutionFailure(
                    "actuate_state", "articulation_release_unreachable"
                )
            retreat = execution_profile.get("retreat") or {}
            xyz = [float(value) for value in retreat.get("xyz", [0.0, 0.0, 0.05])]
            self._reset_planning()
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
                raise _SkillExecutionFailure(
                    "actuate_state", "articulation_retreat_unreachable"
                )
            runtime_state["reference"] = None

        callbacks = ArticulationRuntimeCallbacks(
            read_qpos=read_qpos,
            acquire=acquire,
            command_step=None,
            observe=observe,
            release_retreat=release_retreat,
            settle_steps=lambda steps: [self.scene.step() for _ in range(int(steps))],
            command_segment=command_segment,
        )
        result = run_articulation_closed_loop(
            state=state,
            initial_qpos=float(self.initial_articulations[subject_key]),
            lower=float(limits[0]),
            upper=float(limits[1]),
            state_profile=state_profile,
            execution_profile=execution_profile,
            contract=self._runtime_success_contract(),
            runtime=callbacks,
        )
        self.plan_success = bool(result["success"])
        return result

    def _snapshot_stability_contract(
        self,
        stability: Mapping[str, Any],
        *,
        tracked_components: Sequence[tuple[str, str]] | None,
    ) -> dict[str, Any]:
        """Aggregate component stability into one SuccessSnapshot entity row."""

        result = deepcopy(dict(stability))
        result["component_policy_version"] = STABILITY_COMPONENT_POLICY_VERSION
        entity_stability: dict[str, dict[str, Any]] = {}
        existing = result.get("entities")
        if tracked_components is None and isinstance(existing, Mapping):
            entity_stability.update(
                {
                    str(key): deepcopy(dict(row))
                    for key, row in existing.items()
                    if isinstance(row, Mapping)
                }
            )

        per_entity = result.get("per_entity") or []
        normalized_per_entity: list[dict[str, Any]] = []
        component_rows: dict[str, dict[str, dict[str, Any]]] = {}
        for index, row in enumerate(per_entity):
            if not isinstance(row, Mapping):
                continue
            if (
                tracked_components is not None
                and index < len(tracked_components)
            ):
                key, component_key = tracked_components[index]
            else:
                key = str(row.get("entity_key") or "")
                component_key = str(row.get("component_key") or "root")
            normalized_row = deepcopy(dict(row))
            if key in self.entities:
                normalized_row["entity_key"] = key
                normalized_row["component_key"] = component_key
                component_rows.setdefault(key, {})[component_key] = (
                    normalized_row
                )
            normalized_per_entity.append(normalized_row)
        result["per_entity"] = normalized_per_entity

        if tracked_components is not None:
            expected_components: dict[str, list[str]] = {}
            for key, component_key in tracked_components:
                expected_components.setdefault(key, []).append(component_key)
        else:
            expected_components = {
                key: list(rows) for key, rows in component_rows.items()
            }

        for key, expected in expected_components.items():
            observed = component_rows.get(key, {})
            expected_unique = list(dict.fromkeys(expected))
            missing = [
                component_key
                for component_key in expected_unique
                if component_key not in observed
            ]
            unstable_components = sorted(
                component_key
                for component_key, row in observed.items()
                if not row.get("stable")
            )
            translation_values = [
                float(row["max_translation_m"])
                for row in observed.values()
                if row.get("max_translation_m") is not None
            ]
            rotation_values = [
                float(row["max_rotation_deg"])
                for row in observed.values()
                if row.get("max_rotation_deg") is not None
            ]
            entity_row: dict[str, Any] = {
                "entity_key": key,
                "stable": not missing and not unstable_components,
                "component_count": len(expected_unique),
                "observed_component_count": len(observed),
                "missing_components": missing,
                "unstable_components": unstable_components,
                "components": deepcopy(observed),
            }
            if translation_values:
                entity_row["max_translation_m"] = max(translation_values)
            if rotation_values:
                entity_row["max_rotation_deg"] = max(rotation_values)
            entity_stability[key] = entity_row

        for key in self.entities:
            if key in entity_stability:
                continue
            structurally_fixed = self.dynamics.get(key) not in {
                "dynamic",
                "articulated_fixed",
            }
            if not structurally_fixed:
                raise RuntimeError(
                    f"snapshot_stability_entity_missing:{key}"
                )
            entity_stability[key] = {
                "stable": True,
                "structurally_fixed": True,
                "stability_source": "structural",
            }
        unstable_entities = sorted(
            key
            for key, row in entity_stability.items()
            if not row.get("stable")
        )
        whole_scene_stable = not unstable_entities
        result["stable"] = whole_scene_stable
        result["unstable_entities"] = unstable_entities
        result["entities"] = entity_stability
        result["whole_scene"] = {"stable": whole_scene_stable}
        return result

    def snapshot(
        self,
        *,
        settle: bool,
        settle_config: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        stability = {"stable": True}
        if not settle:
            stability = deepcopy(
                getattr(self, "initial_stability", stability)
            )
        embodiment_profile = self._runtime_embodiment_profile()
        success_contract = self._runtime_success_contract()
        articulation_profiles: dict[str, dict[str, Any]] = {}
        gripper_observations = observe_actual_grippers(
            self.robot, embodiment_profile=embodiment_profile
        )
        active_contacts = self._success_active_contacts(
            embodiment_profile, gripper_observations
        )
        hold_window: dict[str, dict[str, Any]] = {}
        if settle:
            del settle_config
            policy = success_contract.section("stability")
            minimum_steps = int(policy["minimum_settle_steps"])
            window_steps = int(policy["observation_window_steps"])
            maximum_steps = max(
                int(policy["maximum_settle_steps"]),
                minimum_steps + window_steps,
            )
            tracked = self._stability_components()
            self._rollout_video_stage = "settle"
            for step_index in range(minimum_steps):
                self.scene.step()
                self._capture_settle_step(step_index)
            steps = minimum_steps
            while tracked and steps + window_steps <= maximum_steps:
                snapshots: list[list[dict[str, Any]]] = []
                hold_window = {
                    arm: {
                        "subject": self.held_entities.get(arm),
                        "consecutive_bilateral": 0,
                        "maximum_bilateral": 0,
                        "relative": [],
                    }
                    for arm in ("left", "right")
                }
                for step_index in range(window_steps):
                    self.scene.step()
                    self._capture_settle_step(steps + step_index)
                    snapshots.append(
                        [
                            {
                                "p": actor.get_pose().p.tolist(),
                                "q": actor.get_pose().q.tolist(),
                            }
                            for _, _, actor in tracked
                        ]
                    )
                    gripper_observations = observe_actual_grippers(
                        self.robot, embodiment_profile=embodiment_profile
                    )
                    active_contacts = self._success_active_contacts(
                        embodiment_profile, gripper_observations
                    )
                    for arm, row in hold_window.items():
                        subject = row["subject"]
                        if not subject or subject not in self.entities:
                            continue
                        fingers = {
                            contact.get("finger")
                            for contact in active_contacts
                            if {
                                contact.get("entity_a"),
                                contact.get("entity_b"),
                            }
                            == {subject, f"gripper:{arm}"}
                            and contact.get("region") == "inner_pad"
                        }
                        row["consecutive_bilateral"] = (
                            int(row["consecutive_bilateral"]) + 1
                            if len(fingers) >= 2
                            else 0
                        )
                        row["maximum_bilateral"] = max(
                            int(row["maximum_bilateral"]),
                            int(row["consecutive_bilateral"]),
                        )
                        ee_matrix = self._hold_ee_matrix(self._arm(arm))
                        actor_matrix = self.entities[
                            subject
                        ].get_pose().to_transformation_matrix()
                        row["relative"].append(
                            ee_to_actor_transform(ee_matrix, actor_matrix)
                        )
                steps += window_steps
                stability = stability_from_snapshots(snapshots)
                stability = self._apply_axisymmetric_stability(
                    stability, snapshots, tracked
                )
                stability["settle_steps"] = steps
                if stability.get("stable"):
                    break
        stability = self._snapshot_stability_contract(
            stability,
            tracked_components=(
                [
                    (key, component_key)
                    for key, component_key, _ in tracked
                ]
                if settle
                else None
            ),
        )
        poses: dict[str, Any] = {}
        articulations: dict[str, Any] = {}
        aabbs: dict[str, Any] = {}
        entity_rows: dict[str, Any] = {}
        for key, actor in self.entities.items():
            pose = actor.get_pose()
            poses[key] = {
                "position": pose.p.tolist(),
                "quat_wxyz": pose.q.tolist(),
            }
            aabbs[key] = collision_aabb_tight(actor)
            entity_rows[key] = {
                "pose": poses[key],
                "aabb": aabbs[key],
                "support_plane_z": float(
                    self.initial_success_entities[key]["support_plane_z"]
                ),
            }
            if self.dynamics[key] == "articulated_fixed":
                qpos = np.asarray(actor.get_qpos(), dtype=float)
                joint_index = self.articulation_joint_indices[key]
                limits = np.asarray(actor.get_qlimits(), dtype=float)[joint_index]
                state_profile: dict[str, Any] | None = None
                entity_spec = self.scene_spec["entities"][key]
                for candidate_use in ("open", "switch_off"):
                    try:
                        execution_profile = resolve_articulation_execution_profile(
                            str(entity_spec["asset"]),
                            str(entity_spec["model_id"]),
                            candidate_use,
                            contract=success_contract,
                        )
                    except ArticulationExecutionError:
                        continue
                    state_profile = {
                        "profile_version": "benchmarkv1_articulation_state_v1",
                        "positive_use": execution_profile["positive_use"],
                        "source_execution_profile_hash": execution_profile[
                            "profile_hash"
                        ],
                        "source_success_contract_hash": success_contract.hash,
                    }
                    break
                articulation_row = {
                    "qpos": float(qpos[joint_index]),
                    "lower": float(limits[0]),
                    "upper": float(limits[1]),
                }
                if state_profile is not None:
                    state_profile_hash = artifact_hash(state_profile)
                    articulation_row.update(
                        {
                            "state_profile": state_profile,
                            "state_profile_hash": state_profile_hash,
                        }
                    )
                    articulation_profiles[key] = {
                        **state_profile,
                        "profile_hash": state_profile_hash,
                    }
                articulations[key] = articulation_row
        contacts: list[list[str]] = []
        for left_index, left_key in enumerate(sorted(self.entities)):
            for right_key in sorted(self.entities)[left_index + 1 :]:
                metrics = contact_metrics(
                    self.scene,
                    self.entities[left_key].actor,
                    self.entities[right_key].actor,
                )
                if metrics.get("active"):
                    contacts.append([left_key, right_key])
        held_by: dict[str, str] = {}
        for arm_name in ("left", "right"):
            arm = self._arm(arm_name)
            expected = self.held_entities.get(arm_name)
            reference = self.hold_references.get(arm_name) or {}
            if (
                not expected
                or expected not in self.entities
                or reference.get("subject") != expected
            ):
                continue
            expected_relative = np.asarray(
                reference.get("ee_to_actor_transform") or [], dtype=float
            )
            require_hold_ee_frame(str(reference.get("ee_frame") or ""))
            current_relative = np.asarray(
                ee_to_actor_transform(
                    self._hold_ee_matrix(arm),
                    self._pose_matrix(expected),
                ),
                dtype=float,
            )
            if (
                expected_relative.shape == (4, 4)
                and float(
                    np.linalg.norm(
                        current_relative[:3, 3]
                        - expected_relative[:3, 3]
                    )
                )
                <= float(
                    success_contract.section("hold")[
                        "relative_translation_drift_m"
                    ]
                )
            ):
                held_by[arm_name] = expected
        continuity_minimum = int(
            embodiment_profile["classification"][
                "continuous_contact_min_steps"
            ]
        )
        for arm, row in hold_window.items():
            subject = row.get("subject")
            relative = row.get("relative") or []
            if not subject or subject not in self.entities or not relative:
                continue
            state = gripper_observations[arm]
            if int(row["maximum_bilateral"]) >= continuity_minimum:
                state["closure_evidence"] = {
                    "subject": subject,
                    "valid": True,
                    "path": "bilateral_inner_pad",
                    "actual_width_m": state["actual_width_m"],
                    "continuous_steps": int(row["maximum_bilateral"]),
                }
            reference = self.hold_references.get(arm)
            state["hold_consistency"] = hold_consistency_from_relative_history(
                relative,
                subject=str(subject),
                ee_frame=HOLD_EE_FRAME_ARM,
                reference=reference if isinstance(reference, Mapping) else None,
                symmetry_axis=self._hold_symmetry_axis(str(subject)),
            )
            attachment = self.pick_attachments.get(arm)
            certified = (
                reference.get("certified_closure_evidence")
                if isinstance(reference, Mapping)
                else None
            )
            if isinstance(attachment, SapienFixedAttachment):
                attachment_result = self._observe_pick_attachment(self._arm(arm))
                if attachment_result["passed"] and isinstance(certified, Mapping):
                    state["closure_evidence"] = {
                        **deepcopy(dict(certified)),
                        "path": "post_close_fixed_attachment",
                        "actual_width_m": float(state["actual_width_m"]),
                    }
                    state["post_close_attachment"] = {
                        **deepcopy(attachment_result),
                        "subject": str(subject),
                        "certified_close": True,
                    }
        terminal_placements: dict[str, Any] = {}
        terminal_profiles: dict[str, Any] = {}
        reviewed_profiles = load_reviewed_receiver_profiles()
        for unit in self.unit_artifact.get("units", []):
            for region in (unit.get("params") or {}).get(
                "reserved_regions", []
            ):
                relation = str(region.get("relation") or "")
                subject = str(
                    region.get("subject_key")
                    or region.get("subject")
                    or ""
                )
                receivers = [
                    str(value)
                    for value in region.get("receiver_keys", [])
                ]
                binding = terminal_recipe_binding(
                    region,
                    relation,
                    receiver_profiles=reviewed_profiles,
                )
                profile = binding.get("receiver_state_profile")
                if profile is None:
                    continue
                if (
                    subject not in entity_rows
                    or len(receivers) != 1
                    or receivers[0] not in self.entities
                ):
                    raise RuntimeError(
                        "articulated_terminal_snapshot_binding_invalid:"
                        f"{subject}:{relation}:{receivers}"
                    )
                receiver = receivers[0]
                actor = self.entities[receiver]
                _, resolved_targets = resolve_joint_targets(
                    dict(profile), actor.get_qlimits(), actor.get_qpos()
                )
                receiver_state = articulation_state_metrics(
                    final_qpos=np.asarray(
                        actor.get_qpos(), dtype=float
                    ).tolist(),
                    resolved_joint_targets=resolved_targets,
                )
                stability_entities = stability.get("entities") or {}
                subject_stability = stability_entities.get(subject) or {}
                receiver_stability = stability_entities.get(receiver) or {}
                key = terminal_observation_key(
                    subject, relation, receivers
                )
                terminal_placements[key] = {
                    "subject": subject,
                    "relation": relation,
                    "receiver_keys": receivers,
                    "placement_recipe_binding": {
                        name: deepcopy(binding[name])
                        for name in (
                            "placement_recipe",
                            "placement_recipe_source",
                            "placement_recipe_hash",
                            "receiver_state_profile_id",
                            "receiver_state_profile_hash",
                        )
                    },
                    "cavity_world_matrix": (
                        articulated_cavity_world_matrix(
                            actor, profile
                        ).round(12).tolist()
                    ),
                    "cavity": deepcopy(profile["cavity"]),
                    "articulation_state": receiver_state,
                    "terminal_stability": {
                        "stable": bool(subject_stability.get("stable"))
                        and bool(receiver_stability.get("stable")),
                        "subject": deepcopy(subject_stability),
                        "receiver": deepcopy(receiver_stability),
                    },
                }
                profile_id = str(binding["receiver_state_profile_id"])
                terminal_profiles[f"terminal_receiver:{profile_id}"] = {
                    **deepcopy(dict(profile)),
                    "profile_hash": str(
                        binding["receiver_state_profile_hash"]
                    ),
                }
        return {
            "entities": entity_rows,
            "poses": poses,
            "aabbs": aabbs,
            "initial_poses": deepcopy(self.initial_pose_state),
            "articulations": articulations,
            "initial_articulations": deepcopy(self.initial_articulations),
            "held_by": held_by,
            "released_arms": sorted(self.released_arms),
            "action_provenance": {
                "primary_arm": self.primary_execution_arm,
                "primary_op": self.primary_execution_op,
                "primary_event_hash": self.primary_execution_event_hash,
                "participating_arms": sorted(self.participating_arms),
            },
            # The worker carries this field into SuccessSnapshot.terminal.events.
            # Click/Press provenance and the predicate therefore consume the
            # same immutable physical event rather than parallel booleans.
            "events": deepcopy(self.execution_events),
            "contacts": contacts,
            "grippers": gripper_observations,
            "active_contacts": active_contacts,
            "terminal_placements": terminal_placements,
            # This marker survives independently of the observations so the
            # SuccessContract cannot silently fall back to whole-receiver
            # AABB containment when an articulated cavity observation is
            # missing or removed.
            "required_terminal_placement_keys": sorted(terminal_placements),
            "profiles": {
                "embodiment": embodiment_profile,
                **{
                    f"articulation:{key}": profile
                    for key, profile in articulation_profiles.items()
                },
                **terminal_profiles,
            },
            "stable": bool(stability.get("stable")),
            "stability": stability,
        }


class IFSceneValidationRuntime:
    """Worker-facing lifecycle and evidence-sidecar wrapper."""

    def __init__(
        self,
        joined: Mapping[str, Any],
        *,
        evidence_dir: Path,
        config: Mapping[str, Any],
        reusable_env_cache: dict[str, Any] | None = None,
    ) -> None:
        self.joined = deepcopy(dict(joined))
        self.evidence_dir = evidence_dir
        self.config = deepcopy(dict(config))
        self._reusable_env_cache = reusable_env_cache
        self.env: if_scene_validation | None = None
        self.current_seed = 0
        self._evidence: dict[str, Any] = {}
        self._current_video: dict[str, Any] | None = None
        self._expert_capture: Any = None
        self._current_expert_capture: dict[str, Any] | None = None
        self._episode_source_identity: str | None = None
        self._pristine_state_hash: str | None = None
        self._episode_action_epoch = 0
        self._prepared_initial_available = False

    @staticmethod
    def _prepared_source_identity(
        joined: Mapping[str, Any], config: Mapping[str, Any]
    ) -> str:
        # Route reports are produced between setup and M15; filesystem paths
        # may change without changing a joined artifact. Everything else,
        # including complete task payloads and the exact Pick witness, binds.
        return artifact_hash({
            "joined": {key: value for key, value in joined.items()
                       if key not in {"routes", "paths"}},
            "config": config,
        })

    def _physical_state_hash(self) -> str:
        """Read physical state without stepping, settling, or planning."""
        if self.env is None:
            raise RuntimeError("prepared_runtime_not_setup")

        def vector(value: Any) -> list[Any]:
            array = np.asarray(value, dtype=float)
            if not np.isfinite(array).all():
                raise RuntimeError("prepared_runtime_nonfinite_physical_state")
            return array.tolist()

        def body_state(body: Any) -> dict[str, Any]:
            pose = body.get_pose()
            state: dict[str, Any] = {
                "position": vector(pose.p), "quaternion": vector(pose.q),
            }
            for field in ("qpos", "qvel", "velocity", "angular_velocity"):
                getter = getattr(body, f"get_{field}", None)
                if callable(getter):
                    state[field] = vector(getter())
                elif hasattr(body, field):
                    state[field] = vector(getattr(body, field))
            # SAPIEN 3 rigid velocities live on the entity's PhysX component.
            components = []
            for component in getattr(body, "components", ()):
                item = {}
                for field in ("linear_velocity", "angular_velocity"):
                    if hasattr(component, field):
                        item[field] = vector(getattr(component, field))
                if item:
                    components.append(item)
            if components:
                state["components"] = components
            get_joints = getattr(body, "get_active_joints", None)
            if callable(get_joints):
                state["drive_targets"] = [
                    {
                        field: vector(getattr(joint, f"get_{field}")())
                        for field in ("drive_target", "drive_velocity_target")
                        if callable(getattr(joint, f"get_{field}", None))
                    }
                    for joint in get_joints()
                ]
            return state

        return artifact_hash({
            "robot": {
                arm: body_state(getattr(self.env.robot, f"{arm}_entity"))
                for arm in ("left", "right")
            },
            "entities": {
                str(key): body_state(getattr(actor, "actor", actor))
                for key, actor in sorted(self.env.entities.items())
            },
            "action_count": int(getattr(self.env, "take_action_cnt", 0)),
        })

    def assert_pristine_for_planning(
        self, *, joined: Mapping[str, Any], config: Mapping[str, Any]
    ) -> None:
        """Reject a foreign or already-consumed scene before borrowed planning."""
        if (self.env is None or self.current_seed != 0
                or self._pristine_state_hash is None
                or self._episode_action_epoch != 0
                or self.joined.get("planning_only") is True):
            raise RuntimeError("prepared_runtime_not_pristine")
        expected = self._prepared_source_identity(joined, config)
        if (
            self._episode_source_identity != expected
            or self._prepared_source_identity(self.joined, self.config) != expected
            or self._prepared_source_identity(self.env.joined, self.config) != expected
        ):
            raise RuntimeError("prepared_runtime_source_or_config_mismatch")
        if self._physical_state_hash() != self._pristine_state_hash:
            raise RuntimeError("prepared_runtime_physical_state_changed")

    def bind_prepared_batch(
        self, *, joined: Mapping[str, Any], config: Mapping[str, Any],
        evidence_dir: Path, reusable_env_cache: dict[str, Any] | None = None,
    ) -> None:
        """Bind newly validated route records without rebuilding the episode."""
        self.assert_pristine_for_planning(joined=joined, config=config)
        if reusable_env_cache is not None:
            if (self._reusable_env_cache is not None
                    and self._reusable_env_cache is not reusable_env_cache):
                raise RuntimeError("prepared_runtime_cache_owner_mismatch")
            self._reusable_env_cache = reusable_env_cache
        self.joined = deepcopy(dict(joined))
        assert self.env is not None
        self.env.joined = deepcopy(dict(joined))
        self.evidence_dir = Path(evidence_dir)
        self._evidence = {}
        self._current_video = None
        self._prepared_initial_available = True

    def consume_prepared_initial_scene(
        self, seed: int, *, diagnostic_video_render: bool = False,
    ) -> bool:
        """Only the first matching attempt may use the prepared initial state."""
        available = self._prepared_initial_available
        self._prepared_initial_available = False
        if not available or int(seed) != self.current_seed:
            return False
        self.assert_pristine_for_planning(joined=self.joined, config=self.config)
        assert self.env is not None
        # A diagnostic reset can change render resource initialization. Do not
        # silently upgrade a renderless/different setup in a reused episode.
        if bool(diagnostic_video_render) != bool(self.env.diagnostic_video_render):
            return False
        self._episode_action_epoch += 1
        return True

    def _reusable_robot_identity(self) -> str:
        return artifact_hash({
            "actor": self.joined["scene"]["scene_spec"].get("actor"),
            "config": self.config,
            "planning_only": self.joined.get("planning_only") is True,
            "planning_arms": self.joined.get("planning_arms"),
        })

    def _runtime_args(self) -> dict[str, Any]:
        # This helper prepares the frozen demo_clean embodiment without importing
        # it in the simulator-independent planner or aggregator.
        from if_benchmark.asset_probe_worker import _build_runtime_args

        planning_only = self.joined.get("planning_only") is True
        local_runtime_root = os.environ.get("M12_PLANNING_RUNTIME_ROOT")
        if planning_only:
            runtime_root = (
                Path(local_runtime_root)
                if local_runtime_root
                else self.evidence_dir / "runtime"
            )
        else:
            import tempfile

            # Generated embodiment YAMLs are operational scratch, not evidence.
            # Keep their path stable for a persistent lane's reusable Robot,
            # without rewriting them on the shared collection filesystem.  An
            # explicit /tmp also avoids a shared TMPDIR supplied by a launcher.
            # The cache owns cleanup on eviction/process exit; standalone
            # runtimes own cleanup after their physical environment closes.
            if self._reusable_env_cache is not None:
                directory = self._reusable_env_cache.get("_m15_runtime_directory")
                if directory is None:
                    directory = tempfile.TemporaryDirectory(prefix="m15-runtime-", dir="/tmp")
                    self._reusable_env_cache["_m15_runtime_directory"] = directory
            else:
                directory = getattr(self, "_m15_runtime_directory", None)
                if directory is None:
                    directory = tempfile.TemporaryDirectory(prefix="m15-runtime-", dir="/tmp")
                    self._m15_runtime_directory = directory
            runtime_root = Path(directory.name)
        args = _build_runtime_args("demo_clean", str(runtime_root))
        args["task_name"] = "if_scene_validation"
        args["domain_randomization"] = {
            "random_background": False,
            "cluttered_table": False,
            "clean_background_rate": 1.0,
            "random_head_camera_dis": 0.0,
            "random_table_height": 0.0,
            "random_light": False,
            "crazy_random_light_rate": 0.0,
            "random_embodiment": False,
        }
        if planning_only:
            planning_arms = tuple(
                sorted(
                    {
                        str(value)
                        for value in self.joined.get("planning_arms") or []
                    }
                )
            )
            if not planning_arms or any(
                arm not in {"left", "right"} for arm in planning_arms
            ):
                raise RuntimeError("m12_planning_arms_invalid")
            # The M12 oracle consumes CuRobo paths directly.  It does not use
            # Mplib/TOPP, and it must not construct a planner for an arm that
            # is absent from the exact request.  Most current M12 requests are
            # left-only; the old dual-planner setup doubled CUDA/CPU work for
            # them without adding evidence.
            args["need_topp"] = False
            args["curobo_planner_arms"] = list(planning_arms)
        return args

    def _create(self, seed: int, *, diagnostic_video_render: bool = False) -> None:
        self.close()
        self._current_expert_capture = None
        self.current_seed = int(seed)
        self._episode_source_identity = None
        self._pristine_state_hash = None
        self._episode_action_epoch = 0
        self._prepared_initial_available = False
        robot_identity = self._reusable_robot_identity()
        cached_env = (
            None
            if self._reusable_env_cache is None
            else self._reusable_env_cache.pop("env", None)
        )
        self.env = (
            cached_env
            if isinstance(cached_env, if_scene_validation)
            and (self.joined.get("planning_only") is True
                 or self._reusable_env_cache.get("robot_identity") == robot_identity)
            else if_scene_validation()
        )
        timing_enabled = (
            (self.joined.get("planning_only") is True
             and os.environ.get("M12_PLANNING_TIMINGS") == "1")
            or os.environ.get("M15_PROFILE_TIMINGS") == "1"
        )
        if timing_enabled:
            self.env._if_probe_timings = {}
        try:
            self.env.setup_demo(
                joined=self.joined,
                seed=int(seed),
                skill_execution_config=self.config.get(
                    "skill_execution", {}
                ),
                stability_config=(
                    self.config.get("rollout", {}).get("settle", {})
                ),
                diagnostic_video_render=bool(diagnostic_video_render),
                m15_single_pass_contacts=self.joined.get("planning_only") is not True,
                m15_strict_pregrasp_collision=self.joined.get("planning_only") is not True,
                m15_corrected_carry_collision=self.joined.get("planning_only") is not True,
                now_ep_num=0,
                task_variant=None,
                **self._runtime_args(),
            )
            if timing_enabled:
                timing_prefix = "m12" if self.joined.get("planning_only") is True else "m15"
                print(
                    f"[{timing_prefix}-runtime-fast-path] "
                    + json.dumps(
                        self.env.m12_runtime_fast_path,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                print(
                    f"[{timing_prefix}-setup-timing] "
                    + json.dumps(
                        self.env._if_probe_timings,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    file=sys.stderr,
                    flush=True,
                )
            self._episode_source_identity = self._prepared_source_identity(
                self.joined, self.config
            )
            # M12 planning-only clients do not use the route/M15 lifecycle.
            if self.joined.get("planning_only") is not True:
                self._pristine_state_hash = self._physical_state_hash()
        except UnStableError as exc:
            stability = getattr(self.env, "initial_stability", {})
            detail = json.dumps(
                stability,
                sort_keys=True,
                separators=(",", ":"),
            )
            raise SceneValidationError(
                "initial_scene_unstable",
                f"{exc}; stability={detail}",
                stage="semantic",
                scene_spec_id=str(
                    self.joined["scene"]["scene_spec"].get("id") or ""
                ),
            ) from exc
        except InitialPoseContractError as exc:
            raise SceneValidationError(
                "source_initial_pose_mismatch",
                str(exc),
                stage="source",
                scene_spec_id=str(
                    self.joined["scene"]["scene_spec"].get("id") or ""
                ),
            ) from exc
        except RuntimeError as exc:
            if "source_initial_pose" not in str(exc):
                raise
            raise SceneValidationError(
                "source_initial_pose_mismatch",
                str(exc),
                stage="source",
                scene_spec_id=str(
                    self.joined["scene"]["scene_spec"].get("id") or ""
                ),
            ) from exc

    def setup(self, *, diagnostic_video_render: bool = False) -> None:
        self._create(0, diagnostic_video_render=diagnostic_video_render)

    def render_descriptor_frame(self) -> dict[str, Any]:
        assert self.env is not None
        frame = self.env.descriptor_frame()
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        rgb_path = self.evidence_dir / "head_rgb.png"
        depth_path = self.evidence_dir / "head_depth.npy"
        segmentation_path = self.evidence_dir / "head_actor_segmentation.npy"
        from PIL import Image

        Image.fromarray(np.asarray(frame["rgb"], dtype=np.uint8)).save(rgb_path)
        np.save(depth_path, np.asarray(frame["depth"], dtype=np.float32))
        np.save(
            segmentation_path,
            np.asarray(frame["actor_segmentation"], dtype=np.int64),
        )
        self._evidence = {
            "rgb": _sidecar(rgb_path),
            "depth": _sidecar(depth_path),
            "actor_segmentation": _sidecar(segmentation_path),
            "canonical_validation": deepcopy(self.env.canonical_evidence),
        }
        frame["frame_hash"] = hashlib.sha256(
            json.dumps(
                self._evidence,
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return frame

    def reset(
        self, seed: int, *, diagnostic_video_render: bool = False
    ) -> None:
        # Recreating the simulator guarantees the seed cannot perturb layout:
        # load_actors always applies the frozen SceneSpec poses.
        self._create(
            int(seed),
            diagnostic_video_render=bool(diagnostic_video_render),
        )

    def _video_path(self, task_record_id: str, seed: int) -> Path:
        return (
            self.evidence_dir
            / "videos"
            / str(task_record_id)
            / f"seed_{int(seed)}.mp4"
        )

    def begin_attempt_video(
        self,
        task: Mapping[str, Any],
        spec: Mapping[str, Any],
    ) -> None:
        assert self.env is not None
        goal = task.get("goal") or {}
        skill = task.get("skill_config") or {}
        label = (
            f"{spec['task_record_id']} seed={int(spec['seed'])} "
            f"{goal.get('verb') or goal.get('action_family') or 'task'} "
            f"arm={skill.get('execution_arm') or goal.get('arm') or '-'}"
        )
        self._current_video = None
        self.env.begin_rollout_video(
            self._video_path(
                str(spec["task_record_id"]), int(spec["seed"])
            ),
            label=label,
        )

    def end_attempt_video(self, *, outcome: str) -> dict[str, Any]:
        assert self.env is not None
        self._current_video = self.env.finish_rollout_video(
            outcome=outcome
        )
        return deepcopy(self._current_video)

    def capture_setup_failure_video(
        self,
        task: Mapping[str, Any],
        spec: Mapping[str, Any],
        *,
        outcome: str,
    ) -> dict[str, Any]:
        self.begin_attempt_video(task, spec)
        return self.end_attempt_video(outcome=outcome)

    def initial_snapshot(self) -> dict[str, Any]:
        assert self.env is not None
        return self.env.snapshot(settle=False)

    def begin_expert_capture(
        self,
        task: Mapping[str, Any],
        spec: Mapping[str, Any],
        options: Mapping[str, Any] | None = None,
    ) -> None:
        """Begin after t0 admission, immediately before this attempt executes.

        Only this Scene instance is instrumented. The wrapper observes every
        ordinary action, gripper closure, attachment settle and terminal
        settle without modifying any of their existing execution paths.
        """
        from if_benchmark.scene_validation.expert_capture import (
            ExpertCapture, ExpertCaptureError, RobotStateReader,
            capture_options, capture_selected_rgb, runtime_capture_metadata,
        )

        if self.env is None or self.joined.get("planning_only") is True:
            raise ExpertCaptureError("expert_capture_execution_runtime_required")
        if self._expert_capture is not None:
            raise ExpertCaptureError("expert_capture_already_active")
        if self.current_seed != spec.get("seed"):
            raise ExpertCaptureError("expert_capture_runtime_seed_mismatch")
        self._current_expert_capture = None
        env = self.env
        cameras = env.cameras
        if cameras is None:
            raise ExpertCaptureError("expert_capture_cameras_unavailable")
        configured = set(getattr(cameras, "static_camera_name", ()))
        if not getattr(cameras, "collect_head_camera", False):
            configured.discard("head_camera")
        if getattr(cameras, "collect_wrist_camera", False):
            for name in ("left_camera", "right_camera"):
                if getattr(cameras, name, None) is not None:
                    configured.add(name)
        selected = capture_options(options)
        for name in selected["cameras"]:
            if name not in configured:
                raise ExpertCaptureError(f"expert_capture_camera_missing:{name}")
        selective_capture = os.environ.get("M15_SELECTIVE_CAMERA_CAPTURE", "1") != "0"
        rgb_readback = os.environ.get("M15_RGB_READBACK", "cpu")
        verify_readback = os.environ.get("M15_VERIFY_RGB_READBACK", "0") == "1"
        if rgb_readback not in {"cpu", "cuda"}:
            raise ExpertCaptureError(f"expert_capture_readback_invalid:{rgb_readback}")
        batch_capture = os.environ.get("M15_BATCH_CAMERA_RENDER", "0") == "1"
        scene_batch = None
        if batch_capture:
            from if_benchmark.scene_validation.expert_capture import selected_rgb_sensors
            from if_benchmark.scene_validation.render_batch import SceneCameraBatch

            scene_batch = getattr(env, "_m15_scene_camera_batch", None)
            if scene_batch is None:
                scene_batch = SceneCameraBatch(env.scene, selected_rgb_sensors(cameras, selected["cameras"]))
                env._m15_scene_camera_batch = scene_batch
                scene_batch.initialize()
            elif set(scene_batch.sensors) != set(selected["cameras"]):
                raise ExpertCaptureError("expert_batch_camera_selection_changed")

        def read_images() -> dict[str, np.ndarray]:
            # Do not call Base_Task._update_render: that helper can consume
            # physics-task RNG for light randomization. Sensor sync/render
            # itself neither advances physics nor changes scene semantics.
            if scene_batch is not None:
                with scene_batch.frame_scope():
                    cameras.update_wrist_camera(
                        env.robot.left_camera.get_pose(), env.robot.right_camera.get_pose())
                    env.scene.update_render()
                    return scene_batch.read_rgb(verify=verify_readback)
            cameras.update_wrist_camera(
                env.robot.left_camera.get_pose(), env.robot.right_camera.get_pose())
            env.scene.update_render()
            if selective_capture:
                return capture_selected_rgb(cameras, selected["cameras"],
                                            readback=rgb_readback, verify=verify_readback)
            cameras.update_picture()
            return {name: row["rgb"] for name, row in cameras.get_rgb().items()}

        metadata = runtime_capture_metadata(self.joined, task, spec)
        share_review = (options or {}).get(
            "review_from_expert", os.environ.get("M15_REVIEW_FROM_EXPERT", "1") == "1")
        if type(share_review) is not bool:
            raise ExpertCaptureError("expert_capture_review_mode_invalid")
        share_review = (share_review and env._rollout_video_writer is not None
                        and not getattr(env, "random_light", False)
                        and not getattr(env, "crazy_random_light", False)
                        and "head_camera" in selected["cameras"])
        metadata["review_video_backend"] = "expert_rgb_requests_v1" if share_review else "independent_head"
        if scene_batch is not None or (selective_capture and rgb_readback == "cuda"):
            from if_benchmark.scene_validation.expert_capture import CUDA_RGB_TRANSFER

            metadata["rgb_transfer_backend"] = CUDA_RGB_TRANSFER
        if scene_batch is not None:
            from if_benchmark.scene_validation.render_batch import BACKEND

            metadata["render_backend"] = BACKEND
        # The exact external attempt id stays in metadata; a digest is a safe
        # directory component even if the caller's id contains separators.
        directory = hashlib.sha256(str(spec["attempt_id"]).encode("utf-8")).hexdigest()
        capture = ExpertCapture(
            env.scene, output_dir=self.evidence_dir / "expert_staging" / directory,
            state_reader=RobotStateReader(env.robot), read_images=read_images,
            metadata=metadata, options=selected,
            frame_observer=env._record_expert_review_frame if share_review else None,
        )
        capture.begin()
        self._expert_capture = capture
        env._m15_expert_review_active = share_review
        env._m15_expert_review_pending = 0

    def can_finalize_expert_prefix(self) -> bool:
        capture = self._expert_capture
        return capture is not None and capture.can_finalize_prefix()

    def end_expert_capture(self, *, outcome: Any) -> dict[str, Any]:
        from if_benchmark.scene_validation.expert_capture import ExpertCaptureError

        capture = self._expert_capture
        if capture is None:
            raise ExpertCaptureError("expert_capture_not_active")
        try:
            self._current_expert_capture = capture.end(outcome=outcome)
            return deepcopy(self._current_expert_capture)
        except TimeoutError:
            # end() seals any retained incomplete frames before propagating
            # the alarm. Preserve their receipt without another render/close.
            self._current_expert_capture = capture.finalized_result()
            raise
        finally:
            self._expert_capture = None
            if self.env is not None:
                self.env._m15_expert_review_active = False
                self.env._m15_expert_review_pending = 0

    def execute_skill(
        self,
        skill_config: Mapping[str, Any],
        *,
        seed: int,
        checker_config: Mapping[str, Any] | None = None,
        task_record_id: str | None = None,
    ) -> dict[str, Any]:
        assert self.env is not None and seed == self.current_seed
        self._episode_action_epoch += 1
        route_report = (
            None
            if task_record_id is None
            else (self.joined.get("routes") or {}).get(
                str(task_record_id)
            )
        )
        return self.env.execute_skill_config(
            skill_config,
            checker_config=checker_config,
            route_report=route_report,
        )

    def terminal_snapshot(
        self, task: Mapping[str, Any], initial: Mapping[str, Any]
    ) -> dict[str, Any]:
        del task, initial
        assert self.env is not None
        self._episode_action_epoch += 1
        return self.env.snapshot(
            settle=True,
            settle_config=self.config["rollout"]["settle"],
        )

    def evidence(self) -> dict[str, Any]:
        value = deepcopy(self._evidence)
        if self.env is not None:
            value["canonical_validation"] = deepcopy(
                self.env.canonical_evidence
            )
        if self._current_video is not None:
            value["rollout_video"] = deepcopy(self._current_video)
        if self._current_expert_capture is not None:
            value["expert_capture"] = deepcopy(self._current_expert_capture)
        return value

    def close(self) -> None:
        self._prepared_initial_available = False
        self._pristine_state_hash = None
        try:
            if self.env is None:
                return
            env = self.env
            try:
                try:
                    if self._expert_capture is not None:
                        self.end_expert_capture(outcome={"aborted": True, "classification": None})
                finally:
                    # A trajectory/manifest disk failure must not skip either
                    # diagnostic writer finalization or physical scene cleanup.
                    try:
                        if env._rollout_video_writer is not None:
                            self._current_video = env.finish_rollout_video(
                                outcome="ABORTED"
                            )
                    finally:
                        env.close_env()
            finally:
                if self._reusable_env_cache is not None:
                    self._reusable_env_cache["env"] = env
                    self._reusable_env_cache["robot_identity"] = self._reusable_robot_identity()
                self.env = None
        finally:
            directory = getattr(self, "_m15_runtime_directory", None)
            if directory is not None:
                directory.cleanup()
                self._m15_runtime_directory = None


def _sidecar(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }


__all__ = [
    "IFSceneValidationRuntime",
    "entity_dynamics",
    "if_scene_validation",
]
