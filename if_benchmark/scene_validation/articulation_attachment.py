"""SAPIEN runtime adapter for R09 grasp-follow fixed attachments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import sapien
import transforms3d as t3d

from if_benchmark.scene_validation.articulation_execution import (
    ArticulationExecutionError,
    fixed_attachment_frames,
    fixed_attachment_residual,
)
from if_benchmark.success_contract import (
    SuccessContract,
    evaluate_pick_attachment,
    pick_attachment_policy,
)


HARD_D6_TRANSLATION_STIFFNESS = 1000000.0
HARD_D6_TRANSLATION_DAMPING = 2000.0
HARD_D6_ROTATION_STIFFNESS = 1000000.0
HARD_D6_ROTATION_DAMPING = 2000.0


def articulation_attachment_status(
    metrics: Mapping[str, Any],
    *,
    maximum_translation_m: float,
    maximum_rotation_deg: float,
) -> dict[str, Any]:
    """Classify a live articulation attachment without a residual gate.

    Residuals remain observable telemetry for diagnosing the PhysX solver,
    but they are not an M4/M15 acceptance condition.  A grasp-follow
    articulation is interrupted only when its D6 attachment is no longer
    active/enabled; task progress and terminal state remain governed by the
    shared articulation success contract.
    """

    constraint_enabled = bool(metrics.get("constraint_enabled"))
    active = bool(metrics.get("attachment_active")) and constraint_enabled
    failure_code = None if active else "articulation_attachment_lost"
    return {
        "passed": active,
        "failure_code": failure_code,
        "attachment_active": active,
        "checks": {"constraint_enabled": constraint_enabled},
        "metrics": {
            **dict(metrics),
            "maximum_constraint_translation_residual_m": float(
                maximum_translation_m
            ),
            "maximum_constraint_rotation_residual_deg": float(
                maximum_rotation_deg
            ),
            "residual_policy": "diagnostic_only",
        },
    }


def _matrix(body: Any) -> np.ndarray:
    return np.asarray(body.get_pose().to_transformation_matrix(), dtype=float)


def _pose(matrix: Sequence[Sequence[float]]) -> sapien.Pose:
    value = np.asarray(matrix, dtype=float)
    return sapien.Pose(
        value[:3, 3], t3d.quaternions.mat2quat(value[:3, :3])
    )


@dataclass
class SapienFixedAttachment:
    """A hard D6 constraint whose two frames coincide at creation time."""

    drive: Any
    parent_body: Any
    child_body: Any
    parent_local_frame: np.ndarray
    child_local_frame: np.ndarray
    translation_drive_stiffness: float
    translation_drive_damping: float
    rotation_drive_stiffness: float
    rotation_drive_damping: float
    removed: bool = False

    @classmethod
    def create(
        cls,
        scene: Any,
        *,
        parent_body: Any,
        child_body: Any,
        anchor_world_matrix: Sequence[Sequence[float]],
        stiffness: float,
        damping: float,
    ) -> "SapienFixedAttachment":
        parent_local, child_local = fixed_attachment_frames(
            _matrix(parent_body),
            _matrix(child_body),
            anchor_world_matrix=anchor_world_matrix,
        )
        try:
            drive = scene.create_drive(
                parent_body,
                _pose(parent_local),
                child_body,
                _pose(child_local),
            )
            epsilon = 1e-6
            for setter_name in (
                "set_limit_x",
                "set_limit_y",
                "set_limit_z",
            ):
                # A non-zero spring on a PhysX D6 limit makes the limit
                # compliant.  Pick attachments need a hard zero-width lock;
                # the acceleration drives below remain only as numerical
                # stabilization around that lock.
                getattr(drive, setter_name)(-epsilon, epsilon)
            drive.set_limit_twist(
                -epsilon,
                epsilon,
            )
            drive.set_limit_pyramid(
                -epsilon,
                epsilon,
                -epsilon,
                epsilon,
            )
            if float(stiffness) > 0.0:
                # ``fixed_attachment_frames`` already makes the parent and
                # child anchors coincide at creation.  The desired D6-frame
                # relation is therefore identity.  ``drive.relative_pose`` is
                # the live body relation, not the desired anchor relation;
                # feeding it back here makes the stabilizing drive fight the
                # zero-width limits and permits a large rotational residual.
                drive.set_drive_target(sapien.Pose())
                for setter_name in (
                    "set_drive_property_x",
                    "set_drive_property_y",
                    "set_drive_property_z",
                ):
                    getattr(drive, setter_name)(
                        max(float(stiffness), HARD_D6_TRANSLATION_STIFFNESS),
                        max(float(damping), HARD_D6_TRANSLATION_DAMPING),
                        mode="acceleration",
                    )
                drive.set_drive_property_slerp(
                    max(float(stiffness), HARD_D6_ROTATION_STIFFNESS),
                    max(float(damping), HARD_D6_ROTATION_DAMPING),
                    mode="acceleration",
                )
        except Exception as exc:
            raise ArticulationExecutionError(
                "articulation_attachment_creation_failed"
            ) from exc
        return cls(
            drive=drive,
            parent_body=parent_body,
            child_body=child_body,
            parent_local_frame=parent_local,
            child_local_frame=child_local,
            translation_drive_stiffness=max(
                float(stiffness), HARD_D6_TRANSLATION_STIFFNESS
            ),
            translation_drive_damping=max(
                float(damping), HARD_D6_TRANSLATION_DAMPING
            ),
            rotation_drive_stiffness=max(
                float(stiffness), HARD_D6_ROTATION_STIFFNESS
            ),
            rotation_drive_damping=max(
                float(damping), HARD_D6_ROTATION_DAMPING
            ),
        )

    def metrics(self) -> dict[str, Any]:
        enabled_value = getattr(self.drive, "is_enabled", False)
        enabled = bool(
            enabled_value() if callable(enabled_value) else enabled_value
        )
        residual = fixed_attachment_residual(
            _matrix(self.parent_body),
            _matrix(self.child_body),
            self.parent_local_frame,
            self.child_local_frame,
        )
        return {
            "attachment_active": bool(enabled and not self.removed),
            "constraint_enabled": enabled,
            "constraint_translation_residual_m": residual["translation_m"],
            "constraint_rotation_residual_deg": residual["rotation_deg"],
            "translation_drive_stiffness": self.translation_drive_stiffness,
            "translation_drive_damping": self.translation_drive_damping,
            "rotation_drive_stiffness": self.rotation_drive_stiffness,
            "rotation_drive_damping": self.rotation_drive_damping,
        }

    def remove(self) -> None:
        if self.removed:
            return
        try:
            self.drive.disable()
        finally:
            entity = getattr(self.drive, "entity", None)
            remove = getattr(entity, "remove_component", None)
            if callable(remove):
                remove(self.drive)
        self.removed = True


@dataclass
class SapienWorldPoseLock:
    """Temporary hard D6 lock that holds one live rigid link in world space."""

    drive: Any
    body: Any
    removed: bool = False

    @classmethod
    def create(cls, scene: Any, *, body: Any) -> "SapienWorldPoseLock":
        try:
            drive = scene.create_drive(
                None,
                _pose(_matrix(body)),
                body,
                sapien.Pose(),
            )
            epsilon = 1e-6
            for setter_name in ("set_limit_x", "set_limit_y", "set_limit_z"):
                getattr(drive, setter_name)(-epsilon, epsilon)
            drive.set_limit_twist(-epsilon, epsilon)
            drive.set_limit_pyramid(-epsilon, epsilon, -epsilon, epsilon)
            drive.set_drive_target(sapien.Pose())
            for setter_name in (
                "set_drive_property_x",
                "set_drive_property_y",
                "set_drive_property_z",
            ):
                getattr(drive, setter_name)(
                    1000000.0,
                    2000.0,
                    mode="acceleration",
                )
            drive.set_drive_property_slerp(
                1000000.0,
                2000.0,
                mode="acceleration",
            )
        except Exception as exc:
            raise ArticulationExecutionError(
                "handover_source_pose_lock_creation_failed"
            ) from exc
        return cls(drive=drive, body=body)

    def remove(self) -> None:
        if self.removed:
            return
        try:
            self.drive.disable()
        finally:
            entity = getattr(self.drive, "entity", None)
            remove = getattr(entity, "remove_component", None)
            if callable(remove):
                remove(self.drive)
        self.removed = True


def create_evaluated_pick_attachment(
    task: Any,
    actor: Any,
    *,
    arm: str,
) -> tuple[SapienFixedAttachment | None, dict[str, Any]]:
    """Create and validate the one canonical M4/M15 Pick attachment."""

    policy = pick_attachment_policy(contract=SuccessContract.load())
    child_body = getattr(actor, "actor", actor)
    try:
        attachment = SapienFixedAttachment.create(
            task.scene,
            parent_body=physical_ee_link(task.robot, str(arm)),
            child_body=child_body,
            # Put the D6 anchor at the object's origin.  This is rigidly
            # equivalent to a wrist/contact anchor at zero residual, but under
            # finite solver residual it directly minimizes object translation
            # instead of magnifying angular error through a long wrist lever.
            anchor_world_matrix=_matrix(child_body),
            stiffness=float(policy["drive_stiffness"]),
            damping=float(policy["drive_damping"]),
        )
    except ArticulationExecutionError:
        return None, {
            "passed": False,
            "failure_code": "pick_attachment_creation_failed",
            "checks": {"attachment_created": False},
            "metrics": {"policy": policy},
        }
    result = evaluate_pick_attachment(
        attachment.metrics(), contract=SuccessContract.load()
    )
    result["checks"] = {
        "attachment_created": True,
        **dict(result["checks"]),
    }
    if not result["passed"]:
        attachment.remove()
        return None, result
    return attachment, result


def physical_ee_link(robot: Any, arm: str) -> Any:
    joint = robot.left_ee if str(arm) == "left" else robot.right_ee
    child = getattr(joint, "child_link", None)
    if child is None:
        getter = getattr(joint, "get_child_link", None)
        child = getter() if callable(getter) else None
    if child is None:
        raise ArticulationExecutionError("articulation_attachment_parent_missing")
    return child


def physical_ee_world_matrix(robot: Any, arm: str) -> np.ndarray:
    """Return the rigid link frame used as the fixed-attachment parent.

    Robot ``get_*_ee_pose`` is a planning frame derived from the terminal
    joint anchor.  That anchor can move relative to the joint's child link as
    the last revolute joint changes.  Hold consistency must instead use this
    physical body frame, which is exactly the frame constrained by the D6.
    """

    return _matrix(physical_ee_link(robot, str(arm)))


def boost_arm_drive(
    robot: Any,
    arm: str,
    *,
    stiffness: float,
    damping: float,
) -> list[tuple[Any, float, float, float, str]]:
    joints = (
        robot.left_arm_joints
        if str(arm) == "left"
        else robot.right_arm_joints
    )
    saved: list[tuple[Any, float, float, float, str]] = []
    for joint in joints:
        mode_value = joint.get_drive_mode()
        current_mode = str(mode_value)
        saved.append(
            (
                joint,
                float(joint.get_stiffness()),
                float(joint.get_damping()),
                float(joint.get_force_limit()),
                current_mode,
            )
        )
        joint.set_drive_property(
            stiffness=float(stiffness),
            damping=float(damping),
            force_limit=float(joint.get_force_limit()),
            mode=current_mode,
        )
    return saved


def restore_arm_drive(
    saved: Sequence[tuple[Any, float, float, float, str]],
) -> None:
    for joint, stiffness, damping, force_limit, mode in saved:
        joint.set_drive_property(
            stiffness=stiffness,
            damping=damping,
            force_limit=force_limit,
            mode=mode,
        )


class HeldArmDriveLeaseManager:
    """Retain one arm's stronger joint drive for the lifetime of a hold.

    A fixed D6 keeps the object rigidly attached to the end effector, but it
    does not support the robot arm against gravity.  The first acquisition for
    an arm therefore snapshots its original joint drives and applies the held
    motion profile.  Repeated motions by the same owner are idempotent; the
    original snapshot is restored only when the hold is released or the
    runtime closes.
    """

    def __init__(self, robot: Any) -> None:
        self.robot = robot
        self._leases: dict[str, dict[str, Any]] = {}

    def acquire(
        self,
        subject: str,
        arm: str,
        *,
        stiffness: float,
        damping: float,
    ) -> dict[str, Any]:
        arm_name = str(arm)
        owner = str(subject)
        profile = (float(stiffness), float(damping))
        existing = self._leases.get(arm_name)
        if existing is not None:
            if (
                existing["subject"] != owner
                or existing["profile"] != profile
            ):
                raise RuntimeError(
                    f"held_arm_drive_lease_conflict:{arm_name}"
                )
            return self.describe(arm_name)
        saved = boost_arm_drive(
            self.robot,
            arm_name,
            stiffness=profile[0],
            damping=profile[1],
        )
        self._leases[arm_name] = {
            "subject": owner,
            "profile": profile,
            "saved": saved,
        }
        return self.describe(arm_name)

    def require(self, subject: str, arm: str) -> dict[str, Any]:
        arm_name = str(arm)
        lease = self._leases.get(arm_name)
        if lease is None or lease["subject"] != str(subject):
            raise RuntimeError(
                f"held_arm_drive_lease_missing:{arm_name}:{subject}"
            )
        return self.describe(arm_name)

    def describe(self, arm: str) -> dict[str, Any]:
        arm_name = str(arm)
        lease = self._leases.get(arm_name)
        if lease is None:
            return {"active": False, "arm": arm_name}
        return {
            "active": True,
            "arm": arm_name,
            "subject": str(lease["subject"]),
            "stiffness": float(lease["profile"][0]),
            "damping": float(lease["profile"][1]),
        }

    def release(self, subject: str, arm: str) -> bool:
        arm_name = str(arm)
        lease = self._leases.get(arm_name)
        if lease is None:
            return False
        if lease["subject"] != str(subject):
            raise RuntimeError(
                f"held_arm_drive_lease_owner_mismatch:{arm_name}:{subject}"
            )
        self._leases.pop(arm_name)
        restore_arm_drive(lease["saved"])
        return True

    def release_arm(self, arm: str) -> bool:
        arm_name = str(arm)
        lease = self._leases.pop(arm_name, None)
        if lease is None:
            return False
        restore_arm_drive(lease["saved"])
        return True

    def release_all(self) -> None:
        for arm_name in tuple(self._leases):
            self.release_arm(arm_name)


def project_arm_to_terminal_qpos(
    robot: Any,
    arm: str,
    terminal_arm_qpos: Sequence[float],
) -> None:
    entity = robot.left_entity if str(arm) == "left" else robot.right_entity
    arm_joints = (
        robot.left_arm_joints
        if str(arm) == "left"
        else robot.right_arm_joints
    )
    active_joints = list(entity.get_active_joints())
    qpos = np.asarray(entity.get_qpos(), dtype=float)
    target = np.asarray(terminal_arm_qpos, dtype=float).reshape(-1)
    if len(target) < len(arm_joints):
        raise ArticulationExecutionError("articulation_step_projection_invalid")
    for index, joint in enumerate(arm_joints):
        qpos[active_joints.index(joint)] = target[index]
    entity.set_qpos(qpos)


__all__ = [
    "HeldArmDriveLeaseManager",
    "SapienFixedAttachment",
    "SapienWorldPoseLock",
    "HARD_D6_TRANSLATION_DAMPING",
    "HARD_D6_TRANSLATION_STIFFNESS",
    "HARD_D6_ROTATION_DAMPING",
    "HARD_D6_ROTATION_STIFFNESS",
    "articulation_attachment_status",
    "create_evaluated_pick_attachment",
    "boost_arm_drive",
    "physical_ee_link",
    "physical_ee_world_matrix",
    "project_arm_to_terminal_qpos",
    "restore_arm_drive",
]
