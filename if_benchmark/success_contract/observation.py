"""Read-only normalization for physical observations used by success checks."""

from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from if_benchmark.execution_policy import (
    HOLD_EE_FRAME_ARM,
    HOLD_PLANNING_EE_FRAME_ARM,
)
from if_benchmark.schema.serialization import artifact_hash, canonical_data

from .contract import SuccessContract


class SuccessObservationError(ValueError):
    """A physical observation cannot be represented without fabrication."""


DEFAULT_ALOHA_GRIPPER_PROFILE = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "success_contract"
    / "embodiment_profiles"
    / "aloha_agilex_gripper_v1.yml"
)

CONTACT_REGION_CLASSIFICATION_VERSION = "finger_pad_position_and_oriented_normal_v2"


def load_embodiment_profile(path: Path | str) -> dict[str, Any]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, Mapping) or value.get("review_status") != "reviewed":
        raise SuccessObservationError("reviewed embodiment profile missing")
    profile = canonical_data(dict(value))
    profile["profile_hash"] = artifact_hash(profile)
    return profile


def aggregate_active_contacts(
    manifolds: Iterable[Mapping[str, Any]],
    *,
    contract: SuccessContract,
) -> list[dict[str, Any]]:
    """Aggregate PhysX manifold points and retain only non-zero impulse rows.

    Input rows must identify both entity/link endpoints and expose point-level
    position, normal, impulse vector, and separation.  No boolean collision or
    historical contact event is accepted as a substitute.
    """

    epsilon = float(contract.section("active_contact")["impulse_epsilon"])
    grouped: dict[tuple[str, ...], dict[str, Any]] = {}
    vectors: dict[tuple[str, ...], list[float]] = defaultdict(
        lambda: [0.0, 0.0, 0.0]
    )
    for manifold in manifolds:
        endpoints = _endpoints(manifold)
        points = manifold.get("points")
        if not isinstance(points, Sequence):
            raise SuccessObservationError("contact manifold points missing")
        key = (
            endpoints[0],
            endpoints[1],
            str(manifold.get("link_a") or ""),
            str(manifold.get("link_b") or ""),
            str(manifold.get("arm") or ""),
            str(manifold.get("finger") or ""),
            str(manifold.get("region") or ""),
        )
        row = grouped.setdefault(
            key,
            {
                "entity_a": endpoints[0],
                "entity_b": endpoints[1],
                "link_a": key[2] or None,
                "link_b": key[3] or None,
                "arm": key[4] or None,
                "finger": key[5] or None,
                "region": key[6] or None,
                "points": [],
            },
        )
        for point in points:
            if not isinstance(point, Mapping):
                raise SuccessObservationError("contact point must be a mapping")
            required = ("position", "normal", "impulse", "separation")
            missing = [name for name in required if name not in point]
            if missing:
                raise SuccessObservationError(
                    f"contact point fields missing: {missing}"
                )
            impulse = _vector(point["impulse"], "impulse")
            for index, value in enumerate(impulse):
                vectors[key][index] += value
            row["points"].append(
                {
                    "position": _vector(point["position"], "position"),
                    "normal": _vector(point["normal"], "normal"),
                    "impulse": impulse,
                    "separation": float(point["separation"]),
                }
            )
    active: list[dict[str, Any]] = []
    for key in sorted(grouped):
        summed = vectors[key]
        norm = math.sqrt(sum(value * value for value in summed))
        if norm <= epsilon:
            continue
        row = grouped[key]
        row["summed_impulse"] = summed
        row["summed_impulse_norm"] = norm
        row["point_count"] = len(row["points"])
        active.append(canonical_data(row))
    return active


def build_actual_gripper_observation(
    *,
    arm: str,
    actual_qpos: Mapping[str, float] | Sequence[float],
    actual_width_m: float,
    link_poses: Mapping[str, Mapping[str, Any]],
    embodiment_profile: Mapping[str, Any],
    engaged: bool,
    closure_evidence: Mapping[str, Any] | None = None,
    hold_consistency: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build actual gripper state from joints/link poses and reviewed profile."""

    if arm not in {"left", "right"}:
        raise SuccessObservationError("gripper arm must be left or right")
    profile_hash = embodiment_profile.get("profile_hash")
    profile_version = embodiment_profile.get("profile_version")
    regions = embodiment_profile.get("contact_regions")
    if not profile_hash or not profile_version or not isinstance(regions, Mapping):
        raise SuccessObservationError("reviewed embodiment contact profile missing")
    if not isinstance(link_poses, Mapping) or not link_poses:
        raise SuccessObservationError("actual finger/pad link poses missing")
    # Drive targets and commands are intentionally not accepted by this API.
    width = float(actual_width_m)
    if not math.isfinite(width) or width < 0.0:
        raise SuccessObservationError("actual gripper width is invalid")
    value = {
        "arm": arm,
        "actual_qpos": canonical_data(actual_qpos),
        "actual_width_m": width,
        "link_poses": canonical_data(dict(link_poses)),
        "contact_region_profile_version": str(profile_version),
        "contact_region_profile_hash": str(profile_hash),
        "engaged": bool(engaged),
    }
    if closure_evidence is not None:
        value["closure_evidence"] = canonical_data(dict(closure_evidence))
    if hold_consistency is not None:
        value["hold_consistency"] = canonical_data(dict(hold_consistency))
    return canonical_data(value)


def observe_actual_grippers(
    robot: Any, *, embodiment_profile: Mapping[str, Any]
) -> dict[str, Any]:
    """Read actual SAPIEN articulation qpos and live finger link poses."""

    profile_hash = embodiment_profile.get("profile_hash")
    if not profile_hash:
        raise SuccessObservationError("embodiment profile hash missing")
    result: dict[str, Any] = {}
    for arm in ("left", "right"):
        arm_profile = (embodiment_profile.get("arms") or {}).get(arm)
        if not isinstance(arm_profile, Mapping):
            raise SuccessObservationError(f"{arm} gripper profile missing")
        entity = getattr(robot, f"{arm}_entity", None)
        if entity is None:
            raise SuccessObservationError(f"{arm} articulation missing")
        active_joints = list(entity.get_active_joints())
        qpos = list(entity.get_qpos())
        actual: dict[str, float] = {}
        ratios: list[float] = []
        poses: dict[str, Any] = {}
        finger_positions: list[list[float]] = []
        for finger in arm_profile.get("fingers") or []:
            joint_name = str(finger["joint"])
            joint = next(
                (item for item in active_joints if item.get_name() == joint_name),
                None,
            )
            if joint is None:
                raise SuccessObservationError(f"actual joint missing: {joint_name}")
            index = active_joints.index(joint)
            value = float(qpos[index])
            limits = joint.get_limits()
            lower, upper = float(limits[0][0]), float(limits[0][1])
            span = upper - lower
            if span <= 0.0:
                raise SuccessObservationError(f"invalid joint limits: {joint_name}")
            actual[joint_name] = value
            ratios.append((value - lower) / span)
            link_name = str(finger["link"])
            link = next(
                (item for item in entity.get_links() if item.get_name() == link_name),
                None,
            )
            if link is None:
                raise SuccessObservationError(f"finger link missing: {link_name}")
            pose = link.get_pose()
            position = [float(item) for item in pose.p]
            poses[link_name] = {
                "position": position,
                "quat_wxyz": [float(item) for item in pose.q],
            }
            finger_positions.append(position)
        width = math.dist(finger_positions[0], finger_positions[1])
        maximum = float(
            embodiment_profile["classification"][
                "engaged_max_normalized_opening"
            ]
        )
        profile_for_builder = {
            "profile_version": embodiment_profile["profile_version"],
            "profile_hash": profile_hash,
            "contact_regions": {
                str(row["link"]): canonical_data(dict(row))
                for row in arm_profile.get("fingers") or []
            },
        }
        result[arm] = build_actual_gripper_observation(
            arm=arm,
            actual_qpos=actual,
            actual_width_m=width,
            link_poses=poses,
            embodiment_profile=profile_for_builder,
            engaged=max(ratios) <= maximum,
        )
    return canonical_data(result)


def observe_sapien_active_contacts(
    scene: Any,
    *,
    endpoint_by_body_name: Mapping[str, str],
    embodiment_profile: Mapping[str, Any],
    grippers: Mapping[str, Any],
    contract: SuccessContract,
) -> list[dict[str, Any]]:
    """Read link-aware PhysX manifolds from a SAPIEN scene."""

    endpoints = {str(key): str(value) for key, value in endpoint_by_body_name.items()}
    links: dict[str, dict[str, Any]] = {}
    for arm, arm_profile in (embodiment_profile.get("arms") or {}).items():
        for row in arm_profile.get("fingers") or []:
            links[str(row["link"])] = {
                **canonical_data(dict(row)),
                "arm": str(arm),
                "region": "finger",
            }
        for name in arm_profile.get("palm_wrist_links") or []:
            links[str(name)] = {"arm": str(arm), "region": "palm_wrist"}
    centers = {
        arm: [
            sum(float(pose["position"][index]) for pose in state["link_poses"].values())
            / len(state["link_poses"])
            for index in range(3)
        ]
        for arm, state in grippers.items()
    }
    alignment_min = float(
        embodiment_profile["classification"]["inner_pad_normal_alignment_min"]
    )
    if not math.isfinite(alignment_min) or not 0.0 < alignment_min <= 1.0:
        raise SuccessObservationError("inner-pad normal alignment is invalid")
    pad_normals = {}
    for name, link in links.items():
        if link["region"] != "finger":
            continue
        local_normal = _vector(link["inner_normal_local"], "inner_normal_local")
        length = math.sqrt(sum(value * value for value in local_normal))
        if length <= 0.0:
            raise SuccessObservationError("inner-pad normal is invalid")
        pose = grippers[link["arm"]]["link_poses"][name]
        pad_normals[name] = (
            _quat_rotation(pose["quat_wxyz"]) @ local_normal / length
        )
    manifolds: list[dict[str, Any]] = []
    for contact in scene.get_contacts():
        names = [_body_name(body) for body in getattr(contact, "bodies", [])]
        if len(names) != 2 or any(name not in endpoints for name in names):
            continue
        if endpoints[names[0]] == endpoints[names[1]]:
            continue
        gripper_index = next(
            (index for index, name in enumerate(names) if name in links), None
        )
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for point in contact.points:
            position = _vector(point.position, "position")
            normal = _vector(point.normal, "normal")
            region = "entity_contact"
            if gripper_index is not None:
                name = names[gripper_index]
                link = links[name]
                region = str(link["region"])
                if region == "finger":
                    arm = str(link["arm"])
                    origin = grippers[arm]["link_poses"][name]["position"]
                    inward = [centers[arm][i] - float(origin[i]) for i in range(3)]
                    local = [position[i] - float(origin[i]) for i in range(3)]
                    # PhysX normals point from body B to A. Orient toward the
                    # object from the finger before comparing with the pad's
                    # reviewed outward normal. Position alone admits two
                    # fingertips pressing the same top face as a false pinch.
                    sign = -1.0 if gripper_index == 0 else 1.0
                    length = math.sqrt(sum(value * value for value in normal))
                    alignment = (
                        sign * sum(a * b for a, b in zip(pad_normals[name], normal)) / length
                        if length > 0.0 else -1.0
                    )
                    region = (
                        "inner_pad"
                        if sum(a * b for a, b in zip(inward, local)) >= 0.0
                        and alignment >= alignment_min
                        else "outer_finger"
                    )
            grouped[region].append(
                {
                    "position": position,
                    "normal": normal,
                    "impulse": _vector(point.impulse, "impulse"),
                    "separation": float(point.separation),
                }
            )
        for region, points in grouped.items():
            row: dict[str, Any] = {
                "entity_a": endpoints[names[0]],
                "entity_b": endpoints[names[1]],
                "link_a": names[0],
                "link_b": names[1],
                "region": region,
                "points": points,
            }
            if gripper_index is not None:
                profile = links[names[gripper_index]]
                row.update(
                    {"arm": profile["arm"], "finger": profile.get("finger")}
                )
            manifolds.append(row)
    return aggregate_active_contacts(manifolds, contract=contract)


def require_hold_ee_frame(ee_frame: str) -> str:
    """Reject any hold observation that is not the canonical arm EE frame."""

    value = str(ee_frame or "")
    if value != HOLD_EE_FRAME_ARM:
        raise SuccessObservationError(
            "hold_ee_frame_invalid:"
            f" expected {HOLD_EE_FRAME_ARM}, got {value or '<missing>'}"
        )
    return value


def require_hold_planning_ee_frame(ee_frame: str) -> str:
    """Reject a held-motion transform that is not in the planner EE frame."""

    value = str(ee_frame or "")
    if value != HOLD_PLANNING_EE_FRAME_ARM:
        raise SuccessObservationError(
            "hold_planning_ee_frame_invalid:"
            f" expected {HOLD_PLANNING_EE_FRAME_ARM},"
            f" got {value or '<missing>'}"
        )
    return value


def ee_to_actor_transform(
    ee_matrix: Sequence[Sequence[float]],
    actor_matrix: Sequence[Sequence[float]],
) -> list[list[float]]:
    """Return ``inv(EE) @ actor`` for one hold sample."""

    import numpy as np

    ee = _matrix4(ee_matrix, "ee_matrix")
    actor = _matrix4(actor_matrix, "actor_matrix")
    return (np.linalg.inv(ee) @ actor).tolist()


def hold_consistency_from_relative_history(
    relative_transforms: Sequence[Sequence[Sequence[float]]],
    *,
    subject: str,
    ee_frame: str,
    reference: Mapping[str, Any] | None = None,
    symmetry_axis: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Summarize hold drift against an immutable reference in one EE frame.

    Callers must pass the same ``ee_frame`` used to freeze ``reference`` and to
    build every relative sample.  Mixing arm EE with TCP fails closed.
    """

    import numpy as np

    require_hold_ee_frame(ee_frame)
    if not relative_transforms:
        raise SuccessObservationError("hold relative history missing")
    if reference is not None:
        if str(reference.get("ee_frame") or "") != ee_frame:
            raise SuccessObservationError(
                "hold_ee_frame_mismatch:"
                f" reference={reference.get('ee_frame') or '<missing>'}"
                f" observation={ee_frame}"
            )
        frozen = _matrix4(
            reference.get("ee_to_actor_transform"), "ee_to_actor_transform"
        )
    else:
        frozen = _matrix4(relative_transforms[0], "relative_transform")
    maximum_translation = 0.0
    maximum_rotation = 0.0
    for row in relative_transforms:
        observed = _matrix4(row, "relative_transform")
        delta = np.linalg.inv(frozen) @ observed
        maximum_translation = max(
            maximum_translation, float(np.linalg.norm(delta[:3, 3]))
        )
        maximum_rotation = max(
            maximum_rotation,
            _rotation_error_deg(delta[:3, :3], symmetry_axis=symmetry_axis),
        )
    return canonical_data(
        {
            "subject": str(subject),
            "translation_drift_m": maximum_translation,
            "rotation_drift_deg": maximum_rotation,
            "ee_frame": ee_frame,
            "symmetry_aware": symmetry_axis is not None,
        }
    )


def build_hold_reference(
    *,
    subject: str,
    arm: str,
    actor_matrix: Sequence[Sequence[float]],
    ee_matrix: Sequence[Sequence[float]],
    planning_ee_matrix: Sequence[Sequence[float]],
    actual_width_m: float,
    ee_frame: str = HOLD_EE_FRAME_ARM,
) -> dict[str, Any]:
    """Freeze separate physical-hold and planner-motion EE transforms.

    ``ee_matrix`` is the physical child-link frame constrained by the D6 and
    is the only frame used for drift/attachment checks.  ``planning_ee_matrix``
    is the move-group pose accepted by ``move_to_pose`` and is the only frame
    used to turn a desired actor pose into a held-motion target.
    """

    import numpy as np

    require_hold_ee_frame(ee_frame)
    actor = _matrix4(actor_matrix, "actor_matrix")
    ee = _matrix4(ee_matrix, "ee_matrix")
    planning_ee = _matrix4(planning_ee_matrix, "planning_ee_matrix")
    width = float(actual_width_m)
    if not math.isfinite(width) or width < 0.0:
        raise SuccessObservationError("actual gripper width is invalid")
    value = {
        "subject": str(subject),
        "arm": str(arm),
        "ee_frame": require_hold_ee_frame(ee_frame),
        "actual_width_m": width,
        "actor_matrix": actor.tolist(),
        "ee_matrix": ee.tolist(),
        "ee_to_actor_transform": (np.linalg.inv(ee) @ actor).tolist(),
        "actor_to_ee_transform": (np.linalg.inv(actor) @ ee).tolist(),
        "planning_ee_frame": HOLD_PLANNING_EE_FRAME_ARM,
        "planning_ee_matrix": planning_ee.tolist(),
        "planning_ee_to_actor_transform": (
            np.linalg.inv(planning_ee) @ actor
        ).tolist(),
        "actor_to_planning_ee_transform": (
            np.linalg.inv(actor) @ planning_ee
        ).tolist(),
    }
    return canonical_data(value)


def validate_hold_reference(
    reference: Mapping[str, Any],
    *,
    subject: str | None = None,
    arm: str | None = None,
) -> None:
    """Fail closed unless both closure-time EE relations are self-consistent."""

    import numpy as np

    if subject is not None and str(reference.get("subject") or "") != str(
        subject
    ):
        raise SuccessObservationError("hold reference subject mismatch")
    if arm is not None and str(reference.get("arm") or "") != str(arm):
        raise SuccessObservationError("hold reference arm mismatch")
    require_hold_ee_frame(str(reference.get("ee_frame") or ""))
    require_hold_planning_ee_frame(
        str(reference.get("planning_ee_frame") or "")
    )
    actor = _matrix4(reference.get("actor_matrix"), "actor_matrix")
    physical_ee = _matrix4(reference.get("ee_matrix"), "ee_matrix")
    planning_ee = _matrix4(
        reference.get("planning_ee_matrix"), "planning_ee_matrix"
    )
    expected = {
        "ee_to_actor_transform": np.linalg.inv(physical_ee) @ actor,
        "actor_to_ee_transform": np.linalg.inv(actor) @ physical_ee,
        "planning_ee_to_actor_transform": (
            np.linalg.inv(planning_ee) @ actor
        ),
        "actor_to_planning_ee_transform": (
            np.linalg.inv(actor) @ planning_ee
        ),
    }
    for field, expected_matrix in expected.items():
        observed = _matrix4(reference.get(field), field)
        if not np.allclose(
            observed, expected_matrix, rtol=0.0, atol=1e-10
        ):
            raise SuccessObservationError(
                f"hold_reference_transform_mismatch:{field}"
            )


def held_actor_target_to_planning_ee_pose(
    reference: Mapping[str, Any],
    actor_target_pose: Sequence[float],
) -> list[float]:
    """Map one actor target to the planner EE using the closure-time relation."""

    import numpy as np
    import transforms3d as t3d

    validate_hold_reference(reference)
    relative = _matrix4(
        reference.get("actor_to_planning_ee_transform"),
        "actor_to_planning_ee_transform",
    )
    values = np.asarray(actor_target_pose, dtype=float)
    if values.shape != (7,) or not np.all(np.isfinite(values)):
        raise SuccessObservationError(
            "actor target pose must contain finite xyz + quat_wxyz"
        )
    if float(np.linalg.norm(values[3:])) <= 1e-12:
        raise SuccessObservationError("actor target quaternion is invalid")
    actor = np.eye(4, dtype=float)
    actor[:3, :3] = t3d.quaternions.quat2mat(values[3:])
    actor[:3, 3] = values[:3]
    planning_ee = actor @ relative
    quaternion = t3d.quaternions.mat2quat(planning_ee[:3, :3])
    return [
        *planning_ee[:3, 3].tolist(),
        *np.asarray(quaternion, dtype=float).tolist(),
    ]


def summarize_hold_observation(
    samples: Sequence[Mapping[str, Any]],
    *,
    reference: Mapping[str, Any],
    subject: str,
    arm: str,
    embodiment_profile: Mapping[str, Any],
    symmetry_axis: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Summarize an immutable-reference hold window without issuing actions.

    Every sample contains actual ``gripper``, ``active_contacts``, and current
    actor/EE transforms.  Commanded width and planner state are intentionally
    outside this API.
    """

    import numpy as np

    if not samples:
        raise SuccessObservationError("hold observation samples missing")
    validate_hold_reference(reference, subject=subject, arm=arm)
    expected = _matrix4(reference.get("ee_to_actor_transform"), "ee_to_actor_transform")
    minimum = int(
        embodiment_profile["classification"]["continuous_contact_min_steps"]
    )
    consecutive = 0
    maximum_consecutive = 0
    maximum_translation = 0.0
    maximum_rotation = 0.0
    last_gripper: dict[str, Any] | None = None
    last_contacts: list[dict[str, Any]] = []
    width_values: list[float] = []
    for sample in samples:
        gripper = sample.get("gripper")
        contacts = sample.get("active_contacts")
        if not isinstance(gripper, Mapping) or not isinstance(contacts, Sequence):
            raise SuccessObservationError("actual hold sample fields missing")
        sample_frame = sample.get("ee_frame")
        if sample_frame is not None:
            if str(sample_frame) != str(reference.get("ee_frame") or ""):
                raise SuccessObservationError(
                    "hold_ee_frame_mismatch:"
                    f" reference={reference.get('ee_frame')}"
                    f" sample={sample_frame}"
                )
            require_hold_ee_frame(str(sample_frame))
        actor = _matrix4(sample.get("actor_matrix"), "actor_matrix")
        ee = _matrix4(sample.get("ee_matrix"), "ee_matrix")
        observed = np.linalg.inv(ee) @ actor
        delta = np.linalg.inv(expected) @ observed
        maximum_translation = max(
            maximum_translation, float(np.linalg.norm(delta[:3, 3]))
        )
        maximum_rotation = max(
            maximum_rotation,
            _rotation_error_deg(delta[:3, :3], symmetry_axis=symmetry_axis),
        )
        fingers = {
            str(row.get("finger") or "")
            for row in contacts
            if isinstance(row, Mapping)
            and {str(row.get("entity_a") or ""), str(row.get("entity_b") or "")}
            == {str(subject), f"gripper:{arm}"}
            and str(row.get("region") or "") == "inner_pad"
        }
        consecutive = consecutive + 1 if len(fingers) >= 2 else 0
        maximum_consecutive = max(maximum_consecutive, consecutive)
        width_values.append(float(gripper["actual_width_m"]))
        last_gripper = canonical_data(dict(gripper))
        last_contacts = [canonical_data(dict(row)) for row in contacts]
    assert last_gripper is not None
    closure_valid = maximum_consecutive >= minimum
    last_gripper["closure_evidence"] = {
        "subject": str(subject),
        "valid": closure_valid,
        "path": "bilateral_inner_pad",
        "actual_width_m": float(last_gripper["actual_width_m"]),
        "continuous_steps": maximum_consecutive,
        "required_continuous_steps": minimum,
        "failure_code": None if closure_valid else "closure_not_verified",
    }
    last_gripper["hold_consistency"] = {
        "subject": str(subject),
        "translation_drift_m": maximum_translation,
        "rotation_drift_deg": maximum_rotation,
        "ee_frame": str(reference.get("ee_frame") or HOLD_EE_FRAME_ARM),
        "symmetry_aware": symmetry_axis is not None,
    }
    return canonical_data(
        {
            "gripper": last_gripper,
            "active_contacts": last_contacts,
            "metrics": {
                "sample_count": len(samples),
                "maximum_consecutive_bilateral_inner_steps": maximum_consecutive,
                "actual_width_initial_m": width_values[0],
                "actual_width_terminal_m": width_values[-1],
                "actual_width_delta_m": width_values[-1] - width_values[0],
                "translation_drift_m": maximum_translation,
                "rotation_drift_deg": maximum_rotation,
                "engaged": bool(last_gripper.get("engaged")),
            },
        }
    )


def axisymmetric_pose_stability(
    poses: Sequence[Mapping[str, Any]],
    *,
    symmetry_axis: Sequence[float],
    translation_limit_m: float,
    rotation_limit_deg: float,
) -> dict[str, Any]:
    """Evaluate a pose window while ignoring only spin about a reviewed axis."""

    import numpy as np

    if not poses:
        raise SuccessObservationError("stability pose window missing")
    axis = np.asarray(symmetry_axis, dtype=float)
    axis /= float(np.linalg.norm(axis))
    final_p = np.asarray(poses[-1]["p"], dtype=float)
    final_q = np.asarray(poses[-1]["q"], dtype=float)
    final_axis = _quat_rotation(final_q) @ axis
    translations = []
    rotations = []
    for pose in poses:
        translations.append(
            float(np.linalg.norm(np.asarray(pose["p"], dtype=float) - final_p))
        )
        current_axis = _quat_rotation(np.asarray(pose["q"], dtype=float)) @ axis
        cosine = max(-1.0, min(1.0, float(np.dot(current_axis, final_axis))))
        rotations.append(round(math.degrees(math.acos(cosine)), 12))
    translation = max(translations)
    rotation = max(rotations)
    return canonical_data(
        {
            "stable": translation <= float(translation_limit_m)
            and rotation <= float(rotation_limit_deg),
            "max_translation": translation,
            "max_rotation_deg": rotation,
            "rotation_mode": "reviewed_axisymmetric",
        }
    )


def _quat_rotation(quat: Any):
    import numpy as np

    w, x, y, z = (float(value) for value in quat)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 0.0:
        raise SuccessObservationError("quaternion is invalid")
    w, x, y, z = (value / norm for value in (w, x, y, z))
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _rotation_error_deg(
    rotation: Any, *, symmetry_axis: Sequence[float] | None
) -> float:
    import numpy as np

    value = np.asarray(rotation, dtype=float)
    if symmetry_axis is None:
        cosine = max(-1.0, min(1.0, (float(np.trace(value)) - 1.0) / 2.0))
        return round(math.degrees(math.acos(cosine)), 12)
    axis = np.asarray(symmetry_axis, dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm <= 0.0:
        raise SuccessObservationError("symmetry axis is invalid")
    axis /= norm
    observed = value @ axis
    cosine = max(-1.0, min(1.0, float(np.dot(axis, observed))))
    return round(math.degrees(math.acos(cosine)), 12)


def _matrix4(value: Any, field: str):
    import numpy as np

    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise SuccessObservationError(f"{field} must be a finite 4x4 matrix")
    return matrix


def _body_name(body: Any) -> str:
    entity = getattr(body, "entity", body)
    getter = getattr(entity, "get_name", None)
    return str(getter() if callable(getter) else getattr(entity, "name", ""))


def _endpoints(manifold: Mapping[str, Any]) -> tuple[str, str]:
    left = str(manifold.get("entity_a") or "")
    right = str(manifold.get("entity_b") or "")
    if not left or not right:
        raise SuccessObservationError("contact manifold endpoints missing")
    return left, right


def _vector(value: Any, field: str) -> list[float]:
    if isinstance(value, (str, bytes)):
        raise SuccessObservationError(f"{field} must be a 3-vector")
    if hasattr(value, "tolist"):
        value = value.tolist()
    try:
        length = len(value)
    except (TypeError, AttributeError):
        length = 0
    if length < 3:
        raise SuccessObservationError(f"{field} must be a 3-vector")
    try:
        result = [float(value[index]) for index in range(3)]
    except (TypeError, ValueError, IndexError) as exc:
        raise SuccessObservationError(f"{field} must be a 3-vector") from exc
    if not all(math.isfinite(item) for item in result):
        raise SuccessObservationError(f"{field} contains a non-finite value")
    return result


__all__ = [
    "SuccessObservationError",
    "DEFAULT_ALOHA_GRIPPER_PROFILE",
    "HOLD_EE_FRAME_ARM",
    "HOLD_PLANNING_EE_FRAME_ARM",
    "aggregate_active_contacts",
    "axisymmetric_pose_stability",
    "build_actual_gripper_observation",
    "build_hold_reference",
    "ee_to_actor_transform",
    "hold_consistency_from_relative_history",
    "held_actor_target_to_planning_ee_pose",
    "load_embodiment_profile",
    "observe_actual_grippers",
    "observe_sapien_active_contacts",
    "require_hold_ee_frame",
    "require_hold_planning_ee_frame",
    "summarize_hold_observation",
    "validate_hold_reference",
]
