"""Pure runtime helpers for Milestone 4 canonical asset probes."""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from if_benchmark.operation_anchors import (
    build_operation_anchor_contract,
)
from if_benchmark.operation_placement import (
    PICK_OPERATION_PLACEMENT_POLICY,
    resolve_m4_single_object_placement,
    resolve_single_arm_pick_placement,
)


ARTICULATED_TASK_HINTS: dict[str, dict[str, Any]] = {
    "015_laptop": {
        "xlim": [-0.05, 0.05],
        "ylim": [-0.1, 0.05],
        "qpos": [0.7, 0.0, 0.0, 0.7],
        "rotate_rand": True,
        "rotate_lim": [0.0, 0.0, math.pi / 3],
        "source": "envs/open_laptop.py",
    },
}

ARTICULATION_PROGRESS_DELTA = 0.15
ARTICULATION_ACTION_TARGET_DELTA = 0.22
POSITIVE_ARTICULATION_USES = frozenset(("open", "switch_off"))
NEGATIVE_ARTICULATION_USES = frozenset(("close", "switch_on"))
M4_SINGLE_ARM_PICK_ACTOR_X_ABS_RANGE_M = tuple(
    PICK_OPERATION_PLACEMENT_POLICY["single_arm"]["right"][
        "actor_origin_safety_xy_m"
    ]["x"]
)
M4_SINGLE_ARM_PICK_ACTOR_Y_RANGE_M = tuple(
    PICK_OPERATION_PLACEMENT_POLICY["single_arm"]["right"][
        "actor_origin_safety_xy_m"
    ]["y"]
)
M4_SINGLE_ARM_PICK_CONTACT_X_ABS_RANGE_M = tuple(
    PICK_OPERATION_PLACEMENT_POLICY["single_arm"]["right"][
        "target_contact_xy_m"
    ]["x"]
)
M4_SINGLE_ARM_PICK_CONTACT_Y_RANGE_M = tuple(
    PICK_OPERATION_PLACEMENT_POLICY["single_arm"]["right"][
        "target_contact_xy_m"
    ]["y"]
)


def single_arm_pick_placement(
    arm: str,
    *,
    model_data: dict[str, Any] | None = None,
    q0_wxyz: list[float] | tuple[float, ...] | None = None,
) -> dict[str, Any]:
    """Resolve actor bounds from operation and actor safety workspaces.

    The authored Contact narrows placement where that remains inside the
    embodiment's proven actor workspace.  Actor safety wins when the two are
    disjoint because a reachable Contact is not sufficient if the actor body,
    Pre-Grasp, or swept gripper collides with the robot or camera.
    """

    contract = None
    if isinstance(model_data, dict) and q0_wxyz is not None:
        contract = build_operation_anchor_contract(model_data, q0_wxyz)
    return resolve_single_arm_pick_placement(
        str(arm), operation_anchor_contract=contract
    )


def single_arm_pick_placement_bounds(
    arm: str,
    *,
    model_data: dict[str, Any] | None = None,
    q0_wxyz: list[float] | tuple[float, ...] | None = None,
) -> dict[str, list[float]]:
    """Return only the actor-origin bounds for compatibility callers."""

    placement = single_arm_pick_placement(
        arm, model_data=model_data, q0_wxyz=q0_wxyz
    )
    return {
        "xlim": list(placement["xlim"]),
        "ylim": list(placement["ylim"]),
    }


def canonical_probe_operation_anchor_contract(
    *,
    candidate: Mapping[str, Any],
    model_data: Mapping[str, Any],
    canonical_initial_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the exact R01-bound anchor projection used by M4 placement."""

    joint_state = canonical_initial_state.get("articulated_joint_state") or {}
    articulated = str(joint_state.get("mode") or "not_articulated") != (
        "not_articulated"
    )
    urdf_path: Path | None = None
    if articulated:
        metadata_path = Path(
            str(
                (((candidate.get("evidence") or {}).get("metadata") or {}).get("path"))
                or ""
            )
        )
        candidate_urdf = metadata_path.parent / "mobility.urdf"
        if not metadata_path.is_file() or not candidate_urdf.is_file():
            raise ValueError("articulated placement requires model_data and mobility.urdf")
        urdf_path = candidate_urdf
    return build_operation_anchor_contract(
        model_data,
        canonical_initial_state["q0_wxyz"],
        scale_xyz=canonical_initial_state["scale_xyz"],
        articulated_joint_state=(joint_state if articulated else None),
        urdf_path=urdf_path,
        canonical_state_hash=str(canonical_initial_state["canonical_state_hash"]),
    )


def _canonical_probe_contact_selection(
    *,
    use: str,
    candidate: Mapping[str, Any],
    model_data: Mapping[str, Any],
) -> tuple[list[int] | None, list[list[int]]]:
    groups = contact_groups_from_candidate(
        dict(candidate), model_data=dict(model_data)
    )
    raw_points = (
        model_data.get("contact_points")
        if isinstance(model_data.get("contact_points"), list)
        else model_data.get("contact_points_pose")
    )
    point_count = len(raw_points) if isinstance(raw_points, list) else 0
    fallback = list(groups[0]) if groups else list(range(point_count))
    use_name = str(use)
    if use_name == "click_subject":
        selected = 0 if point_count > 0 else (fallback[0] if fallback else 0)
        return [int(selected)], groups
    if use_name == "press_subject":
        selected = 2 if point_count > 2 else (fallback[0] if fallback else 0)
        return [int(selected)], groups
    if use_name in {"rotate_to_orientation_subject", "adjust_orientation_subject"}:
        return (fallback or None), groups
    if use_name in {"open", "close", "switch_on", "switch_off"}:
        drive = candidate.get("articulation_drive") or {}
        contact_ids = drive.get("open_contact_ids")
        return (
            [int(value) for value in contact_ids]
            if isinstance(contact_ids, list) and contact_ids
            else (fallback or None),
            groups,
        )
    if use_name == "bimanual_grab_subject":
        return (
            sorted({value for group in groups[:2] for value in group}) or None,
            groups,
        )
    # Single-arm Pick intentionally exposes the complete eligible Contact
    # domain to the shared acquisition planner.
    return None, groups


def canonical_probe_placement(
    *,
    use: str,
    candidate: Mapping[str, Any],
    model_data: Mapping[str, Any],
    canonical_initial_state: Mapping[str, Any],
    relative_yaw_deg: float = 0.0,
) -> dict[str, Any]:
    """Resolve the deterministic all-use M4 single-subject placement."""

    contract = canonical_probe_operation_anchor_contract(
        candidate=candidate,
        model_data=model_data,
        canonical_initial_state=canonical_initial_state,
    )
    contact_ids, contact_groups = _canonical_probe_contact_selection(
        use=use,
        candidate=candidate,
        model_data=model_data,
    )
    return resolve_m4_single_object_placement(
        str(use),
        origin_relative_bounds=canonical_initial_state["origin_relative_bounds"],
        support=canonical_initial_state["support"],
        operation_anchor_contract=contract,
        canonical_state_hash=str(canonical_initial_state["canonical_state_hash"]),
        relative_yaw_deg=float(relative_yaw_deg),
        allowed_contact_ids=contact_ids,
        contact_groups=contact_groups,
    )


def articulated_task_hint(asset_name: str, model_id: str | int | None = None) -> dict[str, Any]:
    asset_name = str(asset_name)
    profile_hint = _profile_fixture_hints().get((asset_name, str(model_id))) if model_id is not None else None
    if profile_hint is None:
        profile_hint = next(
            (
                value
                for (profile_asset, _), value in sorted(_profile_fixture_hints().items())
                if profile_asset == asset_name
            ),
            None,
        )
    return dict(profile_hint or ARTICULATED_TASK_HINTS.get(asset_name, {}))


@lru_cache(maxsize=1)
def _profile_fixture_hints() -> dict[tuple[str, str], dict[str, Any]]:
    path = Path("if_benchmark/assets/articulated_receiver_profiles.yml")
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as stream:
        catalog = yaml.load(stream, Loader=yaml.CSafeLoader)
    hints: dict[tuple[str, str], dict[str, Any]] = {}
    for profile_id, profile in sorted(((catalog or {}).get("profiles") or {}).items()):
        receiver = profile.get("receiver") or {}
        fixture = profile.get("fixture") or {}
        xy = fixture.get("table_xy")
        q = fixture.get("root_q0_wxyz")
        if not isinstance(xy, list) or len(xy) != 2 or not isinstance(q, list) or len(q) != 4:
            continue
        hints[(str(receiver.get("asset")), str(receiver.get("model_id")))] = {
            "xlim": [float(xy[0]), float(xy[0])],
            "ylim": [float(xy[1]), float(xy[1])],
            "z": float(fixture.get("root_z", 0.741)),
            "qpos": [float(value) for value in q],
            "rotate_rand": False,
            "source": f"receiver_state_profile:{profile_id}.fixture",
        }
    return hints


def contact_groups_from_candidate(
    candidate: dict[str, Any],
    *,
    model_data: dict[str, Any] | None = None,
) -> list[list[int]]:
    metadata = candidate.get("metadata") or {}
    groups = metadata.get("contact_points_group")
    if groups is None and model_data is not None:
        groups = model_data.get("contact_points_group")
    if not isinstance(groups, list):
        return []
    out: list[list[int]] = []
    for group in groups:
        if not isinstance(group, list):
            continue
        ids = [int(item) for item in group]
        if ids:
            out.append(ids)
    return out


def functional_point_count(
    candidate: dict[str, Any],
    *,
    model_data: dict[str, Any] | None = None,
) -> int:
    metadata = candidate.get("metadata") or {}
    for key in ("functional_points_count", "functional_point_count"):
        if metadata.get(key) is None:
            continue
        try:
            return int(metadata.get(key) or 0)
        except (TypeError, ValueError):
            return 0
    functional_matrix = metadata.get("functional_matrix")
    if functional_matrix is None and model_data is not None:
        functional_matrix = model_data.get("functional_matrix")
    return len(functional_matrix) if isinstance(functional_matrix, list) else 0


def as_contact_id_list(contact_ids: Any) -> list[int]:
    if isinstance(contact_ids, list):
        return [int(item) for item in contact_ids]
    if contact_ids is None:
        return []
    return [int(contact_ids)]


def articulation_contact_sequence(
    drive: dict[str, Any],
    fallback_group: Any,
    *,
    use: str,
) -> list[int]:
    del use
    contact_ids = drive.get("open_contact_ids")
    if isinstance(contact_ids, list) and contact_ids:
        sequence = [int(item) for item in contact_ids]
    else:
        sequence = as_contact_id_list(fallback_group)
    return sequence or [0]


def articulation_action_target_delta(drive: dict[str, Any]) -> float:
    raw = drive.get("action_target_delta", ARTICULATION_ACTION_TARGET_DELTA)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = ARTICULATION_ACTION_TARGET_DELTA
    return _clamp(value, ARTICULATION_PROGRESS_DELTA, 1.0)


def articulation_setup_qpos(
    *,
    use: str,
    drive: dict[str, Any],
    qlimits: np.ndarray,
    qpos: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    joint_index = int(drive.get("joint_index", 0))
    if joint_index >= len(qpos) or joint_index >= len(qlimits):
        raise ValueError("joint_index_out_of_range")
    lower, upper = qlimits[joint_index]
    if not np.isfinite(lower) or not np.isfinite(upper) or math.isclose(float(upper - lower), 0.0):
        raise ValueError("invalid_joint_limits")
    state_setup = drive.get("state_setup")
    ratio_source = "articulation_drive.state_setup"
    if state_setup is None:
        state_setup = 0.0 if use in POSITIVE_ARTICULATION_USES else drive.get("open_ratio", 0.5)
        ratio_source = "legacy_open_close_default"
    setup_ratio = _clamp(float(state_setup), 0.0, 1.0)
    out = np.array(qpos, dtype=float)
    out[joint_index] = float(lower) + float(upper - lower) * setup_ratio
    return out, {
        "joint_index": joint_index,
        "lower": float(lower),
        "upper": float(upper),
        "setup_ratio": float(setup_ratio),
        "source": ratio_source,
        "qpos": float(out[joint_index]),
    }


def articulation_normalized_delta(use: str, initial: float, current: float, span: float) -> float:
    if use in POSITIVE_ARTICULATION_USES:
        direction = 1.0
    elif use in NEGATIVE_ARTICULATION_USES:
        direction = -1.0
    else:
        raise ValueError(f"unsupported articulation use: {use}")
    return float(direction * (current - initial) / max(float(span), 1e-6))


def articulation_terminal_state_reached(
    use: str,
    initial: float,
    current: float,
    span: float,
) -> bool:
    """Return whether the requested object-state change crossed the M4 threshold."""

    return articulation_normalized_delta(use, initial, current, span) >= ARTICULATION_PROGRESS_DELTA


def switch_contact_matrix_for_use(matrix: Any, use: str) -> list[list[float]]:
    """Resolve the authored rocker contact point for the requested state edge.

    The reviewed switch contact is on the negative local-y end of a rocker
    whose revolute axis is local x. A downward press there produces positive
    joint motion (switch_off). Mirroring only the contact translation to the
    positive local-y end produces the opposite torque for switch_on without
    changing the fixture pose or the approach frame.
    """

    if use not in {"switch_on", "switch_off"}:
        raise ValueError(f"unsupported switch use: {use}")
    resolved = np.array(matrix, dtype=float, copy=True)
    if resolved.shape != (4, 4) or not np.isfinite(resolved).all():
        raise ValueError("invalid_switch_contact_matrix")
    if use == "switch_on":
        resolved[1, 3] = abs(float(resolved[1, 3]))
    else:
        resolved[1, 3] = -abs(float(resolved[1, 3]))
    return resolved.tolist()


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
