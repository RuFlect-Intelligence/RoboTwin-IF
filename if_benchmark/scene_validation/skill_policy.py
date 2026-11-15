"""Pure policy helpers for Milestone 15 scene-conditioned skill execution.

Milestone 4 canonical probes establish that an asset/use pair is executable in
the canonical fixture.  They are deliberately not action policies for a later
SceneSpec.  This module therefore derives bounded execution targets and
candidate orders only from the symbolic M13 step plus current-scene metadata.
Simulator-specific planning and feedback live in ``envs.if_scene_validation``.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

STABILITY_COMPONENT_POLICY_VERSION = (
    "scene_stability_components_v2_root_and_active_child_links"
)
MAXIMUM_NORMALIZED_JOINT_STEP_EQUIVALENCE = 5e-5
NORMALIZED_JOINT_TRAVEL_EQUIVALENCE = 0.02


def _equivalence_bucket(value: float, width: float) -> int:
    return int(math.floor(float(value) / float(width) + 0.5))


def ordered_rigid_grasp_contact_ids(
    contact_approaches: Iterable[tuple[int, Sequence[float]]],
    *,
    arm: str,
) -> list[int]:
    """Rank rigid contacts by the arm's natural table-side approach.

    ``approach`` points from the pre-grasp pose toward the contact.  On the
    ALOHA table the left arm should normally enter from negative world X
    (therefore approach along +X), and the right arm should mirror that.  A
    small +Y component keeps the wrist in front of the object.  Horizontal
    contacts outrank vertical ones whenever the asset actually provides a
    usable side grasp; top-down remains the deterministic fallback.
    """

    arm_name = str(arm)
    if arm_name not in {"left", "right"}:
        raise ValueError(f"unsupported arm: {arm_name}")
    lateral_sign = 1.0 if arm_name == "left" else -1.0
    desired_x = 0.94 * lateral_sign
    desired_y = 0.34
    scored: list[tuple[tuple[float, ...], int]] = []
    for raw_contact_id, raw_approach in contact_approaches:
        values = [float(value) for value in raw_approach]
        if len(values) != 3:
            raise ValueError("contact approach must contain xyz")
        norm = math.sqrt(sum(value * value for value in values))
        if norm <= 1e-9:
            continue
        x, y, z = (value / norm for value in values)
        horizontal = math.sqrt(x * x + y * y)
        side_usable = 1.0 if horizontal >= 0.70 else 0.0
        alignment = x * desired_x + y * desired_y
        correct_side = x * lateral_sign
        top_down = -z
        # The final negative contact id makes the public descending sort keep
        # the result stable and deterministic for genuinely identical poses.
        score = (
            side_usable,
            alignment if side_usable else top_down,
            correct_side,
            y,
            -abs(z),
            -float(raw_contact_id),
        )
        scored.append((score, int(raw_contact_id)))
    return [
        contact_id
        for _, contact_id in sorted(
            scored,
            key=lambda item: item[0],
            reverse=True,
        )
    ]


def select_reachable_grasp_roll(
    statuses: Sequence[str],
    trajectories: Sequence[Sequence[Sequence[float]]],
    joint_limits: Sequence[Sequence[float]],
    *,
    minimum_normalized_joint_margin: float = 0.10,
) -> int | None:
    """Select a safe reachable wrist roll with the smoothest joint path.

    A tiny terminal-margin advantage must not force a long or discontinuous
    inverse-kinematics branch.  Candidate paths first clear one explicit
    normalized joint-limit floor and are then ordered by continuity, travel,
    terminal displacement, and finally residual joint-limit margin.
    """

    floor = float(minimum_normalized_joint_margin)
    if floor < 0.0:
        raise ValueError("minimum joint margin must be non-negative")
    scored: list[tuple[tuple[float, float, float, float, int], int]] = []
    for index, status in enumerate(statuses):
        if str(status) != "Success" or index >= len(trajectories):
            continue
        path = trajectories[index]
        if len(path) == 0:
            continue
        metrics = grasp_roll_path_metrics(path, joint_limits)
        minimum_margin = float(metrics["minimum_joint_limit_margin"])
        if minimum_margin < floor:
            continue
        scored.append(
            (
                (
                    _equivalence_bucket(
                        metrics["maximum_normalized_joint_step"],
                        MAXIMUM_NORMALIZED_JOINT_STEP_EQUIVALENCE,
                    ),
                    _equivalence_bucket(
                        metrics["normalized_total_travel"],
                        NORMALIZED_JOINT_TRAVEL_EQUIVALENCE,
                    ),
                    _equivalence_bucket(
                        metrics["normalized_terminal_displacement"],
                        NORMALIZED_JOINT_TRAVEL_EQUIVALENCE,
                    ),
                    -minimum_margin,
                    int(index),
                ),
                index,
            )
        )
    if not scored:
        return None
    return min(scored, key=lambda item: item[0])[1]


def grasp_roll_path_metrics(
    trajectory: Sequence[Sequence[float]],
    joint_limits: Sequence[Sequence[float]],
) -> dict[str, float]:
    """Return normalized safety and continuity metrics for one joint path."""

    path = [[float(value) for value in row] for row in trajectory]
    if not path:
        raise ValueError("trajectory must not be empty")
    spans: list[float] = []
    limits: list[tuple[float, float]] = []
    for bounds in joint_limits:
        if len(bounds) != 2:
            raise ValueError("joint limits must contain lower/upper pairs")
        lower, upper = (float(item) for item in bounds)
        span = upper - lower
        if span <= 1e-9:
            raise ValueError("joint limit span must be positive")
        limits.append((lower, upper))
        spans.append(span)
    joint_count = min(len(spans), *(len(row) for row in path))
    if joint_count <= 0:
        raise ValueError("trajectory has no bounded joints")
    terminal = path[-1]
    margins = [
        min(terminal[index] - limits[index][0], limits[index][1] - terminal[index])
        / spans[index]
        for index in range(joint_count)
    ]
    maximum_step = 0.0
    total_travel = 0.0
    for before, after in zip(path, path[1:]):
        normalized = [
            abs(after[index] - before[index]) / spans[index]
            for index in range(joint_count)
        ]
        maximum_step = max(maximum_step, max(normalized, default=0.0))
        total_travel += sum(normalized)
    terminal_displacement = sum(
        abs(path[-1][index] - path[0][index]) / spans[index]
        for index in range(joint_count)
    )
    return {
        "minimum_joint_limit_margin": min(margins, default=0.0),
        "maximum_normalized_joint_step": maximum_step,
        "normalized_total_travel": total_travel,
        "normalized_terminal_displacement": terminal_displacement,
    }


def planar_obstacle_detour(
    start_xy: Sequence[float],
    target_xy: Sequence[float],
    obstacles: Iterable[Mapping[str, Any]],
    *,
    workspace_bounds_xy: Sequence[Sequence[float]] | None = None,
) -> list[list[float]]:
    """Return two deterministic XY waypoints around the first blocker."""

    start = [float(value) for value in start_xy]
    target = [float(value) for value in target_xy]
    if len(start) != 2 or len(target) != 2:
        raise ValueError("planar endpoints must contain xy")
    dx, dy = target[0] - start[0], target[1] - start[1]
    length = math.hypot(dx, dy)
    if length <= 1e-9:
        return []
    ux, uy = dx / length, dy / length
    rows = []
    all_obstacles = []
    for obstacle in obstacles:
        center = [float(value) for value in obstacle["center_xy"]]
        radius = float(obstacle["clearance_m"])
        projection = (
            (center[0] - start[0]) * ux
            + (center[1] - start[1]) * uy
        )
        t = projection / length
        closest = [start[0] + projection * ux, start[1] + projection * uy]
        distance = math.hypot(
            center[0] - closest[0], center[1] - closest[1]
        )
        item = {
            "center": center,
            "radius": radius,
            "projection": projection,
            "t": t,
            "distance": distance,
        }
        all_obstacles.append(item)
        if 0.02 < t < 0.98 and distance < radius:
            rows.append(item)
    if not rows:
        return []
    blocker = min(rows, key=lambda item: (item["t"], item["distance"]))
    offset = blocker["radius"] * 1.15

    def candidate(side: float) -> tuple[float, list[list[float]]]:
        center = blocker["center"]
        if abs(dy) >= abs(dx):
            direction = 1.0 if dy >= 0.0 else -1.0
            before = [
                center[0] + side * offset,
                center[1] - direction * offset,
            ]
            after = [
                center[0] + side * offset,
                center[1] + direction * offset,
            ]
        else:
            direction = 1.0 if dx >= 0.0 else -1.0
            before = [
                center[0] - direction * offset,
                center[1] + side * offset,
            ]
            after = [
                center[0] + direction * offset,
                center[1] + side * offset,
            ]
        route = [start, before, after, target]
        route_length = sum(
            math.hypot(
                route[index + 1][0] - route[index][0],
                route[index + 1][1] - route[index][1],
            )
            for index in range(len(route) - 1)
        )
        penalty = 0.0
        if workspace_bounds_xy is not None:
            if len(workspace_bounds_xy) != 2:
                raise ValueError("workspace bounds must contain x/y ranges")
            for point in (before, after):
                for axis, value in enumerate(point):
                    lower, upper = (
                        float(item) for item in workspace_bounds_xy[axis]
                    )
                    if value < lower:
                        penalty += (lower - value) * 100.0
                    elif value > upper:
                        penalty += (value - upper) * 100.0
        for obstacle in all_obstacles:
            minimum = min(
                math.hypot(
                    point[0] - obstacle["center"][0],
                    point[1] - obstacle["center"][1],
                )
                for point in (before, after)
            )
            if minimum < obstacle["radius"]:
                penalty += (obstacle["radius"] - minimum) * 100.0
        return route_length + penalty, [before, after]

    return min(
        (candidate(-1.0), candidate(1.0)),
        key=lambda item: (item[0], item[1]),
    )[1]


def articulation_use(state: str) -> str:
    """Map the M13 state vocabulary to the validated asset-use vocabulary."""

    value = str(state)
    mapping = {
        "open": "open",
        "close": "close",
        "on": "switch_on",
        "off": "switch_off",
    }
    if value not in mapping:
        raise ValueError(f"unsupported articulation state: {value}")
    return mapping[value]


def articulation_direction(state: str) -> int:
    """Return the expected active-joint direction for an M13 state."""

    return 1 if articulation_use(state) in {"open", "switch_off"} else -1


def articulation_target(
    *,
    initial: float,
    lower: float,
    upper: float,
    state: str,
    minimum_raw_delta: float,
    target_span_ratio: float,
) -> dict[str, float | int]:
    """Build a bounded active-joint target from the current SceneSpec state.

    The target is intentionally independent of canonical attempt traces.  It
    exceeds the checker threshold when the joint has enough remaining travel,
    but never asks the controller to cross a joint limit.
    """

    span = float(upper) - float(lower)
    if span <= 0.0:
        raise ValueError("invalid articulation limits")
    direction = articulation_direction(state)
    requested = max(
        float(minimum_raw_delta),
        span * float(target_span_ratio),
    )
    available = (
        float(upper) - float(initial)
        if direction > 0
        else float(initial) - float(lower)
    )
    delta = min(requested, max(available, 0.0))
    return {
        "direction": direction,
        "span": span,
        "requested_delta": requested,
        "available_delta": available,
        "target_delta": delta,
        "target_qpos": float(initial) + direction * delta,
    }


def grasp_attempts(
    contact_count: int,
    *,
    requested_pre_grasp_m: float,
    preferred_contact_ids: Sequence[int] = (),
    maximum_attempts: int = 6,
) -> list[dict[str, Any]]:
    """Return a deterministic bounded set of current-scene grasp attempts."""

    count = max(int(contact_count), 0)
    preferred = [
        int(value)
        for value in preferred_contact_ids
        if 0 <= int(value) < count
    ]
    contact_ids = list(dict.fromkeys(preferred + list(range(count))))
    pre_distances = list(
        dict.fromkeys(
            (
                max(float(requested_pre_grasp_m), 0.0),
                max(min(float(requested_pre_grasp_m), 0.04), 0.0),
                0.02,
            )
        )
    )
    attempts: list[dict[str, Any]] = []
    seen: set[tuple[int, float]] = set()
    # Contact-distance round-robin: cover distinct contacts while ensuring all
    # three distances appear in the bounded batch.  The top two preferred
    # contacts and one later contact keep the requested clearance; shorter
    # approaches are interleaved as recovery candidates instead of replacing
    # every high-confidence requested-distance attempt.
    distance_base_order = list(range(len(pre_distances)))
    if len(pre_distances) == 3:
        distance_base_order = [0, 0, 2, 1, 0, 2]
    for round_index in range(len(pre_distances)):
        for contact_index, contact_id in enumerate(contact_ids):
            base_index = distance_base_order[
                contact_index % len(distance_base_order)
            ]
            pre_grasp_m = pre_distances[
                (base_index + round_index) % len(pre_distances)
            ]
            key = (contact_id, pre_grasp_m)
            if key in seen:
                continue
            seen.add(key)
            attempts.append(
                {
                    "contact_point_id": contact_id,
                    "pre_grasp_m": pre_grasp_m,
                }
            )
            if len(attempts) >= int(maximum_attempts):
                return attempts
    return attempts


def ordered_articulation_contact_ids(
    contact_points: Iterable[Mapping[str, Any]],
    *,
    moving_link: str | None,
) -> list[int]:
    """Prefer authored contacts attached to the active joint's moving link."""

    rows = list(contact_points)
    moving = [
        index
        for index, item in enumerate(rows)
        if moving_link and str(item.get("base") or "") == moving_link
    ]
    remaining = [index for index in range(len(rows)) if index not in moving]
    return moving + remaining


def checker_articulation_minimum(
    checker_config: Mapping[str, Any] | None,
    *,
    subject: str,
    default: float,
) -> float:
    """Read the intended raw joint-delta threshold without parsing text."""

    values = [
        float(condition.get("minimum_delta", default))
        for condition in (checker_config or {}).get("conditions", [])
        if condition.get("operator") == "articulation_delta"
        and str(condition.get("subject") or "") == str(subject)
    ]
    return max(values, default=float(default))


def lift_completion_minimum(
    checker_config: Mapping[str, Any] | None,
    *,
    subject: str,
    requested_delta_m: float,
    clearance_ratio: float,
    minimum_clearance_m: float,
) -> float:
    """Return the live-state lift postcondition for a symbolic lift step.

    Pick tasks expose an explicit relative-lift checker and use that exact
    threshold.  Placement tasks use lift as a preparatory clearance motion, so
    they require a bounded fraction of the requested displacement instead.
    """

    values = [
        float(condition.get("minimum_delta_m", minimum_clearance_m))
        for condition in (checker_config or {}).get("conditions", [])
        if condition.get("operator") == "relative_lift"
        and str(condition.get("subject") or "") == str(subject)
    ]
    if values:
        return max(values)
    requested = max(float(requested_delta_m), 0.0)
    return min(
        requested,
        max(
            float(minimum_clearance_m),
            requested * float(clearance_ratio),
        ),
    )


def placement_target_pose(
    relation: str,
    target_pose: Sequence[float],
    current_quat_wxyz: Sequence[float],
) -> list[float]:
    """Choose only the orientation needed by the requested relation.

    Planar relations do not constrain object yaw, so retain the live object's
    orientation and the verified grasp transform. Receiver relations retain
    M11's reserved orientation because fitting into or onto the receiver may
    depend on it.
    """

    values = [float(value) for value in target_pose]
    if len(values) != 7:
        raise ValueError("placement target pose must contain xyz + quat_wxyz")
    if str(relation) not in {"in", "on", "stack"}:
        quat = [float(value) for value in current_quat_wxyz]
        if len(quat) != 4:
            raise ValueError("current quaternion must contain four values")
        values[3:] = quat
    return values


__all__ = [
    "STABILITY_COMPONENT_POLICY_VERSION",
    "articulation_direction",
    "articulation_target",
    "articulation_use",
    "checker_articulation_minimum",
    "grasp_roll_path_metrics",
    "grasp_attempts",
    "lift_completion_minimum",
    "ordered_articulation_contact_ids",
    "ordered_rigid_grasp_contact_ids",
    "placement_target_pose",
    "planar_obstacle_detour",
    "select_reachable_grasp_roll",
]
