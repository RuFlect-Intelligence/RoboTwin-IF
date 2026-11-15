"""Shared simulator-independent canonical operation placement contracts.

M4 solves one isolated subject pose while M12 solves a multi-Unit layout.  The
two stages intentionally keep different search spaces, but both define reach
at authored operation anchors and keep the complete collision footprint inside
the embodiment workspace.  This module owns M4's deterministic single-subject
solver plus compatibility helpers for the former Pick-only bounds API.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from if_benchmark.operation_anchors import select_table_pick_anchor
from if_benchmark.schema import artifact_hash
from if_benchmark.schema.serialization import canonical_data


_CANONICAL_LAYOUT_CONFIG_PATH = (
    Path(__file__).resolve().parent / "data" / "layout" / "layout_config.yml"
)


def _shared_layout_geometry() -> dict[str, Any]:
    with _CANONICAL_LAYOUT_CONFIG_PATH.open("r", encoding="utf-8") as stream:
        raw = yaml.load(stream, Loader=yaml.CSafeLoader)
    if not isinstance(raw, Mapping):
        raise ValueError("canonical layout config must be a mapping")
    subset = canonical_data(
        {
            "source_schema_version": raw.get("schema_version"),
            "embodiment": raw.get("embodiment"),
            "table": raw.get("table"),
            "workspace": raw.get("workspace"),
            "camera": {
                "name": (raw.get("camera") or {}).get("name"),
                "table_visible_polygon_xy": (raw.get("camera") or {}).get(
                    "table_visible_polygon_xy"
                ),
            },
            "reach": raw.get("reach"),
        }
    )
    return canonical_data(
        {
            **subset,
            "geometry_subset_hash": artifact_hash(
                {"layout_geometry_subset": subset}
            ),
        }
    )


SHARED_LAYOUT_GEOMETRY = _shared_layout_geometry()


M4_SINGLE_OBJECT_PLACEMENT_POLICY_VERSION = (
    "m4_single_object_placement_v2_r01_anchor_workspace"
)
M4_SINGLE_OBJECT_PLACEMENT_POLICY = canonical_data(
    {
        "version": M4_SINGLE_OBJECT_PLACEMENT_POLICY_VERSION,
        "scope": "m4_isolated_single_subject",
        "embodiment": SHARED_LAYOUT_GEOMETRY["embodiment"],
        # These are the same physical embodiment, table, and camera limits
        # used by the later scene solver.  They are copied into the M4 policy
        # hash; the hash itself is deliberately not an M12 input.
        "layout_geometry_subset_hash": SHARED_LAYOUT_GEOMETRY[
            "geometry_subset_hash"
        ],
        "workspace_bounds_xy_m": SHARED_LAYOUT_GEOMETRY["workspace"][
            "bounds_xy"
        ],
        "camera_visible_polygon_xy_m": SHARED_LAYOUT_GEOMETRY["camera"][
            "table_visible_polygon_xy"
        ],
        "reach_bounds_xy_m": SHARED_LAYOUT_GEOMETRY["reach"],
        "use_arm_mode": {
            "pick_left": "fixed_left",
            "pick_right": "fixed_right",
            "click_subject": "any_explicit",
            "press_subject": "any_explicit",
            "rotate_to_orientation_subject": "any_explicit",
            "adjust_orientation_subject": "any_explicit",
            "open": "both_existential_shared_pose",
            "close": "both_existential_shared_pose",
            "switch_on": "both_existential_shared_pose",
            "switch_off": "both_existential_shared_pose",
            "bimanual_grab_subject": "bimanual_contact_groups",
        },
        "candidate_generation": (
            "constraint_polygon_centroid_bbox_center_vertices_midpoints"
        ),
        "candidate_ranking": (
            "maximum_minimum_constraint_clearance_then_total_clearance"
        ),
        "random_sampling": False,
        "r01_pose_fields_immutable": [
            "q0_wxyz",
            "support.origin_z",
            "support.plane_z",
            "articulated_joint_state.qpos",
            "yaw_policy",
        ],
        # Compatibility-only ranges retained for callers of the old
        # resolve_single_arm_pick_placement API.  The M4 environment no longer
        # uses these ranges to spawn subjects.
        "single_arm": {
            "left": {
                "actor_origin_safety_xy_m": {
                    "x": [-0.25, -0.15],
                    "y": [-0.12, -0.02],
                },
                "target_contact_xy_m": {
                    "x": [-0.22, -0.16],
                    "y": [-0.06, -0.02],
                },
            },
            "right": {
                "actor_origin_safety_xy_m": {
                    "x": [0.15, 0.25],
                    "y": [-0.12, -0.02],
                },
                "target_contact_xy_m": {
                    "x": [0.16, 0.22],
                    "y": [-0.06, -0.02],
                },
            },
        },
        "legacy_single_arm_scope": "compatibility_only_not_m4_spawn",
        "disjoint_resolution": "project_to_nearest_actor_safety_boundary",
        "d4_both_arm_mode": "shared_static_reach_intersection",
    }
)
M4_SINGLE_OBJECT_PLACEMENT_POLICY_HASH = artifact_hash(
    M4_SINGLE_OBJECT_PLACEMENT_POLICY
)

# Click/Press tightens only its own placement quantifier. Keeping this in a
# separate hash domain preserves every non-touch M4 witness/report that still
# binds the unchanged v2 single-object policy above.
M4_TOUCH_PLACEMENT_POLICY_VERSION = (
    "m4_touch_placement_v1_same_anchor_both_arms"
)
M4_TOUCH_PLACEMENT_POLICY = canonical_data(
    {
        "version": M4_TOUCH_PLACEMENT_POLICY_VERSION,
        "scope": "m4_click_press_only",
        "base_policy_version": M4_SINGLE_OBJECT_PLACEMENT_POLICY_VERSION,
        "base_policy_hash": M4_SINGLE_OBJECT_PLACEMENT_POLICY_HASH,
        "uses": ["click_subject", "press_subject"],
        "arm_mode": "touch_both_arms_same_anchor_shared_pose",
        "requested_arms": ["left", "right"],
        "same_root_pose": True,
        "same_authored_anchor": True,
    }
)
M4_TOUCH_PLACEMENT_POLICY_HASH = artifact_hash(M4_TOUCH_PLACEMENT_POLICY)

# The report/probe-plan field keeps its historical name for schema
# compatibility.  It now binds the complete M4 placement policy, not only the
# old single-arm Pick bands.
PICK_OPERATION_PLACEMENT_POLICY_VERSION = (
    M4_SINGLE_OBJECT_PLACEMENT_POLICY_VERSION
)
PICK_OPERATION_PLACEMENT_POLICY = M4_SINGLE_OBJECT_PLACEMENT_POLICY
PICK_OPERATION_PLACEMENT_POLICY_HASH = M4_SINGLE_OBJECT_PLACEMENT_POLICY_HASH


# D4 uses one initial pose for the left/right minimal pair.  It therefore
# cannot consume the disjoint single-arm canonical-probe envelopes above.  The
# separate hash domain keeps an M12-only geometric pruning revision from
# invalidating M4's already-proved single-arm capability evidence.
D4_SHARED_PICK_PLACEMENT_POLICY_VERSION = "d4_shared_pick_placement_v1"
D4_SHARED_PICK_PLACEMENT_POLICY = canonical_data(
    {
        "version": D4_SHARED_PICK_PLACEMENT_POLICY_VERSION,
        "embodiment": "aloha-agilex",
        "scope": "m12_search_pruning_only",
        "actor_origin_safety_xy_m": {
            # The x interval is the overlap of the embodiment's left/right
            # static reach proxies.  The y interval is the common M4
            # actor-safety band, keeping shared picks on the robot-facing half
            # of that overlap instead of anywhere in the broad reach box.
            "x": [-0.08, 0.08],
            "y": [-0.12, -0.02],
        },
        "target_contact_xy_m": {
            "x": [-0.08, 0.08],
            "y": [-0.06, -0.02],
        },
        "contact_selection": "any_eligible_non_upward_grasp_contact",
        "interval_resolution": "exact_intersection_without_projection",
        "planning_witness_required": True,
        "final_feasibility_claim": "none",
    }
)
D4_SHARED_PICK_PLACEMENT_POLICY_HASH = artifact_hash(
    D4_SHARED_PICK_PLACEMENT_POLICY
)


def _ordered_interval(values: Sequence[float], *, field: str) -> list[float]:
    if len(values) != 2:
        raise ValueError(f"{field} must contain two values")
    result = [float(values[0]), float(values[1])]
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{field} must contain finite values")
    if result[0] > result[1]:
        raise ValueError(f"{field} must be ordered")
    return result


def _safe_interval_intersection(
    requested: Sequence[float], safe: Sequence[float]
) -> tuple[list[float], str]:
    requested_interval = _ordered_interval(requested, field="requested interval")
    safe_interval = _ordered_interval(safe, field="safe interval")
    lower = max(requested_interval[0], safe_interval[0])
    upper = min(requested_interval[1], safe_interval[1])
    if lower <= upper:
        return [lower, upper], "intersection"
    boundary = (
        safe_interval[0]
        if requested_interval[1] < safe_interval[0]
        else safe_interval[1]
    )
    return [boundary, boundary], "projected_to_actor_safety_boundary"


def _exact_interval_intersection(
    requested: Sequence[float], safe: Sequence[float]
) -> list[float] | None:
    """Return an exact interval intersection without boundary projection."""

    requested_interval = _ordered_interval(requested, field="requested interval")
    safe_interval = _ordered_interval(safe, field="safe interval")
    lower = max(requested_interval[0], safe_interval[0])
    upper = min(requested_interval[1], safe_interval[1])
    return [lower, upper] if lower <= upper else None


def _rotated_anchor_position(
    position_xyz: Sequence[float], relative_yaw_deg: float
) -> list[float]:
    if len(position_xyz) != 3:
        raise ValueError("operation anchor position must contain xyz")
    position = [float(value) for value in position_xyz]
    if not all(math.isfinite(value) for value in position):
        raise ValueError("operation anchor position must be finite")
    yaw = math.radians(float(relative_yaw_deg))
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return [
        cosine * position[0] - sine * position[1],
        sine * position[0] + cosine * position[1],
        position[2],
    ]


def _origin_bounds_for_target(
    anchor_position_xyz: Sequence[float],
    *,
    target_x: Sequence[float],
    target_y: Sequence[float],
) -> dict[str, list[float]]:
    anchor = _rotated_anchor_position(anchor_position_xyz, 0.0)
    target_x_interval = _ordered_interval(target_x, field="target x")
    target_y_interval = _ordered_interval(target_y, field="target y")
    return {
        "xlim": [
            target_x_interval[0] - anchor[0],
            target_x_interval[1] - anchor[0],
        ],
        "ylim": [
            target_y_interval[0] - anchor[1],
            target_y_interval[1] - anchor[1],
        ],
    }


def _polygon_signed_area(polygon: Sequence[Sequence[float]]) -> float:
    return 0.5 * sum(
        float(row[0]) * float(polygon[(index + 1) % len(polygon)][1])
        - float(polygon[(index + 1) % len(polygon)][0]) * float(row[1])
        for index, row in enumerate(polygon)
    )


def _normalized_convex_polygon(
    polygon: Sequence[Sequence[float]], *, field: str
) -> list[list[float]]:
    points = [[float(value[0]), float(value[1])] for value in polygon]
    if len(points) < 3 or not all(
        math.isfinite(coordinate)
        for point in points
        for coordinate in point
    ):
        raise ValueError(f"{field} must be a finite polygon")
    area = _polygon_signed_area(points)
    if math.isclose(area, 0.0, abs_tol=1e-12):
        raise ValueError(f"{field} has zero area")
    return points if area > 0.0 else list(reversed(points))


def _clip_polygon_halfplane(
    polygon: Sequence[Sequence[float]],
    *,
    normal_xy: Sequence[float],
    minimum_dot: float,
    tolerance: float = 1e-12,
) -> list[list[float]]:
    """Clip a convex polygon by ``normal dot point >= minimum_dot``."""

    if not polygon:
        return []
    nx, ny = float(normal_xy[0]), float(normal_xy[1])
    threshold = float(minimum_dot)

    def signed(point: Sequence[float]) -> float:
        return nx * float(point[0]) + ny * float(point[1]) - threshold

    output: list[list[float]] = []
    previous = [float(value) for value in polygon[-1]]
    previous_signed = signed(previous)
    previous_inside = previous_signed >= -tolerance
    for raw_current in polygon:
        current = [float(value) for value in raw_current]
        current_signed = signed(current)
        current_inside = current_signed >= -tolerance
        if current_inside != previous_inside:
            denominator = previous_signed - current_signed
            if not math.isclose(denominator, 0.0, abs_tol=1e-15):
                fraction = previous_signed / denominator
                output.append(
                    [
                        previous[0] + fraction * (current[0] - previous[0]),
                        previous[1] + fraction * (current[1] - previous[1]),
                    ]
                )
        if current_inside:
            output.append(current)
        previous = current
        previous_signed = current_signed
        previous_inside = current_inside
    return output


def _clip_polygon_to_rect(
    polygon: Sequence[Sequence[float]],
    *,
    xlim: Sequence[float],
    ylim: Sequence[float],
) -> list[list[float]]:
    x_interval = _ordered_interval(xlim, field="x clip")
    y_interval = _ordered_interval(ylim, field="y clip")
    clipped = [list(map(float, row)) for row in polygon]
    for normal, threshold in (
        ([1.0, 0.0], x_interval[0]),
        ([-1.0, 0.0], -x_interval[1]),
        ([0.0, 1.0], y_interval[0]),
        ([0.0, -1.0], -y_interval[1]),
    ):
        clipped = _clip_polygon_halfplane(
            clipped, normal_xy=normal, minimum_dot=threshold
        )
        if not clipped:
            break
    return clipped


def _rotated_xy_bounds(
    origin_relative_bounds: Mapping[str, Any], relative_yaw_deg: float
) -> dict[str, list[float]]:
    minimum = origin_relative_bounds.get("min_xyz")
    maximum = origin_relative_bounds.get("max_xyz")
    if (
        not isinstance(minimum, Sequence)
        or isinstance(minimum, (str, bytes))
        or not isinstance(maximum, Sequence)
        or isinstance(maximum, (str, bytes))
        or len(minimum) != 3
        or len(maximum) != 3
    ):
        raise ValueError("origin_relative_bounds must contain min_xyz/max_xyz")
    minimum_xyz = [float(value) for value in minimum]
    maximum_xyz = [float(value) for value in maximum]
    if not all(math.isfinite(value) for value in minimum_xyz + maximum_xyz):
        raise ValueError("origin_relative_bounds must be finite")
    if any(low > high for low, high in zip(minimum_xyz, maximum_xyz)):
        raise ValueError("origin_relative_bounds must be ordered")
    yaw = math.radians(float(relative_yaw_deg))
    cosine, sine = math.cos(yaw), math.sin(yaw)
    rotated = [
        [cosine * x - sine * y, sine * x + cosine * y]
        for x in (minimum_xyz[0], maximum_xyz[0])
        for y in (minimum_xyz[1], maximum_xyz[1])
    ]
    return {
        "min_xy": [
            min(point[0] for point in rotated),
            min(point[1] for point in rotated),
        ],
        "max_xy": [
            max(point[0] for point in rotated),
            max(point[1] for point in rotated),
        ],
    }


def _eroded_actor_origin_polygon(
    *,
    footprint_bounds: Mapping[str, Sequence[float]],
) -> list[list[float]]:
    policy = M4_SINGLE_OBJECT_PLACEMENT_POLICY
    camera = _normalized_convex_polygon(
        policy["camera_visible_polygon_xy_m"], field="camera polygon"
    )
    minimum = [float(value) for value in footprint_bounds["min_xy"]]
    maximum = [float(value) for value in footprint_bounds["max_xy"]]
    footprint_corners = [
        [x, y]
        for x in (minimum[0], maximum[0])
        for y in (minimum[1], maximum[1])
    ]
    # Erode the convex camera polygon by the asymmetric actor-origin AABB.
    polygon = camera
    for index, edge_start in enumerate(camera):
        edge_end = camera[(index + 1) % len(camera)]
        edge = [
            edge_end[0] - edge_start[0],
            edge_end[1] - edge_start[1],
        ]
        inward_normal = [-edge[1], edge[0]]
        edge_threshold = (
            inward_normal[0] * edge_start[0]
            + inward_normal[1] * edge_start[1]
        )
        minimum_offset = min(
            inward_normal[0] * corner[0]
            + inward_normal[1] * corner[1]
            for corner in footprint_corners
        )
        polygon = _clip_polygon_halfplane(
            polygon,
            normal_xy=inward_normal,
            minimum_dot=edge_threshold - minimum_offset,
        )
        if not polygon:
            return []
    workspace = policy["workspace_bounds_xy_m"]
    return _clip_polygon_to_rect(
        polygon,
        xlim=[
            float(workspace["x"][0]) - minimum[0],
            float(workspace["x"][1]) - maximum[0],
        ],
        ylim=[
            float(workspace["y"][0]) - minimum[1],
            float(workspace["y"][1]) - maximum[1],
        ],
    )


def _polygon_centroid(polygon: Sequence[Sequence[float]]) -> list[float]:
    area_twice = sum(
        float(row[0]) * float(polygon[(index + 1) % len(polygon)][1])
        - float(polygon[(index + 1) % len(polygon)][0]) * float(row[1])
        for index, row in enumerate(polygon)
    )
    if math.isclose(area_twice, 0.0, abs_tol=1e-12):
        return [
            sum(float(row[0]) for row in polygon) / len(polygon),
            sum(float(row[1]) for row in polygon) / len(polygon),
        ]
    factor = 1.0 / (3.0 * area_twice)
    return [
        factor
        * sum(
            (float(row[0]) + float(polygon[(index + 1) % len(polygon)][0]))
            * (
                float(row[0]) * float(polygon[(index + 1) % len(polygon)][1])
                - float(polygon[(index + 1) % len(polygon)][0])
                * float(row[1])
            )
            for index, row in enumerate(polygon)
        ),
        factor
        * sum(
            (float(row[1]) + float(polygon[(index + 1) % len(polygon)][1]))
            * (
                float(row[0]) * float(polygon[(index + 1) % len(polygon)][1])
                - float(polygon[(index + 1) % len(polygon)][0])
                * float(row[1])
            )
            for index, row in enumerate(polygon)
        ),
    ]


def _point_in_convex_polygon(
    point: Sequence[float], polygon: Sequence[Sequence[float]]
) -> bool:
    for index, start in enumerate(polygon):
        end = polygon[(index + 1) % len(polygon)]
        if (
            (float(end[0]) - float(start[0]))
            * (float(point[1]) - float(start[1]))
            - (float(end[1]) - float(start[1]))
            * (float(point[0]) - float(start[0]))
            < -1e-10
        ):
            return False
    return True


def _polygon_feature_candidates(
    polygon: Sequence[Sequence[float]],
) -> list[list[float]]:
    candidates = [_polygon_centroid(polygon)]
    minimum_x = min(float(row[0]) for row in polygon)
    maximum_x = max(float(row[0]) for row in polygon)
    minimum_y = min(float(row[1]) for row in polygon)
    maximum_y = max(float(row[1]) for row in polygon)
    bbox_center = [
        0.5 * (minimum_x + maximum_x),
        0.5 * (minimum_y + maximum_y),
    ]
    if _point_in_convex_polygon(bbox_center, polygon):
        candidates.append(bbox_center)
    for index, row in enumerate(polygon):
        next_row = polygon[(index + 1) % len(polygon)]
        candidates.append([float(row[0]), float(row[1])])
        candidates.append(
            [
                0.5 * (float(row[0]) + float(next_row[0])),
                0.5 * (float(row[1]) + float(next_row[1])),
            ]
        )
    unique: dict[tuple[float, float], list[float]] = {}
    for point in candidates:
        unique.setdefault(
            (round(float(point[0]), 12), round(float(point[1]), 12)),
            [float(point[0]), float(point[1])],
        )
    return list(unique.values())


def _polygon_clearance(
    point: Sequence[float], polygon: Sequence[Sequence[float]]
) -> float:
    clearances: list[float] = []
    for index, start in enumerate(polygon):
        end = polygon[(index + 1) % len(polygon)]
        edge_x = float(end[0]) - float(start[0])
        edge_y = float(end[1]) - float(start[1])
        length = math.hypot(edge_x, edge_y)
        if length <= 1e-12:
            continue
        clearances.append(
            (
                edge_x * (float(point[1]) - float(start[1]))
                - edge_y * (float(point[0]) - float(start[0]))
            )
            / length
        )
    return min(clearances, default=0.0)


def _eligible_grasp_anchors(
    contract: Mapping[str, Any],
    *,
    allowed_contact_ids: Sequence[int] | None,
    require_moves_with_active_joint: bool,
) -> list[dict[str, Any]]:
    allowed = (
        {int(value) for value in allowed_contact_ids}
        if allowed_contact_ids is not None
        else None
    )
    rows: list[dict[str, Any]] = []
    for raw in contract.get("anchors", []):
        if (
            not isinstance(raw, Mapping)
            or raw.get("kind") != "grasp_contact"
            or raw.get("operation_eligible", True) is not True
            or len(raw.get("position_xyz_m") or []) != 3
        ):
            continue
        point_index = int(raw.get("point_index", -1))
        if point_index < 0 or (allowed is not None and point_index not in allowed):
            continue
        if require_moves_with_active_joint and raw.get(
            "moves_with_active_joint"
        ) is not True:
            continue
        # The shared Pick planner owns exact support-envelope admission.  M4
        # placement must expose its complete authored Contact domain rather
        # than pre-reject an orientation tier with a private heuristic.
        rows.append(dict(raw))
    return sorted(rows, key=lambda row: (int(row["point_index"]), str(row.get("id"))))


def _arm_anchor_region(
    polygon: Sequence[Sequence[float]],
    *,
    arm: str,
    anchor: Mapping[str, Any],
    relative_yaw_deg: float,
) -> tuple[list[list[float]], list[float]]:
    rotated = _rotated_anchor_position(
        anchor["position_xyz_m"], relative_yaw_deg
    )
    reach = M4_SINGLE_OBJECT_PLACEMENT_POLICY["reach_bounds_xy_m"][arm]
    return (
        _clip_polygon_to_rect(
            polygon,
            xlim=[
                float(reach["x"][0]) - rotated[0],
                float(reach["x"][1]) - rotated[0],
            ],
            ylim=[
                float(reach["y"][0]) - rotated[1],
                float(reach["y"][1]) - rotated[1],
            ],
        ),
        rotated,
    )


def _candidate_score(
    point: Sequence[float],
    polygon: Sequence[Sequence[float]],
    selected: Sequence[Mapping[str, Any]],
) -> tuple[float, float, float, float]:
    margins = [_polygon_clearance(point, polygon)]
    for row in selected:
        anchor = row["anchor_position_after_yaw_xyz_m"]
        arm = str(row["arm"])
        reach = M4_SINGLE_OBJECT_PLACEMENT_POLICY["reach_bounds_xy_m"][arm]
        world_x = float(point[0]) + float(anchor[0])
        world_y = float(point[1]) + float(anchor[1])
        margins.extend(
            [
                world_x - float(reach["x"][0]),
                float(reach["x"][1]) - world_x,
                world_y - float(reach["y"][0]),
                float(reach["y"][1]) - world_y,
            ]
        )
    # Final coordinates make tie-breaking stable but never asset-named.
    return (
        min(margins),
        sum(margins),
        -abs(float(point[1])),
        -abs(float(point[0])),
    )


def _placement_candidate(
    polygon: Sequence[Sequence[float]],
    *,
    selected: Sequence[Mapping[str, Any]],
    selected_arm: str | None,
) -> dict[str, Any] | None:
    if not polygon:
        return None
    candidates = _polygon_feature_candidates(polygon)
    ranked = [
        (_candidate_score(point, polygon, selected), point)
        for point in candidates
        if _point_in_convex_polygon(point, polygon)
    ]
    if not ranked:
        return None
    score, point = max(
        ranked,
        key=lambda row: (
            row[0],
            -round(float(row[1][0]), 12),
            -round(float(row[1][1]), 12),
        ),
    )
    return {
        "origin_xy": [float(point[0]), float(point[1])],
        "score": [float(value) for value in score],
        "feasible_origin_polygon_xy": [
            [float(value) for value in row] for row in polygon
        ],
        "selected": [dict(row) for row in selected],
        "selected_arm": selected_arm,
    }


def m4_runtime_refinement_origins(
    witness: Mapping[str, Any],
    *,
    spacing_m: float = 0.02,
) -> list[dict[str, Any]]:
    """Enumerate deterministic in-domain origins for runtime pose checks.

    The static M4 solver proves footprint, camera, workspace, and authored
    Contact XY bounds.  Some articulated end-effector frames still lie outside
    the arm's full 6-DoF workspace.  Runtime may use these points to refine the
    already-legal origin with the real collision-aware IK adapter; it must
    never search outside the frozen feasible polygon.
    """

    polygon = witness.get("feasible_origin_polygon_xy")
    origin = witness.get("origin_xy_m")
    selected = witness.get("selected_anchors")
    step = float(spacing_m)
    if (
        not isinstance(polygon, Sequence)
        or isinstance(polygon, (str, bytes))
        or len(polygon) < 3
        or not isinstance(origin, Sequence)
        or isinstance(origin, (str, bytes))
        or len(origin) != 2
        or not isinstance(selected, Sequence)
        or isinstance(selected, (str, bytes))
        or not math.isfinite(step)
        or step <= 0.0
    ):
        return []
    try:
        polygon_xy = [
            [float(point[0]), float(point[1])]
            for point in polygon
        ]
        original = [float(origin[0]), float(origin[1])]
        selected_rows = [dict(row) for row in selected if isinstance(row, Mapping)]
    except (IndexError, TypeError, ValueError):
        return []
    if len(selected_rows) != len(selected) or not all(
        math.isfinite(value)
        for point in polygon_xy
        for value in point
    ) or not all(math.isfinite(value) for value in original):
        return []

    minimum_x = min(point[0] for point in polygon_xy)
    maximum_x = max(point[0] for point in polygon_xy)
    minimum_y = min(point[1] for point in polygon_xy)
    maximum_y = max(point[1] for point in polygon_xy)

    def axis_values(lower: float, upper: float, current: float) -> list[float]:
        span = max(0.0, float(upper) - float(lower))
        interval_count = max(1, int(math.ceil(span / step)))
        values = [
            float(lower) + span * index / interval_count
            for index in range(interval_count + 1)
        ]
        values.append(float(current))
        return sorted({round(value, 12) for value in values})

    raw_points = [original, *_polygon_feature_candidates(polygon_xy)]
    raw_points.extend(
        [original[0], value]
        for value in axis_values(minimum_y, maximum_y, original[1])
    )
    raw_points.extend(
        [value, original[1]]
        for value in axis_values(minimum_x, maximum_x, original[0])
    )

    unique: dict[tuple[float, float], list[float]] = {}
    for point in raw_points:
        candidate = [float(point[0]), float(point[1])]
        if not _point_in_convex_polygon(candidate, polygon_xy):
            continue
        key = (round(candidate[0], 12), round(candidate[1], 12))
        unique[key] = [float(key[0]), float(key[1])]

    rows = []
    for point in unique.values():
        displacement = math.hypot(
            point[0] - original[0], point[1] - original[1]
        )
        rows.append(
            {
                "origin_xy_m": point,
                "translation_from_static_origin_m": float(displacement),
                "static_score": [
                    float(value)
                    for value in _candidate_score(
                        point, polygon_xy, selected_rows
                    )
                ],
            }
        )
    rows.sort(
        key=lambda row: (
            round(float(row["translation_from_static_origin_m"]), 12),
            # The ALOHA bases sit on the negative world-Y side.  Prefer that
            # direction only as a deterministic tie-break; IK remains the
            # actual acceptance gate.
            float(row["origin_xy_m"][1]),
            abs(float(row["origin_xy_m"][0])),
            float(row["origin_xy_m"][0]),
        )
    )
    return canonical_data(rows)


def resolve_m4_single_object_placement(
    use: str,
    *,
    origin_relative_bounds: Mapping[str, Any],
    support: Mapping[str, Any],
    operation_anchor_contract: Mapping[str, Any],
    canonical_state_hash: str,
    relative_yaw_deg: float = 0.0,
    allowed_contact_ids: Sequence[int] | None = None,
    contact_groups: Sequence[Sequence[int]] | None = None,
) -> dict[str, Any]:
    """Solve one deterministic M4 actor XY from physical constraints.

    The result is attempt-owned evidence.  ``any_explicit`` returns the arm
    used to prove the pose. Click/Press solve one shared XY and require the
    same authored Contact to be reachable by both arms. Articulation solves
    one shared XY where each arm has at least one (not necessarily the same)
    reachable authored Contact.
    """

    use_name = str(use)
    touch_placement = use_name in M4_TOUCH_PLACEMENT_POLICY["uses"]
    arm_mode = (
        M4_TOUCH_PLACEMENT_POLICY["arm_mode"]
        if touch_placement
        else M4_SINGLE_OBJECT_PLACEMENT_POLICY["use_arm_mode"].get(use_name)
    )
    if arm_mode is None:
        raise ValueError(f"unsupported M4 placement use: {use_name!r}")
    if not isinstance(operation_anchor_contract, Mapping):
        raise ValueError("operation_anchor_contract is required")
    footprint = _rotated_xy_bounds(origin_relative_bounds, relative_yaw_deg)
    base_polygon = _eroded_actor_origin_polygon(
        footprint_bounds=footprint
    )
    anchors = _eligible_grasp_anchors(
        operation_anchor_contract,
        allowed_contact_ids=allowed_contact_ids,
        require_moves_with_active_joint=use_name in {"open", "close"},
    )
    candidates: list[dict[str, Any]] = []

    def selected_row(
        arm: str, anchor: Mapping[str, Any], rotated: Sequence[float]
    ) -> dict[str, Any]:
        return {
            "arm": str(arm),
            "anchor_id": str(anchor.get("id")),
            "point_index": int(anchor["point_index"]),
            "anchor_position_after_yaw_xyz_m": [float(v) for v in rotated],
        }

    if base_polygon and arm_mode in {"fixed_left", "fixed_right", "any_explicit"}:
        arms = (
            ["left"]
            if arm_mode == "fixed_left"
            else ["right"]
            if arm_mode == "fixed_right"
            # Right first preserves the old single-arm default without using
            # an actor-name or spawn-sign heuristic.
            else ["right", "left"]
        )
        for arm in arms:
            for anchor in anchors:
                region, rotated = _arm_anchor_region(
                    base_polygon,
                    arm=arm,
                    anchor=anchor,
                    relative_yaw_deg=relative_yaw_deg,
                )
                candidate = _placement_candidate(
                    region,
                    selected=[selected_row(arm, anchor, rotated)],
                    selected_arm=arm,
                )
                if candidate is not None:
                    candidates.append(candidate)
    elif (
        base_polygon
        and arm_mode == "touch_both_arms_same_anchor_shared_pose"
    ):
        for anchor in anchors:
            left_region, left_rotated = _arm_anchor_region(
                base_polygon,
                arm="left",
                anchor=anchor,
                relative_yaw_deg=relative_yaw_deg,
            )
            region, right_rotated = _arm_anchor_region(
                left_region,
                arm="right",
                anchor=anchor,
                relative_yaw_deg=relative_yaw_deg,
            )
            selected = [
                selected_row("left", anchor, left_rotated),
                selected_row("right", anchor, right_rotated),
            ]
            candidate = _placement_candidate(
                region, selected=selected, selected_arm=None
            )
            if candidate is not None:
                candidates.append(candidate)
    elif base_polygon and arm_mode == "both_existential_shared_pose":
        for left_anchor in anchors:
            left_region, left_rotated = _arm_anchor_region(
                base_polygon,
                arm="left",
                anchor=left_anchor,
                relative_yaw_deg=relative_yaw_deg,
            )
            for right_anchor in anchors:
                region, right_rotated = _arm_anchor_region(
                    left_region,
                    arm="right",
                    anchor=right_anchor,
                    relative_yaw_deg=relative_yaw_deg,
                )
                selected = [
                    selected_row("left", left_anchor, left_rotated),
                    selected_row("right", right_anchor, right_rotated),
                ]
                candidate = _placement_candidate(
                    region, selected=selected, selected_arm=None
                )
                if candidate is not None:
                    candidates.append(candidate)
    elif base_polygon and arm_mode == "bimanual_contact_groups":
        groups = [
            {int(value) for value in group}
            for group in (contact_groups or [])
            if group
        ]
        if len(groups) >= 2:
            # Bimanual execution enumerates every ordered assignment of two
            # distinct authored groups while the actor remains stationary.
            # Placement must prove the same domain instead of freezing the
            # arbitrary metadata order as left=0/right=1.
            for left_group_index, left_group in enumerate(groups):
                for right_group_index, right_group in enumerate(groups):
                    if left_group_index == right_group_index:
                        continue
                    overlap = left_group & right_group
                    left_unique = left_group - overlap
                    right_unique = right_group - overlap
                    if not left_unique or not right_unique:
                        continue
                    left_anchors = [
                        row
                        for row in anchors
                        if int(row["point_index"]) in left_unique
                    ]
                    right_anchors = [
                        row
                        for row in anchors
                        if int(row["point_index"]) in right_unique
                    ]
                    for left_anchor in left_anchors:
                        left_region, left_rotated = _arm_anchor_region(
                            base_polygon,
                            arm="left",
                            anchor=left_anchor,
                            relative_yaw_deg=relative_yaw_deg,
                        )
                        for right_anchor in right_anchors:
                            region, right_rotated = _arm_anchor_region(
                                left_region,
                                arm="right",
                                anchor=right_anchor,
                                relative_yaw_deg=relative_yaw_deg,
                            )
                            selected = [
                                selected_row("left", left_anchor, left_rotated),
                                selected_row("right", right_anchor, right_rotated),
                            ]
                            candidate = _placement_candidate(
                                region, selected=selected, selected_arm=None
                            )
                            if candidate is not None:
                                candidate["group_assignment"] = {
                                    "left_group_index": left_group_index,
                                    "right_group_index": right_group_index,
                                    "left": sorted(left_unique),
                                    "right": sorted(right_unique),
                                }
                                candidates.append(candidate)

    if not candidates:
        return canonical_data(
            {
                "status": "unsat",
                "failure_code": (
                    "m4_placement_operation_anchor_missing"
                    if not anchors
                    else "m4_placement_constraint_domain_empty"
                ),
                "policy_version": M4_SINGLE_OBJECT_PLACEMENT_POLICY_VERSION,
                "policy_hash": M4_SINGLE_OBJECT_PLACEMENT_POLICY_HASH,
                **(
                    {
                        "touch_placement_policy_version": (
                            M4_TOUCH_PLACEMENT_POLICY_VERSION
                        ),
                        "touch_placement_policy_hash": (
                            M4_TOUCH_PLACEMENT_POLICY_HASH
                        ),
                    }
                    if touch_placement
                    else {}
                ),
                "use": use_name,
                "arm_mode": arm_mode,
                "relative_yaw_deg": float(relative_yaw_deg),
                "canonical_state_hash": str(canonical_state_hash),
                "operation_anchor_contract_hash": operation_anchor_contract.get(
                    "contract_hash"
                ),
                "eligible_anchor_count": len(anchors),
                "footprint_bounds_after_yaw_xy_m": footprint,
                "base_feasible_origin_polygon_xy": base_polygon,
            }
        )

    def candidate_order(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
        selected = candidate["selected"]
        return (
            tuple(float(value) for value in candidate["score"]),
            1 if candidate.get("selected_arm") == "right" else 0,
            tuple(-int(row["point_index"]) for row in selected),
            -abs(float(candidate["origin_xy"][0])),
            -abs(float(candidate["origin_xy"][1])),
        )

    chosen = max(candidates, key=candidate_order)
    support_payload = {
        key: support.get(key)
        for key in ("origin_z", "plane_z", "clearance_m", "formula")
        if support.get(key) is not None
    }
    payload = canonical_data(
        {
            "artifact_kind": "M4CanonicalPlacementWitness",
            "version": M4_SINGLE_OBJECT_PLACEMENT_POLICY_VERSION,
            "policy_version": M4_SINGLE_OBJECT_PLACEMENT_POLICY_VERSION,
            "policy_hash": M4_SINGLE_OBJECT_PLACEMENT_POLICY_HASH,
            **(
                {
                    "touch_placement_policy_version": (
                        M4_TOUCH_PLACEMENT_POLICY_VERSION
                    ),
                    "touch_placement_policy_hash": (
                        M4_TOUCH_PLACEMENT_POLICY_HASH
                    ),
                }
                if touch_placement
                else {}
            ),
            "layout_geometry_subset_hash": M4_SINGLE_OBJECT_PLACEMENT_POLICY[
                "layout_geometry_subset_hash"
            ],
            "status": "passed",
            "use": use_name,
            "arm_mode": arm_mode,
            "selected_arm": chosen.get("selected_arm"),
            "selected_anchors": chosen["selected"],
            "selected_contact_ids_by_arm": {
                str(row["arm"]): int(row["point_index"])
                for row in chosen["selected"]
            },
            "origin_xy_m": chosen["origin_xy"],
            "root_z_m": float(support["origin_z"]),
            "relative_yaw_deg": float(relative_yaw_deg),
            "canonical_state_hash": str(canonical_state_hash),
            "origin_relative_bounds_hash": artifact_hash(
                canonical_data(dict(origin_relative_bounds))
            ),
            "support_hash": artifact_hash(canonical_data(support_payload)),
            "support": support_payload,
            "operation_anchor_contract_hash": operation_anchor_contract.get(
                "contract_hash"
            ),
            "footprint_bounds_after_yaw_xy_m": footprint,
            "feasible_origin_polygon_xy": chosen[
                "feasible_origin_polygon_xy"
            ],
            "candidate_score": chosen["score"],
            "candidate_count": len(candidates),
            **(
                {"group_assignment": chosen["group_assignment"]}
                if chosen.get("group_assignment") is not None
                else {}
            ),
        }
    )
    return canonical_data(
        {
            **payload,
            "witness_hash": artifact_hash(payload),
        }
    )


def m4_canonical_placement_witness_errors(
    witness: Any,
    *,
    expected_use: str | None = None,
    expected_canonical_state_hash: str | None = None,
    require_current_touch_policy: bool = True,
) -> list[str]:
    """Return fail-closed errors for one worker-owned placement witness."""

    if not isinstance(witness, Mapping):
        return ["canonical_placement_witness_missing"]

    def finite_vector(value: Any, length: int) -> bool:
        if (
            not isinstance(value, Sequence)
            or isinstance(value, (str, bytes))
            or len(value) != length
        ):
            return False
        try:
            parsed = [float(item) for item in value]
        except (TypeError, ValueError, OverflowError):
            return False
        return all(
            not isinstance(item, bool) and math.isfinite(number)
            for item, number in zip(value, parsed, strict=True)
        )

    failures: list[str] = []
    if witness.get("artifact_kind") != "M4CanonicalPlacementWitness":
        failures.append("canonical_placement_witness_kind_mismatch")
    if (
        witness.get("version") != M4_SINGLE_OBJECT_PLACEMENT_POLICY_VERSION
        or witness.get("policy_version")
        != M4_SINGLE_OBJECT_PLACEMENT_POLICY_VERSION
    ):
        failures.append("canonical_placement_policy_version_mismatch")
    if witness.get("policy_hash") != M4_SINGLE_OBJECT_PLACEMENT_POLICY_HASH:
        failures.append("canonical_placement_policy_hash_mismatch")
    if witness.get("layout_geometry_subset_hash") != (
        M4_SINGLE_OBJECT_PLACEMENT_POLICY["layout_geometry_subset_hash"]
    ):
        failures.append("canonical_placement_layout_geometry_hash_mismatch")
    if witness.get("status") != "passed":
        failures.append("canonical_placement_not_passed")
    if expected_use is not None and str(witness.get("use")) != str(expected_use):
        failures.append("canonical_placement_use_mismatch")
    if (
        expected_canonical_state_hash is not None
        and witness.get("canonical_state_hash")
        != expected_canonical_state_hash
    ):
        failures.append("canonical_placement_state_hash_mismatch")
    witness_use = str(witness.get("use"))
    touch_placement = bool(
        require_current_touch_policy
        and witness_use in M4_TOUCH_PLACEMENT_POLICY["uses"]
    )
    expected_arm_mode = (
        M4_TOUCH_PLACEMENT_POLICY["arm_mode"]
        if touch_placement
        else M4_SINGLE_OBJECT_PLACEMENT_POLICY["use_arm_mode"].get(
            witness_use
        )
    )
    if witness.get("arm_mode") != expected_arm_mode:
        failures.append("canonical_placement_arm_mode_mismatch")
    if touch_placement:
        if witness.get("touch_placement_policy_version") != (
            M4_TOUCH_PLACEMENT_POLICY_VERSION
        ):
            failures.append("canonical_touch_placement_policy_version_mismatch")
        if witness.get("touch_placement_policy_hash") != (
            M4_TOUCH_PLACEMENT_POLICY_HASH
        ):
            failures.append("canonical_touch_placement_policy_hash_mismatch")
    origin = witness.get("origin_xy_m")
    if not finite_vector(origin, 2):
        failures.append("canonical_placement_origin_invalid")
    payload = {key: value for key, value in witness.items() if key != "witness_hash"}
    if witness.get("witness_hash") != artifact_hash(canonical_data(payload)):
        failures.append("canonical_placement_witness_hash_mismatch")
    if witness.get("arm_mode") == "any_explicit" and witness.get(
        "selected_arm"
    ) not in {"left", "right"}:
        failures.append("canonical_placement_selected_arm_missing")
    if witness.get("arm_mode") == "fixed_left" and witness.get(
        "selected_arm"
    ) != "left":
        failures.append("canonical_placement_fixed_arm_mismatch")
    if witness.get("arm_mode") == "fixed_right" and witness.get(
        "selected_arm"
    ) != "right":
        failures.append("canonical_placement_fixed_arm_mismatch")
    if not witness.get("operation_anchor_contract_hash"):
        failures.append("canonical_placement_anchor_contract_hash_missing")
    if not witness.get("support_hash"):
        failures.append("canonical_placement_support_hash_missing")
    selected_anchors = witness.get("selected_anchors")
    selected_projection: dict[str, int] = {}
    if not isinstance(selected_anchors, list) or not selected_anchors:
        failures.append("canonical_placement_selected_anchors_missing")
    else:
        for row in selected_anchors:
            if (
                not isinstance(row, Mapping)
                or row.get("arm") not in {"left", "right"}
                or not row.get("anchor_id")
                or not isinstance(row.get("point_index"), int)
                or isinstance(row.get("point_index"), bool)
                or not finite_vector(
                    row.get("anchor_position_after_yaw_xyz_m"), 3
                )
            ):
                failures.append("canonical_placement_selected_anchor_invalid")
                continue
            arm = str(row["arm"])
            point_index = int(row["point_index"])
            if arm in selected_projection:
                failures.append("canonical_placement_selected_anchor_duplicate_arm")
            selected_projection[arm] = point_index
    if witness.get("selected_contact_ids_by_arm") != selected_projection:
        failures.append("canonical_placement_selected_contact_projection_mismatch")
    if witness.get("arm_mode") == "both_existential_shared_pose":
        if {str(row.get("arm")) for row in selected_anchors or []} != {
            "left",
            "right",
        }:
            failures.append("canonical_placement_both_arm_witness_missing")
    if witness.get("arm_mode") == "touch_both_arms_same_anchor_shared_pose":
        touch_rows = [
            row for row in selected_anchors or [] if isinstance(row, Mapping)
        ]
        if witness.get("selected_arm") is not None:
            failures.append("canonical_placement_touch_selected_arm_must_be_null")
        if set(selected_projection) != {"left", "right"}:
            failures.append("canonical_placement_touch_both_arm_witness_missing")
        elif len(set(selected_projection.values())) != 1:
            failures.append("canonical_placement_touch_contact_id_mismatch")
        if len({str(row.get("anchor_id") or "") for row in touch_rows}) != 1:
            failures.append("canonical_placement_touch_anchor_id_mismatch")
    if witness.get("arm_mode") == "bimanual_contact_groups":
        if {str(row.get("arm")) for row in selected_anchors or []} != {
            "left",
            "right",
        }:
            failures.append("canonical_placement_bimanual_witness_missing")
    return sorted(set(failures))


def resolve_single_arm_pick_placement(
    arm: str,
    *,
    operation_anchor_contract: Mapping[str, Any] | None = None,
    relative_yaw_deg: float = 0.0,
) -> dict[str, Any]:
    """Resolve the legal actor-origin region for one single-arm Pick.

    The selected Contact follows the same deterministic table-Pick preference
    as the shared M4/M15 acquisition planner.  A disjoint preferred Contact
    region is projected to the closest proven actor-origin safety boundary;
    this preserves the existing M4 safety precedence and makes the resolution
    explicit to M12.
    """

    arm_name = str(arm)
    policies = PICK_OPERATION_PLACEMENT_POLICY["single_arm"]
    if arm_name not in policies:
        raise ValueError(f"unsupported single-arm Pick placement: {arm_name!r}")
    arm_policy = policies[arm_name]
    target = arm_policy["target_contact_xy_m"]
    safe = arm_policy["actor_origin_safety_xy_m"]
    selected = (
        select_table_pick_anchor(operation_anchor_contract)
        if isinstance(operation_anchor_contract, Mapping)
        else None
    )
    canonical_anchor = (
        list(selected["position_xyz_m"])
        if isinstance(selected, Mapping)
        else [0.0, 0.0, 0.0]
    )
    rotated_anchor = _rotated_anchor_position(canonical_anchor, relative_yaw_deg)
    requested = _origin_bounds_for_target(
        rotated_anchor,
        target_x=target["x"],
        target_y=target["y"],
    )
    xlim, x_resolution = _safe_interval_intersection(requested["xlim"], safe["x"])
    ylim, y_resolution = _safe_interval_intersection(requested["ylim"], safe["y"])
    return canonical_data(
        {
            "xlim": xlim,
            "ylim": ylim,
            "policy_version": PICK_OPERATION_PLACEMENT_POLICY_VERSION,
            "policy_hash": PICK_OPERATION_PLACEMENT_POLICY_HASH,
            "relative_yaw_deg": float(relative_yaw_deg),
            "target_contact_x_range_m": list(target["x"]),
            "target_contact_y_range_m": list(target["y"]),
            "actor_safety_x_range_m": list(safe["x"]),
            "actor_safety_y_range_m": list(safe["y"]),
            "requested_origin_bounds": requested,
            "workspace_resolution": {
                "x": x_resolution,
                "y": y_resolution,
            },
            "selected_anchor": dict(selected) if selected is not None else None,
            "selected_anchor_position_after_yaw_xyz_m": rotated_anchor,
            "operation_anchor_contract_hash": (
                operation_anchor_contract.get("contract_hash")
                if isinstance(operation_anchor_contract, Mapping)
                else None
            ),
        }
    )


def evaluate_single_arm_pick_placement(
    arm: str,
    *,
    actor_origin_xy: Sequence[float],
    operation_anchor_contract: Mapping[str, Any],
    relative_yaw_deg: float = 0.0,
    tolerance_m: float = 1e-9,
) -> dict[str, Any]:
    """Evaluate a concrete actor origin against the shared Pick contract."""

    if len(actor_origin_xy) != 2:
        raise ValueError("actor_origin_xy must contain xy")
    origin = [float(actor_origin_xy[0]), float(actor_origin_xy[1])]
    if not all(math.isfinite(value) for value in origin):
        raise ValueError("actor_origin_xy must be finite")
    resolved = resolve_single_arm_pick_placement(
        arm,
        operation_anchor_contract=operation_anchor_contract,
        relative_yaw_deg=relative_yaw_deg,
    )
    tolerance = max(0.0, float(tolerance_m))
    xlim, ylim = resolved["xlim"], resolved["ylim"]
    selected_anchor_exists = resolved["selected_anchor"] is not None
    x_passed = xlim[0] - tolerance <= origin[0] <= xlim[1] + tolerance
    y_passed = ylim[0] - tolerance <= origin[1] <= ylim[1] + tolerance
    passed = bool(selected_anchor_exists and x_passed and y_passed)
    anchor = resolved["selected_anchor_position_after_yaw_xyz_m"]
    contact_xy = [origin[0] + float(anchor[0]), origin[1] + float(anchor[1])]
    target_x = resolved["target_contact_x_range_m"]
    target_y = resolved["target_contact_y_range_m"]
    target_contact_passed = bool(
        target_x[0] - tolerance <= contact_xy[0] <= target_x[1] + tolerance
        and target_y[0] - tolerance <= contact_xy[1] <= target_y[1] + tolerance
    )
    return canonical_data(
        {
            **resolved,
            "actor_origin_xy": origin,
            "selected_contact_world_xy": contact_xy,
            "selected_anchor_exists": selected_anchor_exists,
            "x_passed": x_passed,
            "y_passed": y_passed,
            "target_contact_passed": target_contact_passed,
            "passed": passed,
            "reason": (
                None
                if passed
                else "pick_operation_anchor_missing"
                if not selected_anchor_exists
                else "pick_operation_placement_outside_region"
            ),
        }
    )


def resolve_d4_shared_pick_placement(
    *,
    operation_anchor_contract: Mapping[str, Any],
    relative_yaw_deg: float = 0.0,
) -> dict[str, Any]:
    """Resolve D4 actor-origin regions that survive geometric pruning.

    Every eligible non-upward Contact is considered because M15's shared
    grasp planner may legally choose any authored Contact.  This function only
    narrows M12's search domain; it deliberately does not claim that either
    arm has an IK or Pre-Grasp path.
    """

    target = D4_SHARED_PICK_PLACEMENT_POLICY["target_contact_xy_m"]
    safe = D4_SHARED_PICK_PLACEMENT_POLICY["actor_origin_safety_xy_m"]
    anchors = [
        dict(row)
        for row in operation_anchor_contract.get("anchors", [])
        if row.get("kind") == "grasp_contact"
        and row.get("operation_eligible", True) is True
        and row.get("approach_class") != "upward"
        and len(row.get("position_xyz_m") or []) == 3
    ]

    def anchor_order(row: Mapping[str, Any]) -> tuple[int, str]:
        try:
            point_index = int(row.get("point_index", 10**9))
        except (TypeError, ValueError):
            point_index = 10**9
        return point_index, str(row.get("id") or "")

    anchors.sort(key=anchor_order)
    regions: list[dict[str, Any]] = []
    for anchor_index, anchor in enumerate(anchors):
        try:
            rotated_anchor = _rotated_anchor_position(
                anchor["position_xyz_m"], relative_yaw_deg
            )
        except (TypeError, ValueError):
            continue
        requested = _origin_bounds_for_target(
            rotated_anchor,
            target_x=target["x"],
            target_y=target["y"],
        )
        xlim = _exact_interval_intersection(requested["xlim"], safe["x"])
        ylim = _exact_interval_intersection(requested["ylim"], safe["y"])
        if xlim is None or ylim is None:
            continue
        regions.append(
            {
                "anchor_id": str(
                    anchor.get("id")
                    or f"grasp_contact:{anchor.get('point_index', anchor_index)}"
                ),
                "point_index": anchor.get("point_index"),
                "xlim": xlim,
                "ylim": ylim,
                "requested_origin_bounds": requested,
                "anchor_position_after_yaw_xyz_m": rotated_anchor,
            }
        )
    return canonical_data(
        {
            "policy_version": D4_SHARED_PICK_PLACEMENT_POLICY_VERSION,
            "policy_hash": D4_SHARED_PICK_PLACEMENT_POLICY_HASH,
            "scope": "m12_search_pruning_only",
            "planning_witness_required": True,
            "final_feasibility_proven": False,
            "relative_yaw_deg": float(relative_yaw_deg),
            "actor_safety_x_range_m": list(safe["x"]),
            "actor_safety_y_range_m": list(safe["y"]),
            "target_contact_x_range_m": list(target["x"]),
            "target_contact_y_range_m": list(target["y"]),
            "eligible_anchor_count": len(anchors),
            "compatible_anchor_region_count": len(regions),
            "anchor_regions": regions,
            "operation_anchor_contract_hash": operation_anchor_contract.get(
                "contract_hash"
            ),
        }
    )


def evaluate_d4_shared_pick_placement(
    *,
    actor_origin_xy: Sequence[float],
    operation_anchor_contract: Mapping[str, Any],
    relative_yaw_deg: float = 0.0,
    tolerance_m: float = 1e-9,
) -> dict[str, Any]:
    """Evaluate one D4 pose as a geometric search-pruning candidate."""

    if len(actor_origin_xy) != 2:
        raise ValueError("actor_origin_xy must contain xy")
    origin = [float(actor_origin_xy[0]), float(actor_origin_xy[1])]
    if not all(math.isfinite(value) for value in origin):
        raise ValueError("actor_origin_xy must be finite")
    resolved = resolve_d4_shared_pick_placement(
        operation_anchor_contract=operation_anchor_contract,
        relative_yaw_deg=relative_yaw_deg,
    )
    tolerance = max(0.0, float(tolerance_m))
    matches: list[dict[str, Any]] = []
    for region in resolved["anchor_regions"]:
        xlim, ylim = region["xlim"], region["ylim"]
        if not (
            xlim[0] - tolerance <= origin[0] <= xlim[1] + tolerance
            and ylim[0] - tolerance <= origin[1] <= ylim[1] + tolerance
        ):
            continue
        anchor = region["anchor_position_after_yaw_xyz_m"]
        matches.append(
            {
                "anchor_id": region["anchor_id"],
                "point_index": region.get("point_index"),
                "selected_contact_world_xy": [
                    origin[0] + float(anchor[0]),
                    origin[1] + float(anchor[1]),
                ],
            }
        )
    passed = bool(matches)
    return canonical_data(
        {
            **resolved,
            "actor_origin_xy": origin,
            "matching_anchor_count": len(matches),
            "matching_anchors": matches,
            "passed": passed,
            "reason": (
                None
                if passed
                else "pick_operation_anchor_missing"
                if resolved["eligible_anchor_count"] == 0
                else "d4_shared_pick_anchor_geometry_incompatible"
                if resolved["compatible_anchor_region_count"] == 0
                else "d4_shared_pick_placement_outside_region"
            ),
        }
    )


__all__ = [
    "D4_SHARED_PICK_PLACEMENT_POLICY",
    "D4_SHARED_PICK_PLACEMENT_POLICY_HASH",
    "D4_SHARED_PICK_PLACEMENT_POLICY_VERSION",
    "M4_SINGLE_OBJECT_PLACEMENT_POLICY",
    "M4_SINGLE_OBJECT_PLACEMENT_POLICY_HASH",
    "M4_SINGLE_OBJECT_PLACEMENT_POLICY_VERSION",
    "M4_TOUCH_PLACEMENT_POLICY",
    "M4_TOUCH_PLACEMENT_POLICY_HASH",
    "M4_TOUCH_PLACEMENT_POLICY_VERSION",
    "PICK_OPERATION_PLACEMENT_POLICY",
    "PICK_OPERATION_PLACEMENT_POLICY_HASH",
    "PICK_OPERATION_PLACEMENT_POLICY_VERSION",
    "evaluate_d4_shared_pick_placement",
    "evaluate_single_arm_pick_placement",
    "m4_canonical_placement_witness_errors",
    "m4_runtime_refinement_origins",
    "resolve_d4_shared_pick_placement",
    "resolve_m4_single_object_placement",
    "resolve_single_arm_pick_placement",
]
