"""Pure, bounded terminal proposal domain; shared by Layout and route planning."""
from __future__ import annotations

from .contract import (TerminalCandidateRejected, terminal_candidate_from_sources,
                       terminal_search_offsets, terminal_search_yaws)
from if_benchmark.terminal_placement import load_reviewed_receiver_profiles
from if_benchmark.manipulation_policy import POLICY


def semantic_margin(terminal):
    """Smallest distance to a measured relation boundary, in metres."""
    values = []
    def visit(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key.endswith("margin_m") and isinstance(item, (int, float)):
                    values.append(float(item))
                else:
                    visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
    visit(terminal.get("relation_geometry") or {})
    return min(values) if values else float("inf")


def ranked_terminals(rows):
    # Prefer meaningful semantic slack, then shorter/smaller adjustments. Slack
    # is a ranking objective, never a relaxation of the shared SuccessContract.
    return sorted(rows, key=lambda r: (
        -min(POLICY["terminal_goal_margin_m"], semantic_margin(r)),
        sum(float(x) ** 2 for x in r["offset_xy_m"]),
        min(abs(float(r["yaw_delta_deg"])), abs(360 - float(r["yaw_delta_deg"])))))


def operation_filtered_terminals(unit, scene, step, rows, *, arm, contact_id,
                                 templates=None, boxes=None):
    """Local placement gate for the exact grasp branch, before planner calls.

    Old immutable diagnostics without new M11 resources retain their explicit
    terminal domain. New production M11 artifacts always carry these resources.
    """
    import math
    from transforms3d.quaternions import quat2mat
    from if_benchmark.layout.manipulation import OperationTemplates, scene_boxes, placement_clear
    templates = templates or OperationTemplates(unit["units"])
    subject = step["subject"]
    if subject not in templates.resources:
        return ranked_terminals(rows)
    boxes = boxes if boxes is not None else scene_boxes(scene, unit["units"])
    initial = quat2mat(templates.resources[subject]["geometry"]["quat_wxyz"])
    receivers = [step[k] for k in ("target", "target2") if step.get(k)]
    accepted = []
    for terminal in ranked_terminals(rows):
        rotation = quat2mat(terminal["actor_pose"][3:]) @ initial.T
        yaw = math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))
        template = templates.get(subject, arm, contact_id, step["relation"], yaw_deg=round(yaw, 8))
        if placement_clear(template, boxes, subject, terminal, step["relation"], receivers):
            accepted.append(terminal)
    return accepted


def terminal_candidates(unit, scene, step, *, cavity_world_matrix=None):
    profiles = load_reviewed_receiver_profiles()
    if cavity_world_matrix is None:
        # M11 freezes all receiver link transforms in world-axis coordinates
        # relative to its root. XY layout changes need only a translation.
        from .contract import _reservation_source
        from if_benchmark.terminal_placement import terminal_recipe_binding, validate_se3
        import numpy as np
        _, region, _ = _reservation_source(unit, scene, step)
        binding = terminal_recipe_binding(region, step["relation"], receiver_profiles=profiles)
        profile = binding.get("receiver_state_profile")
        if profile is not None:
            receiver = step["target"]
            frozen = next((u["params"].get("operation_obstacles", {}).get(receiver)
                           for u in unit["units"] if receiver in u["params"].get("operation_obstacles", {})), None)
            if frozen is None:
                raise ValueError("M11 frozen receiver cavity geometry missing")
            cavity = profile["cavity"]
            matrix = np.asarray(frozen["link_matrices"][cavity["frame_link"]], float).copy()
            matrix[:3, 3] += scene["entities"][receiver]["pose"]["position"]
            if cavity.get("frame_transform") is not None:
                matrix = matrix @ np.asarray(cavity["frame_transform"], float)
            cavity_world_matrix = validate_se3(matrix, "frozen receiver cavity")
    for x, y in terminal_search_offsets(step["relation"]):
        for yaw in terminal_search_yaws(step["relation"]):
            try:
                yield terminal_candidate_from_sources(
                    unit, scene, step, offset_xy_m=(x, y), yaw_delta_deg=yaw,
                    cavity_world_matrix=cavity_world_matrix, receiver_profiles=profiles)
            except TerminalCandidateRejected:
                continue
