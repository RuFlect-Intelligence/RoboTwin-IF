"""Versioned, simulator-independent D2b production policy."""

POLICY = {
    "version": "d2b_operation_space_joint_branches_v1",
    "minimum_counterfactual_tasks": 2,
    "robot_padding_m": .002,
    "held_world_box_padding_m": .010,
    "layout_clearance_m": .005,
    "terminal_goal_margin_m": .010,
    "maximum_terminal_plans": 100,
    "maximum_goal_domains": 32,
    "maximum_layouts_per_goal_domain": 2,
    "maximum_cpu_layout_evaluations_per_goal_domain": 16,
    "maximum_arm_assignments": 128,
    "planar_yaws_deg": [0., 90., 180., 270., 15., 30., 45., 60., 75., -15., -30., -45., -60., -75.],
    "new_geometry_changes_publication_identity": True,
}
