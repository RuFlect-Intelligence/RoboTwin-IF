"""Operational helpers for canonical M5 skill execution."""

from __future__ import annotations

from typing import Any


GRIPPER_HOLD_STEPS_PER_ITERATION = 300


def hold_grippers_and_step(
    robot: Any,
    scene: Any,
    controller_iterations: int,
) -> None:
    """Execute Base_Task.delay's unchanged-target physics sequence cheaply.

    Base_Task builds a 200-sample gripper trajectory and pads it with 100
    samples.  When the requested target is the current position, every sample
    is identical and ``per_step`` is zero.  This helper preserves the exact
    left-write, right-write, scene-step order and the exact 300-step count.
    """

    left_position = robot.get_left_gripper_val()
    right_position = robot.get_right_gripper_val()
    for _ in range(
        max(0, int(controller_iterations))
        * GRIPPER_HOLD_STEPS_PER_ITERATION
    ):
        robot.set_gripper(left_position, "left", 0.0)
        robot.set_gripper(right_position, "right", 0.0)
        scene.step()
