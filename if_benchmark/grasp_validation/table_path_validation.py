"""M15 descent table checks without the held sphere cover's shape overhang."""
from contextlib import contextmanager
from copy import deepcopy
import sys

import numpy as np
import transforms3d as t3d

from if_benchmark.schema import artifact_hash


@contextmanager
def table_candidate_world(planner):
    """Omit table only while proposing a path that still requires exact checks."""
    if (getattr(planner, "_collision_geometry_mode", None) != planner.CARRY_COLLISION_MODE
            or not getattr(planner, "_carry_exact_table_geometry", None)
            or getattr(planner, "_omit_table_candidate", False)):
        raise RuntimeError("exact table candidate requires a bound M15 held geometry")
    original_hash = planner._world_hash
    obstacles = deepcopy(planner._last_world_obstacles)
    try:
        planner._omit_table_candidate = True
        planner.update_world_obstacles(obstacles)
        yield {"candidate_world_hash": planner._world_hash, "validated_world_hash": original_hash}
    finally:
        pending = sys.exc_info()[1]
        planner._omit_table_candidate = False
        try:
            planner.update_world_obstacles(obstacles)
            if planner._world_hash != original_hash:
                planner._world_hash = None
                raise RuntimeError("exact table candidate failed to restore its collision world")
        except BaseException as cleanup:
            planner._world_hash = None
            if pending is not None:
                if hasattr(pending, "add_note"):
                    pending.add_note(f"table world restoration also failed: {cleanup!r}")
                raise pending from cleanup
            raise


def validate_table_path(*, robot_spheres_world, ee_world_matrices,
                        held_vertices_parent, padding_m, table_obstacles,
                        trajectory_sha256, geometry_sha256,
                        route_lift_support=False):
    """Check every original waypoint; neither the path nor physics is changed.

    Robot spheres retain their native radii and all five real table boxes.
    Every complete held piece, with its original box envelope already baked
    into the vertices, must remain above the highest point of the entire
    table plus the unchanged planner padding. This infinite support plane is
    more conservative than the finite table boxes.
    """
    spheres = np.asarray(robot_spheres_world, dtype=float)
    matrices = np.asarray(ee_world_matrices, dtype=float)
    vertices = np.asarray(held_vertices_parent, dtype=float)
    padding = float(padding_m)
    if (spheres.ndim != 3 or spheres.shape[-1] != 4 or not len(spheres)
            or matrices.shape != (len(spheres), 4, 4)
            or vertices.ndim != 2 or vertices.shape[1:] != (3,) or not len(vertices)
            or not np.isfinite(spheres).all() or not np.isfinite(matrices).all()
            or not np.isfinite(vertices).all() or not np.isfinite(padding) or padding < 0
            or not table_obstacles):
        raise ValueError("exact table validation requires complete finite path geometry")
    rotations = matrices[:, :3, :3]
    if (not np.all(np.any(spheres[:, :, 3] > 0, axis=1))
            or not np.allclose(matrices[:, 3, :], [0, 0, 0, 1], atol=1e-7, rtol=0)
            or not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-6, rtol=0)
            or not np.allclose(np.linalg.det(rotations), 1, atol=1e-6, rtol=0)):
        raise ValueError("exact table validation requires active robot spheres and rigid FK poses")
    robot_clearance = np.full(len(spheres), np.inf)
    top = -np.inf
    for box in table_obstacles:
        dims, pose = np.asarray(box["dims"], dtype=float), np.asarray(box["pose"], dtype=float)
        if (dims.shape != (3,) or pose.shape != (7,) or not np.isfinite(dims).all()
                or not np.isfinite(pose).all() or np.any(dims <= 0) or np.linalg.norm(pose[3:]) < 1e-12):
            raise ValueError("invalid physical table collision box")
        rotation = t3d.quaternions.quat2mat(pose[3:])
        top = max(top, float(pose[2] + np.abs(rotation[2]) @ (dims / 2)))
        local = (spheres[:, :, :3] - pose[:3]) @ rotation
        delta = np.abs(local) - dims / 2
        distances = (np.linalg.norm(np.maximum(delta, 0), axis=-1)
                     + np.minimum(delta.max(axis=-1), 0) - spheres[:, :, 3])
        distances = np.where(spheres[:, :, 3] > 0, distances, np.inf)
        robot_clearance = np.minimum(robot_clearance, distances.min(axis=1))
    held_clearance = []
    for matrix in matrices:
        # A linear height functional attains its minimum at a convex vertex.
        held_clearance.append(float((vertices @ matrix[2, :3]).min()
                                    + matrix[2, 3] - top - padding))
    held_clearance = np.asarray(held_clearance)
    # Reserve a micrometre for float32 FK/pose roundoff rather than accepting
    # a rounded zero at the collision boundary.
    # A planning-only route Lift begins with the frozen subject on its
    # support. Its path may not worsen that initial contact, and must finish
    # clear of the full envelope/padding. Physical M15 Pick still uses its
    # unchanged shared acquisition and strict 40 mm certification gates.
    floor = min(float(held_clearance[0]), 0.) if route_lift_support else 1e-6
    held_failed = held_clearance < (floor - 1e-9 if route_lift_support else floor)
    if route_lift_support and held_clearance[-1] < 1e-6:
        held_failed[-1] = True
    failed = np.flatnonzero(held_failed | (robot_clearance < 1e-6))
    payload = {"validator": "m15_complete_held_pieces_and_native_robot_table_path_v1",
               "collision_free": len(failed) == 0, "checked_waypoint_count": len(spheres),
               "trajectory_sha256": trajectory_sha256, "geometry_sha256": geometry_sha256,
               "table_geometry_hash": artifact_hash(table_obstacles), "table_top_plane_z": top,
               "padding_m": padding, "held_expanded_vertex_count": len(vertices),
               "route_lift_support": bool(route_lift_support),
               "initial_held_table_clearance_m": float(held_clearance[0]),
               "minimum_robot_table_clearance_m": float(robot_clearance.min()),
               "minimum_held_table_clearance_m": float(held_clearance.min()),
               "failed_waypoint_indices": failed.tolist()}
    return {**payload, "validation_hash": artifact_hash(payload)}
