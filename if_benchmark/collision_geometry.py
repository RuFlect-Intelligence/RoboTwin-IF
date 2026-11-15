"""Live collision-geometry measurements shared by benchmark runtimes."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


COLLISION_AABB_TIGHT_SOURCE = "sapien_collision_compute_global_aabb_tight"
COLLISION_LOCAL_POINT_CLOUD_SOURCE = (
    "sapien_collision_shapes_link_local_vertices_v1"
)
COLLISION_LOCAL_MESH_SOURCE = (
    "sapien_collision_shapes_link_local_triangle_meshes_v1"
)


def frozen_support_plane_z(
    initial_pose: Mapping[str, Any], *, z_bias: float = 0.0
) -> float:
    """Read the immutable support plane from a canonical Initial Pose."""

    support = initial_pose.get("support") or {}
    if not isinstance(support, Mapping) or "plane_z" not in support:
        raise RuntimeError("initial_pose_support_plane_missing")
    value = float(support["plane_z"]) + float(z_bias)
    if not math.isfinite(value):
        raise RuntimeError("initial_pose_support_plane_invalid")
    return value


def collision_aabb_tight(actor: Any) -> dict[str, list[float]]:
    """Return the tight world AABB of an actor's current collision geometry.

    RoboTwin actor wrappers expose the physical SAPIEN object as ``actor``.
    Rigid actors contain collision components; articulated actors contain
    links.  Each collision-bearing component/link owns the authoritative live
    ``compute_global_aabb_tight`` measurement.  Metadata/config AABBs are
    intentionally not accepted as a fallback because they are conservative
    planning geometry rather than a success-predicate measurement.
    """

    physical = getattr(actor, "actor", actor)
    if hasattr(physical, "get_links"):
        candidates = list(physical.get_links())
    elif hasattr(physical, "get_components"):
        candidates = list(physical.get_components())
    else:
        candidates = [physical]

    arrays: list[np.ndarray] = []
    for component in candidates:
        get_collision_shapes = getattr(component, "get_collision_shapes", None)
        compute = getattr(component, "compute_global_aabb_tight", None)
        if not callable(get_collision_shapes) or not callable(compute):
            continue
        if not get_collision_shapes():
            continue
        bounds = np.asarray(compute(), dtype=float)
        if bounds.shape != (2, 3) or not np.all(np.isfinite(bounds)):
            raise RuntimeError("collision_aabb_tight_invalid")
        arrays.append(bounds)

    if not arrays:
        raise RuntimeError("actor_has_no_collision_geometry")

    minimum = np.min(np.stack([row[0] for row in arrays]), axis=0)
    maximum = np.max(np.stack([row[1] for row in arrays]), axis=0)
    return {
        "min": [float(value) for value in minimum],
        "max": [float(value) for value in maximum],
        "center": [float(value) for value in (minimum + maximum) / 2.0],
    }


def collision_support_clearance(actor: Any, support_plane_z_m: float) -> float:
    """Measure live collision-bottom clearance from one frozen support plane."""

    return float(
        collision_aabb_tight(actor)["min"][2] - float(support_plane_z_m)
    )


def _shape_local_vertices(shape: Any) -> np.ndarray:
    """Return a deterministic finite point cloud in one shape-local frame."""

    if hasattr(shape, "vertices"):
        points = np.asarray(shape.vertices, dtype=float)
        scale = np.asarray(getattr(shape, "scale", [1.0, 1.0, 1.0]), dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or scale.shape != (3,):
            raise RuntimeError("collision_shape_vertices_invalid")
        points = points * scale[None, :]
    elif hasattr(shape, "half_size"):
        half = np.asarray(shape.half_size, dtype=float)
        points = np.asarray(
            [
                [sx * half[0], sy * half[1], sz * half[2]]
                for sx in (-1.0, 1.0)
                for sy in (-1.0, 1.0)
                for sz in (-1.0, 1.0)
            ],
            dtype=float,
        )
    elif hasattr(shape, "radius") and hasattr(shape, "half_length"):
        radius = float(shape.radius)
        half_length = float(shape.half_length)
        points = np.asarray(
            [
                [sx * half_length, radius * np.cos(theta), radius * np.sin(theta)]
                for sx in (-1.0, 1.0)
                for theta in np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
            ],
            dtype=float,
        )
    elif hasattr(shape, "radius"):
        radius = float(shape.radius)
        points = np.asarray(
            [
                [radius, 0.0, 0.0],
                [-radius, 0.0, 0.0],
                [0.0, radius, 0.0],
                [0.0, -radius, 0.0],
                [0.0, 0.0, radius],
                [0.0, 0.0, -radius],
            ],
            dtype=float,
        )
    else:
        raise RuntimeError("collision_shape_geometry_unsupported")
    if not len(points) or not np.isfinite(points).all():
        raise RuntimeError("collision_shape_vertices_invalid")
    if len(points) > 4096:
        indices = np.linspace(0, len(points) - 1, 4096, dtype=int)
        points = points[indices]
    return points


def _box_local_mesh(half_size: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(
        [
            [-half_size[0], -half_size[1], -half_size[2]],
            [half_size[0], -half_size[1], -half_size[2]],
            [half_size[0], half_size[1], -half_size[2]],
            [-half_size[0], half_size[1], -half_size[2]],
            [-half_size[0], -half_size[1], half_size[2]],
            [half_size[0], -half_size[1], half_size[2]],
            [half_size[0], half_size[1], half_size[2]],
            [-half_size[0], half_size[1], half_size[2]],
        ],
        dtype=float,
    )
    triangles = np.asarray(
        [
            [0, 2, 1], [0, 3, 2],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [3, 7, 6], [3, 6, 2],
            [0, 4, 7], [0, 7, 3],
            [1, 2, 6], [1, 6, 5],
        ],
        dtype=int,
    )
    return vertices, triangles


def _ring_local_mesh(
    radius: float, half_length: float, *, segments: int = 16
) -> tuple[np.ndarray, np.ndarray]:
    """Return a conservative triangulated X-axis cylinder/capsule hull."""

    vertices = np.asarray(
        [
            [x, radius * np.cos(theta), radius * np.sin(theta)]
            for x in (-half_length, half_length)
            for theta in np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
        ]
        + [[-half_length, 0.0, 0.0], [half_length, 0.0, 0.0]],
        dtype=float,
    )
    left_center = 2 * segments
    right_center = left_center + 1
    triangles: list[list[int]] = []
    for index in range(segments):
        following = (index + 1) % segments
        left = index
        left_next = following
        right = segments + index
        right_next = segments + following
        triangles.extend(
            [
                [left, right, right_next],
                [left, right_next, left_next],
                [left_center, left_next, left],
                [right_center, right, right_next],
            ]
        )
    return vertices, np.asarray(triangles, dtype=int)


def _sphere_local_mesh(radius: float) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(
        [
            [radius, 0.0, 0.0],
            [-radius, 0.0, 0.0],
            [0.0, radius, 0.0],
            [0.0, -radius, 0.0],
            [0.0, 0.0, radius],
            [0.0, 0.0, -radius],
        ],
        dtype=float,
    )
    triangles = np.asarray(
        [
            [0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
            [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5],
        ],
        dtype=int,
    )
    return vertices, triangles


def _shape_local_mesh(shape: Any) -> tuple[np.ndarray, np.ndarray]:
    """Read one collision shape as a deterministic triangle mesh."""

    if hasattr(shape, "vertices") and hasattr(shape, "triangles"):
        vertices = np.asarray(shape.vertices, dtype=float)
        scale = np.asarray(getattr(shape, "scale", [1.0, 1.0, 1.0]), dtype=float)
        triangles = np.asarray(shape.triangles, dtype=int)
        if vertices.ndim != 2 or vertices.shape[1:] != (3,) or scale.shape != (3,):
            raise RuntimeError("collision_shape_vertices_invalid")
        vertices = vertices * scale[None, :]
        if triangles.ndim == 1:
            if len(triangles) % 3:
                raise RuntimeError("collision_shape_triangles_invalid")
            triangles = triangles.reshape(-1, 3)
    elif hasattr(shape, "half_size"):
        half_size = np.asarray(shape.half_size, dtype=float)
        if half_size.shape != (3,):
            raise RuntimeError("collision_shape_vertices_invalid")
        vertices, triangles = _box_local_mesh(half_size)
    elif hasattr(shape, "radius") and hasattr(shape, "half_length"):
        vertices, triangles = _ring_local_mesh(
            float(shape.radius), float(shape.half_length)
        )
    elif hasattr(shape, "radius"):
        vertices, triangles = _sphere_local_mesh(float(shape.radius))
    else:
        raise RuntimeError("collision_shape_geometry_unsupported")
    if (
        not len(vertices)
        or not len(triangles)
        or triangles.ndim != 2
        or triangles.shape[1:] != (3,)
        or not np.isfinite(vertices).all()
        or np.min(triangles) < 0
        or np.max(triangles) >= len(vertices)
    ):
        raise RuntimeError("collision_shape_triangles_invalid")
    if len(triangles) > 8192:
        indices = np.linspace(0, len(triangles) - 1, 8192, dtype=int)
        triangles = triangles[indices]
    return vertices, triangles


def collision_shape_local_meshes(component: Any) -> list[dict[str, np.ndarray]]:
    """Extract collision triangle meshes in one component/link local frame.

    This surface-preserving representation is used only by articulated
    grasp-follow orientation resolution.  Existing rigid Contact planning
    continues to use its authored frames and does not call this function.
    """

    get_shapes = getattr(component, "get_collision_shapes", None)
    shapes: Sequence[Any] = (
        list(get_shapes())
        if callable(get_shapes)
        else list(getattr(component, "collision_shapes", ()) or ())
    )
    meshes: list[dict[str, np.ndarray]] = []
    for shape in shapes:
        try:
            vertices, triangles = _shape_local_mesh(shape)
        except RuntimeError as exc:
            if str(exc) == "collision_shape_geometry_unsupported":
                continue
            raise
        pose = getattr(shape, "local_pose", None)
        matrix = (
            np.eye(4, dtype=float)
            if pose is None
            else np.asarray(pose.to_transformation_matrix(), dtype=float)
        )
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise RuntimeError("collision_shape_local_pose_invalid")
        homogeneous = np.concatenate(
            [vertices, np.ones((len(vertices), 1), dtype=float)], axis=1
        )
        meshes.append(
            {
                "vertices": (homogeneous @ matrix.T)[:, :3],
                "triangles": triangles.copy(),
            }
        )
    if not meshes:
        raise RuntimeError("collision_component_has_no_supported_geometry")
    return meshes


def collision_shape_local_point_clouds(component: Any) -> list[np.ndarray]:
    """Extract authoritative collision points in a component/link local frame.

    Mesh scale and each shape's local pose are applied without mutating the
    simulator. Unsupported non-solid shapes are skipped, while an entirely
    unreadable collision component fails closed.
    """

    get_shapes = getattr(component, "get_collision_shapes", None)
    shapes: Sequence[Any] = (
        list(get_shapes())
        if callable(get_shapes)
        else list(getattr(component, "collision_shapes", ()) or ())
    )
    clouds: list[np.ndarray] = []
    for shape in shapes:
        try:
            points = _shape_local_vertices(shape)
        except RuntimeError as exc:
            if str(exc) == "collision_shape_geometry_unsupported":
                continue
            raise
        pose = getattr(shape, "local_pose", None)
        matrix = (
            np.eye(4, dtype=float)
            if pose is None
            else np.asarray(pose.to_transformation_matrix(), dtype=float)
        )
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise RuntimeError("collision_shape_local_pose_invalid")
        homogeneous = np.concatenate(
            [points, np.ones((len(points), 1), dtype=float)], axis=1
        )
        clouds.append((homogeneous @ matrix.T)[:, :3])
    if not clouds:
        raise RuntimeError("collision_component_has_no_supported_geometry")
    return clouds


__all__ = [
    "COLLISION_AABB_TIGHT_SOURCE",
    "COLLISION_LOCAL_MESH_SOURCE",
    "COLLISION_LOCAL_POINT_CLOUD_SOURCE",
    "collision_aabb_tight",
    "collision_shape_local_meshes",
    "collision_shape_local_point_clouds",
    "collision_support_clearance",
    "frozen_support_plane_z",
]
