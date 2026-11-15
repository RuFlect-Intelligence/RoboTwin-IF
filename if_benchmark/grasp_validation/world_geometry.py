"""Complete live scene collision pieces covered by bounded float32 AABBs.

The scene representation changes, never the robot collision radii or margins.
Each cooked shape remains whole. Budget reduction may join whole pieces only
within the same source entity, preserving phase-specific receiver exclusions.
"""
from __future__ import annotations

from itertools import product
from copy import deepcopy
import hashlib
import heapq
from typing import Any, Mapping, Sequence

import numpy as np

from if_benchmark.schema import artifact_hash
from .convex_cover import _live_pose_matrix

WORLD_GEOMETRY_VERSION = "m15_complete_live_world_pieces_v1"
WORLD_AABB_VERSION = "m15_complete_piece_world_aabb_float32_v1"
NATIVE_CUBOID_VERSION = "curobo_cached_inverse_pose_complete_cuboid_float32_v1"


def _points(value):
    points = np.asarray(value, dtype=float)
    if (points.ndim != 2 or points.shape[1:] != (3,) or not len(points)
            or not np.isfinite(points).all()):
        raise ValueError("world collision piece requires all finite vertices")
    return points


def _array_hash(value, dtype):
    values = np.asarray(value, dtype=dtype)
    return hashlib.sha256(np.asarray(values.shape, dtype="<u8").tobytes()
                          + values.tobytes(order="C")).hexdigest()


def world_geometry_from_pieces(*, name: str, pieces: Sequence, sources: Sequence) -> dict:
    """Seal complete world vertices plus their live shape provenance.

    Pure-data boundary for replay/verification of an already captured scene;
    production callers obtain this data through ``live_world_geometry``.
    """
    if not isinstance(name, str) or not name or name.strip() != name:
        raise ValueError("world geometry requires a source entity name")
    parts = [_points(v) for v in pieces]
    if not parts or len(parts) != len(sources) or any(not isinstance(v, Mapping) for v in sources):
        raise ValueError("world geometry needs one complete source record per piece")
    payload = {"version": WORLD_GEOMETRY_VERSION, "frame": "world", "source_entity": name,
               "pieces": [part.tolist() for part in parts],
               "sources": deepcopy([dict(source) for source in sources])}
    sealed = {**payload, "geometry_hash": artifact_hash(payload)}
    _read_geometries([sealed])
    return sealed


def live_world_geometry(actor: Any, *, name: str) -> dict:
    """Read every live cooked shape with its actual scale and frame chain.

    No visual mesh, metadata bound, downsampling, simulator step, or pose
    mutation is used. Analytic primitives receive containing boxes; a mesh's
    complete vertex set contains all of its triangles and convex interior.
    """
    from sapien.physx import (PhysxCollisionShapeBox, PhysxCollisionShapeCapsule,
                             PhysxCollisionShapeConvexMesh, PhysxCollisionShapeCylinder,
                             PhysxCollisionShapeSphere, PhysxCollisionShapeTriangleMesh)
    physical = getattr(actor, "actor", actor)
    components = (list(physical.get_links()) if hasattr(physical, "get_links") else
                  list(physical.get_components()) if hasattr(physical, "get_components") else [physical])
    signs = np.asarray(list(product((-1., 1.), repeat=3)))
    pieces, sources = [], []
    for component_index, component in enumerate(components):
        getter = getattr(component, "get_collision_shapes", None)
        if not callable(getter):
            continue
        shapes = list(getter())
        if not shapes:
            continue
        component_world = _live_pose_matrix(component.entity.pose)
        for shape_index, shape in enumerate(shapes):
            local = _live_pose_matrix(shape.local_pose)
            source = {"component_index": component_index, "shape_index": shape_index,
                      "shape_type": type(shape).__name__,
                      "component_world_matrix": component_world.tolist(),
                      "shape_local_matrix": local.tolist()}
            if isinstance(shape, (PhysxCollisionShapeConvexMesh, PhysxCollisionShapeTriangleMesh)):
                raw = _points(shape.vertices)
                scale = np.asarray(shape.scale, dtype=float)
                triangles = np.asarray(shape.triangles)
                if (scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0)
                        or triangles.ndim != 2 or triangles.shape[1:] != (3,) or not len(triangles)
                        or not np.issubdtype(triangles.dtype, np.integer)
                        or triangles.min() < 0 or triangles.max() >= len(raw)):
                    raise ValueError("invalid live world collision mesh scale/triangles")
                points = raw * scale
                source.update(scale=scale.tolist(), raw_vertex_count=len(raw), triangle_count=len(triangles),
                              raw_vertices_hash=_array_hash(raw, "<f8"),
                              triangles_hash=_array_hash(triangles, "<i8"), representation="complete_cooked_mesh")
            else:
                if isinstance(shape, PhysxCollisionShapeBox):
                    half = np.asarray(shape.half_size, dtype=float)
                elif isinstance(shape, PhysxCollisionShapeCapsule):
                    half = np.asarray([shape.half_length + shape.radius, shape.radius, shape.radius])
                elif isinstance(shape, PhysxCollisionShapeCylinder):
                    half = np.asarray([shape.half_length, shape.radius, shape.radius])
                elif isinstance(shape, PhysxCollisionShapeSphere):
                    half = np.full(3, float(shape.radius))
                else:
                    raise TypeError(f"unsupported world collision shape: {type(shape).__name__}")
                if half.shape != (3,) or not np.isfinite(half).all() or np.any(half <= 0):
                    raise ValueError("invalid live world collision primitive")
                points = signs * half
                source.update(containing_box_half_size=half.tolist(), representation="complete_primitive_containing_box")
            transform = component_world @ local
            piece = _points(points @ transform[:3, :3].T + transform[:3, 3])
            pieces.append(piece)
            sources.append({**source, "world_vertex_count": len(piece)})
    return world_geometry_from_pieces(name=name, pieces=pieces, sources=sources)


def _read_geometries(geometries):
    result = {}
    for geometry in geometries:
        if (not isinstance(geometry, Mapping) or geometry.get("version") != WORLD_GEOMETRY_VERSION
                or geometry.get("frame") != "world"
                or artifact_hash(geometry, omit_keys=("geometry_hash",)) != geometry.get("geometry_hash")):
            raise ValueError("world collision geometry version/frame/hash mismatch")
        name = geometry.get("source_entity")
        if not isinstance(name, str) or not name or name.strip() != name or name in result:
            raise ValueError("duplicate or invalid world collision source entity")
        parts = [_points(part) for part in geometry.get("pieces", [])]
        sources = geometry.get("sources", [])
        if not parts or len(sources) != len(parts) or any(not isinstance(s, Mapping) for s in sources):
            raise ValueError("incomplete world collision piece provenance")
        source_ids = set()
        for part, source in zip(parts, sources):
            ids = (source.get("component_index"), source.get("shape_index"))
            if (any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in ids)
                    or ids in source_ids or source.get("world_vertex_count") != len(part)
                    or not isinstance(source.get("shape_type"), str) or not source["shape_type"]):
                raise ValueError("invalid world collision piece source identity")
            source_ids.add(ids)
            for key in ("component_world_matrix", "shape_local_matrix"):
                matrix = np.asarray(source.get(key), dtype=float)
                if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
                        or not np.array_equal(matrix[3], [0., 0., 0., 1.])):
                    raise ValueError("missing or invalid world collision piece source transform")
            kind = source.get("representation")
            if kind == "complete_cooked_mesh":
                scale = np.asarray(source.get("scale"), dtype=float)
                if (scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0)
                        or source.get("raw_vertex_count") != len(part)
                        or not isinstance(source.get("triangle_count"), int) or source["triangle_count"] <= 0
                        or any(not isinstance(source.get(key), str) or len(source[key]) != 64
                               for key in ("raw_vertices_hash", "triangles_hash"))):
                    raise ValueError("incomplete world cooked mesh provenance")
            elif kind == "complete_primitive_containing_box":
                half = np.asarray(source.get("containing_box_half_size"), dtype=float)
                if half.shape != (3,) or not np.isfinite(half).all() or np.any(half <= 0) or len(part) != 8:
                    raise ValueError("invalid complete primitive provenance")
            else:
                raise ValueError("unknown world collision piece representation")
        result[name] = (geometry, parts)
    return result


def _budget(value):
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 64:
        raise ValueError("world cuboid budget must be an integer in [0,64]")
    return value


def _groups(data, maximum):
    groups = {(name, (i,)): (part.min(axis=0), part.max(axis=0))
              for name, (_, parts) in data.items() for i, part in enumerate(parts)}
    heap = []
    def offer(left, right):
        left, right = sorted((left, right))
        alo, ahi = groups[left]; blo, bhi = groups[right]
        delta = (np.prod(np.maximum(ahi, bhi) - np.minimum(alo, blo))
                 - np.prod(ahi - alo) - np.prod(bhi - blo))
        if not np.isfinite(delta):
            raise ValueError("world collision bounds exceed finite budget arithmetic")
        heapq.heappush(heap, (float(delta), left, right))
    if len(groups) > maximum:
        keys = sorted(groups)
        for i, left in enumerate(keys):
            for right in keys[i + 1:]:
                if left[0] == right[0]:
                    offer(left, right)
    while len(groups) > maximum:
        if not heap:
            raise ValueError("world cuboid budget cannot preserve every source entity")
        _, left, right = heapq.heappop(heap)
        if left not in groups or right not in groups:
            continue
        alo, ahi = groups.pop(left); blo, bhi = groups.pop(right)
        merged = (left[0], tuple(sorted(left[1] + right[1])))
        groups[merged] = (np.minimum(alo, blo), np.maximum(ahi, bhi))
        for other in sorted(groups):
            if other[0] == merged[0] and other != merged:
                offer(merged, other)
    return sorted(groups)


def _cover(points):
    # Recompute the half sizes AFTER quantizing the center. Rounding just the
    # original min/max box dimensions can silently shave its outside surface.
    lower, upper = points.min(axis=0), points.max(axis=0)
    center = np.asarray(lower * .5 + upper * .5, dtype=np.float32)
    if not np.isfinite(center).all():
        raise ValueError("world cuboid center is not representable in float32")
    required = np.abs(points - center.astype(float)).max(axis=0)
    guard = 16 * np.finfo(float).eps * max(1., float(np.abs(points).max()))
    dims = np.nextafter(np.asarray(2 * (required + guard), dtype=np.float32), np.float32(np.inf))
    if not np.isfinite(dims).all() or np.any(dims <= 0):
        raise ValueError("world cuboid dimensions are not representable in float32")
    margin = float((dims.astype(float) / 2 - np.abs(points - center.astype(float))).min())
    if margin < 0:
        raise RuntimeError("complete world piece coverage proof failed")
    return center, dims, margin


def world_piece_obstacles(geometries: Sequence[Mapping], *, max_obstacles: int = 59) -> list[dict]:
    """Build compatible world-frame cuboids, with independently checkable cover.

    Reserve the live table count before passing the 64-slot planner budget.
    Piece merging is a conservative fallback, never pruning or sampling.
    """
    maximum = _budget(max_obstacles)
    data = _read_geometries(geometries)
    if len(data) > maximum:
        raise ValueError("world cuboid budget is smaller than source entity count")
    rows = []
    for name, indices in _groups(data, maximum):
        geometry, parts = data[name]
        points = np.vstack([parts[i] for i in indices])
        center, dims, margin = _cover(points)
        payload = {"name": f"{name}:complete-world-aabb-v1:{'-'.join(map(str, indices))}",
                   "dims": dims.astype(float).tolist(), "pose": [*center.astype(float).tolist(), 1., 0., 0., 0.],
                   "padding_m": 0., "source": WORLD_AABB_VERSION, "source_entity": name,
                   "geometry_hash": geometry["geometry_hash"], "source_piece_indices": list(indices),
                   "coverage": {"version": WORLD_AABB_VERSION, "coordinate_dtype": "float32",
                                "checked_world_vertex_count": len(points),
                                "minimum_world_axis_coverage_margin_m": margin}}
        rows.append({**payload, "coverage_hash": artifact_hash(payload)})
    verify_world_piece_obstacles(rows, geometries, max_obstacles=maximum)
    return rows


def verify_world_piece_obstacles(obstacles: Sequence[Mapping], geometries: Sequence[Mapping],
                                 *, max_obstacles: int = 59) -> dict:
    """Fail closed unless every complete piece appears exactly once and fits.

    This certifies emitted float32 WORLD boxes. The planner must preserve their
    full extent when transforming into its internal collision frame.
    """
    maximum = _budget(max_obstacles)
    data = _read_geometries(geometries)
    if len(obstacles) > maximum:
        raise ValueError("world obstacle cache budget exceeded")
    accounted = {name: [] for name in data}
    names = set()
    for row in obstacles:
        if (not isinstance(row, Mapping) or row.get("source") != WORLD_AABB_VERSION
                or artifact_hash(row, omit_keys=("coverage_hash",)) != row.get("coverage_hash")):
            raise ValueError("world piece coverage version/hash mismatch")
        name = row.get("source_entity")
        if (name not in data or not isinstance(row.get("name"), str) or not row["name"]
                or row.get("name") in names):
            raise ValueError("invalid world obstacle source/name")
        names.add(row.get("name"))
        geometry, parts = data[name]
        if row.get("geometry_hash") != geometry["geometry_hash"]:
            raise ValueError("world obstacle source geometry mismatch")
        indices = row.get("source_piece_indices")
        if (not isinstance(indices, list) or not indices
                or any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(parts) for i in indices)
                or indices != sorted(set(indices))):
            raise ValueError("invalid world obstacle piece partition")
        points = np.vstack([parts[i] for i in indices])
        dims, pose = np.asarray(row.get("dims"), dtype=float), np.asarray(row.get("pose"), dtype=float)
        if (dims.shape != (3,) or pose.shape != (7,) or not np.isfinite(dims).all()
                or not np.isfinite(pose).all() or np.any(dims <= 0)
                or not np.array_equal(pose[3:], [1., 0., 0., 0.])
                or not np.array_equal(dims, dims.astype(np.float32).astype(float))
                or not np.array_equal(pose[:3], pose[:3].astype(np.float32).astype(float))
                or row.get("padding_m") != 0.):
            raise ValueError("invalid emitted float32 world cuboid")
        margin = float((dims / 2 - np.abs(points - pose[:3])).min())
        if margin < 0:
            raise ValueError("world cuboid omits complete collision geometry")
        expected = {"version": WORLD_AABB_VERSION, "coordinate_dtype": "float32",
                    "checked_world_vertex_count": len(points), "minimum_world_axis_coverage_margin_m": margin}
        if row.get("coverage") != expected:
            raise ValueError("world cuboid coverage receipt mismatch")
        accounted[name].extend(indices)
    if any(sorted(accounted[name]) != list(range(len(parts))) for name, (_, parts) in data.items()):
        raise ValueError("world cuboid partition omits or duplicates collision pieces")
    payload = {"version": WORLD_AABB_VERSION, "verified": True, "cuboid_count": len(obstacles),
               "piece_count": sum(len(parts) for _, parts in data.values()),
               "source_geometry_hashes": {name: geometry["geometry_hash"]
                                           for name, (geometry, _) in sorted(data.items())},
               "obstacles_hash": artifact_hash(list(obstacles))}
    return {**payload, "verification_hash": artifact_hash(payload)}


def _native_required_half(internal_points, native_inverse_pose):
    points = _points(internal_points)
    inverse = np.asarray(native_inverse_pose, dtype=float)
    if (inverse.shape != (7,) or not np.isfinite(inverse).all()
            or not np.array_equal(inverse, inverse.astype(np.float32).astype(float))
            or abs(float(inverse[3:] @ inverse[3:]) - 1.) > 16 * np.finfo(np.float32).eps):
        raise ValueError("native cuboid proof requires the actual finite float32 inverse pose")
    w, x, y, z = inverse[3:]
    if x == 0 and y == 0 and z == 0:
        # Match the kernel's identity branch, including its treatment of w.
        rotation = np.eye(3)
        absolute_terms = np.eye(3)
    else:
        # This is q*p*conj(q), NOT the normalized quaternion rotation used by
        # transforms3d. The CUDA sphere_obb_kernel.cu does not renormalize q.
        rotation = np.asarray([
            [w*w + x*x - y*y - z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, w*w - x*x + y*y - z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, w*w - x*x - y*y + z*z],
        ])
        absolute_terms = np.asarray([
            [w*w + x*x + y*y + z*z, 2*abs(x*y) + 2*abs(w*z), 2*abs(x*z) + 2*abs(w*y)],
            [2*abs(x*y) + 2*abs(w*z), w*w + x*x + y*y + z*z, 2*abs(y*z) + 2*abs(w*x)],
            [2*abs(x*z) + 2*abs(w*y), 2*abs(y*z) + 2*abs(w*x), w*w + x*x + y*y + z*z],
        ])
    local = points @ rotation.T + inverse[:3]
    # Every coordinate has fewer than 64 elementary arithmetic operations,
    # counting input float32 conversion and all unfused products/additions.
    # gamma_64 times the absolute-term sum bounds round-to-nearest error; FMA
    # contraction only reduces that bound. The +1 also dominates subnormal
    # flush-to-zero error and double-precision reference arithmetic here.
    unit = np.finfo(np.float32).eps / 2
    gamma = 64 * unit / (1 - 64 * unit)
    magnitude = np.abs(inverse[:3]) + absolute_terms @ np.abs(points).max(axis=0)
    error = gamma * (magnitude + 1.)
    required = np.abs(local).max(axis=0) + error
    if not np.isfinite(required).all() or np.any(required > np.finfo(np.float32).max / 4):
        raise ValueError("native cuboid coverage exceeds finite float32 arithmetic")
    return points, inverse, required, error


def native_cuboid_cover(internal_points, native_inverse_pose, *, minimum_dims=None) -> dict:
    """Expand dimensions around the ACTUAL cached inverse pose, never change it.

    ``internal_points`` must contain all eight transformed corners of the
    already-certified world AABB. Call separately for both solver readbacks.
    No robot margin is used to absorb obstacle-coordinate rounding errors.
    """
    points, inverse, required, _ = _native_required_half(internal_points, native_inverse_pose)
    target = 2 * required
    if minimum_dims is not None:
        minimum = np.asarray(minimum_dims, dtype=float)
        if minimum.shape != (3,) or not np.isfinite(minimum).all() or np.any(minimum <= 0):
            raise ValueError("invalid current native cuboid dimensions")
        target = np.maximum(target, minimum)
    dims = np.nextafter(target.astype(np.float32), np.float32(np.inf))
    return {"dims": dims.astype(float).tolist(),
            **verify_native_cuboid_cover(points, inverse, dims)}


def verify_native_cuboid_cover(internal_points, native_inverse_pose, actual_dims) -> dict:
    """Independently recheck the actual read-back dimensions and inverse pose."""
    points, inverse, required, error = _native_required_half(internal_points, native_inverse_pose)
    dims = np.asarray(actual_dims, dtype=float)
    if (dims.shape != (3,) or not np.isfinite(dims).all() or np.any(dims <= 0)
            or not np.array_equal(dims, dims.astype(np.float32).astype(float))):
        raise ValueError("native cuboid proof requires actual positive float32 dimensions")
    margin = dims / 2 - required
    if np.any(margin < 0):
        raise ValueError("native cuboid undercovers world geometry after pose quantization")
    payload = {"version": NATIVE_CUBOID_VERSION, "verified": True,
               "internal_points_hash": _array_hash(points, "<f8"),
               "native_inverse_pose_hash": _array_hash(inverse, "<f4"),
               "native_dims_hash": _array_hash(dims, "<f4"),
               "checked_point_count": len(points), "float32_arithmetic_bound_m": error.tolist(),
               "required_half_dimensions_m": required.tolist(),
               "minimum_coverage_margin_after_arithmetic_bound_m": float(margin.min())}
    return {**payload, "verification_hash": artifact_hash(payload)}
