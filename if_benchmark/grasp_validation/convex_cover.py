"""Conservative M15 held geometry from complete physical collision pieces.

No simulator imports are required for the cover/proof routines. The optional
live adapter imports SAPIEN only when a physical actor is supplied.
"""
from __future__ import annotations

import hashlib
import heapq
from itertools import product
from typing import Any, Sequence

import numpy as np

from if_benchmark.schema import artifact_hash

CONVEX_GEOMETRY_VERSION = "m15_live_collision_pieces_world_vertices_v1"
CONVEX_COVER_VERSION = "complete_piece_box_partition_spheres_float32_v2"


def _rigid_matrix(value):
    matrix = np.asarray(value, dtype=float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], rtol=0, atol=1e-10)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), rtol=0, atol=1e-6)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, rtol=0, atol=1e-6)):
        raise ValueError("collision cover transform must be finite, rigid and right handed")
    return matrix


def _live_pose_matrix(pose):
    """Keep the SDK transform, checking its float32 conversion against its pose.

    SAPIEN normalizes a float32 quaternion before forming a float32 matrix.
    Near a half turn, those operations can exceed _rigid_matrix's 1e-6
    orthogonality bound.  Only a live pose with an independently verified
    quaternion may use this path; arbitrary attachment matrices remain strict.
    Neither the simulator pose nor the matrix/vertices used for collision
    coverage are normalized or projected here.
    """
    position = np.asarray(pose.p, dtype=float)
    quaternion = np.asarray(pose.q, dtype=float)
    eps32 = np.finfo(np.float32).eps
    if (position.shape != (3,) or quaternion.shape != (4,)
            or not np.isfinite(position).all() or not np.isfinite(quaternion).all()
            or abs(float(quaternion @ quaternion) - 1.) > 8 * eps32):
        raise ValueError("collision cover live pose requires a finite unit quaternion")
    matrix = np.asarray(pose.to_transformation_matrix(), dtype=float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.array_equal(matrix[3], [0., 0., 0., 1.])
            or not np.array_equal(matrix[:3, 3], position)):
        raise ValueError("collision cover live pose matrix disagrees with its position")
    # Bound only the SDK's quaternion-normalization and matrix-conversion
    # rounding.  This reference is validation evidence, never replacement
    # geometry. Distortions beyond this SDK conversion error bound are rejected.
    w, x, y, z = quaternion / np.linalg.norm(quaternion)
    reference = np.asarray([
        [1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y)],
        [2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x)],
        [2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)],
    ])
    if (not np.allclose(matrix[:3, :3], reference, rtol=0, atol=16 * eps32)
            or np.linalg.det(matrix[:3, :3]) <= 0):
        raise ValueError("collision cover live pose matrix disagrees with its quaternion")
    return matrix


def _points(value):
    points = np.asarray(value, dtype=float)
    if points.ndim != 2 or points.shape[1:] != (3,) or not len(points) or not np.isfinite(points).all():
        raise ValueError("collision piece requires every finite vertex, shape (N, 3)")
    return points


def _geometry_hash(parts):
    digest = hashlib.sha256()
    for part in parts:
        digest.update(np.asarray(part.shape, dtype="<u8").tobytes())
        digest.update(np.asarray(part, dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


def live_collision_pieces(actor: Any) -> dict[str, Any]:
    """Read complete cooked shapes, including scale and local/world poses.

    Mesh vertices enclose their complete convex hull, hence every mesh face.
    Analytic primitives use containing boxes rather than inscribed samples.
    Nothing is decimated or obtained from visual meshes or asset metadata.
    """
    from sapien.physx import (
        PhysxCollisionShapeBox, PhysxCollisionShapeCapsule,
        PhysxCollisionShapeConvexMesh, PhysxCollisionShapeCylinder,
        PhysxCollisionShapeSphere, PhysxCollisionShapeTriangleMesh,
    )

    physical = getattr(actor, "actor", actor)
    components = (list(physical.get_links()) if hasattr(physical, "get_links") else
                  list(physical.get_components()) if hasattr(physical, "get_components") else [physical])
    parts, sources = [], []
    signs = np.asarray(list(product((-1., 1.), repeat=3)))
    for component in components:
        get_shapes = getattr(component, "get_collision_shapes", None)
        if not callable(get_shapes):
            continue
        shapes = get_shapes()
        if not shapes:
            continue
        world = _live_pose_matrix(component.entity.pose)
        for shape in shapes:
            source = {"shape_type": type(shape).__name__}
            if isinstance(shape, (PhysxCollisionShapeConvexMesh, PhysxCollisionShapeTriangleMesh)):
                vertices = _points(shape.vertices)
                scale = np.asarray(shape.scale, dtype=float)
                if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0):
                    raise ValueError("collision piece scale must be finite and positive")
                points = vertices * scale
                source["scale"] = scale.tolist()
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
                    raise TypeError(f"unsupported held collision shape: {type(shape).__name__}")
                if half.shape != (3,) or not np.isfinite(half).all() or np.any(half <= 0):
                    raise ValueError("held primitive bounds must be finite and positive")
                points = signs * half
                source["containing_box_half_size"] = half.tolist()
            transform = world @ _live_pose_matrix(shape.local_pose)
            part = _points(points @ transform[:3, :3].T + transform[:3, 3])
            parts.append(part)
            sources.append({**source, "vertex_count": len(part)})
    if not parts:
        raise ValueError("held actor has no physical collision geometry")
    return {"version": CONVEX_GEOMETRY_VERSION, "frame": "world",
            "geometry_sha256": _geometry_hash(parts), "sources": sources,
            "pieces": [part.tolist() for part in parts]}


def _bounding_sphere(points):
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    return center, float(np.linalg.norm(points - center, axis=1).max())


def _group_pieces(parts, slot_count):
    # A heap reuses unchanged pair fits. Each merge keeps complete pieces;
    # spatial clustering individual vertices could fail to cover a convex face.
    groups = {(i,): (part, _bounding_sphere(part)) for i, part in enumerate(parts)}
    pairs = []

    def offer(left, right):
        left, right = sorted((left, right))
        lv, (_, lr) = groups[left]
        rv, (_, rr) = groups[right]
        _, radius = _bounding_sphere(np.vstack((lv, rv)))
        heapq.heappush(pairs, (radius, radius ** 3 - lr ** 3 - rr ** 3, left, right))

    if len(groups) > slot_count:
        keys = sorted(groups)
        for i, left in enumerate(keys):
            for right in keys[i + 1:]:
                offer(left, right)
    while len(groups) > slot_count:
        _, _, left, right = heapq.heappop(pairs)
        if left not in groups or right not in groups:
            continue
        vertices = np.vstack((groups.pop(left)[0], groups.pop(right)[0]))
        merged = tuple(sorted((*left, *right)))
        groups[merged] = (vertices, _bounding_sphere(vertices))
        for other in sorted(groups):
            if other != merged:
                offer(merged, other)
    return sorted(groups)


def _cover_inputs(geometry, world_to_parent, dimension_envelope_m, padding_m, slot_count):
    if isinstance(slot_count, bool) or not isinstance(slot_count, int) or not 1 <= slot_count <= 32:
        raise ValueError("collision cover slot_count must be an integer in [1, 32]")
    if isinstance(dimension_envelope_m, bool) or isinstance(padding_m, bool):
        raise ValueError("collision cover margins must be finite and nonnegative")
    envelope, padding = float(dimension_envelope_m), float(padding_m)
    if not np.isfinite([envelope, padding]).all() or min(envelope, padding) < 0:
        raise ValueError("collision cover margins must be finite and nonnegative")
    transform = _rigid_matrix(world_to_parent)
    if (not isinstance(geometry, dict) or geometry.get("version") != CONVEX_GEOMETRY_VERSION
            or geometry.get("frame") != "world"):
        raise ValueError("held collision piece geometry version/frame mismatch")
    world_parts = [_points(part) for part in geometry.get("pieces", [])]
    if not world_parts or _geometry_hash(world_parts) != geometry.get("geometry_sha256"):
        raise ValueError("held collision piece geometry hash mismatch or empty geometry")
    with np.errstate(over="ignore", invalid="ignore"):
        parts = [_points(v @ transform[:3, :3].T + transform[:3, 3]) for v in world_parts]
        offsets = np.asarray(list(product((-envelope, envelope), repeat=3))) @ transform[:3, :3].T
        expanded = [_points((v[:, None, :] + offsets[None, :, :]).reshape(-1, 3)) for v in parts]
    return transform, envelope, padding, parts, expanded


def _cell_corners(minimum, maximum):
    # Select endpoints directly: midpoint +/- half-extent can round inward.
    return np.asarray(list(product(*zip(minimum, maximum))), dtype=float)


def _cell_sphere(minimum, maximum, padding):
    corners = _cell_corners(minimum, maximum)
    with np.errstate(over="ignore", invalid="ignore"):
        center32 = (minimum * .5 + maximum * .5).astype(np.float32)
        distance = float(np.linalg.norm(corners - center32.astype(float), axis=1).max())
        required = distance + padding
        required += 16 * np.finfo(float).eps * max(1., float(np.abs(corners).max()), required)
        radius32 = np.nextafter(np.float32(required), np.float32(np.inf))
        if not np.isfinite(center32).all() or not np.isfinite(radius32) or radius32 <= 0:
            raise ValueError("collision cover is not representable in float32")
    margin = float(radius32) - distance - padding
    if margin < 0:
        raise ValueError("collision cell sphere does not cover its padded corners")
    return np.asarray([*center32, radius32], dtype=np.float32), margin


def _split_cell(minimum, maximum):
    extent = maximum - minimum
    axis = int(np.argmax(extent))
    split = float(minimum[axis] * .5 + maximum[axis] * .5)
    if not minimum[axis] < split < maximum[axis]:
        return None
    left_max, right_min = maximum.copy(), minimum.copy()
    left_max[axis] = right_min[axis] = split
    return axis, split, (minimum.copy(), left_max), (right_min, maximum.copy())


def _piece_proof(roots, cells, parts):
    rows = []
    for root in roots:
        sphere_indices = [row["sphere_index"] for row in cells if row["root_index"] == root["root_index"]]
        for piece in root["piece_indices"]:
            rows.append({"piece_index": piece, "root_index": root["root_index"],
                         "sphere_indices": sphere_indices, "vertex_count": len(parts[piece]),
                         "expanded_vertex_count": 8 * len(parts[piece])})
    return sorted(rows, key=lambda row: row["piece_index"])


def conservative_piece_cover(
    geometry: dict[str, Any], *, world_to_parent: Sequence[Sequence[float]],
    dimension_envelope_m: float, padding_m: float, slot_count: int = 32,
) -> dict[str, Any]:
    """Cover complete pieces with a certified partition of enclosing boxes.

    Each root contains every vertex of its complete convex pieces plus the
    unchanged world-axis envelope. Closed child boxes tile the parent; each
    leaf sphere contains the entire child box plus ball padding. Thus faces,
    edges and interiors stay covered even when one long piece uses many balls.
    The finite slot budget controls approximation, not a monotonicity promise.
    """
    transform, envelope, padding, parts, expanded = _cover_inputs(
        geometry, world_to_parent, dimension_envelope_m, padding_m, slot_count)
    groups = _group_pieces(parts, slot_count)
    roots, leaves = [], {}
    for root_index, ids in enumerate(groups):
        vertices = np.vstack([expanded[i] for i in ids])
        minimum, maximum = vertices.min(axis=0), vertices.max(axis=0)
        # Fail before ranking/splitting if even the root is not representable.
        _cell_sphere(minimum, maximum, padding)
        roots.append({"root_index": root_index, "piece_indices": list(ids),
                      "minimum": minimum.tolist(), "maximum": maximum.tolist()})
        leaves[(root_index, "")] = (minimum, maximum)
    splits = []
    while len(leaves) < slot_count:
        candidates = [(key, _split_cell(*box)) for key, box in leaves.items()]
        candidates = [(key, split) for key, split in candidates if split is not None]
        if not candidates:
            break
        key, (axis, split, left, right) = min(
            candidates, key=lambda item: (-float(np.linalg.norm(leaves[item[0]][1] - leaves[item[0]][0])), item[0]))
        del leaves[key]
        leaves[(key[0], key[1] + "0")] = left
        leaves[(key[0], key[1] + "1")] = right
        splits.append({"root_index": key[0], "node_path": key[1], "axis": axis, "split": split})
    rows = np.zeros((slot_count, 4), dtype=np.float32)
    rows[:, 3] = -10
    cells = []
    for index, ((root_index, node_path), (minimum, maximum)) in enumerate(sorted(leaves.items())):
        rows[index], margin = _cell_sphere(minimum, maximum, padding)
        cells.append({"root_index": root_index, "node_path": node_path, "sphere_index": index,
                      "minimum": minimum.tolist(), "maximum": maximum.tolist(),
                      "minimum_padded_coverage_margin_m": margin})
    payload = {"version": CONVEX_COVER_VERSION, "geometry_sha256": geometry["geometry_sha256"],
               "world_to_parent": transform.tolist(), "dimension_envelope_m": envelope,
               "padding_m": padding, "slot_count": slot_count, "active_sphere_count": len(cells),
               "piece_count": len(parts), "vertex_count": sum(map(len, parts)),
               "piece_coverage": _piece_proof(roots, cells, parts),
               "partition_roots": roots, "partition_splits": splits, "cell_coverage": cells,
               "sphere_tensor_sha256": hashlib.sha256(rows.astype("<f4").tobytes(order="C")).hexdigest(),
               "sphere_dtype": "float32", "spheres": rows.tolist()}
    return {**payload, "cover_hash": artifact_hash(payload)}


def verify_piece_cover(
    cover: dict[str, Any], geometry: dict[str, Any], *,
    world_to_parent: Sequence[Sequence[float]], dimension_envelope_m: float,
    padding_m: float, slot_count: int = 32,
) -> dict[str, Any]:
    """Independently validate the full coverage proof against caller inputs.

    A valid self-hash is insufficient. Re-read every source vertex, reconstruct
    the complete split tree, and check every leaf's actual float32 sphere.
    Call before adding installation receipts to the producer's certificate.
    """
    transform, envelope, padding, parts, expanded = _cover_inputs(
        geometry, world_to_parent, dimension_envelope_m, padding_m, slot_count)

    def require(test, message):
        if not test:
            raise ValueError("invalid collision cover certificate: " + message)

    def index(value, limit, name):
        require(isinstance(value, int) and not isinstance(value, bool) and 0 <= value < limit, name)
        return value

    def bounds(row):
        low, high = np.asarray(row.get("minimum"), dtype=float), np.asarray(row.get("maximum"), dtype=float)
        require(low.shape == high.shape == (3,) and np.isfinite(low).all()
                and np.isfinite(high).all() and np.all(low <= high), "nonfinite or invalid cell bounds")
        return low, high

    def node_key(row, root_count):
        root = index(row.get("root_index"), root_count, "root index")
        path = row.get("node_path")
        require(isinstance(path, str) and len(path) < slot_count and all(c in "01" for c in path), "node path")
        return root, path

    require(isinstance(cover, dict), "not a mapping")
    require(cover.get("version") == CONVEX_COVER_VERSION, "version")
    require(cover.get("cover_hash") == artifact_hash(cover, omit_keys=("cover_hash",)), "self hash")
    require(cover.get("geometry_sha256") == geometry["geometry_sha256"], "source geometry hash")
    require(np.array_equal(cover.get("world_to_parent"), transform), "caller transform")
    require(cover.get("dimension_envelope_m") == envelope and cover.get("padding_m") == padding, "caller margins")
    require(isinstance(cover.get("slot_count"), int) and cover.get("slot_count") == slot_count
            and not isinstance(cover.get("slot_count"), bool), "slot budget")
    for field, expected in (("piece_count", len(parts)), ("vertex_count", sum(map(len, parts)))):
        require(isinstance(cover.get(field), int) and not isinstance(cover.get(field), bool)
                and cover.get(field) == expected, "source counts")
    raw_spheres = np.asarray(cover.get("spheres"), dtype=float)
    require(raw_spheres.shape == (slot_count, 4) and np.isfinite(raw_spheres).all(), "sphere shape or nonfinite sphere")
    with np.errstate(over="ignore", invalid="ignore"):
        spheres = raw_spheres.astype(np.float32)
    require(cover.get("sphere_dtype") == "float32" and np.isfinite(spheres).all()
            and np.array_equal(raw_spheres, spheres), "spheres are not exact float32 values")
    sphere_sha = hashlib.sha256(spheres.astype("<f4").tobytes(order="C")).hexdigest()
    require(cover.get("sphere_tensor_sha256") == sphere_sha, "sphere tensor bytes")

    roots = cover.get("partition_roots")
    require(isinstance(roots, list) and len(roots) == min(len(parts), slot_count), "root count")
    leaves, source_ids = {}, []
    for root_index, root in enumerate(roots):
        require(isinstance(root, dict) and root.get("root_index") == root_index, "root ordering")
        index(root.get("root_index"), len(roots), "root index")
        ids = root.get("piece_indices")
        require(isinstance(ids, list) and bool(ids), "root source membership")
        for piece in ids:
            index(piece, len(parts), "source piece index")
        require(ids == sorted(set(ids)), "duplicate or unordered source membership")
        source_ids.extend(ids)
        low, high = bounds(root)
        vertices = np.vstack([expanded[piece] for piece in ids])
        require(np.array_equal(low, vertices.min(axis=0)) and np.array_equal(high, vertices.max(axis=0)),
                "root does not bind complete expanded source geometry")
        leaves[(root_index, "")] = (low, high)
    require(sorted(source_ids) == list(range(len(parts))), "missing or repeated source piece")

    splits = cover.get("partition_splits")
    require(isinstance(splits, list) and len(splits) <= slot_count - len(roots), "split budget")
    for split in splits:
        require(isinstance(split, dict), "split mapping")
        key = node_key(split, len(roots))
        require(key in leaves, "split refers to a missing or already split cell")
        partition = _split_cell(*leaves.pop(key))
        require(partition is not None, "split is not representable")
        axis, coordinate, left, right = partition
        index(split.get("axis"), 3, "split axis")
        require(split["axis"] == axis and split.get("split") == coordinate, "incorrect split plane")
        leaves[(key[0], key[1] + "0")] = left
        leaves[(key[0], key[1] + "1")] = right
    require(len(leaves) == slot_count or all(_split_cell(*box) is None for box in leaves.values()),
            "unused available partition slots")
    cells = cover.get("cell_coverage")
    require(isinstance(cells, list) and len(cells) == len(leaves), "missing or repeated leaf")
    index(cover.get("active_sphere_count"), slot_count + 1, "active sphere count")
    require(cover.get("active_sphere_count") == len(cells), "active sphere count")
    seen = set()
    for sphere_index, cell in enumerate(cells):
        require(isinstance(cell, dict), "leaf mapping")
        key = node_key(cell, len(roots))
        require(key in leaves and key not in seen, "missing or repeated leaf")
        seen.add(key)
        index(cell.get("sphere_index"), slot_count, "leaf sphere index")
        require(cell.get("sphere_index") == sphere_index, "leaf sphere index")
        low, high = bounds(cell)
        expected_low, expected_high = leaves[key]
        require(np.array_equal(low, expected_low) and np.array_equal(high, expected_high), "leaf does not tile its parent")
        expected_sphere, margin = _cell_sphere(low, high, padding)
        require(np.array_equal(spheres[sphere_index], expected_sphere), "sphere shrunk or moved from certified cell")
        require(cell.get("minimum_padded_coverage_margin_m") == margin and margin >= 0, "incorrect coverage margin")
    require(seen == set(leaves), "missing leaf")
    inactive = np.zeros((slot_count - len(cells), 4), dtype=np.float32)
    inactive[:, 3] = -10
    require(np.array_equal(spheres[len(cells):], inactive), "unbound active sphere")
    require(cover.get("piece_coverage") == _piece_proof(roots, cells, parts), "source-to-leaf coverage membership")
    return {"verified": True, "cover_hash": cover["cover_hash"], "geometry_sha256": geometry["geometry_sha256"],
            "sphere_tensor_sha256": sphere_sha, "active_sphere_count": len(cells), "slot_count": slot_count}
