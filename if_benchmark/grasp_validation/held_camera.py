"""Exact, attachment-scoped replacement of the held/fixed-camera sphere pair.

All held pieces keep the existing world-axis expansion and padding. Native
held spheres remain installed for every other pair and every scene obstacle.
The CPU geometry routines do not move the simulator or alter a planned path.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from itertools import product
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import transforms3d as t3d

from if_benchmark.schema import artifact_hash
from .convex_cover import _geometry_hash, _live_pose_matrix, _points, _rigid_matrix, live_collision_pieces
from .core import trajectory_summary

HELD_CAMERA_POLICY = "if_held_fixed_camera_exact_pair_v1"
NUMERICAL_GUARD_M = 1e-5
# This kernel skips each (-1, -1) pair without changing launch shape or storage.
# Unknown ABIs cannot use an in-place mask until independently reviewed.
_REVIEWED_KERNEL_SHA256 = "612be710756be5d4b80cf7bb7211df94feab9108b4114c4dabb6773e04c03328"


def _matrix(p, q):
    return t3d.affines.compose(p, t3d.quaternions.quat2mat(q), [1, 1, 1])


def _transform(points, matrix):
    return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]


def fixed_camera_model(urdf_path, ee_link):
    """Only the reviewed Aloha fixed wrist-camera relation is eligible."""
    arm = {"fl_link6": "left", "fr_link6": "right"}.get(ee_link)
    if arm is None:
        raise RuntimeError("held_camera_unsupported_end_link")
    path = Path(urdf_path)
    camera = arm + "_camera"
    joints = [j for j in ET.parse(path).getroot().findall("joint")
              if j.find("child") is not None and j.find("child").get("link") == camera]
    if (len(joints) != 1 or joints[0].get("type") != "fixed"
            or joints[0].find("parent").get("link") != ee_link):
        raise RuntimeError("held_camera_requires_direct_fixed_joint")
    origin = joints[0].find("origin")
    p = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
    rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
    if p.shape != (3,) or rpy.shape != (3,) or not np.isfinite([p, rpy]).all():
        raise RuntimeError("held_camera_fixed_transform_invalid")
    fixed = np.eye(4)
    fixed[:3, :3] = t3d.euler.euler2mat(*rpy)
    fixed[:3, 3] = p
    return {"arm": arm, "camera": camera, "ee_link": ee_link,
            "fixed_transform": fixed.tolist(), "urdf_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def live_camera_descriptor(robot, arm):
    """Serializable complete live camera geometry, suitable for planner RPC."""
    if arm not in {"left", "right"}:
        raise ValueError("invalid held camera arm")
    articulation = getattr(robot, arm + "_entity")
    ee_name = "fl_link6" if arm == "left" else "fr_link6"
    links = {link.name: link for link in articulation.links}
    camera = links[arm + "_camera"]
    camera_world = _live_pose_matrix(camera.entity.pose)
    ee_world = _live_pose_matrix(links[ee_name].entity.pose)
    geometry = live_collision_pieces(camera)
    # A hull is exact for these cooked convex/box camera shapes. Other live
    # shapes need their own reviewed exact conversion, never a visual mesh.
    if any(row["shape_type"] not in {"PhysxCollisionShapeConvexMesh", "PhysxCollisionShapeBox"}
           for row in geometry["sources"]):
        raise RuntimeError("held_camera_shape_not_reviewed")
    parts = [_transform(p, np.linalg.inv(camera_world)) for p in geometry["pieces"]]
    result = {"camera": arm + "_camera", "ee_link": ee_name,
              "live_fixed_transform": (np.linalg.inv(ee_world) @ camera_world).tolist(),
              "pieces_local": [p.tolist() for p in parts],
              "geometry_sha256": _geometry_hash(parts), "sources": geometry["sources"]}
    result["descriptor_hash"] = artifact_hash(result)
    return result


def _piece_object(parts, name):
    from scipy.spatial import ConvexHull
    from mplib import Pose
    from mplib.collision_detection.fcl import CollisionObject, Convex, FCLObject
    shapes = []
    for points in parts:
        vertices = np.unique(_points(points), axis=0)
        hull = ConvexHull(vertices)
        faces = hull.simplices.copy()
        for index, face in enumerate(faces):
            if np.dot(np.cross(vertices[face[1]] - vertices[face[0]],
                               vertices[face[2]] - vertices[face[0]]), hull.equations[index, :3]) < 0:
                faces[index, [1, 2]] = faces[index, [2, 1]]
        ids = sorted(set(faces.flat))
        lookup = {old: new for new, old in enumerate(ids)}
        faces = np.asarray([[lookup[i] for i in face] for face in faces], dtype=int)
        shapes.append(CollisionObject(Convex(vertices=vertices[ids], faces=faces)))
    if not shapes:
        raise RuntimeError("held_camera_empty_geometry")
    return FCLObject(name, Pose(), shapes, [Pose() for _ in shapes])


def exact_pair_geometry(descriptor, model, world_to_parent, native_camera_spheres, held_self_offset):
    """Check each whole held convex piece; never fill gaps between pieces."""
    from mplib import Pose
    from mplib.collision_detection.fcl import CollisionObject, FCLObject, Sphere, collide, distance
    transform = _rigid_matrix(world_to_parent)
    camera = descriptor["fixed_camera"]
    if (camera.get("descriptor_hash") != artifact_hash(camera, omit_keys=("descriptor_hash",))
            or camera.get("camera") != model["camera"] or camera.get("ee_link") != model["ee_link"]):
        raise RuntimeError("held_camera_descriptor_mismatch")
    fixed = _rigid_matrix(model["fixed_transform"])
    live_fixed = np.asarray(camera["live_fixed_transform"], dtype=float)
    if live_fixed.shape != (4, 4) or not np.allclose(live_fixed, fixed, rtol=0, atol=2e-5):
        raise RuntimeError("held_camera_live_fixed_joint_mismatch")
    camera_parts = [_points(p) for p in camera["pieces_local"]]
    if _geometry_hash(camera_parts) != camera["geometry_sha256"]:
        raise RuntimeError("held_camera_geometry_hash_mismatch")
    parts = [_points(p) for p in descriptor["collision_pieces"]["pieces"]]
    if _geometry_hash(parts) != descriptor["collision_pieces"]["geometry_sha256"]:
        raise RuntimeError("held_camera_subject_geometry_hash_mismatch")
    envelope, padding = float(descriptor["dimension_envelope_m"]), float(descriptor["padding_m"])
    if envelope != .005 or padding != .005 or not np.isfinite(held_self_offset):
        raise RuntimeError("held_camera_unreviewed_margin")
    offsets = np.asarray(list(product((-envelope, envelope), repeat=3)))
    expanded = [_transform((p[:, None, :] + offsets[None, :, :]).reshape(-1, 3), transform) for p in parts]
    raw = _piece_object([_transform(p, transform) for p in parts], "held_raw")
    held = _piece_object(expanded, "held_expanded")
    actual_camera = _piece_object([_transform(p, fixed) for p in camera_parts], "fixed_camera")
    def check(a, b):
        gap = float(distance(a, b).min_distance)
        if not np.isfinite(gap):
            raise RuntimeError("held_camera_distance_nonfinite")
        return gap, bool(collide(a, b).is_collision())
    raw_gap, raw_collision = check(raw, actual_camera)
    gap, collision = check(held, actual_camera)
    sphere_checks = []
    for row in native_camera_spheres:
        center = np.asarray(row["center_parent"], dtype=float)
        radius = float(row["native_radius_m"]) + float(row["native_self_offset_m"])
        if center.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(radius) or radius <= 0:
            raise RuntimeError("held_camera_native_sphere_invalid")
        ball = FCLObject("native_camera", Pose(p=center), [CollisionObject(Sphere(radius=radius))], [Pose()])
        sphere_gap, sphere_collision = check(held, ball)
        sphere_checks.append({**row, "effective_radius_m": radius, "expanded_gap_m": sphere_gap,
                              "remaining_margin_m": sphere_gap - padding - held_self_offset,
                              "collision": sphere_collision})
    if not sphere_checks:
        raise RuntimeError("held_camera_native_spheres_missing")
    mesh_margin = gap - padding
    native_margin = min(row["remaining_margin_m"] for row in sphere_checks)
    passed = (not raw_collision and not collision and mesh_margin >= NUMERICAL_GUARD_M
              and not any(row["collision"] for row in sphere_checks) and native_margin >= NUMERICAL_GUARD_M)
    receipt = {"policy": HELD_CAMERA_POLICY, "passed": bool(passed), "model": model,
               "geometry_sha256": descriptor["collision_pieces"]["geometry_sha256"],
               "camera_geometry_sha256": camera["geometry_sha256"], "piece_count": len(parts),
               "camera_piece_count": len(camera_parts), "dimension_envelope_m": envelope, "padding_m": padding,
               "held_self_offset_m": float(held_self_offset), "numerical_guard_m": NUMERICAL_GUARD_M,
               "raw_gap_m": raw_gap, "raw_collision": raw_collision, "expanded_gap_m": gap,
               "expanded_collision": collision, "remaining_exact_mesh_margin_m": mesh_margin,
               "remaining_native_camera_margin_m": native_margin, "camera_spheres": sphere_checks}
    receipt["validation_hash"] = artifact_hash(receipt)
    return receipt


def _numpy(tensor):
    return tensor.detach().to("cpu").numpy()


def _sphere_names(kinematics):
    config = kinematics.kinematics_config
    names = {int(i): name for name, i in config.link_name_to_idx_map.items()}
    return [names[int(i)] for i in _numpy(config.link_sphere_idx_map).reshape(-1)]


def native_pair_frame(planner, qpos, *, generator=None):
    generator = planner.motion_gen if generator is None else generator
    q = np.asarray([planner._active_joint_angles(row) for row in np.atleast_2d(qpos)], dtype=float)
    state = generator.kinematics.get_state(generator.tensor_args.to_device(q))
    ee = np.stack([_matrix(p, quat) for p, quat in zip(_numpy(state.ee_position), _numpy(state.ee_quaternion))])
    spheres = _numpy(state.link_spheres_tensor).reshape(len(q), -1, 4)
    names = _sphere_names(generator.kinematics)
    camera = planner._held_camera_model["camera"]
    ci = [i for i, name in enumerate(names) if name == camera]
    hi = [i for i, name in enumerate(names) if name == planner.ATTACHED_LINK_NAME]
    if not ci or not hi or not np.isfinite(spheres).all():
        raise RuntimeError("held_camera_native_link_mapping_invalid")
    centers = np.einsum("nki,nij->nkj", spheres[:, ci, :3] - ee[:, None, :3, 3], ee[:, :3, :3])
    offset = _numpy(generator.kinematics.get_self_collision_config().offset).reshape(-1)
    if not np.isfinite(offset).all():
        raise RuntimeError("held_camera_native_offset_invalid")
    rows = [{"index": i, "center_parent": centers[0, j].tolist(), "native_radius_m": float(spheres[0, i, 3]),
             "native_self_offset_m": float(offset[i])} for j, i in enumerate(ci)]
    origin_p, origin_q = planner._collision_pose_in_base([0., 0., 0., 1., 0., 0., 0.])
    world_to_parent = np.linalg.inv(ee[0]) @ _matrix(origin_p, origin_q)
    return world_to_parent, rows, float(np.max(offset[hi])), centers, spheres[:, ci, 3]


class NativeCameraPairMask:
    """Preserve CUDA graph tensor addresses and all non-target native pairs."""
    def __init__(self, planner):
        import curobo
        kernel = Path(curobo.__file__).parent / "curobolib/cpp/self_collision_kernel.cu"
        self.rows = []
        self.masked = False
        self.camera = planner._held_camera_model["camera"]
        seen = set()
        for label, gen in (("single", planner.motion_gen), ("batch", planner.motion_gen_batch)):
            names = _sphere_names(gen.kinematics)
            configs = [gen.kinematics.get_self_collision_config()]
            for rollout in [gen.rollout_fn, *gen.get_all_rollout_instances()]:
                for attr in ("robot_self_collision_constraint", "robot_self_collision_cost"):
                    cost = getattr(rollout, attr, None)
                    if cost is not None:
                        configs.append(cost.self_collision_kin_config)
            for cfg in configs:
                tensor = cfg.thread_location if cfg.experimental_kernel else cfg.collision_matrix
                key = (str(tensor.device), tensor.data_ptr())
                if key in seen:
                    continue
                seen.add(key)
                original = tensor.clone()
                masked = original.clone()
                raw = _numpy(original)
                target = {i for i, n in enumerate(names) if n == planner.ATTACHED_LINK_NAME}
                camera = {i for i, n in enumerate(names) if n == self.camera}
                if not target or not camera:
                    raise RuntimeError("held_camera_pair_indices_missing")
                if cfg.experimental_kernel:
                    if hashlib.sha256(kernel.read_bytes()).hexdigest() != _REVIEWED_KERNEL_SHA256:
                        raise RuntimeError("held_camera_native_mask_abi_unreviewed")
                    pairs = raw.reshape(-1, 2)
                    selected = [i for i, (a, b) in enumerate(pairs)
                                if (a in target and b in camera) or (b in target and a in camera)]
                    expected = {tuple(sorted((a, b))) for a in target for b in camera}
                    actual = {tuple(sorted(map(int, pairs[i]))) for i in selected}
                    if actual != expected:
                        raise RuntimeError("held_camera_original_pair_coverage_invalid")
                    masked.reshape(-1, 2)[selected] = -1
                    count = len(selected)
                else:
                    if raw.shape != (len(names), len(names)):
                        raise RuntimeError("held_camera_native_matrix_shape_invalid")
                    count = 0
                    for a in target:
                        for b in camera:
                            if not raw[a, b] or not raw[b, a]:
                                raise RuntimeError("held_camera_original_matrix_pair_missing")
                            masked[a, b] = masked[b, a] = 0
                            count += 1
                self.rows.append({"label": label, "tensor": tensor, "original": original,
                                  "masked": masked, "offset": cfg.offset, "offset_snapshot": cfg.offset.clone(),
                                  "removed_pairs": count, "experimental": cfg.experimental_kernel,
                                  "thread_max": cfg.thread_max, "cfg": cfg, "pointer": tensor.data_ptr()})
        if not self.rows:
            raise RuntimeError("held_camera_native_pair_masks_missing")

    def verify(self):
        import torch
        for row in self.rows:
            current = row["cfg"].thread_location if row["experimental"] else row["cfg"].collision_matrix
            if (current.data_ptr() != row["pointer"] or row["tensor"].data_ptr() != row["pointer"]
                    or row["cfg"].offset.data_ptr() != row["offset"].data_ptr()
                    or row["cfg"].thread_max != row["thread_max"]
                    or not torch.equal(row["tensor"], row["masked" if self.masked else "original"])
                    or not torch.equal(row["offset"], row["offset_snapshot"])):
                raise RuntimeError("held_camera_native_pair_mask_changed")

    def restore(self):
        # Cleanup must restore even when an admission/readback check failed.
        for row in self.rows:
            row["tensor"].copy_(row["original"])
        self.masked = False
        self.verify()

    def set_masked(self, enabled):
        self.verify()
        try:
            for row in self.rows:
                row["tensor"].copy_(row["masked" if enabled else "original"])
            self.masked = bool(enabled)
            self.verify()
        except BaseException:
            for row in self.rows:
                row["tensor"].copy_(row["original"])
            self.masked = False
            raise
        return [{"solver": row["label"], "removed_pair_count": row["removed_pairs"] if enabled else 0,
                 "only_held_fixed_camera_pair_changed": True, "tensor_address_preserved": True,
                 "self_offsets_unchanged": True} for row in self.rows]


def bind_camera_pair(planner, qpos, descriptor, coverage):
    model = fixed_camera_model(planner._held_camera_urdf, planner._held_camera_ee_link)
    planner._held_camera_model = model
    if getattr(planner, "_held_camera_mask", None) is None:
        planner._held_camera_mask = NativeCameraPairMask(planner)
    transform, spheres, offset, _, _ = native_pair_frame(planner, qpos)
    if not np.allclose(transform, coverage["world_to_parent"], rtol=0, atol=2e-6):
        raise RuntimeError("held_camera_attachment_frame_mismatch")
    _, other_spheres, other_offset, _, _ = native_pair_frame(planner, qpos, generator=planner.motion_gen_batch)
    if other_spheres != spheres or other_offset != offset:
        raise RuntimeError("held_camera_solver_geometry_disagreement")
    gate = exact_pair_geometry(descriptor, model, transform, spheres, offset)
    gate["native_pair_masks"] = planner._held_camera_mask.set_masked(gate["passed"])
    gate["sphere_tensor_sha256"] = coverage["sphere_tensor_sha256"]
    gate["validation_hash"] = artifact_hash(gate, omit_keys=("validation_hash",))
    planner._held_camera_gate = gate
    return deepcopy(gate)


def require_camera_pair(planner):
    if not getattr(planner, "_attachment_hash", None) or getattr(planner, "_collision_geometry_mode", None) != planner.CARRY_COLLISION_MODE:
        return None
    if getattr(planner, "_held_camera_observation_failed", False):
        raise RuntimeError("held_camera_actual_pose_invalidated_binding")
    gate = getattr(planner, "_held_camera_gate", None)
    if (not gate or gate.get("policy") != HELD_CAMERA_POLICY
            or gate.get("validation_hash") != artifact_hash(gate, omit_keys=("validation_hash",))
            or getattr(planner, "_held_camera_attachment_hash", None) != planner._attachment_hash):
        raise RuntimeError("held_camera_binding_missing_or_stale")
    planner._held_camera_mask.verify()
    if planner._held_camera_mask.masked is not gate["passed"]:
        raise RuntimeError("held_camera_pair_mask_admission_mismatch")
    # Geometry tensors stay installed for every other pair and world check.
    for gen in (planner.motion_gen, planner.motion_gen_batch):
        tensor = _numpy(gen.robot_cfg.kinematics.kinematics_config.get_link_spheres(planner.ATTACHED_LINK_NAME))
        if hashlib.sha256(np.asarray(tensor, dtype="<f4").tobytes()).hexdigest() != gate["sphere_tensor_sha256"]:
            raise RuntimeError("held_camera_attachment_spheres_changed")
    return gate


def validate_camera_path(planner, path, *, generator=None):
    gate = require_camera_pair(planner)
    if gate is None:
        return None
    values = np.asarray(path, dtype=float)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("held camera validation requires a finite complete path")
    drift = 0.
    if gate["passed"]:
        expected = np.asarray([r["center_parent"] for r in gate["camera_spheres"]])
        expected_radii = np.asarray([r["native_radius_m"] for r in gate["camera_spheres"]])
        for start in range(0, len(values), 128):
            _, rows, offset, centers, radii = native_pair_frame(planner, values[start:start + 128], generator=generator)
            if (offset != gate["held_self_offset_m"] or not np.array_equal(radii, np.broadcast_to(expected_radii, radii.shape))
                    or [r["native_self_offset_m"] for r in rows] != [r["native_self_offset_m"] for r in gate["camera_spheres"]]):
                raise RuntimeError("held_camera_native_pair_geometry_changed")
            drift = max(drift, float(np.linalg.norm(centers - expected, axis=-1).max()))
    passed = gate["passed"] and drift < min(NUMERICAL_GUARD_M, gate["remaining_exact_mesh_margin_m"], gate["remaining_native_camera_margin_m"])
    receipt = {"policy": HELD_CAMERA_POLICY, "collision_free": bool(passed),
               "attachment_hash": planner._attachment_hash, "pair_validation_hash": gate["validation_hash"],
               "trajectory_sha256": trajectory_summary(values)["trajectory_sha256"],
               "checked_waypoint_count": len(values), "max_camera_relative_translation_drift_m": drift}
    receipt["validation_hash"] = artifact_hash(receipt)
    return receipt


def observe_camera_pair(planner, qpos, descriptor, *, expected_attachment_hash):
    planner._validate_expected_state(expected_attachment_hash=expected_attachment_hash)
    gate = require_camera_pair(planner)
    if gate is None:
        raise RuntimeError("held_camera_observation_requires_binding")
    try:
        transform, spheres, offset, _, _ = native_pair_frame(planner, qpos)
        measured = exact_pair_geometry(descriptor, planner._held_camera_model, transform, spheres, offset)
    except BaseException:
        planner._held_camera_observation_failed = True
        planner._held_camera_mask.restore()
        raise
    proof = {**measured, "bound_pair_validation_hash": gate["validation_hash"],
             "attachment_hash": planner._attachment_hash, "observed_qpos": np.asarray(qpos).tolist(),
             "stage": "actual_held_pose_recheck", "passed": gate["passed"] and measured["passed"]}
    proof["validation_hash"] = artifact_hash(proof, omit_keys=("validation_hash",))
    if not proof["passed"]:
        planner._held_camera_observation_failed = True
        planner._held_camera_mask.restore()
    return proof


def verify_camera_path_proof(proof, path, attachment_hash, *, pair_validation_hash=None):
    """Consumers cannot accept old, altered or incomplete replacement evidence."""
    if (not isinstance(proof, dict) or proof.get("policy") != HELD_CAMERA_POLICY
            or proof.get("collision_free") is not True
            or proof.get("attachment_hash") != attachment_hash
            or not proof.get("pair_validation_hash")
            or (pair_validation_hash is not None and proof["pair_validation_hash"] != pair_validation_hash)
            or proof.get("trajectory_sha256") != trajectory_summary(np.asarray(path))["trajectory_sha256"]
            or proof.get("checked_waypoint_count") != len(path)
            or proof.get("validation_hash") != artifact_hash(proof, omit_keys=("validation_hash",))):
        raise RuntimeError("held_fixed_camera_path_proof_missing_or_stale")
