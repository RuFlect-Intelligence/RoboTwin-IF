"""Cartesian acquisition proposal and independent, all-waypoint pose admission.

This module never certifies collision freedom. The caller must run the original
native world/self, exact support-exit, fixed-camera and complete mesh gates on
these SAME arrays. No free-space trajectory is a fallback for a failed lift.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from if_benchmark.schema import artifact_hash
from .core import trajectory_summary

VERTICAL_LIFT_VERSION = "if_acquisition_cartesian_world_z_v1"
# These are pose-tracking limits, not collision/support allowances.
XY_TOLERANCE_M = 5e-5
ORIENTATION_TOLERANCE_RAD = 5e-4
FK_PROGRESS_ROUNDOFF_M = 5e-7


class VerticalLiftFailure(RuntimeError):
    pass


def _transform(xyz, rpy):
    out = np.eye(4)
    out[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    out[:3, 3] = xyz
    return out


class SerialChain:
    """Double-precision proposal FK from the same URDF and native joint order.

    Native FK remains the admission authority; every proposed pose is compared
    against it before returning a path. Unsupported/mimic chain joints fail.
    """
    def __init__(self, urdf, base_link, ee_link, active_names, locked_positions):
        raw = Path(urdf).read_bytes()
        self.urdf_sha256 = hashlib.sha256(raw).hexdigest()
        tree = ET.fromstring(raw)
        by_child = {j.find("child").get("link"): j for j in tree.findall("joint")}
        nodes, link, seen = [], ee_link, set()
        while link != base_link:
            if link in seen or link not in by_child:
                raise VerticalLiftFailure("vertical_lift_urdf_chain_missing")
            seen.add(link)
            node = by_child[link]
            nodes.append(node)
            link = node.find("parent").get("link")
        self.names = list(active_names)
        self.nodes = []
        used = []
        for node in reversed(nodes):
            name, kind = node.get("name"), node.get("type")
            if node.find("mimic") is not None or kind not in {"fixed", "revolute", "continuous", "prismatic"}:
                raise VerticalLiftFailure("vertical_lift_unsupported_chain_joint")
            origin = node.find("origin")
            values = lambda key: np.fromstring(origin.get(key, "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
            transform = _transform(values("xyz"), values("rpy"))
            axis = node.find("axis")
            axis = np.fromstring(axis.get("xyz", "1 0 0"), sep=" ") if axis is not None else np.array([1., 0., 0.])
            if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) == 0:
                raise VerticalLiftFailure("vertical_lift_invalid_joint_axis")
            axis = axis / np.linalg.norm(axis)
            index = self.names.index(name) if name in self.names else None
            if kind != "fixed" and index is None and name not in locked_positions:
                raise VerticalLiftFailure("vertical_lift_missing_locked_joint")
            if index is not None:
                used.append(name)
            self.nodes.append((transform, axis, kind, index, float(locked_positions.get(name, 0.))))
        if len(used) != len(self.names) or set(used) != set(self.names):
            raise VerticalLiftFailure("vertical_lift_active_chain_mismatch")

    def state(self, q, world_from_base):
        transform = np.array(world_from_base, dtype=float, copy=True)
        axes = []
        for origin, axis, kind, index, locked in self.nodes:
            transform = transform @ origin
            if index is not None:
                axes.append((index, kind, transform[:3, :3] @ axis, transform[:3, 3].copy()))
            value = float(q[index]) if index is not None else locked
            if kind in {"revolute", "continuous"}:
                transform[:3, :3] = transform[:3, :3] @ Rotation.from_rotvec(axis * value).as_matrix()
            elif kind == "prismatic":
                transform[:3, 3] += transform[:3, :3] @ (axis * value)
        jacobian = np.zeros((6, len(q)))
        for index, kind, axis, origin in axes:
            if kind == "prismatic":
                jacobian[:3, index] = axis
            elif kind in {"revolute", "continuous"}:
                jacobian[:3, index] = np.cross(axis, transform[:3, 3] - origin)
                jacobian[3:, index] = axis
        return transform, jacobian


def _pose_error(target, actual):
    return np.concatenate((target[:3, 3] - actual[:3, 3],
        Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).as_rotvec()))


def _solve(chain, world_from_base, seed, target, bounds, *, position_tolerance=1e-10, rotation_tolerance=1e-9):
    q = seed.copy()
    for _ in range(12):
        actual, jacobian = chain.state(q, world_from_base)
        error = _pose_error(target, actual)
        if np.linalg.norm(error[:3]) <= position_tolerance and np.linalg.norm(error[3:]) <= rotation_tolerance:
            return q
        step = np.linalg.lstsq(jacobian, error, rcond=1e-6)[0]
        if not np.isfinite(step).all():
            break
        step *= min(1., .08 / max(float(np.max(np.abs(step))), 1e-12))
        q += step
        if np.any(q < bounds[0]) or np.any(q > bounds[1]):
            raise VerticalLiftFailure("vertical_lift_joint_limit")
    raise VerticalLiftFailure("vertical_lift_continuation_ik_failed")


def pose_receipt(path, native_matrices, target_world, *, dt):
    matrices = np.asarray(native_matrices, dtype=float)
    values = np.asarray(path)
    if (values.ndim != 2 or len(values) < 2 or matrices.shape != (len(values), 4, 4)
            or not np.isfinite(values).all() or not np.isfinite(matrices).all()):
        raise VerticalLiftFailure("vertical_lift_invalid_path")
    start = matrices[0]
    xy = np.linalg.norm(matrices[:, :2, 3] - start[None, :2, 3], axis=1)
    rotation = Rotation.from_matrix(matrices[:, :3, :3] @ start[:3, :3].T).magnitude()
    z = matrices[:, 2, 3]
    target_error = _pose_error(target_world, matrices[-1])
    failures = []
    if np.max(xy) > XY_TOLERANCE_M:
        failures.append("lateral_drift")
    if np.max(rotation) > ORIENTATION_TOLERANCE_RAD:
        failures.append("orientation_drift")
    # Compare with the running maximum as well as the start: small drops cannot
    # accumulate into a large descent. Exact held-support checks are stricter.
    backtrack = np.maximum.accumulate(z) - z
    if np.max(backtrack) > FK_PROGRESS_ROUNDOFF_M:
        failures.append("downward_motion")
    if (np.linalg.norm(target_error[:3]) > XY_TOLERANCE_M
            or np.linalg.norm(target_error[3:]) > ORIENTATION_TOLERANCE_RAD):
        failures.append("frozen_endpoint_missed")
    if z[-1] <= z[0]:
        failures.append("nonpositive_lift")
    receipt = {"version": VERTICAL_LIFT_VERSION, "passed": not failures,
               "failure_reasons": failures, "trajectory_sha256": trajectory_summary(values)["trajectory_sha256"],
               "checked_waypoint_count": len(values), "sample_dt_s": dt,
               "reference_start_world": start.tolist(), "requested_target_world": np.asarray(target_world).tolist(),
               "maximum_xy_drift_m": float(np.max(xy)), "maximum_orientation_drift_rad": float(np.max(rotation)),
               "maximum_downward_excursion_m": float(np.max(backtrack)),
               "endpoint_position_error_m": float(np.linalg.norm(target_error[:3])),
               "endpoint_rotation_error_rad": float(np.linalg.norm(target_error[3:])),
               "xy_tolerance_m": XY_TOLERANCE_M, "orientation_tolerance_rad": ORIENTATION_TOLERANCE_RAD,
               "fk_progress_roundoff_m": FK_PROGRESS_ROUNDOFF_M,
               "support_and_collision_thresholds_changed": False}
    receipt["validation_hash"] = artifact_hash(receipt)
    return receipt


def _propose_cartesian(chain, world_from_base, start_q, target_world, limits, native_fk, *,
                       admit_goal, make_receipt, dt=.004, smooth_start=False):
    """Direct Cartesian continuation, fixed sample rate, bounded time scaling.

    Uses smooth scalar progress rather than a free joint-space optimizer. The
    original endpoint is retained; an incompatible XY/orientation goal fails.
    No path is repaired after admission or handed out on any failed proof.
    """
    start_q = np.asarray(start_q, dtype=np.float32).copy()
    target_world = np.asarray(target_world, dtype=float)
    if (start_q.ndim != 1 or not len(start_q) or not np.isfinite(start_q).all()
            or target_world.shape != (4, 4) or not np.isfinite(target_world).all()
            or not math.isfinite(dt) or dt <= 0):
        raise VerticalLiftFailure("vertical_lift_invalid_input")
    bounds = np.asarray(limits["position"], dtype=float)
    if bounds.shape != (2, len(start_q)) or not np.isfinite(bounds).all():
        raise VerticalLiftFailure("vertical_lift_invalid_joint_limits")
    if np.any(bounds[0] >= bounds[1]) or np.any(start_q < bounds[0]) or np.any(start_q > bounds[1]):
        raise VerticalLiftFailure("vertical_lift_start_joint_limit")
    initial, _ = chain.state(start_q, world_from_base)
    native_start = native_fk(start_q[None])[0]
    if np.max(np.abs(initial - native_start)) > 2e-5:
        raise VerticalLiftFailure("vertical_lift_urdf_native_fk_mismatch")
    delta = _pose_error(target_world, initial)
    admit_goal(delta)
    duration = max(1.2, abs(float(delta[2])) / .025)
    for attempt in range(3):
        count = int(math.ceil(duration / dt)) + 1
        if count > 5001:
            raise VerticalLiftFailure("vertical_lift_duration_budget")
        u = np.linspace(0, 1, count)
        # Cubic progress has zero endpoint velocity. Quintic's initial cubic
        # increments can be smaller than float32 FK resolution at 250 Hz,
        # producing a false downward tick at support. Generate a resolvable
        # start directly; discrete acceleration/jerk limits still gate the path.
        # Carry may start near an extended-arm singularity: use zero first,
        # second and third Cartesian time derivatives at both ends. Keep
        # acquisition's resolvable cubic support-exit start unchanged.
        progress = (35*u**4 - 84*u**5 + 70*u**6 - 20*u**7
                    if smooth_start else 3*u**2 - 2*u**3)
        rows, q = [start_q], start_q.astype(float)
        relative_rotation = Rotation.from_matrix(target_world[:3, :3] @ initial[:3, :3].T).as_rotvec()
        for fraction in progress[1:]:
            target = initial.copy()
            target[:3, 3] += fraction * (target_world[:3, 3] - initial[:3, 3])
            target[:3, :3] = Rotation.from_rotvec(fraction * relative_rotation).as_matrix() @ initial[:3, :3]
            q = _solve(chain, world_from_base, q, target, bounds,
                **({"position_tolerance": 1e-13, "rotation_tolerance": 1e-12} if smooth_start else {}))
            rows.append(q.astype(np.float32))
        path = np.asarray(rows, dtype=np.float32)
        if np.any(path < bounds[0]) or np.any(path > bounds[1]):
            raise VerticalLiftFailure("vertical_lift_quantized_joint_limit")
        velocity = np.gradient(path.astype(float), dt, axis=0, edge_order=2)
        velocity[[0, -1]] = 0.
        acceleration = np.gradient(velocity, dt, axis=0, edge_order=2)
        jerk = np.gradient(acceleration, dt, axis=0, edge_order=2)
        ratios = []
        dynamics = {}
        for key, array, exponent in [("velocity", velocity, 1), ("acceleration", acceleration, 2), ("jerk", jerk, 3)]:
            bound = np.asarray(limits[key], dtype=float)
            if bound.shape != (2, len(start_q)) or not np.isfinite(bound).all():
                raise VerticalLiftFailure("vertical_lift_invalid_dynamic_limits")
            cap = np.minimum(np.abs(bound[0]), np.abs(bound[1]))
            if np.any(cap <= 0):
                raise VerticalLiftFailure("vertical_lift_nonpositive_dynamic_limit")
            ratio = float(np.max(np.abs(array) / cap))
            ratios.append(ratio ** (1 / exponent))
            dynamics[key] = {"maximum_abs": np.max(np.abs(array), axis=0).tolist(), "limits_abs": cap.tolist(), "maximum_ratio": ratio}
        if max(ratios) > .9:
            duration *= max(1.3, max(ratios) / .8)
            continue
        matrices = native_fk(path)
        receipt = make_receipt(path, matrices, target_world, dt=dt)
        double_matrices = np.stack([chain.state(q, world_from_base)[0] for q in path])
        fk_error = float(np.max(np.abs(matrices - double_matrices)))
        if fk_error > 2e-5:
            raise VerticalLiftFailure("vertical_lift_path_fk_mismatch")
        receipt.update(urdf_sha256=chain.urdf_sha256, native_vs_urdf_max_matrix_error=fk_error,
                       dynamics=dynamics, duration_s=(len(path)-1)*dt, proposal_attempt=attempt+1)
        receipt["validation_hash"] = artifact_hash(receipt, omit_keys=("validation_hash",))
        if not receipt["passed"]:
            raise VerticalLiftFailure("vertical_lift_pose_contract:" + ",".join(receipt["failure_reasons"]))
        return path, velocity.astype(np.float32), receipt
    raise VerticalLiftFailure("vertical_lift_dynamic_limits")


def verify_vertical_lift_receipt(proof, path):
    """Reject missing, legacy or rebound lift evidence at consumer boundaries."""
    if (not isinstance(proof, dict) or proof.get("version") != VERTICAL_LIFT_VERSION
            or proof.get("passed") is not True or proof.get("failure_reasons") != []
            or proof.get("checked_waypoint_count") != len(path)
            or proof.get("trajectory_sha256") != trajectory_summary(path)["trajectory_sha256"]
            or proof.get("xy_tolerance_m") != XY_TOLERANCE_M
            or proof.get("orientation_tolerance_rad") != ORIENTATION_TOLERANCE_RAD
            or proof.get("fk_progress_roundoff_m") != FK_PROGRESS_ROUNDOFF_M
            or proof.get("support_and_collision_thresholds_changed") is not False
            or proof.get("validation_hash") != artifact_hash(proof, omit_keys=("validation_hash",))):
        raise RuntimeError("vertical_lift_proof_missing_or_stale")


def propose_vertical_lift(chain, world_from_base, start_q, target_world, limits, native_fk, *, dt=.004):
    """Acquire by world-Z lift; preserve the original strict pose contract."""
    def admit(delta):
        if (np.linalg.norm(delta[:2]) > XY_TOLERANCE_M / 2
                or np.linalg.norm(delta[3:]) > ORIENTATION_TOLERANCE_RAD / 2 or delta[2] <= 0):
            raise VerticalLiftFailure("vertical_lift_nonvertical_frozen_goal")
    return _propose_cartesian(chain, world_from_base, start_q, target_world, limits,
        native_fk, admit_goal=admit, make_receipt=pose_receipt, dt=dt)
