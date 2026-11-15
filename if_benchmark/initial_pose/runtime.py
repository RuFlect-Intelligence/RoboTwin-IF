"""Runtime preflight checks shared by M4/M5/M15 workers."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .core import (
    InitialPoseContractError,
    normalize_quaternion,
    validate_dynamic_mass_properties,
)


def apply_dynamic_mass_properties(
    actor: Any, canonical_state: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply the R01-frozen mass choice through the shared actor interface."""

    contract = canonical_state.get("dynamic_mass_properties")
    if not isinstance(contract, Mapping):
        raise InitialPoseContractError(
            "canonical state lacks dynamic mass contract"
        )
    validate_dynamic_mass_properties(contract)
    if contract["mode"] == "explicit_target":
        actor.set_mass(float(contract["target_mass_kg"]))
    return dict(contract)


def bind_loaded_actor_scale(
    actor: Any, expected_scale_xyz: Sequence[float]
) -> list[float]:
    """Record an explicit scale when legacy model metadata omitted it.

    RoboTwin's rigid loader honors its ``scale`` argument only when the model
    JSON has no scale field, but in that case it also returns ``config=None``.
    Callers invoke this immediately after loading with the exact canonical
    scale, so runtime preflight can observe the scale that built the actor.
    Existing metadata-owned scale is never overwritten and remains available
    for mismatch detection.
    """

    expected = [float(value) for value in expected_scale_xyz]
    if len(expected) == 1:
        expected *= 3
    if len(expected) != 3:
        raise InitialPoseContractError("canonical actor scale must have 3 values")
    config = getattr(actor, "config", None)
    if not isinstance(config, dict) or config.get("scale") is None:
        actor.config = {**(config or {}), "scale": expected}
        return expected
    actual = config["scale"]
    if isinstance(actual, (int, float)):
        return [float(actual)] * 3
    values = [float(value) for value in actual]
    return values * 3 if len(values) == 1 else values


def _multiply(left: Sequence[float], right: Sequence[float]) -> list[float]:
    lw, lx, ly, lz = normalize_quaternion(left)
    rw, rx, ry, rz = normalize_quaternion(right)
    return normalize_quaternion(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ]
    )


def _inverse(value: Sequence[float]) -> list[float]:
    w, x, y, z = normalize_quaternion(value)
    return [w, -x, -y, -z]


def approved_world_yaw_deg(canonical_state: Mapping[str, Any]) -> float:
    """Return the only default yaw that may be used without scene sampling."""

    policy = canonical_state.get("yaw_policy") or {}
    mode = str(policy.get("mode") or "")
    if mode == "free":
        return 0.0
    if mode == "fixed":
        if policy.get("yaw_deg") is None:
            raise InitialPoseContractError("fixed yaw policy lacks yaw_deg")
        return float(policy["yaw_deg"])
    raise InitialPoseContractError(f"invalid approved yaw policy: {mode!r}")


def apply_world_yaw(
    q0_wxyz: Sequence[float], yaw_deg: float
) -> list[float]:
    """Compose an approved world-Z yaw on the left of canonical q0."""

    half = math.radians(float(yaw_deg)) / 2.0
    return _multiply(
        [math.cos(half), 0.0, 0.0, math.sin(half)],
        q0_wxyz,
    )


def validate_loaded_initial_pose(
    *,
    canonical_state: Mapping[str, Any],
    actual_q_wxyz: Sequence[float],
    actual_joint_qpos: Sequence[float] | None,
    actual_scale_xyz: Sequence[float],
    actual_origin_z: float,
    expected_world_yaw_deg: float | None = None,
    quaternion_tolerance_deg: float = 0.05,
    scalar_tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Require exact q0 roll/pitch and only policy-authorized world yaw."""

    q0 = normalize_quaternion(canonical_state["q0_wxyz"])
    actual = normalize_quaternion(actual_q_wxyz)
    relative = _multiply(actual, _inverse(q0))
    # Because yaw is added in world coordinates, actual = qz * q0.  Relative
    # must therefore be a pure world-Z quaternion.
    tilt_residual = math.degrees(
        2.0 * math.asin(min(1.0, math.sqrt(relative[1] ** 2 + relative[2] ** 2)))
    )
    if abs(tilt_residual) > quaternion_tolerance_deg:
        raise InitialPoseContractError(
            f"loaded roll/pitch differs from approved q0 ({tilt_residual} deg)"
        )
    yaw_deg = math.degrees(2.0 * math.atan2(relative[3], relative[0]))
    yaw_deg = ((yaw_deg + 180.0) % 360.0) - 180.0
    policy = canonical_state.get("yaw_policy") or {}
    mode = str(policy.get("mode") or "")
    if mode == "fixed":
        expected = float(
            policy.get("yaw_deg")
            if policy.get("yaw_deg") is not None
            else expected_world_yaw_deg or 0.0
        )
        error = ((yaw_deg - expected + 180.0) % 360.0) - 180.0
        if abs(error) > quaternion_tolerance_deg:
            raise InitialPoseContractError(
                f"loaded yaw {yaw_deg} violates fixed yaw {expected}"
            )
    elif mode == "free":
        if expected_world_yaw_deg is not None:
            error = ((yaw_deg - float(expected_world_yaw_deg) + 180.0) % 360.0) - 180.0
            if abs(error) > quaternion_tolerance_deg:
                raise InitialPoseContractError(
                    f"loaded yaw {yaw_deg} differs from requested yaw {expected_world_yaw_deg}"
                )
    else:
        raise InitialPoseContractError(f"invalid approved yaw policy: {mode!r}")

    expected_scale = [float(value) for value in canonical_state["scale_xyz"]]
    actual_scale = [float(value) for value in actual_scale_xyz]
    if len(actual_scale) != 3 or any(
        abs(left - right) > scalar_tolerance
        for left, right in zip(actual_scale, expected_scale)
    ):
        raise InitialPoseContractError(
            f"loaded scale {actual_scale} differs from approved {expected_scale}"
        )
    expected_z = float((canonical_state.get("support") or {})["origin_z"])
    if abs(float(actual_origin_z) - expected_z) > scalar_tolerance:
        raise InitialPoseContractError(
            f"loaded origin_z {actual_origin_z} differs from approved {expected_z}"
        )
    joint_state = canonical_state.get("articulated_joint_state") or {}
    expected_qpos = [float(value) for value in joint_state.get("qpos") or []]
    actual_qpos = [
        float(value)
        for value in (
            actual_joint_qpos if actual_joint_qpos is not None else []
        )
    ]
    if expected_qpos or actual_qpos:
        if len(expected_qpos) != len(actual_qpos) or any(
            abs(left - right) > scalar_tolerance
            for left, right in zip(actual_qpos, expected_qpos)
        ):
            raise InitialPoseContractError(
                f"loaded joint qpos {actual_qpos} differs from approved {expected_qpos}"
            )
    return {
        "status": "matched",
        "approved_q0_wxyz": q0,
        "actual_q_wxyz": actual,
        "world_yaw_deg": yaw_deg,
        "tilt_residual_deg": tilt_residual,
        "actual_scale_xyz": actual_scale,
        "actual_origin_z": float(actual_origin_z),
        "actual_joint_qpos": actual_qpos,
        "geometry_hash": canonical_state.get("geometry_hash"),
        "revision_hash": canonical_state.get("revision_hash"),
    }
