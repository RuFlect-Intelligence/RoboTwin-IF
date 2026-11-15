"""Bounded Carry descent and collision-checked duplicate-target holds.

Only the path producer changes. Frozen endpoints, support/collision tolerances
and receiver exclusions remain authoritative. A failure has no free-path fallback.
"""
from functools import partial
import numpy as np
from scipy.spatial.transform import Rotation
from if_benchmark.schema import artifact_hash
from .core import trajectory_summary
from .vertical_lift import (
    VerticalLiftFailure, _propose_cartesian, _pose_error,
    XY_TOLERANCE_M, ORIENTATION_TOLERANCE_RAD, FK_PROGRESS_ROUNDOFF_M,
)

CARRY_MOTION_POLICY = {
    "version": "if_carry_deduplicate_cartesian_descent_v1",
    "maximum_start_xy_error_m": .002,
    "maximum_start_orientation_error_rad": float(np.deg2rad(.5)),
    "duplicate_position_tolerance_m": .002,
    "path_position_tolerance_m": XY_TOLERANCE_M,
    "path_orientation_tolerance_rad": ORIENTATION_TOLERANCE_RAD,
    "fk_progress_roundoff_m": FK_PROGRESS_ROUNDOFF_M,
    "free_path_fallback": False,
    "time_progress": "septic_zero_endpoint_velocity_acceleration_jerk",
}


def carry_motion_mode(phase, phase_actor_matrices):
    """Deduplicate equal semantic targets, never merely nearby distinct goals."""
    if phase == "transit":
        return None
    if phase == "approach":
        if np.allclose(phase_actor_matrices["transit"], phase_actor_matrices["approach"],
                       atol=1e-8, rtol=0):
            return "duplicate"
        return "descent"
    if phase == "descent":
        return "descent"
    raise ValueError("unknown_carry_phase")


def carry_pose_receipt(path, matrices, target_world, *, dt, mode):
    q=np.asarray(path); poses=np.asarray(matrices, dtype=float)
    if (mode not in {"duplicate", "descent"} or q.ndim != 2 or len(q)<2
            or poses.shape != (len(q),4,4) or not np.isfinite(poses).all()
            or not np.isfinite(q).all()):
        raise VerticalLiftFailure("carry_cartesian_invalid_path")
    initial=poses[0]; delta=_pose_error(target_world,initial)
    error=_pose_error(target_world,poses[-1]); policy=CARRY_MOTION_POLICY
    xy=np.linalg.norm(poses[:,:2,3]-initial[None,:2,3],axis=1)
    rotations=Rotation.from_matrix(poses[:,:3,:3]@initial[:3,:3].T).magnitude()
    z=poses[:,2,3]; upward=z-np.minimum.accumulate(z)
    failures=[]; line_error=0.; rotation_error=0.
    if np.linalg.norm(delta[:2])>policy["maximum_start_xy_error_m"]:
        failures.append("start_lateral_error")
    if np.linalg.norm(delta[3:])>policy["maximum_start_orientation_error_rad"]:
        failures.append("start_orientation_error")
    if mode=="duplicate":
        if not np.array_equal(q,np.broadcast_to(q[0],q.shape)):
            failures.append("duplicate_target_must_hold")
        if np.linalg.norm(error[:3])>policy["duplicate_position_tolerance_m"]:
            failures.append("duplicate_target_not_reached")
    else:
        if delta[2]>=-FK_PROGRESS_ROUNDOFF_M:
            failures.append("non_descending_goal")
        else:
            progress=np.clip((z-z[0])/delta[2],0,1)
            expected=initial[:3,3]+progress[:,None]*delta[:3]
            line_error=float(np.max(np.linalg.norm(poses[:,:3,3]-expected,axis=1)))
            expected_rotation=Rotation.from_rotvec(progress[:,None]*delta[3:]).as_matrix()@initial[:3,:3]
            rotation_error=float(np.max(Rotation.from_matrix(
                poses[:,:3,:3]@np.swapaxes(expected_rotation,1,2)).magnitude()))
            if line_error>XY_TOLERANCE_M or rotation_error>ORIENTATION_TOLERANCE_RAD:
                failures.append("cartesian_path_deviation")
        if np.max(upward)>FK_PROGRESS_ROUNDOFF_M:
            failures.append("upward_motion")
        if np.linalg.norm(error[:3])>XY_TOLERANCE_M or np.linalg.norm(error[3:])>ORIENTATION_TOLERANCE_RAD:
            failures.append("frozen_endpoint_missed")
    if np.max(xy)>policy["maximum_start_xy_error_m"]+XY_TOLERANCE_M:
        failures.append("lateral_drift")
    if np.max(rotations)>policy["maximum_start_orientation_error_rad"]+ORIENTATION_TOLERANCE_RAD:
        failures.append("orientation_drift")
    receipt={"version":policy["version"],"policy":dict(policy),"passed":not failures,
        "mode":mode,"motion_skipped":mode=="duplicate","failure_reasons":failures,
        "trajectory_sha256":trajectory_summary(q)["trajectory_sha256"],
        "checked_waypoint_count":len(q),"sample_dt_s":dt,
        "reference_start_world":initial.tolist(),"requested_target_world":np.asarray(target_world).tolist(),
        "maximum_xy_drift_m":float(np.max(xy)),"maximum_orientation_drift_rad":float(np.max(rotations)),
        "maximum_upward_excursion_m":float(np.max(upward)),
        "maximum_line_error_m":line_error,"maximum_rotation_interpolation_error_rad":rotation_error,
        "endpoint_position_error_m":float(np.linalg.norm(error[:3])),
        "endpoint_rotation_error_rad":float(np.linalg.norm(error[3:])),
        "support_and_collision_thresholds_changed":False}
    receipt["validation_hash"]=artifact_hash(receipt)
    return receipt


def propose_carry_motion(chain, world_from_base, start_q, target_world, limits, native_fk, *, mode, dt=.004):
    policy=CARRY_MOTION_POLICY
    def admit(delta):
        if (np.linalg.norm(delta[:2])>policy["maximum_start_xy_error_m"]
                or np.linalg.norm(delta[3:])>policy["maximum_start_orientation_error_rad"]):
            raise VerticalLiftFailure("carry_descent_start_outside_corridor")
        if delta[2]>=-FK_PROGRESS_ROUNDOFF_M:
            raise VerticalLiftFailure("carry_non_descending_frozen_goal")
    if mode=="duplicate":
        q=np.asarray(start_q,dtype=np.float32); bounds=np.asarray(limits["position"])
        if (q.ndim!=1 or not len(q) or not np.isfinite(q).all()
                or bounds.shape!=(2,len(q)) or not np.isfinite(bounds).all()
                or np.any(q<bounds[0]) or np.any(q>bounds[1])):
            raise VerticalLiftFailure("carry_hold_invalid_start")
        path=np.stack([q,q]); poses=native_fk(path)
        proof=carry_pose_receipt(path,poses,target_world,dt=dt,mode=mode)
        if not proof["passed"]:
            raise VerticalLiftFailure("carry_duplicate_target_not_reached:"+",".join(proof["failure_reasons"]))
        return path,np.zeros_like(path),proof
    if mode!="descent":
        raise ValueError("unknown_carry_motion_mode")
    return _propose_cartesian(chain,world_from_base,start_q,target_world,limits,native_fk,
        admit_goal=admit,make_receipt=partial(carry_pose_receipt,mode=mode),dt=dt,smooth_start=True)


def verify_carry_motion_receipt(proof,path,*,mode):
    if (not isinstance(proof,dict) or proof.get("version")!=CARRY_MOTION_POLICY["version"]
            or proof.get("policy")!=CARRY_MOTION_POLICY or proof.get("mode")!=mode
            or proof.get("passed") is not True or proof.get("failure_reasons")!=[]
            or proof.get("motion_skipped")!=(mode=="duplicate")
            or proof.get("checked_waypoint_count")!=len(path)
            or proof.get("trajectory_sha256")!=trajectory_summary(path)["trajectory_sha256"]
            or proof.get("support_and_collision_thresholds_changed") is not False
            or proof.get("validation_hash")!=artifact_hash(proof,omit_keys=("validation_hash",))):
        raise RuntimeError("carry_motion_proof_missing_or_stale")
    if mode=="duplicate" and not np.array_equal(path,np.broadcast_to(path[0],np.shape(path))):
        raise RuntimeError("carry_duplicate_path_moved")
