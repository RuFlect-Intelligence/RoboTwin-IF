"""SAPIEN-collision-derived rigid R01 stable-support revisions.

Candidate generation always stops at an unapproved proposal.  A separate,
explicit benchmark-owner promotion can accept only bounded revisions whose
parent is the currently approved pose and whose seed-zero four-yaw physical
sweep passed.  This narrow path records that no new per-item render review was
performed; it is not a general approval bypass.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from if_benchmark.schema import artifact_hash, deterministic_id

from .core import (
    DEFAULT_ASSET_CATALOG_PATH,
    SUPPORT_CONTRACT_VERSION,
    InitialPoseContractError,
    _append_current_revision,
    _current_revision,
    _current_support_config,
    _dynamic_mass_properties_for_state,
    _load_mapping,
    _proposal_catalog_for_legacy_mass_derivation,
    _revision_record,
    _scale3,
    atomic_json,
    atomic_yaml,
    file_sha256,
    normalize_quaternion,
    validate_dynamic_mass_properties,
    validate_initial_pose_set,
    validate_support_report,
)


STABLE_SUPPORT_DERIVATION_VERSION = (
    "sapien_convex_stable_support_candidate_v1"
)
STABLE_SUPPORT_REPORT_FILE = "stable_support_revision_report.json"
STABLE_SUPPORT_QUICK_REPORTS_DIR = "stable_support_quick_reports"
DEFAULT_MAX_SEMANTIC_TILT_DEG = 12.0
DEFAULT_MAX_CANDIDATES = 12
DEFAULT_SUPPORT_BAND_M = 0.00025
DEFAULT_QUICK_SWEEP_SEED = 0
QUICK_SWEEP_YAWS_DEG = (0.0, 90.0, 180.0, 270.0)
QUICK_SWEEP_EVIDENCE_SCOPE = "unapproved_stable_support_quick_sweep_v1"
BOUNDED_STABLE_SUPPORT_OWNER_POLICY_VERSION = (
    "bounded_stable_support_owner_authorization_v1"
)


def _candidate_physical_trial_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    """Order a screened subset by smallest semantic change first."""

    return (
        float(row["semantic_delta_deg"]),
        -float(row["com_margin_m"]),
        -float(row["support_area_m2"]),
        -float(row["probability"]),
        int(row["stable_index"]),
    )


def _diverse_candidate_subset(
    candidates: Sequence[Mapping[str, Any]],
    *,
    max_candidates: int,
) -> list[dict[str, Any]]:
    """Keep semantic and robustness extrema within the physical-trial budget.

    Stable-pose probability, a vertex-band support polygon, and its COM margin
    are imperfect proxies for PhysX.  Reserving each metric's optimum avoids a
    semantic-only top-k silently dropping the one pose which survives the exact
    four-yaw physical sweep.  Trials remain ordered by semantic delta, so a tiny
    correction wins whenever it is sufficient.
    """

    ordered = sorted(
        (dict(row) for row in candidates), key=_candidate_physical_trial_key
    )
    if len(ordered) <= int(max_candidates):
        return ordered

    representatives = [
        min(ordered, key=_candidate_physical_trial_key),
        max(
            ordered,
            key=lambda row: (
                float(row["com_margin_m"]),
                float(row["support_area_m2"]),
                float(row["probability"]),
                -float(row["semantic_delta_deg"]),
            ),
        ),
        max(
            ordered,
            key=lambda row: (
                float(row["support_area_m2"]),
                float(row["com_margin_m"]),
                float(row["probability"]),
                -float(row["semantic_delta_deg"]),
            ),
        ),
        max(
            ordered,
            key=lambda row: (
                float(row["probability"]),
                float(row["com_margin_m"]),
                float(row["support_area_m2"]),
                -float(row["semantic_delta_deg"]),
            ),
        ),
    ]
    selected: dict[str, dict[str, Any]] = {
        str(row["candidate_id"]): row for row in representatives
    }
    for row in ordered:
        if len(selected) >= int(max_candidates):
            break
        selected.setdefault(str(row["candidate_id"]), row)
    return sorted(selected.values(), key=_candidate_physical_trial_key)


def _q_to_matrix(q_wxyz: Sequence[float]) -> np.ndarray:
    w, x, y, z = normalize_quaternion(q_wxyz)
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _matrix_to_q(matrix: np.ndarray) -> list[float]:
    """Convert one proper rotation matrix to canonical wxyz form."""

    value = np.asarray(matrix, dtype=float)
    if value.shape != (3, 3) or not np.isfinite(value).all():
        raise InitialPoseContractError("stable-support rotation matrix is invalid")
    trace = float(np.trace(value))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        q = [
            0.25 * scale,
            (value[2, 1] - value[1, 2]) / scale,
            (value[0, 2] - value[2, 0]) / scale,
            (value[1, 0] - value[0, 1]) / scale,
        ]
    else:
        axis = int(np.argmax(np.diag(value)))
        if axis == 0:
            scale = math.sqrt(1.0 + value[0, 0] - value[1, 1] - value[2, 2]) * 2.0
            q = [
                (value[2, 1] - value[1, 2]) / scale,
                0.25 * scale,
                (value[0, 1] + value[1, 0]) / scale,
                (value[0, 2] + value[2, 0]) / scale,
            ]
        elif axis == 1:
            scale = math.sqrt(1.0 + value[1, 1] - value[0, 0] - value[2, 2]) * 2.0
            q = [
                (value[0, 2] - value[2, 0]) / scale,
                (value[0, 1] + value[1, 0]) / scale,
                0.25 * scale,
                (value[1, 2] + value[2, 1]) / scale,
            ]
        else:
            scale = math.sqrt(1.0 + value[2, 2] - value[0, 0] - value[1, 1]) * 2.0
            q = [
                (value[1, 0] - value[0, 1]) / scale,
                (value[0, 2] + value[2, 0]) / scale,
                (value[1, 2] + value[2, 1]) / scale,
                0.25 * scale,
            ]
    return normalize_quaternion(q)


def _yaw_align_to_reference(
    candidate: np.ndarray, reference: np.ndarray
) -> np.ndarray:
    """Choose the free world-Z yaw closest to the reviewed reference q0."""

    relative = candidate @ reference.T
    yaw = math.atan2(
        float(relative[1, 0] - relative[0, 1]),
        float(relative[0, 0] + relative[1, 1]),
    )
    cosine = math.cos(-yaw)
    sine = math.sin(-yaw)
    world_yaw = np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    return world_yaw @ candidate


def _semantic_tilt_deg(candidate: np.ndarray, reference: np.ndarray) -> float:
    world_up = np.asarray([0.0, 0.0, 1.0], dtype=float)
    left = candidate.T @ world_up
    right = reference.T @ world_up
    return float(
        math.degrees(
            math.acos(np.clip(float(np.dot(left, right)), -1.0, 1.0))
        )
    )


def _support_patch_metrics(
    vertices: np.ndarray,
    center_mass: np.ndarray,
    rotation: np.ndarray,
    *,
    support_band_m: float,
) -> dict[str, Any]:
    from scipy.spatial import ConvexHull, QhullError

    rotated = np.asarray(vertices, dtype=float) @ rotation.T
    rotated_com = rotation @ np.asarray(center_mass, dtype=float)
    local_z_min = float(np.min(rotated[:, 2]))
    patch = rotated[
        rotated[:, 2] <= local_z_min + float(support_band_m), :2
    ]
    if len(patch) < 3:
        return {
            "local_z_min_m": local_z_min,
            "support_area_m2": 0.0,
            "com_margin_m": -1.0,
            "support_vertex_count": int(len(patch)),
        }
    try:
        hull = ConvexHull(patch)
    except QhullError:
        return {
            "local_z_min_m": local_z_min,
            "support_area_m2": 0.0,
            "com_margin_m": -1.0,
            "support_vertex_count": int(len(patch)),
        }
    equations = np.asarray(hull.equations, dtype=float)
    normal_norm = np.linalg.norm(equations[:, :2], axis=1)
    signed_inside = -(
        equations[:, :2] @ rotated_com[:2] + equations[:, 2]
    ) / np.maximum(normal_norm, 1e-12)
    return {
        "local_z_min_m": local_z_min,
        "support_area_m2": float(hull.volume),
        "com_margin_m": float(np.min(signed_inside)),
        "support_vertex_count": int(len(patch)),
    }


def _collision_mesh_bounds(
    collision_meshes: Sequence[Mapping[str, Any]],
    q_wxyz: Sequence[float],
) -> dict[str, Any]:
    """Measure bounds from the same live post-convex mesh used for poses."""

    rotation = _q_to_matrix(q_wxyz)
    transformed = []
    for raw in collision_meshes:
        vertices = np.asarray(raw.get("vertices"), dtype=float)
        if (
            vertices.ndim != 2
            or vertices.shape[1:] != (3,)
            or not len(vertices)
            or not np.isfinite(vertices).all()
        ):
            raise InitialPoseContractError(
                "stable-support collision mesh bounds are invalid"
            )
        transformed.append(vertices @ rotation.T)
    if not transformed:
        raise InitialPoseContractError(
            "stable-support actor has no collision mesh bounds"
        )
    points = np.concatenate(transformed, axis=0)
    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)
    return {
        "frame": "actor_origin_after_q0_before_world_yaw",
        "min_xyz": [round(float(value), 12) for value in minimum],
        "max_xyz": [round(float(value), 12) for value in maximum],
        "extents_xyz": [
            round(float(maximum[index] - minimum[index]), 12)
            for index in range(3)
        ],
        "source": "sapien_collision_shapes_actor_local_convex_meshes_v1",
    }


def enumerate_stable_support_candidates(
    *,
    collision_meshes: Sequence[Mapping[str, Any]],
    center_mass: Sequence[float],
    q0_wxyz: Sequence[float],
    collision_geometry_hash: str,
    max_semantic_tilt_deg: float = DEFAULT_MAX_SEMANTIC_TILT_DEG,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    support_band_m: float = DEFAULT_SUPPORT_BAND_M,
) -> list[dict[str, Any]]:
    """Enumerate stable faces near q0 in the free-world-yaw quotient.

    Geometry scores only order the physical trials.  They are intentionally
    not treated as proof: PhysX contact resolution can distinguish candidates
    which have nearly identical convex-hull area and COM margin.
    """

    if not math.isfinite(float(max_semantic_tilt_deg)) or not (
        0.0 < float(max_semantic_tilt_deg) <= 45.0
    ):
        raise InitialPoseContractError(
            "stable-support maximum semantic tilt must be within (0, 45] degrees"
        )
    if int(max_candidates) < 4 or int(max_candidates) > 64:
        raise InitialPoseContractError(
            "stable-support candidate count must be within [4, 64] so every "
            "geometry-ordering objective remains represented"
        )
    if not math.isfinite(float(support_band_m)) or not (
        0.0 < float(support_band_m) <= 0.002
    ):
        raise InitialPoseContractError(
            "stable-support contact band must be within (0, 0.002] meters"
        )
    try:
        import trimesh
    except ImportError as exc:  # pragma: no cover - production dependency
        raise InitialPoseContractError(
            "stable-support candidate generation requires trimesh"
        ) from exc

    meshes = []
    all_vertices = []
    for raw in collision_meshes:
        vertices = np.asarray(raw.get("vertices"), dtype=float)
        triangles = np.asarray(raw.get("triangles"), dtype=int)
        if triangles.ndim == 1 and len(triangles) % 3 == 0:
            triangles = triangles.reshape(-1, 3)
        if (
            vertices.ndim != 2
            or vertices.shape[1:] != (3,)
            or triangles.ndim != 2
            or triangles.shape[1:] != (3,)
            or not len(vertices)
            or not len(triangles)
            or not np.isfinite(vertices).all()
            or np.min(triangles) < 0
            or np.max(triangles) >= len(vertices)
        ):
            raise InitialPoseContractError(
                "stable-support collision mesh is invalid"
            )
        meshes.append(
            trimesh.Trimesh(
                vertices=vertices,
                faces=triangles,
                process=False,
            )
        )
        all_vertices.append(vertices)
    if not meshes:
        raise InitialPoseContractError(
            "stable-support actor has no readable collision mesh"
        )
    merged = trimesh.util.concatenate(meshes)
    vertices = np.concatenate(all_vertices, axis=0)
    com = np.asarray(center_mass, dtype=float)
    if com.shape != (3,) or not np.isfinite(com).all():
        raise InitialPoseContractError(
            "stable-support center of mass is invalid"
        )
    transforms, probabilities = trimesh.poses.compute_stable_poses(
        merged,
        center_mass=com,
        sigma=0.0,
        n_samples=1,
        threshold=0.0,
    )
    reference = _q_to_matrix(q0_wxyz)
    unique_up: list[np.ndarray] = []
    candidates: list[dict[str, Any]] = []
    for stable_index, (transform, probability) in enumerate(
        zip(transforms, probabilities)
    ):
        rotation = _yaw_align_to_reference(
            np.asarray(transform[:3, :3], dtype=float), reference
        )
        local_up = rotation.T @ np.asarray([0.0, 0.0, 1.0])
        if any(
            math.degrees(
                math.acos(np.clip(float(np.dot(local_up, previous)), -1.0, 1.0))
            )
            < 0.05
            for previous in unique_up
        ):
            continue
        unique_up.append(local_up)
        semantic_delta = _semantic_tilt_deg(rotation, reference)
        if semantic_delta > float(max_semantic_tilt_deg) + 1e-9:
            continue
        metrics = _support_patch_metrics(
            vertices,
            com,
            rotation,
            support_band_m=float(support_band_m),
        )
        q = _matrix_to_q(rotation)
        payload = {
            "derivation_version": STABLE_SUPPORT_DERIVATION_VERSION,
            "collision_geometry_hash": str(collision_geometry_hash),
            "stable_index": int(stable_index),
            "q0_wxyz": q,
            "semantic_delta_deg": round(float(semantic_delta), 12),
            "local_up": [round(float(value), 12) for value in local_up],
            "probability": round(float(probability), 12),
            "support_band_m": round(float(support_band_m), 12),
            **{
                key: round(float(value), 12)
                if isinstance(value, (float, np.floating))
                else int(value)
                for key, value in metrics.items()
            },
        }
        payload["candidate_id"] = deterministic_id(
            "issc", payload, length=20
        )
        payload["candidate_hash"] = artifact_hash(payload)
        candidates.append(payload)
    return _diverse_candidate_subset(
        candidates,
        max_candidates=int(max_candidates),
    )


def extract_sapien_rigid_support_geometry(
    *,
    asset: str,
    model_id: str,
    scale_xyz: Sequence[float],
    dynamic_mass_properties: Mapping[str, Any],
) -> dict[str, Any]:
    """Load one rigid actor and copy its authoritative post-convex meshes."""

    validate_dynamic_mass_properties(dynamic_mass_properties)
    try:
        import sapien
        from envs.utils.create_actor import create_actor
        from if_benchmark.collision_geometry import collision_shape_local_meshes
    except ImportError as exc:  # pragma: no cover - target runtime dependency
        raise InitialPoseContractError(
            "stable-support geometry extraction requires the SAPIEN runtime"
        ) from exc

    scene = sapien.Scene()
    actor = create_actor(
        scene=scene,
        pose=sapien.Pose(),
        modelname=str(asset),
        scale=[float(value) for value in scale_xyz],
        model_id=int(model_id) if str(model_id).isdigit() else str(model_id),
        convex=True,
        mass=None,
    )
    if actor is None:
        raise InitialPoseContractError(
            f"stable-support actor load failed: {asset}/{model_id}"
        )
    expected_scale = _scale3(scale_xyz)
    actor_config = getattr(actor, "config", None)
    actual_scale = (
        _scale3(actor_config.get("scale"))
        if isinstance(actor_config, Mapping)
        and actor_config.get("scale") is not None
        else expected_scale
    )
    if any(abs(left - right) > 1e-9 for left, right in zip(actual_scale, expected_scale)):
        raise InitialPoseContractError(
            f"stable-support actor scale mismatch: {actual_scale} != {expected_scale}"
        )
    if dynamic_mass_properties["mode"] == "explicit_target":
        actor.set_mass(float(dynamic_mass_properties["target_mass_kg"]))
    components = [
        component
        for component in actor.actor.get_components()
        if callable(getattr(component, "get_collision_shapes", None))
        and component.get_collision_shapes()
        and hasattr(component, "cmass_local_pose")
    ]
    if len(components) != 1:
        raise InitialPoseContractError(
            "stable-support rigid actor must have one collision-bearing dynamic component"
        )
    component = components[0]
    copied_meshes = []
    for row in collision_shape_local_meshes(component):
        copied_meshes.append(
            {
                "vertices": np.asarray(row["vertices"], dtype=float).copy(),
                "triangles": np.asarray(row["triangles"], dtype=int).copy(),
            }
        )
    center_mass = np.asarray(component.cmass_local_pose.p, dtype=float)
    geometry_payload = {
        "source": "sapien_collision_shapes_actor_local_convex_meshes_v1",
        "asset": str(asset),
        "model_id": str(model_id),
        "scale_xyz": [round(float(value), 12) for value in expected_scale],
        "meshes": [
            {
                "vertices": np.round(row["vertices"], 12).tolist(),
                "triangles": row["triangles"].tolist(),
            }
            for row in copied_meshes
        ],
        "center_mass": np.round(center_mass, 12).tolist(),
    }
    geometry_hash = artifact_hash(geometry_payload)
    return {
        "collision_meshes": copied_meshes,
        "center_mass": center_mass,
        "collision_geometry_hash": geometry_hash,
        "source": geometry_payload["source"],
        "shape_count": len(copied_meshes),
        "mass_properties": {
            "mass_kg": float(component.mass),
            "principal_inertia_kg_m2": np.asarray(
                component.inertia, dtype=float
            ).tolist(),
            "cmass_local_pose_p": center_mass.tolist(),
        },
    }


def _quick_support_jobs(
    *,
    state: Mapping[str, Any],
    parent: Mapping[str, Any],
    candidate: Mapping[str, Any],
    bounds: Mapping[str, Any],
    dynamic_mass_properties: Mapping[str, Any],
    seed: int,
    task_config: str,
) -> tuple[str, list[dict[str, Any]]]:
    if int(seed) != DEFAULT_QUICK_SWEEP_SEED:
        raise InitialPoseContractError(
            "stable-support physical prefilter is fixed to seed 0"
        )
    support_config = _current_support_config()
    support_config_hash = artifact_hash(support_config)
    support = {
        "plane_z": float(parent["support"]["plane_z"]),
        "clearance_m": float(parent["support"]["clearance_m"]),
        "origin_z": (
            float(parent["support"]["plane_z"])
            - float(bounds["min_xyz"][2])
            + float(parent["support"]["clearance_m"])
        ),
        "formula": "support_z-local_z_min+support_clearance",
    }
    binding = {
        "initial_state_id": str(state["initial_state_id"]),
        "parent_revision_hash": str(parent["revision_hash"]),
        "candidate_hash": str(candidate["candidate_hash"]),
        "q0_wxyz": list(candidate["q0_wxyz"]),
        "scale_xyz": list(parent["scale_xyz"]),
        "support": support,
        "dynamic_mass_properties": dict(dynamic_mass_properties),
        "support_config_hash": support_config_hash,
    }
    quick_plan_hash = artifact_hash(
        {
            "artifact_kind": "stable_support_quick_sweep_plan",
            "derivation_version": STABLE_SUPPORT_DERIVATION_VERSION,
            "binding": binding,
            "seed": int(seed),
            "yaw_degrees": list(QUICK_SWEEP_YAWS_DEG),
            "task_config": str(task_config),
        }
    )
    jobs = []
    candidate_geometry_hash = artifact_hash(
        {
            "candidate_hash": candidate["candidate_hash"],
            "bounds": dict(bounds),
            "scale_xyz": list(parent["scale_xyz"]),
        }
    )
    unapproved_binding_hash = artifact_hash(
        {
            "evidence_scope": QUICK_SWEEP_EVIDENCE_SCOPE,
            "candidate_hash": candidate["candidate_hash"],
            "quick_plan_hash": quick_plan_hash,
        }
    )
    for yaw_deg in QUICK_SWEEP_YAWS_DEG:
        job = {
            "job_id": deterministic_id(
                "ipq",
                {
                    "quick_plan_hash": quick_plan_hash,
                    "yaw_deg": yaw_deg,
                    "seed": int(seed),
                },
                length=20,
            ),
            "initial_state_id": str(state["initial_state_id"]),
            "revision": int(parent["revision"]) + 1,
            # The shared support worker requires this legacy field.  Bind it
            # to an explicitly unapproved scope rather than impersonating a
            # human approval hash; the persisted wrapper has its own schema.
            "approval_hash": unapproved_binding_hash,
            "candidate_hash": str(candidate["candidate_hash"]),
            "evidence_scope": QUICK_SWEEP_EVIDENCE_SCOPE,
            "geometry_hash": candidate_geometry_hash,
            "dynamic_mass_properties": dict(dynamic_mass_properties),
            "asset": str(state["asset"]),
            "model_id": str(state["model_id"]),
            "initial_state_profile_id": str(state["initial_state_profile_id"]),
            "q0_wxyz": list(candidate["q0_wxyz"]),
            "articulated_joint_state": dict(parent["articulated_joint_state"]),
            "scale_xyz": list(parent["scale_xyz"]),
            "support": support,
            "yaw_policy": dict(parent["yaw_policy"]),
            "yaw_deg": float(yaw_deg),
            "seed": int(seed),
            "required_metrics": list(support_config["required_metrics"]),
            "support_contract_version": SUPPORT_CONTRACT_VERSION,
            "support_config_hash": support_config_hash,
            "support_mode": support_config["root_modes"]["rigid"],
            "task_config": str(task_config),
        }
        jobs.append(job)
    return quick_plan_hash, jobs


def _quick_report_wrapper(
    *,
    job: Mapping[str, Any],
    quick_plan_hash: str,
    support_report: Mapping[str, Any],
) -> dict[str, Any]:
    portable_report = dict(support_report)
    portable_report.pop("report_hash", None)
    error = portable_report.get("error")
    if isinstance(error, Mapping):
        # Traceback filenames contain host checkout roots.  Preserve the
        # stable failure identity without allowing those roots into revision
        # lineage.
        portable_report["error"] = {
            "type": error.get("type"),
            "message": error.get("message"),
        }
    portable_support_report_hash = artifact_hash(portable_report)
    wrapper: dict[str, Any] = {
        "artifact_kind": "initial_pose_stable_support_quick_report",
        "schema_version": 1,
        "derivation_version": STABLE_SUPPORT_DERIVATION_VERSION,
        "evidence_scope": QUICK_SWEEP_EVIDENCE_SCOPE,
        "quick_plan_hash": str(quick_plan_hash),
        "job_id": str(job["job_id"]),
        "initial_state_id": str(job["initial_state_id"]),
        "candidate_hash": str(job["candidate_hash"]),
        "yaw_deg": float(job["yaw_deg"]),
        "seed": int(job["seed"]),
        "outcome": str(support_report["outcome"]),
        "support_report_hash": str(support_report["report_hash"]),
        "portable_support_report_hash": portable_support_report_hash,
        "support_report": dict(support_report),
    }
    wrapper["report_hash"] = artifact_hash(wrapper)
    return wrapper


def _validate_quick_report_wrapper(
    wrapper: Mapping[str, Any],
    *,
    job: Mapping[str, Any],
    quick_plan_hash: str,
) -> dict[str, Any]:
    if wrapper.get("artifact_kind") != "initial_pose_stable_support_quick_report":
        raise InitialPoseContractError(
            "stable-support quick report has unexpected artifact kind"
        )
    if wrapper.get("derivation_version") != STABLE_SUPPORT_DERIVATION_VERSION:
        raise InitialPoseContractError(
            "stable-support quick report derivation is stale"
        )
    if wrapper.get("evidence_scope") != QUICK_SWEEP_EVIDENCE_SCOPE:
        raise InitialPoseContractError(
            "stable-support quick report scope mismatch"
        )
    if wrapper.get("report_hash") != artifact_hash(
        wrapper, omit_keys=("report_hash",)
    ):
        raise InitialPoseContractError(
            "stable-support quick report self-hash mismatch"
        )
    expected = {
        "quick_plan_hash": str(quick_plan_hash),
        "job_id": str(job["job_id"]),
        "initial_state_id": str(job["initial_state_id"]),
        "candidate_hash": str(job["candidate_hash"]),
        "yaw_deg": float(job["yaw_deg"]),
        "seed": int(job["seed"]),
    }
    for key, value in expected.items():
        if wrapper.get(key) != value:
            raise InitialPoseContractError(
                f"stable-support quick report {key} mismatch"
            )
    support_report = wrapper.get("support_report")
    if not isinstance(support_report, Mapping):
        raise InitialPoseContractError(
            "stable-support quick wrapper lacks support report"
        )
    validate_support_report(
        support_report,
        job,
        expected_plan_hash=quick_plan_hash,
    )
    if wrapper.get("support_report_hash") != support_report.get("report_hash"):
        raise InitialPoseContractError(
            "stable-support nested support report hash mismatch"
        )
    portable_report = dict(support_report)
    portable_report.pop("report_hash", None)
    error = portable_report.get("error")
    if isinstance(error, Mapping):
        portable_report["error"] = {
            "type": error.get("type"),
            "message": error.get("message"),
        }
    if wrapper.get("portable_support_report_hash") != artifact_hash(
        portable_report
    ):
        raise InitialPoseContractError(
            "stable-support portable support report hash mismatch"
        )
    if wrapper.get("outcome") != support_report.get("outcome"):
        raise InitialPoseContractError(
            "stable-support nested support outcome mismatch"
        )
    return dict(support_report)


def run_candidate_quick_sweep(
    *,
    state: Mapping[str, Any],
    parent: Mapping[str, Any],
    candidate: Mapping[str, Any],
    bounds: Mapping[str, Any],
    dynamic_mass_properties: Mapping[str, Any],
    reports_dir: Path,
    runtime_dir: Path,
    task_config: str,
    seed: int = DEFAULT_QUICK_SWEEP_SEED,
) -> dict[str, Any]:
    """Run and persist the four-yaw seed-zero physical prefilter."""

    from .support_worker import run_support_job

    quick_plan_hash, jobs = _quick_support_jobs(
        state=state,
        parent=parent,
        candidate=candidate,
        bounds=bounds,
        dynamic_mass_properties=dynamic_mass_properties,
        seed=int(seed),
        task_config=str(task_config),
    )
    candidate_dir = reports_dir / str(candidate["candidate_id"])
    candidate_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for job in jobs:
        path = candidate_dir / f"{job['job_id']}.json"
        report = None
        wrapper = None
        if path.is_file():
            existing = _load_mapping(path)
            try:
                existing_report = _validate_quick_report_wrapper(
                    existing,
                    job=job,
                    quick_plan_hash=quick_plan_hash,
                )
                # Runtime/infrastructure failures are not immutable physical
                # evidence.  A repeated command must actually retry them.
                if existing_report.get("outcome") not in {
                    "source_failure",
                    "internal_failure",
                }:
                    report = existing_report
                    wrapper = existing
            except InitialPoseContractError:
                report = None
        if report is None:
            report = run_support_job(
                job=job,
                plan_hash=quick_plan_hash,
                runtime_dir=runtime_dir
                / str(candidate["candidate_id"])
                / str(job["job_id"]),
                task_config=task_config,
            )
            validate_support_report(
                report,
                job,
                expected_plan_hash=quick_plan_hash,
            )
            wrapper = _quick_report_wrapper(
                job=job,
                quick_plan_hash=quick_plan_hash,
                support_report=report,
            )
            atomic_json(path, wrapper)
        if wrapper is None:  # pragma: no cover - defensive invariant
            raise InitialPoseContractError(
                "stable-support quick wrapper was not materialized"
            )
        outcome = str(report["outcome"])
        if outcome in {"source_failure", "internal_failure"}:
            raise InitialPoseContractError(
                "stable-support quick sweep infrastructure failure: "
                f"{job['job_id']}:{outcome}:{(report.get('error') or {}).get('message')}"
            )
        metrics = report.get("metrics") or {}
        rows.append(
            {
                "job_id": str(job["job_id"]),
                "yaw_deg": float(job["yaw_deg"]),
                "seed": int(job["seed"]),
                "outcome": outcome,
                # This evidence is embedded in a canonical revision.  Keep it
                # independent of the host-specific output root.
                "report_file": (
                    f"{candidate['candidate_id']}/{job['job_id']}.json"
                ),
                "portable_support_report_hash": str(
                    wrapper["portable_support_report_hash"]
                ),
                "failed_checks": list(metrics.get("failed_checks") or []),
                "spawn_to_settled_translation_m": metrics.get(
                    "spawn_to_settled_translation_m"
                ),
                "spawn_to_settled_rotation_deg": metrics.get(
                    "spawn_to_settled_rotation_deg"
                ),
                "tail_translation_drift_m": metrics.get(
                    "tail_translation_drift_m"
                ),
                "tail_rotation_drift_deg": metrics.get(
                    "tail_rotation_drift_deg"
                ),
            }
        )
    result = {
        "evidence_scope": QUICK_SWEEP_EVIDENCE_SCOPE,
        "quick_plan_hash": quick_plan_hash,
        "seed": int(seed),
        "required_yaws_deg": list(QUICK_SWEEP_YAWS_DEG),
        "attempts": rows,
        "all_pass": len(rows) == len(QUICK_SWEEP_YAWS_DEG)
        and all(row["outcome"] == "pass" for row in rows),
    }
    result["quick_sweep_hash"] = artifact_hash(result)
    return result


def select_stable_support_candidate(
    candidates: Sequence[Mapping[str, Any]],
    quick_sweep: Callable[[Mapping[str, Any]], Mapping[str, Any]],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Return the first semantic-minimal candidate passing every quick yaw."""

    trials = []
    for raw_candidate in candidates:
        candidate = dict(raw_candidate)
        evidence = dict(quick_sweep(candidate))
        if evidence.get("quick_sweep_hash") != artifact_hash(
            evidence, omit_keys=("quick_sweep_hash",)
        ):
            raise InitialPoseContractError(
                "stable-support quick sweep self-hash mismatch"
            )
        if evidence.get("evidence_scope") != QUICK_SWEEP_EVIDENCE_SCOPE:
            raise InitialPoseContractError(
                "stable-support quick sweep evidence scope mismatch"
            )
        if int(evidence.get("seed", -1)) != DEFAULT_QUICK_SWEEP_SEED:
            raise InitialPoseContractError(
                "stable-support quick sweep must use seed 0"
            )
        required_yaws = [float(value) for value in evidence.get("required_yaws_deg") or []]
        attempts = list(evidence.get("attempts") or [])
        observed = sorted(float(row.get("yaw_deg")) for row in attempts)
        if sorted(required_yaws) != sorted(QUICK_SWEEP_YAWS_DEG) or observed != sorted(
            QUICK_SWEEP_YAWS_DEG
        ):
            raise InitialPoseContractError(
                "stable-support quick sweep lacks exact four-yaw coverage"
            )
        if any(
            int(row.get("seed", -1)) != DEFAULT_QUICK_SWEEP_SEED
            for row in attempts
        ):
            raise InitialPoseContractError(
                "stable-support quick sweep attempt seed mismatch"
            )
        if any(
            not str(row.get("portable_support_report_hash") or "")
            for row in attempts
        ):
            raise InitialPoseContractError(
                "stable-support quick sweep lacks portable report identity"
            )
        attempts_all_pass = all(
            str(row.get("outcome")) == "pass" for row in attempts
        )
        if bool(evidence.get("all_pass")) != attempts_all_pass:
            raise InitialPoseContractError(
                "stable-support quick sweep pass summary contradicts attempts"
            )
        trial = {
            "candidate_id": candidate["candidate_id"],
            "candidate_hash": candidate["candidate_hash"],
            "semantic_delta_deg": candidate["semantic_delta_deg"],
            "quick_sweep_hash": evidence.get("quick_sweep_hash"),
            "all_pass": attempts_all_pass,
            "attempts": attempts,
        }
        trials.append(trial)
        if trial["all_pass"]:
            return {**candidate, "quick_sweep": evidence}, trials
    return None, trials


def materialize_stable_support_revisions(
    *,
    proposal_path: Path,
    state_ids: set[str],
    output_report_path: Path,
    quick_reports_dir: Path,
    runtime_dir: Path,
    task_config: str = "demo_clean",
    max_semantic_tilt_deg: float = DEFAULT_MAX_SEMANTIC_TILT_DEG,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    quick_seed: int = DEFAULT_QUICK_SWEEP_SEED,
    geometry_provider: Callable[..., Mapping[str, Any]] | None = None,
    quick_sweep_runner: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Append physically screened, explicitly unapproved rigid revisions."""

    if int(quick_seed) != DEFAULT_QUICK_SWEEP_SEED:
        raise InitialPoseContractError(
            "stable-support physical prefilter is fixed to seed 0"
        )
    if not state_ids:
        raise InitialPoseContractError(
            "stable-support revision requires at least one explicit state id"
        )
    proposal = _load_mapping(proposal_path)
    if proposal.get("proposal_set_hash") != artifact_hash(
        proposal, omit_keys=("proposal_set_hash",)
    ):
        raise InitialPoseContractError("Initial Pose proposal set self-hash mismatch")
    state_rows = list(proposal.get("states") or [])
    state_keys = [str(row["initial_state_id"]) for row in state_rows]
    if len(state_keys) != len(set(state_keys)):
        raise InitialPoseContractError(
            "Initial Pose proposal contains duplicate state ids"
        )
    for row in state_rows:
        if row.get("state_hash") != artifact_hash(
            row, omit_keys=("state_hash",)
        ):
            raise InitialPoseContractError(
                f"Initial Pose state self-hash mismatch: {row.get('initial_state_id')}"
            )
    states = {str(row["initial_state_id"]): row for row in state_rows}
    unknown = sorted(set(state_ids) - set(states))
    if unknown:
        raise InitialPoseContractError(
            f"unknown stable-support state ids: {unknown}"
        )
    catalog_path = Path(
        str(
            (proposal.get("source_paths") or {}).get("asset_catalog")
            or DEFAULT_ASSET_CATALOG_PATH
        )
    )
    catalog = _load_mapping(catalog_path)
    geometry_fn = geometry_provider or extract_sapien_rigid_support_geometry
    sweep_fn = quick_sweep_runner or run_candidate_quick_sweep
    revisions = []
    unresolved = []
    implementation_hash = file_sha256(Path(__file__))
    for state_id in sorted(state_ids):
        state = states[state_id]
        parent = _current_revision(state)
        if (parent.get("articulated_joint_state") or {}).get("mode") != "not_articulated":
            raise InitialPoseContractError(
                f"stable-support revision is rigid-only: {state_id}"
            )
        if (parent.get("yaw_policy") or {}).get("mode") != "free":
            raise InitialPoseContractError(
                f"stable-support four-yaw revision requires free yaw: {state_id}"
            )
        if (parent.get("source") or {}).get("kind") == STABLE_SUPPORT_DERIVATION_VERSION:
            raise InitialPoseContractError(
                f"current revision is already a stable-support proposal: {state_id}"
            )
        dynamic_mass = _dynamic_mass_properties_for_state(state, catalog)
        geometry = dict(
            geometry_fn(
                asset=str(state["asset"]),
                model_id=str(state["model_id"]),
                scale_xyz=parent["scale_xyz"],
                dynamic_mass_properties=dynamic_mass,
            )
        )
        candidates = enumerate_stable_support_candidates(
            collision_meshes=geometry["collision_meshes"],
            center_mass=geometry["center_mass"],
            q0_wxyz=parent["q0_wxyz"],
            collision_geometry_hash=str(geometry["collision_geometry_hash"]),
            max_semantic_tilt_deg=max_semantic_tilt_deg,
            max_candidates=max_candidates,
        )
        candidate_bounds = {
            str(candidate["candidate_id"]): _collision_mesh_bounds(
                geometry["collision_meshes"],
                candidate["q0_wxyz"],
            )
            for candidate in candidates
        }

        def evaluate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
            return sweep_fn(
                state=state,
                parent=parent,
                candidate=candidate,
                bounds=candidate_bounds[str(candidate["candidate_id"])],
                dynamic_mass_properties=dynamic_mass,
                reports_dir=quick_reports_dir,
                runtime_dir=runtime_dir,
                task_config=task_config,
                seed=int(quick_seed),
            )

        selected, trials = select_stable_support_candidate(candidates, evaluate)
        if selected is None:
            unresolved.append(
                {
                    "initial_state_id": state_id,
                    "asset": state["asset"],
                    "model_id": state["model_id"],
                    "reason": (
                        "no_semantic_neighborhood_candidate"
                        if not candidates
                        else "no_candidate_passed_four_yaw_quick_sweep"
                    ),
                    "candidate_count": len(candidates),
                    "trials": trials,
                }
            )
            continue
        bounds = candidate_bounds[str(selected["candidate_id"])]
        quick_evidence = dict(selected.pop("quick_sweep"))
        source_hashes = {
            **dict(parent.get("source_hashes") or {}),
            "stable_support_implementation": implementation_hash,
            "stable_support_collision_geometry": str(
                geometry["collision_geometry_hash"]
            ),
            "stable_support_candidate": str(selected["candidate_hash"]),
            "stable_support_quick_sweep": str(
                quick_evidence["quick_sweep_hash"]
            ),
        }
        source = {
            "kind": STABLE_SUPPORT_DERIVATION_VERSION,
            "parent_revision_hash": parent["revision_hash"],
            "collision_geometry_source": geometry.get("source"),
            "collision_geometry_hash": geometry["collision_geometry_hash"],
            "mass_properties": geometry.get("mass_properties"),
            "candidate": selected,
            "candidate_count": len(candidates),
            "max_semantic_tilt_deg": float(max_semantic_tilt_deg),
            "quick_sweep": quick_evidence,
            "quick_sweep_trials": trials,
            "evidence_only_not_approval": True,
            "requires_four_view_human_review": True,
        }
        revision = _revision_record(
            state_id=state_id,
            revision=int(parent["revision"]) + 1,
            parent_revision=int(parent["revision"]),
            q0_wxyz=selected["q0_wxyz"],
            articulated_joint_state=parent["articulated_joint_state"],
            yaw_policy=parent["yaw_policy"],
            scale=parent["scale_xyz"],
            bounds=bounds,
            support_z=float(parent["support"]["plane_z"]),
            support_clearance_m=float(parent["support"]["clearance_m"]),
            source=source,
            source_hashes=source_hashes,
            rotation_sequence=(),
            original_request=(
                "SAPIEN collision stable-support candidate; unapproved until rerendered and human-reviewed"
            ),
        )
        # A newly materialized physical revision is also the migration point
        # for legacy proposal states that predate the mass/inertia field.  Keep
        # the exact contract used for geometry extraction and replay so later
        # stages never need to infer it again.
        state["dynamic_mass_properties"] = dict(dynamic_mass)
        _append_current_revision(state, revision)
        revisions.append(
            {
                "initial_state_id": state_id,
                "asset": state["asset"],
                "model_id": state["model_id"],
                "parent_revision": parent["revision"],
                "revision": revision["revision"],
                "revision_hash": revision["revision_hash"],
                "candidate_id": selected["candidate_id"],
                "candidate_hash": selected["candidate_hash"],
                "semantic_delta_deg": selected["semantic_delta_deg"],
                "quick_sweep_hash": quick_evidence["quick_sweep_hash"],
                "quick_evidence_hashes": sorted(
                    str(row["portable_support_report_hash"])
                    for row in quick_evidence["attempts"]
                ),
                "approval_status": "unapproved_requires_human_review",
            }
        )
    if revisions:
        proposal["states"] = list(states.values())
        proposal.pop("proposal_set_hash", None)
        proposal["proposal_set_hash"] = artifact_hash(proposal)
    report: dict[str, Any] = {
        "artifact_kind": "initial_pose_stable_support_revision_report",
        "schema_version": 1,
        "derivation_version": STABLE_SUPPORT_DERIVATION_VERSION,
        "source_proposal_path": str(proposal_path.resolve()),
        "proposal_set_hash": proposal["proposal_set_hash"],
        "requested_state_ids": sorted(state_ids),
        "configuration": {
            "max_semantic_tilt_deg": float(max_semantic_tilt_deg),
            "max_candidates": int(max_candidates),
            "quick_seed": int(quick_seed),
            "quick_yaws_deg": list(QUICK_SWEEP_YAWS_DEG),
            "task_config": str(task_config),
            "quick_reports_dir": str(quick_reports_dir.resolve()),
        },
        "summary": {
            "requested_state_count": len(state_ids),
            "revised_state_count": len(revisions),
            "unresolved_state_count": len(unresolved),
        },
        "revisions": revisions,
        "unresolved": unresolved,
        "writes_human_review": False,
        "writes_approved_pose_set": False,
    }
    report["report_hash"] = artifact_hash(report)
    if revisions:
        atomic_yaml(proposal_path, proposal)
    atomic_json(output_report_path, report)
    return report


def promote_bounded_stable_support_revisions(
    *,
    proposal_path: Path,
    base_approved_set_path: Path,
    output_path: Path,
    state_ids: set[str],
    authority: str,
    maximum_semantic_delta_deg: float = DEFAULT_MAX_SEMANTIC_TILT_DEG,
) -> dict[str, Any]:
    """Promote explicit bounded support revisions under benchmark-owner authority.

    The old approval remains the semantic/appearance judgment.  The overlay
    authorizes only a small support-stability correction backed by the exact
    seed-zero four-yaw physical sweep embedded in the proposal revision.
    """

    owner = str(authority).strip()
    if not owner:
        raise InitialPoseContractError(
            "bounded stable-support promotion requires a named authority"
        )
    if not state_ids:
        raise InitialPoseContractError(
            "bounded stable-support promotion requires explicit state ids"
        )
    maximum_delta = float(maximum_semantic_delta_deg)
    if (
        not math.isfinite(maximum_delta)
        or maximum_delta <= 0.0
        or maximum_delta > DEFAULT_MAX_SEMANTIC_TILT_DEG
    ):
        raise InitialPoseContractError(
            "bounded stable-support promotion exceeds the 12 degree policy"
        )

    proposal = _load_mapping(proposal_path)
    if proposal.get("proposal_set_hash") != artifact_hash(
        proposal, omit_keys=("proposal_set_hash",)
    ):
        raise InitialPoseContractError("Initial Pose proposal set self-hash mismatch")
    base = _load_mapping(base_approved_set_path)
    validate_initial_pose_set(base, require_support_validated=False)

    proposal_rows = list(proposal.get("states") or [])
    proposal_ids = [str(row.get("initial_state_id")) for row in proposal_rows]
    if len(proposal_ids) != len(set(proposal_ids)):
        raise InitialPoseContractError(
            "Initial Pose proposal contains duplicate state ids"
        )
    proposal_by_id = {
        str(row["initial_state_id"]): row for row in proposal_rows
    }
    base_states = {
        str(row["initial_state_id"]): dict(row)
        for row in base.get("canonical_initial_states") or []
    }
    base_approvals = {
        str(row["initial_state_id"]): dict(row)
        for row in base.get("approvals") or []
    }
    missing = sorted(
        (set(state_ids) - set(proposal_by_id))
        | (set(state_ids) - set(base_states))
    )
    if missing:
        raise InitialPoseContractError(
            f"bounded stable-support promotion lacks states: {missing}"
        )

    catalog, catalog_hash = _proposal_catalog_for_legacy_mass_derivation(
        proposal
    )
    authorizations = []
    for state_id in sorted(state_ids):
        state = proposal_by_id[state_id]
        if state.get("state_hash") != artifact_hash(
            state, omit_keys=("state_hash",)
        ):
            raise InitialPoseContractError(
                f"Initial Pose state self-hash mismatch: {state_id}"
            )
        revision = _current_revision(state)
        source = revision.get("source") or {}
        if source.get("kind") != STABLE_SUPPORT_DERIVATION_VERSION:
            raise InitialPoseContractError(
                f"state is not a current stable-support revision: {state_id}"
            )
        old = base_states[state_id]
        old_approval = base_approvals[state_id]
        if old.get("status") != "approved" or old_approval.get("decision") != "approved":
            raise InitialPoseContractError(
                f"bounded stable-support parent is not approved: {state_id}"
            )
        if int(revision.get("parent_revision", -1)) != int(old["revision"]):
            raise InitialPoseContractError(
                f"bounded stable-support parent revision mismatch: {state_id}"
            )
        if source.get("parent_revision_hash") != old.get("revision_hash"):
            raise InitialPoseContractError(
                f"bounded stable-support parent hash mismatch: {state_id}"
            )
        for key in ("asset", "model_id", "initial_state_profile_id"):
            if str(state.get(key)) != str(old.get(key)):
                raise InitialPoseContractError(
                    f"bounded stable-support identity mismatch: {state_id}:{key}"
                )
        candidate = source.get("candidate") or {}
        semantic_delta = float(candidate.get("semantic_delta_deg", math.inf))
        if not math.isfinite(semantic_delta) or semantic_delta > maximum_delta:
            raise InitialPoseContractError(
                f"bounded stable-support semantic delta exceeds policy: {state_id}"
            )
        quick_sweep = source.get("quick_sweep")
        if not isinstance(quick_sweep, Mapping):
            raise InitialPoseContractError(
                f"bounded stable-support quick sweep missing: {state_id}"
            )
        selected, _ = select_stable_support_candidate(
            [candidate], lambda _candidate: quick_sweep
        )
        if selected is None:
            raise InitialPoseContractError(
                f"bounded stable-support quick sweep did not pass: {state_id}"
            )

        dynamic_mass = (
            dict(state["dynamic_mass_properties"])
            if isinstance(state.get("dynamic_mass_properties"), Mapping)
            else _dynamic_mass_properties_for_state(state, catalog)
        )
        validate_dynamic_mass_properties(dynamic_mass)
        canonical = {
            "artifact_kind": "CanonicalInitialState",
            "schema_version": int(base.get("schema_version") or 1),
            "initial_state_id": state_id,
            "asset": state["asset"],
            "model_id": state["model_id"],
            "initial_state_profile_id": state["initial_state_profile_id"],
            "revision": revision["revision"],
            "revision_hash": revision["revision_hash"],
            "q0_wxyz": revision["q0_wxyz"],
            "articulated_joint_state": revision["articulated_joint_state"],
            "yaw_policy": revision["yaw_policy"],
            "scale_xyz": revision["scale_xyz"],
            "origin_relative_bounds": revision["origin_relative_bounds"],
            "support": revision["support"],
            "geometry_hash": revision["geometry_hash"],
            "candidate_roles": list(state.get("candidate_roles") or []),
            "candidate_uses": list(state.get("candidate_uses") or []),
            "dynamic_mass_properties": dynamic_mass,
            "assignment_eligible": bool(state.get("assignment_eligible")),
            "status": "approved",
        }
        canonical["canonical_state_hash"] = artifact_hash(canonical)

        authorization = {
            "artifact_kind": "InitialPoseBoundedStableSupportAuthorization",
            "schema_version": 1,
            "policy_version": BOUNDED_STABLE_SUPPORT_OWNER_POLICY_VERSION,
            "authority": owner,
            "initial_state_id": state_id,
            "parent_revision": old["revision"],
            "parent_revision_hash": old["revision_hash"],
            "revision": revision["revision"],
            "revision_hash": revision["revision_hash"],
            "candidate_hash": candidate["candidate_hash"],
            "semantic_delta_deg": semantic_delta,
            "quick_sweep_hash": quick_sweep["quick_sweep_hash"],
            "quick_yaws_deg": list(QUICK_SWEEP_YAWS_DEG),
            "quick_seed": DEFAULT_QUICK_SWEEP_SEED,
            "catalog_hash": catalog_hash,
            "reuses_parent_semantic_review": True,
            "per_item_render_review_performed": False,
        }
        authorization["authorization_hash"] = artifact_hash(authorization)
        approval = {
            **old_approval,
            "revision": revision["revision"],
            "revision_hash": revision["revision_hash"],
            "geometry_hash": revision["geometry_hash"],
            "review_hash": authorization["authorization_hash"],
            "reviewer": owner,
            "decision": "approved",
            "notes": (
                "Benchmark-owner bounded support-stability authorization; "
                "parent semantic review retained; no per-item rerender review."
            ),
        }
        approval.pop("approval_hash", None)
        approval["approval_hash"] = artifact_hash(approval)
        base_states[state_id] = canonical
        base_approvals[state_id] = approval
        authorizations.append(authorization)

    authorization_set = {
        "artifact_kind": "InitialPoseBoundedStableSupportAuthorizationSet",
        "schema_version": 1,
        "policy_version": BOUNDED_STABLE_SUPPORT_OWNER_POLICY_VERSION,
        "authority": owner,
        "source_base_approved_pose_set_hash": base["approved_pose_set_hash"],
        "source_proposal_set_hash": proposal["proposal_set_hash"],
        "authorizations": authorizations,
    }
    authorization_set["authorization_set_hash"] = artifact_hash(
        authorization_set
    )
    result = {
        **base,
        "proposal_set_hash": proposal["proposal_set_hash"],
        "canonical_initial_states": sorted(
            base_states.values(), key=lambda row: str(row["initial_state_id"])
        ),
        "approvals": sorted(
            base_approvals.values(), key=lambda row: str(row["initial_state_id"])
        ),
        "bounded_stable_support_authorization": authorization_set,
    }
    result.pop("approved_pose_set_hash", None)
    result["approved_pose_set_hash"] = artifact_hash(result)
    validate_initial_pose_set(result, require_support_validated=False)
    atomic_yaml(output_path, result)
    return result


__all__ = [
    "BOUNDED_STABLE_SUPPORT_OWNER_POLICY_VERSION",
    "DEFAULT_MAX_CANDIDATES",
    "DEFAULT_MAX_SEMANTIC_TILT_DEG",
    "QUICK_SWEEP_YAWS_DEG",
    "STABLE_SUPPORT_DERIVATION_VERSION",
    "STABLE_SUPPORT_QUICK_REPORTS_DIR",
    "STABLE_SUPPORT_REPORT_FILE",
    "enumerate_stable_support_candidates",
    "extract_sapien_rigid_support_geometry",
    "materialize_stable_support_revisions",
    "promote_bounded_stable_support_revisions",
    "run_candidate_quick_sweep",
    "select_stable_support_candidate",
]
