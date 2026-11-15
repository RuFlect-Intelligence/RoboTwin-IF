"""Canonical Initial Pose proposal, review, freeze, and support contracts.

This module is deliberately simulator-independent.  SAPIEN rendering and
support execution live behind manifests, while every gate here can be checked
in ordinary unit tests and by downstream planners before a worker starts.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence
import xml.etree.ElementTree as ET

import yaml

from if_benchmark.asset_probe_pose import resolve_canonical_pose
from if_benchmark.asset_role_contract import CANDIDATE_TO_ASSET_ROLE
from if_benchmark.execution_policy import (
    DYNAMIC_ASSET_MASS_INERTIA_POLICY_VERSION,
)
from if_benchmark.schema import artifact_hash, canonical_yaml, deterministic_id


SCHEMA_VERSION = 1
POSE_CONTRACT_VERSION = 1
ASSIGNMENT_STATE_SELECTION_VERSION = "canonical_assignment_state_selection_v1"
RENDERER_CONFIG_VERSION = 1
SUPPORT_CONTRACT_VERSION = 6

DEFAULT_DATA_DIR = Path("if_benchmark/data/initial_pose")
DEFAULT_ASSET_CATALOG_PATH = Path("if_benchmark/assets/asset_catalog.yml")
DEFAULT_INVENTORY_PATH = Path("if_benchmark/data/inventory/asset_inventory.lock.yml")
DEFAULT_ARCHETYPES_PATH = Path("if_benchmark/data/families/archetypes.yml")
DEFAULT_ARTICULATED_PROFILES_PATH = Path(
    "if_benchmark/assets/articulated_receiver_profiles.yml"
)
DEFAULT_SCALE_POLICY_PATH = Path(
    "if_benchmark/assets/initial_pose_scale_policy.yml"
)

PROPOSAL_FILE = "initial_pose_proposals.yml"
AUDIT_MANIFEST_FILE = "audit_manifest.json"
HUMAN_REVIEWS_FILE = "human_reviews.json"
AUDIT_SUMMARY_FILE = "audit_summary.json"
APPROVED_SET_FILE = "approved_pose_set.yml"
SUPPORT_PLAN_FILE = "support_probe_plan.yml"
SUPPORT_REBIND_FILE = "support_rebind_report.json"
VALIDATED_SET_FILE = "validated_initial_states.yml"
ASSIGNMENT_STATE_SELECTION_FILE = "canonical_assignment_state_selection_v1.yml"
ASSIGNMENT_SELECTION_SUPPORT_PLAN_FILE = (
    "assignment_selection_support_probe_plan.yml"
)
ASSIGNMENT_SELECTION_SUPPORT_REPORTS_DIR = "assignment_selection_support_reports"
ASSIGNMENT_SELECTION_SUPPORT_RUNTIME_DIR = "assignment_selection_support_runtime"
ASSIGNMENT_SELECTION_SUPPORT_OUTPUT_FILE = (
    "canonical_assignment_state_selection_v1.current_support.yml"
)
ASSIGNMENT_SELECTION_SUPPORT_SCOPE = (
    "joint_coordinate_revision_effective_states_v1"
)
DEFAULT_ASSIGNMENT_STATE_SELECTION_PATH = (
    DEFAULT_DATA_DIR / ASSIGNMENT_STATE_SELECTION_FILE
)
REVISION_ROUND_FILE = "revision_round_report.json"

REQUIRED_VIEWS = ("robot_front", "robot_right", "three_quarter", "top")
TERMINAL_REVIEW_DECISIONS = frozenset({"approved", "unsupported"})
REVIEW_DECISIONS = frozenset({"approved", "revision_requested", "unsupported"})
AXES = frozenset({"X", "Y", "Z"})
DEFAULT_SUPPORT_Z = 0.74
DEFAULT_SUPPORT_CLEARANCE_M = 0.002
DEFAULT_SUPPORT_SEEDS = (0, 1, 2)
DEFAULT_FREE_YAWS_DEG = (0.0, 90.0, 180.0, 270.0)
ALLOWED_SCALE_TARGET_LONGEST_EXTENTS_M = (
    0.04,
    0.06,
    0.08,
    0.10,
    0.12,
    0.15,
    0.20,
    0.25,
    0.30,
    0.40,
)

_DYNAMIC_MASS_CONTRACT_FIELDS = frozenset(
    {
        "mode",
        "target_mass_kg",
        "inertia_mode",
        "policy_version",
        "source_fields",
    }
)
_DYNAMIC_MASS_SOURCE_FIELDS = frozenset(
    {"physical_defaults.mass", "articulation_drive.mass"}
)

RENDERER_CONFIG: dict[str, Any] = {
    "version": RENDERER_CONFIG_VERSION,
    "engine": "sapien",
    "physics_steps": 0,
    "width": 800,
    "height": 600,
    "fovy_deg": 52.0,
    "views": list(REQUIRED_VIEWS),
    "world_axes": {
        "+X": "robot_right_red",
        "+Y": "away_from_robot_green",
        "+Z": "up_blue",
        "rotation": "right_hand_positive_world_axis",
    },
    "overlays": ["world_axes", "positive_rotation", "actor_origin", "support_plane"],
    "collision_bounds": "manifest_and_toggle_overlay",
}

SUPPORT_CONFIG: dict[str, Any] = {
    "version": SUPPORT_CONTRACT_VERSION,
    "dynamic_asset_mass_inertia_policy_version": (
        DYNAMIC_ASSET_MASS_INERTIA_POLICY_VERSION
    ),
    "settled_bounds_source": "sapien_collision_compute_global_aabb_tight",
    "root_modes": {
        "rigid": "dynamic_table_support",
        "articulated": "fixed_root_fixture",
    },
    "contact_identity": "physx_body_entity",
    "settle_steps": 1600,
    "tail_steps": 200,
    "tail_drift_measure": "sum_adjacent_pose_path",
    "seeds": list(DEFAULT_SUPPORT_SEEDS),
    "free_yaw_degrees": list(DEFAULT_FREE_YAWS_DEG),
    "required_metrics": [
        "spawn_to_settled_translation_m",
        "spawn_to_settled_rotation_deg",
        "tail_translation_drift_m",
        "tail_rotation_drift_deg",
        "bottom_gap_m",
        "bottom_penetration_m",
        "support_impulse_ns",
        "table_edge_margin_m",
        "initial_robot_contact",
        "settled_collision_aabb",
        "collision_aabb_source",
        "support_mode",
    ],
    "fallback_pose": "forbidden",
    "pass_thresholds": {
        "spawn_to_settled_translation_m_max": 0.02,
        "spawn_to_settled_rotation_deg_max": 10.0,
        "tail_translation_drift_m_max": 0.003,
        "tail_rotation_drift_deg_max": 2.0,
        "bottom_gap_m_max": 0.005,
        "bottom_penetration_m_max": 0.003,
        "table_edge_margin_m_min": 0.02,
        "support_impulse_ns_min_exclusive": 0.0,
    },
}


class InitialPoseContractError(ValueError):
    """Raised when a pose artifact cannot cross a BenchmarkV1 gate."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
    else:
        value = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.CSafeLoader)
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping in {path}")
    return value


def _atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
    )


def atomic_yaml(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_text(path, canonical_yaml(value))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean_float(value: float) -> float:
    result = round(float(value), 12)
    return 0.0 if result == -0.0 else result


def normalize_quaternion(q_wxyz: Sequence[float]) -> list[float]:
    if len(q_wxyz) != 4:
        raise InitialPoseContractError("q0_wxyz must contain four values")
    q = [float(value) for value in q_wxyz]
    norm = math.sqrt(sum(value * value for value in q))
    if not math.isfinite(norm) or norm < 1e-12:
        raise InitialPoseContractError("q0_wxyz must be finite and nonzero")
    q = [value / norm for value in q]
    if q[0] < 0 or (abs(q[0]) < 1e-12 and next((v for v in q[1:] if abs(v) > 1e-12), 1) < 0):
        q = [-value for value in q]
    return [_clean_float(value) for value in q]


def _quaternion_multiply(left: Sequence[float], right: Sequence[float]) -> list[float]:
    lw, lx, ly, lz = [float(value) for value in left]
    rw, rx, ry, rz = [float(value) for value in right]
    return normalize_quaternion(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ]
    )


def apply_world_axis_rotations(
    q0_wxyz: Sequence[float], rotations: Sequence[Mapping[str, Any]]
) -> list[float]:
    """Apply an ordered right-hand rotation sequence about fixed world axes."""

    current = normalize_quaternion(q0_wxyz)
    for index, row in enumerate(rotations):
        axis = str(row.get("axis") or "").upper()
        if axis not in AXES:
            raise InitialPoseContractError(f"rotation {index} has invalid axis {axis!r}")
        degrees = float(row.get("degrees"))
        if not math.isfinite(degrees) or abs(degrees) > 3600:
            raise InitialPoseContractError(f"rotation {index} has invalid degrees")
        half = math.radians(degrees) / 2.0
        vector = {
            "X": [math.sin(half), 0.0, 0.0],
            "Y": [0.0, math.sin(half), 0.0],
            "Z": [0.0, 0.0, math.sin(half)],
        }[axis]
        world_rotation = [math.cos(half), *vector]
        # Pre-multiplication is the fixed-world-axis convention.
        current = _quaternion_multiply(world_rotation, current)
    return current


def quaternion_matrix(q_wxyz: Sequence[float]) -> list[list[float]]:
    w, x, y, z = normalize_quaternion(q_wxyz)
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def _rotate(point: Sequence[float], matrix: Sequence[Sequence[float]]) -> list[float]:
    return [
        sum(float(matrix[row][column]) * float(point[column]) for column in range(3))
        for row in range(3)
    ]


def _scale3(value: Any) -> list[float]:
    if isinstance(value, (int, float)):
        return [float(value)] * 3
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = [float(item) for item in value]
        if len(values) == 1:
            return values * 3
        if len(values) == 3:
            return values
    return [1.0, 1.0, 1.0]


def _model_dynamic_mass_properties(
    candidates: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze one model-wide dynamic mass contract for R01/M4/M15.

    Candidate mass hints are model properties in the current catalog: every
    use of a model must either omit a target mass or name the same target.
    Keeping this decision model-wide prevents R01 from proving one mass while
    M4 or M15 silently executes another.
    """

    if not isinstance(candidates, Mapping) or not candidates:
        raise InitialPoseContractError(
            "cannot derive dynamic mass contract without candidate records"
        )

    rows = []
    for use, raw_candidate in sorted(candidates.items()):
        if not isinstance(raw_candidate, Mapping):
            raise InitialPoseContractError(
                f"candidate record is not a mapping: {use}"
            )
        candidate = raw_candidate
        declared: list[tuple[str, float]] = []
        for container_name, source_field in (
            ("physical_defaults", "physical_defaults.mass"),
            ("articulation_drive", "articulation_drive.mass"),
        ):
            raw_container = candidate.get(container_name)
            if raw_container is None:
                container: Mapping[str, Any] = {}
            elif isinstance(raw_container, Mapping):
                container = raw_container
            else:
                raise InitialPoseContractError(
                    f"candidate {use} has invalid {container_name}"
                )
            raw_mass = container.get("mass")
            if raw_mass is None:
                continue
            if isinstance(raw_mass, bool) or not isinstance(raw_mass, (int, float)):
                raise InitialPoseContractError(
                    f"candidate {use} dynamic mass must be numeric"
                )
            mass = float(raw_mass)
            if not math.isfinite(mass) or mass <= 0.0:
                raise InitialPoseContractError(
                    f"candidate {use} dynamic mass must be finite and positive"
                )
            declared.append((source_field, mass))
        if declared and len({_clean_float(value) for _, value in declared}) != 1:
            raise InitialPoseContractError(
                f"candidate {use} dynamic mass fields disagree"
            )
        rows.append(
            (
                str(use),
                declared[0][1] if declared else None,
                tuple(source for source, _ in declared),
            )
        )

    explicit = [row for row in rows if row[1] is not None]
    if not explicit:
        return {
            "mode": "collision_geometry_auto",
            "target_mass_kg": None,
            "inertia_mode": "collision_geometry_auto",
            "policy_version": DYNAMIC_ASSET_MASS_INERTIA_POLICY_VERSION,
            "source_fields": [],
        }
    if len(explicit) != len(rows):
        raise InitialPoseContractError(
            "candidate uses mix explicit and automatic dynamic mass"
        )
    masses = {_clean_float(float(row[1])) for row in explicit}
    if len(masses) != 1:
        raise InitialPoseContractError(
            "candidate uses disagree on model dynamic mass"
        )
    target_mass = masses.pop()
    return {
        "mode": "explicit_target",
        "target_mass_kg": _clean_float(target_mass),
        "inertia_mode": "density_proportional",
        "policy_version": DYNAMIC_ASSET_MASS_INERTIA_POLICY_VERSION,
        "source_fields": sorted(
            {source for row in explicit for source in row[2]}
        ),
    }


def validate_dynamic_mass_properties(contract: Mapping[str, Any]) -> None:
    if not isinstance(contract, Mapping):
        raise InitialPoseContractError("dynamic mass contract is not a mapping")
    actual_fields = set(contract)
    if actual_fields != _DYNAMIC_MASS_CONTRACT_FIELDS:
        raise InitialPoseContractError(
            "dynamic mass contract field set mismatch: "
            f"missing={sorted(_DYNAMIC_MASS_CONTRACT_FIELDS - actual_fields)} "
            f"extra={sorted(actual_fields - _DYNAMIC_MASS_CONTRACT_FIELDS)}"
        )
    mode = contract.get("mode")
    if not isinstance(mode, str):
        raise InitialPoseContractError("dynamic mass contract mode is not a string")
    if contract.get("policy_version") != DYNAMIC_ASSET_MASS_INERTIA_POLICY_VERSION:
        raise InitialPoseContractError("dynamic mass contract policy is stale")
    source_fields = contract.get("source_fields")
    if not isinstance(source_fields, list):
        raise InitialPoseContractError(
            "dynamic mass contract source_fields must be a list"
        )
    if any(not isinstance(value, str) for value in source_fields):
        raise InitialPoseContractError(
            "dynamic mass contract source_fields must contain strings"
        )
    if source_fields != sorted(set(source_fields)):
        raise InitialPoseContractError(
            "dynamic mass contract source_fields must be sorted and unique"
        )
    unknown_sources = set(source_fields) - _DYNAMIC_MASS_SOURCE_FIELDS
    if unknown_sources:
        raise InitialPoseContractError(
            f"dynamic mass contract has unknown source fields: {sorted(unknown_sources)}"
        )
    if mode == "collision_geometry_auto":
        if contract.get("target_mass_kg") is not None:
            raise InitialPoseContractError(
                "automatic dynamic mass contract has a target mass"
            )
        if contract.get("inertia_mode") != "collision_geometry_auto":
            raise InitialPoseContractError(
                "automatic dynamic mass contract has invalid inertia mode"
            )
        if source_fields:
            raise InitialPoseContractError(
                "automatic dynamic mass contract has source fields"
            )
        return
    if mode != "explicit_target":
        raise InitialPoseContractError("invalid dynamic mass contract mode")
    raw_target = contract.get("target_mass_kg")
    if isinstance(raw_target, bool) or not isinstance(raw_target, (int, float)):
        raise InitialPoseContractError(
            "explicit dynamic mass contract target must be numeric"
        )
    target = float(raw_target)
    if not math.isfinite(target) or target <= 0.0:
        raise InitialPoseContractError(
            "explicit dynamic mass target must be finite and positive"
        )
    if contract.get("inertia_mode") != "density_proportional":
        raise InitialPoseContractError(
            "explicit dynamic mass contract has invalid inertia mode"
        )
    if not source_fields:
        raise InitialPoseContractError(
            "explicit dynamic mass contract lacks source fields"
        )


def _catalog_candidates_for_state(
    state: Mapping[str, Any], catalog: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Resolve the exact catalog model used for legacy mass-contract derivation."""

    assets = catalog.get("assets")
    if not isinstance(assets, Mapping):
        raise InitialPoseContractError("asset catalog lacks an assets mapping")
    asset = state.get("asset")
    model_id = state.get("model_id")
    if not isinstance(asset, str) or not asset:
        raise InitialPoseContractError(
            "cannot derive dynamic mass contract without a valid asset id"
        )
    asset_row = assets.get(asset)
    if not isinstance(asset_row, Mapping):
        raise InitialPoseContractError(
            f"cannot derive dynamic mass contract: catalog asset not found: {asset}"
        )
    models = asset_row.get("models")
    if not isinstance(models, Mapping):
        raise InitialPoseContractError(
            f"cannot derive dynamic mass contract: catalog models missing: {asset}"
        )
    model_key = str(model_id)
    model = models.get(model_key)
    if not isinstance(model, Mapping):
        raise InitialPoseContractError(
            "cannot derive dynamic mass contract: catalog model not found: "
            f"{asset}/{model_key}"
        )
    candidates = model.get("candidates")
    if not isinstance(candidates, Mapping) or not candidates:
        raise InitialPoseContractError(
            "cannot derive dynamic mass contract: catalog candidate records missing: "
            f"{asset}/{model_key}"
        )
    return candidates


def _catalog_hash_for_derivation(path: Path) -> str:
    if not path.is_file():
        raise InitialPoseContractError(
            f"dynamic mass derivation catalog does not exist: {path}"
        )
    return file_sha256(path)


def _proposal_catalog_for_legacy_mass_derivation(
    proposal: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    """Load the exact catalog frozen by an old proposal before deriving mass."""

    source_paths = proposal.get("source_paths")
    source_hashes = proposal.get("source_hashes")
    if not isinstance(source_paths, Mapping) or not isinstance(
        source_hashes, Mapping
    ):
        raise InitialPoseContractError(
            "legacy Initial Pose proposal lacks source catalog provenance"
        )
    raw_path = source_paths.get("asset_catalog")
    if not isinstance(raw_path, str) or not raw_path:
        raise InitialPoseContractError(
            "legacy Initial Pose proposal lacks asset catalog path"
        )
    path = Path(raw_path)
    expected_hash = source_hashes.get(raw_path)
    if not isinstance(expected_hash, str) or not expected_hash:
        raise InitialPoseContractError(
            "legacy Initial Pose proposal lacks asset catalog hash"
        )
    actual_hash = _catalog_hash_for_derivation(path)
    if actual_hash != expected_hash:
        raise InitialPoseContractError(
            "legacy Initial Pose proposal asset catalog hash mismatch"
        )
    return _load_mapping(path), actual_hash


def _metadata_bounds(
    model: Mapping[str, Any],
    q0_wxyz: Sequence[float],
    *,
    scale_override: Sequence[float] | None = None,
) -> dict[str, Any]:
    summary = model.get("metadata_summary") or {}
    center = summary.get("center") or [0.0, 0.0, 0.0]
    extents = summary.get("extents")
    scale = _scale3(scale_override if scale_override is not None else summary.get("scale"))
    if not isinstance(extents, Sequence) or len(extents) != 3:
        scaled_extents = summary.get("scaled_extents")
        if not isinstance(scaled_extents, Sequence) or len(scaled_extents) != 3:
            raise InitialPoseContractError("model has no usable rigid bounds")
        extents = [float(value) / max(abs(scale[index]), 1e-12) for index, value in enumerate(scaled_extents)]
    center_scaled = [float(center[index]) * scale[index] for index in range(3)]
    half = [abs(float(extents[index]) * scale[index]) / 2.0 for index in range(3)]
    matrix = quaternion_matrix(q0_wxyz)
    points = [
        _rotate(
            [
                center_scaled[0] + sx * half[0],
                center_scaled[1] + sy * half[1],
                center_scaled[2] + sz * half[2],
            ],
            matrix,
        )
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
        for sz in (-1.0, 1.0)
    ]
    return _bounds_record(points, source="model_metadata_center_extents")


def _mesh_bounds(
    collision_path: Path,
    scale: Sequence[float],
    q0_wxyz: Sequence[float],
) -> dict[str, Any]:
    try:
        import numpy as np
        import trimesh

        loaded = trimesh.load(collision_path, process=False)
        vertices: list[Any] = []
        if isinstance(loaded, trimesh.Scene):
            for node_name in loaded.graph.nodes_geometry:
                transform, geometry_name = loaded.graph[node_name]
                geometry = loaded.geometry[geometry_name]
                vertices.append(trimesh.transform_points(geometry.vertices, transform))
        else:
            vertices.append(loaded.vertices)
        if not vertices:
            raise ValueError("collision mesh has no vertices")
        points = np.concatenate(vertices, axis=0) * np.asarray(scale, dtype=float)
        rotation = np.asarray(quaternion_matrix(q0_wxyz), dtype=float)
        points = points @ rotation.T
        return _bounds_record(points.tolist(), source="collision_mesh_vertices")
    except ImportError as exc:
        raise InitialPoseContractError("trimesh is required to compute collision bounds") from exc


def _bounds_record(points: Sequence[Sequence[float]], *, source: str) -> dict[str, Any]:
    if not points:
        raise InitialPoseContractError("cannot compute empty geometry bounds")
    minimum = [min(float(point[axis]) for point in points) for axis in range(3)]
    maximum = [max(float(point[axis]) for point in points) for axis in range(3)]
    return {
        "frame": "actor_origin_after_q0_before_world_yaw",
        "min_xyz": [_clean_float(value) for value in minimum],
        "max_xyz": [_clean_float(value) for value in maximum],
        "extents_xyz": [_clean_float(maximum[index] - minimum[index]) for index in range(3)],
        "source": source,
    }


def compute_origin_relative_bounds(
    model: Mapping[str, Any],
    q0_wxyz: Sequence[float],
    *,
    scale_override: Sequence[float] | None = None,
) -> dict[str, Any]:
    files = model.get("files") or {}
    collision = files.get("collision") or {}
    collision_path = Path(str(collision.get("path") or ""))
    scale = _scale3(
        scale_override
        if scale_override is not None
        else (model.get("metadata_summary") or {}).get("scale")
    )
    if collision.get("present") and collision_path.is_file():
        try:
            return _mesh_bounds(collision_path, scale, q0_wxyz)
        except Exception:
            # Metadata is a declared source in the inventory and is preferable
            # to guessing or silently dropping a state when a mesh parser lacks
            # support for one file.  The source remains explicit in the record.
            pass
    return _metadata_bounds(model, q0_wxyz, scale_override=scale)


def _allowed_asset_roles(archetypes: Mapping[str, Any]) -> set[str]:
    return {
        str(asset_role)
        for archetype in archetypes.get("archetypes") or []
        for role in archetype.get("roles") or []
        for asset_role in role.get("allowed_asset_roles") or []
    }


def _initial_state_id(asset: str, model_id: str, profile_id: str) -> str:
    return deterministic_id(
        "cis",
        {"asset": asset, "model_id": model_id, "initial_state_profile_id": profile_id},
        length=20,
    )


def _articulation_joint_catalog(model: Mapping[str, Any]) -> list[dict[str, Any]]:
    articulation = model.get("articulation") or {}
    urdf_row = (articulation.get("files") or {}).get("mobility.urdf") or {}
    urdf_path = Path(str(urdf_row.get("path") or ""))
    if not urdf_row.get("present") or not urdf_path.is_file():
        raise InitialPoseContractError("articulated model has no readable mobility.urdf")
    root = ET.parse(urdf_path).getroot()
    joints: list[dict[str, Any]] = []
    for joint in root.findall("joint"):
        joint_type = str(joint.attrib.get("type") or "")
        if joint_type == "fixed":
            continue
        limit = joint.find("limit")
        lower = float((limit.attrib if limit is not None else {}).get("lower", 0.0))
        upper = float((limit.attrib if limit is not None else {}).get("upper", lower))
        joints.append(
            {
                "joint_index": len(joints),
                "joint_name": str(joint.attrib.get("name") or f"joint_{len(joints)}"),
                "joint_type": joint_type,
                "lower": _clean_float(lower),
                "upper": _clean_float(upper),
            }
        )
    return joints


def _model_init_qpos(model: Mapping[str, Any], joint_count: int) -> list[float]:
    metadata_row = ((model.get("files") or {}).get("metadata") or {})
    metadata_path = Path(str(metadata_row.get("path") or ""))
    values: list[float] = []
    if metadata_row.get("present") and metadata_path.is_file():
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        values = [float(value) for value in payload.get("init_qpos") or []]
    if len(values) > joint_count:
        raise InitialPoseContractError("model_data init_qpos exceeds active URDF joints")
    return [_clean_float(value) for value in values + [0.0] * (joint_count - len(values))]


def _exact_articulated_joint_state(
    model: Mapping[str, Any],
    *,
    profile_id: str,
    joint_targets: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    if not (model.get("articulation") or {}).get("has_articulation"):
        return {"mode": "not_articulated", "qpos": []}
    joints = _articulation_joint_catalog(model)
    qpos = _model_init_qpos(model, len(joints))
    resolved_targets: list[dict[str, Any]] = []
    for target in joint_targets:
        index = int(target.get("joint_index"))
        if index < 0 or index >= len(joints):
            raise InitialPoseContractError(
                f"profile {profile_id} targets invalid joint index {index}"
            )
        joint = joints[index]
        expected_name = target.get("joint_name")
        if expected_name and str(expected_name) != joint["joint_name"]:
            raise InitialPoseContractError(
                f"profile {profile_id} joint name/index mismatch: {expected_name} != {joint['joint_name']}"
            )
        ratio = float(target.get("target_ratio"))
        if not 0.0 <= ratio <= 1.0:
            raise InitialPoseContractError(
                f"profile {profile_id} target ratio is outside [0,1]"
            )
        value = float(joint["lower"]) + ratio * (
            float(joint["upper"]) - float(joint["lower"])
        )
        qpos[index] = _clean_float(value)
        resolved_targets.append(
            {
                **dict(target),
                "joint_index": index,
                "joint_name": joint["joint_name"],
                "target_ratio": _clean_float(ratio),
                "target_qpos": _clean_float(value),
            }
        )
    return {
        "mode": "exact_reviewed_profile" if joint_targets else "exact_model_data_default",
        "profile_id": profile_id,
        "qpos": qpos,
        "joint_index": int(resolved_targets[0]["joint_index"]) if resolved_targets else None,
        "joints": joints,
        "joint_targets": resolved_targets,
    }


def _profiles_for_model(
    profiles: Mapping[str, Any], asset: str, model_id: str
) -> list[tuple[str, dict[str, Any]]]:
    matches: list[tuple[str, dict[str, Any]]] = []
    for profile_id, profile in sorted((profiles.get("profiles") or {}).items()):
        receiver = profile.get("receiver") or {}
        if str(receiver.get("asset")) == asset and str(receiver.get("model_id")) == model_id:
            matches.append((str(profile_id), dict(profile)))
    return matches


def _revision_record(
    *,
    state_id: str,
    revision: int,
    q0_wxyz: Sequence[float],
    articulated_joint_state: Mapping[str, Any],
    yaw_policy: Mapping[str, Any],
    scale: Sequence[float],
    bounds: Mapping[str, Any],
    support_z: float,
    support_clearance_m: float,
    source: Mapping[str, Any],
    source_hashes: Mapping[str, Any],
    parent_revision: int | None = None,
    rotation_sequence: Sequence[Mapping[str, Any]] = (),
    original_request: str | None = None,
) -> dict[str, Any]:
    q0 = normalize_quaternion(q0_wxyz)
    origin_z = float(support_z) - float(bounds["min_xyz"][2]) + float(support_clearance_m)
    geometry_payload = {
        "scale": [_clean_float(value) for value in scale],
        "bounds": dict(bounds),
        "q0_wxyz": q0,
        "articulated_joint_state": dict(articulated_joint_state),
        "source_hashes": dict(source_hashes),
    }
    row: dict[str, Any] = {
        "revision": int(revision),
        "parent_revision": parent_revision,
        "q0_wxyz": q0,
        "articulated_joint_state": dict(articulated_joint_state),
        "yaw_policy": dict(yaw_policy),
        "scale_xyz": [_clean_float(value) for value in scale],
        "origin_relative_bounds": dict(bounds),
        "support": {
            "plane_z": _clean_float(support_z),
            "clearance_m": _clean_float(support_clearance_m),
            "origin_z": _clean_float(origin_z),
            "formula": "support_z-local_z_min+support_clearance",
        },
        "source": dict(source),
        "source_hashes": dict(source_hashes),
        "geometry_hash": artifact_hash(geometry_payload),
        "rotation_frame": "fixed_world_table_axes",
        "rotation_sequence": [
            {"axis": str(item["axis"]).upper(), "degrees": _clean_float(float(item["degrees"]))}
            for item in rotation_sequence
        ],
        "original_revision_request": original_request,
    }
    row["revision_hash"] = artifact_hash(row)
    return row


def build_initial_pose_proposals(
    *,
    asset_catalog_path: Path = DEFAULT_ASSET_CATALOG_PATH,
    inventory_path: Path = DEFAULT_INVENTORY_PATH,
    archetypes_path: Path = DEFAULT_ARCHETYPES_PATH,
    articulated_profiles_path: Path = DEFAULT_ARTICULATED_PROFILES_PATH,
    output_path: Path = DEFAULT_DATA_DIR / PROPOSAL_FILE,
    support_z: float = DEFAULT_SUPPORT_Z,
    support_clearance_m: float = DEFAULT_SUPPORT_CLEARANCE_M,
) -> dict[str, Any]:
    """Enumerate the raw normative role universe and freeze revision zero."""

    catalog = _load_mapping(asset_catalog_path)
    inventory = _load_mapping(inventory_path)
    archetypes = _load_mapping(archetypes_path)
    articulated_profiles = _load_mapping(articulated_profiles_path)
    allowed_roles = _allowed_asset_roles(archetypes)
    inventory_models = {
        f"{asset}|{model_id}": model
        for asset, asset_row in (inventory.get("assets") or {}).items()
        for model_id, model in (asset_row.get("models") or {}).items()
    }
    states = []
    excluded = Counter()
    for asset, asset_row in sorted((catalog.get("assets") or {}).items()):
        for model_id, model_row in sorted(
            (asset_row.get("models") or {}).items(), key=lambda item: str(item[0])
        ):
            candidates = model_row.get("candidates") or {}
            uses = sorted(str(use) for use in candidates)
            roles = sorted(
                {
                    CANDIDATE_TO_ASSET_ROLE[use]
                    for use in uses
                    if use in CANDIDATE_TO_ASSET_ROLE
                    and CANDIDATE_TO_ASSET_ROLE[use] in allowed_roles
                }
            )
            if not roles:
                excluded["outside_normative_raw_role_domain"] += 1
                continue
            key = f"{asset}|{model_id}"
            inventory_model = inventory_models.get(key)
            if inventory_model is None:
                excluded["missing_inventory_model"] += 1
                continue
            articulated = (inventory_model.get("articulation") or {}).get(
                "has_articulation"
            )
            if articulated:
                representative_use = next(
                    (
                        use
                        for use in uses
                        if (candidates.get(use) or {}).get("articulation_drive")
                    ),
                    uses[0],
                )
            else:
                representative_use = "pick_subject" if "pick_subject" in candidates else uses[0]
            candidate = dict(candidates[representative_use])
            candidate.setdefault("asset", asset)
            candidate.setdefault("model_id", str(model_id))
            candidate.setdefault("category", str(model_row.get("category") or ""))
            pose = resolve_canonical_pose(candidate, representative_use)
            dynamic_mass_properties = _model_dynamic_mass_properties(candidates)
            scale = _scale3((inventory_model.get("metadata_summary") or {}).get("scale"))
            source_hashes = {
                "inventory_model": artifact_hash(inventory_model),
                "candidate_records": artifact_hash(candidates),
                "metadata": str(((inventory_model.get("files") or {}).get("metadata") or {}).get("sha256") or "missing"),
                "visual_mesh": str(((inventory_model.get("files") or {}).get("visual") or {}).get("sha256") or "missing"),
                "collision_mesh": str(((inventory_model.get("files") or {}).get("collision") or {}).get("sha256") or "missing"),
                "articulated_profiles": file_sha256(articulated_profiles_path),
            }
            state_specs: list[tuple[str, dict[str, Any] | None]] = [("default", None)]
            if articulated:
                state_specs.extend(
                    _profiles_for_model(
                        articulated_profiles, str(asset), str(model_id)
                    )
                )
            renderable = bool(
                ((inventory_model.get("files") or {}).get("visual") or {}).get("present")
                or articulated
            )
            for profile_id, receiver_profile in state_specs:
                profile_state = (receiver_profile or {}).get("state") or {}
                profile_fixture = (receiver_profile or {}).get("fixture") or {}
                q0_wxyz = profile_fixture.get("root_q0_wxyz") or pose["q0_wxyz"]
                try:
                    bounds = compute_origin_relative_bounds(inventory_model, q0_wxyz)
                    geometry_status = "resolved"
                except InitialPoseContractError as exc:
                    bounds = {
                        "frame": "actor_origin_after_q0_before_world_yaw",
                        "min_xyz": [0.0, 0.0, 0.0],
                        "max_xyz": [0.0, 0.0, 0.0],
                        "extents_xyz": [0.0, 0.0, 0.0],
                        "source": "unresolved",
                        "error": str(exc),
                    }
                    geometry_status = "unresolved"
                articulated_joint_state = _exact_articulated_joint_state(
                    inventory_model,
                    profile_id=profile_id,
                    joint_targets=profile_state.get("joint_targets") or [],
                )
                state_id = _initial_state_id(
                    str(asset), str(model_id), profile_id
                )
                revision = _revision_record(
                    state_id=state_id,
                    revision=0,
                    q0_wxyz=q0_wxyz,
                    articulated_joint_state=articulated_joint_state,
                    yaw_policy=pose["yaw"],
                    scale=scale,
                    bounds=bounds,
                    support_z=support_z,
                    support_clearance_m=support_clearance_m,
                    source={
                        "kind": "initial_pose_proposal",
                        "resolver_source": pose.get("source"),
                        "representative_use": representative_use,
                        "resolved_by_legacy_policy": bool(pose.get("resolved")),
                        "receiver_state_profile_id": (
                            profile_id if receiver_profile is not None else None
                        ),
                        "receiver_state_profile_hash": (
                            artifact_hash(receiver_profile)
                            if receiver_profile is not None
                            else None
                        ),
                        "reviewed_profile_evidence": (
                            receiver_profile if receiver_profile is not None else {}
                        ),
                        "evidence_only_not_approval": True,
                    },
                    source_hashes=source_hashes,
                )
                state = {
                    "initial_state_id": state_id,
                    "asset": str(asset),
                    "model_id": str(model_id),
                    "initial_state_profile_id": profile_id,
                    "category": str(model_row.get("category") or ""),
                    "candidate_roles": roles,
                    "candidate_uses": uses,
                    "dynamic_mass_properties": dynamic_mass_properties,
                    "assignment_eligible": bool(
                        (receiver_profile or {}).get("acceptance_required", True)
                    ),
                    "renderable": renderable,
                    "geometry_status": geometry_status,
                    "appearance_contact_sheet": (asset_row.get("contact_sheet") or {}),
                    "current_revision": 0,
                    "revisions": [revision],
                }
                state["state_hash"] = artifact_hash(state)
                states.append(state)
    source_hashes = {
        str(asset_catalog_path): file_sha256(asset_catalog_path),
        str(inventory_path): file_sha256(inventory_path),
        str(archetypes_path): file_sha256(archetypes_path),
        str(articulated_profiles_path): file_sha256(articulated_profiles_path),
    }
    result: dict[str, Any] = {
        "artifact_kind": "initial_pose_proposal_set",
        "schema_version": SCHEMA_VERSION,
        "pose_contract_version": POSE_CONTRACT_VERSION,
        "proposal_set_id": deterministic_id("ipps", source_hashes, length=20),
        "universe": "all_normative_archetype_raw_role_domains",
        "axis_convention": RENDERER_CONFIG["world_axes"],
        "source_paths": {
            "asset_catalog": str(asset_catalog_path),
            "inventory": str(inventory_path),
            "archetypes": str(archetypes_path),
            "articulated_profiles": str(articulated_profiles_path),
        },
        "source_hashes": source_hashes,
        "summary": {
            "inventory_model_count": int((inventory.get("summary") or {}).get("model_count") or len(inventory_models)),
            "raw_role_domain_state_count": len(states),
            "renderable_state_count": sum(bool(state["renderable"]) for state in states),
            "geometry_resolved_state_count": sum(state["geometry_status"] == "resolved" for state in states),
            "excluded_model_counts": dict(sorted(excluded.items())),
        },
        "states": states,
    }
    result["proposal_set_hash"] = artifact_hash(result)
    atomic_yaml(output_path, result)
    return result


def validate_render_manifest(
    manifest: Mapping[str, Any],
    revision: Mapping[str, Any],
    *,
    require_files: bool = True,
) -> None:
    if manifest.get("artifact_kind") != "initial_pose_render_manifest":
        raise InitialPoseContractError("unexpected render manifest kind")
    if manifest.get("manifest_hash") != artifact_hash(manifest, omit_keys=("manifest_hash",)):
        raise InitialPoseContractError("render manifest self-hash mismatch")
    if manifest.get("revision_hash") != revision.get("revision_hash"):
        raise InitialPoseContractError("render is stale for current revision")
    if manifest.get("geometry_hash") != revision.get("geometry_hash"):
        raise InitialPoseContractError("render geometry hash is stale")
    if manifest.get("renderer_config_hash") != artifact_hash(RENDERER_CONFIG):
        raise InitialPoseContractError("render renderer-config hash is stale")
    views = {str(row.get("view")): row for row in manifest.get("views") or []}
    if set(views) != set(REQUIRED_VIEWS):
        raise InitialPoseContractError("render manifest must contain exactly four required views")
    if bool(manifest.get("physics_advanced")):
        raise InitialPoseContractError("semantic review render advanced physics")
    if require_files:
        for view in REQUIRED_VIEWS:
            row = views[view]
            path = Path(str(row.get("path") or ""))
            if not path.is_file() or file_sha256(path) != row.get("sha256"):
                raise InitialPoseContractError(f"render image missing or stale: {view}")
            collision_path = Path(str(row.get("collision_overlay_path") or ""))
            if (
                not collision_path.is_file()
                or file_sha256(collision_path)
                != row.get("collision_overlay_sha256")
            ):
                raise InitialPoseContractError(
                    f"collision overlay missing or stale: {view}"
                )


def _current_revision(state: Mapping[str, Any]) -> dict[str, Any]:
    revision_id = int(state.get("current_revision") or 0)
    matches = [
        dict(row)
        for row in state.get("revisions") or []
        if int(row.get("revision") or 0) == revision_id
    ]
    if len(matches) != 1:
        raise InitialPoseContractError(
            f"state {state.get('initial_state_id')} has no unique current revision"
        )
    revision = matches[0]
    if revision.get("revision_hash") != artifact_hash(revision, omit_keys=("revision_hash",)):
        raise InitialPoseContractError("pose revision self-hash mismatch")
    return revision


def build_pose_audit_workspace(
    *,
    proposal_path: Path,
    render_root: Path,
    audit_dir: Path,
    designated_reviewer: str,
) -> dict[str, Any]:
    """Join current revisions to render manifests and preserve review history."""

    reviewer = str(designated_reviewer).strip()
    if not reviewer:
        raise InitialPoseContractError("designated_reviewer must be non-empty")
    proposal = _load_mapping(proposal_path)
    if proposal.get("proposal_set_hash") != artifact_hash(proposal, omit_keys=("proposal_set_hash",)):
        raise InitialPoseContractError("proposal set self-hash mismatch")
    items = []
    for state in proposal.get("states") or []:
        revision = _current_revision(state)
        render_path = render_root / str(state["initial_state_id"]) / f"revision_{revision['revision']}" / "manifest.json"
        render = _load_mapping(render_path) if render_path.is_file() else None
        render_status = "missing"
        render_error = None
        if render is not None:
            try:
                validate_render_manifest(render, revision)
                render_status = "current"
            except (InitialPoseContractError, FileNotFoundError) as exc:
                render_status = "stale"
                render_error = str(exc)
        items.append(
            {
                "item_id": f"{state['initial_state_id']}@{revision['revision']}",
                "initial_state_id": state["initial_state_id"],
                "asset": state["asset"],
                "model_id": state["model_id"],
                "initial_state_profile_id": state["initial_state_profile_id"],
                "category": state.get("category"),
                "candidate_roles": state.get("candidate_roles") or [],
                "candidate_uses": state.get("candidate_uses") or [],
                "assignment_eligible": bool(state.get("assignment_eligible")),
                "revision": revision,
                "render": render,
                "render_status": render_status,
                "render_error": render_error,
                "appearance_contact_sheet": state.get("appearance_contact_sheet") or {},
                "revision_history": [
                    {
                        "revision": row.get("revision"),
                        "parent_revision": row.get("parent_revision"),
                        "revision_hash": row.get("revision_hash"),
                        "rotation_sequence": row.get("rotation_sequence") or [],
                        "scale_xyz": row.get("scale_xyz") or [],
                        "origin_relative_bounds": row.get("origin_relative_bounds") or {},
                        "scale_repair": (row.get("source") or {}).get("scale_repair"),
                        "original_revision_request": row.get("original_revision_request"),
                    }
                    for row in state.get("revisions") or []
                ],
            }
        )
    manifest: dict[str, Any] = {
        "artifact_kind": "initial_pose_audit_manifest",
        "schema_version": SCHEMA_VERSION,
        "pose_contract_version": POSE_CONTRACT_VERSION,
        "audit_id": deterministic_id(
            "ipa",
            {
                "proposal_set_id": proposal["proposal_set_id"],
                "designated_reviewer": reviewer,
            },
            length=20,
        ),
        "proposal_path": str(proposal_path.resolve()),
        "proposal_set_hash": proposal["proposal_set_hash"],
        "designated_reviewer": reviewer,
        "axis_convention": RENDERER_CONFIG["world_axes"],
        "item_count": len(items),
        "items": items,
    }
    manifest["manifest_hash"] = artifact_hash(manifest)
    audit_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(audit_dir / AUDIT_MANIFEST_FILE, manifest)
    reviews_path = audit_dir / HUMAN_REVIEWS_FILE
    if reviews_path.is_file():
        reviews = _load_mapping(reviews_path)
        retained = list(reviews.get("reviews") or [])
        revision_number = int(reviews.get("revision") or 0)
    else:
        retained = []
        revision_number = 0
    review_doc = {
        "artifact_kind": "initial_pose_human_reviews",
        "schema_version": SCHEMA_VERSION,
        "audit_id": manifest["audit_id"],
        "manifest_hash": manifest["manifest_hash"],
        "revision": revision_number,
        "reviews": retained,
    }
    review_doc["reviews_hash"] = artifact_hash(review_doc)
    atomic_json(reviews_path, review_doc)
    return manifest


def _inventory_model_for_state(
    inventory: Mapping[str, Any], state: Mapping[str, Any]
) -> dict[str, Any]:
    model = (
        ((inventory.get("assets") or {}).get(str(state["asset"])) or {})
        .get("models", {})
        .get(str(state["model_id"]))
    )
    if not isinstance(model, dict):
        raise InitialPoseContractError("inventory model missing while revising pose")
    return model


def _materialize_revision_record(
    *,
    state: Mapping[str, Any],
    parent: Mapping[str, Any],
    model: Mapping[str, Any],
    review: Mapping[str, Any],
    reviewer: str,
    rotations: Sequence[Mapping[str, Any]],
    scale_target_longest_extent_m: float | None = None,
    scale_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not rotations and scale_target_longest_extent_m is None:
        raise InitialPoseContractError("revision has neither rotation nor scale change")
    q0 = apply_world_axis_rotations(parent["q0_wxyz"], rotations)
    parent_scale = _scale3(parent.get("scale_xyz"))
    scale = list(parent_scale)
    scale_repair = None
    source_hashes = dict(parent.get("source_hashes") or {})
    if scale_target_longest_extent_m is not None:
        target = float(scale_target_longest_extent_m)
        if not math.isfinite(target) or not 0.01 <= target <= 1.0:
            raise InitialPoseContractError(
                "scale target longest extent must be within [0.01, 1.0] meters"
            )
        before = compute_origin_relative_bounds(
            model, q0, scale_override=parent_scale
        )
        current_longest = max(float(value) for value in before["extents_xyz"])
        if current_longest <= 0:
            raise InitialPoseContractError("cannot scale geometry with zero extent")
        multiplier = target / current_longest
        scale = [_clean_float(value * multiplier) for value in parent_scale]
        scale_repair = {
            "mode": "uniform_target_longest_extent",
            "parent_scale_xyz": parent_scale,
            "parent_longest_extent_m": _clean_float(current_longest),
            "target_longest_extent_m": _clean_float(target),
            "uniform_multiplier": _clean_float(multiplier),
        }
        if scale_policy is not None:
            policy_hash = artifact_hash(scale_policy)
            scale_repair["policy_id"] = str(scale_policy.get("policy_id") or "")
            scale_repair["policy_hash"] = policy_hash
            source_hashes["scale_repair_policy"] = policy_hash
    bounds = compute_origin_relative_bounds(model, q0, scale_override=scale)
    source_kind = (
        "human_revision_and_scale_repair_proposal"
        if rotations and scale_repair is not None
        else "human_revision_request"
        if rotations
        else "scale_repair_proposal"
    )
    source: dict[str, Any] = {
        "kind": source_kind,
        "reviewer": reviewer,
        "parent_revision_hash": parent["revision_hash"],
        "review_hash": review["review_hash"],
    }
    if scale_repair is not None:
        source["scale_repair"] = scale_repair
    return _revision_record(
        state_id=str(state["initial_state_id"]),
        revision=int(parent["revision"]) + 1,
        parent_revision=int(parent["revision"]),
        q0_wxyz=q0,
        articulated_joint_state=parent.get("articulated_joint_state") or {},
        yaw_policy=parent["yaw_policy"],
        scale=scale,
        bounds=bounds,
        support_z=float(
            (parent.get("support") or {}).get("plane_z") or DEFAULT_SUPPORT_Z
        ),
        support_clearance_m=float(
            (parent.get("support") or {}).get("clearance_m")
            or DEFAULT_SUPPORT_CLEARANCE_M
        ),
        source=source,
        source_hashes=source_hashes,
        rotation_sequence=rotations,
        original_request=str(review.get("notes") or ""),
    )


def _append_current_revision(
    state: dict[str, Any], revision: Mapping[str, Any]
) -> None:
    state["revisions"] = [*list(state.get("revisions") or []), dict(revision)]
    state["current_revision"] = int(revision["revision"])
    state.pop("state_hash", None)
    state["state_hash"] = artifact_hash(state)


def materialize_requested_revision(
    *,
    proposal_path: Path,
    reviews_path: Path,
    initial_state_id: str,
    reviewer: str,
) -> dict[str, Any]:
    """Deterministically apply one reviewer's structured revision request."""

    proposal = _load_mapping(proposal_path)
    reviews = _load_mapping(reviews_path)
    states = list(proposal.get("states") or [])
    state = next(
        (row for row in states if row.get("initial_state_id") == initial_state_id),
        None,
    )
    if state is None:
        raise KeyError(initial_state_id)
    parent = _current_revision(state)
    review = next(
        (
            row
            for row in reversed(reviews.get("reviews") or [])
            if row.get("initial_state_id") == initial_state_id
            and row.get("revision_hash") == parent.get("revision_hash")
            and row.get("reviewer") == reviewer
            and row.get("decision") == "revision_requested"
        ),
        None,
    )
    if review is None:
        raise InitialPoseContractError("no matching current revision request")
    rotations = review.get("rotation_sequence") or []
    scale_target = review.get("scale_target_longest_extent_m")
    if not rotations and scale_target is None:
        raise InitialPoseContractError(
            "revision request has neither structured rotation nor scale target"
        )
    inventory_path = Path(
        str(
            (proposal.get("source_paths") or {}).get("inventory")
            or DEFAULT_INVENTORY_PATH
        )
    )
    inventory = _load_mapping(inventory_path)
    model = _inventory_model_for_state(inventory, state)
    revision = _materialize_revision_record(
        state=state,
        parent=parent,
        model=model,
        review=review,
        reviewer=reviewer,
        rotations=rotations,
        scale_target_longest_extent_m=(
            float(scale_target) if scale_target is not None else None
        ),
    )
    _append_current_revision(state, revision)
    proposal["states"] = states
    proposal["summary"] = {
        **(proposal.get("summary") or {}),
        "revised_state_count": sum(
            len(row.get("revisions") or []) > 1 for row in states
        ),
    }
    proposal.pop("proposal_set_hash", None)
    proposal["proposal_set_hash"] = artifact_hash(proposal)
    atomic_yaml(proposal_path, proposal)
    return revision


def materialize_audit_revision_round(
    *,
    proposal_path: Path,
    reviews_path: Path,
    reviewer: str,
    scale_policy_path: Path = DEFAULT_SCALE_POLICY_PATH,
    report_path: Path | None = None,
) -> dict[str, Any]:
    """Materialize all current human revisions plus deterministic scale repairs.

    Missing metadata scale is never repaired by silently assuming a new runtime
    value.  Only states beyond the policy's absolute tabletop threshold receive
    a versioned, uniformly-scaled proposal, which must be rendered and approved
    again before it can cross the freeze gate.
    """

    proposal = _load_mapping(proposal_path)
    reviews = _load_mapping(reviews_path)
    policy = _load_mapping(scale_policy_path)
    if policy.get("artifact_kind") != "initial_pose_scale_repair_policy":
        raise InitialPoseContractError("unexpected scale repair policy kind")
    if reviews.get("reviews_hash") != artifact_hash(
        reviews, omit_keys=("reviews_hash",)
    ):
        raise InitialPoseContractError("human review self-hash mismatch")
    inventory_path = Path(
        str(
            (proposal.get("source_paths") or {}).get("inventory")
            or DEFAULT_INVENTORY_PATH
        )
    )
    inventory = _load_mapping(inventory_path)
    selection = policy.get("selection") or {}
    threshold = float(
        selection.get("current_longest_extent_m_min_exclusive") or 0.5
    )
    targets = policy.get("targets") or {}
    states = list(proposal.get("states") or [])
    rows = []
    unresolved_categories = set()
    for state in states:
        parent = _current_revision(state)
        review = next(
            (
                row
                for row in reversed(reviews.get("reviews") or [])
                if row.get("initial_state_id") == state.get("initial_state_id")
                and row.get("revision_hash") == parent.get("revision_hash")
                and row.get("reviewer") == reviewer
            ),
            None,
        )
        if review is None:
            continue
        rotations = (
            list(review.get("rotation_sequence") or [])
            if review.get("decision") == "revision_requested"
            else []
        )
        explicit_scale_target = review.get("scale_target_longest_extent_m")
        model = _inventory_model_for_state(inventory, state)
        metadata_scale = (model.get("metadata_summary") or {}).get("scale")
        parent_longest = max(
            float(value)
            for value in parent["origin_relative_bounds"]["extents_xyz"]
        )
        automatic_scale_target = None
        target_policy = targets.get(str(state.get("category") or ""))
        if (
            bool(selection.get("metadata_scale_must_be_missing", True))
            and metadata_scale is None
            and parent_longest > threshold
        ):
            if not isinstance(target_policy, Mapping):
                unresolved_categories.add(str(state.get("category") or "unknown"))
            else:
                automatic_scale_target = float(
                    target_policy["target_longest_extent_m"]
                )
        scale_target = (
            float(explicit_scale_target)
            if explicit_scale_target is not None
            else automatic_scale_target
        )
        if not rotations and scale_target is None:
            continue
        if scale_target is not None and (
            parent.get("articulated_joint_state") or {}
        ).get("mode") != "not_articulated":
            raise InitialPoseContractError(
                f"articulated scale repair is unsupported: {state['initial_state_id']}"
            )
        revision = _materialize_revision_record(
            state=state,
            parent=parent,
            model=model,
            review=review,
            reviewer=reviewer,
            rotations=rotations,
            scale_target_longest_extent_m=scale_target,
            scale_policy=policy if automatic_scale_target is not None else None,
        )
        _append_current_revision(state, revision)
        rows.append(
            {
                "initial_state_id": state["initial_state_id"],
                "asset": state["asset"],
                "model_id": state["model_id"],
                "category": state.get("category"),
                "parent_revision": parent["revision"],
                "revision": revision["revision"],
                "revision_hash": revision["revision_hash"],
                "review_decision": review["decision"],
                "rotation_sequence": rotations,
                "scale_target_longest_extent_m": scale_target,
                "scale_policy_basis": (
                    (target_policy or {}).get("basis")
                    if automatic_scale_target is not None
                    else None
                ),
            }
        )
    if unresolved_categories:
        raise InitialPoseContractError(
            "scale repair policy lacks target categories: "
            + ", ".join(sorted(unresolved_categories))
        )
    proposal["states"] = states
    proposal["summary"] = {
        **(proposal.get("summary") or {}),
        "revised_state_count": sum(
            len(row.get("revisions") or []) > 1 for row in states
        ),
        "scale_repaired_state_count": sum(
            row["scale_target_longest_extent_m"] is not None for row in rows
        ),
    }
    proposal.pop("proposal_set_hash", None)
    proposal["proposal_set_hash"] = artifact_hash(proposal)
    report: dict[str, Any] = {
        "artifact_kind": "initial_pose_revision_round_report",
        "schema_version": SCHEMA_VERSION,
        "reviewer": reviewer,
        "input_reviews_hash": reviews["reviews_hash"],
        "scale_policy_path": str(scale_policy_path),
        "scale_policy_hash": artifact_hash(policy),
        "proposal_set_hash": proposal["proposal_set_hash"],
        "summary": {
            "revised_state_count": len(rows),
            "rotation_revision_count": sum(bool(row["rotation_sequence"]) for row in rows),
            "scale_repair_count": sum(
                row["scale_target_longest_extent_m"] is not None for row in rows
            ),
            "combined_rotation_scale_count": sum(
                bool(row["rotation_sequence"])
                and row["scale_target_longest_extent_m"] is not None
                for row in rows
            ),
            "terminal_decision_invalidated_count": sum(
                row["review_decision"] in TERMINAL_REVIEW_DECISIONS for row in rows
            ),
        },
        "revisions": sorted(rows, key=lambda row: row["initial_state_id"]),
    }
    report["report_hash"] = artifact_hash(report)
    atomic_yaml(proposal_path, proposal)
    atomic_json(
        report_path or proposal_path.with_name(REVISION_ROUND_FILE), report
    )
    return report


def freeze_approved_pose_set(
    *,
    proposal_path: Path,
    audit_dir: Path,
    output_path: Path,
    scale_policy_path: Path = DEFAULT_SCALE_POLICY_PATH,
) -> dict[str, Any]:
    """Freeze only when every proposed current state has a designated decision."""

    proposal = _load_mapping(proposal_path)
    manifest = _load_mapping(audit_dir / AUDIT_MANIFEST_FILE)
    reviews = _load_mapping(audit_dir / HUMAN_REVIEWS_FILE)
    scale_policy = _load_mapping(scale_policy_path)
    inventory_path = Path(
        str(
            (proposal.get("source_paths") or {}).get("inventory")
            or DEFAULT_INVENTORY_PATH
        )
    )
    inventory = _load_mapping(inventory_path)
    scale_threshold = float(
        ((scale_policy.get("selection") or {}).get(
            "current_longest_extent_m_min_exclusive"
        ))
        or 0.5
    )
    if reviews.get("reviews_hash") != artifact_hash(reviews, omit_keys=("reviews_hash",)):
        raise InitialPoseContractError("human review self-hash mismatch")
    if manifest.get("proposal_set_hash") != proposal.get("proposal_set_hash"):
        raise InitialPoseContractError("audit manifest is stale for proposal set")
    designated = str(manifest.get("designated_reviewer") or "")
    review_index = {
        (str(row.get("initial_state_id")), str(row.get("revision_hash")), str(row.get("reviewer"))): row
        for row in reviews.get("reviews") or []
    }
    item_index = {str(row["initial_state_id"]): row for row in manifest.get("items") or []}
    canonical_states = []
    approvals = []
    blockers = []
    legacy_mass_catalog: dict[str, Any] | None = None
    for state in proposal.get("states") or []:
        state_id = str(state["initial_state_id"])
        revision = _current_revision(state)
        model = _inventory_model_for_state(inventory, state)
        if (
            (model.get("metadata_summary") or {}).get("scale") is None
            and max(
                float(value)
                for value in revision["origin_relative_bounds"]["extents_xyz"]
            )
            > scale_threshold
        ):
            blockers.append(
                {
                    "initial_state_id": state_id,
                    "reason": "unresolved_missing_metadata_scale_anomaly",
                }
            )
            continue
        item = item_index.get(state_id)
        if item is None or item.get("render_status") != "current":
            blockers.append({"initial_state_id": state_id, "reason": "missing_or_stale_render"})
            continue
        try:
            validate_render_manifest(item.get("render") or {}, revision)
        except InitialPoseContractError as exc:
            blockers.append({"initial_state_id": state_id, "reason": str(exc)})
            continue
        review = review_index.get((state_id, str(revision["revision_hash"]), designated))
        if review is None or review.get("decision") not in TERMINAL_REVIEW_DECISIONS:
            blockers.append({"initial_state_id": state_id, "reason": "designated_reviewer_decision_missing"})
            continue
        if review.get("review_hash") != artifact_hash(
            review, omit_keys=("review_hash", "updated_at")
        ):
            blockers.append(
                {"initial_state_id": state_id, "reason": "review_self_hash_mismatch"}
            )
            continue
        if review.get("render_manifest_hash") != (item.get("render") or {}).get("manifest_hash"):
            blockers.append({"initial_state_id": state_id, "reason": "approval_render_hash_stale"})
            continue
        if "dynamic_mass_properties" in state:
            embedded_mass = state.get("dynamic_mass_properties")
            if not isinstance(embedded_mass, Mapping):
                blockers.append(
                    {
                        "initial_state_id": state_id,
                        "reason": "invalid_dynamic_mass_contract",
                    }
                )
                continue
            dynamic_mass_properties = dict(embedded_mass)
            try:
                validate_dynamic_mass_properties(dynamic_mass_properties)
            except InitialPoseContractError as exc:
                blockers.append(
                    {"initial_state_id": state_id, "reason": str(exc)}
                )
                continue
        else:
            try:
                if legacy_mass_catalog is None:
                    legacy_mass_catalog, _ = (
                        _proposal_catalog_for_legacy_mass_derivation(proposal)
                    )
                dynamic_mass_properties = _dynamic_mass_properties_for_state(
                    state, legacy_mass_catalog
                )
            except InitialPoseContractError as exc:
                blockers.append(
                    {"initial_state_id": state_id, "reason": str(exc)}
                )
                continue
        canonical = {
            "artifact_kind": "CanonicalInitialState",
            "schema_version": SCHEMA_VERSION,
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
            "candidate_roles": state.get("candidate_roles") or [],
            "candidate_uses": state.get("candidate_uses") or [],
            "dynamic_mass_properties": dynamic_mass_properties,
            "assignment_eligible": bool(state.get("assignment_eligible")),
            "status": str(review["decision"]),
        }
        canonical["canonical_state_hash"] = artifact_hash(canonical)
        approval = {
            "artifact_kind": "InitialPoseApproval",
            "schema_version": SCHEMA_VERSION,
            "initial_state_id": state_id,
            "revision": revision["revision"],
            "revision_hash": revision["revision_hash"],
            "geometry_hash": revision["geometry_hash"],
            "render_manifest_hash": review["render_manifest_hash"],
            "review_hash": review["review_hash"],
            "reviewer": designated,
            "decision": review["decision"],
            "notes": review.get("notes") or "",
        }
        approval["approval_hash"] = artifact_hash(approval)
        canonical_states.append(canonical)
        approvals.append(approval)
    if blockers:
        preview = ", ".join(f"{row['initial_state_id']}:{row['reason']}" for row in blockers[:8])
        raise InitialPoseContractError(
            f"Approved Pose Set is blocked by {len(blockers)} state(s): {preview}"
        )
    result: dict[str, Any] = {
        "artifact_kind": "approved_initial_pose_set",
        "schema_version": SCHEMA_VERSION,
        "pose_contract_version": POSE_CONTRACT_VERSION,
        "proposal_set_hash": proposal["proposal_set_hash"],
        "audit_manifest_hash": manifest["manifest_hash"],
        "human_reviews_hash": reviews["reviews_hash"],
        "scale_repair_policy_hash": artifact_hash(scale_policy),
        "designated_reviewer": designated,
        "summary": {
            "state_count": len(canonical_states),
            "approved_count": sum(row["status"] == "approved" for row in canonical_states),
            "unsupported_count": sum(row["status"] == "unsupported" for row in canonical_states),
        },
        "canonical_initial_states": sorted(canonical_states, key=lambda row: row["initial_state_id"]),
        "approvals": sorted(approvals, key=lambda row: row["initial_state_id"]),
    }
    result["approved_pose_set_hash"] = artifact_hash(result)
    validate_initial_pose_set(result, require_support_validated=False)
    atomic_yaml(output_path, result)
    return result


def load_approved_pose_index(
    path: Path,
    *,
    require_support_validated: bool = False,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    document = _load_mapping(path)
    validate_initial_pose_set(
        document, require_support_validated=require_support_validated
    )
    records = document.get("canonical_initial_states") or []
    result = {}
    for record in records:
        if record.get("status") != "approved":
            continue
        key = (str(record["asset"]), str(record["model_id"]), str(record["initial_state_profile_id"]))
        result[key] = dict(record)
    return result


def validate_canonical_initial_state(record: Mapping[str, Any]) -> None:
    """Validate the immutable canonical identity and optional support join."""

    if record.get("artifact_kind") != "CanonicalInitialState":
        raise InitialPoseContractError("unexpected CanonicalInitialState kind")
    canonical_payload = dict(record)
    canonical_payload.pop("canonical_state_hash", None)
    canonical_payload.pop("support_validation", None)
    canonical_payload.pop("support_validation_hash", None)
    if record.get("canonical_state_hash") != artifact_hash(canonical_payload):
        raise InitialPoseContractError("CanonicalInitialState self-hash mismatch")
    if "dynamic_mass_properties" in record:
        dynamic_mass_properties = record.get("dynamic_mass_properties")
        if not isinstance(dynamic_mass_properties, Mapping):
            raise InitialPoseContractError(
                "CanonicalInitialState dynamic mass contract is invalid"
            )
        validate_dynamic_mass_properties(dynamic_mass_properties)
    support = record.get("support_validation")
    support_hash = record.get("support_validation_hash")
    if support is None and support_hash is not None:
        raise InitialPoseContractError("orphan support validation hash")
    if support is not None and support_hash != artifact_hash(support):
        raise InitialPoseContractError("support validation self-hash mismatch")
    if support is not None:
        _validate_support_validation_identity(support)


def _validate_support_validation_identity(
    support: Mapping[str, Any],
) -> None:
    """Validate the frozen R01 policy identity without rebinding to source.

    R01 is an immutable, reviewed input.  Its stored support configuration
    remains part of the artifact lineage and is checked for presence and
    consistency by :func:`validate_initial_pose_set`, but later source-file
    edits must not retroactively invalidate an already frozen pose set.
    """

    if support.get("support_contract_version") != SUPPORT_CONTRACT_VERSION:
        raise InitialPoseContractError(
            "support validation contract is stale for current runtime"
        )
    if (
        support.get("dynamic_asset_mass_inertia_policy_version")
        != DYNAMIC_ASSET_MASS_INERTIA_POLICY_VERSION
    ):
        raise InitialPoseContractError(
            "support validation mass/inertia policy is stale"
        )
    config_hash = support.get("support_config_hash")
    if not isinstance(config_hash, str) or not config_hash:
        raise InitialPoseContractError(
            "support validation lacks support config identity"
        )


def validate_initial_pose_approval(record: Mapping[str, Any]) -> None:
    if record.get("artifact_kind") != "InitialPoseApproval":
        raise InitialPoseContractError("unexpected InitialPoseApproval kind")
    if record.get("approval_hash") != artifact_hash(
        record, omit_keys=("approval_hash",)
    ):
        raise InitialPoseContractError("approval self-hash mismatch")


def validate_assignment_state_selection(
    selection: Mapping[str, Any],
    document: Mapping[str, Any],
    *,
    require_support_validated: bool,
) -> None:
    """Validate the optional project-local canonical state selector.

    R01 may contain multiple reviewed states for one articulated model (for
    example a receiver-open state and the model-data default).  Downstream
    assignment and A2O stages must nevertheless bind exactly one shared state
    for all variants.  Older unit-test fixtures omit this policy and retain the
    historical ``default`` behavior.
    """

    policy = selection
    if not isinstance(policy, Mapping):
        raise InitialPoseContractError(
            "canonical assignment state selection must be a mapping"
        )
    if policy.get("artifact_kind") != "CanonicalAssignmentStateSelection":
        raise InitialPoseContractError(
            "unexpected canonical assignment state selection kind"
        )
    if policy.get("selection_version") != ASSIGNMENT_STATE_SELECTION_VERSION:
        raise InitialPoseContractError(
            "unsupported canonical assignment state selection version"
        )
    if policy.get("review_status") != "reviewed":
        raise InitialPoseContractError(
            "canonical assignment state selection is not reviewed"
        )
    if policy.get("fallback_profile_id") != "default":
        raise InitialPoseContractError(
            "canonical assignment state selection fallback must be default"
        )
    if policy.get("selection_hash") != artifact_hash(
        policy, omit_keys=("selection_hash",)
    ):
        raise InitialPoseContractError(
            "canonical assignment state selection self-hash mismatch"
        )

    # The neutral-state decision is defined against the one global Success
    # Contract.  A contract threshold change must make this R01 selector stale.
    from if_benchmark.success_contract import (
        SuccessContract,
        articulation_target_interval,
        resolve_articulation_goal,
    )

    contract = SuccessContract.load()
    if policy.get("success_contract_hash") != contract.hash:
        raise InitialPoseContractError(
            "canonical assignment state selection success contract mismatch"
        )

    states = list(document.get("canonical_initial_states") or [])
    states_by_id = {str(row.get("initial_state_id")): row for row in states}
    selected_models: set[str] = set()
    for model_key, raw_selection in sorted((policy.get("selections") or {}).items()):
        if not isinstance(raw_selection, Mapping):
            raise InitialPoseContractError(
                f"canonical assignment selection is malformed: {model_key}"
            )
        asset, separator, model_id = str(model_key).partition("|")
        if not separator or not asset or not model_id:
            raise InitialPoseContractError(
                f"invalid canonical assignment model key: {model_key}"
            )
        if model_key in selected_models:
            raise InitialPoseContractError(
                f"duplicate canonical assignment model selection: {model_key}"
            )
        selected_models.add(str(model_key))
        state_id = str(raw_selection.get("initial_state_id") or "")
        state = states_by_id.get(state_id)
        if state is None:
            raise InitialPoseContractError(
                f"canonical assignment state is missing: {model_key}:{state_id}"
            )
        if (str(state.get("asset")), str(state.get("model_id"))) != (
            asset,
            model_id,
        ):
            raise InitialPoseContractError(
                f"canonical assignment state/model mismatch: {model_key}:{state_id}"
            )
        if str(state.get("initial_state_profile_id")) != str(
            raw_selection.get("initial_state_profile_id")
        ):
            raise InitialPoseContractError(
                f"canonical assignment state/profile mismatch: {model_key}:{state_id}"
            )
        if state.get("status") != "approved":
            raise InitialPoseContractError(
                f"canonical assignment state is not approved: {model_key}:{state_id}"
            )
        if raw_selection.get("canonical_state_hash") != state.get(
            "canonical_state_hash"
        ):
            raise InitialPoseContractError(
                f"canonical assignment state hash mismatch: {model_key}:{state_id}"
            )
        source_approval = approval_for_state(document, state_id)
        if raw_selection.get("joint_coordinate_revision") is not None:
            if raw_selection.get("source_approval_hash") != source_approval.get(
                "approval_hash"
            ):
                raise InitialPoseContractError(
                    f"canonical assignment source approval hash mismatch: {model_key}:{state_id}"
                )
        effective_state = apply_assignment_joint_coordinate_revision(
            state,
            raw_selection,
        )
        effective_approval = assignment_approval_for_state(
            document,
            effective_state,
            raw_selection,
        )
        if raw_selection.get("approval_hash") != effective_approval.get(
            "approval_hash"
        ):
            raise InitialPoseContractError(
                f"canonical assignment approval hash mismatch: {model_key}:{state_id}"
            )
        if raw_selection.get("effective_canonical_state_hash") not in (
            None,
            effective_state.get("canonical_state_hash"),
        ):
            raise InitialPoseContractError(
                f"canonical assignment effective state hash mismatch: {model_key}:{state_id}"
            )
        if require_support_validated:
            support_validation = effective_state.get("support_validation") or {}
            if support_validation.get("status") != "validated":
                raise InitialPoseContractError(
                    f"canonical assignment state lacks support validation: {model_key}:{state_id}"
                )
            # A current support-validated R01 set must never admit stale
            # selector-local evidence.  Legacy fixtures without the set-level
            # v5 identity remain readable only as migration context; canonical
            # production consumers validate the set itself before this join.
            if document.get("support_contract_version") == SUPPORT_CONTRACT_VERSION:
                _validate_support_validation_identity(support_validation)
        if require_support_validated and raw_selection.get(
            "support_validation_hash"
        ) != effective_state.get("support_validation_hash"):
            raise InitialPoseContractError(
                f"canonical assignment support hash mismatch: {model_key}:{state_id}"
            )

        required_uses = [str(value) for value in raw_selection.get("required_uses") or []]
        if not required_uses:
            continue
        joint_state = effective_state.get("articulated_joint_state") or {}
        qpos = list(joint_state.get("qpos") or [])
        joints = list(joint_state.get("joints") or [])
        joint_index = int(raw_selection.get("joint_index", -1))
        if joint_index < 0 or joint_index >= len(qpos):
            raise InitialPoseContractError(
                f"canonical assignment joint index is invalid: {model_key}:{joint_index}"
            )
        joint = next(
            (
                row
                for row in joints
                if int(row.get("joint_index", -1)) == joint_index
            ),
            None,
        )
        if joint is None:
            raise InitialPoseContractError(
                f"canonical assignment joint metadata is missing: {model_key}:{joint_index}"
            )
        profile = {"positive_use": str(raw_selection.get("positive_use") or "open")}
        for use in required_uses:
            goal = resolve_articulation_goal(
                state=use,
                initial_qpos=float(qpos[joint_index]),
                lower=float(joint["lower"]),
                upper=float(joint["upper"]),
                state_profile=profile,
                contract=contract,
            )
            if goal.get("status") != "resolved":
                raise InitialPoseContractError(
                    "canonical assignment state cannot reach required use: "
                    f"{model_key}:{use}:{goal.get('failure_code')}"
                )
            interval = articulation_target_interval(use, profile, contract)
            initial_ratio = float(goal["initial_ratio"])
            if abs(
                initial_ratio - float(raw_selection.get("reviewed_initial_ratio"))
            ) > 1e-9:
                raise InitialPoseContractError(
                    "canonical assignment reviewed ratio mismatch: "
                    f"{model_key}:{initial_ratio}"
                )
            if interval is None or float(interval[0]) <= initial_ratio <= float(interval[1]):
                raise InitialPoseContractError(
                    "canonical assignment state already satisfies target interval: "
                    f"{model_key}:{use}:{initial_ratio}"
                )


def load_assignment_state_selection(
    path: Path = DEFAULT_ASSIGNMENT_STATE_SELECTION_PATH,
) -> dict[str, Any]:
    """Load the project-local state selector without reading shared assets."""

    policy = _load_mapping(path)
    if policy.get("selection_hash") != artifact_hash(
        policy, omit_keys=("selection_hash",)
    ):
        raise InitialPoseContractError(
            "canonical assignment state selection self-hash mismatch"
        )
    return policy


def apply_assignment_joint_coordinate_revision(
    canonical_state: Mapping[str, Any],
    selection_entry: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply one explicit project-local R01 joint-state correction.

    Some legacy R01 articulated records stored prismatic coordinates in the
    unscaled URDF frame even though SAPIEN scales prismatic limits with the
    actor.  A reviewed selector may also replace a model-data joint default
    with one explicitly reviewed shared state.  Both forms are hash-bound,
    check every source coordinate, and replace stale geometry and support
    evidence.  There is no runtime inference from live limits.
    """

    revision = selection_entry.get("joint_coordinate_revision")
    if revision is None:
        return dict(canonical_state)
    if not isinstance(revision, Mapping):
        raise InitialPoseContractError("joint coordinate revision must be a mapping")
    if revision.get("artifact_kind") != "AssignmentJointCoordinateRevision":
        raise InitialPoseContractError("unexpected joint coordinate revision kind")
    if revision.get("schema_version") != 1:
        raise InitialPoseContractError("unsupported joint coordinate revision schema")
    if revision.get("review_status") != "reviewed":
        raise InitialPoseContractError("joint coordinate revision is not reviewed")
    if revision.get("revision_hash") != artifact_hash(
        revision,
        omit_keys=("revision_hash", "support_validation", "support_validation_hash"),
    ):
        raise InitialPoseContractError("joint coordinate revision self-hash mismatch")
    if revision.get("source_canonical_state_hash") != canonical_state.get(
        "canonical_state_hash"
    ):
        raise InitialPoseContractError("joint coordinate revision source state mismatch")

    result = deepcopy(dict(canonical_state))
    joint_state = deepcopy(result.get("articulated_joint_state") or {})
    source_qpos = [float(value) for value in revision.get("source_qpos") or []]
    actual_source_qpos = [float(value) for value in joint_state.get("qpos") or []]
    if source_qpos != actual_source_qpos:
        raise InitialPoseContractError("joint coordinate revision source qpos mismatch")
    runtime_qpos = [float(value) for value in revision.get("runtime_qpos") or []]
    if len(runtime_qpos) != len(source_qpos):
        raise InitialPoseContractError("joint coordinate revision qpos length mismatch")

    revised_joints = list(revision.get("runtime_joints") or [])
    source_joints = list(revision.get("source_joints") or [])
    actual_joints = list(joint_state.get("joints") or [])
    if source_joints != actual_joints:
        raise InitialPoseContractError("joint coordinate revision source limits mismatch")
    if len(revised_joints) != len(actual_joints):
        raise InitialPoseContractError("joint coordinate revision joint count mismatch")
    semantics = str(revision.get("coordinate_semantics") or "")
    if semantics == "reviewed_shared_revolute_initial_qpos":
        if revised_joints != actual_joints:
            raise InitialPoseContractError(
                "reviewed shared joint state may not change joint metadata"
            )
        changed = False
        for qpos, source_q, joint in zip(
            runtime_qpos, actual_source_qpos, actual_joints
        ):
            if "revolute" not in str(joint.get("joint_type") or "").lower():
                raise InitialPoseContractError(
                    "reviewed shared revolute state requires revolute joints"
                )
            lower = float(joint["lower"])
            upper = float(joint["upper"])
            if not lower <= qpos <= upper:
                raise InitialPoseContractError(
                    "reviewed shared revolute qpos is outside joint limits"
                )
            changed = changed or qpos != source_q
        if not changed:
            raise InitialPoseContractError(
                "reviewed shared revolute state does not change qpos"
            )
    for source_joint, runtime_joint in zip(actual_joints, revised_joints):
        if "prismatic" not in str(source_joint.get("joint_type") or "").lower():
            if source_joint != runtime_joint:
                raise InitialPoseContractError(
                    "joint coordinate revision may only change prismatic joints"
                )

    joint_state["qpos"] = runtime_qpos
    joint_state["joints"] = deepcopy(revised_joints)
    targets = []
    runtime_by_index = {
        int(row["joint_index"]): row for row in revised_joints
    }
    for target in joint_state.get("joint_targets") or []:
        row = deepcopy(target)
        index = int(row["joint_index"])
        joint = runtime_by_index[index]
        ratio = float(row["target_ratio"])
        row["target_qpos"] = _clean_float(
            float(joint["lower"])
            + ratio * (float(joint["upper"]) - float(joint["lower"]))
        )
        targets.append(row)
    joint_state["joint_targets"] = targets
    result["articulated_joint_state"] = joint_state
    source_bounds = revision.get("source_origin_relative_bounds")
    runtime_bounds = revision.get("runtime_origin_relative_bounds")
    if source_bounds is not None:
        if source_bounds != canonical_state.get("origin_relative_bounds"):
            raise InitialPoseContractError(
                "joint coordinate revision source bounds mismatch"
            )
        if not isinstance(runtime_bounds, Mapping):
            raise InitialPoseContractError(
                "joint coordinate revision runtime bounds missing"
            )
        if revision.get("source_geometry_hash") != canonical_state.get(
            "geometry_hash"
        ):
            raise InitialPoseContractError(
                "joint coordinate revision source geometry mismatch"
            )
        result["origin_relative_bounds"] = deepcopy(dict(runtime_bounds))
        result["geometry_hash"] = str(revision.get("runtime_geometry_hash") or "")
        if not result["geometry_hash"]:
            raise InitialPoseContractError(
                "joint coordinate revision runtime geometry hash missing"
            )
        runtime_support = revision.get("runtime_support")
        if not isinstance(runtime_support, Mapping):
            raise InitialPoseContractError(
                "joint coordinate revision runtime support missing"
            )
        result["support"] = deepcopy(dict(runtime_support))
    result["assignment_state_revision"] = {
        "artifact_kind": revision["artifact_kind"],
        "schema_version": revision["schema_version"],
        "revision_hash": revision["revision_hash"],
        "source_canonical_state_hash": revision["source_canonical_state_hash"],
        "coordinate_semantics": revision.get("coordinate_semantics"),
        "review_authority": revision.get("review_authority"),
        "runtime_geometry_hash": revision.get("runtime_geometry_hash"),
    }
    support = revision.get("support_validation")
    support_hash = revision.get("support_validation_hash")
    result.pop("support_validation", None)
    result.pop("support_validation_hash", None)
    canonical_payload = dict(result)
    canonical_payload.pop("canonical_state_hash", None)
    result["canonical_state_hash"] = artifact_hash(canonical_payload)
    expected_effective_hash = selection_entry.get("effective_canonical_state_hash")
    if expected_effective_hash not in (None, result["canonical_state_hash"]):
        raise InitialPoseContractError("joint coordinate effective state hash mismatch")
    if support is not None:
        if support_hash != artifact_hash(support):
            raise InitialPoseContractError("joint coordinate support evidence hash mismatch")
        result["support_validation"] = deepcopy(support)
        result["support_validation_hash"] = str(support_hash)
    elif support_hash is not None:
        raise InitialPoseContractError("orphan joint coordinate support hash")
    return result


def canonical_assignment_states(
    document: Mapping[str, Any],
    selection: Mapping[str, Any] | None = None,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Select one immutable R01 state per asset/model for M4 and M8+.

    Explicit reviewed selections override the legacy ``default`` profile.
    Most rows retain the frozen R01 identity verbatim; a selector may also
    carry an explicit reviewed joint-coordinate revision with its own geometry,
    approval, support evidence, and hashes.  Live runtime state is never used
    to infer or repair qpos here.
    """

    policy = selection or {}
    if selection is not None:
        validate_assignment_state_selection(
            policy, document, require_support_validated=True
        )
    explicit = {
        str(key): str(value.get("initial_state_id") or "")
        for key, value in (policy.get("selections") or {}).items()
        if isinstance(value, Mapping)
    }
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for row in document.get("canonical_initial_states") or []:
        if row.get("status") != "approved":
            continue
        if (row.get("support_validation") or {}).get("status") != "validated":
            continue
        key = (str(row["asset"]), str(row["model_id"]))
        model_key = f"{key[0]}|{key[1]}"
        selected_id = explicit.get(model_key)
        if selected_id:
            if str(row.get("initial_state_id")) == selected_id:
                selected[key] = apply_assignment_joint_coordinate_revision(
                    row,
                    (policy.get("selections") or {}).get(model_key) or {},
                )
            continue
        if str(row.get("initial_state_profile_id")) == "default":
            selected[key] = dict(row)
    missing = sorted(set(explicit) - {f"{a}|{m}" for a, m in selected})
    if missing:
        raise InitialPoseContractError(
            f"canonical assignment state selections are unresolved: {missing}"
        )
    _validate_open_close_neutral_assignment_states(
        selected,
        selections=policy.get("selections") or {},
    )
    return selected


def _validate_open_close_neutral_assignment_states(
    states: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    selections: Mapping[str, Any],
) -> None:
    """Require one genuinely neutral shared state for every open/close model.

    Model-data ``qpos=0`` is not necessarily neutral because articulated joint
    limits are often asymmetric.  This gate runs after explicit selection and
    default fallback have both been resolved, so M4 and M8+ cannot silently
    consume a state that already satisfies either terminal predicate.
    """

    from if_benchmark.success_contract import (
        SuccessContract,
        articulation_target_interval,
        resolve_articulation_goal,
    )

    contract = SuccessContract.load()
    for (asset, model_id), state in sorted(states.items()):
        if "articulated_open_close" not in {
            str(value) for value in state.get("candidate_uses") or []
        }:
            continue
        model_key = f"{asset}|{model_id}"
        selection = selections.get(model_key) or {}
        joint_state = state.get("articulated_joint_state") or {}
        qpos = list(joint_state.get("qpos") or [])
        joints = list(joint_state.get("joints") or [])
        joint_index = selection.get("joint_index")
        if joint_index is None:
            joint_index = joint_state.get("joint_index")
        joint_index = 0 if joint_index is None else int(joint_index)
        joint = next(
            (
                row
                for row in joints
                if int(row.get("joint_index", -1)) == joint_index
            ),
            None,
        )
        if joint is None or joint_index < 0 or joint_index >= len(qpos):
            raise InitialPoseContractError(
                f"open/close neutral state lacks active joint: {model_key}"
            )
        profile = {
            "positive_use": str(selection.get("positive_use") or "open")
        }
        for use in ("open", "close"):
            goal = resolve_articulation_goal(
                state=use,
                initial_qpos=float(qpos[joint_index]),
                lower=float(joint["lower"]),
                upper=float(joint["upper"]),
                state_profile=profile,
                contract=contract,
            )
            if goal.get("status") != "resolved":
                raise InitialPoseContractError(
                    "open/close shared initial state cannot reach target: "
                    f"{model_key}:{use}:{goal.get('failure_code')}"
                )
            interval = articulation_target_interval(use, profile, contract)
            ratio = float(goal["initial_ratio"])
            if interval is None or float(interval[0]) <= ratio <= float(interval[1]):
                raise InitialPoseContractError(
                    "open/close shared initial state is not neutral: "
                    f"{model_key}:{use}:{ratio}"
                )


def canonical_assignment_selection_identity(
    selection: Mapping[str, Any] | None,
) -> dict[str, str]:
    """Return the selector identity recorded by downstream artifacts."""

    policy = selection
    if not isinstance(policy, Mapping):
        return {
            "selection_version": "legacy_default_profile_selection",
            "selection_hash": artifact_hash(
                {"selection_version": "legacy_default_profile_selection"}
            ),
        }
    return {
        "selection_version": str(policy["selection_version"]),
        "selection_hash": str(policy["selection_hash"]),
    }


def validate_initial_pose_set(
    document: Mapping[str, Any], *, require_support_validated: bool
) -> None:
    expected_kind = (
        "support_validated_initial_pose_set"
        if require_support_validated
        else "approved_initial_pose_set"
    )
    hash_key = (
        "validated_pose_set_hash"
        if require_support_validated
        else "approved_pose_set_hash"
    )
    if document.get("artifact_kind") != expected_kind:
        raise InitialPoseContractError(f"expected {expected_kind}")
    if document.get(hash_key) != artifact_hash(document, omit_keys=(hash_key,)):
        raise InitialPoseContractError("Initial Pose Set self-hash mismatch")
    support_config_hash: str | None = None
    if require_support_validated:
        if document.get("support_contract_version") != SUPPORT_CONTRACT_VERSION:
            raise InitialPoseContractError(
                "Initial Pose Set support contract is stale"
            )
        if (
            document.get("dynamic_asset_mass_inertia_policy_version")
            != DYNAMIC_ASSET_MASS_INERTIA_POLICY_VERSION
        ):
            raise InitialPoseContractError(
                "Initial Pose Set mass/inertia policy is stale"
            )
        support_config_hash = document.get("support_config_hash")
        if not isinstance(support_config_hash, str) or not support_config_hash:
            raise InitialPoseContractError(
                "Initial Pose Set lacks support config identity"
            )
    states = list(document.get("canonical_initial_states") or [])
    approvals = list(document.get("approvals") or [])
    state_ids = [str(row.get("initial_state_id")) for row in states]
    approval_ids = [str(row.get("initial_state_id")) for row in approvals]
    if len(state_ids) != len(set(state_ids)):
        raise InitialPoseContractError("duplicate CanonicalInitialState id")
    if len(approval_ids) != len(set(approval_ids)):
        raise InitialPoseContractError("duplicate InitialPoseApproval id")
    if set(state_ids) != set(approval_ids):
        raise InitialPoseContractError(
            "CanonicalInitialState/InitialPoseApproval coverage mismatch"
        )
    approval_by_id = {
        str(row["initial_state_id"]): row for row in approvals
    }
    for state in states:
        if require_support_validated and not isinstance(
            state.get("dynamic_mass_properties"), Mapping
        ):
            raise InitialPoseContractError(
                "support-validated state lacks dynamic mass contract"
            )
        validate_canonical_initial_state(state)
        approval = approval_by_id[str(state["initial_state_id"])]
        validate_initial_pose_approval(approval)
        for key in (
            "initial_state_id",
            "revision",
            "revision_hash",
            "geometry_hash",
        ):
            if state.get(key) != approval.get(key):
                raise InitialPoseContractError(
                    f"state/approval identity mismatch: {key}"
                )
        if state.get("status") != approval.get("decision"):
            raise InitialPoseContractError("state/approval decision mismatch")
        if require_support_validated:
            support_validation = state.get("support_validation") or {}
            support_status = support_validation.get("status")
            expected_support_status = (
                "validated"
                if state.get("status") == "approved"
                else "not_applicable_human_unsupported"
            )
            if support_status != expected_support_status:
                raise InitialPoseContractError(
                    f"invalid support validation status: {support_status!r}"
                )
            if support_validation.get("support_config_hash") != support_config_hash:
                raise InitialPoseContractError(
                    "state/support-set config identity mismatch"
                )


def approval_for_state(approved_set: Mapping[str, Any], initial_state_id: str) -> dict[str, Any]:
    approval = next(
        (dict(row) for row in approved_set.get("approvals") or [] if row.get("initial_state_id") == initial_state_id),
        None,
    )
    if approval is None or approval.get("decision") != "approved":
        raise InitialPoseContractError(f"state is not human-approved: {initial_state_id}")
    validate_initial_pose_approval(approval)
    return approval


def assignment_approval_for_state(
    document: Mapping[str, Any],
    canonical_state: Mapping[str, Any],
    selection_entry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve the approval bound to an effective assignment-state revision."""

    entry = selection_entry or {}
    effective = entry.get("effective_approval")
    if effective is None:
        return approval_for_state(document, str(canonical_state["initial_state_id"]))
    if not isinstance(effective, Mapping):
        raise InitialPoseContractError("effective assignment approval is malformed")
    approval = dict(effective)
    validate_initial_pose_approval(approval)
    expected = {
        "initial_state_id": canonical_state.get("initial_state_id"),
        "revision": canonical_state.get("revision"),
        "revision_hash": canonical_state.get("revision_hash"),
        "geometry_hash": canonical_state.get("geometry_hash"),
        "decision": canonical_state.get("status"),
    }
    mismatches = {
        key: {"expected": value, "actual": approval.get(key)}
        for key, value in expected.items()
        if approval.get(key) != value
    }
    if mismatches:
        raise InitialPoseContractError(
            f"effective assignment approval identity mismatch: {mismatches}"
        )
    return approval


def validate_pose_binding(
    binding: Mapping[str, Any],
    canonical_state: Mapping[str, Any],
    approval: Mapping[str, Any],
) -> None:
    """Validate a planner/worker binding against exact approved identities."""

    expected = {
        "initial_state_id": canonical_state.get("initial_state_id"),
        "revision": canonical_state.get("revision"),
        "approval_hash": approval.get("approval_hash"),
        "geometry_hash": canonical_state.get("geometry_hash"),
    }
    mismatches = {
        key: {"expected": value, "actual": binding.get(key)}
        for key, value in expected.items()
        if binding.get(key) != value
    }
    if mismatches:
        raise InitialPoseContractError(f"initial pose binding mismatch: {mismatches}")


def _current_support_config() -> dict[str, Any]:
    """Return the exact source- and policy-bound support configuration."""

    repository_root = Path(__file__).resolve().parents[2]
    return {
        **SUPPORT_CONFIG,
        "implementation_hashes": {
            "actor_mass_dispatch": file_sha256(
                repository_root / "envs" / "utils" / "actor_utils.py"
            ),
            "actor_construction": file_sha256(
                repository_root / "envs" / "utils" / "create_actor.py"
            ),
            "dynamic_mass_properties": file_sha256(
                repository_root / "envs" / "utils" / "physical_properties.py"
            ),
            "support_worker": file_sha256(
                Path(__file__).with_name("support_worker.py")
            ),
            "support_environment": file_sha256(
                repository_root / "envs" / "if_initial_pose_support.py"
            ),
            "initial_pose_runtime": file_sha256(
                Path(__file__).with_name("runtime.py")
            ),
        },
    }


def _validate_current_support_plan(plan: Mapping[str, Any]) -> None:
    """Require a support plan produced by this exact physical-policy epoch."""

    if plan.get("artifact_kind") != "initial_pose_support_probe_plan":
        raise InitialPoseContractError("unexpected support plan kind")
    if plan.get("plan_hash") != artifact_hash(plan, omit_keys=("plan_hash",)):
        raise InitialPoseContractError("support plan self-hash mismatch")
    if plan.get("support_contract_version") != SUPPORT_CONTRACT_VERSION:
        raise InitialPoseContractError("support plan contract is stale")
    expected_config = _current_support_config()
    expected_config_hash = artifact_hash(expected_config)
    if plan.get("config") != expected_config:
        raise InitialPoseContractError(
            "support plan configuration is stale for current runtime"
        )
    if plan.get("config_hash") != expected_config_hash:
        raise InitialPoseContractError("support plan config hash mismatch")
    for job in plan.get("jobs") or []:
        if job.get("support_contract_version") != SUPPORT_CONTRACT_VERSION:
            raise InitialPoseContractError(
                "support plan contains a stale-contract job"
            )
        if job.get("support_config_hash") != expected_config_hash:
            raise InitialPoseContractError(
                "support plan job/config identity mismatch"
            )


def _dynamic_mass_properties_for_state(
    state: Mapping[str, Any], catalog: Mapping[str, Any]
) -> dict[str, Any]:
    if "dynamic_mass_properties" in state:
        embedded = state.get("dynamic_mass_properties")
        if not isinstance(embedded, Mapping):
            raise InitialPoseContractError(
                "embedded dynamic mass contract is not a mapping"
            )
        contract = dict(embedded)
        validate_dynamic_mass_properties(contract)
        return contract
    candidates = _catalog_candidates_for_state(state, catalog)
    contract = _model_dynamic_mass_properties(candidates)
    validate_dynamic_mass_properties(contract)
    return contract


def _dynamic_mass_properties_map(
    states: Iterable[Mapping[str, Any]], catalog: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for state in states:
        state_id = state.get("initial_state_id")
        if not isinstance(state_id, str) or not state_id:
            raise InitialPoseContractError(
                "cannot bind dynamic mass contract without an initial state id"
            )
        if state_id in result:
            raise InitialPoseContractError(
                f"duplicate dynamic mass state id: {state_id}"
            )
        result[state_id] = _dynamic_mass_properties_for_state(state, catalog)
    return dict(sorted(result.items()))


def build_support_plan(
    *,
    approved_set_path: Path,
    output_path: Path,
    asset_catalog_path: Path = DEFAULT_ASSET_CATALOG_PATH,
) -> dict[str, Any]:
    approved = _load_mapping(approved_set_path)
    validate_initial_pose_set(approved, require_support_validated=False)
    catalog = _load_mapping(asset_catalog_path)
    dynamic_mass_catalog_hash = _catalog_hash_for_derivation(asset_catalog_path)
    dynamic_mass_by_state = _dynamic_mass_properties_map(
        approved.get("canonical_initial_states") or [], catalog
    )
    support_config = _current_support_config()
    support_config_hash = artifact_hash(support_config)
    jobs = []
    for state in approved.get("canonical_initial_states") or []:
        if state.get("status") != "approved":
            continue
        approval = approval_for_state(approved, str(state["initial_state_id"]))
        dynamic_mass_properties = dynamic_mass_by_state[
            str(state["initial_state_id"])
        ]
        articulated = (
            (state.get("articulated_joint_state") or {}).get("mode")
            != "not_articulated"
        )
        support_mode = (
            support_config["root_modes"]["articulated"]
            if articulated
            else support_config["root_modes"]["rigid"]
        )
        yaw_policy = state.get("yaw_policy") or {}
        if yaw_policy.get("mode") == "free":
            yaws = DEFAULT_FREE_YAWS_DEG
        elif yaw_policy.get("mode") == "fixed":
            yaws = (float(yaw_policy.get("yaw_deg") or 0.0),)
        else:
            raise InitialPoseContractError(f"invalid yaw policy for {state['initial_state_id']}")
        for yaw_deg in yaws:
            for seed in DEFAULT_SUPPORT_SEEDS:
                binding = {
                    "initial_state_id": state["initial_state_id"],
                    "revision": state["revision"],
                    "approval_hash": approval["approval_hash"],
                    "geometry_hash": state["geometry_hash"],
                    "dynamic_mass_properties": dynamic_mass_properties,
                }
                job = {
                    "job_id": deterministic_id(
                        "ips",
                        {**binding, "yaw_deg": yaw_deg, "seed": seed},
                        length=20,
                    ),
                    **binding,
                    "asset": state["asset"],
                    "model_id": state["model_id"],
                    "initial_state_profile_id": state["initial_state_profile_id"],
                    "q0_wxyz": state["q0_wxyz"],
                    "articulated_joint_state": state["articulated_joint_state"],
                    "scale_xyz": state["scale_xyz"],
                    "support": state["support"],
                    "yaw_policy": state["yaw_policy"],
                    "yaw_deg": _clean_float(yaw_deg),
                    "seed": int(seed),
                    "required_metrics": support_config["required_metrics"],
                    "support_contract_version": SUPPORT_CONTRACT_VERSION,
                    "support_config_hash": support_config_hash,
                    "support_mode": support_mode,
                }
                validate_pose_binding(job, state, approval)
                jobs.append(job)
    result: dict[str, Any] = {
        "artifact_kind": "initial_pose_support_probe_plan",
        "schema_version": SCHEMA_VERSION,
        "support_contract_version": SUPPORT_CONTRACT_VERSION,
        "approved_pose_set_hash": approved["approved_pose_set_hash"],
        "dynamic_mass_catalog_hash": dynamic_mass_catalog_hash,
        "dynamic_mass_properties_by_state": dynamic_mass_by_state,
        "config": support_config,
        "config_hash": support_config_hash,
        "summary": {
            "approved_state_count": int((approved.get("summary") or {}).get("approved_count") or 0),
            "job_count": len(jobs),
            "seed_count": len(DEFAULT_SUPPORT_SEEDS),
        },
        "jobs": sorted(jobs, key=lambda row: row["job_id"]),
    }
    result["plan_hash"] = artifact_hash(result)
    _validate_current_support_plan(result)
    atomic_yaml(output_path, result)
    return result


def _assignment_selection_revision_states(
    approved: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> list[tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]]:
    """Resolve only the selector-owned effective joint-state revisions.

    The selector may contain old support evidence so that its reviewed state
    semantics remain readable during a policy transition.  That evidence is
    deliberately removed here: every returned state must earn fresh support
    evidence for its effective qpos through the new plan.
    """

    source_is_validated = (
        approved.get("artifact_kind") == "support_validated_initial_pose_set"
    )
    validate_initial_pose_set(
        approved, require_support_validated=source_is_validated
    )
    validate_assignment_state_selection(
        selection,
        approved,
        require_support_validated=source_is_validated,
    )
    states_by_id = {
        str(row.get("initial_state_id")): row
        for row in approved.get("canonical_initial_states") or []
    }
    resolved = []
    for model_key, raw_entry in sorted((selection.get("selections") or {}).items()):
        if not isinstance(raw_entry, Mapping):
            continue
        revision = raw_entry.get("joint_coordinate_revision")
        if revision is None:
            continue
        state_id = str(raw_entry.get("initial_state_id") or "")
        source_state = states_by_id.get(state_id)
        if source_state is None:
            raise InitialPoseContractError(
                f"assignment selection support source state is missing: {model_key}:{state_id}"
            )
        effective_state = apply_assignment_joint_coordinate_revision(
            source_state,
            raw_entry,
        )
        # A revised qpos is the subject of this sweep.  Neither the base
        # state's support proof nor a stale selector-local proof is input.
        effective_state.pop("support_validation", None)
        effective_state.pop("support_validation_hash", None)
        validate_canonical_initial_state(effective_state)
        approval = assignment_approval_for_state(
            approved,
            effective_state,
            raw_entry,
        )
        resolved.append(
            (
                str(model_key),
                effective_state,
                approval,
                dict(raw_entry),
            )
        )
    if not resolved:
        raise InitialPoseContractError(
            "canonical assignment selector has no joint-coordinate revisions"
        )
    return resolved


def _assignment_selection_support_plan_payload(
    approved: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    support_config = _current_support_config()
    support_config_hash = artifact_hash(support_config)
    catalog = _load_mapping(DEFAULT_ASSET_CATALOG_PATH)
    dynamic_mass_catalog_hash = _catalog_hash_for_derivation(
        DEFAULT_ASSET_CATALOG_PATH
    )
    jobs = []
    revisions = _assignment_selection_revision_states(approved, selection)
    dynamic_mass_by_state = _dynamic_mass_properties_map(
        (state for _, state, _, _ in revisions), catalog
    )
    for model_key, state, approval, selection_entry in revisions:
        revision = selection_entry["joint_coordinate_revision"]
        articulated = (
            (state.get("articulated_joint_state") or {}).get("mode")
            != "not_articulated"
        )
        if not articulated:
            raise InitialPoseContractError(
                f"assignment joint-coordinate revision is not articulated: {model_key}"
            )
        support_mode = support_config["root_modes"]["articulated"]
        dynamic_mass_properties = dynamic_mass_by_state[
            str(state["initial_state_id"])
        ]
        yaw_policy = state.get("yaw_policy") or {}
        if yaw_policy.get("mode") == "free":
            yaws = DEFAULT_FREE_YAWS_DEG
        elif yaw_policy.get("mode") == "fixed":
            yaws = (float(yaw_policy.get("yaw_deg") or 0.0),)
        else:
            raise InitialPoseContractError(
                f"invalid yaw policy for {state['initial_state_id']}"
            )
        revision_binding = {
            "assignment_model_key": model_key,
            "assignment_selection_hash": selection["selection_hash"],
            "assignment_revision_hash": revision["revision_hash"],
            "effective_canonical_state_hash": state["canonical_state_hash"],
        }
        for yaw_deg in yaws:
            for seed in DEFAULT_SUPPORT_SEEDS:
                binding = {
                    "initial_state_id": state["initial_state_id"],
                    "revision": state["revision"],
                    "approval_hash": approval["approval_hash"],
                    "geometry_hash": state["geometry_hash"],
                    "dynamic_mass_properties": dynamic_mass_properties,
                }
                job = {
                    "job_id": deterministic_id(
                        "ips",
                        {
                            **binding,
                            **revision_binding,
                            "yaw_deg": yaw_deg,
                            "seed": seed,
                        },
                        length=20,
                    ),
                    **binding,
                    **revision_binding,
                    "asset": state["asset"],
                    "model_id": state["model_id"],
                    "initial_state_profile_id": state[
                        "initial_state_profile_id"
                    ],
                    "q0_wxyz": state["q0_wxyz"],
                    "articulated_joint_state": state[
                        "articulated_joint_state"
                    ],
                    "scale_xyz": state["scale_xyz"],
                    "support": state["support"],
                    "yaw_policy": state["yaw_policy"],
                    "yaw_deg": _clean_float(yaw_deg),
                    "seed": int(seed),
                    "required_metrics": support_config["required_metrics"],
                    "support_contract_version": SUPPORT_CONTRACT_VERSION,
                    "support_config_hash": support_config_hash,
                    "support_mode": support_mode,
                }
                validate_pose_binding(job, state, approval)
                jobs.append(job)
    result: dict[str, Any] = {
        # Keep the existing kind/report schema so run_support_plan remains the
        # one physical execution implementation.
        "artifact_kind": "initial_pose_support_probe_plan",
        "schema_version": SCHEMA_VERSION,
        "support_plan_scope": ASSIGNMENT_SELECTION_SUPPORT_SCOPE,
        "support_contract_version": SUPPORT_CONTRACT_VERSION,
        "approved_pose_set_hash": approved["approved_pose_set_hash"],
        "dynamic_mass_catalog_hash": dynamic_mass_catalog_hash,
        "dynamic_mass_properties_by_state": dynamic_mass_by_state,
        "assignment_state_selection_version": selection["selection_version"],
        "source_assignment_state_selection_hash": selection["selection_hash"],
        "config": support_config,
        "config_hash": support_config_hash,
        "summary": {
            "joint_revision_state_count": len(revisions),
            "job_count": len(jobs),
            "seed_count": len(DEFAULT_SUPPORT_SEEDS),
        },
        "jobs": sorted(jobs, key=lambda row: row["job_id"]),
    }
    result["plan_hash"] = artifact_hash(result)
    return result


def validate_assignment_selection_support_plan(
    plan: Mapping[str, Any],
    *,
    approved: Mapping[str, Any] | None = None,
    selection: Mapping[str, Any] | None = None,
) -> None:
    """Validate the current scoped plan, optionally against exact sources."""

    _validate_current_support_plan(plan)
    if plan.get("support_plan_scope") != ASSIGNMENT_SELECTION_SUPPORT_SCOPE:
        raise InitialPoseContractError(
            "unexpected assignment selection support plan scope"
        )
    if (
        plan.get("assignment_state_selection_version")
        != ASSIGNMENT_STATE_SELECTION_VERSION
    ):
        raise InitialPoseContractError(
            "assignment selection support plan selector version is stale"
        )
    jobs = list(plan.get("jobs") or [])
    job_ids = [str(row.get("job_id") or "") for row in jobs]
    if not jobs or "" in job_ids or len(job_ids) != len(set(job_ids)):
        raise InitialPoseContractError(
            "assignment selection support plan has invalid job coverage"
        )
    source_selection_hash = str(
        plan.get("source_assignment_state_selection_hash") or ""
    )
    for job in jobs:
        for key in (
            "assignment_model_key",
            "assignment_revision_hash",
            "effective_canonical_state_hash",
        ):
            if not str(job.get(key) or ""):
                raise InitialPoseContractError(
                    f"assignment selection support job lacks binding: {key}"
                )
        if job.get("assignment_selection_hash") != source_selection_hash:
            raise InitialPoseContractError(
                "assignment selection support job/selector hash mismatch"
            )
    if approved is not None or selection is not None:
        if approved is None or selection is None:
            raise InitialPoseContractError(
                "exact assignment support plan validation requires both sources"
            )
        expected = _assignment_selection_support_plan_payload(
            approved,
            selection,
        )
        if dict(plan) != expected:
            raise InitialPoseContractError(
                "assignment selection support plan is stale for exact sources"
            )


def build_assignment_selection_support_plan(
    *,
    approved_set_path: Path,
    selection_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    approved = _load_mapping(approved_set_path)
    selection = _load_mapping(selection_path)
    result = _assignment_selection_support_plan_payload(approved, selection)
    validate_assignment_selection_support_plan(
        result,
        approved=approved,
        selection=selection,
    )
    atomic_yaml(output_path, result)
    return result


def validate_support_report(
    report: Mapping[str, Any],
    job: Mapping[str, Any],
    *,
    expected_plan_hash: str | None = None,
) -> None:
    if report.get("artifact_kind") != "initial_pose_support_report":
        raise InitialPoseContractError("unexpected support report kind")
    if report.get("report_hash") != artifact_hash(report, omit_keys=("report_hash",)):
        raise InitialPoseContractError("support report self-hash mismatch")
    if job.get("support_contract_version") != SUPPORT_CONTRACT_VERSION:
        raise InitialPoseContractError(
            "support report job contract is stale for current runtime"
        )
    for key in (
        "job_id",
        "initial_state_id",
        "revision",
        "approval_hash",
        "geometry_hash",
        "support_contract_version",
        "support_config_hash",
        "support_mode",
    ):
        if job.get(key) in (None, ""):
            raise InitialPoseContractError(f"support job lacks binding: {key}")
        if report.get(key) != job.get(key):
            raise InitialPoseContractError(f"support report binding mismatch: {key}")
    if expected_plan_hash is not None and report.get("source_plan_hash") != expected_plan_hash:
        raise InitialPoseContractError("support report is stale for support plan")
    outcome = report.get("outcome")
    if outcome not in {"pass", "fail", "source_failure", "internal_failure"}:
        raise InitialPoseContractError("support report has invalid outcome")
    metrics = report.get("metrics") or {}
    if bool(report.get("fallback_pose_used")):
        raise InitialPoseContractError("support report used forbidden fallback pose")
    if outcome in {"pass", "fail"}:
        missing = sorted(set(SUPPORT_CONFIG["required_metrics"]) - set(metrics))
        if missing:
            raise InitialPoseContractError(f"support report lacks metrics: {missing}")
        if metrics.get("support_mode") != report.get("support_mode"):
            raise InitialPoseContractError(
                "support report metric/root mode binding mismatch"
            )
        _validate_support_mass_preflight(report, job)
    if outcome == "pass" and bool(metrics.get("initial_robot_contact")):
        raise InitialPoseContractError(
            "passing support report has initial robot contact"
        )
    if outcome == "pass":
        thresholds = SUPPORT_CONFIG["pass_thresholds"]
        maximums = {
            "spawn_to_settled_translation_m": thresholds[
                "spawn_to_settled_translation_m_max"
            ],
            "spawn_to_settled_rotation_deg": thresholds[
                "spawn_to_settled_rotation_deg_max"
            ],
            "tail_translation_drift_m": thresholds[
                "tail_translation_drift_m_max"
            ],
            "tail_rotation_drift_deg": thresholds[
                "tail_rotation_drift_deg_max"
            ],
            "bottom_gap_m": thresholds["bottom_gap_m_max"],
            "bottom_penetration_m": thresholds[
                "bottom_penetration_m_max"
            ],
        }
        violations = {
            key: {"actual": float(metrics[key]), "maximum": float(maximum)}
            for key, maximum in maximums.items()
            if not math.isfinite(float(metrics[key]))
            or float(metrics[key]) > float(maximum)
        }
        edge_margin = float(metrics["table_edge_margin_m"])
        if not math.isfinite(edge_margin) or edge_margin < float(
            thresholds["table_edge_margin_m_min"]
        ):
            violations["table_edge_margin_m"] = {
                "actual": edge_margin,
                "minimum": thresholds["table_edge_margin_m_min"],
            }
        impulse = float(metrics["support_impulse_ns"])
        if not math.isfinite(impulse):
            violations["support_impulse_ns"] = {
                "actual": impulse,
                "requirement": "finite",
            }
        elif report.get("support_mode") == "dynamic_table_support":
            if impulse <= float(
                thresholds["support_impulse_ns_min_exclusive"]
            ):
                violations["support_impulse_ns"] = {
                    "actual": impulse,
                    "minimum_exclusive": thresholds[
                        "support_impulse_ns_min_exclusive"
                    ],
                }
        elif report.get("support_mode") != "fixed_root_fixture":
            violations["support_mode"] = {
                "actual": report.get("support_mode"),
                "allowed": ["dynamic_table_support", "fixed_root_fixture"],
            }
        if violations:
            raise InitialPoseContractError(
                f"passing support report violates thresholds: {violations}"
            )


def _validate_support_mass_preflight(
    report: Mapping[str, Any], job: Mapping[str, Any]
) -> None:
    expected = job.get("dynamic_mass_properties")
    if not isinstance(expected, Mapping):
        raise InitialPoseContractError(
            "support job lacks dynamic mass contract"
        )
    validate_dynamic_mass_properties(expected)
    preflight = report.get("preflight")
    if not isinstance(preflight, Mapping):
        raise InitialPoseContractError(
            "support report lacks mass preflight"
        )
    if preflight.get("dynamic_mass_properties") != expected:
        raise InitialPoseContractError(
            "support mass contract/preflight mismatch"
        )
    rows = preflight.get("actual_mass_properties")
    if not isinstance(rows, list) or not rows:
        raise InitialPoseContractError(
            "support preflight lacks actual mass properties"
        )
    target = expected.get("target_mass_kg")
    for row in rows:
        try:
            mass = float(row["mass_kg"])
            inertia = [
                float(value)
                for value in row["principal_inertia_kg_m2"]
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise InitialPoseContractError(
                "support preflight mass properties are incomplete"
            ) from exc
        if (
            not math.isfinite(mass)
            or mass <= 0.0
            or len(inertia) != 3
            or any(not math.isfinite(value) or value <= 0.0 for value in inertia)
        ):
            raise InitialPoseContractError(
                "support preflight mass properties are invalid"
            )
        if target is not None and not math.isclose(
            mass,
            float(target),
            rel_tol=1e-6,
            abs_tol=1e-8,
        ):
            raise InitialPoseContractError(
                "support preflight explicit mass was not applied"
            )


def _validate_direct_support_preflight(
    report: Mapping[str, Any],
    job: Mapping[str, Any],
) -> None:
    """Prove that the worker loaded the plan's effective state, including qpos."""

    preflight = report.get("preflight")
    if not isinstance(preflight, Mapping) or preflight.get("status") != "matched":
        raise InitialPoseContractError(
            "assignment selection support report lacks matched preflight"
        )
    if preflight.get("geometry_hash") != job.get("geometry_hash"):
        raise InitialPoseContractError(
            "assignment selection support preflight geometry mismatch"
        )
    from .runtime import validate_loaded_initial_pose

    try:
        validate_loaded_initial_pose(
            canonical_state=job,
            actual_q_wxyz=preflight["actual_q_wxyz"],
            actual_joint_qpos=preflight.get("actual_joint_qpos"),
            actual_scale_xyz=preflight["actual_scale_xyz"],
            actual_origin_z=float(preflight["actual_origin_z"]),
            expected_world_yaw_deg=float(job["yaw_deg"]),
        )
    except InitialPoseContractError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise InitialPoseContractError(
            "assignment selection support preflight is incomplete"
        ) from exc


def freeze_assignment_selection_support(
    *,
    approved_set_path: Path,
    validated_pose_set_path: Path | None = None,
    selection_path: Path,
    support_plan_path: Path,
    reports_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Freeze fresh support evidence into a new run-local selector."""

    if output_path.resolve() == selection_path.resolve():
        raise InitialPoseContractError(
            "assignment selection support refresh must write a run-local selector"
        )
    approved = _load_mapping(approved_set_path)
    validated_pose_set: dict[str, Any] | None = None
    validated_states: dict[str, dict[str, Any]] = {}
    if validated_pose_set_path is not None:
        validated_pose_set = _load_mapping(validated_pose_set_path)
        validate_initial_pose_set(
            validated_pose_set, require_support_validated=True
        )
        if (
            validated_pose_set.get("approved_pose_set_hash")
            != approved.get("approved_pose_set_hash")
        ):
            raise InitialPoseContractError(
                "assignment support validated/approved pose-set mismatch"
            )
        validated_states = {
            str(state["initial_state_id"]): dict(state)
            for state in validated_pose_set.get("canonical_initial_states") or []
        }
    selection = _load_mapping(selection_path)
    plan = _load_mapping(support_plan_path)
    validate_assignment_selection_support_plan(
        plan,
        approved=(
            validated_pose_set
            if validated_pose_set is not None
            else approved
        ),
        selection=selection,
    )

    by_model: dict[str, list[dict[str, Any]]] = {}
    blockers = []
    for job in plan.get("jobs") or []:
        path = reports_dir / f"{job['job_id']}.json"
        if not path.is_file():
            blockers.append({"job_id": job["job_id"], "reason": "missing_report"})
            continue
        report = _load_mapping(path)
        try:
            validate_support_report(
                report,
                job,
                expected_plan_hash=str(plan["plan_hash"]),
            )
            if report.get("outcome") != "pass":
                raise InitialPoseContractError(
                    f"support_{report.get('outcome')}"
                )
            _validate_direct_support_preflight(report, job)
        except InitialPoseContractError as exc:
            blockers.append({"job_id": job["job_id"], "reason": str(exc)})
            continue
        by_model.setdefault(str(job["assignment_model_key"]), []).append(report)
    if blockers:
        preview = ", ".join(
            f"{row['job_id']}:{row['reason']}" for row in blockers[:8]
        )
        raise InitialPoseContractError(
            "assignment selection support validation blocked by "
            f"{len(blockers)} job(s): {preview}"
        )

    result = deepcopy(selection)
    approved_states = {
        str(state["initial_state_id"]): dict(state)
        for state in approved.get("canonical_initial_states") or []
    }
    jobs_by_model: dict[str, list[dict[str, Any]]] = {}
    for job in plan.get("jobs") or []:
        jobs_by_model.setdefault(str(job["assignment_model_key"]), []).append(job)

    def pose_projection(state: Mapping[str, Any]) -> dict[str, Any]:
        projection = dict(state)
        for key in (
            "canonical_state_hash",
            "dynamic_mass_properties",
            "support_validation",
            "support_validation_hash",
        ):
            projection.pop(key, None)
        return projection

    for model_key, raw_entry in sorted((result.get("selections") or {}).items()):
        if not isinstance(raw_entry, dict):
            continue
        validated_state: dict[str, Any] | None = None
        if validated_pose_set is not None:
            state_id = str(raw_entry["initial_state_id"])
            approved_state = approved_states.get(state_id)
            validated_state = validated_states.get(state_id)
            if approved_state is None or validated_state is None:
                raise InitialPoseContractError(
                    f"assignment support rebind state missing: {model_key}"
                )
            if pose_projection(approved_state) != pose_projection(validated_state):
                raise InitialPoseContractError(
                    "assignment support rebind changes pose semantics: "
                    f"{model_key}"
                )
            raw_entry["canonical_state_hash"] = validated_state[
                "canonical_state_hash"
            ]
        revision = raw_entry.get("joint_coordinate_revision")
        if not isinstance(revision, dict):
            if validated_state is not None:
                raw_entry["effective_canonical_state_hash"] = validated_state[
                    "canonical_state_hash"
                ]
                raw_entry["support_validation_hash"] = validated_state[
                    "support_validation_hash"
                ]
            continue
        source_assignment_revision_hash = str(revision["revision_hash"])
        source_effective_state_hash = str(
            raw_entry["effective_canonical_state_hash"]
        )
        effective_state: dict[str, Any] | None = None
        if validated_state is not None:
            revision.pop("support_validation", None)
            revision.pop("support_validation_hash", None)
            revision["source_canonical_state_hash"] = validated_state[
                "canonical_state_hash"
            ]
            revision["revision_hash"] = artifact_hash(
                revision,
                omit_keys=(
                    "revision_hash",
                    "support_validation",
                    "support_validation_hash",
                ),
            )
            raw_entry["effective_canonical_state_hash"] = None
            effective_state = apply_assignment_joint_coordinate_revision(
                validated_state, raw_entry
            )
            raw_entry["effective_canonical_state_hash"] = effective_state[
                "canonical_state_hash"
            ]
            for job in jobs_by_model.get(str(model_key), []):
                expected = {
                    "geometry_hash": effective_state["geometry_hash"],
                    "dynamic_mass_properties": effective_state[
                        "dynamic_mass_properties"
                    ],
                    "q0_wxyz": effective_state["q0_wxyz"],
                    "articulated_joint_state": effective_state[
                        "articulated_joint_state"
                    ],
                    "scale_xyz": effective_state["scale_xyz"],
                    "support": effective_state["support"],
                    "yaw_policy": effective_state["yaw_policy"],
                }
                mismatches = {
                    key: {"expected": value, "actual": job.get(key)}
                    for key, value in expected.items()
                    if job.get(key) != value
                }
                if mismatches:
                    raise InitialPoseContractError(
                        "assignment support execution-input rebind mismatch: "
                        f"{model_key}:{mismatches}"
                    )
        reports = by_model.get(str(model_key), [])
        if not reports:
            raise InitialPoseContractError(
                f"assignment revision support coverage missing: {model_key}"
            )
        support_validation = {
            "support_contract_version": SUPPORT_CONTRACT_VERSION,
            "support_config_hash": plan["config_hash"],
            "dynamic_asset_mass_inertia_policy_version": (
                DYNAMIC_ASSET_MASS_INERTIA_POLICY_VERSION
            ),
            "status": "validated",
            "plan_hash": plan["plan_hash"],
            "report_hashes": sorted(
                str(report["report_hash"]) for report in reports
            ),
            "attempt_count": len(reports),
            "evidence_mode": (
                "exact_execution_input_validated_state_rebind"
                if validated_pose_set is not None
                else "direct_plan_execution"
            ),
            "evidence_scope": ASSIGNMENT_SELECTION_SUPPORT_SCOPE,
            "source_assignment_state_selection_hash": selection[
                "selection_hash"
            ],
            "assignment_revision_hash": revision["revision_hash"],
            "effective_canonical_state_hash": raw_entry[
                "effective_canonical_state_hash"
            ],
        }
        if validated_pose_set is not None:
            support_validation.update(
                {
                    "validated_pose_set_hash": validated_pose_set[
                        "validated_pose_set_hash"
                    ],
                    "source_assignment_revision_hash": (
                        source_assignment_revision_hash
                    ),
                    "source_effective_canonical_state_hash": (
                        source_effective_state_hash
                    ),
                    "rebind_scope": (
                        "dynamic_mass_contract_added_by_r01_support_freeze"
                    ),
                }
            )
        support_hash = artifact_hash(support_validation)
        revision["support_validation"] = support_validation
        revision["support_validation_hash"] = support_hash
        raw_entry["support_validation_hash"] = support_hash
    if set(by_model) != {
        str(key)
        for key, value in (result.get("selections") or {}).items()
        if isinstance(value, Mapping)
        and value.get("joint_coordinate_revision") is not None
    }:
        raise InitialPoseContractError(
            "assignment selection support report/model coverage mismatch"
        )
    result["selection_hash"] = artifact_hash(
        result,
        omit_keys=("selection_hash",),
    )
    # The approved set intentionally has no support evidence for ordinary
    # (non-revised) states.  Validate selector semantics against it, then
    # validate every refreshed revision's support identity directly.  The
    # eventual downstream join validates ordinary states against the separate
    # current support-validated R01 set.
    validate_assignment_state_selection(
        result,
        validated_pose_set if validated_pose_set is not None else approved,
        require_support_validated=validated_pose_set is not None,
    )
    for raw_entry in (result.get("selections") or {}).values():
        if not isinstance(raw_entry, Mapping):
            continue
        revision = raw_entry.get("joint_coordinate_revision")
        if not isinstance(revision, Mapping):
            continue
        support_validation = revision.get("support_validation") or {}
        _validate_support_validation_identity(support_validation)
        if support_validation.get("status") != "validated":
            raise InitialPoseContractError(
                "refreshed assignment revision support is not validated"
            )
        if revision.get("support_validation_hash") != artifact_hash(
            support_validation
        ):
            raise InitialPoseContractError(
                "refreshed assignment revision support hash mismatch"
            )
    atomic_yaml(output_path, result)
    return result


SUPPORT_EXECUTION_INPUT_FIELDS = (
    "job_id",
    "initial_state_id",
    "revision",
    "approval_hash",
    "geometry_hash",
    "dynamic_mass_properties",
    "asset",
    "model_id",
    "initial_state_profile_id",
    "q0_wxyz",
    "articulated_joint_state",
    "scale_xyz",
    "support",
    "yaw_policy",
    "yaw_deg",
    "seed",
    "required_metrics",
    "support_contract_version",
    "support_config_hash",
    "support_mode",
)


def support_execution_input(job: Mapping[str, Any]) -> dict[str, Any]:
    """Return the complete plan-recorded input consumed by a support job."""

    missing = [key for key in SUPPORT_EXECUTION_INPUT_FIELDS if key not in job]
    if missing:
        raise InitialPoseContractError(
            f"support job lacks execution input fields: {missing}"
        )
    return {key: job[key] for key in SUPPORT_EXECUTION_INPUT_FIELDS}


def _validate_rebind_equivalence(
    report: Mapping[str, Any], job: Mapping[str, Any]
) -> None:
    """Prove that an immutable old-plan report executed the current job input.

    The original report remains bound to its original plan.  Rebinding is
    allowed only because the report identity/config fields and the simulator
    preflight together cover every mutable actor setup input in the target
    support job.
    """

    validate_support_report(report, job)
    if report.get("schema_version") != 2:
        raise InitialPoseContractError("rebind requires support report schema v2")
    source_plan_hash = report.get("source_plan_hash")
    if not isinstance(source_plan_hash, str) or len(source_plan_hash) != 64:
        raise InitialPoseContractError("support report lacks source plan lineage")
    identity_mismatches = {}
    for key in ("asset", "initial_state_profile_id"):
        if report.get(key) != job.get(key):
            identity_mismatches[key] = {
                "expected": job.get(key),
                "actual": report.get(key),
            }
    if str(report.get("model_id")) != str(job.get("model_id")):
        identity_mismatches["model_id"] = {
            "expected": str(job.get("model_id")),
            "actual": str(report.get("model_id")),
        }
    try:
        report_yaw = float(report["yaw_deg"])
        report_seed = int(report["seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise InitialPoseContractError(
            "support report lacks exact yaw/seed execution identity"
        ) from exc
    if report_yaw != float(job["yaw_deg"]):
        identity_mismatches["yaw_deg"] = {
            "expected": float(job["yaw_deg"]),
            "actual": report_yaw,
        }
    if report_seed != int(job["seed"]):
        identity_mismatches["seed"] = {
            "expected": int(job["seed"]),
            "actual": report_seed,
        }
    if identity_mismatches:
        raise InitialPoseContractError(
            f"support report execution identity mismatch: {identity_mismatches}"
        )
    if report.get("outcome") != "pass" or report.get("error") is not None:
        raise InitialPoseContractError("only clean passing reports may be rebound")

    preflight = report.get("preflight")
    if not isinstance(preflight, Mapping) or preflight.get("status") != "matched":
        raise InitialPoseContractError("rebind requires a matched simulator preflight")
    if preflight.get("geometry_hash") != job.get("geometry_hash"):
        raise InitialPoseContractError("support preflight geometry mismatch")
    try:
        approved_q0 = normalize_quaternion(preflight["approved_q0_wxyz"])
        expected_q0 = normalize_quaternion(job["q0_wxyz"])
    except (KeyError, TypeError, ValueError) as exc:
        raise InitialPoseContractError(
            "support preflight lacks approved q0 identity"
        ) from exc
    if any(abs(left - right) > 1e-9 for left, right in zip(approved_q0, expected_q0)):
        raise InitialPoseContractError("support preflight approved q0 mismatch")

    # Import locally because runtime itself imports the core contract helpers.
    from .runtime import validate_loaded_initial_pose

    try:
        validate_loaded_initial_pose(
            canonical_state=job,
            actual_q_wxyz=preflight["actual_q_wxyz"],
            actual_joint_qpos=preflight.get("actual_joint_qpos"),
            actual_scale_xyz=preflight["actual_scale_xyz"],
            actual_origin_z=float(preflight["actual_origin_z"]),
            expected_world_yaw_deg=float(job["yaw_deg"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise InitialPoseContractError(
            "support preflight lacks complete loaded actor identity"
        ) from exc


def _support_rebind_entry(
    report: Mapping[str, Any], job: Mapping[str, Any]
) -> dict[str, Any]:
    _validate_rebind_equivalence(report, job)
    entry: dict[str, Any] = {
        "job_id": job["job_id"],
        "initial_state_id": job["initial_state_id"],
        "source_plan_hash": report["source_plan_hash"],
        "target_plan_hash": None,
        "report_hash": report["report_hash"],
        "execution_input_hash": artifact_hash(support_execution_input(job)),
    }
    return entry


def rebind_support_evidence(
    *,
    approved_set_path: Path,
    support_plan_path: Path,
    reports_dir: Path,
    output_path: Path,
    expected_source_plan_hashes: set[str] | None = None,
) -> dict[str, Any]:
    """Bind immutable old-plan reports to an execution-identical target plan."""

    approved = _load_mapping(approved_set_path)
    validate_initial_pose_set(approved, require_support_validated=False)
    plan = _load_mapping(support_plan_path)
    _validate_current_support_plan(plan)
    if plan.get("approved_pose_set_hash") != approved.get("approved_pose_set_hash"):
        raise InitialPoseContractError("support plan is stale for approved pose set")
    jobs = list(plan.get("jobs") or [])
    job_ids = [str(row.get("job_id")) for row in jobs]
    if len(job_ids) != len(set(job_ids)):
        raise InitialPoseContractError("support plan contains duplicate job ids")

    entries = []
    blockers = []
    for job in jobs:
        path = reports_dir / f"{job['job_id']}.json"
        if not path.is_file():
            blockers.append({"job_id": job["job_id"], "reason": "missing_report"})
            continue
        report = _load_mapping(path)
        try:
            entry = _support_rebind_entry(report, job)
            source_hash = str(entry["source_plan_hash"])
            if (
                expected_source_plan_hashes is not None
                and source_hash not in expected_source_plan_hashes
            ):
                raise InitialPoseContractError(
                    f"unexpected source plan lineage: {source_hash}"
                )
            entry["target_plan_hash"] = plan["plan_hash"]
            entry["binding_hash"] = artifact_hash(entry)
            entries.append(entry)
        except InitialPoseContractError as exc:
            blockers.append({"job_id": job["job_id"], "reason": str(exc)})
    if blockers:
        preview = ", ".join(
            f"{row['job_id']}:{row['reason']}" for row in blockers[:8]
        )
        raise InitialPoseContractError(
            f"support evidence rebind blocked by {len(blockers)} job(s): {preview}"
        )

    entries.sort(key=lambda row: row["job_id"])
    source_hashes = sorted({str(row["source_plan_hash"]) for row in entries})
    state_counts = Counter(str(row["initial_state_id"]) for row in entries)
    result: dict[str, Any] = {
        "artifact_kind": "initial_pose_support_evidence_rebind",
        "schema_version": 1,
        "approved_pose_set_hash": approved["approved_pose_set_hash"],
        "target_support_plan_hash": plan["plan_hash"],
        "target_support_config_hash": plan["config_hash"],
        "source_support_plan_hashes": source_hashes,
        "equivalence_contract": {
            "execution_input_fields": list(SUPPORT_EXECUTION_INPUT_FIELDS),
            "report_identity_fields": [
                "job_id",
                "initial_state_id",
                "revision",
                "approval_hash",
                "geometry_hash",
                "support_contract_version",
                "support_config_hash",
                "support_mode",
                "asset",
                "model_id",
                "initial_state_profile_id",
                "yaw_deg",
                "seed",
            ],
            "simulator_preflight_fields": [
                "approved_q0_wxyz",
                "actual_q_wxyz",
                "actual_joint_qpos",
                "actual_scale_xyz",
                "actual_origin_z",
                "geometry_hash",
            ],
            "original_reports_immutable": True,
        },
        "summary": {
            "target_job_count": len(jobs),
            "rebound_job_count": len(entries),
            "state_count": len(state_counts),
            "source_plan_count": len(source_hashes),
            "all_reports_passed": True,
        },
        "state_attempt_counts": dict(sorted(state_counts.items())),
        "bindings": entries,
    }
    result["rebind_hash"] = artifact_hash(result)
    atomic_json(output_path, result)
    return result


def validate_support_rebind(
    *,
    rebind: Mapping[str, Any],
    approved: Mapping[str, Any],
    plan: Mapping[str, Any],
    reports_dir: Path,
) -> dict[str, dict[str, Any]]:
    """Revalidate a rebind manifest against the untouched source reports."""

    _validate_current_support_plan(plan)
    if rebind.get("artifact_kind") != "initial_pose_support_evidence_rebind":
        raise InitialPoseContractError("unexpected support rebind artifact kind")
    if rebind.get("rebind_hash") != artifact_hash(
        rebind, omit_keys=("rebind_hash",)
    ):
        raise InitialPoseContractError("support rebind self-hash mismatch")
    expected_top = {
        "approved_pose_set_hash": approved.get("approved_pose_set_hash"),
        "target_support_plan_hash": plan.get("plan_hash"),
        "target_support_config_hash": plan.get("config_hash"),
    }
    mismatches = {
        key: {"expected": value, "actual": rebind.get(key)}
        for key, value in expected_top.items()
        if rebind.get(key) != value
    }
    if mismatches:
        raise InitialPoseContractError(f"support rebind target mismatch: {mismatches}")
    bindings = list(rebind.get("bindings") or [])
    by_job = {str(row.get("job_id")): row for row in bindings}
    if len(bindings) != len(by_job):
        raise InitialPoseContractError("support rebind contains duplicate job ids")
    expected_jobs = {str(row["job_id"]): row for row in plan.get("jobs") or []}
    if set(by_job) != set(expected_jobs):
        raise InitialPoseContractError("support rebind job coverage mismatch")

    verified: dict[str, dict[str, Any]] = {}
    for job_id, job in expected_jobs.items():
        path = reports_dir / f"{job_id}.json"
        if not path.is_file():
            raise InitialPoseContractError(f"rebound support report missing: {job_id}")
        report = _load_mapping(path)
        expected_entry = _support_rebind_entry(report, job)
        expected_entry["target_plan_hash"] = plan["plan_hash"]
        expected_entry["binding_hash"] = artifact_hash(expected_entry)
        if by_job[job_id] != expected_entry:
            raise InitialPoseContractError(
                f"support rebind binding mismatch: {job_id}"
            )
        verified[job_id] = expected_entry
    source_hashes = sorted(
        {str(row["source_plan_hash"]) for row in verified.values()}
    )
    if rebind.get("source_support_plan_hashes") != source_hashes:
        raise InitialPoseContractError("support rebind source lineage mismatch")
    return verified


def freeze_support_validated_set(
    *,
    approved_set_path: Path,
    support_plan_path: Path,
    reports_dir: Path,
    output_path: Path,
    rebind_report_path: Path | None = None,
) -> dict[str, Any]:
    approved = _load_mapping(approved_set_path)
    validate_initial_pose_set(approved, require_support_validated=False)
    plan = _load_mapping(support_plan_path)
    _validate_current_support_plan(plan)
    if plan.get("approved_pose_set_hash") != approved.get("approved_pose_set_hash"):
        raise InitialPoseContractError("support plan is stale for approved pose set")
    rebound_entries: dict[str, dict[str, Any]] | None = None
    rebind: dict[str, Any] | None = None
    if rebind_report_path is not None:
        rebind = _load_mapping(rebind_report_path)
        rebound_entries = validate_support_rebind(
            rebind=rebind,
            approved=approved,
            plan=plan,
            reports_dir=reports_dir,
        )
    by_state: dict[str, list[dict[str, Any]]] = {}
    blockers = []
    for job in plan.get("jobs") or []:
        if rebound_entries is not None:
            entry = rebound_entries[str(job["job_id"])]
            by_state.setdefault(str(job["initial_state_id"]), []).append(entry)
            continue
        path = reports_dir / f"{job['job_id']}.json"
        if not path.is_file():
            blockers.append({"job_id": job["job_id"], "reason": "missing_report"})
            continue
        report = _load_mapping(path)
        try:
            validate_support_report(
                report,
                job,
                expected_plan_hash=str(plan["plan_hash"]),
            )
        except InitialPoseContractError as exc:
            blockers.append({"job_id": job["job_id"], "reason": str(exc)})
            continue
        if report.get("outcome") != "pass":
            blockers.append({"job_id": job["job_id"], "reason": f"support_{report.get('outcome')}"})
            continue
        by_state.setdefault(str(job["initial_state_id"]), []).append(report)
    if blockers:
        preview = ", ".join(f"{row['job_id']}:{row['reason']}" for row in blockers[:8])
        raise InitialPoseContractError(f"support validation blocked by {len(blockers)} job(s): {preview}")
    support_identity = {
        "support_contract_version": SUPPORT_CONTRACT_VERSION,
        "support_config_hash": plan["config_hash"],
        "dynamic_asset_mass_inertia_policy_version": (
            DYNAMIC_ASSET_MASS_INERTIA_POLICY_VERSION
        ),
    }
    raw_mass_by_state = plan.get("dynamic_mass_properties_by_state")
    if not isinstance(raw_mass_by_state, Mapping):
        raise InitialPoseContractError(
            "support plan lacks per-state dynamic mass contracts"
        )
    approved_state_ids = {
        str(state["initial_state_id"])
        for state in approved.get("canonical_initial_states") or []
    }
    if set(map(str, raw_mass_by_state)) != approved_state_ids:
        raise InitialPoseContractError(
            "support plan dynamic mass coverage mismatch"
        )
    mass_by_state: dict[str, dict[str, Any]] = {}
    for state_id, raw_contract in raw_mass_by_state.items():
        if not isinstance(raw_contract, Mapping):
            raise InitialPoseContractError(
                f"support plan has invalid dynamic mass contract: {state_id}"
            )
        contract = dict(raw_contract)
        validate_dynamic_mass_properties(contract)
        mass_by_state[str(state_id)] = contract
    for job in plan.get("jobs") or []:
        state_id = str(job["initial_state_id"])
        contract = dict(job.get("dynamic_mass_properties") or {})
        validate_dynamic_mass_properties(contract)
        if mass_by_state.get(state_id) != contract:
            raise InitialPoseContractError(
                f"support plan has inconsistent dynamic mass contract: {state_id}"
            )
    states = []
    approvals = []
    for state in approved.get("canonical_initial_states") or []:
        record = dict(state)
        record["dynamic_mass_properties"] = mass_by_state[
            str(state["initial_state_id"])
        ]
        canonical_payload = dict(record)
        canonical_payload.pop("canonical_state_hash", None)
        canonical_payload.pop("support_validation", None)
        canonical_payload.pop("support_validation_hash", None)
        record["canonical_state_hash"] = artifact_hash(canonical_payload)
        if state.get("status") == "approved":
            reports = by_state.get(str(state["initial_state_id"]), [])
            record["support_validation"] = {
                **support_identity,
                "status": "validated",
                "plan_hash": plan["plan_hash"],
                "report_hashes": sorted(str(row["report_hash"]) for row in reports),
                "attempt_count": len(reports),
            }
            if rebind is not None:
                record["support_validation"].update(
                    {
                        "evidence_mode": "exact_execution_input_rebind",
                        "rebind_hash": rebind["rebind_hash"],
                        "source_plan_hashes": sorted(
                            {
                                str(row["source_plan_hash"])
                                for row in reports
                            }
                        ),
                    }
                )
            else:
                record["support_validation"]["evidence_mode"] = "direct_plan_execution"
        else:
            record["support_validation"] = {
                **support_identity,
                "status": "not_applicable_human_unsupported",
                "plan_hash": plan["plan_hash"],
                "report_hashes": [],
                "attempt_count": 0,
            }
        record["support_validation_hash"] = artifact_hash(record["support_validation"])
        states.append(record)
        matching_approval = next(
            row
            for row in approved.get("approvals") or []
            if row.get("initial_state_id") == state.get("initial_state_id")
        )
        approvals.append(dict(matching_approval))
    result: dict[str, Any] = {
        "artifact_kind": "support_validated_initial_pose_set",
        "schema_version": SCHEMA_VERSION,
        "pose_contract_version": POSE_CONTRACT_VERSION,
        "approved_pose_set_hash": approved["approved_pose_set_hash"],
        "support_plan_hash": plan["plan_hash"],
        **support_identity,
        "summary": {
            "state_count": len(states),
            "validated_state_count": sum(row.get("status") == "approved" for row in states),
            "unsupported_state_count": sum(row.get("status") == "unsupported" for row in states),
            "support_attempt_count": len(plan.get("jobs") or []),
        },
        "canonical_initial_states": sorted(states, key=lambda row: row["initial_state_id"]),
        "approvals": sorted(approvals, key=lambda row: row["initial_state_id"]),
    }
    if rebind is not None:
        result["support_evidence_rebind"] = {
            "rebind_hash": rebind["rebind_hash"],
            "source_plan_hashes": rebind["source_support_plan_hashes"],
            "target_plan_hash": plan["plan_hash"],
        }
    result["validated_pose_set_hash"] = artifact_hash(result)
    validate_initial_pose_set(result, require_support_validated=True)
    atomic_yaml(output_path, result)
    return result
