"""Milestone 11 Unit Builder and frozen artifact lifecycle.

The module is intentionally simulator-independent.  It consumes the frozen
Milestone 10 UnitInput handoff, canonical geometry metadata, and pair evidence,
then emits deterministic local layout primitives.  Global table placement is a
Milestone 12 responsibility.
"""

from __future__ import annotations

from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence
import xml.etree.ElementTree as ET

import numpy as np
import yaml

from if_benchmark.asset_probe_pose import (
    canonical_pose_policy_hash,
    resolve_canonical_pose,
)
from if_benchmark.asset_probe_evidence import (
    m4_attempt_is_blocking,
    m4_row_attempt_accounting_errors,
)
from if_benchmark.execution_policy import (
    m4_worker_execution_policy_manifest_errors,
)
from if_benchmark.asset_probe_runtime import contact_groups_from_candidate
from if_benchmark.asset_validation_contract import (
    ASSET_VALIDATION_REPORT_SCHEMA_VERSION,
    LEGACY_CONTACT_REPORT_SCHEMA_VERSION,
    M4_EXECUTABLE_USES,
)
from if_benchmark.grasp_validation.contact_contract import (
    build_contact_contract,
    resolved_contact_preflight_policy_version,
)
from if_benchmark.operation_anchors import (
    OPERATION_ANCHOR_POLICY_VERSION,
    build_operation_anchor_contract,
)
from if_benchmark.operation_placement import (
    PICK_OPERATION_PLACEMENT_POLICY_HASH,
    PICK_OPERATION_PLACEMENT_POLICY_VERSION,
)
from if_benchmark.pair_yaw_geometry import pose_aware_box_geometry
from if_benchmark.pair_recipe import (
    LEGACY_M5_REPORT_V3_PROJECTIONS,
    rebind_legacy_terminal_placement_recipe,
    validate_terminal_placement_recipe,
)
from if_benchmark.relation_contract import (
    ANCHOR_PAIR_DISTANCE_BANDS_M,
    ANCHOR_PAIR_TERMINAL_CLEARANCE_M,
    BETWEEN_MAXIMUM_PERPENDICULAR_M,
    RELATION_SURFACE_GAP_BANDS_M,
    RELATION_SURFACE_GAP_CONTRACT_VERSION,
    relation_anchor_pair_acceptance,
    relation_surface_gap_acceptance,
    relation_surface_gap_band,
)
from if_benchmark.schema import (
    artifact_hash,
    canonical_yaml,
    deterministic_id,
    validate_document,
)
from if_benchmark.schema.serialization import canonical_data
from if_benchmark.articulation_capability import (
    aggregate_articulation_capabilities,
    articulation_capability_errors,
)
from if_benchmark.touch_capability import (
    touch_capability_errors,
)


SCHEMA_VERSION = 1
UNIT_BUILD_VERSION = 16

DEFAULT_REALIZATION_DIR = Path("if_benchmark/data/realized_families")
DEFAULT_ASSET_CATALOG_PATH = Path("if_benchmark/assets/asset_catalog.yml")
DEFAULT_GEOMETRY_PATH = Path("if_benchmark/data/inventory/geometry_summary.yml")
DEFAULT_TERMINAL_PAIRS_PATH = Path(
    "if_benchmark/data/pipeline_runs/release/m05_pair_validation/frozen/validated_terminal_pairs.yml"
)
DEFAULT_SKILL_PAIRS_PATH = Path(
    "if_benchmark/data/pipeline_runs/release/m05_pair_validation/frozen/validated_skill_pairs.yml"
)
DEFAULT_RECEIVER_PROFILES_PATH = Path(
    "if_benchmark/assets/articulated_receiver_profiles.yml"
)
DEFAULT_OUTPUT_DIR = Path("if_benchmark/data/units")
DEFAULT_PROCESS_WORKERS = min(32, os.cpu_count() or 1)

REPORT_FILE = "unit_build_report.yml"
UNSAT_FILE = "unit_unsat_report.yml"

LOCAL_RELATION_GAP_M = 0.04
TERMINAL_CLEARANCE_M = 0.015
SUPPORT_CLEARANCE_M = 0.002
ARTICULATION_SWEEP_STEP_DEG = 5.0
ARTICULATION_SWEEP_PADDING_M = 0.002
MAX_ARTICULATION_CONFIGS = 4096

PAIR_RELATION = {"in": "place_in", "on": "place_on", "stack": "stack"}
PAIR_REQUIRED_RELATIONS = frozenset(PAIR_RELATION)
SUPPORTED_LOCAL_PREDICATES = frozenset(
    ("left_of", "right_of", "in_front_of", "behind", "in", "on")
)
SUPPORTED_TERMINAL_RELATIONS = frozenset(
    (
        "in",
        "on",
        "stack",
        "next_to",
        "left_of",
        "right_of",
        "in_front_of",
        "behind",
        "between",
        "left_of_pair",
        "right_of_pair",
    )
)

CONFIG: dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "unit_build_version": UNIT_BUILD_VERSION,
    "local_relation_gap_m": LOCAL_RELATION_GAP_M,
    "terminal_clearance_m": TERMINAL_CLEARANCE_M,
    "support_clearance_m": SUPPORT_CLEARANCE_M,
    "articulation_sweep_step_deg": ARTICULATION_SWEEP_STEP_DEG,
    "articulation_sweep_padding_m": ARTICULATION_SWEEP_PADDING_M,
    "max_articulation_configs": MAX_ARTICULATION_CONFIGS,
    "canonical_pose_policy_hash": canonical_pose_policy_hash(),
    "pair_projection_policy": "m11_exact_stage_evidence_recipe_payload_v2",
    "pair_required_relations": dict(sorted(PAIR_RELATION.items())),
    "footprint_shape": "unit_local_rect_aabb",
    "receiver_subject_membership": "subject_never_in_receiver_unit",
    "support_anchor_policy": "entity_origin_xy_plus_local_z_lower_bound_v1",
    "relation_surface_gap_contract_version": (
        RELATION_SURFACE_GAP_CONTRACT_VERSION
    ),
    "relation_surface_gap_bands_m": RELATION_SURFACE_GAP_BANDS_M,
    "anchor_pair_distance_bands_m": ANCHOR_PAIR_DISTANCE_BANDS_M,
    "between_maximum_perpendicular_m": BETWEEN_MAXIMUM_PERPENDICULAR_M,
    "operation_anchor_policy_version": OPERATION_ANCHOR_POLICY_VERSION,
    "operation_space_policy": "d2b_operation_space_joint_branches_v1",
}


@dataclass
class UnitBuildError(ValueError):
    """Stable M11 failure with stage attribution."""

    code: str
    message: str
    realized_family_id: str | None = None
    unit_input_id: str | None = None
    geometry_ref: str | None = None

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"

    def to_record(self) -> dict[str, Any]:
        row: dict[str, Any] = {"code": self.code, "message": self.message}
        for key in ("realized_family_id", "unit_input_id", "geometry_ref"):
            value = getattr(self, key)
            if value is not None:
                row[key] = value
        return row


def _error(
    code: str,
    message: str,
    *,
    realized_family_id: str | None = None,
    unit_input_id: str | None = None,
    geometry_ref: str | None = None,
) -> UnitBuildError:
    return UnitBuildError(
        code,
        message,
        realized_family_id=realized_family_id,
        unit_input_id=unit_input_id,
        geometry_ref=geometry_ref,
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.load(stream, Loader=yaml.CSafeLoader)
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _try_load_yaml(path: Path) -> dict[str, Any] | None:
    try:
        return _load_yaml(path)
    except (OSError, ValueError, yaml.YAMLError):
        return None


def _pose_key_from_attempt(attempt: Mapping[str, Any]) -> tuple[Any, ...] | None:
    pose = (attempt.get("diagnostics") or {}).get("root_pose") or {}
    qpos = pose.get("qpos")
    if not isinstance(qpos, list) or len(qpos) != 4:
        return None
    return (
        tuple(round(float(value), 12) for value in qpos),
        round(float(pose.get("z", 0.741)), 12),
        str(pose.get("source") or ""),
    )


def select_common_canonical_pose(
    reports: Sequence[Mapping[str, Any]],
    *,
    asset: str,
    model_id: str,
    required_uses: Sequence[str],
    minimum_success_count: int = 2,
) -> dict[str, Any] | None:
    """Select one M4-successful pose shared by every required asset use."""

    uses = tuple(sorted(set(str(value) for value in required_uses)))
    if not uses:
        return None
    accepted_records = {
        str(row.get("use"))
        for report in reports
        for row in report.get("records", [])
        if str(row.get("asset")) == str(asset)
        and str(row.get("model_id")) == str(model_id)
        and str(row.get("use")) in uses
        and row.get("pass") is True
    }
    if set(uses) - accepted_records:
        return None
    seeds_by_use_pose: dict[
        str, dict[tuple[Any, ...], set[int]]
    ] = {use: {} for use in uses}
    option_indices: dict[tuple[Any, ...], list[int]] = {}
    for report in reports:
        for attempt in report.get("attempts", []):
            use = str(attempt.get("use") or "")
            if (
                use not in seeds_by_use_pose
                or str(attempt.get("asset")) != str(asset)
                or str(attempt.get("model_id")) != str(model_id)
                or attempt.get("success") is not True
            ):
                continue
            key = _pose_key_from_attempt(attempt)
            if key is None:
                continue
            seeds_by_use_pose[use].setdefault(key, set()).add(
                int(attempt.get("seed", -1))
            )
            root_pose = (attempt.get("diagnostics") or {}).get(
                "root_pose"
            ) or {}
            attempts = (attempt.get("diagnostics") or {}).get(
                "root_pose_attempts"
            ) or []
            selected_index = next(
                (
                    index
                    for index, candidate in enumerate(attempts)
                    if candidate.get("qpos") == root_pose.get("qpos")
                    and str(candidate.get("source") or "")
                    == str(root_pose.get("source") or "")
                ),
                len(attempts),
            )
            option_indices.setdefault(key, []).append(selected_index)
    common = set.intersection(
        *(
            {
                key
                for key, seeds in seeds_by_use_pose[use].items()
                if len(seeds) >= int(minimum_success_count)
            }
            for use in uses
        )
    )
    if not common:
        return None

    def rank(key: tuple[Any, ...]) -> tuple[Any, ...]:
        counts = [len(seeds_by_use_pose[use][key]) for use in uses]
        return (
            -min(counts),
            -sum(counts),
            min(option_indices.get(key) or [10**9]),
            key,
        )

    selected = min(common, key=rank)
    qpos, z, source = selected
    successful_seeds = {
        use: sorted(seeds_by_use_pose[use][selected]) for use in uses
    }
    return {
        "q0_wxyz": list(qpos),
        "z": float(z),
        "source": str(source),
        "required_uses": list(uses),
        "minimum_success_count": int(minimum_success_count),
        "successful_seeds": successful_seeds,
        "minimum_observed_success_count": min(
            len(value) for value in successful_seeds.values()
        ),
        "total_observed_success_count": sum(
            len(value) for value in successful_seeds.values()
        ),
        "candidate_option_index": min(
            option_indices.get(selected) or [10**9]
        ),
    }


def _write_yaml(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(canonical_yaml(value))
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _clean(value: float) -> float:
    result = round(float(value), 12)
    return 0.0 if result == -0.0 else result


def _vector(values: Iterable[float]) -> list[float]:
    return [_clean(value) for value in values]


def _rect(
    center_xy: Sequence[float],
    half_extents_xy: Sequence[float],
    *,
    source: str,
) -> dict[str, Any]:
    half = [float(value) for value in half_extents_xy]
    if len(half) != 2 or min(half) <= 0:
        raise _error("invalid_footprint", f"invalid half extents: {half}")
    return {
        "shape": "rect",
        "frame": "unit_local",
        "center_xy": _vector(center_xy),
        "half_extents_xy": _vector(half),
        "source": source,
    }


def _translated(rect: Mapping[str, Any], offset_xy: Sequence[float]) -> dict[str, Any]:
    center = [
        float(rect["center_xy"][index]) + float(offset_xy[index])
        for index in range(2)
    ]
    return {
        **deepcopy(dict(rect)),
        "center_xy": _vector(center),
    }


def _rotated(rect: Mapping[str, Any], yaw_deg: float) -> dict[str, Any]:
    """Return the conservative AABB of a local rectangle after yaw rotation."""

    radians = math.radians(float(yaw_deg))
    signed_cosine = math.cos(radians)
    signed_sine = math.sin(radians)
    cosine = abs(signed_cosine)
    sine = abs(signed_sine)
    center_x, center_y = [float(value) for value in rect["center_xy"]]
    half_x, half_y = [float(value) for value in rect["half_extents_xy"]]
    return {
        **deepcopy(dict(rect)),
        "center_xy": _vector(
            [
                signed_cosine * center_x - signed_sine * center_y,
                signed_sine * center_x + signed_cosine * center_y,
            ]
        ),
        "half_extents_xy": _vector(
            [
                cosine * half_x + sine * half_y,
                sine * half_x + cosine * half_y,
            ]
        ),
        "source": f"{rect['source']}_yaw_aabb",
    }


def _envelope(
    rectangles: Sequence[Mapping[str, Any]], *, source: str
) -> dict[str, Any]:
    if not rectangles:
        raise _error("invalid_envelope", "cannot envelope an empty rectangle list")
    lower = [math.inf, math.inf]
    upper = [-math.inf, -math.inf]
    for rect in rectangles:
        center = [float(value) for value in rect["center_xy"]]
        half = [float(value) for value in rect["half_extents_xy"]]
        for axis in range(2):
            lower[axis] = min(lower[axis], center[axis] - half[axis])
            upper[axis] = max(upper[axis], center[axis] + half[axis])
    return _rect(
        [(lower[axis] + upper[axis]) / 2.0 for axis in range(2)],
        [(upper[axis] - lower[axis]) / 2.0 for axis in range(2)],
        source=source,
    )


def _rect_aabb(rect: Mapping[str, Any]) -> dict[str, list[float]]:
    """Represent one unit-local footprint as a finite 3D AABB."""

    center = [float(value) for value in rect["center_xy"]]
    half = [float(value) for value in rect["half_extents_xy"]]
    return {
        "min": [center[0] - half[0], center[1] - half[1], 0.0],
        "max": [center[0] + half[0], center[1] + half[1], 1.0],
        "center": [center[0], center[1], 0.5],
    }


def _xyz_rpy_transform(element: ET.Element | None) -> np.ndarray:
    matrix = np.eye(4, dtype=float)
    if element is None:
        return matrix
    xyz = [float(value) for value in element.get("xyz", "0 0 0").split()]
    rpy = [float(value) for value in element.get("rpy", "0 0 0").split()]
    roll, pitch, yaw = rpy
    cx, sx = math.cos(roll), math.sin(roll)
    cy, sy = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(yaw), math.sin(yaw)
    rotation = np.asarray(
        [
            [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
            [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
            [-sy, cy * sx, cy * cx],
        ],
        dtype=float,
    )
    matrix[:3, :3] = rotation
    matrix[:3, 3] = xyz
    return matrix


def _axis_motion(joint_type: str, axis: np.ndarray, value: float) -> np.ndarray:
    matrix = np.eye(4, dtype=float)
    norm = float(np.linalg.norm(axis))
    axis = axis / max(norm, 1e-12)
    if joint_type == "prismatic":
        matrix[:3, 3] = axis * float(value)
        return matrix
    if joint_type not in {"revolute", "continuous"}:
        return matrix
    x, y, z = axis
    cosine = math.cos(value)
    sine = math.sin(value)
    cross = 1.0 - cosine
    matrix[:3, :3] = np.asarray(
        [
            [cosine + x * x * cross, x * y * cross - z * sine, x * z * cross + y * sine],
            [y * x * cross + z * sine, cosine + y * y * cross, y * z * cross - x * sine],
            [z * x * cross - y * sine, z * y * cross + x * sine, cosine + z * z * cross],
        ],
        dtype=float,
    )
    return matrix


def _quaternion_matrix(q_wxyz: Sequence[float]) -> np.ndarray:
    w, x, y, z = [float(value) for value in q_wxyz]
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = [value / max(norm, 1e-12) for value in (w, x, y, z)]
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _box_corners(lower: Sequence[float], upper: Sequence[float]) -> np.ndarray:
    return np.asarray(
        [
            [x, y, z, 1.0]
            for x in (float(lower[0]), float(upper[0]))
            for y in (float(lower[1]), float(upper[1]))
            for z in (float(lower[2]), float(upper[2]))
        ],
        dtype=float,
    )


class FootprintProvider:
    """Resolve and cache deterministic rigid or articulated footprints."""

    def __init__(
        self,
        geometry_summary: Mapping[str, Any],
        *,
        asset_catalog: Mapping[str, Any] | None = None,
        terminal_pairs: Mapping[str, Any] | None = None,
        skill_pairs: Mapping[str, Any] | None = None,
        receiver_profiles: Mapping[str, Any] | None = None,
        terminal_pair_projection_sha256: str | None = None,
        skill_pair_projection_sha256: str | None = None,
        workspace_root: Path = Path("."),
    ) -> None:
        self.geometry_summary = dict(geometry_summary)
        self.models = dict(self.geometry_summary.get("models") or {})
        self.asset_catalog = dict(asset_catalog or {})
        self.terminal_pairs = {
            str(row["pair_key"]): row
            for row in (terminal_pairs or {}).get("records", [])
        }
        self.terminal_pair_schema_version = int(
            (terminal_pairs or {}).get("schema_version") or 0
        )
        self.terminal_pair_stage = str(
            (terminal_pairs or {}).get("stage") or ""
        )
        self.terminal_pair_source_ledger_hash = str(
            (terminal_pairs or {}).get("source_ledger_hash") or ""
        )
        self.skill_pairs = {
            str(row["pair_key"]): row
            for row in (skill_pairs or {}).get("records", [])
        }
        self.skill_pair_schema_version = int(
            (skill_pairs or {}).get("schema_version") or 0
        )
        self.skill_pair_stage = str(
            (skill_pairs or {}).get("stage") or ""
        )
        self.skill_pair_source_ledger_hash = str(
            (skill_pairs or {}).get("source_ledger_hash") or ""
        )
        self.receiver_profiles = dict(
            (receiver_profiles or {}).get("profiles") or {}
        )
        self.terminal_pair_projection_sha256 = (
            terminal_pair_projection_sha256
        )
        self.skill_pair_projection_sha256 = skill_pair_projection_sha256
        self.workspace_root = workspace_root
        self._cache: dict[tuple[str, str], dict[str, Any]] = {}
        self._report_cache: dict[Path, dict[str, Any]] = {}
        self._report_rows_by_key: dict[
            tuple[str, str, str, str, str], list[dict[str, Any]]
        ] = {}
        self._report_attempts_by_key: dict[
            tuple[str, str, str, str, str], list[dict[str, Any]]
        ] = {}
        self._probe_plan_cache: dict[Path, dict[str, Any]] = {}
        self._mesh_bounds: dict[Path, tuple[np.ndarray, np.ndarray]] = {}
        self._operation_anchor_cache: dict[tuple[str, ...], dict[str, Any]] = {}
        self.cache_hits = 0
        self.cache_misses = 0

    @staticmethod
    def _m4_report_matches_evidence_ref(
        report: Mapping[str, Any], row: Mapping[str, Any]
    ) -> bool:
        """Require the live outward M4 identity named by the EvidenceRef."""

        return (
            str(row.get("record_id") or ""),
            str(row.get("hash") or ""),
        ) == (
            str(report.get("evidence_id") or ""),
            str(report.get("report_hash") or ""),
        )

    def prewarm_m4_release_report(self) -> None:
        """Load the sole M4 release once before forking M11 workers."""

        release_reports = (
            self.workspace_root
            / "if_benchmark/data/pipeline_runs/release/"
            "m04_asset_validation/reports"
        ).resolve()
        if not release_reports.is_dir():
            return
        reports = sorted(release_reports.glob("canonical_skill_*.json"))
        if len(reports) == 1:
            self._load_m4_report(str(reports[0]))
            # The validated source report contains multi-gigabyte physical
            # trajectories that the compact M11 projection deliberately
            # drops.  Return those released pages before forking workers;
            # otherwise glibc keeps the address space resident and every
            # child pays avoidable copy-on-write/memory-bandwidth pressure.
            import gc

            gc.collect()
            try:
                import ctypes

                ctypes.CDLL(None).malloc_trim(0)
            except (AttributeError, OSError):  # pragma: no cover - non-glibc
                pass

    @staticmethod
    def _m4_report_lookup_key(
        report: Mapping[str, Any], *, asset: str, model_id: str, use: str
    ) -> tuple[str, str, str, str, str]:
        return (
            str(report.get("evidence_id") or ""),
            str(report.get("report_hash") or ""),
            asset,
            model_id,
            use,
        )

    def _index_m4_report(self, report: Mapping[str, Any]) -> None:
        """Build compact per-operation indexes before M11 workers fork."""

        for field, destination in (
            ("records", self._report_rows_by_key),
            ("attempts", self._report_attempts_by_key),
        ):
            for value in report.get(field) or []:
                if not isinstance(value, dict):
                    continue
                key = self._m4_report_lookup_key(
                    report,
                    asset=str(value.get("asset") or ""),
                    model_id=str(value.get("model_id") or ""),
                    use=str(value.get("use") or ""),
                )
                destination.setdefault(key, []).append(value)

    @staticmethod
    def _compact_articulation_action(
        action: Mapping[str, Any], attempt: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Project a validated articulation trace to its capability witness."""

        compact = {
            key: action.get(key)
            for key in (
                "execution_profile_version",
                "execution_profile_hash",
                "execution_profile_source_hash",
                "motion_policy_version",
                "motion_policy_hash",
                "selected_arm",
                "requested_arm",
                "successful_capability",
            )
            if action.get(key) is not None
        }
        capability = action.get("successful_capability")
        if attempt.get("success") is True and isinstance(capability, Mapping):
            candidate = capability.get("candidate") or {}
            if isinstance(candidate, Mapping):
                contact_id = candidate.get("contact_id")
                compact["articulation_result"] = {
                    "success": True,
                    "trace": [
                        {
                            "phase": "acquire",
                            "passed": True,
                            "contact_id": contact_id,
                            "moving_link": capability.get("moving_link"),
                            "engagement_mode": capability.get(
                                "engagement_mode"
                            ),
                            "attempts": [
                                {
                                    "contact_id": contact_id,
                                    "planned": True,
                                    "pre_grasp_m": candidate.get(
                                        "pre_grasp_m"
                                    ),
                                    "roll_rad": candidate.get("roll_rad"),
                                    "orientation_variant": candidate.get(
                                        "orientation_variant"
                                    ),
                                    "observation": {"passed": True},
                                }
                            ],
                        }
                    ],
                }
        return compact

    @staticmethod
    def _compact_m4_attempt(attempt: Mapping[str, Any]) -> dict[str, Any]:
        """Keep only fields used by M11 after the full report is validated."""

        diagnostics = attempt.get("diagnostics") or {}
        compact_diagnostics: dict[str, Any] = {}
        # The supported common-pose fallback consumes these fields after the
        # full report is validated. Dropping them makes a valid report appear
        # to contain no repeated successful pose.
        for field in ("root_pose", "root_pose_attempts"):
            if field in diagnostics:
                compact_diagnostics[field] = deepcopy(diagnostics[field])
        if "contact_contract_preflight" in diagnostics:
            compact_diagnostics["contact_contract_preflight"] = diagnostics[
                "contact_contract_preflight"
            ]
        action = diagnostics.get("articulation_action")
        if isinstance(action, Mapping):
            compact_diagnostics["articulation_action"] = (
                FootprintProvider._compact_articulation_action(
                    action, attempt
                )
            )
        exact_grasp_plans = []
        for plan in diagnostics.get("exact_grasp_plans") or []:
            if not isinstance(plan, Mapping):
                continue
            candidate = plan.get("best_candidate") or {}
            if not isinstance(candidate, Mapping):
                continue
            ranked_contact_ids: list[int] = []
            for ranked in plan.get("ranked_contact_candidates") or []:
                if not isinstance(ranked, Mapping):
                    continue
                contact_id = ranked.get("contact_id")
                if (
                    isinstance(contact_id, int)
                    and not isinstance(contact_id, bool)
                    and contact_id not in ranked_contact_ids
                ):
                    ranked_contact_ids.append(int(contact_id))
                if len(ranked_contact_ids) >= 2:
                    break
            if not ranked_contact_ids and isinstance(
                candidate.get("contact_id"), int
            ):
                ranked_contact_ids.append(int(candidate["contact_id"]))
            exact_grasp_plans.append(
                {
                    "arm": plan.get("arm"),
                    "allowed_contact_ids": plan.get("allowed_contact_ids"),
                    "ranked_contact_ids": ranked_contact_ids,
                    "best_candidate": {
                        "contact_id": candidate.get("contact_id"),
                        "contact_contract_hash": candidate.get(
                            "contact_contract_hash"
                        ),
                    },
                }
            )
        if exact_grasp_plans:
            compact_diagnostics["exact_grasp_plans"] = exact_grasp_plans
        witness = attempt.get("canonical_placement_witness") or {}
        return {
            key: value
            for key, value in {
                "asset": attempt.get("asset"),
                "model_id": attempt.get("model_id"),
                "use": attempt.get("use"),
                "seed": attempt.get("seed"),
                "requested_arm": attempt.get("requested_arm"),
                "failure_reason": attempt.get("failure_reason"),
                "plan_success": attempt.get("plan_success"),
                "success": attempt.get("success"),
                "execution_policy_manifest": attempt.get(
                    "execution_policy_manifest"
                ),
                "canonical_placement_witness": (
                    {
                        "witness_hash": witness.get("witness_hash"),
                        "selected_contact_ids_by_arm": deepcopy(
                            witness.get("selected_contact_ids_by_arm") or {}
                        ),
                    }
                    if witness
                    else None
                ),
                "diagnostics": compact_diagnostics,
            }.items()
            if value is not None
        }

    @classmethod
    def _compact_m4_report(
        cls, report: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Discard physical trajectories after validating their envelope."""

        compact = {
            key: value
            for key, value in report.items()
            if key not in {"attempts", "source_probe_plan_partitions"}
        }
        compact["attempts"] = [
            cls._compact_m4_attempt(attempt)
            for attempt in report.get("attempts") or []
            if isinstance(attempt, Mapping)
        ]
        return compact

    def _m4_rows_for_key(
        self,
        report: Mapping[str, Any],
        *,
        asset: str,
        model_id: str,
        use: str,
    ) -> list[dict[str, Any]]:
        return self._report_rows_by_key.get(
            self._m4_report_lookup_key(
                report, asset=asset, model_id=model_id, use=use
            ),
            [],
        )

    def operation_anchor_contract(
        self,
        binding: Mapping[str, Any],
        q0_wxyz: Sequence[float],
        *,
        scale_xyz: Sequence[float] | None = None,
        articulated_joint_state: Mapping[str, Any] | None = None,
        canonical_state_hash: str | None = None,
        assignment_state_revision: Mapping[str, Any] | None = None,
        validated_evidence: Sequence[Mapping[str, Any]] = (),
    ) -> dict[str, Any] | None:
        """Freeze reviewed model-data operation points for one entity."""

        asset = str(binding.get("asset") or "")
        model_id = str(binding.get("model_id") or "")
        catalog_model = (
            (
                (
                    (self.asset_catalog.get("assets") or {}).get(asset)
                    or {}
                ).get("models")
                or {}
            ).get(model_id)
            or {}
        )
        candidates = catalog_model.get("candidates") or {}
        validation_rows = [
            canonical_data(dict(row))
            for row in validated_evidence
            if row.get("kind") == "asset_validation"
            and str(row.get("asset")) == asset
            and str(row.get("model_id")) == model_id
            and str(row.get("status")) == "validated"
        ]
        validated_canonical_evidence: list[dict[str, Any]] = []
        for row in validation_rows:
            if str(row.get("stage") or "") != "canonical_skill":
                continue
            if str(row.get("use") or "") not in M4_EXECUTABLE_USES:
                raise _error(
                    "stale_operation_anchor_evidence",
                    f"unsupported live M4 canonical use for "
                    f"{asset}|{model_id}: {row.get('use')!r}",
                    geometry_ref=f"{asset}|{model_id}",
                )
            metadata = row.get("metadata") or {}
            if isinstance(
                metadata.get("contact_contract_binding"), Mapping
            ):
                stale_code = "stale_contact_contract_evidence"
            elif isinstance(
                metadata.get("articulation_execution_binding"), Mapping
            ):
                stale_code = "stale_articulation_execution_evidence"
            elif isinstance(metadata.get("touch_execution_binding"), Mapping):
                stale_code = "stale_touch_execution_evidence"
            else:
                stale_code = "stale_operation_anchor_evidence"
            report = self._load_m4_report(
                str(row.get("path") or ""),
                error_code=stale_code,
                geometry_ref=f"{asset}|{model_id}",
            )
            if report.get("schema_version") == (
                LEGACY_CONTACT_REPORT_SCHEMA_VERSION
            ) and (
                str(row.get("use"))
                not in {
                    "pick_left",
                    "pick_right",
                    "bimanual_grab_subject",
                }
                or not isinstance(
                    metadata.get("contact_contract_binding"), Mapping
                )
            ):
                raise _error(
                    stale_code,
                    "legacy M4 schema v2 is only valid for Contact-bound "
                    f"evidence, not {asset}|{model_id}|{row.get('use')}",
                    geometry_ref=f"{asset}|{model_id}",
                )
            if not self._m4_report_matches_evidence_ref(report, row):
                raise _error(
                    stale_code,
                    f"M4 report join changed for {asset}|{model_id}|"
                    f"{row.get('use')}",
                    geometry_ref=f"{asset}|{model_id}",
                )
            matching_rows = self._m4_rows_for_key(
                report,
                asset=asset,
                model_id=model_id,
                use=str(row.get("use")),
            )
            if len(matching_rows) != 1 or matching_rows[0].get("pass") is not True:
                raise _error(
                    stale_code,
                    f"M4 EvidenceRef has no unique passing target row for "
                    f"{asset}|{model_id}|{row.get('use')}",
                    geometry_ref=f"{asset}|{model_id}",
                )
            target_attempts = self._m4_attempts_for_row(
                report,
                asset=asset,
                model_id=model_id,
                use=str(row.get("use")),
            )
            if any(
                m4_worker_execution_policy_manifest_errors(
                    attempt.get("execution_policy_manifest"),
                    use=str(row.get("use") or ""),
                )
                for attempt in target_attempts
                if not m4_attempt_is_blocking(attempt)
            ):
                raise _error(
                    stale_code,
                    f"M4 EvidenceRef target attempt policy changed for "
                    f"{asset}|{model_id}|{row.get('use')}",
                    geometry_ref=f"{asset}|{model_id}",
                )
            validated_canonical_evidence.append(
                {
                    "use": str(row.get("use")),
                    "evidence_id": str(row.get("record_id")),
                    "report_path": str(row.get("path")),
                    "report_hash": str(row.get("hash")),
                }
            )
        explicit_contact_rows = [
            row
            for row in validation_rows
            if isinstance(
                (row.get("metadata") or {}).get(
                    "contact_contract_binding"
                ),
                Mapping,
            )
        ]
        # Orientation adjustment was historically published as acquisition
        # eligibility rather than an exact-grasp row.  Its worker attempts do
        # nevertheless bind the live Contact contract used for acquisition.
        # M11 may deterministically rebind that exact attempt-owned contract;
        # it must never infer one for an unrelated generic use.
        implicit_adjust_contact_rows = [
            row
            for row in validation_rows
            if str(row.get("use") or "") == "adjust_orientation_subject"
            and row not in explicit_contact_rows
        ]
        contact_rows = explicit_contact_rows + implicit_adjust_contact_rows
        preferred_candidate_uses = {
            str(
                ((row.get("metadata") or {}).get(
                    "contact_contract_binding"
                ) or {}).get("candidate_use")
                or (
                    "adjust_orientation_subject"
                    if row in implicit_adjust_contact_rows
                    else ""
                )
                or ""
            )
            for row in contact_rows
        }
        preferred_candidate_uses.discard("")
        if len(preferred_candidate_uses) > 1:
            raise _error(
                "ambiguous_contact_contract_evidence",
                f"multiple Contact candidate uses for {asset}|{model_id}: "
                f"{sorted(preferred_candidate_uses)}",
                geometry_ref=f"{asset}|{model_id}",
            )
        preferred_candidate_use = next(
            iter(preferred_candidate_uses), ""
        )
        candidate_rows = sorted(
            candidates.items(),
            key=lambda item: (
                0
                if str(item[0])
                == (preferred_candidate_use or "pick_subject")
                else 1,
                str(item[0]),
            ),
        )
        candidate_use, candidate = next(
            (
                (str(use), dict(row))
                for use, row in candidate_rows
                if str(
                    (
                        (row.get("evidence") or {}).get("metadata") or {}
                    ).get("path")
                    or ""
                )
            ),
            ("", {}),
        )
        evidence = (candidate.get("evidence") or {}).get("metadata") or {}
        raw_path = str(evidence.get("path") or "")
        if not raw_path:
            return None
        path = (self.workspace_root / raw_path).resolve()
        try:
            payload = path.read_bytes()
            observed_sha = hashlib.sha256(payload).hexdigest()
            model_data = json.loads(payload.decode("utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise _error(
                "missing_operation_anchor_evidence",
                f"cannot load operation-anchor metadata {raw_path}: {exc}",
                geometry_ref=f"{asset}|{model_id}",
            ) from exc
        if not isinstance(model_data, dict):
            raise _error(
                "missing_operation_anchor_evidence",
                f"operation-anchor metadata is not a mapping: {raw_path}",
                geometry_ref=f"{asset}|{model_id}",
            )
        joint_state = dict(articulated_joint_state or {})
        articulated = (
            str(joint_state.get("mode") or "not_articulated")
            != "not_articulated"
        )
        urdf_source_path = (
            Path(raw_path).with_name("mobility.urdf")
            if articulated
            else None
        )
        urdf_path = (
            (self.workspace_root / urdf_source_path).resolve()
            if urdf_source_path is not None
            else None
        )
        observed_urdf_sha = ""
        if urdf_path is not None:
            try:
                observed_urdf_sha = hashlib.sha256(
                    urdf_path.read_bytes()
                ).hexdigest()
            except OSError as exc:
                raise _error(
                    "missing_operation_anchor_evidence",
                    f"cannot load operation-anchor URDF {urdf_source_path}: {exc}",
                    geometry_ref=f"{asset}|{model_id}",
                ) from exc
        cache_key = (
            asset,
            model_id,
            observed_sha,
            observed_urdf_sha,
            artifact_hash(list(q0_wxyz)),
            artifact_hash(list(scale_xyz or [])),
            artifact_hash(joint_state),
            str(canonical_state_hash or ""),
            artifact_hash(dict(assignment_state_revision or {})),
            artifact_hash(validation_rows),
        )
        if cache_key in self._operation_anchor_cache:
            return deepcopy(self._operation_anchor_cache[cache_key])
        try:
            contract = build_operation_anchor_contract(
                model_data,
                q0_wxyz,
                scale_xyz=scale_xyz,
                articulated_joint_state=joint_state,
                urdf_path=urdf_path,
                canonical_state_hash=canonical_state_hash,
            )
        except ValueError as exc:
            raise _error(
                "invalid_operation_anchor_evidence",
                f"cannot project operation anchors for {asset}|{model_id}: {exc}",
                geometry_ref=f"{asset}|{model_id}",
            ) from exc
        validated_contact_contracts: list[dict[str, Any]] = []
        live_contact_projection_hash: str | None = None
        if contact_rows:
            matrices = model_data.get("contact_points_pose")
            if not isinstance(matrices, list) or not matrices:
                raise _error(
                    "missing_operation_anchor_evidence",
                    f"Contact matrices missing for {asset}|{model_id}",
                    geometry_ref=f"{asset}|{model_id}",
                )
            groups = contact_groups_from_candidate(
                candidate, model_data=model_data
            )
            for row in contact_rows:
                explicit_binding = (row.get("metadata") or {}).get(
                    "contact_contract_binding"
                )
                implicit_adjust = not isinstance(explicit_binding, Mapping)
                binding_ref = (
                    dict(explicit_binding)
                    if isinstance(explicit_binding, Mapping)
                    else {}
                )
                candidate_use_ref = (
                    str(row.get("use"))
                    if implicit_adjust
                    else str(binding_ref["candidate_use"])
                )
                use_profile_ref = (
                    "single_arm_pick"
                    if implicit_adjust
                    else str(binding_ref["use_profile"])
                )
                observed_contact = build_contact_contract(
                    asset=asset,
                    model_id=model_id,
                    use=candidate_use_ref,
                    contact_matrices=matrices,
                    contact_groups=groups,
                    use_profile=use_profile_ref,
                )
                if implicit_adjust:
                    binding_ref = {
                        "policy_version": observed_contact["policy_version"],
                        "contact_contract_id": observed_contact[
                            "contact_contract_id"
                        ],
                        "contact_contract_hash": observed_contact[
                            "contact_contract_hash"
                        ],
                        "contact_projection_hash": observed_contact[
                            "contact_projection_hash"
                        ],
                        "candidate_use": candidate_use_ref,
                        "use_profile": use_profile_ref,
                        "contact_count": len(matrices),
                        "contact_groups": groups,
                    }
                if live_contact_projection_hash not in (
                    None,
                    observed_contact["contact_projection_hash"],
                ):
                    raise _error(
                        "ambiguous_contact_contract_evidence",
                        f"Contact projections disagree for {asset}|{model_id}",
                        geometry_ref=f"{asset}|{model_id}",
                    )
                live_contact_projection_hash = observed_contact[
                    "contact_projection_hash"
                ]
                for field in (
                    "policy_version",
                    "contact_contract_id",
                    "contact_contract_hash",
                    "contact_projection_hash",
                    "candidate_use",
                    "use_profile",
                    "contact_count",
                    "contact_groups",
                ):
                    expected_value = (
                        len(matrices)
                        if field == "contact_count"
                        else groups
                        if field == "contact_groups"
                        else observed_contact.get("use")
                        if field == "candidate_use"
                        else observed_contact.get(field)
                    )
                    if binding_ref.get(field) != expected_value:
                        raise _error(
                            "stale_contact_contract_evidence",
                            f"{field} changed for {asset}|{model_id}",
                            geometry_ref=f"{asset}|{model_id}",
                        )
                report = self._load_m4_report(
                    str(row.get("path") or ""),
                    error_code="stale_contact_contract_evidence",
                    geometry_ref=f"{asset}|{model_id}",
                )
                if not self._m4_report_matches_evidence_ref(report, row):
                    raise _error(
                        "stale_contact_contract_evidence",
                        f"M4 report join changed for {asset}|{model_id}",
                        geometry_ref=f"{asset}|{model_id}",
                    )
                matching_rows = [
                    item
                    for item in self._m4_rows_for_key(
                        report,
                        asset=asset,
                        model_id=model_id,
                        use=str(row.get("use")),
                    )
                    if item.get("pass") is True
                ]
                if len(matching_rows) != 1:
                    raise _error(
                        "stale_contact_contract_evidence",
                        f"M4 report has no unique passing row for "
                        f"{asset}|{model_id}|{row.get('use')}",
                        geometry_ref=f"{asset}|{model_id}",
                    )
                if implicit_adjust:
                    self._verify_m4_adjust_contact_report_binding(
                        report,
                        row=matching_rows[0],
                        binding=binding_ref,
                        asset=asset,
                        model_id=model_id,
                        use=str(row.get("use")),
                    )
                else:
                    self._verify_m4_contact_report_binding(
                        report,
                        row=matching_rows[0],
                        binding=binding_ref,
                        asset=asset,
                        model_id=model_id,
                        use=str(row.get("use")),
                    )
                validated_contact_ids_by_arm = (
                    self._validated_m4_contact_ids(
                        report,
                        asset=asset,
                        model_id=model_id,
                        use=str(row.get("use")),
                        binding=binding_ref,
                    )
                )
                validated_contact_contracts.append(
                    {
                        "use": str(row.get("use")),
                        "evidence_id": str(row.get("record_id")),
                        "report_path": str(row.get("path")),
                        "report_hash": str(row.get("hash")),
                        "binding": canonical_data(binding_ref),
                        "validated_contact_ids_by_arm": canonical_data(
                            validated_contact_ids_by_arm
                        ),
                        "validated_contact_ids": sorted(
                            {
                                contact_id
                                for ids in validated_contact_ids_by_arm.values()
                                for contact_id in ids
                            }
                        ),
                    }
                )
        validated_articulation_executions: list[dict[str, Any]] = []
        for row in validation_rows:
            binding_value = (row.get("metadata") or {}).get(
                "articulation_execution_binding"
            )
            if not isinstance(binding_value, Mapping):
                continue
            report = self._load_m4_report(
                str(row.get("path") or ""),
                error_code="stale_articulation_execution_evidence",
                geometry_ref=f"{asset}|{model_id}",
            )
            if not self._m4_report_matches_evidence_ref(report, row):
                raise _error(
                    "stale_articulation_execution_evidence",
                    f"M4 articulation report join changed for "
                    f"{asset}|{model_id}",
                    geometry_ref=f"{asset}|{model_id}",
                )
            matching_rows = [
                item
                for item in self._m4_rows_for_key(
                    report,
                    asset=asset,
                    model_id=model_id,
                    use=str(row.get("use")),
                )
                if item.get("pass") is True
            ]
            if len(matching_rows) != 1:
                raise _error(
                    "stale_articulation_execution_evidence",
                    f"M4 report has no unique passing articulation row for "
                    f"{asset}|{model_id}|{row.get('use')}",
                    geometry_ref=f"{asset}|{model_id}",
                )
            self._verify_m4_articulation_report_binding(
                report,
                row=matching_rows[0],
                binding=dict(binding_value),
                asset=asset,
                model_id=model_id,
                use=str(row.get("use")),
            )
            validated_articulation_executions.append(
                {
                    "use": str(row.get("use")),
                    "evidence_id": str(row.get("record_id")),
                    "report_path": str(row.get("path")),
                    "report_hash": str(row.get("hash")),
                    "binding": canonical_data(dict(binding_value)),
                }
            )
        validated_touch_executions: list[dict[str, Any]] = []
        for row in validation_rows:
            binding_value = (row.get("metadata") or {}).get(
                "touch_execution_binding"
            )
            if not isinstance(binding_value, Mapping):
                continue
            report = self._load_m4_report(
                str(row.get("path") or ""),
                error_code="stale_touch_execution_evidence",
                geometry_ref=f"{asset}|{model_id}",
            )
            if not self._m4_report_matches_evidence_ref(report, row):
                raise _error(
                    "stale_touch_execution_evidence",
                    f"M4 touch report join changed for {asset}|{model_id}",
                    geometry_ref=f"{asset}|{model_id}",
                )
            matching_rows = [
                item
                for item in self._m4_rows_for_key(
                    report,
                    asset=asset,
                    model_id=model_id,
                    use=str(row.get("use")),
                )
                if item.get("pass") is True
            ]
            if len(matching_rows) != 1:
                raise _error(
                    "stale_touch_execution_evidence",
                    f"M4 report has no unique passing touch row for "
                    f"{asset}|{model_id}|{row.get('use')}",
                    geometry_ref=f"{asset}|{model_id}",
                )
            self._verify_m4_touch_report_binding(
                report,
                row=matching_rows[0],
                binding=dict(binding_value),
                asset=asset,
                model_id=model_id,
                use=str(row.get("use")),
            )
            validated_touch_executions.append(
                {
                    "use": str(row.get("use")),
                    "evidence_id": str(row.get("record_id")),
                    "report_path": str(row.get("path")),
                    "report_hash": str(row.get("hash")),
                    "binding": canonical_data(dict(binding_value)),
                }
            )
        source = {
            "asset": asset,
            "model_id": model_id,
            "candidate_use": candidate_use,
            "metadata_path": raw_path,
            "metadata_sha256": observed_sha,
            "operation_metadata_hash": contract["operation_metadata_hash"],
        }
        if validated_canonical_evidence:
            source["validated_canonical_evidence"] = sorted(
                validated_canonical_evidence,
                key=lambda row: (
                    row["use"],
                    row["evidence_id"],
                    row["report_path"],
                    row["report_hash"],
                ),
            )
        if validated_contact_contracts:
            source["validated_contact_contracts"] = sorted(
                validated_contact_contracts,
                key=lambda row: (row["use"], row["report_hash"]),
            )
            source["contact_projection_hash"] = live_contact_projection_hash
        if validated_articulation_executions:
            source["validated_articulation_executions"] = sorted(
                validated_articulation_executions,
                key=lambda row: (row["use"], row["report_hash"]),
            )
        if validated_touch_executions:
            source["validated_touch_executions"] = sorted(
                validated_touch_executions,
                key=lambda row: (row["use"], row["report_hash"]),
            )
        if articulated and urdf_path is not None:
            if contract.get("urdf_sha256") != observed_urdf_sha:
                raise _error(
                    "stale_operation_anchor_evidence",
                    f"operation-anchor URDF changed while building {asset}|{model_id}",
                    geometry_ref=f"{asset}|{model_id}",
                )
            source.update(
                {
                    "urdf_path": str(urdf_source_path),
                    "urdf_sha256": contract["urdf_sha256"],
                }
            )
        if canonical_state_hash:
            source["canonical_state_hash"] = str(canonical_state_hash)
        if assignment_state_revision:
            source["assignment_state_revision"] = canonical_data(
                dict(assignment_state_revision)
            )
        payload = {
            key: value
            for key, value in contract.items()
            if key != "contract_hash"
        }
        if live_contact_projection_hash:
            payload["contact_projection_hash"] = (
                live_contact_projection_hash
            )
        payload["source"] = source
        result = canonical_data(
            {**payload, "contract_hash": artifact_hash(payload)}
        )
        self._operation_anchor_cache[cache_key] = result
        return deepcopy(result)

    def pair_evidence(
        self, precondition: Mapping[str, Any], relation: str
    ) -> dict[str, Any]:
        """Resolve one M10 pair projection to its fresh M5 witness."""

        pair_key = str(precondition.get("pair_key") or "")
        expected_prefix = PAIR_RELATION[relation] + "|"
        if not pair_key.startswith(expected_prefix):
            raise _error(
                "stale_pair_evidence",
                f"pair key {pair_key!r} does not prove {relation}",
            )
        terminal = self.terminal_pairs.get(pair_key)
        if not isinstance(terminal, dict) or terminal.get("status") != "validated_terminal":
            raise _error(
                "missing_pair_evidence",
                f"no validated terminal projection for {pair_key}",
            )
        stage = str(precondition.get("stage") or "")
        source = terminal
        if stage == "canonical_skill":
            source = self.skill_pairs.get(pair_key)
            if not isinstance(source, dict) or source.get("status") != "validated_skill":
                raise _error(
                    "missing_pair_evidence",
                    f"no validated skill projection for {pair_key}",
                )
            expected_recipe = str(source.get("validated_skill_recipe_id") or "")
            expected_placement_recipe = source.get(
                "validated_skill_placement_recipe"
            )
            expected_bands = [
                str(value) for value in source.get("validated_skill_band_ids", [])
            ]
        elif stage == "terminal_settle":
            expected_recipe = str(terminal.get("validated_recipe_id") or "")
            expected_placement_recipe = terminal.get(
                "validated_placement_recipe"
            )
            expected_bands = [
                str(value) for value in terminal.get("validated_band_ids", [])
            ]
        else:
            raise _error(
                "stale_pair_evidence",
                f"unsupported pair evidence stage {stage!r} for {pair_key}",
            )
        if str(precondition.get("evidence_id") or "") != str(
            source.get("evidence_id") or ""
        ):
            raise _error(
                "stale_pair_evidence",
                f"evidence id changed for {pair_key}",
            )
        recipe_id = str(precondition.get("recipe_id") or "")
        band_ids = [str(value) for value in precondition.get("band_ids", [])]
        if recipe_id != expected_recipe or band_ids != expected_bands:
            raise _error(
                "stale_pair_evidence",
                f"recipe/band projection changed for {pair_key}",
            )
        pair_relation = PAIR_RELATION[relation]
        if isinstance(expected_placement_recipe, Mapping):
            try:
                expected_placement_recipe = (
                    validate_terminal_placement_recipe(
                        expected_placement_recipe,
                        pair_relation,
                        expected_recipe_id=recipe_id,
                    )
                )
            except ValueError as exc:
                raise _error(
                    "stale_pair_evidence",
                    f"invalid frozen M5 placement recipe for {pair_key}: {exc}",
                ) from exc
        else:
            try:
                expected_placement_recipe = (
                    rebind_legacy_terminal_placement_recipe(
                        source,
                        pair_relation,
                        stage=stage,
                        projection_schema_version=(
                            self.skill_pair_schema_version
                            if stage == "canonical_skill"
                            else self.terminal_pair_schema_version
                        ),
                        projection_stage=(
                            self.skill_pair_stage
                            if stage == "canonical_skill"
                            else self.terminal_pair_stage
                        ),
                        projection_sha256=(
                            self.skill_pair_projection_sha256
                            if stage == "canonical_skill"
                            else self.terminal_pair_projection_sha256
                        ),
                        source_ledger_hash=str(
                            (
                                source.get("source_ledger_hash")
                                or (
                                    self.skill_pair_source_ledger_hash
                                    if stage == "canonical_skill"
                                    else self.terminal_pair_source_ledger_hash
                                )
                            )
                            or ""
                        ),
                        receiver_profiles=self.receiver_profiles,
                    )
                )
            except ValueError as exc:
                raise _error(
                    "stale_pair_evidence",
                    f"M5 placement recipe cannot be rebound for {pair_key}: {exc}",
                ) from exc
        precondition_placement_recipe = precondition.get(
            "placement_recipe"
        )
        placement_recipe_source = str(
            precondition.get("placement_recipe_source") or ""
        )
        if not isinstance(precondition_placement_recipe, Mapping):
            raise _error(
                "missing_pair_evidence",
                f"M8/M10 pair projection lacks placement recipe for {pair_key}",
            )
        if not placement_recipe_source:
            raise _error(
                "missing_pair_evidence",
                f"M8/M10 pair projection lacks recipe source for {pair_key}",
            )
        try:
            precondition_placement_recipe = validate_terminal_placement_recipe(
                precondition_placement_recipe,
                pair_relation,
                expected_recipe_id=recipe_id,
            )
        except ValueError as exc:
            raise _error(
                "stale_pair_evidence",
                f"invalid M8/M10 placement recipe for {pair_key}: {exc}",
            ) from exc
        if precondition_placement_recipe != expected_placement_recipe:
            raise _error(
                "stale_pair_evidence",
                f"placement recipe projection changed for {pair_key}",
            )
        band_results = list((terminal.get("metrics") or {}).get("band_results") or [])
        selected = next(
            (
                row
                for row in band_results
                if str(row.get("recipe_id") or "") == recipe_id
                and (
                    not band_ids
                    or str(row.get("band_id") or "") in set(band_ids)
                )
            ),
            None,
        )
        if selected is None and stage == "terminal_settle":
            selected = next(
                (
                    row
                    for row in band_results
                    if str(row.get("recipe_id") or "") == recipe_id
                ),
                None,
            )
        yaw_witness = (selected or {}).get("yaw_witness") or {}
        relative_yaw = terminal.get("relative_yaw") or {}
        usable_region = relative_yaw.get("usable_region") or {}
        if (
            selected is None
            or selected.get("pass") is not True
            or "witness_delta_deg" not in yaw_witness
            or not all(key in usable_region for key in ("x", "y"))
        ):
            raise _error(
                "missing_pair_evidence",
                f"pair projection lacks validated yaw/usable-region evidence: {pair_key}",
            )
        return canonical_data(
            {
                "pair_key": pair_key,
                "relation": PAIR_RELATION[relation],
                "stage": stage,
                "evidence_id": source["evidence_id"],
                "terminal_evidence_id": terminal["evidence_id"],
                "recipe_id": recipe_id,
                "placement_recipe": precondition_placement_recipe,
                "placement_recipe_source": placement_recipe_source,
                "band_ids": band_ids,
                "yaw_witness": {
                    key: yaw_witness[key]
                    for key in (
                        "band_id",
                        "connected_component_id",
                        "covered_band_ids",
                        "geometry_source",
                        "max_margin_m",
                        "witness_delta_deg",
                    )
                    if key in yaw_witness
                },
                "relative_yaw_deg": float(yaw_witness["witness_delta_deg"]),
                "usable_region": {
                    "x": float(usable_region["x"]),
                    "y": float(usable_region["y"]),
                    "factor": usable_region.get("factor"),
                    "edge_clearance_m": relative_yaw.get("edge_clearance_m"),
                    "solver_policy_hash": relative_yaw.get("solver_policy_hash"),
                },
            }
        )

    def _load_m4_report(
        self,
        raw_path: str,
        *,
        error_code: str = "stale_canonical_pose_evidence",
        geometry_ref: str | None = None,
    ) -> dict[str, Any]:
        path = (self.workspace_root / raw_path).resolve()
        if path in self._report_cache:
            return self._report_cache[path]
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise _error(
                error_code,
                f"cannot load M4 report {raw_path}: {exc}",
                geometry_ref=geometry_ref,
            ) from exc
        if not isinstance(report, dict):
            raise _error(
                error_code,
                f"M4 report is not a mapping: {raw_path}",
                geometry_ref=geometry_ref,
            )
        try:
            # Import lazily: asset_validation -> scene_validation -> layout
            # imports this module's CONFIG during module initialization.
            from if_benchmark.asset_validation import (
                validate_normalized_report,
            )

            validate_normalized_report(report, expected_path=path)
        except ValueError as exc:
            raise _error(
                error_code,
                f"M4 report validation failed: {raw_path}: {exc}",
                geometry_ref=geometry_ref,
            ) from exc
        compact_report = self._compact_m4_report(report)
        self._report_cache[path] = compact_report
        self._index_m4_report(compact_report)
        return compact_report

    def _load_m4_probe_plan(
        self, report: Mapping[str, Any]
    ) -> dict[str, Any]:
        raw_path = str(report.get("source_probe_plan_path") or "")
        if not raw_path:
            raise _error(
                "stale_contact_contract_evidence",
                "legacy M4 report has no source probe-plan path",
            )
        path = (self.workspace_root / raw_path).resolve()
        if path in self._probe_plan_cache:
            return self._probe_plan_cache[path]
        try:
            plan = _load_yaml(path)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            raise _error(
                "stale_contact_contract_evidence",
                f"cannot load legacy M4 probe plan {raw_path}: {exc}",
            ) from exc
        if (
            not isinstance(plan, dict)
            or plan.get("probe_plan_hash")
            != artifact_hash(plan, omit_keys=("probe_plan_hash",))
            or report.get("source_probe_plan_hash")
            != plan.get("probe_plan_hash")
            or plan.get("pick_operation_placement_policy_version")
            != PICK_OPERATION_PLACEMENT_POLICY_VERSION
            or plan.get("pick_operation_placement_policy_hash")
            != PICK_OPERATION_PLACEMENT_POLICY_HASH
        ):
            raise _error(
                "stale_contact_contract_evidence",
                f"legacy M4 probe-plan join is invalid: {raw_path}",
            )
        self._probe_plan_cache[path] = plan
        return plan

    def _m4_attempts_for_row(
        self,
        report: Mapping[str, Any],
        *,
        asset: str,
        model_id: str,
        use: str,
    ) -> list[dict[str, Any]]:
        return [
            dict(attempt)
            for attempt in self._report_attempts_by_key.get(
                self._m4_report_lookup_key(
                    report, asset=asset, model_id=model_id, use=use
                ),
                [],
            )
        ]

    def _validated_m4_contact_ids(
        self,
        report: Mapping[str, Any],
        *,
        asset: str,
        model_id: str,
        use: str,
        binding: Mapping[str, Any],
    ) -> dict[str, list[int]]:
        """Return the top Contacts from successful physical M4 attempts."""

        expected_hash = str(binding.get("contact_contract_hash") or "")
        contact_count = int(binding.get("contact_count") or 0)
        by_arm: dict[str, list[int]] = {"left": [], "right": []}
        for attempt in self._m4_attempts_for_row(
            report,
            asset=asset,
            model_id=model_id,
            use=use,
        ):
            if (
                m4_attempt_is_blocking(attempt)
                or attempt.get("success") is not True
            ):
                continue
            found_plan = False
            for plan in (attempt.get("diagnostics") or {}).get(
                "exact_grasp_plans"
            ) or []:
                if not isinstance(plan, Mapping):
                    continue
                arm = str(plan.get("arm") or "")
                candidate = plan.get("best_candidate") or {}
                if arm not in by_arm or candidate.get(
                    "contact_contract_hash"
                ) != expected_hash:
                    continue
                ranked_ids = list(plan.get("ranked_contact_ids") or [])
                if not ranked_ids:
                    ranked_ids = [candidate.get("contact_id")]
                for contact_id in ranked_ids[:2]:
                    if (
                        not isinstance(contact_id, int)
                        or isinstance(contact_id, bool)
                        or not 0 <= contact_id < contact_count
                    ):
                        continue
                    if int(contact_id) not in by_arm[arm]:
                        by_arm[arm].append(int(contact_id))
                    found_plan = True
            if found_plan:
                continue
            # Historical compact reports may only expose the placement winner.
            selected = (
                attempt.get("canonical_placement_witness") or {}
            ).get("selected_contact_ids_by_arm") or {}
            if not isinstance(selected, Mapping):
                continue
            for arm, raw_contact_id in selected.items():
                if str(arm) not in by_arm:
                    continue
                try:
                    contact_id = int(raw_contact_id)
                except (TypeError, ValueError):
                    continue
                if 0 <= contact_id < contact_count:
                    if contact_id not in by_arm[str(arm)]:
                        by_arm[str(arm)].append(contact_id)
        result = {
            arm: list(ids) for arm, ids in by_arm.items() if ids
        }
        if not result:
            raise _error(
                "stale_contact_contract_evidence",
                f"M4 has no successful Contact winner for "
                f"{asset}|{model_id}|{use}",
                geometry_ref=f"{asset}|{model_id}",
            )
        return result

    def _verify_m4_contact_report_binding(
        self,
        report: Mapping[str, Any],
        *,
        row: Mapping[str, Any],
        binding: Mapping[str, Any],
        asset: str,
        model_id: str,
        use: str,
    ) -> None:
        """Prove that the joined passing M4 row ran this exact Contact ref."""

        report_version = report.get("schema_version")
        if report_version not in {2, 3}:
            raise _error(
                "stale_contact_contract_evidence",
                f"M4 Contact report schema is stale: {report_version!r}",
                geometry_ref=f"{asset}|{model_id}",
            )
        if m4_worker_execution_policy_manifest_errors(
            report.get("worker_execution_policy_manifest")
        ):
            raise _error(
                "stale_contact_contract_evidence",
                f"M4 worker execution policy changed for "
                f"{asset}|{model_id}|{use}",
                geometry_ref=f"{asset}|{model_id}",
            )
        legacy_fields = (
            "contact_contract_id",
            "contact_contract_hash",
            "candidate_use",
            "use_profile",
            "contact_count",
            "contact_groups",
        )
        if report_version == 3:
            report_binding = row.get("contact_contract_binding")
            if (
                not isinstance(report_binding, Mapping)
                or canonical_data(report_binding) != canonical_data(binding)
                or row.get("contact_contract_hash")
                != binding.get("contact_contract_hash")
                or report.get("pick_operation_placement_policy_version")
                != PICK_OPERATION_PLACEMENT_POLICY_VERSION
                or report.get("pick_operation_placement_policy_hash")
                != PICK_OPERATION_PLACEMENT_POLICY_HASH
            ):
                raise _error(
                    "stale_contact_contract_evidence",
                    f"M4 Contact row binding changed for "
                    f"{asset}|{model_id}|{use}",
                    geometry_ref=f"{asset}|{model_id}",
                )
        else:
            plan = self._load_m4_probe_plan(report)
            jobs = [
                job
                for job in plan.get("jobs") or []
                if str(job.get("asset")) == asset
                and str(job.get("model_id")) == model_id
                and str(job.get("use")) == use
            ]
            plan_binding = (
                jobs[0].get("contact_contract_binding")
                if len(jobs) == 1
                else None
            )
            if not isinstance(plan_binding, Mapping) or any(
                plan_binding.get(field) != binding.get(field)
                for field in legacy_fields
            ):
                raise _error(
                    "stale_contact_contract_evidence",
                    f"legacy M4 Contact plan binding changed for "
                    f"{asset}|{model_id}|{use}",
                    geometry_ref=f"{asset}|{model_id}",
                )

        all_attempts = self._m4_attempts_for_row(
            report,
            asset=asset,
            model_id=model_id,
            use=use,
        )
        accounting_errors = m4_row_attempt_accounting_errors(
            row,
            all_attempts,
            require_seed_coverage=True,
        )
        if accounting_errors:
            raise _error(
                "stale_contact_contract_evidence",
                f"M4 Contact attempt accounting changed for "
                f"{asset}|{model_id}|{use}: "
                + ", ".join(accounting_errors),
                geometry_ref=f"{asset}|{model_id}",
            )
        attempts = [
            attempt
            for attempt in all_attempts
            if not m4_attempt_is_blocking(attempt)
        ]
        if len(attempts) != int(row.get("denominator") or 0):
            raise _error(
                "stale_contact_contract_evidence",
                f"M4 Contact denominator changed for {asset}|{model_id}|{use}",
                geometry_ref=f"{asset}|{model_id}",
            )
        for attempt in attempts:
            if m4_worker_execution_policy_manifest_errors(
                attempt.get("execution_policy_manifest"),
                use=use,
            ):
                raise _error(
                    "stale_contact_contract_evidence",
                    f"M4 worker attempt policy changed for "
                    f"{asset}|{model_id}|{use}",
                    geometry_ref=f"{asset}|{model_id}",
                )
            observed = (attempt.get("diagnostics") or {}).get(
                "contact_contract_preflight"
            )
            if not isinstance(observed, Mapping) or any(
                observed.get(field) != binding.get(field)
                for field in legacy_fields
            ):
                raise _error(
                    "stale_contact_contract_evidence",
                    f"M4 Contact runtime preflight changed for "
                    f"{asset}|{model_id}|{use}",
                    geometry_ref=f"{asset}|{model_id}",
                )
            if report_version == 3 and (
                resolved_contact_preflight_policy_version(
                    observed, binding
                )
                != binding.get("policy_version")
                or observed.get("contact_projection_hash")
                != binding.get("contact_projection_hash")
            ):
                raise _error(
                    "stale_contact_contract_evidence",
                    f"M4 Contact projection changed for "
                    f"{asset}|{model_id}|{use}",
                    geometry_ref=f"{asset}|{model_id}",
                )

    def _verify_m4_adjust_contact_report_binding(
        self,
        report: Mapping[str, Any],
        *,
        row: Mapping[str, Any],
        binding: Mapping[str, Any],
        asset: str,
        model_id: str,
        use: str,
    ) -> None:
        """Verify the attempt-owned Contact contract of orientation adjust.

        Older M4 adjustment rows intentionally described acquisition
        eligibility and did not duplicate the Contact binding at row level.
        Their physical attempts still froze the exact grasp plan and Contact
        contract hash.  This narrow rebind accepts only that exact evidence.
        """

        if use != "adjust_orientation_subject" or report.get(
            "schema_version"
        ) != ASSET_VALIDATION_REPORT_SCHEMA_VERSION:
            raise _error(
                "stale_contact_contract_evidence",
                f"orientation-adjust Contact rebind is not legal for "
                f"{asset}|{model_id}|{use}",
                geometry_ref=f"{asset}|{model_id}",
            )
        if m4_worker_execution_policy_manifest_errors(
            report.get("worker_execution_policy_manifest")
        ):
            raise _error(
                "stale_contact_contract_evidence",
                f"M4 worker execution policy changed for "
                f"{asset}|{model_id}|{use}",
                geometry_ref=f"{asset}|{model_id}",
            )
        all_attempts = self._m4_attempts_for_row(
            report,
            asset=asset,
            model_id=model_id,
            use=use,
        )
        accounting_errors = m4_row_attempt_accounting_errors(
            row,
            all_attempts,
            require_seed_coverage=True,
        )
        if accounting_errors:
            raise _error(
                "stale_contact_contract_evidence",
                f"M4 orientation-adjust attempt accounting changed for "
                f"{asset}|{model_id}|{use}: "
                + ", ".join(accounting_errors),
                geometry_ref=f"{asset}|{model_id}",
            )
        attempts = [
            attempt
            for attempt in all_attempts
            if not m4_attempt_is_blocking(attempt)
        ]
        if len(attempts) != int(row.get("denominator") or 0):
            raise _error(
                "stale_contact_contract_evidence",
                f"M4 orientation-adjust denominator changed for "
                f"{asset}|{model_id}|{use}",
                geometry_ref=f"{asset}|{model_id}",
            )
        expected_hash = str(binding.get("contact_contract_hash") or "")
        expected_ids = set(range(int(binding.get("contact_count") or 0)))
        for attempt in attempts:
            if m4_worker_execution_policy_manifest_errors(
                attempt.get("execution_policy_manifest"), use=use
            ):
                raise _error(
                    "stale_contact_contract_evidence",
                    f"M4 orientation-adjust worker attempt policy changed "
                    f"for {asset}|{model_id}|{use}",
                    geometry_ref=f"{asset}|{model_id}",
                )
            plans = (attempt.get("diagnostics") or {}).get(
                "exact_grasp_plans"
            )
            if not isinstance(plans, list) or not plans:
                raise _error(
                    "stale_contact_contract_evidence",
                    f"M4 orientation-adjust attempt lacks exact grasp "
                    f"lineage for {asset}|{model_id}|{use}",
                    geometry_ref=f"{asset}|{model_id}",
                )
            for plan in plans:
                candidate = (plan or {}).get("best_candidate") or {}
                allowed = (plan or {}).get("allowed_contact_ids") or []
                contact_id = candidate.get("contact_id")
                if (
                    candidate.get("contact_contract_hash") != expected_hash
                    or not isinstance(contact_id, int)
                    or contact_id not in expected_ids
                    or contact_id not in allowed
                    or any(value not in expected_ids for value in allowed)
                ):
                    raise _error(
                        "stale_contact_contract_evidence",
                        f"M4 orientation-adjust Contact plan changed for "
                        f"{asset}|{model_id}|{use}",
                        geometry_ref=f"{asset}|{model_id}",
                    )

    def _verify_m4_touch_report_binding(
        self,
        report: Mapping[str, Any],
        *,
        row: Mapping[str, Any],
        binding: Mapping[str, Any],
        asset: str,
        model_id: str,
        use: str,
    ) -> None:
        """Prove exact per-arm click/press capability and shared anchor lineage."""

        if touch_capability_errors(binding, use=use):
            raise _error(
                "stale_touch_execution_evidence",
                f"M4 touch binding is invalid for {asset}|{model_id}|{use}",
                geometry_ref=f"{asset}|{model_id}",
            )
        expected = {
            "touch_capability_version": binding.get("capability_version"),
            "touch_arm_sampling_version": binding.get("sampling_version"),
            "touch_requested_arms": binding.get("requested_arms"),
            "touch_arm_results": binding.get("arm_results"),
            "touch_arm_capabilities": binding.get("arm_capabilities"),
            "touch_canonical_placement_witness_hash": binding.get(
                "canonical_placement_witness_hash"
            ),
        }
        if any(canonical_data(row.get(key)) != canonical_data(value) for key, value in expected.items()):
            raise _error(
                "stale_touch_execution_evidence",
                f"M4 touch row binding changed for {asset}|{model_id}|{use}",
                geometry_ref=f"{asset}|{model_id}",
            )
        attempts = self._m4_attempts_for_row(
            report, asset=asset, model_id=model_id, use=use
        )
        accounting_errors = m4_row_attempt_accounting_errors(
            row, attempts, require_seed_coverage=True
        )
        if accounting_errors:
            raise _error(
                "stale_touch_execution_evidence",
                f"M4 touch attempt accounting changed for "
                f"{asset}|{model_id}|{use}: {', '.join(accounting_errors)}",
                geometry_ref=f"{asset}|{model_id}",
            )
        observed_hashes = {
            str((attempt.get("canonical_placement_witness") or {}).get("witness_hash"))
            for attempt in attempts
            if not m4_attempt_is_blocking(attempt)
        }
        if observed_hashes != {str(binding.get("canonical_placement_witness_hash"))}:
            raise _error(
                "stale_touch_execution_evidence",
                f"M4 touch placement witness changed for {asset}|{model_id}|{use}",
                geometry_ref=f"{asset}|{model_id}",
            )

    def _verify_m4_articulation_report_binding(
        self,
        report: Mapping[str, Any],
        *,
        row: Mapping[str, Any],
        binding: Mapping[str, Any],
        asset: str,
        model_id: str,
        use: str,
    ) -> None:
        """Prove that the passing M4 row executed the frozen A2O profile."""

        if m4_worker_execution_policy_manifest_errors(
            report.get("worker_execution_policy_manifest")
        ):
            raise _error(
                "stale_articulation_execution_evidence",
                f"M4 worker execution policy changed for "
                f"{asset}|{model_id}|{use}",
                geometry_ref=f"{asset}|{model_id}",
            )

        expected_lists = {
            "articulation_execution_profile_versions": [
                binding.get("profile_version")
            ],
            "articulation_execution_profile_hashes": [
                binding.get("profile_hash")
            ],
            "articulation_execution_profile_source_hashes": [
                binding.get("profile_source_hash")
            ],
            "articulation_motion_policy_versions": [
                binding.get("motion_policy_version")
            ],
            "articulation_motion_policy_hashes": [
                binding.get("motion_policy_hash")
            ],
        }
        if any(row.get(key) != value for key, value in expected_lists.items()):
            raise _error(
                "stale_articulation_execution_evidence",
                f"M4 articulation row binding changed for "
                f"{asset}|{model_id}|{use}",
                geometry_ref=f"{asset}|{model_id}",
            )
        all_attempts = self._m4_attempts_for_row(
            report,
            asset=asset,
            model_id=model_id,
            use=use,
        )
        accounting_errors = m4_row_attempt_accounting_errors(
            row,
            all_attempts,
            require_seed_coverage=True,
        )
        if accounting_errors:
            raise _error(
                "stale_articulation_execution_evidence",
                f"M4 articulation attempt accounting changed for "
                f"{asset}|{model_id}|{use}: "
                + ", ".join(accounting_errors),
                geometry_ref=f"{asset}|{model_id}",
            )
        attempts = [
            attempt
            for attempt in all_attempts
            if not m4_attempt_is_blocking(attempt)
        ]
        if len(attempts) != int(row.get("denominator") or 0):
            raise _error(
                "stale_articulation_execution_evidence",
                f"M4 articulation denominator changed for "
                f"{asset}|{model_id}|{use}",
                geometry_ref=f"{asset}|{model_id}",
            )
        expected_runtime = {
            "execution_profile_version": binding.get("profile_version"),
            "execution_profile_hash": binding.get("profile_hash"),
            "execution_profile_source_hash": binding.get(
                "profile_source_hash"
            ),
            "motion_policy_version": binding.get("motion_policy_version"),
            "motion_policy_hash": binding.get("motion_policy_hash"),
        }
        for attempt in attempts:
            if m4_worker_execution_policy_manifest_errors(
                attempt.get("execution_policy_manifest"),
                use=use,
            ):
                raise _error(
                    "stale_articulation_execution_evidence",
                    f"M4 worker attempt policy changed for "
                    f"{asset}|{model_id}|{use}",
                    geometry_ref=f"{asset}|{model_id}",
                )
            observed = (attempt.get("diagnostics") or {}).get(
                "articulation_action"
            )
            if not isinstance(observed, Mapping) or any(
                observed.get(field) != value
                for field, value in expected_runtime.items()
            ):
                raise _error(
                    "stale_articulation_execution_evidence",
                    f"M4 articulation runtime profile changed for "
                    f"{asset}|{model_id}|{use}",
                    geometry_ref=f"{asset}|{model_id}",
                )
        capability = aggregate_articulation_capabilities(
            attempts,
            use=use,
            denominator=len(attempts),
            threshold=float(row.get("threshold") or 0.0),
            require_explicit_runtime_capability=(
                report.get("schema_version")
                == ASSET_VALIDATION_REPORT_SCHEMA_VERSION
            ),
        )
        if (
            capability["failures"]
            or binding.get("capability_version")
            != capability["capability_version"]
            or canonical_data(binding.get("arm_capabilities") or [])
            != capability["arm_capabilities"]
            or articulation_capability_errors(binding, use=use)
        ):
            raise _error(
                "stale_articulation_execution_evidence",
                f"M4 articulation arm/Contact capability changed for "
                f"{asset}|{model_id}|{use}",
                geometry_ref=f"{asset}|{model_id}",
            )

    def _m4_pose_evidence(
        self,
        realized: Mapping[str, Any] | None,
        binding: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        if not realized:
            return None
        assignment = (realized.get("evidence") or {}).get("assignment") or {}
        refs = [
            row
            for row in assignment.get("evidence_refs", [])
            if row.get("kind") == "asset_validation"
            and row.get("stage") == "canonical_skill"
            and row.get("status") == "validated"
            and str(row.get("asset")) == str(binding.get("asset"))
            and str(row.get("model_id")) == str(binding.get("model_id"))
            and str(row.get("use") or "")
            in {"pick_left", "pick_right", "handover"}
            and row.get("path")
        ]
        if not refs:
            return None
        required_uses = sorted({str(row["use"]) for row in refs})
        paths = sorted({str(row["path"]) for row in refs})
        reports = [self._load_m4_report(path) for path in paths]
        selected = select_common_canonical_pose(
            reports,
            asset=str(binding["asset"]),
            model_id=str(binding["model_id"]),
            required_uses=required_uses,
            minimum_success_count=2,
        )
        if selected is None:
            raise _error(
                "missing_common_canonical_pose",
                "no variant-invariant M4 pose has at least two successful "
                f"seeds for {binding['asset']}|{binding['model_id']} across "
                f"{required_uses}",
                geometry_ref=f"{binding['asset']}|{binding['model_id']}",
            )
        return {
            **selected,
            "report_paths": paths,
            "report_hashes": {
                path: str(report["report_hash"])
                for path, report in zip(paths, reports)
            },
            "evidence_record_ids": sorted(
                {str(row.get("record_id")) for row in refs}
            ),
        }

    def footprint(
        self,
        geometry_ref: str,
        binding: Mapping[str, Any],
        *,
        realized: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        pose_evidence = self._m4_pose_evidence(realized, binding)
        pose_hash = artifact_hash(pose_evidence) if pose_evidence else ""
        cache_key = (geometry_ref, pose_hash)
        if cache_key in self._cache:
            self.cache_hits += 1
            return deepcopy(self._cache[cache_key])
        self.cache_misses += 1
        geometry = self.models.get(geometry_ref)
        if not isinstance(geometry, dict):
            raise _error(
                "missing_geometry",
                f"geometry summary has no record for {geometry_ref}",
                geometry_ref=geometry_ref,
            )
        endpoint = {
            "asset": str(binding["asset"]),
            "model_id": str(binding["model_id"]),
            "category": str(binding.get("category") or ""),
            "metadata": geometry,
        }
        use = "open" if geometry.get("has_articulation") else "layout"
        if geometry.get("has_articulation") and self.asset_catalog:
            catalog_model = (
                (
                    (self.asset_catalog.get("assets") or {}).get(
                        endpoint["asset"]
                    )
                    or {}
                )
                .get("models", {})
                .get(endpoint["model_id"])
                or {}
            )
            drives = [
                dict(candidate.get("articulation_drive") or {})
                for _, candidate in sorted(
                    (catalog_model.get("candidates") or {}).items()
                )
                if (candidate.get("articulation_drive") or {}).get(
                    "root_qpos"
                )
            ]
            if not drives:
                raise _error(
                    "missing_canonical_pose",
                    "asset catalog has no articulated root pose for "
                    f"{geometry_ref}",
                    geometry_ref=geometry_ref,
                )
            root_poses = {
                tuple(round(float(value), 8) for value in drive["root_qpos"])
                for drive in drives
            }
            if len(root_poses) != 1:
                raise _error(
                    "contradictory_canonical_pose",
                    "asset catalog has conflicting articulated root poses for "
                    f"{geometry_ref}",
                    geometry_ref=geometry_ref,
                )
            endpoint["articulation_drive"] = drives[0]
        canonical_pose = resolve_canonical_pose(endpoint, use, role="layout")
        if pose_evidence is not None and not geometry.get("has_articulation"):
            canonical_pose = {
                **canonical_pose,
                "q0_wxyz": list(pose_evidence["q0_wxyz"]),
                "z": float(pose_evidence["z"]),
                "source": f"m4_common:{pose_evidence['source']}",
                "resolved": True,
            }
        if not canonical_pose.get("resolved"):
            raise _error(
                "missing_canonical_pose",
                f"canonical resting pose is unresolved for {geometry_ref}",
                geometry_ref=geometry_ref,
            )
        if geometry.get("has_articulation"):
            result = self._articulated_footprint(
                geometry_ref, geometry, canonical_pose
            )
        else:
            result = self._rigid_footprint(
                geometry_ref,
                endpoint,
                canonical_pose,
                pose_evidence=pose_evidence,
            )
        self._cache[cache_key] = canonical_data(result)
        return deepcopy(self._cache[cache_key])

    def _rigid_footprint(
        self,
        geometry_ref: str,
        endpoint: Mapping[str, Any],
        canonical_pose: Mapping[str, Any],
        *,
        pose_evidence: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        projected = pose_aware_box_geometry(dict(endpoint), dict(canonical_pose))
        if projected is None:
            raise _error(
                "missing_geometry",
                f"pose-aware geometry cannot be built for {geometry_ref}",
                geometry_ref=geometry_ref,
            )
        footprint = _rect(
            [0.0, 0.0],
            [float(projected["x"]) / 2.0, float(projected["y"]) / 2.0],
            source="rigid_resting_pose_aabb",
        )
        return {
            "geometry_ref": geometry_ref,
            "footprint": footprint,
            "height": _clean(float(projected["height"])),
            "local_z_bounds": _vector(
                [-float(projected["height"]) / 2.0, float(projected["height"]) / 2.0]
            ),
            "canonical_pose": canonical_data(canonical_pose),
            "geometry_evidence": {
                "geometry_ref": geometry_ref,
                "source": projected["source"],
                "pose_policy_hash": canonical_pose_policy_hash(),
                "canonical_pose_source": canonical_pose["source"],
                "canonical_pose_evidence": (
                    canonical_data(dict(pose_evidence))
                    if pose_evidence is not None
                    else None
                ),
                "articulated": False,
                "state_range": [],
                "padding_m": 0.0,
            },
        }

    def _mesh_local_bounds(
        self, path: Path, mesh_scale: Sequence[float]
    ) -> tuple[np.ndarray, np.ndarray]:
        resolved = path.resolve()
        if resolved not in self._mesh_bounds:
            try:
                import trimesh

                mesh = trimesh.load(resolved, force="scene", process=False)
                bounds = np.asarray(mesh.bounds, dtype=float)
            except Exception as exc:
                raise _error(
                    "articulated_sweep_failure",
                    f"cannot read collision mesh {path}: {exc}",
                ) from exc
            if bounds.shape != (2, 3):
                raise _error(
                    "articulated_sweep_failure",
                    f"collision mesh has invalid bounds: {path}",
                )
            self._mesh_bounds[resolved] = (bounds[0], bounds[1])
        lower, upper = self._mesh_bounds[resolved]
        scale = np.asarray(mesh_scale, dtype=float)
        scaled = np.vstack((lower * scale, upper * scale))
        return scaled.min(axis=0), scaled.max(axis=0)

    def _articulated_footprint(
        self,
        geometry_ref: str,
        geometry: Mapping[str, Any],
        canonical_pose: Mapping[str, Any],
    ) -> dict[str, Any]:
        model_data_path = self.workspace_root / str(geometry["source_path"])
        model_dir = model_data_path.parent
        urdf_path = model_dir / "mobility.urdf"
        if not urdf_path.is_file():
            raise _error(
                "articulated_sweep_failure",
                f"missing URDF for {geometry_ref}: {urdf_path}",
                geometry_ref=geometry_ref,
            )
        try:
            model_data = json.loads(model_data_path.read_text(encoding="utf-8"))
            root = ET.parse(urdf_path).getroot()
        except Exception as exc:
            raise _error(
                "articulated_sweep_failure",
                f"cannot parse articulated sources for {geometry_ref}: {exc}",
                geometry_ref=geometry_ref,
            ) from exc

        raw_scale = model_data.get("scale", 1.0)
        if isinstance(raw_scale, list):
            global_scale = float(raw_scale[0])
        else:
            global_scale = float(raw_scale)
        link_collisions: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
        max_radius = 0.0
        for link in root.findall("link"):
            rows: list[tuple[np.ndarray, np.ndarray]] = []
            for collision in link.findall("collision"):
                origin = _xyz_rpy_transform(collision.find("origin"))
                geometry_node = collision.find("geometry")
                if geometry_node is None:
                    continue
                local_lower: np.ndarray
                local_upper: np.ndarray
                mesh = geometry_node.find("mesh")
                box = geometry_node.find("box")
                sphere = geometry_node.find("sphere")
                cylinder = geometry_node.find("cylinder")
                if mesh is not None and mesh.get("filename"):
                    mesh_scale = [
                        float(value)
                        for value in mesh.get("scale", "1 1 1").split()
                    ]
                    local_lower, local_upper = self._mesh_local_bounds(
                        model_dir / str(mesh.get("filename")), mesh_scale
                    )
                elif box is not None:
                    size = np.asarray(
                        [float(value) for value in box.get("size", "").split()],
                        dtype=float,
                    )
                    local_lower, local_upper = -size / 2.0, size / 2.0
                elif sphere is not None:
                    radius = float(sphere.get("radius", "0"))
                    local_lower = np.asarray([-radius] * 3)
                    local_upper = np.asarray([radius] * 3)
                elif cylinder is not None:
                    radius = float(cylinder.get("radius", "0"))
                    length = float(cylinder.get("length", "0"))
                    local_lower = np.asarray([-radius, -radius, -length / 2.0])
                    local_upper = np.asarray([radius, radius, length / 2.0])
                else:
                    continue
                corners = (origin @ _box_corners(local_lower, local_upper).T).T[:, :3]
                corners *= global_scale
                lower, upper = corners.min(axis=0), corners.max(axis=0)
                max_radius = max(
                    max_radius,
                    max(float(np.linalg.norm(point)) for point in corners),
                )
                rows.append((lower, upper))
            link_collisions[str(link.get("name"))] = rows
        if not any(link_collisions.values()):
            raise _error(
                "articulated_sweep_failure",
                f"URDF has no usable collision geometry for {geometry_ref}",
                geometry_ref=geometry_ref,
            )

        joints: list[dict[str, Any]] = []
        children: set[str] = set()
        for joint in root.findall("joint"):
            parent = joint.find("parent")
            child = joint.find("child")
            if parent is None or child is None:
                continue
            joint_type = str(joint.get("type") or "fixed")
            axis_node = joint.find("axis")
            axis = np.asarray(
                [
                    float(value)
                    for value in (
                        axis_node.get("xyz", "1 0 0")
                        if axis_node is not None
                        else "1 0 0"
                    ).split()
                ],
                dtype=float,
            )
            limit = joint.find("limit")
            lower = upper = 0.0
            if joint_type in {"revolute", "prismatic"} and limit is not None:
                lower = float(limit.get("lower", "0"))
                upper = float(limit.get("upper", "0"))
                if joint_type == "prismatic":
                    lower *= global_scale
                    upper *= global_scale
            samples = [0.0]
            step = 0.0
            if joint_type == "revolute" and upper > lower:
                step = math.radians(ARTICULATION_SWEEP_STEP_DEG)
                count = max(1, int(math.ceil((upper - lower) / step)))
                samples = [
                    lower + (upper - lower) * index / count
                    for index in range(count + 1)
                ]
                step = (upper - lower) / count
            elif joint_type == "continuous":
                lower = 0.0
                upper = 2.0 * math.pi
                count = max(
                    1,
                    int(
                        math.ceil(
                            (upper - lower)
                            / math.radians(ARTICULATION_SWEEP_STEP_DEG)
                        )
                    ),
                )
                samples = [
                    lower + (upper - lower) * index / count
                    for index in range(count)
                ]
                step = (upper - lower) / count
            elif joint_type == "prismatic" and upper > lower:
                samples = [lower, upper]
                step = upper - lower
            joint_origin = _xyz_rpy_transform(joint.find("origin"))
            joint_origin[:3, 3] *= global_scale
            joints.append(
                {
                    "name": str(joint.get("name") or ""),
                    "type": joint_type,
                    "parent": str(parent.get("link")),
                    "child": str(child.get("link")),
                    "origin": joint_origin,
                    "axis": axis,
                    "samples": samples,
                    "step": step,
                    "lower": lower,
                    "upper": upper,
                }
            )
            children.add(str(child.get("link")))
        moving = [row for row in joints if len(row["samples"]) > 1]
        parent_by_link = {
            row["child"]: row["parent"] for row in joints
        }

        def is_ancestor(link: str, descendant: str) -> bool:
            if link == descendant:
                return True
            current = descendant
            while current in parent_by_link:
                current = parent_by_link[current]
                if current == link:
                    return True
            return False

        has_nested_motion = any(
            first is not second
            and is_ancestor(first["child"], second["parent"])
            for first in moving
            for second in moving
        )
        moving_names = [row["name"] for row in moving]
        sample_domains = [row["samples"] for row in moving]
        if not moving:
            configurations: list[tuple[float, ...]] = [()]
            sweep_mode = "static"
        elif has_nested_motion:
            configurations = list(product(*sample_domains))
            sweep_mode = "joint_chain_cartesian"
        else:
            baseline = tuple(float(values[0]) for values in sample_domains)
            configuration_set = {baseline}
            for joint_index, values in enumerate(sample_domains):
                for value in values:
                    candidate = list(baseline)
                    candidate[joint_index] = float(value)
                    configuration_set.add(tuple(candidate))
            configurations = sorted(configuration_set)
            sweep_mode = "independent_branch_envelope"
        config_count = len(configurations)
        if config_count > MAX_ARTICULATION_CONFIGS:
            raise _error(
                "articulated_sweep_failure",
                f"{geometry_ref} requires {config_count} sweep configurations",
                geometry_ref=geometry_ref,
            )
        roots = sorted(set(link_collisions) - children)
        joint_by_child = {row["child"]: row for row in joints}

        transform_matrix = np.asarray(
            model_data.get("transform_matrix") or np.eye(4), dtype=float
        )
        root_transform = np.eye(4, dtype=float)
        root_transform[:3, :3] = (
            transform_matrix[:3, :3]
            @ _quaternion_matrix(canonical_pose["q0_wxyz"])
        )

        lower_xyz = np.asarray([math.inf, math.inf, math.inf], dtype=float)
        upper_xyz = np.asarray([-math.inf, -math.inf, -math.inf], dtype=float)
        for values in configurations:
            values_by_name = dict(zip(moving_names, values))
            link_world: dict[str, np.ndarray] = {
                link: root_transform.copy() for link in roots
            }
            pending = set(link_collisions) - set(roots)
            while pending:
                progressed = False
                for child in sorted(list(pending)):
                    joint = joint_by_child.get(child)
                    if joint is None:
                        link_world[child] = root_transform.copy()
                        pending.remove(child)
                        progressed = True
                        continue
                    if joint["parent"] not in link_world:
                        continue
                    value = float(values_by_name.get(joint["name"], 0.0))
                    link_world[child] = (
                        link_world[joint["parent"]]
                        @ joint["origin"]
                        @ _axis_motion(joint["type"], joint["axis"], value)
                    )
                    pending.remove(child)
                    progressed = True
                if not progressed:
                    raise _error(
                        "articulated_sweep_failure",
                        f"URDF link graph is unresolved for {geometry_ref}",
                        geometry_ref=geometry_ref,
                    )
            for link, collision_rows in link_collisions.items():
                matrix = link_world.get(link, root_transform)
                for local_lower, local_upper in collision_rows:
                    points = (
                        matrix @ _box_corners(local_lower, local_upper).T
                    ).T[:, :3]
                    lower_xyz = np.minimum(lower_xyz, points.min(axis=0))
                    upper_xyz = np.maximum(upper_xyz, points.max(axis=0))

        interpolation_padding = ARTICULATION_SWEEP_PADDING_M
        for joint in moving:
            if joint["type"] == "revolute":
                interpolation_padding += max_radius * math.sin(
                    abs(float(joint["step"])) / 2.0
                )
        lower_xyz[:2] -= interpolation_padding
        upper_xyz[:2] += interpolation_padding
        footprint = _rect(
            (lower_xyz[:2] + upper_xyz[:2]) / 2.0,
            (upper_xyz[:2] - lower_xyz[:2]) / 2.0,
            source="articulated_conservative_swept_aabb",
        )
        return {
            "geometry_ref": geometry_ref,
            "footprint": footprint,
            "height": _clean(upper_xyz[2] - lower_xyz[2]),
            "local_z_bounds": _vector([lower_xyz[2], upper_xyz[2]]),
            "canonical_pose": canonical_data(canonical_pose),
            "geometry_evidence": {
                "geometry_ref": geometry_ref,
                "source": "urdf_collision_kinematic_sweep",
                "urdf_path": str(urdf_path),
                "pose_policy_hash": canonical_pose_policy_hash(),
                "canonical_pose_source": canonical_pose["source"],
                "articulated": True,
                "state_range": [
                    {
                        "name": row["name"],
                        "type": row["type"],
                        "lower": _clean(row["lower"]),
                        "upper": _clean(row["upper"]),
                        "sample_count": len(row["samples"]),
                    }
                    for row in moving
                ],
                "configuration_count": config_count,
                "sweep_mode": sweep_mode,
                "max_step_deg": ARTICULATION_SWEEP_STEP_DEG,
                "interpolation_padding_m": _clean(interpolation_padding),
                "padding_m": _clean(interpolation_padding),
            },
        }


def _role_geometry(
    realized: Mapping[str, Any],
    unit_input: Mapping[str, Any],
    provider: FootprintProvider,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for role in unit_input.get("roles", []):
        binding = (realized.get("role_bindings") or {}).get(role)
        geometry_ref = (unit_input.get("geometry_refs") or {}).get(role)
        if not isinstance(binding, dict) or not geometry_ref:
            raise _error(
                "invalid_unit_membership",
                f"unit input lacks binding/geometry for role {role}",
                realized_family_id=str(realized.get("id")),
                unit_input_id=str(unit_input.get("id")),
            )
        initial_state = (unit_input.get("initial_state_refs") or {}).get(role)
        if isinstance(initial_state, Mapping):
            result[str(role)] = _initial_state_geometry(
                str(geometry_ref), initial_state
            )
        else:
            result[str(role)] = provider.footprint(
                str(geometry_ref), binding, realized=realized
            )
    return result


def _role_operation_anchors(
    realized: Mapping[str, Any],
    unit_input: Mapping[str, Any],
    geometry: Mapping[str, Mapping[str, Any]],
    provider: FootprintProvider,
) -> dict[str, dict[str, Any]]:
    """Bind entity keys to immutable Contact/functional anchor contracts."""

    result: dict[str, dict[str, Any]] = {}
    roles = [str(value) for value in unit_input.get("roles", [])]
    entities = [str(value) for value in unit_input.get("entity_keys", [])]
    assignment_evidence = (
        ((realized.get("evidence") or {}).get("assignment") or {}).get(
            "evidence_refs"
        )
        or []
    )
    for role, entity in zip(roles, entities):
        binding = (realized.get("role_bindings") or {}).get(role) or {}
        canonical_pose = (geometry.get(role) or {}).get("canonical_pose") or {}
        q0_wxyz = canonical_pose.get("q0_wxyz")
        if not isinstance(q0_wxyz, list) or len(q0_wxyz) != 4:
            continue
        contract = provider.operation_anchor_contract(
            binding,
            q0_wxyz,
            scale_xyz=canonical_pose.get("scale_xyz"),
            articulated_joint_state=canonical_pose.get(
                "articulated_joint_state"
            ),
            canonical_state_hash=canonical_pose.get(
                "canonical_state_hash"
            ),
            assignment_state_revision=canonical_pose.get(
                "assignment_state_revision"
            ),
            validated_evidence=[
                row
                for row in assignment_evidence
                if str((row.get("metadata") or {}).get("role") or "")
                == role
            ],
        )
        if isinstance(contract, dict) and contract.get("anchors"):
            result[entity] = contract
    return result


def _initial_state_geometry(
    geometry_ref: str, initial_state: Mapping[str, Any]
) -> dict[str, Any]:
    bounds = initial_state.get("origin_relative_bounds") or {}
    minimum = _vector(bounds.get("min_xyz") or [])
    maximum = _vector(bounds.get("max_xyz") or [])
    if len(minimum) != 3 or len(maximum) != 3:
        raise _error(
            "stale_initial_pose_geometry",
            f"Initial Pose bounds missing for {geometry_ref}",
            geometry_ref=geometry_ref,
        )
    size_x = float(maximum[0]) - float(minimum[0])
    size_y = float(maximum[1]) - float(minimum[1])
    height = float(maximum[2]) - float(minimum[2])
    if min(size_x, size_y, height) <= 0:
        raise _error(
            "stale_initial_pose_geometry",
            f"Initial Pose bounds are degenerate for {geometry_ref}",
            geometry_ref=geometry_ref,
        )
    dynamic_mass_properties = initial_state.get("dynamic_mass_properties")
    if not isinstance(dynamic_mass_properties, Mapping):
        raise _error(
            "stale_initial_pose_geometry",
            f"Initial Pose dynamic mass contract missing for {geometry_ref}",
            geometry_ref=geometry_ref,
        )
    support = initial_state.get("support") or {}
    return {
        "geometry_ref": geometry_ref,
        "footprint": _rect(
            [
                (float(minimum[0]) + float(maximum[0])) / 2.0,
                (float(minimum[1]) + float(maximum[1])) / 2.0,
            ],
            [size_x / 2.0, size_y / 2.0],
            source="approved_initial_pose_origin_relative_bounds",
        ),
        "height": _clean(height),
        "local_z_bounds": minimum[2:3] + maximum[2:3],
        "origin_relative_bounds": canonical_data(bounds),
        "support_clearance_m": float(support.get("clearance_m") or 0.0),
        "canonical_pose": {
            "q0_wxyz": _vector(initial_state["q0_wxyz"]),
            "yaw": canonical_data(initial_state.get("yaw_policy") or {}),
            "scale_xyz": _vector(initial_state.get("scale_xyz") or []),
            "origin_relative_bounds": canonical_data(bounds),
            "support": canonical_data(support),
            "articulated_joint_state": canonical_data(
                initial_state.get("articulated_joint_state") or {}
            ),
            "dynamic_mass_properties": canonical_data(
                dynamic_mass_properties
            ),
            "source": "support_validated_human_approved_initial_pose",
            "initial_state_id": initial_state.get("initial_state_id"),
            "revision": initial_state.get("revision"),
            "revision_hash": initial_state.get("revision_hash"),
            "geometry_hash": initial_state.get("geometry_hash"),
            "canonical_state_hash": initial_state.get(
                "canonical_state_hash"
            ),
            "assignment_state_revision": canonical_data(
                initial_state.get("assignment_state_revision") or {}
            ),
        },
        "geometry_evidence": {
            "geometry_ref": geometry_ref,
            "source": "approved_initial_pose_origin_relative_bounds",
            "initial_state_id": initial_state.get("initial_state_id"),
            "revision": initial_state.get("revision"),
            "revision_hash": initial_state.get("revision_hash"),
            "geometry_hash": initial_state.get("geometry_hash"),
            "support_validation_hash": initial_state.get(
                "support_validation_hash"
            ),
            "origin_relative_bounds": canonical_data(bounds),
            "support": canonical_data(support),
            "padding_m": 0.0,
        },
    }


def _resting_origin_z(geometry: Mapping[str, Any]) -> float:
    return _clean(
        -float(geometry["local_z_bounds"][0])
        + float(geometry.get("support_clearance_m") or 0.0)
    )


def _supported_origin_z(geometry: Mapping[str, Any]) -> float:
    """Return source origin height above another support surface.

    Canonical Initial Pose geometry carries its approved support clearance.
    Legacy/synthetic test geometry retains the historical unit-level clearance.
    """

    clearance = geometry.get("support_clearance_m")
    if clearance is None:
        clearance = SUPPORT_CLEARANCE_M
    return _clean(-float(geometry["local_z_bounds"][0]) + float(clearance))


def _support_plane_position(
    pose: Mapping[str, Any], z_bounds: Sequence[float]
) -> list[float]:
    """Recover the Unit support plane from an R01 placement contract."""

    translation = pose.get("translation_xyz") or []
    clearance = float((pose.get("support") or {}).get("clearance_m") or 0.0)
    return _vector(
        [
            translation[0],
            translation[1],
            float(translation[2]) + float(z_bounds[0]) - clearance,
        ]
    )


def _external_role_geometry(
    realized: Mapping[str, Any],
    role: str,
    provider: FootprintProvider,
) -> dict[str, Any]:
    binding = (realized.get("role_bindings") or {}).get(role)
    resolved = (
        (realized.get("evidence") or {})
        .get("assignment", {})
        .get("resolved_values", {})
        .get(role, {})
    )
    geometry_ref = resolved.get("geometry_ref")
    if not isinstance(binding, dict) or not geometry_ref:
        raise _error(
            "invalid_unit_membership",
            f"cannot resolve external endpoint role {role}",
            realized_family_id=str(realized.get("id")),
        )
    initial_state = resolved.get("initial_state_ref")
    if isinstance(initial_state, Mapping):
        return _initial_state_geometry(str(geometry_ref), initial_state)
    return provider.footprint(str(geometry_ref), binding, realized=realized)


def _local_pose(
    geometry: Mapping[str, Any],
    translation_xyz: Sequence[float],
    *,
    placement_mode: str,
    relative_yaw_deg: float = 0.0,
) -> dict[str, Any]:
    canonical = geometry["canonical_pose"]
    row = {
        "translation_xyz": _vector(translation_xyz),
        "q0_wxyz": _vector(canonical["q0_wxyz"]),
        "relative_yaw_deg": _clean(relative_yaw_deg),
        "placement_mode": placement_mode,
    }
    if canonical.get("initial_state_id"):
        row.update(
            {
                "yaw_policy": canonical_data(canonical.get("yaw") or {}),
                "initial_state_id": canonical.get("initial_state_id"),
                "initial_state_revision": canonical.get("revision"),
                "initial_state_revision_hash": canonical.get("revision_hash"),
                "geometry_hash": canonical.get("geometry_hash"),
                "canonical_state_hash": canonical.get(
                    "canonical_state_hash"
                ),
                "scale_xyz": canonical_data(canonical.get("scale_xyz") or []),
                "origin_relative_bounds": canonical_data(
                    canonical.get("origin_relative_bounds") or {}
                ),
                "support": canonical_data(canonical.get("support") or {}),
                "articulated_joint_state": canonical_data(
                    canonical.get("articulated_joint_state") or {}
                ),
                "dynamic_mass_properties": canonical_data(
                    canonical["dynamic_mass_properties"]
                ),
            }
        )
        if canonical.get("assignment_state_revision"):
            row["assignment_state_revision"] = canonical_data(
                canonical["assignment_state_revision"]
            )
    return row


def _pair_evidence(
    unit_input: Mapping[str, Any],
    pair_ids: Sequence[str],
    relation: str,
    provider: FootprintProvider,
) -> list[dict[str, Any]]:
    rows = unit_input.get("pair_preconditions") or {}
    result: list[dict[str, Any]] = []
    if not pair_ids:
        raise _error(
            "missing_pair_evidence",
            f"no pair precondition proves {relation}",
            unit_input_id=str(unit_input.get("id")),
        )
    expected = PAIR_RELATION[relation] + "|"
    for pair_id in pair_ids:
        row = rows.get(pair_id)
        if not isinstance(row, dict):
            raise _error(
                "missing_pair_evidence",
                f"missing pair precondition {pair_id} for {relation}",
                unit_input_id=str(unit_input.get("id")),
            )
        if not str(row.get("pair_key") or "").startswith(expected):
            raise _error(
                "stale_pair_evidence",
                f"pair precondition {pair_id} does not prove {relation}",
                unit_input_id=str(unit_input.get("id")),
            )
        result.append(
            {
                "constraint_id": pair_id,
                **provider.pair_evidence(row, relation),
            }
        )
    return result


def _matching_pair_ids(
    realized: Mapping[str, Any],
    unit_input: Mapping[str, Any],
    *,
    relation: str,
    source_role: str,
    reference_role: str,
    allowed_ids: Sequence[str] | None = None,
) -> list[str]:
    bindings = realized.get("role_bindings") or {}
    source = bindings.get(source_role) or {}
    reference = bindings.get(reference_role) or {}
    expected = (
        f"{PAIR_RELATION[relation]}|"
        f"{source.get('asset')}:{source.get('model_id')}|"
        f"{reference.get('asset')}:{reference.get('model_id')}"
    )
    allowed = (
        set(str(value) for value in allowed_ids)
        if allowed_ids is not None
        else set(str(value) for value in unit_input.get("source_constraints", []))
    )
    return [
        str(constraint_id)
        for constraint_id, row in (
            unit_input.get("pair_preconditions") or {}
        ).items()
        if (
            isinstance(row, dict)
            and constraint_id in allowed
            and str(row.get("pair_key") or "") == expected
        )
    ]


def _object_unit(
    realized: Mapping[str, Any],
    unit_input: Mapping[str, Any],
    provider: FootprintProvider,
) -> dict[str, Any]:
    roles = list(unit_input["roles"])
    if len(roles) != 1 or len(unit_input["entity_keys"]) != 1:
        raise _error(
            "invalid_unit_membership",
            "ObjectUnit must contain exactly one role/entity",
            unit_input_id=str(unit_input.get("id")),
        )
    role = str(roles[0])
    entity = str(unit_input["entity_keys"][0])
    geometry = _role_geometry(realized, unit_input, provider)[role]
    operation_anchors = _role_operation_anchors(
        realized, unit_input, {role: geometry}, provider
    )
    local_z_bounds = _vector(geometry["local_z_bounds"])
    origin_z = _resting_origin_z(geometry)
    return {
        "id": deterministic_id("unit", {"unit_input_id": unit_input["id"]}),
        "type": "ObjectUnit",
        "roles": roles,
        "entity_keys": [entity],
        "params": {
            "object_key": entity,
            "footprint": geometry["footprint"],
            "anchors": {
                "origin": [0.0, 0.0, origin_z],
                "footprint_center": [
                    *geometry["footprint"]["center_xy"],
                    origin_z,
                ],
                "support": [0.0, 0.0, 0.0],
            },
            "local_pose": _local_pose(
                geometry, [0.0, 0.0, origin_z], placement_mode="table_resting"
            ),
            "local_z_bounds": local_z_bounds,
            "geometry_evidence": geometry["geometry_evidence"],
            "operation_anchors": operation_anchors,
        },
    }


def _landmark_relation_unit(
    realized: Mapping[str, Any],
    unit_input: Mapping[str, Any],
    provider: FootprintProvider,
) -> dict[str, Any]:
    roles = [str(value) for value in unit_input["roles"]]
    entity_by_role = dict(zip(roles, unit_input["entity_keys"]))
    geometry = _role_geometry(realized, unit_input, provider)
    operation_anchors = _role_operation_anchors(
        realized, unit_input, geometry, provider
    )
    params = unit_input.get("params") or {}
    references = [str(value) for value in params.get("reference_roles", [])]
    if len(references) != 1 or references[0] not in roles:
        raise _error(
            "invalid_unit_membership",
            "LandmarkRelationUnit requires exactly one reference role",
            unit_input_id=str(unit_input.get("id")),
        )
    landmark_role = references[0]
    landmark_key = str(entity_by_role[landmark_role])
    landmark_origin_z = _resting_origin_z(geometry[landmark_role])
    offsets: dict[str, list[float]] = {
        landmark_role: [0.0, 0.0, landmark_origin_z]
    }
    placement_modes: dict[str, str] = {landmark_role: "table_resting"}
    relative_yaw_by_role: dict[str, float] = {}
    facts: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for relation_row in params.get("relations", []):
        source_role = str(relation_row["source_role"])
        reference_role = str(relation_row["reference_role"])
        if reference_role != landmark_role or source_role not in roles:
            raise _error(
                "invalid_unit_membership",
                "local relation endpoint does not match unit membership",
                unit_input_id=str(unit_input.get("id")),
            )
        predicates = [str(value) for value in relation_row.get("predicates", [])]
        unsupported = set(predicates) - SUPPORTED_LOCAL_PREDICATES
        if unsupported:
            raise _error(
                "unsupported_relation",
                f"unsupported local predicates: {sorted(unsupported)}",
                unit_input_id=str(unit_input.get("id")),
            )
        if {"left_of", "right_of"} <= set(predicates) or {
            "in_front_of",
            "behind",
        } <= set(predicates):
            raise _error(
                "contradictory_relation",
                f"contradictory local predicates: {predicates}",
                unit_input_id=str(unit_input.get("id")),
            )
        source_fp = geometry[source_role]["footprint"]
        reference_fp = geometry[reference_role]["footprint"]
        dx = dy = 0.0
        dz = _resting_origin_z(geometry[source_role])
        if "left_of" in predicates:
            dx = -(
                float(source_fp["half_extents_xy"][0])
                + float(reference_fp["half_extents_xy"][0])
                + LOCAL_RELATION_GAP_M
            )
        elif "right_of" in predicates:
            dx = (
                float(source_fp["half_extents_xy"][0])
                + float(reference_fp["half_extents_xy"][0])
                + LOCAL_RELATION_GAP_M
            )
        if "in_front_of" in predicates:
            dy = -(
                float(source_fp["half_extents_xy"][1])
                + float(reference_fp["half_extents_xy"][1])
                + LOCAL_RELATION_GAP_M
            )
        elif "behind" in predicates:
            dy = (
                float(source_fp["half_extents_xy"][1])
                + float(reference_fp["half_extents_xy"][1])
                + LOCAL_RELATION_GAP_M
            )
        topological = [value for value in predicates if value in {"in", "on"}]
        if topological:
            if len(topological) != 1 or len(predicates) != 1:
                raise _error(
                    "contradictory_relation",
                    f"topological relation cannot combine with {predicates}",
                    unit_input_id=str(unit_input.get("id")),
                )
            relation = topological[0]
            pair_ids = _matching_pair_ids(
                realized,
                unit_input,
                relation=relation,
                source_role=source_role,
                reference_role=reference_role,
            )
            relation_pair_rows = _pair_evidence(
                unit_input, pair_ids, relation, provider
            )
            pair_rows.extend(relation_pair_rows)
            relative_yaw = float(
                relation_pair_rows[0]["relative_yaw_deg"]
            )
            if relation == "on":
                dz = (
                    landmark_origin_z
                    + float(geometry[reference_role]["local_z_bounds"][1])
                    + _supported_origin_z(geometry[source_role])
                )
                placement_modes[source_role] = "support_on_landmark"
            else:
                dz = (
                    landmark_origin_z
                    + float(geometry[reference_role]["local_z_bounds"][1])
                )
                placement_modes[source_role] = "settle_into_landmark"
            relative_yaw_by_role[source_role] = relative_yaw
        else:
            placement_modes[source_role] = "table_resting"
        offsets[source_role] = [dx, dy, dz]
        for predicate in predicates:
            facts.append(
                {
                    "predicate": "spatial",
                    "subject": str(entity_by_role[source_role]),
                    "relation": predicate,
                    "reference": str(entity_by_role[reference_role]),
                    "source_constraint_ids": list(
                        unit_input.get("source_constraints", [])
                    ),
                }
            )
    if set(offsets) != set(roles):
        raise _error(
            "invalid_unit_membership",
            f"local layout does not cover roles {sorted(set(roles) - set(offsets))}",
            unit_input_id=str(unit_input.get("id")),
        )
    member_footprints = {}
    for role in roles:
        rotated = _rotated(
            geometry[role]["footprint"], relative_yaw_by_role.get(role, 0.0)
        )
        member_footprints[role] = _translated(rotated, offsets[role][:2])
    footprint = _envelope(
        list(member_footprints.values()),
        source="landmark_relation_member_envelope",
    )
    return {
        "id": deterministic_id("unit", {"unit_input_id": unit_input["id"]}),
        "type": "LandmarkRelationUnit",
        "roles": roles,
        "entity_keys": [str(value) for value in unit_input["entity_keys"]],
        "params": {
            "landmark_key": landmark_key,
            "internal_layout": {
                str(entity_by_role[role]): _local_pose(
                    geometry[role],
                    offsets[role],
                    placement_mode=placement_modes[role],
                    relative_yaw_deg=relative_yaw_by_role.get(role, 0.0),
                )
                for role in roles
            },
            "member_footprints": {
                str(entity_by_role[role]): member_footprints[role]
                for role in roles
            },
            "member_z_bounds": {
                str(entity_by_role[role]): _vector(
                    geometry[role]["local_z_bounds"]
                )
                for role in roles
            },
            "footprint": footprint,
            "anchors": {
                key: value
                for role in roles
                for key, value in (
                    (
                        f"{entity_by_role[role]}:origin",
                        _vector(offsets[role]),
                    ),
                    (
                        f"{entity_by_role[role]}:support",
                        _support_plane_position(
                            _local_pose(
                                geometry[role],
                                offsets[role],
                                placement_mode=placement_modes[role],
                                relative_yaw_deg=relative_yaw_by_role.get(
                                    role, 0.0
                                ),
                            ),
                            geometry[role]["local_z_bounds"],
                        ),
                    ),
                )
            },
            "guaranteed_facts": facts,
            "geometry_evidence": {
                str(entity_by_role[role]): geometry[role]["geometry_evidence"]
                for role in roles
            },
            "pair_evidence": pair_rows,
            "operation_anchors": operation_anchors,
        },
    }


def _receiver_reservation_unit(
    realized: Mapping[str, Any],
    unit_input: Mapping[str, Any],
    provider: FootprintProvider,
) -> dict[str, Any]:
    roles = [str(value) for value in unit_input["roles"]]
    entity_by_role = dict(zip(roles, unit_input["entity_keys"]))
    receiver_geometry = _role_geometry(realized, unit_input, provider)
    operation_anchors = _role_operation_anchors(
        realized, unit_input, receiver_geometry, provider
    )
    params = unit_input.get("params") or {}
    receiver_roles = [str(value) for value in params.get("receiver_roles", [])]
    subject_roles = [str(value) for value in params.get("subject_roles", [])]
    relations = [str(value) for value in params.get("relations", [])]
    endpoints = params.get("relation_endpoints") or {}
    if receiver_roles != roles or not subject_roles or set(endpoints) != set(relations):
        raise _error(
            "invalid_unit_membership",
            "receiver endpoint contract is incomplete",
            unit_input_id=str(unit_input.get("id")),
        )
    unsupported = set(relations) - SUPPORTED_TERMINAL_RELATIONS
    if unsupported:
        raise _error(
            "unsupported_relation",
            f"unsupported terminal relations: {sorted(unsupported)}",
            unit_input_id=str(unit_input.get("id")),
        )
    subject_geometry = {
        role: _external_role_geometry(realized, role, provider)
        for role in subject_roles
    }

    receiver_offsets: dict[str, list[float]] = {
        role: [
            0.0,
            0.0,
            _resting_origin_z(receiver_geometry[role]),
        ]
        for role in receiver_roles
    }
    if len(receiver_roles) == 2:
        max_subject_half_x = max(
            float(value["footprint"]["half_extents_xy"][0])
            for value in subject_geometry.values()
        )
        left, right = receiver_roles
        center_clearance = max_subject_half_x + TERMINAL_CLEARANCE_M
        for role, direction in ((left, -1.0), (right, 1.0)):
            footprint = receiver_geometry[role]["footprint"]
            target_aabb_center_x = direction * (
                float(footprint["half_extents_xy"][0])
                + center_clearance
            )
            receiver_offsets[role][0] = (
                target_aabb_center_x - float(footprint["center_xy"][0])
            )
            receiver_offsets[role][1] = -float(
                footprint["center_xy"][1]
            )
    elif len(receiver_roles) != 1:
        raise _error(
            "invalid_unit_membership",
            "ReceiverReservationUnit requires one or two receivers",
            unit_input_id=str(unit_input.get("id")),
        )

    receiver_footprints = {
        role: _translated(
            receiver_geometry[role]["footprint"], receiver_offsets[role][:2]
        )
        for role in receiver_roles
    }
    regions: list[dict[str, Any]] = []
    facts: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for relation in relations:
        endpoint = endpoints[relation]
        endpoint_subjects = [
            str(value) for value in endpoint.get("subject_roles", [])
        ]
        endpoint_receivers = [
            str(value) for value in endpoint.get("receiver_roles", [])
        ]
        pair_ids = [
            str(value) for value in endpoint.get("pair_constraint_ids", [])
        ]
        if (
            not endpoint_subjects
            or set(endpoint_subjects) - set(subject_roles)
            or not endpoint_receivers
            or set(endpoint_receivers) - set(receiver_roles)
        ):
            raise _error(
                "invalid_unit_membership",
                f"invalid receiver endpoints for {relation}",
                unit_input_id=str(unit_input.get("id")),
            )
        if relation not in PAIR_REQUIRED_RELATIONS and pair_ids:
            raise _error(
                "stale_pair_evidence",
                f"{relation} unexpectedly carries pair evidence",
                unit_input_id=str(unit_input.get("id")),
            )
        for subject_role in endpoint_subjects:
            relation_pair_rows: list[dict[str, Any]] = []
            if relation in PAIR_REQUIRED_RELATIONS:
                subject_pair_ids = _matching_pair_ids(
                    realized,
                    unit_input,
                    relation=relation,
                    source_role=subject_role,
                    reference_role=endpoint_receivers[0],
                    allowed_ids=pair_ids,
                )
                relation_pair_rows = _pair_evidence(
                    unit_input, subject_pair_ids, relation, provider
                )
                pair_rows.extend(relation_pair_rows)
            subject_key = str(
                realized["role_bindings"][subject_role]["entity_key"]
            )
            relative_yaw = (
                float(relation_pair_rows[0]["relative_yaw_deg"])
                if relation_pair_rows
                else 0.0
            )
            terminal_placement_recipe = (
                relation_pair_rows[0]["placement_recipe"]
                if relation_pair_rows
                else {
                    "clearance_m": TERMINAL_CLEARANCE_M,
                    "policy": "m11_non_pair_terminal_clearance_v1",
                }
            )
            terminal_clearance_m = float(
                terminal_placement_recipe.get(
                    "clearance_m", TERMINAL_CLEARANCE_M
                )
            )
            subject_fp = _rotated(
                subject_geometry[subject_role]["footprint"], relative_yaw
            )
            subject_half = [
                float(value) for value in subject_fp["half_extents_xy"]
            ]
            half = [
                float(value) + TERMINAL_CLEARANCE_M
                for value in subject_half
            ]
            primary_receiver = endpoint_receivers[0]
            receiver_fp = receiver_footprints[primary_receiver]
            center = list(receiver_fp["center_xy"])
            subject_center_offset = [
                float(value) for value in subject_fp["center_xy"]
            ]
            surface_gap_band = relation_surface_gap_band(relation)
            reserved_surface_gap: float | None = None
            if relation in {"left_of", "right_of", "next_to"}:
                direction = -1.0 if relation == "left_of" else 1.0
                if surface_gap_band is None:
                    raise _error(
                        "terminal_surface_gap_band_missing",
                        f"no shared surface-gap band for {relation}",
                        unit_input_id=str(unit_input.get("id")),
                    )
                reserved_surface_gap = (
                    TERMINAL_CLEARANCE_M + LOCAL_RELATION_GAP_M
                )
                if not (
                    surface_gap_band[0]
                    <= reserved_surface_gap
                    <= surface_gap_band[1]
                ):
                    raise _error(
                        "terminal_surface_gap_policy_unsat",
                        f"{relation} target surface gap "
                        f"{reserved_surface_gap:.6f}m is outside shared band "
                        f"{list(surface_gap_band)}",
                        unit_input_id=str(unit_input.get("id")),
                    )
                separation = (
                    float(receiver_fp["half_extents_xy"][0])
                    + subject_half[0]
                    + reserved_surface_gap
                )
                center[0] += direction * separation
                center = [
                    center[index] - subject_center_offset[index]
                    for index in range(2)
                ]
            elif relation in {"in_front_of", "behind"}:
                direction = -1.0 if relation == "in_front_of" else 1.0
                if surface_gap_band is None:
                    raise _error(
                        "terminal_surface_gap_band_missing",
                        f"no shared surface-gap band for {relation}",
                        unit_input_id=str(unit_input.get("id")),
                    )
                reserved_surface_gap = (
                    TERMINAL_CLEARANCE_M + LOCAL_RELATION_GAP_M
                )
                if not (
                    surface_gap_band[0]
                    <= reserved_surface_gap
                    <= surface_gap_band[1]
                ):
                    raise _error(
                        "terminal_surface_gap_policy_unsat",
                        f"{relation} target surface gap "
                        f"{reserved_surface_gap:.6f}m is outside shared band "
                        f"{list(surface_gap_band)}",
                        unit_input_id=str(unit_input.get("id")),
                    )
                separation = (
                    float(receiver_fp["half_extents_xy"][1])
                    + subject_half[1]
                    + reserved_surface_gap
                )
                center[1] += direction * separation
                center = [
                    center[index] - subject_center_offset[index]
                    for index in range(2)
                ]
            elif relation in {"between", "left_of_pair", "right_of_pair"}:
                if len(endpoint_receivers) != 2:
                    raise _error(
                        "invalid_unit_membership",
                        f"{relation} requires two receiver endpoints",
                        unit_input_id=str(unit_input.get("id")),
                    )
                rectangles = [
                    receiver_footprints[role] for role in endpoint_receivers
                ]
                receiver_centers = [
                    [float(value) for value in rect["center_xy"]]
                    for rect in rectangles
                ]
                if relation == "between":
                    desired_subject_center = [
                        sum(value[axis] for value in receiver_centers)
                        / len(receiver_centers)
                        for axis in range(2)
                    ]
                else:
                    pair_band = ANCHOR_PAIR_DISTANCE_BANDS_M[relation]
                    boundary_index = (
                        min(
                            range(len(rectangles)),
                            key=lambda index: receiver_centers[index][0],
                        )
                        if relation == "left_of_pair"
                        else max(
                            range(len(rectangles)),
                            key=lambda index: receiver_centers[index][0],
                        )
                    )
                    required_distance = (
                        float(
                            rectangles[boundary_index][
                                "half_extents_xy"
                            ][0]
                        )
                        + subject_half[0]
                        + ANCHOR_PAIR_TERMINAL_CLEARANCE_M
                    )
                    selected_distance = max(pair_band[0], required_distance)
                    if selected_distance > pair_band[1]:
                        raise _error(
                            "terminal_anchor_pair_policy_unsat",
                            f"{relation} collision-free center distance "
                            f"{selected_distance:.6f}m exceeds shared band "
                            f"{list(pair_band)}",
                            unit_input_id=str(unit_input.get("id")),
                        )
                    direction = -1.0 if relation == "left_of_pair" else 1.0
                    desired_subject_center = [
                        receiver_centers[boundary_index][0]
                        + direction * selected_distance,
                        sum(value[1] for value in receiver_centers)
                        / len(receiver_centers),
                    ]
                center = [
                    desired_subject_center[index]
                    - subject_center_offset[index]
                    for index in range(2)
                ]

            subject_terminal_aabb = _rect_aabb(
                _translated(subject_fp, center)
            )
            receiver_terminal_aabbs = [
                _rect_aabb(receiver_footprints[role])
                for role in endpoint_receivers
            ]
            if relation in RELATION_SURFACE_GAP_BANDS_M:
                geometry_acceptance = relation_surface_gap_acceptance(
                    subject_terminal_aabb,
                    receiver_terminal_aabbs[0],
                    relation,
                )
            elif relation in {
                "between",
                "left_of_pair",
                "right_of_pair",
            }:
                geometry_acceptance = relation_anchor_pair_acceptance(
                    subject_terminal_aabb,
                    receiver_terminal_aabbs[0],
                    receiver_terminal_aabbs[1],
                    relation,
                )
            else:
                geometry_acceptance = None
            if (
                geometry_acceptance is not None
                and geometry_acceptance["passed"] is not True
            ):
                raise _error(
                    "terminal_relation_geometry_unsat",
                    f"{relation} central reservation is outside the shared "
                    "SuccessContract geometry",
                    unit_input_id=str(unit_input.get("id")),
                )
            region_id = deterministic_id(
                "region",
                {
                    "unit_input_id": unit_input["id"],
                    "relation": relation,
                    "subject_role": subject_role,
                    "receiver_roles": endpoint_receivers,
                },
            )
            region = {
                "id": region_id,
                "relation": relation,
                "subject_key": subject_key,
                "receiver_keys": [
                    str(entity_by_role[role]) for role in endpoint_receivers
                ],
                "footprint": _rect(
                    center, half, source=f"terminal_{relation}_reservation"
                ),
                "support_clearance_m": SUPPORT_CLEARANCE_M,
                "terminal_clearance_m": _clean(terminal_clearance_m),
                "placement_recipe": terminal_placement_recipe,
                "placement_recipe_source": (
                    relation_pair_rows[0].get("placement_recipe_source")
                    if relation_pair_rows
                    else "relation_default"
                ),
                "pair_evidence_ids": [
                    row["evidence_id"]
                    for row in relation_pair_rows
                    if row.get("evidence_id")
                ],
                "relative_yaw_deg": _clean(relative_yaw),
                "usable_region": (
                    relation_pair_rows[0]["usable_region"]
                    if relation_pair_rows
                    else None
                ),
                "relation_surface_gap_band_m": (
                    list(surface_gap_band or []) or None
                ),
                "reserved_surface_gap_m": (
                    _clean(reserved_surface_gap)
                    if reserved_surface_gap is not None
                    else None
                ),
                "reserved_center_distance_m": _clean(
                    math.hypot(
                        float(center[0]) - float(receiver_fp["center_xy"][0]),
                        float(center[1]) - float(receiver_fp["center_xy"][1]),
                    )
                ),
            }
            regions.append(region)
            facts.append(
                {
                    "predicate": "terminal_region",
                    "subject": subject_key,
                    "relation": relation,
                    "receivers": region["receiver_keys"],
                    "region_id": region_id,
                    "source_constraint_ids": list(
                        unit_input.get("source_constraints", [])
                    ),
                }
            )
    if not regions:
        raise _error(
            "invalid_reservation",
            "ReceiverReservationUnit produced no terminal regions",
            unit_input_id=str(unit_input.get("id")),
        )
    footprint = _envelope(
        [
            *receiver_footprints.values(),
            *(row["footprint"] for row in regions),
        ],
        source="receiver_and_terminal_region_envelope",
    )
    return {
        "id": deterministic_id("unit", {"unit_input_id": unit_input["id"]}),
        "type": "ReceiverReservationUnit",
        "roles": roles,
        "entity_keys": [str(value) for value in unit_input["entity_keys"]],
        "params": {
            "receiver_keys": [str(entity_by_role[role]) for role in receiver_roles],
            "subject_keys": [
                str(realized["role_bindings"][role]["entity_key"])
                for role in subject_roles
            ],
            "relations": relations,
            "receiver_layout": {
                str(entity_by_role[role]): _local_pose(
                    receiver_geometry[role],
                    receiver_offsets[role],
                    placement_mode="table_resting",
                )
                for role in receiver_roles
            },
            "receiver_footprints": {
                str(entity_by_role[role]): receiver_footprints[role]
                for role in receiver_roles
            },
            "receiver_z_bounds": {
                str(entity_by_role[role]): _vector(
                    receiver_geometry[role]["local_z_bounds"]
                )
                for role in receiver_roles
            },
            "reserved_regions": regions,
            "footprint": footprint,
            "anchors": {
                key: value
                for role in receiver_roles
                for key, value in (
                    (
                        f"{entity_by_role[role]}:origin",
                        _vector(receiver_offsets[role]),
                    ),
                    (
                        f"{entity_by_role[role]}:support",
                        _support_plane_position(
                            _local_pose(
                                receiver_geometry[role],
                                receiver_offsets[role],
                                placement_mode="table_resting",
                            ),
                            receiver_geometry[role]["local_z_bounds"],
                        ),
                    ),
                )
            },
            "guaranteed_facts": facts,
            "geometry_evidence": {
                str(entity_by_role[role]): receiver_geometry[role][
                    "geometry_evidence"
                ]
                for role in receiver_roles
            },
            "subject_geometry_evidence": {
                str(realized["role_bindings"][role]["entity_key"]): subject_geometry[
                    role
                ]["geometry_evidence"]
                for role in subject_roles
            },
            "subject_z_bounds": {
                str(realized["role_bindings"][role]["entity_key"]): _vector(
                    subject_geometry[role]["local_z_bounds"]
                )
                for role in subject_roles
            },
            "pair_evidence": pair_rows,
            "operation_anchors": operation_anchors,
        },
    }


def build_units(
    realized: Mapping[str, Any], provider: FootprintProvider
) -> list[dict[str, Any]]:
    """Build the complete deterministic Unit list for one realized family."""

    builders = {
        "ObjectUnit": _object_unit,
        "LandmarkRelationUnit": _landmark_relation_unit,
        "ReceiverReservationUnit": _receiver_reservation_unit,
    }
    result: list[dict[str, Any]] = []
    for unit_input in realized.get("unit_inputs", []):
        unit_type = str(unit_input.get("type") or "")
        builder = builders.get(unit_type)
        if builder is None:
            raise _error(
                "unsupported_unit_type",
                f"unsupported Unit type: {unit_type}",
                realized_family_id=str(realized.get("id")),
                unit_input_id=str(unit_input.get("id")),
            )
        try:
            result.append(builder(realized, unit_input, provider))
        except UnitBuildError as exc:
            if exc.realized_family_id is None:
                exc.realized_family_id = str(realized.get("id"))
            if exc.unit_input_id is None:
                exc.unit_input_id = str(unit_input.get("id"))
            raise
    result = sorted(result, key=lambda row: row["id"])
    if any(v.get("goal", {}).get("action_family") == "O2O_relation"
           for v in realized.get("goal_variants", [])):
        from if_benchmark.layout.operation_catalog import OperationGeometryCatalog
        from if_benchmark.layout.manipulation import materialize_operation_spaces, reserve_pair_operation_space
        catalog = getattr(provider, "_operation_geometry_catalog", None)
        if catalog is None:
            catalog = OperationGeometryCatalog(provider.workspace_root)
            provider._operation_geometry_catalog = catalog
        materialize_operation_spaces(realized, result, catalog=catalog)
        reserve_pair_operation_space(realized, result)
    return result


def required_operation_anchor_kinds(
    realized: Mapping[str, Any],
) -> dict[str, set[str]]:
    """Return the shared M11/M12 operation-point requirements.

    Pick, handover and object-to-object skills acquire a grasp.  Articulation
    and short-contact state skills may use either a reviewed functional point
    or an eligible Contact on the active moving-link subtree.
    """

    result: dict[str, set[str]] = {}
    grasp_required: set[str] = set()
    for variant in realized.get("goal_variants", []):
        goal = variant.get("goal") or {}
        subject = goal.get("subject")
        if not subject:
            continue
        family = str(goal.get("action_family") or "")
        verb = str(goal.get("verb") or "")
        kinds = result.setdefault(str(subject), set())
        if family == "O2O_relation" or verb in {
            "pick",
            "handover",
            "adjust_orientation",
        }:
            grasp_required.add(str(subject))
        else:
            kinds.update(("functional", "grasp_contact"))
    for subject in grasp_required:
        # One shared initial scene must satisfy every variant.  A functional
        # point cannot stand in for the Contact required by a Pick variant.
        result[subject] = {"grasp_contact"}
    return result


def m4_contact_evidence_required_entities(
    realized: Mapping[str, Any],
) -> set[str]:
    """Return grasp subjects whose capability is explicitly covered by M4."""

    result: set[str] = set()
    for variant in realized.get("goal_variants", []):
        goal = variant.get("goal") or {}
        subject = goal.get("subject")
        if not subject:
            continue
        if (
            str(goal.get("action_family") or "") == "O2O_relation"
            or str(goal.get("verb") or "")
            in {"pick", "adjust_orientation"}
        ):
            result.add(str(subject))
    return result


def m4_articulation_evidence_required_entities(
    realized: Mapping[str, Any],
) -> dict[str, set[str]]:
    """Return operated entities and the exact M4 articulation uses required."""

    use_by_state = {
        "open": "open",
        "close": "close",
        "on": "switch_on",
        "off": "switch_off",
        "switch_on": "switch_on",
        "switch_off": "switch_off",
    }
    result: dict[str, set[str]] = {}
    for variant in realized.get("goal_variants", []):
        goal = variant.get("goal") or {}
        subject = goal.get("subject")
        use = use_by_state.get(
            str(goal.get("verb") or goal.get("state") or "")
        )
        if subject and use:
            result.setdefault(str(subject), set()).add(use)
    return result


def m4_touch_evidence_required_entities(
    realized: Mapping[str, Any],
) -> dict[str, set[str]]:
    """Return event-action entities and exact M4 click/press uses required."""

    use_by_verb = {"click": "click_subject", "press": "press_subject"}
    result: dict[str, set[str]] = {}
    for variant in realized.get("goal_variants", []):
        goal = variant.get("goal") or {}
        subject = goal.get("subject")
        use = use_by_verb.get(str(goal.get("verb") or ""))
        if subject and use:
            result.setdefault(str(subject), set()).add(use)
    return result


def preflight_units(
    realized: Mapping[str, Any], units: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Validate Unit alignment and the completed canonical RSF contract."""

    failures: list[dict[str, Any]] = []
    required_subjects = {v["goal"]["subject"] for v in realized.get("goal_variants", [])
                         if v.get("goal", {}).get("action_family") == "O2O_relation"}
    if required_subjects:
        from if_benchmark.layout.manipulation import resources
        from if_benchmark.manipulation_policy import POLICY
        try:
            operation = resources(units)
            if not required_subjects <= set(operation) or any(
                    operation[k].get("policy") != POLICY for k in required_subjects):
                raise ValueError("missing or stale production manipulation resources")
        except (ValueError, KeyError) as exc:
            failures.append({"code": "operation_space_resources_invalid", "message": str(exc)})

    expected = {
        (str(row["type"]), tuple(row["roles"]), tuple(row["entity_keys"]))
        for row in realized.get("unit_inputs", [])
    }
    actual = {
        (str(row["type"]), tuple(row["roles"]), tuple(row["entity_keys"]))
        for row in units
    }
    if actual != expected:
        failures.append(
            {
                "code": "unit_input_alignment",
                "message": "built Units do not exactly match UnitInput membership",
            }
        )
    unit_ids = [str(row.get("id")) for row in units]
    if len(unit_ids) != len(set(unit_ids)):
        failures.append(
            {"code": "duplicate_unit_id", "message": "built Unit ids are not unique"}
        )
    operation_contracts = {
        str(entity): contract
        for unit in units
        for entity, contract in (
            (unit.get("params") or {}).get("operation_anchors") or {}
        ).items()
    }
    m4_contact_required = m4_contact_evidence_required_entities(realized)
    m4_articulation_required = (
        m4_articulation_evidence_required_entities(realized)
    )
    m4_touch_required = m4_touch_evidence_required_entities(realized)
    for entity, kinds in sorted(
        required_operation_anchor_kinds(realized).items()
    ):
        contract = operation_contracts.get(entity)
        if not isinstance(contract, Mapping):
            failures.append(
                {
                    "code": "missing_operation_anchor_evidence",
                    "message": (
                        f"operated entity {entity} has no frozen "
                        f"operation-anchor contract"
                    ),
                }
            )
            continue
        if contract.get("version") != OPERATION_ANCHOR_POLICY_VERSION:
            failures.append(
                {
                    "code": "stale_operation_anchor_evidence",
                    "message": (
                        f"operated entity {entity} uses operation-anchor "
                        f"policy {contract.get('version')!r}"
                    ),
                }
            )
        if contract.get("contract_hash") != artifact_hash(
            contract, omit_keys=("contract_hash",)
        ):
            failures.append(
                {
                    "code": "stale_operation_anchor_evidence",
                    "message": (
                        f"operated entity {entity} has an invalid "
                        "operation-anchor self-hash"
                    ),
                }
            )
        if entity in m4_contact_required and not (
            (contract.get("source") or {}).get(
                "validated_contact_contracts"
            )
        ):
            failures.append(
                {
                    "code": "missing_contact_contract_evidence",
                    "message": (
                        f"operated entity {entity} has no M4 Contact "
                        "contract lineage"
                    ),
                }
            )
        required_articulation_uses = m4_articulation_required.get(
            entity, set()
        )
        articulation_rows = list(
            (contract.get("source") or {}).get(
                "validated_articulation_executions"
            )
            or []
        )
        observed_articulation_uses = {
            str(row.get("use") or "") for row in articulation_rows
        }
        if not required_articulation_uses <= observed_articulation_uses:
            failures.append(
                {
                    "code": "missing_articulation_execution_evidence",
                    "message": (
                        f"operated entity {entity} lacks M4 articulation "
                        "lineage for "
                        f"{sorted(required_articulation_uses - observed_articulation_uses)}"
                    ),
                }
            )
        for use in sorted(required_articulation_uses):
            matches = [
                row
                for row in articulation_rows
                if str(row.get("use") or "") == use
            ]
            if len(matches) != 1 or articulation_capability_errors(
                (matches[0].get("binding") or {}) if matches else {},
                use=use,
            ):
                failures.append(
                    {
                        "code": "missing_articulation_arm_capability",
                        "message": (
                            f"operated entity {entity} lacks exact per-arm "
                            f"M4 capability for {use}"
                        ),
                    }
                )
        required_touch_uses = m4_touch_required.get(entity, set())
        touch_rows = list(
            (contract.get("source") or {}).get("validated_touch_executions")
            or []
        )
        observed_touch_uses = {str(row.get("use") or "") for row in touch_rows}
        if not required_touch_uses <= observed_touch_uses:
            failures.append(
                {
                    "code": "missing_touch_execution_evidence",
                    "message": (
                        f"operated entity {entity} lacks M4 touch lineage for "
                        f"{sorted(required_touch_uses - observed_touch_uses)}"
                    ),
                }
            )
        for use in sorted(required_touch_uses):
            matches = [
                row for row in touch_rows if str(row.get("use") or "") == use
            ]
            if len(matches) != 1 or touch_capability_errors(
                (matches[0].get("binding") or {}) if matches else {}, use=use
            ):
                failures.append(
                    {
                        "code": "missing_touch_arm_capability",
                        "message": (
                            f"operated entity {entity} lacks exact per-arm "
                            f"M4 capability for {use}"
                        ),
                    }
                )
        anchors = [
            row
            for row in contract.get("anchors", [])
            if str(row.get("kind")) in kinds
            and row.get("operation_eligible", True) is True
        ]
        if not anchors:
            failures.append(
                {
                    "code": "missing_operation_anchor_evidence",
                    "message": (
                        f"operated entity {entity} has no eligible anchor "
                        f"of kinds {sorted(kinds)}"
                    ),
                }
            )
    for unit in units:
        params = unit.get("params") or {}
        footprint = params.get("footprint")
        if not isinstance(footprint, dict):
            failures.append(
                {
                    "code": "missing_unit_footprint",
                    "message": f"{unit.get('id')} has no footprint",
                }
            )
        if unit.get("type") == "ReceiverReservationUnit":
            if set(params.get("subject_keys", [])) & set(unit.get("entity_keys", [])):
                failures.append(
                    {
                        "code": "receiver_contains_subject",
                        "message": f"{unit.get('id')} contains its subject",
                    }
                )
        unit_type = unit.get("type")
        if unit_type == "ObjectUnit":
            support_rows = [
                (
                    str(params.get("object_key")),
                    params.get("local_pose"),
                    params.get("local_z_bounds"),
                    (params.get("anchors") or {}).get("support"),
                )
            ]
        elif unit_type == "LandmarkRelationUnit":
            support_rows = [
                (
                    str(entity),
                    pose,
                    (params.get("member_z_bounds") or {}).get(entity),
                    (params.get("anchors") or {}).get(f"{entity}:support"),
                )
                for entity, pose in (params.get("internal_layout") or {}).items()
            ]
        elif unit_type == "ReceiverReservationUnit":
            support_rows = [
                (
                    str(entity),
                    pose,
                    (params.get("receiver_z_bounds") or {}).get(entity),
                    (params.get("anchors") or {}).get(f"{entity}:support"),
                )
                for entity, pose in (params.get("receiver_layout") or {}).items()
            ]
        else:
            support_rows = []
        for entity, pose, z_bounds, support in support_rows:
            if not isinstance(pose, dict) or not isinstance(z_bounds, list):
                continue
            translation = pose.get("translation_xyz") or []
            if len(translation) != 3 or len(z_bounds) != 2:
                continue
            expected_support = _support_plane_position(pose, z_bounds)
            if support != expected_support:
                failures.append(
                    {
                        "code": "invalid_support_anchor",
                        "message": (
                            f"{unit.get('id')} support anchor for {entity} "
                            "does not match its local pose and Z lower bound"
                        ),
                    }
                )
            if (
                pose.get("placement_mode") == "table_resting"
                and expected_support[2] != 0.0
            ):
                failures.append(
                    {
                        "code": "invalid_table_support",
                        "message": (
                            f"{unit.get('id')} table-resting member {entity} "
                            "does not rest on local z=0"
                        ),
                    }
                )
    completed = deepcopy(dict(realized))
    completed["units"] = canonical_data(list(units))
    schema = validate_document(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "realized_scene_family",
            "realized_scene_family": canonical_data(completed),
        }
    )
    failures.extend(
        {
            "code": "unit_schema_invalid",
            "message": f"{row['path']}: {row['message']}",
        }
        for row in schema["errors"]
    )
    return {
        "status": "passed" if not failures else "failed",
        "unit_count": len(units),
        "failures": failures,
    }


def _artifact(
    realized_artifact: Mapping[str, Any],
    units: Sequence[Mapping[str, Any]],
    preflight: Mapping[str, Any],
    *,
    shared_source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    realized = realized_artifact["realized_scene_family"]
    pose_source_hashes = _canonical_pose_report_hashes(units)
    operation_source_hashes = _operation_source_hashes(units)
    source_hashes = {
        **dict(shared_source_hashes),
        **pose_source_hashes,
        **operation_source_hashes,
        f"{DEFAULT_REALIZATION_DIR}/{realized_artifact['realized_family_id']}.yml": str(
            realized_artifact["artifact_hash"]
        ),
    }
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "unit_build",
        "unit_build_version": UNIT_BUILD_VERSION,
        "realized_family_id": realized["id"],
        "abstract_family_id": realized["abstract_family_id"],
        "assignment_id": realized["assignment_id"],
        "source_realization_hash": realized_artifact["artifact_hash"],
        "source_hashes": dict(sorted(source_hashes.items())),
        "config_hash": artifact_hash(CONFIG),
        "units": canonical_data(list(units)),
        "preflight": canonical_data(preflight),
    }
    result["artifact_hash"] = artifact_hash(result)
    return result


def _canonical_pose_report_hashes(
    units: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    hashes: dict[str, str] = {}

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            report_path = value.get("report_path")
            report_hash = value.get("report_hash")
            if report_path is not None or report_hash is not None:
                if not report_path or not report_hash:
                    raise _error(
                        "contradictory_canonical_pose_evidence",
                        "M4 report source requires both path and hash",
                    )
                prior = hashes.get(str(report_path))
                if prior is not None and prior != str(report_hash):
                    raise _error(
                        "contradictory_canonical_pose_evidence",
                        f"conflicting M4 report hashes for {report_path}",
                    )
                hashes[str(report_path)] = str(report_hash)
            report_hashes = value.get("report_hashes")
            if isinstance(report_hashes, Mapping):
                for path, digest in report_hashes.items():
                    prior = hashes.get(str(path))
                    if prior is not None and prior != str(digest):
                        raise _error(
                            "contradictory_canonical_pose_evidence",
                            f"conflicting M4 report hashes for {path}",
                        )
                    hashes[str(path)] = str(digest)
            for child in value.values():
                visit(child)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for child in value:
                visit(child)

    visit(list(units))
    return dict(sorted(hashes.items()))


def _operation_source_hashes(
    units: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    """Collect exact live metadata/URDF digests bound into operation anchors."""

    hashes: dict[str, str] = {}

    def add(path: Any, digest: Any, *, label: str) -> None:
        if path is None and digest is None:
            return
        if not path or not digest:
            raise _error(
                "contradictory_operation_anchor_evidence",
                f"operation-anchor {label} source requires both path and sha256",
            )
        normalized_path = str(path)
        normalized_digest = str(digest)
        prior = hashes.get(normalized_path)
        if prior is not None and prior != normalized_digest:
            raise _error(
                "contradictory_operation_anchor_evidence",
                f"conflicting operation-anchor hashes for {normalized_path}",
            )
        hashes[normalized_path] = normalized_digest

    # Only the operation-anchor ``source`` mapping owns live file provenance.
    # Evidence bindings nested below that source legitimately carry digest-only
    # identities (for example an articulation profile's ``urdf_sha256``); they
    # are not file-source records and must not be mistaken for a missing path.
    for unit in units:
        for path, digest in (unit.get("params") or {}).get("operation_sources", {}).items():
            add(path, digest, label="operation-space")
        operation_anchors = (
            (unit.get("params") or {}).get("operation_anchors") or {}
        )
        if not isinstance(operation_anchors, Mapping):
            continue
        for contract in operation_anchors.values():
            if not isinstance(contract, Mapping):
                continue
            source = contract.get("source")
            if not isinstance(source, Mapping):
                continue
            add(
                source.get("metadata_path"),
                source.get("metadata_sha256"),
                label="metadata",
            )
            add(
                source.get("urdf_path"),
                source.get("urdf_sha256"),
                label="URDF",
            )
    return dict(sorted(hashes.items()))


@lru_cache(maxsize=32)
def _current_m4_report_hash_cached(
    path: str,
    stat_identity: tuple[int, int, int, int, int],
) -> str | None:
    try:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(report, dict):
        return None
    report_hash = str(report.get("report_hash") or "")
    return (
        report_hash
        if report_hash
        and report_hash
        == artifact_hash(report, omit_keys=("report_hash",))
        else None
    )


def _current_m4_report_hash(path: str) -> str | None:
    try:
        stat = Path(path).stat()
    except OSError:
        return None
    identity = (
        int(stat.st_dev),
        int(stat.st_ino),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    )
    return _current_m4_report_hash_cached(str(path), identity)


def _prewarm_current_m4_release_report_hash(workspace_root: Path) -> None:
    """Verify the sole current M4 report before M11 check workers fork."""

    release_dir = (
        workspace_root
        / "if_benchmark/data/pipeline_runs/release/"
        "m04_asset_validation/reports"
    ).resolve()
    if not release_dir.is_dir():
        return
    reports = sorted(release_dir.glob("canonical_skill_*.json"))
    if len(reports) != 1:
        return
    if _current_m4_report_hash(str(reports[0])) is None:
        raise _error(
            "invalid_m4_release_report",
            f"invalid unified M4 release report: {reports[0]}",
        )


def _current_source_file_hash(path: str) -> str | None:
    source_path = Path(path)
    if not source_path.is_absolute() and not source_path.is_file():
        source_path = Path(__file__).resolve().parents[2] / source_path
    try:
        return hashlib.sha256(source_path.read_bytes()).hexdigest()
    except OSError:
        return None


def _self_hash_valid(artifact: Mapping[str, Any]) -> bool:
    return artifact.get("artifact_hash") == artifact_hash(
        artifact, omit_keys=("artifact_hash",)
    )


def _report_hash_valid(report: Mapping[str, Any]) -> bool:
    return report.get("report_hash") == artifact_hash(
        report, omit_keys=("report_hash",)
    )


def _shared_inputs(
    asset_catalog_path: Path,
    geometry_path: Path,
    terminal_pairs_path: Path,
    skill_pairs_path: Path,
    receiver_profiles_path: Path = DEFAULT_RECEIVER_PROFILES_PATH,
    *,
    load_pair_records: bool = True,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, str],
]:
    asset_catalog = _load_yaml(asset_catalog_path)
    geometry = _load_yaml(geometry_path)

    def projection_hash(path: Path, *, stage: str, artifact_name: str) -> str:
        document = _load_yaml(path)
        schema_version = int(document.get("schema_version") or 0)
        if (
            str(document.get("stage") or "") != stage
            or str(document.get("artifact") or "") != artifact_name
            or not str(document.get("source_ledger_hash") or "")
        ):
            raise ValueError(f"invalid M5 projection envelope: {path}")
        declared = str(document.get("projection_hash") or "")
        observed = artifact_hash(
            document, omit_keys=("projection_hash",)
        )
        if not declared or declared != observed:
            raise ValueError(f"M5 projection content hash mismatch: {path}")
        if schema_version == 2:
            allowed = LEGACY_M5_REPORT_V3_PROJECTIONS.get(stage)
            if (
                not allowed
                or _file_sha256(path) != str(allowed["sha256"])
                or str(document.get("source_ledger_hash") or "")
                != str(allowed["source_ledger_hash"])
            ):
                raise ValueError(f"unsupported legacy M5 projection: {path}")
        elif schema_version != 3:
            raise ValueError(f"unsupported M5 projection schema: {path}")
        return declared

    hashes = {
        str(asset_catalog_path): artifact_hash(asset_catalog),
        str(geometry_path): artifact_hash(geometry),
        str(terminal_pairs_path): projection_hash(
            terminal_pairs_path,
            stage="terminal_settle",
            artifact_name="validated_terminal_pairs.yml",
        ),
        str(skill_pairs_path): projection_hash(
            skill_pairs_path,
            stage="canonical_skill",
            artifact_name="validated_skill_pairs.yml",
        ),
        str(receiver_profiles_path): _file_sha256(receiver_profiles_path),
    }
    terminal_pairs = _load_yaml(terminal_pairs_path) if load_pair_records else {}
    skill_pairs = _load_yaml(skill_pairs_path) if load_pair_records else {}
    receiver_profiles = (
        _load_yaml(receiver_profiles_path) if load_pair_records else {}
    )
    return (
        asset_catalog,
        geometry,
        terminal_pairs,
        skill_pairs,
        receiver_profiles,
        hashes,
    )


def _reusable(
    artifact: Mapping[str, Any],
    *,
    realized_artifact: Mapping[str, Any],
    shared_source_hashes: Mapping[str, str],
) -> bool:
    realized_id = str(realized_artifact["realized_family_id"])
    expected_sources = {
        **dict(shared_source_hashes),
        f"{DEFAULT_REALIZATION_DIR}/{realized_id}.yml": str(
            realized_artifact["artifact_hash"]
        ),
    }
    actual_sources = dict(artifact.get("source_hashes") or {})
    try:
        pose_sources = _canonical_pose_report_hashes(
            artifact.get("units") or []
        )
        operation_sources = _operation_source_hashes(
            artifact.get("units") or []
        )
    except UnitBuildError:
        # A pre-report_path Unit v6 artifact is stale migration input, not a
        # fatal freeze error.  Rebuild it so the live M4 file join is added.
        return False
    sources_valid = bool(
        all(actual_sources.get(path) == digest for path, digest in expected_sources.items())
        and {
            path: digest
            for path, digest in actual_sources.items()
            if path not in expected_sources
        }
        == {**pose_sources, **operation_sources}
        and all(
            _current_m4_report_hash(path) == digest
            for path, digest in pose_sources.items()
        )
        and all(
            _current_source_file_hash(path) == digest
            for path, digest in operation_sources.items()
        )
    )
    return (
        artifact.get("schema_version") == SCHEMA_VERSION
        and artifact.get("artifact_kind") == "unit_build"
        and artifact.get("unit_build_version") == UNIT_BUILD_VERSION
        and artifact.get("realized_family_id") == realized_id
        and artifact.get("source_realization_hash")
        == realized_artifact.get("artifact_hash")
        and sources_valid
        and artifact.get("config_hash") == artifact_hash(CONFIG)
        and artifact.get("preflight", {}).get("status") == "passed"
        and _self_hash_valid(artifact)
    )


def _reports(
    rows: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
    *,
    source_hashes: Mapping[str, str],
    expected_family_count: int,
    provider: FootprintProvider,
) -> tuple[dict[str, Any], dict[str, Any]]:
    type_counts = Counter(
        str(unit_type)
        for row in rows
        for unit_type, count in row.get("unit_type_counts", {}).items()
        for _ in range(int(count))
    )
    relation_counts = Counter(
        str(relation)
        for row in rows
        for relation, count in row.get("relation_counts", {}).items()
        for _ in range(int(count))
    )
    unit_footprint_source_counts = Counter(
        str(source)
        for row in rows
        for source, count in row.get(
            "unit_footprint_source_counts", {}
        ).items()
        for _ in range(int(count))
    )
    geometry_footprint_source_counts = Counter(
        str(source)
        for row in rows
        for source, count in row.get(
            "geometry_footprint_source_counts", {}
        ).items()
        for _ in range(int(count))
    )
    geometry_ref_counts = Counter(
        str(geometry_ref)
        for row in rows
        for geometry_ref, count in row.get("geometry_ref_counts", {}).items()
        for _ in range(int(count))
    )
    summary = {
        "expected_realized_family_count": expected_family_count,
        "realized_family_count": len(rows),
        "unit_count": sum(int(row["unit_count"]) for row in rows),
        "preflight_pass_count": sum(
            row.get("preflight_status") == "passed" for row in rows
        ),
        "failure_count": len(failures),
    }
    acceptance_ready = (
        not failures
        and summary["realized_family_count"] == expected_family_count
        and summary["preflight_pass_count"] == expected_family_count
    )
    artifact_rows = [
        {"id": row["realized_family_id"], "hash": row["artifact_hash"]}
        for row in sorted(rows, key=lambda value: value["realized_family_id"])
    ]
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "unit_build_report",
        "unit_build_version": UNIT_BUILD_VERSION,
        "config_hash": artifact_hash(CONFIG),
        "source_hashes": dict(source_hashes),
        "summary": summary,
        "unit_type_counts": dict(sorted(type_counts.items())),
        "relation_counts": dict(sorted(relation_counts.items())),
        "unit_footprint_source_counts": dict(
            sorted(unit_footprint_source_counts.items())
        ),
        "geometry_footprint_source_counts": dict(
            sorted(geometry_footprint_source_counts.items())
        ),
        "footprint_cache": {
            "hits": sum(geometry_ref_counts.values()) - len(geometry_ref_counts),
            "misses": len(geometry_ref_counts),
            "entry_count": len(geometry_ref_counts),
            "accounting": "frozen_geometry_lookup_provenance",
        },
        "artifact_set_hash": artifact_hash(artifact_rows),
        "acceptance_ready": acceptance_ready,
    }
    report["report_hash"] = artifact_hash(report)
    unsat: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "unit_unsat_report",
        "unit_build_version": UNIT_BUILD_VERSION,
        "config_hash": artifact_hash(CONFIG),
        "source_hashes": dict(source_hashes),
        "failure_count": len(failures),
        "failures": canonical_data(list(failures)),
        "acceptance_ready": acceptance_ready,
    }
    unsat["report_hash"] = artifact_hash(unsat)
    return report, unsat


def _row_from_artifact(artifact: Mapping[str, Any]) -> dict[str, Any]:
    units = artifact.get("units", [])
    geometry_sources: list[str] = []
    geometry_refs: list[str] = []
    for unit in units:
        params = unit.get("params") or {}
        for evidence_key in ("geometry_evidence", "subject_geometry_evidence"):
            evidence = params.get(evidence_key)
            if not isinstance(evidence, dict):
                continue
            if "source" in evidence:
                geometry_sources.append(str(evidence["source"]))
                if evidence.get("geometry_ref"):
                    geometry_refs.append(str(evidence["geometry_ref"]))
                continue
            for row in evidence.values():
                if not isinstance(row, dict):
                    continue
                if row.get("source"):
                    geometry_sources.append(str(row["source"]))
                if row.get("geometry_ref"):
                    geometry_refs.append(str(row["geometry_ref"]))
    return {
        "realized_family_id": artifact["realized_family_id"],
        "artifact_hash": artifact["artifact_hash"],
        "unit_count": len(units),
        "unit_type_counts": dict(
            Counter(str(row["type"]) for row in units)
        ),
        "relation_counts": dict(
            Counter(
                str(relation)
                for row in units
                if row.get("type") == "ReceiverReservationUnit"
                for relation in (row.get("params") or {}).get("relations", [])
            )
        ),
        "unit_footprint_source_counts": dict(
            Counter(
                str((row.get("params") or {}).get("footprint", {}).get("source"))
                for row in units
            )
        ),
        "geometry_footprint_source_counts": dict(
            Counter(geometry_sources)
        ),
        "geometry_ref_counts": dict(Counter(geometry_refs)),
        "preflight_status": artifact.get("preflight", {}).get("status"),
    }


_UNIT_WORKER_PROVIDER: FootprintProvider | None = None
_UNIT_WORKER_SHARED_HASHES: Mapping[str, str] = {}


def _init_unit_worker(
    provider: FootprintProvider | None,
    shared_hashes: Mapping[str, str],
) -> None:
    global _UNIT_WORKER_PROVIDER, _UNIT_WORKER_SHARED_HASHES
    _UNIT_WORKER_PROVIDER = provider
    _UNIT_WORKER_SHARED_HASHES = shared_hashes


def _validated_worker_count(workers: int) -> int:
    worker_count = int(workers)
    if worker_count < 1:
        raise ValueError("workers must be a positive integer")
    return worker_count


def _run_source_workers(
    function: Callable[[tuple[str, str]], dict[str, Any]],
    tasks: Sequence[tuple[str, str]],
    *,
    workers: int,
    provider: FootprintProvider | None,
    shared_hashes: Mapping[str, str],
) -> Iterable[dict[str, Any]]:
    """Run independent source files while preserving source-order results."""

    worker_count = min(_validated_worker_count(workers), max(1, len(tasks)))
    if worker_count == 1:
        _init_unit_worker(provider, shared_hashes)
        for task in tasks:
            yield function(task)
        return

    try:
        process_context = multiprocessing.get_context("fork")
    except ValueError:  # pragma: no cover - non-POSIX compatibility fallback
        process_context = multiprocessing.get_context()
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=process_context,
        initializer=_init_unit_worker,
        initargs=(provider, shared_hashes),
    ) as executor:
        # Keep full-corpus M11 runs bounded.  Python 3.10's executor.map
        # eagerly creates all work items, which can fill multiprocessing pipes
        # before the parent begins consuming completed results.
        task_iterator = iter(tasks)
        pending = deque()
        for _ in range(worker_count * 2):
            try:
                task = next(task_iterator)
            except StopIteration:
                break
            pending.append(executor.submit(function, task))

        while pending:
            future = pending.popleft()
            yield future.result()
            try:
                task = next(task_iterator)
            except StopIteration:
                continue
            pending.append(executor.submit(function, task))


def _freeze_unit_source(task: tuple[str, str]) -> dict[str, Any]:
    source_path = Path(task[0])
    output_dir = Path(task[1])
    source = _load_yaml(source_path)
    realized_id = str(source["realized_family_id"])
    output_path = output_dir / f"{realized_id}.yml"
    existing = _try_load_yaml(output_path)
    if existing is not None and _reusable(
        existing,
        realized_artifact=source,
        shared_source_hashes=_UNIT_WORKER_SHARED_HASHES,
    ):
        return {
            "realized_id": realized_id,
            "row": _row_from_artifact(existing),
            "failure": None,
        }
    if _UNIT_WORKER_PROVIDER is None:
        raise RuntimeError("M11 freeze worker was not initialized")
    try:
        realized = source["realized_scene_family"]
        units = build_units(realized, _UNIT_WORKER_PROVIDER)
        preflight = preflight_units(realized, units)
        if preflight["status"] != "passed":
            first_failure = preflight["failures"][0]
            raise _error(
                str(first_failure["code"]),
                "; ".join(
                    f"{row['code']}: {row['message']}"
                    for row in preflight["failures"][:10]
                ),
                realized_family_id=realized_id,
            )
        artifact = _artifact(
            source,
            units,
            preflight,
            shared_source_hashes=_UNIT_WORKER_SHARED_HASHES,
        )
        _write_yaml(output_path, artifact)
        return {
            "realized_id": realized_id,
            "row": _row_from_artifact(artifact),
            "failure": None,
        }
    except UnitBuildError as exc:
        failure = exc.to_record()
        failure.setdefault("realized_family_id", realized_id)
        return {
            "realized_id": realized_id,
            "row": None,
            "failure": failure,
        }


def _check_unit_source(task: tuple[str, str]) -> dict[str, Any]:
    source_path = Path(task[0])
    output_dir = Path(task[1])
    source = _load_yaml(source_path)
    realized_id = str(source["realized_family_id"])
    artifact_path = output_dir / f"{realized_id}.yml"
    artifact = _try_load_yaml(artifact_path)
    reusable = artifact is not None and _reusable(
        artifact,
        realized_artifact=source,
        shared_source_hashes=_UNIT_WORKER_SHARED_HASHES,
    )
    return {
        "realized_id": realized_id,
        "row": _row_from_artifact(artifact) if reusable else None,
        "stale_path": None if reusable else str(artifact_path),
    }


def _validate_unit_source(task: tuple[str, str]) -> dict[str, Any]:
    source_path = Path(task[0])
    output_dir = Path(task[1])
    source = _load_yaml(source_path)
    realized_id = str(source["realized_family_id"])
    artifact_path = output_dir / f"{realized_id}.yml"
    actual = _try_load_yaml(artifact_path)
    if _UNIT_WORKER_PROVIDER is None:
        raise RuntimeError("M11 validate worker was not initialized")
    try:
        realized = source["realized_scene_family"]
        units = build_units(realized, _UNIT_WORKER_PROVIDER)
        preflight = preflight_units(realized, units)
        expected = _artifact(
            source,
            units,
            preflight,
            shared_source_hashes=_UNIT_WORKER_SHARED_HASHES,
        )
        error = None if actual == expected else f"{artifact_path}: content mismatch"
    except UnitBuildError as exc:
        error = str(exc)
    return {"realized_id": realized_id, "error": error}


def freeze_units(
    *,
    realization_dir: Path = DEFAULT_REALIZATION_DIR,
    asset_catalog_path: Path = DEFAULT_ASSET_CATALOG_PATH,
    geometry_path: Path = DEFAULT_GEOMETRY_PATH,
    terminal_pairs_path: Path = DEFAULT_TERMINAL_PAIRS_PATH,
    skill_pairs_path: Path = DEFAULT_SKILL_PAIRS_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    workers: int = 1,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Freeze all M11 Unit artifacts and aggregate reports."""

    (
        asset_catalog,
        geometry,
        terminal_pairs,
        skill_pairs,
        receiver_profiles,
        shared_hashes,
    ) = _shared_inputs(
        asset_catalog_path, geometry_path, terminal_pairs_path, skill_pairs_path
    )
    provider = FootprintProvider(
        geometry,
        asset_catalog=asset_catalog,
        terminal_pairs=terminal_pairs,
        skill_pairs=skill_pairs,
        receiver_profiles=receiver_profiles,
        terminal_pair_projection_sha256=_file_sha256(terminal_pairs_path),
        skill_pair_projection_sha256=_file_sha256(skill_pairs_path),
    )
    if realization_dir.resolve() == Path(
        "if_benchmark/data/pipeline_runs/release/m10_realized_families"
    ).resolve():
        provider.prewarm_m4_release_report()
    source_paths = sorted(realization_dir.glob("rsf_*.yml"))
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    expected_names = {path.name for path in source_paths}
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(str(path), str(output_dir)) for path in source_paths]
    results = _run_source_workers(
        _freeze_unit_source,
        tasks,
        workers=workers,
        provider=provider,
        shared_hashes=shared_hashes,
    )
    for index, result in enumerate(results, start=1):
        realized_id = str(result["realized_id"])
        if result["failure"] is not None:
            failures.append(result["failure"])
        else:
            rows.append(result["row"])
        if progress is not None:
            progress(index, len(source_paths), realized_id)
    for extra in sorted(output_dir.glob("rsf_*.yml")):
        if extra.name not in expected_names:
            extra.unlink()
    report, unsat = _reports(
        rows,
        failures,
        source_hashes=shared_hashes,
        expected_family_count=len(source_paths),
        provider=provider,
    )
    _write_yaml(output_dir / REPORT_FILE, report)
    _write_yaml(output_dir / UNSAT_FILE, unsat)
    return report


def check_units(
    *,
    realization_dir: Path = DEFAULT_REALIZATION_DIR,
    asset_catalog_path: Path = DEFAULT_ASSET_CATALOG_PATH,
    geometry_path: Path = DEFAULT_GEOMETRY_PATH,
    terminal_pairs_path: Path = DEFAULT_TERMINAL_PAIRS_PATH,
    skill_pairs_path: Path = DEFAULT_SKILL_PAIRS_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    workers: int = 1,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Check M11 artifacts without rebuilding Unit geometry."""

    if realization_dir.resolve() == Path(
        "if_benchmark/data/pipeline_runs/release/m10_realized_families"
    ).resolve():
        _prewarm_current_m4_release_report_hash(
            Path(__file__).resolve().parents[2]
        )
    _, _, _, _, _, shared_hashes = _shared_inputs(
        asset_catalog_path,
        geometry_path,
        terminal_pairs_path,
        skill_pairs_path,
        load_pair_records=False,
    )
    source_paths = sorted(realization_dir.glob("rsf_*.yml"))
    stale_paths: list[str] = []
    current_rows: list[dict[str, Any]] = []
    expected_names = {path.name for path in source_paths}
    tasks = [(str(path), str(output_dir)) for path in source_paths]
    results = _run_source_workers(
        _check_unit_source,
        tasks,
        workers=workers,
        provider=None,
        shared_hashes=shared_hashes,
    )
    for index, result in enumerate(results, start=1):
        realized_id = str(result["realized_id"])
        if result["stale_path"] is not None:
            stale_paths.append(result["stale_path"])
        else:
            current_rows.append(result["row"])
        if progress is not None:
            progress(index, len(source_paths), realized_id)
    for extra in sorted(output_dir.glob("rsf_*.yml")):
        if extra.name not in expected_names:
            stale_paths.append(str(extra))
    report = _try_load_yaml(output_dir / REPORT_FILE)
    unsat = _try_load_yaml(output_dir / UNSAT_FILE)
    expected_artifact_set_hash = artifact_hash(
        [
            {"id": row["realized_family_id"], "hash": row["artifact_hash"]}
            for row in sorted(
                current_rows, key=lambda value: value["realized_family_id"]
            )
        ]
    )
    expected_unit_count = sum(int(row["unit_count"]) for row in current_rows)
    expected_type_counts = dict(
        sorted(
            Counter(
                unit_type
                for row in current_rows
                for unit_type, count in row["unit_type_counts"].items()
                for _ in range(int(count))
            ).items()
        )
    )
    expected_relation_counts = dict(
        sorted(
            Counter(
                relation
                for row in current_rows
                for relation, count in row["relation_counts"].items()
                for _ in range(int(count))
            ).items()
        )
    )
    expected_unit_footprint_sources = dict(
        sorted(
            Counter(
                source
                for row in current_rows
                for source, count in row[
                    "unit_footprint_source_counts"
                ].items()
                for _ in range(int(count))
            ).items()
        )
    )
    expected_geometry_footprint_sources = dict(
        sorted(
            Counter(
                source
                for row in current_rows
                for source, count in row[
                    "geometry_footprint_source_counts"
                ].items()
                for _ in range(int(count))
            ).items()
        )
    )
    expected_geometry_refs = Counter(
        geometry_ref
        for row in current_rows
        for geometry_ref, count in row["geometry_ref_counts"].items()
        for _ in range(int(count))
    )
    expected_footprint_cache = {
        "hits": sum(expected_geometry_refs.values()) - len(expected_geometry_refs),
        "misses": len(expected_geometry_refs),
        "entry_count": len(expected_geometry_refs),
        "accounting": "frozen_geometry_lookup_provenance",
    }
    if (
        report is None
        or report.get("artifact_kind") != "unit_build_report"
        or report.get("unit_build_version") != UNIT_BUILD_VERSION
        or report.get("config_hash") != artifact_hash(CONFIG)
        or report.get("source_hashes") != shared_hashes
        or not report.get("acceptance_ready")
        or not _report_hash_valid(report)
        or report.get("artifact_set_hash") != expected_artifact_set_hash
        or report.get("unit_type_counts") != expected_type_counts
        or report.get("relation_counts") != expected_relation_counts
        or report.get("unit_footprint_source_counts")
        != expected_unit_footprint_sources
        or report.get("geometry_footprint_source_counts")
        != expected_geometry_footprint_sources
        or report.get("footprint_cache") != expected_footprint_cache
        or report.get("summary", {}).get("expected_realized_family_count")
        != len(source_paths)
        or report.get("summary", {}).get("realized_family_count")
        != len(current_rows)
        or report.get("summary", {}).get("unit_count")
        != expected_unit_count
        or report.get("summary", {}).get("preflight_pass_count")
        != len(current_rows)
        or report.get("summary", {}).get("failure_count") != 0
    ):
        stale_paths.append(str(output_dir / REPORT_FILE))
    if (
        unsat is None
        or unsat.get("artifact_kind") != "unit_unsat_report"
        or unsat.get("unit_build_version") != UNIT_BUILD_VERSION
        or unsat.get("config_hash") != artifact_hash(CONFIG)
        or unsat.get("source_hashes") != shared_hashes
        or unsat.get("failure_count") != 0
        or unsat.get("failures") != []
        or not unsat.get("acceptance_ready")
        or not _report_hash_valid(unsat)
    ):
        stale_paths.append(str(output_dir / UNSAT_FILE))
    return {
        "stale": bool(stale_paths),
        "stale_paths": sorted(set(stale_paths)),
        "realized_family_count": len(source_paths),
    }


def validate_units(
    *,
    realization_dir: Path = DEFAULT_REALIZATION_DIR,
    asset_catalog_path: Path = DEFAULT_ASSET_CATALOG_PATH,
    geometry_path: Path = DEFAULT_GEOMETRY_PATH,
    terminal_pairs_path: Path = DEFAULT_TERMINAL_PAIRS_PATH,
    skill_pairs_path: Path = DEFAULT_SKILL_PAIRS_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    workers: int = 1,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Rebuild every M11 payload in memory and compare canonical hashes."""

    (
        asset_catalog,
        geometry,
        terminal_pairs,
        skill_pairs,
        receiver_profiles,
        shared_hashes,
    ) = _shared_inputs(
        asset_catalog_path, geometry_path, terminal_pairs_path, skill_pairs_path
    )
    provider = FootprintProvider(
        geometry,
        asset_catalog=asset_catalog,
        terminal_pairs=terminal_pairs,
        skill_pairs=skill_pairs,
        receiver_profiles=receiver_profiles,
        terminal_pair_projection_sha256=_file_sha256(terminal_pairs_path),
        skill_pair_projection_sha256=_file_sha256(skill_pairs_path),
    )
    if realization_dir.resolve() == Path(
        "if_benchmark/data/pipeline_runs/release/m10_realized_families"
    ).resolve():
        provider.prewarm_m4_release_report()
    source_paths = sorted(realization_dir.glob("rsf_*.yml"))
    errors: list[str] = []
    tasks = [(str(path), str(output_dir)) for path in source_paths]
    results = _run_source_workers(
        _validate_unit_source,
        tasks,
        workers=workers,
        provider=provider,
        shared_hashes=shared_hashes,
    )
    for index, result in enumerate(results, start=1):
        realized_id = str(result["realized_id"])
        if result["error"] is not None:
            errors.append(result["error"])
        if progress is not None:
            progress(index, len(source_paths), realized_id)
    lifecycle = check_units(
        realization_dir=realization_dir,
        asset_catalog_path=asset_catalog_path,
        geometry_path=geometry_path,
        terminal_pairs_path=terminal_pairs_path,
        skill_pairs_path=skill_pairs_path,
        output_dir=output_dir,
        workers=workers,
    )
    errors.extend(
        f"{path}: stale lifecycle artifact"
        for path in lifecycle["stale_paths"]
    )
    return {
        "valid": not errors,
        "errors": errors,
        "realized_family_count": len(source_paths),
    }
