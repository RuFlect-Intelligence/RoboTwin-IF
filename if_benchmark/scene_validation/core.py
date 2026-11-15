"""Milestone 15 simulator-independent scene validation.

This module owns the deterministic contracts used by both the real simulator
worker and unit-test/mock runtimes.  In particular, source joins, descriptor
geometry, checker evaluation, rollout classification, aggregation, and
publication decisions do not import a simulator.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from copy import deepcopy
from functools import lru_cache, partial
import hashlib
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence

import yaml

from if_benchmark.articulation_capability import (
    ARTICULATION_CAPABILITY_VERSION,
    validated_articulation_arm_capability,
)
from if_benchmark.grasp_validation.planning_contract import (
    validated_scene_pick_planning_contract,
)
from if_benchmark.operation_anchors import OPERATION_ANCHOR_POLICY_VERSION
from if_benchmark.layout.core import LAYOUT_SOLVER_VERSION
from if_benchmark.realization.core import REALIZATION_VERSION
from if_benchmark.reach_validation.contract import (
    validate_route_witness_report,
)
from if_benchmark.schema import artifact_hash, build_provenance, deterministic_id
from if_benchmark.schema.cli import validate_document
from if_benchmark.schema.serialization import canonical_data
from if_benchmark.scene_integrity import (
    build_scene_integrity_binding,
    evaluate_scene_integrity,
)
from if_benchmark.scene_validation.skill_policy import (
    STABILITY_COMPONENT_POLICY_VERSION,
)
from if_benchmark.success_contract import (
    RelationContract,
    SuccessContract,
    evaluate_m15_success,
)
from if_benchmark.tasks.core import CHECKER_CONFIG_VERSION, TASK_RENDER_VERSION
from if_benchmark.touch_capability import (
    TOUCH_CAPABILITY_VERSION,
    TOUCH_EXECUTABLE_USES,
    validated_touch_arm_capability,
)
from if_benchmark.units.core import UNIT_BUILD_VERSION


SCHEMA_VERSION = 1
SCENE_VALIDATION_VERSION = 9
PLAN_VERSION = 9
PUBLICATION_POLICY = "validated_task_subset_with_audited_exclusions_v1"

DEFAULT_REALIZATION_DIR = Path("if_benchmark/data/realized_families")
DEFAULT_UNIT_DIR = Path("if_benchmark/data/units")
DEFAULT_SCENE_SPEC_DIR = Path("if_benchmark/data/scene_specs")
DEFAULT_TASK_DIR = Path("if_benchmark/data/tasks")
DEFAULT_ROUTE_DIR = Path("if_benchmark/data/reach_validation/route_reports")
DEFAULT_FAMILY_LIBRARY = Path("if_benchmark/data/families/abstract_families.yml")
DEFAULT_CONFIG_PATH = Path(
    "if_benchmark/data/scene_validation/scene_validation_config.yml"
)
DEFAULT_OUTPUT_DIR = Path("if_benchmark/data/scene_validation")

PLAN_FILE = "validation_plan.yml"
DESCRIPTOR_REPORT_FILE = "descriptor_gate_report.yml"
T0_REPORT_FILE = "not_success_at_t0_report.yml"
ROLLOUT_REPORT_FILE = "sim_rollout_report.yml"
FAILURE_REPORT_FILE = "failure_attribution_report.yml"
REPORT_SEQUENCE = (
    DESCRIPTOR_REPORT_FILE,
    T0_REPORT_FILE,
    ROLLOUT_REPORT_FILE,
    FAILURE_REPORT_FILE,
)
REPORT_FILES = frozenset(REPORT_SEQUENCE)

CHECKER_OPERATORS = frozenset(
    {
        "articulation_state_success",
        "click_press_success",
        "pick_success",
        "placement_completed",
        "orientation_success",
    }
)
CHECKER_GUARDS = frozenset({"scene_integrity"})
SUCCESS_CONTRACT = SuccessContract.load()
RELATION_CONTRACT = RelationContract.from_success_contract(SUCCESS_CONTRACT)
SKILL_OPERATORS = frozenset(
    {
        "actuate_state",
        "adjust_orientation",
        "carry_to_relation",
        "grasp",
        "release",
        "retreat_relative",
        "settle_stack",
        "touch_event",
    }
)
TOUCH_EVENT_BY_USE = {
    "click_subject": "click",
    "press_subject": "press",
}
DESCRIPTOR_SUBTYPES = frozenset(
    {
        "color",
        "landmark",
        "material",
        "position",
        "robot",
        "shape",
        "size",
        "spatial",
        "type",
    }
)
ROUTES = frozenset({"direct"})
M13_EXCLUSION_REPORTS = (
    ("m13_instruction", "instruction_report.yml", "instruction_report"),
    (
        "m13_checker_alignment",
        "checker_alignment_report.yml",
        "checker_alignment_report",
    ),
)
_M13_EXCLUSION_CACHE: dict[
    str,
    tuple[
        tuple[tuple[str, int, int], ...],
        set[tuple[str, str]],
        dict[str, dict[str, str]],
    ],
] = {}


@dataclass
class SceneValidationError(ValueError):
    """Stable source, infrastructure, or semantic validation failure."""

    code: str
    message: str
    stage: str = "source"
    path: str | None = None
    scene_spec_id: str | None = None
    task_record_id: str | None = None

    def __str__(self) -> str:
        context = [
            f"{key}={value}"
            for key, value in (
                ("scene_spec_id", self.scene_spec_id),
                ("task_record_id", self.task_record_id),
                ("path", self.path),
            )
            if value is not None
        ]
        suffix = f" ({', '.join(context)})" if context else ""
        return f"{self.code}: {self.message}{suffix}"

    def to_record(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "stage": self.stage,
        }
        for key in ("path", "scene_spec_id", "task_record_id"):
            item = getattr(self, key)
            if item is not None:
                value[key] = item
        return value


@dataclass(frozen=True)
class ValidationConfig:
    """Versioned validation policy."""

    data: dict[str, Any]

    @classmethod
    def load(cls, path: Path = DEFAULT_CONFIG_PATH) -> "ValidationConfig":
        value = _load_yaml(path)
        if int(value.get("schema_version", 0)) != SCHEMA_VERSION:
            raise ValueError(f"unsupported scene validation schema: {path}")
        if int(value.get("scene_validation_version", 0)) != SCENE_VALIDATION_VERSION:
            raise ValueError(f"unsupported scene validation version: {path}")
        seeds = tuple(int(seed) for seed in value["rollout"]["seeds"])
        if seeds != (0, 1, 2):
            raise ValueError("Milestone 15 v1 rollout seeds must be [0, 1, 2]")
        if int(value["rollout"]["minimum_success_count"]) != 2:
            raise ValueError("Milestone 15 v1 requires at least 2/3 successes")
        settle_policy = (value.get("rollout") or {}).get("settle") or {}
        if (
            str(settle_policy.get("component_policy_version") or "")
            != STABILITY_COMPONENT_POLICY_VERSION
        ):
            raise ValueError(
                "Milestone 15 stability component policy is missing or stale"
            )
        waiver_ids: set[str] = set()
        waiver_tokens: set[str] = set()
        for waiver in value.get("coverage_waivers", []):
            waiver_id = str(waiver.get("id") or "")
            if not waiver_id or waiver_id in waiver_ids:
                raise ValueError("coverage waiver ids must be non-empty and unique")
            waiver_ids.add(waiver_id)
            scopes = {str(item) for item in waiver.get("scopes", [])}
            if not scopes or not scopes <= {"pilot", "full"}:
                raise ValueError(
                    f"coverage waiver {waiver_id} has invalid scopes"
                )
            tokens = {str(item) for item in waiver.get("tokens", [])}
            if not tokens or "" in tokens or tokens & waiver_tokens:
                raise ValueError(
                    "coverage waiver tokens must be non-empty and uniquely owned"
                )
            waiver_tokens.update(tokens)
        return cls(canonical_data(value))

    @property
    def hash(self) -> str:
        return artifact_hash(self.data)

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(int(seed) for seed in self.data["rollout"]["seeds"])

    @property
    def minimum_success_count(self) -> int:
        return int(self.data["rollout"]["minimum_success_count"])


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        # JSON numeric syntax is not a strict subset of YAML 1.1 as parsed by
        # PyYAML: values such as ``3e-05`` are loaded as strings.  Route
        # witnesses are JSON and their content hash is type-sensitive, so they
        # must be decoded with the JSON parser that wrote them.
        value = (
            json.load(stream)
            if path.suffix.lower() == ".json"
            else yaml.load(stream, Loader=yaml.CSafeLoader)
        )
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return value


@lru_cache(maxsize=64)
def _parsed_source(payload: bytes, is_json: bool) -> Any:
    return json.loads(payload) if is_json else yaml.load(payload, Loader=yaml.CSafeLoader)


def _load_source_yaml(path: Path) -> dict[str, Any]:
    """Reuse parsing only: re-read every byte and retain every source self-hash.

    The cache is keyed by the actual bytes, never timestamps. A fresh deep copy
    prevents a caller from mutating the cached source seen by another join.
    Oversized reports bypass this small per-process cache.
    """
    payload = path.read_bytes()
    is_json = path.suffix.lower() == ".json"
    value = (_parsed_source(payload, is_json) if len(payload) <= 256 * 1024
             else json.loads(payload) if is_json else yaml.load(payload, Loader=yaml.CSafeLoader))
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return deepcopy(value)


def _write_yaml(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.dump(
            canonical_data(dict(value)),
            stream,
            Dumper=yaml.CSafeDumper,
            sort_keys=True,
            allow_unicode=True,
            default_flow_style=False,
            width=100,
        )


def _self_hash_valid(value: Mapping[str, Any]) -> bool:
    return value.get("artifact_hash") == artifact_hash(
        value, omit_keys=("artifact_hash",)
    )


def _report_hash_valid(value: Mapping[str, Any]) -> bool:
    return value.get("report_hash") == artifact_hash(
        value, omit_keys=("report_hash",)
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pose_within_tolerance(
    expected_position: Sequence[float],
    expected_quaternion_wxyz: Sequence[float],
    actual_position: Sequence[float],
    actual_quaternion_wxyz: Sequence[float],
    *,
    translation_tolerance_m: float = 1e-6,
    quaternion_tolerance: float = 1e-6,
) -> bool:
    """Compare a frozen pose after a simulator's normal float32 round-trip."""

    def distance(left: Sequence[float], right: Sequence[float]) -> float:
        return math.sqrt(
            sum(
                (float(a) - float(b)) ** 2
                for a, b in zip(left, right, strict=True)
            )
        )

    position_error = distance(expected_position, actual_position)
    quaternion_error = min(
        distance(expected_quaternion_wxyz, actual_quaternion_wxyz),
        distance(
            expected_quaternion_wxyz,
            [-float(item) for item in actual_quaternion_wxyz],
        ),
    )
    return (
        position_error <= float(translation_tolerance_m)
        and quaternion_error <= float(quaternion_tolerance)
    )


def _m13_exclusion_ledger(
    task_dir: Path,
    *,
    scene_spec_id: str | None = None,
) -> tuple[set[tuple[str, str]], dict[str, dict[str, str]]]:
    """Load the complete hashed M13 instruction/skill exclusion ledger."""

    report_paths = [
        (key, task_dir / filename, artifact_kind)
        for key, filename, artifact_kind in M13_EXCLUSION_REPORTS
    ]
    missing = [path for _, path, _ in report_paths if not path.is_file()]
    if missing:
        raise _source_error(
            "missing_m13_exclusion_report",
            "M13 strict subset accounting requires instruction and checker "
            f"reports; missing {[str(path) for path in missing]}",
            path=missing[0],
            scene_spec_id=scene_spec_id,
        )
    versions = tuple(
        (str(path.resolve()), path.stat().st_mtime_ns, path.stat().st_size)
        for _, path, _ in report_paths
    )
    cache_key = str(task_dir.resolve())
    cached = _M13_EXCLUSION_CACHE.get(cache_key)
    if cached is not None and cached[0] == versions:
        return set(cached[1]), canonical_data(cached[2])

    reports: list[tuple[str, Path, dict[str, Any]]] = []
    for key, path, expected_kind in report_paths:
        report = _load_yaml(path)
        if report.get("artifact_kind") != expected_kind:
            raise _source_error(
                "m13_report_kind_mismatch",
                f"expected {expected_kind}, found {report.get('artifact_kind')}",
                path=path,
                scene_spec_id=scene_spec_id,
            )
        if not _report_hash_valid(report):
            raise _source_error(
                "m13_report_hash_mismatch",
                f"M13 {expected_kind} self-hash does not match",
                path=path,
                scene_spec_id=scene_spec_id,
            )
        reports.append((key, path, report))

    artifact_set_hashes = {
        str(report.get("artifact_set_hash") or "")
        for _, _, report in reports
    }
    config_hashes = {
        str(report.get("config_hash") or "")
        for _, _, report in reports
    }
    if (
        len(artifact_set_hashes) != 1
        or "" in artifact_set_hashes
        or len(config_hashes) != 1
        or "" in config_hashes
    ):
        raise _source_error(
            "m13_report_set_mismatch",
            "M13 instruction/checker reports do not describe the same "
            "artifact set and configuration",
            path=task_dir,
            scene_spec_id=scene_spec_id,
        )

    exclusions = {
        (
            str(item.get("scene_spec_id") or ""),
            str(item.get("goal_variant_id") or ""),
        )
        for _, _, report in reports
        for item in report.get("failures", [])
    }
    metadata = {
        key: {
            "path": str(path),
            "report_hash": str(report["report_hash"]),
            "artifact_set_hash": str(report["artifact_set_hash"]),
            "config_hash": str(report["config_hash"]),
        }
        for key, path, report in reports
    }
    _M13_EXCLUSION_CACHE[cache_key] = (
        versions,
        exclusions,
        metadata,
    )
    return set(exclusions), canonical_data(metadata)


def _evidence_sidecar_errors(
    value: Any,
    *,
    record_path: Path,
) -> list[str]:
    errors: list[str] = []
    if isinstance(value, Mapping):
        path_value = value.get("path")
        expected_hash = value.get("sha256") or value.get("file_hash")
        if path_value is not None and expected_hash is not None:
            path = Path(str(path_value))
            if (
                not path.is_file()
                or _sha256_file(path) != str(expected_hash)
                or (
                    value.get("size_bytes") is not None
                    and path.is_file()
                    and path.stat().st_size != int(value["size_bytes"])
                )
            ):
                errors.append(f"{record_path}: stale evidence sidecar {path}")
        for child in value.values():
            errors.extend(
                _evidence_sidecar_errors(child, record_path=record_path)
            )
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes)
    ):
        for child in value:
            errors.extend(
                _evidence_sidecar_errors(child, record_path=record_path)
            )
    return errors


def _source_error(
    code: str,
    message: str,
    *,
    path: Path | None = None,
    scene_spec_id: str | None = None,
    task_record_id: str | None = None,
) -> SceneValidationError:
    return SceneValidationError(
        code,
        message,
        stage="source",
        path=str(path) if path is not None else None,
        scene_spec_id=scene_spec_id,
        task_record_id=task_record_id,
    )


def _require_self_hash(
    value: Mapping[str, Any],
    path: Path,
    *,
    scene_spec_id: str | None = None,
) -> None:
    if not _self_hash_valid(value):
        raise _source_error(
            "source_self_hash_mismatch",
            f"artifact self-hash does not match {path}",
            path=path,
            scene_spec_id=scene_spec_id,
        )


def _require_artifact_version(
    value: Mapping[str, Any],
    path: Path | None,
    *,
    field: str,
    expected: int,
    code: str,
    scene_spec_id: str | None = None,
    task_record_id: str | None = None,
) -> None:
    observed = value.get(field)
    if observed != expected:
        raise _source_error(
            code,
            f"{field} is {observed!r}, expected {expected!r}",
            path=path,
            scene_spec_id=scene_spec_id,
            task_record_id=task_record_id,
        )


def _find_scene_path(scene_spec_id: str, scene_spec_dir: Path) -> Path:
    matches = sorted(scene_spec_dir.glob("rsf_*/*.yml"))
    for path in matches:
        # Full YAML loads are avoided for the common case by using the stable id
        # token.  A fallback load confirms candidates with unusual formatting.
        if scene_spec_id in path.read_text(encoding="utf-8", errors="ignore"):
            value = _load_yaml(path)
            if value.get("scene_spec", {}).get("id") == scene_spec_id:
                return path
    raise _source_error(
        "missing_scene_spec",
        f"SceneSpec {scene_spec_id} was not found",
        scene_spec_id=scene_spec_id,
    )


def strict_source_join(
    scene: Mapping[str, Any] | Path,
    *,
    realization_dir: Path = DEFAULT_REALIZATION_DIR,
    unit_dir: Path = DEFAULT_UNIT_DIR,
    task_dir: Path = DEFAULT_TASK_DIR,
    route_dir: Path | None = None,
) -> dict[str, Any]:
    """Join one SceneSpec to exact M10, M11, and every available M13 task.

    Every source must be self-consistent and all recorded upstream hashes must
    agree.  The function intentionally rejects a partially refreshed pipeline.
    """

    scene_path = Path(scene) if isinstance(scene, (str, Path)) else None
    scene_artifact = _load_source_yaml(scene_path) if scene_path is not None else dict(scene)
    scene_spec_id = str(scene_artifact.get("scene_spec", {}).get("id") or "")
    if not scene_spec_id:
        raise _source_error(
            "invalid_scene_spec", "SceneSpec id is missing", path=scene_path
        )
    if scene_path is not None:
        _require_self_hash(scene_artifact, scene_path, scene_spec_id=scene_spec_id)
    elif not _self_hash_valid(scene_artifact):
        raise _source_error(
            "source_self_hash_mismatch",
            "in-memory SceneSpec self-hash does not match",
            scene_spec_id=scene_spec_id,
        )
    if (
        scene_artifact.get("artifact_kind") != "solved_scene_spec"
        or scene_artifact.get("preflight", {}).get("status") != "passed"
    ):
        raise _source_error(
            "invalid_scene_spec_source",
            "SceneSpec is not a preflight-passed solved artifact",
            path=scene_path,
            scene_spec_id=scene_spec_id,
        )
    _require_artifact_version(
        scene_artifact,
        scene_path,
        field="layout_solver_version",
        expected=LAYOUT_SOLVER_VERSION,
        code="stale_scene_spec_version",
        scene_spec_id=scene_spec_id,
    )

    realized_family_id = str(scene_artifact.get("realized_family_id") or "")
    if not realized_family_id:
        raise _source_error(
            "invalid_scene_spec",
            "SceneSpec realized_family_id is missing",
            scene_spec_id=scene_spec_id,
        )
    realization_path = realization_dir / f"{realized_family_id}.yml"
    unit_path = unit_dir / f"{realized_family_id}.yml"
    if not realization_path.is_file():
        raise _source_error(
            "missing_realized_family",
            f"missing RealizedSceneFamily {realized_family_id}",
            path=realization_path,
            scene_spec_id=scene_spec_id,
        )
    if not unit_path.is_file():
        raise _source_error(
            "missing_unit_build",
            f"missing Unit build {realized_family_id}",
            path=unit_path,
            scene_spec_id=scene_spec_id,
        )
    realized = _load_source_yaml(realization_path)
    unit = _load_source_yaml(unit_path)
    _require_self_hash(realized, realization_path, scene_spec_id=scene_spec_id)
    _require_self_hash(unit, unit_path, scene_spec_id=scene_spec_id)
    if (
        realized.get("artifact_kind") != "realized_scene_family"
        or realized.get("preflight", {}).get("status") != "passed"
    ):
        raise _source_error(
            "invalid_realized_family_source",
            "RealizedSceneFamily preflight is not passed",
            path=realization_path,
            scene_spec_id=scene_spec_id,
        )
    _require_artifact_version(
        realized,
        realization_path,
        field="realization_version",
        expected=REALIZATION_VERSION,
        code="stale_realization_version",
        scene_spec_id=scene_spec_id,
    )
    if (
        unit.get("artifact_kind") != "unit_build"
        or unit.get("preflight", {}).get("status") != "passed"
    ):
        raise _source_error(
            "invalid_unit_source",
            "Unit build preflight is not passed",
            path=unit_path,
            scene_spec_id=scene_spec_id,
        )
    _require_artifact_version(
        unit,
        unit_path,
        field="unit_build_version",
        expected=UNIT_BUILD_VERSION,
        code="stale_unit_version",
        scene_spec_id=scene_spec_id,
    )

    if realized.get("realized_family_id") != realized_family_id:
        raise _source_error(
            "stale_realization_join",
            "RealizedSceneFamily id does not match SceneSpec",
            path=realization_path,
            scene_spec_id=scene_spec_id,
        )
    if scene_artifact.get("source_realization_hash") != realized.get("artifact_hash"):
        raise _source_error(
            "stale_realization_join",
            "SceneSpec source_realization_hash is stale",
            path=scene_path or realization_path,
            scene_spec_id=scene_spec_id,
        )
    if unit.get("realized_family_id") != realized_family_id:
        raise _source_error(
            "stale_unit_join",
            "Unit realized_family_id does not match SceneSpec",
            path=unit_path,
            scene_spec_id=scene_spec_id,
        )
    if unit.get("source_realization_hash") != realized.get("artifact_hash"):
        raise _source_error(
            "stale_unit_join",
            "Unit source_realization_hash is stale",
            path=unit_path,
            scene_spec_id=scene_spec_id,
        )
    if scene_artifact.get("source_unit_hash") != unit.get("artifact_hash"):
        raise _source_error(
            "stale_unit_join",
            "SceneSpec source_unit_hash is stale",
            path=scene_path or unit_path,
            scene_spec_id=scene_spec_id,
        )
    try:
        pick_planning_contract = validated_scene_pick_planning_contract(
            scene_artifact,
            realized_artifact=realized,
            unit_artifact=unit,
        )
    except ValueError as exc:
        raise _source_error(
            "invalid_pick_planning_evidence",
            f"SceneSpec Pick planning evidence is invalid: {exc}",
            path=scene_path or unit_path,
            scene_spec_id=scene_spec_id,
        ) from exc

    task_paths = sorted((task_dir / scene_spec_id).glob("*.yml"))
    tasks: list[dict[str, Any]] = []
    seen_task_ids: set[str] = set()
    seen_variants: set[str] = set()
    for path in task_paths:
        task = _load_source_yaml(path)
        _require_self_hash(task, path, scene_spec_id=scene_spec_id)
        task_id = str(task.get("task_record", {}).get("id") or "")
        variant_id = str(task.get("goal_variant_id") or "")
        if not task_id or task_id in seen_task_ids or variant_id in seen_variants:
            raise _source_error(
                "duplicate_or_invalid_task",
                f"duplicate or invalid TaskRecord {task_id!r}/{variant_id!r}",
                path=path,
                scene_spec_id=scene_spec_id,
                task_record_id=task_id or None,
            )
        if (
            task.get("artifact_kind") != "rendered_task_record"
            or task.get("preflight", {}).get("status") != "passed"
        ):
            raise _source_error(
                "invalid_task_source",
                f"TaskRecord {task_id} preflight is not passed",
                path=path,
                scene_spec_id=scene_spec_id,
                task_record_id=task_id,
            )
        _require_artifact_version(
            task,
            path,
            field="task_render_version",
            expected=TASK_RENDER_VERSION,
            code="stale_task_version",
            scene_spec_id=scene_spec_id,
            task_record_id=task_id,
        )
        if (
            task.get("scene_spec_id") != scene_spec_id
            or task.get("realized_family_id") != realized_family_id
            or task.get("source_scene_hash") != scene_artifact.get("artifact_hash")
            or task.get("source_realization_hash") != realized.get("artifact_hash")
        ):
            raise _source_error(
                "stale_task_join",
                f"TaskRecord {task_id} has a stale source join",
                path=path,
                scene_spec_id=scene_spec_id,
                task_record_id=task_id,
            )
        record = task.get("task_record") or {}
        if (
            record.get("scene_spec_id") != scene_spec_id
            or record.get("realized_family_id") != realized_family_id
        ):
            raise _source_error(
                "stale_task_payload_join",
                f"TaskRecord payload {task_id} has a stale scene join",
                path=path,
                scene_spec_id=scene_spec_id,
                task_record_id=task_id,
            )
        skill = record.get("skill_config") or {}
        if str(skill.get("route") or "") != "direct" or any(
            isinstance(step, Mapping)
            and str(step.get("op") or "") == "handover"
            for step in skill.get("steps") or []
        ):
            raise _source_error(
                "disabled_handover_route",
                f"TaskRecord {task_id} is not direct-only",
                path=path,
                scene_spec_id=scene_spec_id,
                task_record_id=task_id,
            )
        checker = record.get("checker_config") or {}
        expected_integrity = build_scene_integrity_binding(
            record.get("goal") or {},
            scene_artifact["scene_spec"]["entities"],
            SUCCESS_CONTRACT,
        )
        if (
            checker.get("version") != CHECKER_CONFIG_VERSION
            or "conditions" in checker
            or not isinstance(checker.get("predicate"), Mapping)
            or checker.get("scene_integrity") != expected_integrity
            or checker.get("success_contract") != SUCCESS_CONTRACT.identity
        ):
            raise _source_error(
                "stale_success_checker_contract",
                f"TaskRecord {task_id} does not use canonical checker v5",
                path=path,
                scene_spec_id=scene_spec_id,
                task_record_id=task_id,
            )
        if (
            (record.get("goal") or {}).get("action_family") == "O2O_relation"
            and checker.get("relation_contract") != RELATION_CONTRACT.identity
        ):
            raise _source_error(
                "stale_relation_checker_contract",
                f"TaskRecord {task_id} relation contract is stale",
                path=path,
                scene_spec_id=scene_spec_id,
                task_record_id=task_id,
            )
        seen_task_ids.add(task_id)
        seen_variants.add(variant_id)
        tasks.append(task)

    expected_variants = set(
        str(item)
        for item in scene_artifact.get("scene_spec", {}).get("goal_variant_ids", [])
    )
    missing_variants = sorted(expected_variants - seen_variants)
    unexpected_variants = sorted(seen_variants - expected_variants)
    if unexpected_variants:
        raise _source_error(
            "unexpected_task_variants",
            f"TaskRecords are not declared by SceneSpec: {unexpected_variants}",
            path=task_dir / scene_spec_id,
            scene_spec_id=scene_spec_id,
        )
    if missing_variants:
        # M13 can publish a validated subset.  The missing variants are accepted
        # only when the complete M13 instruction + skill report set explicitly
        # accounts for them.
        excluded, _ = _m13_exclusion_ledger(
            task_dir,
            scene_spec_id=scene_spec_id,
        )
        unaccounted = [
            variant
            for variant in missing_variants
            if (scene_spec_id, variant) not in excluded
        ]
        if unaccounted:
            raise _source_error(
                "unaccounted_missing_tasks",
                f"missing TaskRecords are not accounted by M13: {unaccounted}",
                path=task_dir / scene_spec_id,
                scene_spec_id=scene_spec_id,
            )

    routes: dict[str, dict[str, Any]] = {}
    route_paths: list[str] = []
    if route_dir is not None:
        for task in tasks:
            record = task["task_record"]
            skill = record.get("skill_config") or {}
            carry_steps = [
                step
                for step in skill.get("steps", [])
                if str(step.get("op") or "") == "carry_to_relation"
            ]
            if str(skill.get("route") or "") != "direct" or not carry_steps:
                continue
            task_id = str(record["id"])
            path = route_dir / scene_spec_id / f"{task_id}.json"
            if not path.is_file():
                raise _source_error(
                    "missing_route_witness",
                    f"missing exact-scene route witness for {task_id}",
                    path=path,
                    scene_spec_id=scene_spec_id,
                    task_record_id=task_id,
                )
            try:
                route = validate_route_witness_report(
                    _load_source_yaml(path),
                    scene_spec_id=scene_spec_id,
                    task_record_id=task_id,
                    source_hashes={
                        "scene": str(scene_artifact["artifact_hash"]),
                        "realized": str(realized["artifact_hash"]),
                        "unit": str(unit["artifact_hash"]),
                        "task": str(task["artifact_hash"]),
                    },
                    skill_config=skill,
                    unit_artifact=unit,
                    scene_spec=scene_artifact["scene_spec"],
                )
                selected = route["selected_witness"]
                target_key = f"{route['subject']}:{route['arm']}"
                planning_targets = {
                    f"{item['entity_key']}:{item['arm']}": item
                    for item in (
                        (pick_planning_contract.get("request") or {}).get(
                            "targets", []
                        )
                    )
                }
                planning_target = planning_targets.get(target_key)
                if not isinstance(planning_target, Mapping):
                    raise ValueError(
                        "route witness has no matching M12 Pick target"
                    )
                planning_request = planning_target.get("planning_request") or {}
                if str(route.get("m12_pick_planning_request_hash") or "") != str(
                    artifact_hash(planning_request)
                ):
                    raise ValueError(
                        "route witness M12 Pick planning request is stale"
                    )
                planning_witness_rows = [
                    row
                    for row in (
                        (pick_planning_contract.get("witness") or {}).get(
                            "results", []
                        )
                    )
                    if isinstance(row, Mapping)
                    and str(row.get("entity_key") or "")
                    == str(route["subject"])
                    and str(row.get("arm") or "") == str(route["arm"])
                    and row.get("status") == "passed"
                ]
                if len(planning_witness_rows) != 1:
                    raise ValueError(
                        "route witness has no unique M12 Pick witness"
                    )
                m12_selected = planning_witness_rows[0].get(
                    "selected_candidate"
                )
                if (
                    int(selected["contact_id"])
                    not in {
                        int(value)
                        for value in planning_request.get("contact_ids", [])
                    }
                    or str(selected.get("contact_contract_hash") or "")
                    != str(
                        (
                            planning_target.get("contact_contract_binding")
                            or {}
                        ).get("contact_contract_hash")
                        or ""
                    )
                    or not isinstance(m12_selected, Mapping)
                    or int(m12_selected.get("contact_id", -1))
                    != int(selected["contact_id"])
                ):
                    raise ValueError(
                        "route witness Contact differs from M12 frozen Pick witness"
                    )
            except (TypeError, ValueError) as exc:
                raise _source_error(
                    "invalid_route_witness",
                    f"route witness for {task_id} is invalid: {exc}",
                    path=path,
                    scene_spec_id=scene_spec_id,
                    task_record_id=task_id,
                ) from exc
            routes[task_id] = route
            route_paths.append(str(path))

    return {
        "scene": canonical_data(scene_artifact),
        "realized": canonical_data(realized),
        "unit": canonical_data(unit),
        "tasks": canonical_data(tasks),
        "routes": canonical_data(routes),
        "paths": {
            "scene": str(scene_path) if scene_path is not None else None,
            "realized": str(realization_path),
            "unit": str(unit_path),
            "tasks": [str(path) for path in task_paths],
            "routes": route_paths,
        },
        "missing_goal_variant_ids": missing_variants,
        "pick_planning_contract": pick_planning_contract,
    }


def _mask_metrics(
    frame: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    if isinstance(frame.get("entities"), Mapping):
        result: dict[str, dict[str, Any]] = {}
        for key, raw in frame["entities"].items():
            item = dict(raw)
            bbox = item.get("bbox") or [0, 0, -1, -1]
            area = int(item.get("area_px", item.get("area", 0)))
            width = max(0, int(bbox[2]) - int(bbox[0]) + 1)
            height = max(0, int(bbox[3]) - int(bbox[1]) + 1)
            bbox_area = width * height
            item["area_px"] = area
            item["bbox"] = [int(value) for value in bbox]
            item["bbox_fill"] = float(
                item.get("bbox_fill", area / bbox_area if bbox_area else 0.0)
            )
            item["centroid_px"] = [
                float(value)
                for value in item.get(
                    "centroid_px",
                    [
                        (float(bbox[0]) + float(bbox[2])) / 2.0,
                        (float(bbox[1]) + float(bbox[3])) / 2.0,
                    ],
                )
            ]
            result[str(key)] = item
        return result

    segmentation = frame.get("actor_segmentation")
    actor_ids = frame.get("entity_actor_ids") or {}
    if segmentation is None or not isinstance(actor_ids, Mapping):
        raise ValueError("frame requires entities or actor_segmentation/entity_actor_ids")
    depth = frame.get("depth")
    rgb = frame.get("rgb")
    result = {}
    for entity_key, actor_id in actor_ids.items():
        accepted_actor_ids = (
            {int(item) for item in actor_id}
            if isinstance(actor_id, Sequence)
            and not isinstance(actor_id, (str, bytes))
            else {int(actor_id)}
        )
        xs: list[int] = []
        ys: list[int] = []
        depths: list[float] = []
        colors: list[tuple[float, float, float]] = []
        for y, row in enumerate(segmentation):
            for x, value in enumerate(row):
                if int(value) not in accepted_actor_ids:
                    continue
                xs.append(x)
                ys.append(y)
                if depth is not None:
                    depths.append(float(depth[y][x]))
                if rgb is not None:
                    pixel = rgb[y][x]
                    colors.append(tuple(float(pixel[index]) for index in range(3)))
        if not xs:
            result[str(entity_key)] = {
                "area_px": 0,
                "bbox": [0, 0, -1, -1],
                "bbox_fill": 0.0,
                "centroid_px": [0.0, 0.0],
            }
            continue
        bbox = [min(xs), min(ys), max(xs), max(ys)]
        bbox_area = (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)
        item: dict[str, Any] = {
            "area_px": len(xs),
            "bbox": bbox,
            "bbox_fill": len(xs) / bbox_area,
            "centroid_px": [sum(xs) / len(xs), sum(ys) / len(ys)],
        }
        if depths:
            item["median_depth_m"] = sorted(depths)[len(depths) // 2]
        if colors:
            item["mean_rgb"] = [
                sum(pixel[index] for pixel in colors) / len(colors)
                for index in range(3)
            ]
        result[str(entity_key)] = item
    return result


def _visible(item: Mapping[str, Any], thresholds: Mapping[str, Any]) -> bool:
    return (
        int(item.get("area_px", 0)) >= int(thresholds["minimum_mask_area_px"])
        and float(item.get("bbox_fill", 0.0))
        >= float(thresholds["minimum_bbox_fill"])
    )


def _entity_role_map(realized: Mapping[str, Any]) -> dict[str, str]:
    payload = realized.get("realized_scene_family") or realized
    result: dict[str, str] = {}
    for role, binding in (payload.get("role_bindings") or {}).items():
        if isinstance(binding, Mapping) and binding.get("entity_key"):
            result[str(role)] = str(binding["entity_key"])
    return result


def _resolved_attributes(realized: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    payload = realized.get("realized_scene_family") or realized
    assignment = (payload.get("evidence") or {}).get("assignment") or {}
    values = assignment.get("resolved_values") or {}
    return {
        str(role): dict((item or {}).get("attributes") or {})
        for role, item in values.items()
    }


def _descriptor_requests(
    tasks: Sequence[Mapping[str, Any]], role_map: Mapping[str, str]
) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    for artifact in tasks:
        task = artifact.get("task_record") or artifact
        provenance = task.get("goal_provenance") or {}
        goal = task.get("goal") or {}
        role_refs = provenance.get("role_refs") or {}
        entity_refs = provenance.get("entity_refs") or {}
        for axis in provenance.get("intended_axes") or []:
            code = str(axis.get("axis") or "")
            if code not in {"D1a", "D1b", "D1c"}:
                continue
            semantic_role = str(axis.get("role") or "subject")
            role = role_refs.get(semantic_role)
            entity = entity_refs.get(semantic_role) or goal.get(semantic_role)
            if role and not entity:
                entity = role_map.get(str(role))
            requests.append(
                {
                    "task_record_id": task.get("id"),
                    "axis": code,
                    "subtype": axis.get("subtype"),
                    "semantic_role": semantic_role,
                    "role_ref": role,
                    "entity_key": entity,
                    "goal": canonical_data(goal),
                }
            )
    unique = {
        artifact_hash(request): request
        for request in requests
        if request.get("entity_key")
    }
    return [unique[key] for key in sorted(unique)]


def _descriptor_peer_roles(
    role: str, role_map: Mapping[str, str]
) -> list[str]:
    """Return semantic peers without pulling landmarks/receivers into the set."""

    if role.startswith("subj"):
        return sorted(
            candidate
            for candidate in role_map
            if candidate != role and candidate.startswith("subj")
        )
    prefix = role.rsplit("_", 1)[0] if "_" in role else role
    return sorted(
        candidate
        for candidate in role_map
        if candidate != role
        and (
            candidate == prefix
            or candidate.startswith(prefix + "_")
        )
    )


def _bbox_contains(
    outer: Sequence[int], inner: Sequence[int], tolerance_px: int = 2
) -> bool:
    return (
        int(inner[0]) >= int(outer[0]) - tolerance_px
        and int(inner[1]) >= int(outer[1]) - tolerance_px
        and int(inner[2]) <= int(outer[2]) + tolerance_px
        and int(inner[3]) <= int(outer[3]) + tolerance_px
    )


def evaluate_descriptor_gate(
    scene_spec: Mapping[str, Any],
    realized_family: Mapping[str, Any],
    frame: Mapping[str, Any],
    tasks: Sequence[Mapping[str, Any]] = (),
    *,
    config: ValidationConfig | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate D1 descriptor grounding from one settled head-camera frame."""

    if config is None:
        cfg = ValidationConfig.load().data
    elif isinstance(config, ValidationConfig):
        cfg = config.data
    else:
        cfg = dict(config)
    thresholds = cfg["descriptor_gate"]
    metrics = _mask_metrics(frame)
    role_map = _entity_role_map(realized_family)
    attributes = _resolved_attributes(realized_family)
    requests = _descriptor_requests(tasks, role_map)
    checks: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for request in requests:
        entity = str(request["entity_key"])
        item = metrics.get(entity, {})
        visible = _visible(item, thresholds)
        check: dict[str, Any] = {
            **request,
            "visible": visible,
            "metrics": canonical_data(item),
            "passed": visible,
            "diagnostics": {},
        }
        if not visible:
            check["failure_code"] = "descriptor_target_not_visible"
        axis = request["axis"]
        subtype = str(request.get("subtype") or "")
        role = str(request.get("role_ref") or "")

        if axis == "D1a":
            value = attributes.get(role, {}).get(subtype)
            semantic_peer_roles = set(_descriptor_peer_roles(role, role_map))
            peer_roles = [
                candidate
                for candidate, candidate_attrs in attributes.items()
                if candidate in semantic_peer_roles and subtype in candidate_attrs
            ]
            peers = [role_map[candidate] for candidate in peer_roles if candidate in role_map]
            peer_visibility = {
                peer: _visible(metrics.get(peer, {}), thresholds) for peer in peers
            }
            reviewed_unique = value is not None and all(
                attributes[candidate].get(subtype) != value
                for candidate in peer_roles
            )
            check["diagnostics"].update(
                {
                    "reviewed_value": value,
                    "reviewed_unique": reviewed_unique,
                    "peer_visibility": peer_visibility,
                }
            )
            check["passed"] = bool(
                check["passed"]
                and reviewed_unique
                and peers
                and all(peer_visibility.values())
            )
            if subtype == "size" and visible and peers:
                target_area = float(item.get("area_px", 0))
                visible_peer_roles = [
                    peer_role
                    for peer_role in peer_roles
                    if peer_visibility.get(role_map.get(peer_role, ""))
                ]
                peer_areas = [
                    float(metrics[role_map[peer_role]].get("area_px", 0))
                    for peer_role in visible_peer_roles
                ]
                ratios = [
                    max(target_area, area) / max(1.0, min(target_area, area))
                    for area in peer_areas
                ]
                minimum_ratio = float(thresholds["minimum_area_ratio"])
                rank = {"small": 0, "medium": 1, "large": 2}
                direction_results = []
                for peer_role, peer_area in zip(
                    visible_peer_roles, peer_areas
                ):
                    peer_value = attributes[peer_role].get(subtype)
                    if value in rank and peer_value in rank:
                        direction_results.append(
                            target_area * minimum_ratio <= peer_area
                            if rank[str(value)] < rank[str(peer_value)]
                            else target_area >= peer_area * minimum_ratio
                        )
                    else:
                        direction_results.append(True)
                size_ok = (
                    bool(ratios)
                    and min(ratios) >= minimum_ratio
                    and all(direction_results)
                )
                check["diagnostics"]["projected_area_ratios"] = ratios
                check["diagnostics"][
                    "projected_area_direction_ok"
                ] = direction_results
                check["passed"] = bool(check["passed"] and size_ok)
            if subtype == "color" and visible:
                target_rgb = item.get("mean_rgb")
                distances = {}
                if target_rgb is not None:
                    for peer in peers:
                        peer_rgb = metrics.get(peer, {}).get("mean_rgb")
                        if peer_rgb is not None:
                            distances[peer] = math.sqrt(
                                sum(
                                    (float(target_rgb[index]) - float(peer_rgb[index]))
                                    ** 2
                                    for index in range(3)
                                )
                            )
                # Calibration diagnostic only: reviewed labels remain normative.
                check["diagnostics"]["color_mask_rgb_distances"] = distances

        elif axis == "D1b":
            goal = request["goal"]
            descriptors = (goal.get("descriptors") or {}).get(
                request["semantic_role"], {}
            )
            spatial = descriptors.get("spatial") or {}
            predicates = [
                str(item)
                for item in spatial.get(
                    "predicates",
                    [
                        spatial.get("relation")
                        or spatial.get("value")
                        or ""
                    ],
                )
                if item
            ]
            reference = (
                spatial.get("reference")
                or spatial.get("landmark")
                or goal.get("target")
            )
            if isinstance(reference, Mapping):
                reference = reference.get("entity_key")
            reference = str(reference or "")
            ref_item = metrics.get(reference, {})
            ref_visible = _visible(ref_item, thresholds)
            check["diagnostics"]["reference_entity"] = reference
            check["diagnostics"]["reference_visible"] = ref_visible
            check["passed"] = bool(check["passed"] and ref_visible)
            if visible and ref_visible and predicates:
                cx, cy = item["centroid_px"]
                rx, ry = ref_item["centroid_px"]
                gap = float(thresholds["minimum_image_gap_px"])
                relation_results: dict[str, bool] = {}
                for relation in predicates:
                    relation_ok = {
                        "left_of": cx <= rx - gap,
                        "right_of": cx >= rx + gap,
                        "in_front_of": cy >= ry + gap,
                        "behind": cy <= ry - gap,
                    }.get(relation)
                    if relation in {"in", "on"}:
                        world_relations = frame.get("world_relations") or {}
                        relation_facts = set(
                            tuple(value)
                            for value in world_relations.get(relation, [])
                        )
                        world_ok = (entity, reference) in relation_facts
                        if relation == "in":
                            image_ok = _bbox_contains(
                                ref_item.get("bbox", [0, 0, -1, -1]),
                                item.get("bbox", [0, 0, -1, -1]),
                            )
                        else:
                            subject_bbox = item.get("bbox", [0, 0, -1, -1])
                            target_bbox = ref_item.get("bbox", [0, 0, -1, -1])
                            horizontal_overlap = min(
                                subject_bbox[2], target_bbox[2]
                            ) - max(subject_bbox[0], target_bbox[0])
                            image_ok = (
                                horizontal_overlap >= 0
                                and abs(
                                    float(subject_bbox[3])
                                    - float(target_bbox[1])
                                )
                                <= gap * 2
                            )
                        relation_ok = bool(image_ok and world_ok)
                        check["diagnostics"].update(
                            {
                                f"{relation}_image_relation_unique": bool(image_ok),
                                f"{relation}_world_relation_evidence": bool(world_ok),
                            }
                        )
                    relation_results[relation] = bool(relation_ok)
                check["passed"] = bool(
                    check["passed"]
                    and relation_results
                    and all(relation_results.values())
                )
                check["diagnostics"]["relations"] = relation_results

        elif axis == "D1c":
            goal = request["goal"]
            descriptors = (goal.get("descriptors") or {}).get(
                request["semantic_role"], {}
            )
            ordinal = descriptors.get("ordinal") or {}
            basis = str(ordinal.get("basis") or subtype)
            order = str(
                ordinal.get("order")
                or ordinal.get("value")
                or ordinal.get("rank")
                or ""
            )
            peer_roles = _descriptor_peer_roles(role, role_map)
            peers = [
                role_map[candidate]
                for candidate in peer_roles
                if candidate in role_map
                and _visible(metrics.get(role_map[candidate], {}), thresholds)
            ]
            ok = bool(peers)
            if visible and peers:
                if basis == "position":
                    values = {
                        key: float(candidate["centroid_px"][0])
                        for key, candidate in metrics.items()
                        if key == entity or key in peers
                    }
                    ordered = sorted(values, key=lambda key: (values[key], key))
                    if order in {"rightmost", "farthest"}:
                        ordered.reverse()
                    ok = ordered[0] == entity and (
                        len(ordered) == 1
                        or abs(values[ordered[0]] - values[ordered[1]])
                        >= float(thresholds["minimum_image_gap_px"])
                    )
                    check["diagnostics"]["ordered_projection"] = ordered
                elif basis == "landmark":
                    reference = str(ordinal.get("reference") or "")
                    reference_item = metrics.get(reference, {})
                    reference_visible = _visible(reference_item, thresholds)
                    values = {
                        key: math.dist(
                            [
                                float(candidate["centroid_px"][0]),
                                float(candidate["centroid_px"][1]),
                            ],
                            [
                                float(reference_item.get("centroid_px", [0, 0])[0]),
                                float(reference_item.get("centroid_px", [0, 0])[1]),
                            ],
                        )
                        for key, candidate in metrics.items()
                        if key == entity or key in peers
                    }
                    reverse = order == "farthest"
                    ordered = sorted(
                        values,
                        key=lambda key: (values[key], key),
                        reverse=reverse,
                    )
                    ok = reference_visible and ordered[0] == entity and (
                        len(ordered) == 1
                        or abs(values[ordered[0]] - values[ordered[1]])
                        >= float(thresholds["minimum_image_gap_px"])
                    )
                    check["diagnostics"].update(
                        {
                            "reference_entity": reference,
                            "reference_visible": reference_visible,
                            "ordered_landmark_distance": ordered,
                        }
                    )
                elif basis == "robot":
                    values = {
                        key: float(
                            candidate.get("median_depth_m", float("inf"))
                        )
                        for key, candidate in metrics.items()
                        if key == entity or key in peers
                    }
                    reverse = order == "farthest"
                    ordered = sorted(values, key=lambda key: (values[key], key), reverse=reverse)
                    ok = ordered[0] == entity and (
                        len(ordered) == 1
                        or abs(values[ordered[0]] - values[ordered[1]])
                        >= float(thresholds["minimum_depth_gap_m"])
                    )
                    check["diagnostics"]["ordered_depth"] = ordered
                elif basis == "size":
                    values = {
                        key: float(candidate.get("area_px", 0))
                        for key, candidate in metrics.items()
                        if key == entity or key in peers
                    }
                    reverse = order in {"largest", "farthest"}
                    ordered = sorted(values, key=lambda key: (values[key], key), reverse=reverse)
                    ratio = max(values[ordered[0]], values[ordered[1]]) / max(
                        1.0, min(values[ordered[0]], values[ordered[1]])
                    )
                    ok = ordered[0] == entity and ratio >= float(
                        thresholds["minimum_area_ratio"]
                    )
                    check["diagnostics"]["adjacent_area_ratio"] = ratio
            check["passed"] = bool(check["passed"] and ok)

        if not check["passed"]:
            code = str(check.get("failure_code") or "descriptor_disambiguation_failed")
            failures.append(
                {
                    "code": code,
                    "task_record_id": request.get("task_record_id"),
                    "axis": axis,
                    "subtype": subtype,
                    "entity_key": entity,
                }
            )
        checks.append(canonical_data(check))

    visible_entities = {
        key: _visible(value, thresholds) for key, value in metrics.items()
    }
    return {
        "status": "passed" if not failures else "failed",
        "passed": not failures,
        "render_count": 1,
        "frame_hash": str(frame.get("frame_hash") or artifact_hash(frame)),
        "entity_metrics": canonical_data(metrics),
        "visible_entities": visible_entities,
        "checks": checks,
        "failures": failures,
    }


def evaluate_checker_config(
    checker_config: Mapping[str, Any], snapshot: Mapping[str, Any]
) -> dict[str, Any]:
    """Evaluate one M13 v5 predicate plus its scene-integrity guard."""

    if (
        checker_config.get("version") != CHECKER_CONFIG_VERSION
        or "conditions" in checker_config
    ):
        return {
            "status": "blocked",
            "passed": None,
            "predicate": None,
            "failed_predicates": ["checker_config_v5_required"],
            "metrics": {},
            **SUCCESS_CONTRACT.identity,
            "success_snapshot_hash": snapshot.get("success_snapshot_hash"),
        }
    if checker_config.get("success_contract") != SUCCESS_CONTRACT.identity:
        return {
            "status": "blocked",
            "passed": None,
            "predicate": None,
            "failed_predicates": ["checker_success_contract_hash_mismatch"],
            "metrics": {},
            **SUCCESS_CONTRACT.identity,
            "success_snapshot_hash": snapshot.get("success_snapshot_hash"),
        }
    binding = checker_config.get("predicate") or {}
    if str(binding.get("name") or "") == "click_press_success":
        arguments = binding.get("arguments") or {}
        use = str(arguments.get("use") or "")
        event = str(arguments.get("event") or "")
        arm = str(arguments.get("requested_arm") or "")
        subject = str(arguments.get("subject") or "")
        anchor_id = str(arguments.get("operation_anchor_id") or "")
        contact_id = arguments.get("contact_id")
        valid_contact_id = (
            isinstance(contact_id, int)
            and not isinstance(contact_id, bool)
            and contact_id >= 0
        )
        expected_anchor_id = (
            f"grasp_contact:{int(contact_id)}" if valid_contact_id else ""
        )
        if (
            checker_config.get("family") != "A2O_event"
            or use not in TOUCH_EXECUTABLE_USES
            or event != TOUCH_EVENT_BY_USE.get(use)
            or arm not in {"left", "right"}
            or not subject
            or not valid_contact_id
            or anchor_id != expected_anchor_id
        ):
            return {
                "status": "blocked",
                "passed": None,
                "predicate": "click_press_success",
                "failed_predicates": ["checker_touch_event_binding_invalid"],
                "metrics": {
                    "use": use or None,
                    "event": event or None,
                    "requested_arm": arm or None,
                    "subject": subject or None,
                    "contact_id": contact_id,
                    "operation_anchor_id": anchor_id or None,
                },
                **SUCCESS_CONTRACT.identity,
                "success_snapshot_hash": snapshot.get("success_snapshot_hash"),
            }
    if (
        str(binding.get("name") or "") == "placement_completed"
        and checker_config.get("relation_contract") != RELATION_CONTRACT.identity
    ):
        return {
            "status": "blocked",
            "passed": None,
            "predicate": "placement_completed",
            "failed_predicates": ["checker_relation_contract_hash_mismatch"],
            "metrics": {},
            **SUCCESS_CONTRACT.identity,
            "success_snapshot_hash": snapshot.get("success_snapshot_hash"),
        }
    primary = evaluate_m15_success(
        str(binding.get("name") or ""),
        binding.get("arguments") or {},
        snapshot,
        SUCCESS_CONTRACT,
    )
    integrity = evaluate_scene_integrity(
        checker_config.get("scene_integrity") or {},
        snapshot,
        SUCCESS_CONTRACT,
    )
    primary["scene_integrity_gate"] = integrity
    if integrity.get("status") == "blocked":
        primary["status"] = "blocked"
        primary["passed"] = None
        primary.setdefault("failed_predicates", []).extend(
            value
            for value in integrity.get("failed_predicates", [])
            if value not in primary.get("failed_predicates", [])
        )
    elif primary.get("status") != "blocked" and not integrity.get("passed"):
        primary["status"] = "failed"
        primary["passed"] = False
        if "scene_integrity" not in primary.setdefault(
            "failed_predicates", []
        ):
            primary["failed_predicates"].append("scene_integrity")
    return canonical_data(primary)


def evaluate_task_checker(
    task_record: Mapping[str, Any], snapshot: Mapping[str, Any]
) -> dict[str, Any]:
    """Evaluate an M13 checker plus M15's D4 action-provenance gate.

    Some D4 contrasts intentionally have the same physical terminal state.
    M15 therefore checks which arm actually completed the primary action; the
    M13 checker configuration itself remains unchanged.
    """

    result = evaluate_checker_config(
        task_record.get("checker_config") or {}, snapshot
    )
    intended_axes = (
        (task_record.get("goal_provenance") or {}).get("intended_axes") or []
    )
    required = any(
        str(axis.get("axis") or "") == "D4"
        for axis in intended_axes
        if isinstance(axis, Mapping)
    )
    semantic_expected_arm = (
        str((task_record.get("goal") or {}).get("arm") or "")
        if required
        else ""
    )
    checker_predicate = (
        (task_record.get("checker_config") or {}).get("predicate") or {}
    )
    predicate_name = str(checker_predicate.get("name") or "")
    touch_binding_passed = True
    provenance_consistent = True
    if predicate_name == "click_press_success":
        # Click/Press D4 provenance is intentionally the arm recorded on the
        # exact physical contact event that the success predicate consumed.
        # A second generic action-provenance record is not authoritative.
        event = result.get("metrics") or {}
        observed_arm = str(event.get("arm") or "")
        arguments = checker_predicate.get("arguments") or {}
        expected_arm = str(arguments.get("requested_arm") or "")
        touch_binding_passed = all(
            (
                str(event.get("arm") or "") == expected_arm,
                str(event.get("subject") or "")
                == str(arguments.get("subject") or ""),
                event.get("contact_id") == arguments.get("contact_id"),
                str(event.get("use") or "")
                == str(arguments.get("use") or ""),
                str(event.get("event") or "")
                == str(arguments.get("event") or ""),
                str(event.get("operation_anchor_id") or "")
                == str(arguments.get("operation_anchor_id") or ""),
            )
        )
        reported_value = (
            (snapshot.get("terminal") or {}).get("action_provenance")
            or snapshot.get("action_provenance")
            or {}
        )
        reported = (
            reported_value if isinstance(reported_value, Mapping) else {}
        )
        reported_fields = {
            "primary_arm": observed_arm,
            "primary_op": "touch_event",
            "use": str(event.get("use") or ""),
            "event": str(event.get("event") or ""),
            "contact_id": event.get("contact_id"),
            "operation_anchor_id": str(
                event.get("operation_anchor_id") or ""
            ),
        }
        event_hash = str(event.get("event_hash") or "")
        reported_event_hash = str(
            reported.get("primary_event_hash") or ""
        )
        event_payload = {
            key: value
            for key, value in event.items()
            if key not in {"checks", "tolerance_m"}
        }
        event_hash_valid = bool(event_hash) and event_hash == artifact_hash(
            event_payload, omit_keys=("event_hash",)
        )
        provenance_consistent = bool(
            event_hash_valid
            and reported_event_hash == event_hash
            and all(
                key not in reported or reported.get(key) == expected
                for key, expected in reported_fields.items()
            )
        )
        action_provenance = {
            "source": "click_press_success_contact_event",
            "contact_event": canonical_data(event),
            "primary_arm": observed_arm or None,
            "event_hash_valid": event_hash_valid,
            "primary_event_hash": event_hash or None,
            "reported_primary_event_hash": reported_event_hash or None,
            "reported_primary_action": canonical_data(reported),
            "reported_primary_action_consistent": provenance_consistent,
        }
    else:
        expected_arm = semantic_expected_arm
        action_provenance = (
            (snapshot.get("terminal") or {}).get("action_provenance")
            or snapshot.get("action_provenance")
            or {}
        )
        observed_arm = str(action_provenance.get("primary_arm") or "")
    provenance_passed = bool(
        touch_binding_passed
        and provenance_consistent
        and (
            not required
            or (
                semantic_expected_arm in {"left", "right"}
                and observed_arm == semantic_expected_arm
                and expected_arm == semantic_expected_arm
            )
        )
    )
    result["action_provenance_gate"] = {
        "required": required,
        "passed": provenance_passed,
        "expected_primary_arm": (
            semantic_expected_arm or expected_arm or None
        ),
        "observed_primary_arm": observed_arm or None,
        "contact_event_binding_passed": touch_binding_passed,
        "reported_primary_action_consistent": provenance_consistent,
        "evidence": canonical_data(action_provenance),
    }
    if result.get("status") != "blocked":
        result["passed"] = bool(result["passed"] and provenance_passed)
        if not provenance_passed:
            result["status"] = "failed"
            failure = (
                "touch_event_provenance"
                if predicate_name == "click_press_success"
                else "primary_execution_arm"
            )
            result.setdefault("failed_predicates", []).append(failure)
    return result


def classify_checker_matches(
    intended_task_id: str,
    checker_results: Mapping[str, Any],
) -> dict[str, Any]:
    """Classify cross-checker terminal matches without renormalization."""

    matches = sorted(
        str(task_id)
        for task_id, result in checker_results.items()
        if bool(result.get("passed") if isinstance(result, Mapping) else result)
    )
    if matches == [intended_task_id]:
        classification = "intended"
    elif not matches:
        classification = "failure"
    elif len(matches) == 1:
        classification = "other"
    else:
        classification = "ambiguous"
    return {
        "classification": classification,
        "intended_task_id": intended_task_id,
        "matched_task_ids": matches,
    }


def resolve_entity_dynamics(joined: Mapping[str, Any]) -> dict[str, str]:
    """Resolve rigid/static/articulated loading once across all scene tasks."""

    scene_entities = joined["scene"]["scene_spec"]["entities"]
    realized = joined["realized"].get("realized_scene_family") or joined["realized"]
    bindings = realized.get("role_bindings") or {}
    articulated: set[str] = set()
    for binding in bindings.values():
        key = str(binding.get("entity_key") or "")
        if key and (
            bool(binding.get("articulated"))
            or str(binding.get("asset_role") or "") == "articulated"
        ):
            articulated.add(key)
    assignment = (realized.get("evidence") or {}).get("assignment") or {}
    for evidence in assignment.get("evidence_refs", []):
        metadata = evidence.get("metadata") or {}
        role = metadata.get("role")
        if (
            metadata.get("asset_role") == "articulated"
            and role in bindings
        ):
            articulated.add(str(bindings[role]["entity_key"]))

    manipulands: set[str] = set()
    for artifact in joined["tasks"]:
        task = artifact.get("task_record") or artifact
        for step in (task.get("skill_config") or {}).get("steps", []):
            op = str(step.get("op") or "")
            if op in {"grasp", "touch_event"} and step.get("subject"):
                manipulands.add(str(step["subject"]))
            if op == "actuate_state" and step.get("subject"):
                articulated.add(str(step["subject"]))
        predicate = (task.get("checker_config") or {}).get("predicate") or {}
        arguments = predicate.get("arguments") or {}
        if (
            predicate.get("name") == "articulation_state_success"
            and arguments.get("subject")
        ):
            articulated.add(str(arguments["subject"]))
    return {
        str(key): (
            "articulated_fixed"
            if key in articulated
            else "dynamic"
            if key in manipulands
            else "static"
        )
        for key in scene_entities
    }


def resolve_reservation_anchor(
    unit_artifact: Mapping[str, Any],
    scene_spec: Mapping[str, Any],
    *,
    relation: str,
    subject: str,
    receiver_top_z: float,
    table_top_z: float | None = None,
) -> list[float] | None:
    """Resolve an M11 terminal reservation center into world coordinates."""

    for unit in unit_artifact.get("units", []):
        params = unit.get("params") or {}
        for region in params.get("reserved_regions", []):
            if (
                str(region.get("relation")) != relation
                or str(region.get("subject_key") or region.get("subject") or subject)
                != subject
            ):
                continue
            center_xy = (region.get("footprint") or {}).get("center_xy")
            if center_xy is None:
                continue
            unit_pose = next(
                (
                    item
                    for item in scene_spec.get("unit_poses", [])
                    if item.get("unit_id") == unit.get("id")
                ),
                None,
            )
            if unit_pose is None:
                raise SceneValidationError(
                    "reservation_unit_pose_missing",
                    f"Unit pose is missing for {unit.get('id')}",
                    stage="source",
                )
            yaw = math.radians(float(unit_pose.get("yaw_deg", 0.0)))
            local_x, local_y = float(center_xy[0]), float(center_xy[1])
            world_x = (
                float(unit_pose["position"][0])
                + math.cos(yaw) * local_x
                - math.sin(yaw) * local_y
            )
            world_y = (
                float(unit_pose["position"][1])
                + math.sin(yaw) * local_x
                + math.cos(yaw) * local_y
            )
            clearance = float(
                region.get(
                    "terminal_clearance_m",
                    region.get("support_clearance_m", 0.015),
                )
            )
            support_top_z = (
                float(receiver_top_z)
                if relation in {"in", "on", "stack"}
                else float(table_top_z)
                if table_top_z is not None
                else float(receiver_top_z)
            )
            subject_bounds = (
                (params.get("subject_z_bounds") or {}).get(subject)
                or [0.0, 0.0]
            )
            subject_bottom_z = float(subject_bounds[0])
            return [
                world_x,
                world_y,
                support_top_z + clearance - subject_bottom_z,
            ]
    return None


def _assignment_validation_refs_for_entity(
    realized_artifact: Mapping[str, Any] | None,
    *,
    entity: str,
    use: str,
) -> list[dict[str, Any]]:
    if not isinstance(realized_artifact, Mapping):
        return []
    realized = (
        realized_artifact.get("realized_scene_family")
        or realized_artifact
    )
    roles = {
        str(role)
        for role, binding in (realized.get("role_bindings") or {}).items()
        if str((binding or {}).get("entity_key") or "") == entity
    }
    refs = (
        ((realized.get("evidence") or {}).get("assignment") or {}).get(
            "evidence_refs"
        )
        or []
    )
    return [
        canonical_data(dict(row))
        for row in refs
        if row.get("kind") == "asset_validation"
        and str(row.get("use") or "") == use
        and str((row.get("metadata") or {}).get("role") or "") in roles
        and row.get("status") == "validated"
    ]


def preflight_skill_operation_anchors(
    skill_config: Mapping[str, Any],
    unit_artifact: Mapping[str, Any],
    realized_artifact: Mapping[str, Any] | None = None,
    scene_spec: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Reject a stale M15 skill before it can issue a physical action."""

    observed_version = str(
        skill_config.get("operation_anchor_policy_version") or ""
    )
    if observed_version != OPERATION_ANCHOR_POLICY_VERSION:
        raise SceneValidationError(
            "stale_operation_anchor_policy",
            f"Skill IR uses {observed_version!r}, expected "
            f"{OPERATION_ANCHOR_POLICY_VERSION!r}",
            stage="source",
        )
    units = list(unit_artifact.get("units", []))
    contracts = {
        str(entity): contract
        for unit in units
        for entity, contract in (
            (unit.get("params") or {}).get("operation_anchors") or {}
        ).items()
    }
    local_poses: dict[str, Mapping[str, Any]] = {}
    for unit in units:
        params = unit.get("params") or {}
        if unit.get("type") == "ObjectUnit":
            entity = str(params.get("object_key") or "")
            pose = params.get("local_pose")
            if entity and isinstance(pose, Mapping):
                local_poses[entity] = pose
        for field in ("internal_layout", "receiver_layout"):
            for entity, pose in (params.get(field) or {}).items():
                if isinstance(pose, Mapping):
                    local_poses[str(entity)] = pose
    entity_keys = {
        str(entity)
        for unit in units
        for entity in unit.get("entity_keys", [])
    }
    reservations = [
        region
        for unit in units
        for region in (unit.get("params") or {}).get(
            "reserved_regions", []
        )
    ]
    checked = []
    for step in skill_config.get("steps", []):
        op = str(step.get("op") or "")
        if op not in {
            "grasp",
            "actuate_state",
            "carry_to_relation",
            "touch_event",
        }:
            continue
        subject = str(step.get("subject") or "")
        anchor_kind = str(step.get("operation_anchor_kind") or "")
        if op == "grasp":
            expected_kinds = {"grasp_contact"}
            available_kinds = {"grasp_contact"}
        elif op in {"actuate_state", "touch_event"}:
            expected_kinds = {"functional_or_contact"}
            available_kinds = {"functional", "grasp_contact"}
        else:
            expected_kinds = {"receiver_reservation"}
            available_kinds = set()
        if anchor_kind not in expected_kinds:
            raise SceneValidationError(
                "skill_operation_anchor_kind_missing",
                f"{op}({subject}) declares {anchor_kind!r}, expected "
                f"{sorted(expected_kinds)!r}",
                stage="source",
            )
        if op == "carry_to_relation":
            receivers = {
                str(step.get(key) or "")
                for key in ("target", "target2")
                if step.get(key) is not None
            }
            matches = [
                region
                for region in reservations
                if str(region.get("subject_key") or "") == subject
                and str(region.get("relation") or "")
                == str(step.get("relation") or "")
                and receivers
                <= {
                    str(value)
                    for value in region.get("receiver_keys", [])
                }
            ]
            if not matches or not receivers <= entity_keys:
                raise SceneValidationError(
                    "skill_operation_anchor_missing",
                    f"{op}({subject}) has no frozen receiver reservation",
                    stage="source",
                )
            checked.append(
                {
                    "op": op,
                    "subject": subject,
                    "anchor_kind": anchor_kind,
                    "anchor_count": len(matches),
                    "reservation_ids": [
                        str(row.get("id") or "") for row in matches
                    ],
                }
            )
            continue
        contract = contracts.get(subject)
        if not isinstance(contract, Mapping):
            raise SceneValidationError(
                "skill_operation_anchor_missing",
                f"{op}({subject}) has no frozen operation-anchor contract",
                stage="source",
            )
        if contract.get("version") != OPERATION_ANCHOR_POLICY_VERSION:
            raise SceneValidationError(
                "stale_operation_anchor_policy",
                f"{op}({subject}) has stale Unit operation-anchor policy",
                stage="source",
            )
        if contract.get("contract_hash") != artifact_hash(
            contract, omit_keys=("contract_hash",)
        ):
            raise SceneValidationError(
                "stale_operation_anchor_contract",
                f"{op}({subject}) has an invalid Unit operation-anchor hash",
                stage="source",
            )
        source = contract.get("source") or {}
        if source.get("operation_metadata_hash") != contract.get(
            "operation_metadata_hash"
        ):
            raise SceneValidationError(
                "stale_operation_anchor_source",
                f"{op}({subject}) operation metadata provenance is stale",
                stage="source",
            )
        local_pose = local_poses.get(subject) or {}
        expected_state_hash = str(
            local_pose.get("canonical_state_hash") or ""
        )
        if expected_state_hash and (
            contract.get("canonical_state_hash") != expected_state_hash
            or source.get("canonical_state_hash") != expected_state_hash
        ):
            raise SceneValidationError(
                "stale_operation_anchor_state",
                f"{op}({subject}) operation anchors do not bind the "
                "frozen canonical state",
                stage="source",
            )
        expected_revision = local_pose.get("assignment_state_revision")
        if expected_revision and canonical_data(
            source.get("assignment_state_revision") or {}
        ) != canonical_data(expected_revision):
            raise SceneValidationError(
                "stale_operation_anchor_state_revision",
                f"{op}({subject}) operation anchors do not bind the "
                "assignment state revision",
                stage="source",
            )
        anchors = [
            row
            for row in contract.get("anchors", [])
            if row.get("kind") in available_kinds
            and row.get("operation_eligible", True) is True
        ]
        if not anchors:
            raise SceneValidationError(
                "skill_operation_anchor_missing",
                f"{op}({subject}) has no frozen {anchor_kind} in its Unit",
                stage="source",
            )
        lineage: dict[str, Any] = {}
        if op == "grasp":
            source_rows = list(
                source.get("validated_contact_contracts") or []
            )
            arm = str(step.get("arm") or "")
            expected_use = {
                "left": "pick_left",
                "right": "pick_right",
            }.get(arm)
            if expected_use is None:
                raise SceneValidationError(
                    "stale_contact_contract_arm",
                    f"grasp({subject}) has invalid arm {arm!r}",
                    stage="source",
                )
            m10_rows = _assignment_validation_refs_for_entity(
                realized_artifact,
                entity=subject,
                use=expected_use,
            )
            m11_rows = [
                row
                for row in source_rows
                if str(row.get("use") or "") == expected_use
            ]
            if len(m10_rows) != 1 or len(m11_rows) != 1:
                raise SceneValidationError(
                    "stale_contact_contract_arm",
                    f"grasp({subject}) lacks unique {expected_use} "
                    "M4 lineage",
                    stage="source",
                )
            m10 = m10_rows[0]
            m11 = m11_rows[0]
            binding = (m10.get("metadata") or {}).get(
                "contact_contract_binding"
            )
            if (
                not isinstance(binding, Mapping)
                or canonical_data(m11.get("binding") or {})
                != canonical_data(binding)
                or m11.get("evidence_id") != m10.get("record_id")
                or m11.get("report_path") != m10.get("path")
                or m11.get("report_hash") != m10.get("hash")
                or contract.get("contact_projection_hash")
                != binding.get("contact_projection_hash")
                or source.get("contact_projection_hash")
                != binding.get("contact_projection_hash")
            ):
                raise SceneValidationError(
                    "stale_contact_contract_evidence",
                    f"grasp({subject}) M4/M10/M11 Contact lineage "
                    "does not match",
                    stage="source",
                )
            lineage["contact_contract_binding"] = canonical_data(
                dict(binding)
            )
            lineage["contact_validation_use"] = expected_use
        elif op == "actuate_state":
            expected_use = {
                "open": "open",
                "close": "close",
                "on": "switch_on",
                "off": "switch_off",
                "switch_on": "switch_on",
                "switch_off": "switch_off",
            }.get(str(step.get("state") or ""))
            source_rows = list(
                source.get("validated_articulation_executions") or []
            )
            if expected_use is None:
                raise SceneValidationError(
                    "stale_articulation_execution_evidence",
                    f"actuate_state({subject}) has unsupported state "
                    f"{step.get('state')!r}",
                    stage="source",
                )
            m10_rows = _assignment_validation_refs_for_entity(
                realized_artifact,
                entity=subject,
                use=expected_use,
            )
            m11_rows = [
                row
                for row in source_rows
                if str(row.get("use") or "") == expected_use
            ]
            if len(m10_rows) != 1 or len(m11_rows) != 1:
                raise SceneValidationError(
                    "stale_articulation_execution_evidence",
                    f"actuate_state({subject}) lacks unique M4 "
                    "execution-profile lineage",
                    stage="source",
                )
            m10 = m10_rows[0]
            m11 = m11_rows[0]
            binding = (m10.get("metadata") or {}).get(
                "articulation_execution_binding"
            )
            if (
                not isinstance(binding, Mapping)
                or canonical_data(m11.get("binding") or {})
                != canonical_data(binding)
                or m11.get("evidence_id") != m10.get("record_id")
                or m11.get("report_path") != m10.get("path")
                or m11.get("report_hash") != m10.get("hash")
            ):
                raise SceneValidationError(
                    "stale_articulation_execution_evidence",
                    f"actuate_state({subject}) M4/M10/M11 profile "
                    "lineage does not match",
                    stage="source",
                )
            arm = str(step.get("arm") or "")
            arm_capability = validated_articulation_arm_capability(
                binding,
                use=expected_use,
                arm=arm,
            )
            skill_capability = step.get("articulation_capability") or {}
            expected_skill_capability = {
                "capability_version": ARTICULATION_CAPABILITY_VERSION,
                "use": expected_use,
                "arm_capability": arm_capability,
                "evidence": "m11_exact_m4_articulation_capability_v1",
            }
            if (
                arm_capability is None
                or canonical_data(skill_capability)
                != canonical_data(expected_skill_capability)
            ):
                raise SceneValidationError(
                    "stale_articulation_arm_capability",
                    f"actuate_state({subject}) arm {arm!r} lacks exact "
                    f"M4 {expected_use} capability",
                    stage="source",
                )
            lineage["articulation_execution_binding"] = canonical_data(
                dict(binding)
            )
            lineage["articulation_arm_capability"] = canonical_data(
                dict(arm_capability)
            )
            lineage["articulation_validation_use"] = expected_use
        elif op == "touch_event":
            expected_use = str(step.get("use") or "")
            expected_event = TOUCH_EVENT_BY_USE.get(expected_use)
            if (
                expected_use not in TOUCH_EXECUTABLE_USES
                or str(step.get("event") or "") != expected_event
            ):
                raise SceneValidationError(
                    "stale_touch_operation_binding",
                    f"touch_event({subject}) has invalid use/event binding",
                    stage="source",
                )
            source_rows = list(
                source.get("validated_touch_executions") or []
            )
            m10_rows = _assignment_validation_refs_for_entity(
                realized_artifact,
                entity=subject,
                use=expected_use,
            )
            m11_rows = [
                row
                for row in source_rows
                if str(row.get("use") or "") == expected_use
            ]
            if len(m10_rows) != 1 or len(m11_rows) != 1:
                raise SceneValidationError(
                    "stale_touch_execution_evidence",
                    f"touch_event({subject}) lacks unique M4 {expected_use} "
                    "lineage",
                    stage="source",
                )
            m10 = m10_rows[0]
            m11 = m11_rows[0]
            binding = (m10.get("metadata") or {}).get(
                "touch_execution_binding"
            )
            if (
                not isinstance(binding, Mapping)
                or canonical_data(m11.get("binding") or {})
                != canonical_data(binding)
                or m11.get("evidence_id") != m10.get("record_id")
                or m11.get("report_path") != m10.get("path")
                or m11.get("report_hash") != m10.get("hash")
            ):
                raise SceneValidationError(
                    "stale_touch_execution_evidence",
                    f"touch_event({subject}) M4/M10/M11 lineage does not "
                    "match",
                    stage="source",
                )
            arm = str(step.get("arm") or "")
            arm_capability = validated_touch_arm_capability(
                binding,
                use=expected_use,
                arm=arm,
            )
            expected_skill_capability = {
                "capability_version": TOUCH_CAPABILITY_VERSION,
                "use": expected_use,
                "arm_capability": arm_capability,
                "evidence": "m11_exact_m4_touch_capability_v1",
            }
            if (
                arm_capability is None
                or canonical_data(step.get("touch_capability") or {})
                != canonical_data(expected_skill_capability)
            ):
                raise SceneValidationError(
                    "stale_touch_arm_capability",
                    f"touch_event({subject}) arm {arm!r} lacks exact M4 "
                    f"{expected_use} capability",
                    stage="source",
                )
            contact_id = int(arm_capability["contact_id"])
            observed_contact_id = step.get("contact_id")
            exact_anchor_id = f"grasp_contact:{contact_id}"
            exact_anchors = [
                row
                for row in anchors
                if row.get("kind") == "grasp_contact"
                and row.get("point_index") == contact_id
                and str(row.get("id") or "") == exact_anchor_id
            ]
            if (
                not isinstance(observed_contact_id, int)
                or isinstance(observed_contact_id, bool)
                or observed_contact_id != contact_id
                or str(step.get("operation_anchor_id") or "")
                != exact_anchor_id
                or len(exact_anchors) != 1
            ):
                raise SceneValidationError(
                    "stale_touch_operation_binding",
                    f"touch_event({subject}) does not bind exact eligible "
                    f"Contact {contact_id} ({exact_anchor_id})",
                    stage="source",
                )
            if scene_spec is not None:
                reach = (scene_spec.get("reach") or {}).get(subject) or {}
                group_id = (
                    f"touch_reach:{subject}:{expected_use}:{arm}"
                )
                reach_groups = [
                    row
                    for row in reach.get("touch_operation_groups") or []
                    if str(row.get("id") or "") == group_id
                ]
                exact_reach_options = [
                    option
                    for row in reach_groups
                    for option in row.get("options") or []
                    if str(option.get("arm") or "") == arm
                    and option.get("contact_id") == contact_id
                    and option.get("reachable") is True
                ]
                if (
                    len(reach_groups) != 1
                    or reach_groups[0].get("passed") is not True
                    or str(reach_groups[0].get("use") or "")
                    != expected_use
                    or len(exact_reach_options) != 1
                ):
                    raise SceneValidationError(
                        "stale_touch_scene_reach_evidence",
                        f"touch_event({subject}) lacks exact passing M12 "
                        f"reach group {group_id}",
                        stage="source",
                    )
            lineage["touch_execution_binding"] = canonical_data(
                dict(binding)
            )
            lineage["touch_arm_capability"] = canonical_data(
                dict(arm_capability)
            )
            lineage["touch_validation_use"] = expected_use
            lineage["touch_event"] = expected_event
            lineage["touch_arm"] = arm
            lineage["touch_contact_id"] = contact_id
            lineage["operation_anchor_id"] = exact_anchor_id
            if scene_spec is not None:
                lineage["touch_operation_group_id"] = group_id
        checked.append(
            {
                "op": op,
                "subject": subject,
                "anchor_kind": anchor_kind,
                "anchor_count": len(anchors),
                "contract_hash": contract.get("contract_hash"),
                **lineage,
            }
        )
    return {
        "status": "passed",
        "policy_version": OPERATION_ANCHOR_POLICY_VERSION,
        "checks": checked,
    }


def dispatch_skill_config(
    skill_config: Mapping[str, Any],
    dispatch: Callable[[str, Mapping[str, Any]], None],
) -> list[str]:
    """Validate and dispatch M13 Skill IR without parsing compatibility text."""

    dispatched: list[str] = []
    for step in skill_config.get("steps", []):
        op = str(step.get("op") or "")
        if op not in SKILL_OPERATORS:
            raise SceneValidationError(
                "unsupported_skill_operator",
                f"unsupported skill operator {op!r}",
                stage="internal",
            )
        dispatch(op, step)
        dispatched.append(op)
    if not dispatched:
        raise SceneValidationError(
            "empty_skill_config",
            "skill config contains no steps",
            stage="internal",
        )
    return dispatched


def _family_composition(path: Path) -> dict[str, tuple[str, ...]]:
    library = _load_yaml(path)
    result: dict[str, tuple[str, ...]] = {}
    for family_id, value in (library.get("composition_by_family") or {}).items():
        canonical_axes = {
            "d1a": "D1a",
            "d1b": "D1b",
            "d1c": "D1c",
            "d2a": "D2a",
            "d2b": "D2b",
            "d4": "D4",
        }
        axes = tuple(
            sorted(
                {
                    canonical_axes[str(fragment).lower()]
                    for fragment in value.get("fragment_ids", [])
                    if str(fragment).lower() in canonical_axes
                }
            )
        )
        result[str(family_id)] = axes
    return result


def _realized_ids_for_family(
    realization_dir: Path,
    abstract_family_id: str,
) -> set[str]:
    result: set[str] = set()
    for path in sorted(realization_dir.glob("rsf_*.yml")):
        family_id = None
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("abstract_family_id:"):
                    family_id = line.split(":", 1)[1].strip()
                    break
        if family_id == abstract_family_id:
            result.add(path.stem)
    return result


def _apply_coverage_waivers(
    *,
    config: ValidationConfig,
    scope: str,
    required: set[str],
    uncovered: set[str],
    realization_dir: Path,
) -> tuple[
    set[str],
    list[dict[str, Any]],
    dict[str, dict[str, str]],
]:
    """Apply only explicit, source-backed waivers to otherwise blocking gaps."""

    waived: set[str] = set()
    records: list[dict[str, Any]] = []
    reports: dict[str, dict[str, str]] = {}
    for raw in config.data.get("coverage_waivers", []):
        if scope not in {str(item) for item in raw.get("scopes", [])}:
            continue
        waiver_id = str(raw["id"])
        tokens = {str(item) for item in raw["tokens"]}
        unknown = tokens - required
        if unknown:
            raise SceneValidationError(
                "coverage_waiver_unknown_tokens",
                f"{waiver_id} names tokens outside the required universe: "
                + ", ".join(sorted(unknown)),
                stage="plan",
            )
        applied = tokens & uncovered
        if not applied:
            continue

        source = raw.get("source_report") or {}
        report_path = Path(str(source.get("path") or ""))
        if not report_path.is_file():
            raise SceneValidationError(
                "coverage_waiver_missing_evidence",
                f"{waiver_id} source report is missing",
                stage="plan",
                path=str(report_path),
            )
        report = _load_yaml(report_path)
        if (
            report.get("artifact_kind") != source.get("artifact_kind")
            or not _report_hash_valid(report)
            or report.get("report_hash") != source.get("report_hash")
            or not bool(report.get("implementation_complete"))
            or not bool(report.get("publication_ready"))
        ):
            raise SceneValidationError(
                "coverage_waiver_stale_evidence",
                f"{waiver_id} source report is stale or not publishable",
                stage="plan",
                path=str(report_path),
            )

        expected = canonical_data(dict(raw.get("evidence") or {}))
        family_id = str(expected.get("abstract_family_id") or "")
        realized_ids = _realized_ids_for_family(realization_dir, family_id)
        family_failures = [
            item
            for item in report.get("failures", [])
            if str(item.get("realized_family_id") or "") in realized_ids
        ]
        failure_ids = {
            str(item.get("realized_family_id") or "")
            for item in family_failures
        }
        observed = {
            "abstract_family_id": family_id,
            "realized_family_count": len(realized_ids),
            "unsat_count": len(family_failures),
            "failure_code_counts": dict(
                sorted(Counter(item.get("code") for item in family_failures).items())
            ),
        }
        if (
            not family_id
            or not realized_ids
            or failure_ids != realized_ids
            or len(family_failures) != len(failure_ids)
            or any(
                not bool(item.get("geometric_unsat"))
                or item.get("owner") != "layout"
                for item in family_failures
            )
            or observed != expected
        ):
            raise SceneValidationError(
                "coverage_waiver_evidence_mismatch",
                f"{waiver_id} evidence no longer proves a fully audited "
                f"geometric-unsat family: expected {expected}, observed {observed}",
                stage="plan",
                path=str(report_path),
            )

        waived.update(applied)
        records.append(
            {
                "id": waiver_id,
                "reason_code": str(raw.get("reason_code") or ""),
                "reason": str(raw.get("reason") or ""),
                "scopes": sorted(str(item) for item in raw.get("scopes", [])),
                "tokens": sorted(tokens),
                "applied_tokens": sorted(applied),
                "source_report": {
                    "path": str(report_path),
                    "artifact_kind": str(report["artifact_kind"]),
                    "report_hash": str(report["report_hash"]),
                    "config_hash": str(report.get("config_hash") or ""),
                },
                "evidence": observed,
            }
        )
        reports[f"coverage_waiver_{waiver_id}"] = {
            "path": str(report_path),
            "report_hash": str(report["report_hash"]),
        }
    return waived, canonical_data(records), canonical_data(reports)


def _task_tokens(task_artifact: Mapping[str, Any]) -> set[str]:
    task = task_artifact.get("task_record") or task_artifact
    tokens: set[str] = set()
    goal = task.get("goal") or {}
    if goal.get("verb"):
        tokens.add(f"verb:{goal['verb']}")
    if goal.get("relation"):
        tokens.add(f"relation:{goal['relation']}")
    provenance = task.get("goal_provenance") or {}
    for axis in provenance.get("intended_axes") or []:
        if axis.get("subtype"):
            tokens.add(f"descriptor_subtype:{axis['subtype']}")
    checker = task.get("checker_config") or {}
    predicate = checker.get("predicate") or {}
    if predicate.get("name"):
        tokens.add(f"checker_operator:{predicate['name']}")
    if checker.get("scene_integrity"):
        tokens.add("checker_guard:scene_integrity")
    skill = task.get("skill_config") or {}
    if skill.get("route"):
        tokens.add(f"route:{skill['route']}")
    for step in skill.get("steps") or []:
        tokens.add(f"skill_op:{step.get('op')}")
    return tokens


def _candidate_scene_paths(
    scene_spec_dir: Path,
    realization_dir: Path,
    family_ids: Iterable[str],
    *,
    candidates_per_family: int | None = 24,
) -> list[Path]:
    """Return published SceneSpecs without collapsing a family's layouts.

    A full M15 run consumes every solved M12 SceneSpec.  Pilot discovery is
    bounded per abstract family and is ordered in two round-robin layers:
    first across realized families at the same successful-layout rank, then
    across abstract families.  This preserves assignment breadth before using
    a second layout of the same realization and prevents early-sorted families
    from monopolizing the candidate stream.
    """

    if candidates_per_family is not None and candidates_per_family < 1:
        raise ValueError("candidates_per_family must be positive or None")

    def layout_path_key(path: Path) -> tuple[int, int, str]:
        try:
            return (0, int(path.stem), path.name)
        except ValueError:
            return (1, 0, path.name)

    wanted = set(family_ids)
    realized_buckets: dict[str, dict[str, list[Path]]] = defaultdict(dict)
    for realization_path in sorted(realization_dir.glob("rsf_*.yml")):
        family_id = None
        realized_id = realization_path.stem
        with realization_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("abstract_family_id:"):
                    family_id = line.split(":", 1)[1].strip()
                    break
        if family_id not in wanted:
            continue
        scene_paths = sorted(
            (scene_spec_dir / realized_id).glob("*.yml"),
            key=layout_path_key,
        )
        if scene_paths:
            realized_buckets[family_id][realized_id] = scene_paths

    family_buckets: dict[str, list[Path]] = {}
    for family_id, realized_rows in sorted(realized_buckets.items()):
        ordered: list[Path] = []
        maximum_rank = max(len(paths) for paths in realized_rows.values())
        for layout_rank in range(maximum_rank):
            for realized_id in sorted(realized_rows):
                paths = realized_rows[realized_id]
                if layout_rank < len(paths):
                    ordered.append(paths[layout_rank])
                    if (
                        candidates_per_family is not None
                        and len(ordered) >= candidates_per_family
                    ):
                        break
            if (
                candidates_per_family is not None
                and len(ordered) >= candidates_per_family
            ):
                break
        family_buckets[family_id] = ordered

    result: list[Path] = []
    maximum_family_count = max(
        (len(paths) for paths in family_buckets.values()), default=0
    )
    for candidate_rank in range(maximum_family_count):
        for family_id in sorted(family_buckets):
            paths = family_buckets[family_id]
            if candidate_rank < len(paths):
                result.append(paths[candidate_rank])
    return result


def _plan_candidate(
    scene_path: Path,
    *,
    realization_dir: Path,
    unit_dir: Path,
    task_dir: Path,
    route_dir: Path,
    config_hash: str,
    composition: Mapping[str, tuple[str, ...]],
) -> dict[str, Any] | None:
    joined = strict_source_join(
        scene_path,
        realization_dir=realization_dir,
        unit_dir=unit_dir,
        task_dir=task_dir,
        route_dir=route_dir,
    )
    if not joined["tasks"]:
        return None
    stale_routes = sorted(
        task_id
        for task_id, route in joined["routes"].items()
        if str(route.get("scene_validation_config_hash") or "")
        != str(config_hash)
    )
    if stale_routes:
        raise _source_error(
            "stale_route_config",
            f"route witnesses use a stale M15 config: {stale_routes}",
            path=scene_path,
            scene_spec_id=str(joined["scene"]["scene_spec"]["id"]),
        )
    scene = joined["scene"]
    realized = joined["realized"]
    scene_id = str(scene["scene_spec"]["id"])
    family_id = str(realized["abstract_family_id"])
    axes = composition.get(family_id, ())
    tokens = {f"abstract_family:{family_id}", f"fusion_signature:{'+'.join(axes)}"}
    for task in joined["tasks"]:
        tokens.update(_task_tokens(task))
    return {
        "scene_spec_id": scene_id,
        "realized_family_id": scene["realized_family_id"],
        "abstract_family_id": family_id,
        "source_scene_hash": scene["artifact_hash"],
        "source_realization_hash": realized["artifact_hash"],
        "source_unit_hash": joined["unit"]["artifact_hash"],
        "task_records": [
            {
                "task_record_id": task["task_record"]["id"],
                "goal_variant_id": task["goal_variant_id"],
                "source_task_hash": task["artifact_hash"],
            }
            for task in joined["tasks"]
        ],
        "route_records": [
            {
                "task_record_id": task_id,
                "report_hash": route["report_hash"],
                "witness_hash": route["selected_witness"]["witness_hash"],
            }
            for task_id, route in sorted(joined["routes"].items())
        ],
        "missing_goal_variant_ids": joined["missing_goal_variant_ids"],
        "coverage_tokens": sorted(tokens),
        "scene_path": str(scene_path),
    }


def _checked_plan_candidate(path: Path, **kwargs):
    try:
        return _plan_candidate(path, **kwargs), None
    except SceneValidationError as exc:
        # Explicit attributed errors are ordinary results, not missing scenes.
        return None, exc.to_record()


def build_validation_plan(
    *,
    scope: str = "pilot",
    config_path: Path = DEFAULT_CONFIG_PATH,
    family_library_path: Path = DEFAULT_FAMILY_LIBRARY,
    realization_dir: Path = DEFAULT_REALIZATION_DIR,
    unit_dir: Path = DEFAULT_UNIT_DIR,
    scene_spec_dir: Path = DEFAULT_SCENE_SPEC_DIR,
    task_dir: Path = DEFAULT_TASK_DIR,
    route_dir: Path = DEFAULT_ROUTE_DIR,
    output_dir: Path | None = None,
    max_scene_specs: int | None = None,
    candidates: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a deterministic set-cover plan for a pilot or full run."""

    if scope not in {"pilot", "full"}:
        raise ValueError("scope must be 'pilot' or 'full'")
    config = ValidationConfig.load(config_path)
    composition = _family_composition(family_library_path)
    if not composition:
        raise SceneValidationError(
            "empty_family_library",
            "no abstract family composition exists",
            stage="source",
            path=str(family_library_path),
        )
    limit = (
        int(max_scene_specs)
        if max_scene_specs is not None
        else int(config.data["pilot"]["maximum_scene_specs"])
    )
    source_failures: list[SceneValidationError] = []
    accounted_no_task_count = 0
    source_reports: dict[str, dict[str, str]] = {}

    if candidates is None:
        _, m13_source_reports = _m13_exclusion_ledger(task_dir)
        source_reports.update(m13_source_reports)
        paths = _candidate_scene_paths(
            scene_spec_dir,
            realization_dir,
            composition,
            candidates_per_family=(
                int(config.data["pilot"]["candidates_per_family"])
                if scope == "pilot"
                else None
            ),
        )
        from .validation_parallel import ordered_validation_map
        candidate_rows = []
        check_candidate = partial(_checked_plan_candidate,
            realization_dir=realization_dir, unit_dir=unit_dir, task_dir=task_dir,
            route_dir=route_dir, config_hash=config.hash, composition=composition)
        for row, failure in ordered_validation_map(check_candidate, paths):
            if failure is not None:
                source_failures.append(SceneValidationError(**failure))
                continue
            if row is None:
                accounted_no_task_count += 1
                continue
            candidate_rows.append(row)
    else:
        candidate_rows = [canonical_data(dict(item)) for item in candidates]

    required = {
        *(f"abstract_family:{family_id}" for family_id in composition),
        *(f"fusion_signature:{'+'.join(axes)}" for axes in set(composition.values())),
        *(f"descriptor_subtype:{item}" for item in DESCRIPTOR_SUBTYPES),
        *(f"checker_operator:{item}" for item in CHECKER_OPERATORS),
        *(f"checker_guard:{item}" for item in CHECKER_GUARDS),
        *(f"skill_op:{item}" for item in SKILL_OPERATORS),
        *(f"route:{item}" for item in ROUTES),
    }
    # Verb and relation universes are derived from legal, rendered M13 tasks.
    # Unlike fixed operators, this avoids introducing a legacy vocabulary.
    for row in candidate_rows:
        required.update(
            token
            for token in row.get("coverage_tokens", [])
            if str(token).startswith(("verb:", "relation:"))
        )

    selected: list[dict[str, Any]] = []
    uncovered = set(required)
    remaining = sorted(candidate_rows, key=lambda item: str(item["scene_spec_id"]))
    while scope == "pilot" and remaining and uncovered and len(selected) < limit:
        ranked = sorted(
            remaining,
            key=lambda item: (
                -len(set(item.get("coverage_tokens", [])) & uncovered),
                str(item["scene_spec_id"]),
            ),
        )
        best = ranked[0]
        gain = set(best.get("coverage_tokens", [])) & uncovered
        if not gain:
            break
        selected.append(best)
        uncovered -= gain
        remaining = [
            item
            for item in remaining
            if item["scene_spec_id"] != best["scene_spec_id"]
        ]

    if scope == "full":
        selected = sorted(candidate_rows, key=lambda item: str(item["scene_spec_id"]))
        uncovered = required - {
            token
            for row in selected
            for token in row.get("coverage_tokens", [])
        }

    waived, waiver_records, waiver_reports = _apply_coverage_waivers(
        config=config,
        scope=scope,
        required=required,
        uncovered=uncovered,
        realization_dir=realization_dir,
    )
    source_reports.update(waiver_reports)
    blocking_uncovered = uncovered - waived

    plan: dict[str, Any] = {
        "artifact_kind": "scene_validation_plan",
        "schema_version": SCHEMA_VERSION,
        "scene_validation_version": SCENE_VALIDATION_VERSION,
        "plan_version": PLAN_VERSION,
        "scope": scope,
        "calibration_only": scope == "pilot",
        "publication_ready": False,
        "config_hash": config.hash,
        "selection_policy": "deterministic_greedy_set_cover_with_waivers_v2",
        "maximum_scene_specs": limit if scope == "pilot" else None,
        "source_roots": {
            "realization_dir": str(realization_dir),
            "unit_dir": str(unit_dir),
            "scene_spec_dir": str(scene_spec_dir),
            "task_dir": str(task_dir),
            "route_dir": str(route_dir),
            "config_path": str(config_path),
            "family_library_path": str(family_library_path),
        },
        "source_reports": source_reports,
        "coverage_waivers": waiver_records,
        "coverage": {
            "required_tokens": sorted(required),
            "covered_tokens": sorted(required - uncovered),
            "uncovered_tokens": sorted(uncovered),
            "waived_tokens": sorted(waived),
            "blocking_uncovered_tokens": sorted(blocking_uncovered),
            "token_complete": not uncovered,
            "gate_complete": not blocking_uncovered,
            "source_join_complete": not source_failures,
            "complete": not blocking_uncovered and not source_failures,
        },
        "source_failures": [
            item.to_record()
            for item in sorted(
                source_failures,
                key=lambda item: (
                    item.scene_spec_id or "",
                    item.task_record_id or "",
                    item.code,
                ),
            )
        ],
        "summary": {
            "accounted_no_task_scene_count": accounted_no_task_count,
            "candidate_scene_count": len(candidate_rows),
            "source_failure_count": len(source_failures),
            "waived_token_count": len(waived),
            "selected_scene_count": len(selected),
            "selected_task_count": sum(len(row["task_records"]) for row in selected),
            "abstract_family_count": len(
                {
                    row["abstract_family_id"]
                    for row in selected
                }
            ),
            "fusion_signature_count": len(
                {
                    token
                    for row in selected
                    for token in row["coverage_tokens"]
                    if token.startswith("fusion_signature:")
                }
            ),
        },
        "scenes": canonical_data(selected),
        "provenance": build_provenance(
            generator="if_benchmark.scene_validation.plan",
            config_hash=config.hash,
            sources=[
                {
                    "kind": "abstract_family_library",
                    "record_id": family_library_path.name,
                    "hash": _sha256_file(family_library_path),
                }
            ],
            metadata={"scope": scope, "plan_version": PLAN_VERSION},
        ),
    }
    plan["plan_id"] = deterministic_id(
        "svp",
        {
            "scope": scope,
            "config_hash": config.hash,
            **SUCCESS_CONTRACT.identity,
            "scene_hashes": [row["source_scene_hash"] for row in selected],
        },
    )
    plan["artifact_hash"] = artifact_hash(plan)
    if output_dir is not None:
        _write_yaml(output_dir / PLAN_FILE, plan)
    if source_failures:
        details = "; ".join(str(item) for item in source_failures[:20])
        if len(source_failures) > 20:
            details += f"; ... {len(source_failures) - 20} more"
        coverage_detail = (
            "; blocking uncovered tokens: "
            + ", ".join(sorted(blocking_uncovered))
            if blocking_uncovered
            else ""
        )
        raise SceneValidationError(
            "blocking_source_failures",
            f"{len(source_failures)} candidate SceneSpec joins failed: "
            f"{details}{coverage_detail}; blocked plan_id={plan['plan_id']} "
            f"artifact_hash={plan['artifact_hash']}",
            stage="source",
        )
    if blocking_uncovered:
        raise SceneValidationError(
            "pilot_coverage_incomplete",
            "pilot plan cannot cover: "
            + ", ".join(sorted(blocking_uncovered)),
            stage="plan",
        )
    return canonical_data(plan)


def stable_attempt_id(
    plan: Mapping[str, Any],
    scene_spec_id: str,
    task_record_id: str,
    seed: int,
) -> str:
    from .execution_binding import execution_plan_hash

    return deterministic_id(
        "sva",
        {
            "plan_hash": execution_plan_hash(plan, scene_spec_id),
            "config_hash": plan.get("config_hash"),
            "scene_spec_id": scene_spec_id,
            "task_record_id": task_record_id,
            "seed": int(seed),
        },
    )


def _verify_plan_sources(plan: Mapping[str, Any]) -> list[str]:
    from .execution_binding import verify_execution_bindings

    binding_errors = verify_execution_bindings(plan)
    if binding_errors:
        return binding_errors
    roots = plan.get("source_roots")
    if not isinstance(roots, Mapping):
        # In-memory unit-test plans may omit roots. Every planner-produced
        # artifact includes them.
        return []
    errors: list[str] = []
    family_path = Path(str(roots.get("family_library_path") or ""))
    family_source = next(
        (
            item
            for item in (plan.get("provenance") or {}).get("sources", [])
            if item.get("kind") == "abstract_family_library"
        ),
        None,
    )
    if (
        not family_path.is_file()
        or family_source is None
        or _sha256_file(family_path) != family_source.get("hash")
    ):
        errors.append(f"{family_path}: plan family library source is stale")
    for report in (plan.get("source_reports") or {}).values():
        path = Path(str(report.get("path") or ""))
        try:
            value = _load_yaml(path)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            errors.append(f"{path}: {exc}")
            continue
        if (
            not _report_hash_valid(value)
            or value.get("report_hash") != report.get("report_hash")
        ):
            errors.append(f"{path}: plan source report is stale")
    from .validation_parallel import ordered_validation_map
    verify_source = partial(_verify_scene_source, roots=roots)
    for scene_errors in ordered_validation_map(verify_source, plan.get("scenes", [])):
        errors.extend(scene_errors)
    return errors


def _verify_scene_source(source: Mapping[str, Any], *, roots: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    scene_path = Path(str(source.get("scene_path") or ""))
    try:
        joined = strict_source_join(
            scene_path,
            realization_dir=Path(str(roots["realization_dir"])),
            unit_dir=Path(str(roots["unit_dir"])),
            task_dir=Path(str(roots["task_dir"])),
            route_dir=Path(str(roots["route_dir"])),
        )
    except (SceneValidationError, OSError, ValueError, yaml.YAMLError) as exc:
        errors.append(f"{scene_path}: {exc}")
        return errors
    current_tasks = {
        str(item["task_record"]["id"]): str(item["artifact_hash"])
        for item in joined["tasks"]
    }
    planned_tasks = {
        str(item["task_record_id"]): str(item["source_task_hash"])
        for item in source.get("task_records", [])
    }
    current_routes = {
        str(task_id): {
            "report_hash": str(route["report_hash"]),
            "witness_hash": str(
                route["selected_witness"]["witness_hash"]
            ),
        }
        for task_id, route in joined["routes"].items()
    }
    planned_routes = {
        str(item["task_record_id"]): {
            "report_hash": str(item["report_hash"]),
            "witness_hash": str(item["witness_hash"]),
        }
        for item in source.get("route_records", [])
    }
    mismatches = (
        joined["scene"].get("artifact_hash") != source.get("source_scene_hash")
        or joined["realized"].get("artifact_hash")
        != source.get("source_realization_hash")
        or joined["unit"].get("artifact_hash") != source.get("source_unit_hash")
        or current_tasks != planned_tasks
        or current_routes != planned_routes
    )
    if mismatches:
        errors.append(f"{scene_path}: plan source hashes/task set are stale")
    return errors


def _load_attempts(path: Path) -> list[dict[str, Any]]:
    return list(_iter_attempts(path))


def _iter_attempts(path: Path):
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SceneValidationError(
                    "invalid_attempt_checkpoint",
                    f"invalid JSONL line {line_number}: {exc}",
                    stage="source",
                    path=str(path),
                ) from exc
            if not isinstance(item, dict):
                raise SceneValidationError(
                    "invalid_attempt_checkpoint",
                    f"attempt line {line_number} is not a mapping",
                    stage="source",
                    path=str(path),
                )
            yield item


def _attempt_index(
    plan: Mapping[str, Any],
    attempts: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    *,
    for_freeze: bool = False,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    from .execution_binding import execution_plan_hash

    expected: dict[str, tuple[str, str, int]] = {}
    for scene in plan.get("scenes", []):
        scene_id = str(scene["scene_spec_id"])
        for task in scene.get("task_records", []):
            task_id = str(task["task_record_id"])
            for seed in seeds:
                attempt_id = stable_attempt_id(plan, scene_id, task_id, seed)
                expected[attempt_id] = (scene_id, task_id, int(seed))
    index: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    for raw in attempts:
        item = dict(raw)
        attempt_id = str(item.get("attempt_id") or "")
        if item.get("attempt_hash") != artifact_hash(
            item, omit_keys=("attempt_hash",)
        ):
            failures.append(
                {"code": "attempt_self_hash_mismatch", "attempt_id": attempt_id}
            )
            continue
        if attempt_id in index:
            failures.append({"code": "duplicate_attempt", "attempt_id": attempt_id})
            continue
        if attempt_id not in expected:
            failures.append({"code": "unexpected_attempt", "attempt_id": attempt_id})
            continue
        scene_id, task_id, seed = expected[attempt_id]
        if (
            item.get("plan_hash") != execution_plan_hash(plan, scene_id)
            or item.get("config_hash") != plan.get("config_hash")
            or item.get("scene_spec_id") != scene_id
            or item.get("task_record_id") != task_id
            or int(item.get("seed", -1)) != seed
        ):
            failures.append({"code": "stale_attempt", "attempt_id": attempt_id})
            continue
        if for_freeze:
            # Hash/identity/duplicate checks above always consume the complete
            # original row. Physical traces remain in immutable JSONL evidence;
            # aggregation retains only the exact fields it actually consumes.
            fields = ('attempt_id', 'scene_spec_id', 'task_record_id', 'seed',
                      'status', 'classification', 'failure_code', 'failure_stage',
                      'skill_completed', 'stable', 'success_snapshot_hash',
                      'descriptor_gate', 't0_checker', 'evidence')
            retained = {key: item[key] for key in fields if key in item}
            checker = (item.get('terminal_checkers') or {}).get(task_id)
            if checker is not None:
                retained['terminal_checkers'] = {task_id: checker}
            index[attempt_id] = canonical_data(retained)
        else:
            index[attempt_id] = canonical_data(item)
    for attempt_id, (scene_id, task_id, seed) in expected.items():
        if attempt_id not in index:
            failures.append(
                {
                    "code": "missing_attempt",
                    "attempt_id": attempt_id,
                    "scene_spec_id": scene_id,
                    "task_record_id": task_id,
                    "seed": seed,
                }
            )
    return index, failures


def _failure_owner(code: str, stage: str) -> str:
    if stage == "infrastructure":
        return "infrastructure"
    if stage == "source":
        return "source"
    if stage == "internal":
        return "internal"
    if code.startswith(("generated_scene_", "initial_scene_")):
        return "generated_scene"
    if code.startswith(("planner_", "planning_")):
        return "planning"
    if code.startswith(("grasp_", "handover_")):
        return "manipulation"
    if code.startswith(("checker_", "other", "ambiguous")):
        return "checker"
    if code.startswith(("settle_", "stability_")):
        return "settling"
    if code.startswith("descriptor_"):
        return "descriptor"
    return "execution"


def freeze_scene_validation(
    *,
    plan: Mapping[str, Any] | Path | None = None,
    attempts: Sequence[Mapping[str, Any]] | Path | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """Aggregate checkpointed attempts into deterministic M15 artifacts."""

    config = ValidationConfig.load(config_path)
    if plan is None:
        plan_value = _load_yaml(output_dir / PLAN_FILE)
    elif isinstance(plan, (str, Path)):
        plan_value = _load_yaml(Path(plan))
    else:
        plan_value = dict(plan)
    if (plan_value.get("artifact_kind") == "scene_validation_execution_plan"
            or plan_value.get("execution_only") or plan_value.get("parent_dispatch_hash")
            or plan_value.get("parent_dispatch_path")):
        raise SceneValidationError(
            "execution_only_plan",
            "execution children/diagnostic dispatches cannot be frozen as full M15 validation",
            stage="source",
        )
    if not _self_hash_valid(plan_value):
        raise SceneValidationError(
            "plan_self_hash_mismatch",
            "validation plan self-hash does not match",
            stage="source",
        )
    if plan_value.get("source_failures"):
        raise SceneValidationError(
            "plan_source_failures",
            "cannot freeze a plan with blocking source failures: "
            + ", ".join(
                str(
                    item.get("scene_spec_id")
                    or item.get("path")
                    or item.get("code")
                )
                for item in plan_value["source_failures"][:20]
            ),
            stage="source",
        )
    if isinstance(plan_value.get("coverage"), Mapping) and not bool(
        plan_value["coverage"].get("complete")
    ):
        raise SceneValidationError(
            "plan_coverage_incomplete",
            "cannot freeze a plan with uncovered tokens: "
            + ", ".join(plan_value["coverage"].get("uncovered_tokens", [])),
            stage="source",
        )
    if plan_value.get("config_hash") != config.hash:
        raise SceneValidationError(
            "stale_plan_config",
            "validation plan was built with a different config",
            stage="source",
        )
    source_errors = _verify_plan_sources(plan_value)
    if source_errors:
        raise SceneValidationError(
            "stale_plan_sources",
            "; ".join(source_errors[:20]),
            stage="source",
        )
    if attempts is None:
        attempt_path = (
            output_dir
            / "runs"
            / str(plan_value["plan_id"])
            / "attempts.jsonl"
        )
        attempt_rows = _iter_attempts(attempt_path)
    elif isinstance(attempts, (str, Path)):
        attempt_path = Path(attempts)
        attempt_rows = _iter_attempts(attempt_path)
    else:
        attempt_path = None
        attempt_rows = attempts

    index, checkpoint_failures = _attempt_index(
        plan_value, attempt_rows, config.seeds, for_freeze=True
    )
    attempts_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in index.values():
        attempts_by_scene[str(item["scene_spec_id"])].append(item)
    descriptor_rows: list[dict[str, Any]] = []
    t0_rows: list[dict[str, Any]] = []
    rollout_rows: list[dict[str, Any]] = []
    attribution_rows: list[dict[str, Any]] = []
    scene_records: list[dict[str, Any]] = []

    for scene in plan_value.get("scenes", []):
        scene_id = str(scene["scene_spec_id"])
        scene_attempts = attempts_by_scene[scene_id]
        descriptor_values = [
            item.get("descriptor_gate")
            for item in scene_attempts
            if isinstance(item.get("descriptor_gate"), Mapping)
        ]
        descriptor = (
            sorted(
                descriptor_values,
                key=lambda item: artifact_hash(item),
            )[0]
            if descriptor_values
            else {
                "status": "infrastructure_failure",
                "passed": False,
                "failures": [{"code": "missing_descriptor_gate"}],
            }
        )
        descriptor_hashes = {artifact_hash(value) for value in descriptor_values}
        if len(descriptor_hashes) > 1:
            descriptor = {
                "status": "internal_failure",
                "passed": False,
                "failures": [{"code": "non_invariant_descriptor_render"}],
            }
        descriptor_internal_failure = (
            descriptor.get("status") == "internal_failure"
        )
        descriptor_row = {
            "scene_spec_id": scene_id,
            "passed": bool(descriptor.get("passed")),
            "result": canonical_data(descriptor),
        }
        descriptor_rows.append(descriptor_row)
        for failure in descriptor.get("failures", []):
            code = str(failure.get("code") or "descriptor_gate_failure")
            stage = (
                "internal"
                if descriptor_internal_failure
                else "semantic"
            )
            attribution_rows.append(
                {
                    "scene_spec_id": scene_id,
                    "task_record_id": None,
                    "attempt_id": None,
                    "seed": None,
                    "code": code,
                    "stage": stage,
                    "owner": _failure_owner(code, stage),
                }
            )

        task_results: list[dict[str, Any]] = []
        for task in scene.get("task_records", []):
            task_id = str(task["task_record_id"])
            task_attempts = sorted(
                (
                    item
                    for item in scene_attempts
                    if item["task_record_id"] == task_id
                ),
                key=lambda item: int(item["seed"]),
            )
            infra = [
                item
                for item in task_attempts
                if item.get("status") == "infrastructure_failure"
            ]
            source_failures = [
                item
                for item in task_attempts
                if item.get("status") == "source_failure"
            ]
            internal_failures_for_task = [
                item
                for item in task_attempts
                if item.get("status") == "internal_failure"
            ]
            nonsemantic_statuses = {
                "infrastructure_failure",
                "source_failure",
                "internal_failure",
            }
            semantic = [
                item
                for item in task_attempts
                if item.get("status") not in nonsemantic_statuses
            ]
            t0_values = [
                item.get("t0_checker")
                for item in task_attempts
                if isinstance(item.get("t0_checker"), Mapping)
            ]
            t0_passed = all(
                (item.get("t0_checker") or {}).get("status") == "failed"
                for item in task_attempts
                if isinstance(item.get("t0_checker"), Mapping)
            ) and len(t0_values) == len(config.seeds)
            successes = sum(
                1
                for item in semantic
                if item.get("classification") == "intended"
                and bool(item.get("skill_completed", True))
                and bool(item.get("stable", True))
            )
            counts = Counter(
                str(item.get("classification") or "failure") for item in semantic
            )
            validated = bool(
                descriptor.get("passed")
                and t0_passed
                and not infra
                and not source_failures
                and not internal_failures_for_task
                and len(semantic) == len(config.seeds)
                and successes >= config.minimum_success_count
            )
            result = {
                "task_record_id": task_id,
                "goal_variant_id": task["goal_variant_id"],
                "source_task_hash": task["source_task_hash"],
                "validated": validated,
                "excluded": not validated,
                "success_count": successes,
                "denominator": len(semantic),
                "minimum_success_count": config.minimum_success_count,
                "success_rate": successes / len(semantic) if semantic else None,
                "classification_counts": dict(sorted(counts.items())),
                "t0_false": t0_passed,
                **SUCCESS_CONTRACT.identity,
                "success_snapshot_hashes": sorted(
                    {
                        str(item["success_snapshot_hash"])
                        for item in semantic
                        if item.get("success_snapshot_hash")
                    }
                ),
                "infrastructure_failure_count": len(infra),
                "source_failure_count": len(source_failures),
                "internal_failure_count": len(internal_failures_for_task),
                "attempts": [
                    {
                        "attempt_id": item["attempt_id"],
                        "seed": item["seed"],
                        "status": item.get("status"),
                        "classification": item.get("classification"),
                        "failure_code": item.get("failure_code"),
                        "failure_stage": item.get("failure_stage"),
                        "success_snapshot_hash": item.get(
                            "success_snapshot_hash"
                        ),
                        "predicate_result": canonical_data(
                            (item.get("terminal_checkers") or {}).get(task_id)
                            or {}
                        ),
                        "evidence": item.get("evidence", {}),
                    }
                    for item in task_attempts
                ],
            }
            task_results.append(result)
            t0_rows.append(
                {
                    "scene_spec_id": scene_id,
                    "task_record_id": task_id,
                    "passed": t0_passed,
                }
            )
            rollout_rows.append(
                {
                    "scene_spec_id": scene_id,
                    **canonical_data(result),
                }
            )
            for item in task_attempts:
                if (
                    item.get("classification") == "intended"
                    and not item.get("failure_code")
                ):
                    continue
                code = str(
                    item.get("failure_code")
                    or item.get("classification")
                    or item.get("status")
                    or "execution_failure"
                )
                stage = str(item.get("failure_stage") or "semantic")
                attribution_rows.append(
                    {
                        "scene_spec_id": scene_id,
                        "task_record_id": task_id,
                        "attempt_id": item["attempt_id"],
                        "seed": item["seed"],
                        "code": code,
                        "stage": stage,
                        "owner": _failure_owner(code, stage),
                    }
                )

        infrastructure_failure = any(
            item.get("status") == "infrastructure_failure"
            for item in scene_attempts
        ) or any(
            item.get("code") in {"missing_attempt", "duplicate_attempt", "stale_attempt"}
            and item.get("scene_spec_id") in {None, scene_id}
            for item in checkpoint_failures
        )
        source_failure = any(
            item.get("status") == "source_failure"
            for item in scene_attempts
        )
        attempt_internal_failure = any(
            item.get("status") == "internal_failure"
            for item in scene_attempts
        )
        record: dict[str, Any] = {
            "artifact_kind": "scene_validation_record",
            "schema_version": SCHEMA_VERSION,
            "scene_validation_version": SCENE_VALIDATION_VERSION,
            "id": deterministic_id(
                "svr",
                {
                    "scene_spec_id": scene_id,
                    "plan_hash": plan_value["artifact_hash"],
                    "config_hash": config.hash,
                },
            ),
            "scene_spec_id": scene_id,
            "realized_family_id": scene["realized_family_id"],
            "source_scene_hash": scene["source_scene_hash"],
            "source_realization_hash": scene["source_realization_hash"],
            "source_unit_hash": scene["source_unit_hash"],
            "plan_hash": plan_value["artifact_hash"],
            "config_hash": config.hash,
            **SUCCESS_CONTRACT.identity,
            "calibration_only": bool(plan_value.get("calibration_only")),
            "publication_ready": False,
            "descriptor_gate": canonical_data(descriptor),
            "not_success_at_t0": {
                "passed": all(item["t0_false"] for item in task_results),
                "task_count": len(task_results),
            },
            "infrastructure_failure": infrastructure_failure,
            "source_failure": source_failure,
            "internal_failure": bool(
                descriptor_internal_failure or attempt_internal_failure
            ),
            "task_results": canonical_data(task_results),
            "summary": {
                "task_count": len(task_results),
                "validated_task_count": sum(
                    1 for item in task_results if item["validated"]
                ),
                "excluded_task_count": sum(
                    1 for item in task_results if item["excluded"]
                ),
            },
            "provenance": build_provenance(
                generator="if_benchmark.scene_validation.freeze",
                config_hash=config.hash,
                sources=[
                    {
                        "kind": "scene_validation_plan",
                        "record_id": plan_value["plan_id"],
                        "hash": plan_value["artifact_hash"],
                    },
                    {
                        "kind": "scene_spec",
                        "record_id": scene_id,
                        "hash": scene["source_scene_hash"],
                    },
                ],
                metadata={
                    "calibration_only": bool(plan_value.get("calibration_only")),
                    "scene_validation_version": SCENE_VALIDATION_VERSION,
                },
            ),
        }
        record["artifact_hash"] = artifact_hash(record)
        scene_records.append(record)

    infrastructure_failures = sum(
        1 for record in scene_records if record["infrastructure_failure"]
    )
    internal_failures = sum(
        1 for record in scene_records if record["internal_failure"]
    )
    source_failures = sum(
        1 for record in scene_records if record["source_failure"]
    )
    implementation_complete = (
        not checkpoint_failures
        and infrastructure_failures == 0
        and source_failures == 0
        and internal_failures == 0
    )
    publication_ready = bool(
        not plan_value.get("calibration_only") and implementation_complete
    )
    record_dir = output_dir / "scene_validation"
    active_scene_ids = {str(record["scene_spec_id"]) for record in scene_records}
    for stale_path in sorted(record_dir.glob("ss_*.yml")):
        if stale_path.stem not in active_scene_ids:
            stale_path.unlink()
    for record in scene_records:
        record["publication_ready"] = publication_ready
        record["artifact_hash"] = artifact_hash(
            record, omit_keys=("artifact_hash",)
        )
        _write_yaml(record_dir / f"{record['scene_spec_id']}.yml", record)

    def report(
        kind: str,
        rows: Sequence[Mapping[str, Any]],
        summary: Mapping[str, Any],
    ) -> dict[str, Any]:
        value: dict[str, Any] = {
            "artifact_kind": kind,
            "schema_version": SCHEMA_VERSION,
            "scene_validation_version": SCENE_VALIDATION_VERSION,
            "plan_hash": plan_value["artifact_hash"],
            "config_hash": config.hash,
            "calibration_only": bool(plan_value.get("calibration_only")),
            "publication_ready": publication_ready,
            "summary": canonical_data(dict(summary)),
            "records": canonical_data(list(rows)),
        }
        value["report_hash"] = artifact_hash(value)
        return value

    descriptor_report = report(
        "descriptor_gate_report",
        descriptor_rows,
        {
            "scene_count": len(scene_records),
            "passed_scene_count": sum(1 for item in descriptor_rows if item["passed"]),
            "failed_scene_count": sum(1 for item in descriptor_rows if not item["passed"]),
        },
    )
    t0_report = report(
        "not_success_at_t0_report",
        t0_rows,
        {
            "task_count": len(t0_rows),
            "passed_task_count": sum(1 for item in t0_rows if item["passed"]),
            "failed_task_count": sum(1 for item in t0_rows if not item["passed"]),
        },
    )
    rollout_report = report(
        "sim_rollout_report",
        rollout_rows,
        {
            "scene_count": len(scene_records),
            "task_count": len(rollout_rows),
            "attempt_count": len(index),
            "expected_attempt_count": sum(
                len(scene["task_records"]) * len(config.seeds)
                for scene in plan_value.get("scenes", [])
            ),
            "validated_task_count": sum(
                1 for item in rollout_rows if item["validated"]
            ),
            "excluded_task_count": sum(
                1 for item in rollout_rows if item["excluded"]
            ),
            "checkpoint_failure_count": len(checkpoint_failures),
        },
    )
    failure_report = report(
        "failure_attribution_report",
        attribution_rows,
        {
            "failure_count": len(attribution_rows),
            "owner_counts": dict(
                sorted(Counter(item["owner"] for item in attribution_rows).items())
            ),
            "code_counts": dict(
                sorted(Counter(item["code"] for item in attribution_rows).items())
            ),
            "checkpoint_failures": canonical_data(checkpoint_failures),
        },
    )
    for filename, value in (
        (DESCRIPTOR_REPORT_FILE, descriptor_report),
        (T0_REPORT_FILE, t0_report),
        (ROLLOUT_REPORT_FILE, rollout_report),
        (FAILURE_REPORT_FILE, failure_report),
    ):
        _write_yaml(output_dir / filename, value)

    artifact_set_hash = artifact_hash(
        {
            "scene_records": [
                record["artifact_hash"]
                for record in sorted(scene_records, key=lambda item: item["scene_spec_id"])
            ],
            "reports": [
                descriptor_report["report_hash"],
                t0_report["report_hash"],
                rollout_report["report_hash"],
                failure_report["report_hash"],
            ],
        }
    )
    return {
        "artifact_set_hash": artifact_set_hash,
        "calibration_only": bool(plan_value.get("calibration_only")),
        "publication_ready": publication_ready,
        "coverage_complete": bool(
            (plan_value.get("coverage") or {}).get("token_complete", True)
        )
        and all(item["validated"] for item in rollout_rows),
        "implementation_complete": implementation_complete,
        "summary": {
            **rollout_report["summary"],
            "infrastructure_failure_scene_count": infrastructure_failures,
            "source_failure_scene_count": source_failures,
            "internal_failure_scene_count": internal_failures,
        },
        "reports": {
            DESCRIPTOR_REPORT_FILE: descriptor_report,
            T0_REPORT_FILE: t0_report,
            ROLLOUT_REPORT_FILE: rollout_report,
            FAILURE_REPORT_FILE: failure_report,
        },
        "scene_records": scene_records,
    }


def check_scene_validation(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """Cheap staleness and hash check for frozen M15 artifacts."""

    errors: list[str] = []
    config = ValidationConfig.load(config_path)
    plan_path = output_dir / PLAN_FILE
    if not plan_path.is_file():
        return {"stale": True, "stale_paths": [str(plan_path)]}
    plan = _load_yaml(plan_path)
    if (
        not _self_hash_valid(plan)
        or plan.get("config_hash") != config.hash
        or bool(plan.get("source_failures"))
        or (
            isinstance(plan.get("coverage"), Mapping)
            and not bool(plan["coverage"].get("complete"))
        )
    ):
        errors.append(str(plan_path))
    errors.extend(_verify_plan_sources(plan))
    report_hashes: dict[str, str] = {}
    for filename in REPORT_SEQUENCE:
        path = output_dir / filename
        if not path.is_file():
            errors.append(str(path))
            continue
        value = _load_yaml(path)
        if (
            not _report_hash_valid(value)
            or value.get("plan_hash") != plan.get("artifact_hash")
            or value.get("config_hash") != config.hash
        ):
            errors.append(str(path))
        else:
            report_hashes[filename] = str(value["report_hash"])
    expected_scenes = {
        str(item["scene_spec_id"]): item for item in plan.get("scenes", [])
    }
    scene_hashes: dict[str, str] = {}
    for scene_id, source in expected_scenes.items():
        path = output_dir / "scene_validation" / f"{scene_id}.yml"
        if not path.is_file():
            errors.append(str(path))
            continue
        value = _load_yaml(path)
        if (
            not _self_hash_valid(value)
            or value.get("source_scene_hash") != source.get("source_scene_hash")
            or value.get("source_realization_hash")
            != source.get("source_realization_hash")
            or value.get("source_unit_hash") != source.get("source_unit_hash")
            or value.get("plan_hash") != plan.get("artifact_hash")
        ):
            errors.append(str(path))
        else:
            scene_hashes[scene_id] = str(value["artifact_hash"])
            for task in value.get("task_results", []):
                for attempt in task.get("attempts", []):
                    errors.extend(
                        _evidence_sidecar_errors(
                            attempt.get("evidence", {}),
                            record_path=path,
                        )
                    )
    attempt_path = (
        output_dir
        / "runs"
        / str(plan.get("plan_id"))
        / "attempts.jsonl"
    )
    if not attempt_path.is_file():
        errors.append(str(attempt_path))
    artifact_set_hash = (
        artifact_hash(
            {
                "scene_records": [
                    scene_hashes[key] for key in sorted(scene_hashes)
                ],
                "reports": [
                    report_hashes[key] for key in REPORT_SEQUENCE
                ],
            }
        )
        if len(scene_hashes) == len(expected_scenes)
        and len(report_hashes) == len(REPORT_SEQUENCE)
        else None
    )
    return {
        "stale": bool(errors),
        "stale_paths": sorted(set(errors)),
        "scene_count": len(expected_scenes),
        "plan_id": plan.get("plan_id"),
        "artifact_set_hash": artifact_set_hash,
    }


def validate_scene_validation(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """Deep schema, aggregation, and invariant validation."""

    checked = check_scene_validation(
        output_dir=output_dir, config_path=config_path
    )
    errors = [f"stale: {path}" for path in checked.get("stale_paths", [])]
    if errors:
        return {"valid": False, "errors": errors}
    plan = _load_yaml(output_dir / PLAN_FILE)
    for scene in plan.get("scenes", []):
        path = output_dir / "scene_validation" / f"{scene['scene_spec_id']}.yml"
        value = _load_yaml(path)
        try:
            schema_result = validate_document(
                {
                    "schema_version": SCHEMA_VERSION,
                    "kind": "scene_validation_record",
                    "scene_validation_record": value,
                }
            )
            if not schema_result["valid"]:
                for error in schema_result["errors"]:
                    errors.append(
                        f"{path}: {error.get('path')}: {error.get('message')}"
                    )
        except Exception as exc:  # schema library raises several useful types
            errors.append(f"{path}: {exc}")
        task_ids = [item["task_record_id"] for item in value.get("task_results", [])]
        if len(task_ids) != len(set(task_ids)):
            errors.append(f"{path}: duplicate task results")
        for result in value.get("task_results", []):
            denominator = int(result.get("denominator", 0))
            counts = sum(int(value) for value in result.get("classification_counts", {}).values())
            if denominator != counts:
                errors.append(
                    f"{path}: classification denominator mismatch for {result.get('task_record_id')}"
                )
    attempt_path = (
        output_dir
        / "runs"
        / str(plan["plan_id"])
        / "attempts.jsonl"
    )
    if attempt_path.is_file() and not errors:
        with tempfile.TemporaryDirectory(prefix="if-m15-validate-") as tmp:
            rebuilt_dir = Path(tmp)
            rebuilt = freeze_scene_validation(
                plan=plan,
                attempts=attempt_path,
                output_dir=rebuilt_dir,
                config_path=config_path,
            )
            for filename in REPORT_SEQUENCE:
                current = _load_yaml(output_dir / filename)
                expected = _load_yaml(rebuilt_dir / filename)
                if canonical_data(current) != canonical_data(expected):
                    errors.append(
                        f"{output_dir / filename}: report reconstruction mismatch"
                    )
            for scene in plan.get("scenes", []):
                filename = f"{scene['scene_spec_id']}.yml"
                current = _load_yaml(
                    output_dir / "scene_validation" / filename
                )
                expected = _load_yaml(
                    rebuilt_dir / "scene_validation" / filename
                )
                if canonical_data(current) != canonical_data(expected):
                    errors.append(
                        f"{output_dir / 'scene_validation' / filename}: "
                        "record reconstruction mismatch"
                    )
            expected_hashes = {
                filename: rebuilt["reports"][filename]["report_hash"]
                for filename in REPORT_SEQUENCE
            }
            actual_hashes = {
                filename: _load_yaml(output_dir / filename)["report_hash"]
                for filename in REPORT_SEQUENCE
            }
            if expected_hashes != actual_hashes:
                errors.append("artifact-set report hash reconstruction mismatch")
    return {
        "valid": not errors,
        "errors": errors,
        "scene_count": checked.get("scene_count", 0),
        "plan_id": checked.get("plan_id"),
    }


__all__ = [
    "CHECKER_GUARDS",
    "CHECKER_OPERATORS",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_REALIZATION_DIR",
    "DEFAULT_ROUTE_DIR",
    "DEFAULT_SCENE_SPEC_DIR",
    "DEFAULT_TASK_DIR",
    "DEFAULT_UNIT_DIR",
    "DESCRIPTOR_SUBTYPES",
    "PLAN_FILE",
    "REPORT_FILES",
    "ROUTES",
    "SCENE_VALIDATION_VERSION",
    "SKILL_OPERATORS",
    "SceneValidationError",
    "ValidationConfig",
    "build_validation_plan",
    "check_scene_validation",
    "classify_checker_matches",
    "evaluate_checker_config",
    "evaluate_task_checker",
    "evaluate_descriptor_gate",
    "dispatch_skill_config",
    "preflight_skill_operation_anchors",
    "freeze_scene_validation",
    "stable_attempt_id",
    "strict_source_join",
    "resolve_entity_dynamics",
    "resolve_reservation_anchor",
    "validate_scene_validation",
]
