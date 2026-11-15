"""Milestone 8 finite-domain assignment solving.

The solver is deliberately simulator-independent.  It consumes only frozen
BenchmarkV1 artifacts, treats every emitted AssignmentCandidate as one atomic
joint binding, and never samples roles independently.
"""

from __future__ import annotations

import hashlib
import heapq
import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
import multiprocessing
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence

import yaml

from if_benchmark.asset_role_contract import CANDIDATE_TO_ASSET_ROLE
from if_benchmark.asset_validation_contract import (
    ASSET_VALIDATION_SCHEMA_VERSION,
    M4_ASSET_SCOPE_USES,
    M4_EXECUTABLE_USES,
)
from if_benchmark.asset_semantic_exclusions import (
    DEFAULT_SEMANTIC_EXCLUSIONS_PATH,
    expand_asset_semantic_exclusions,
    load_asset_semantic_exclusions,
)
from if_benchmark.initial_pose.core import (
    DEFAULT_ASSIGNMENT_STATE_SELECTION_PATH,
    canonical_assignment_selection_identity,
    canonical_assignment_states,
    load_assignment_state_selection,
    validate_initial_pose_set,
)
from if_benchmark.schema import (
    artifact_hash,
    canonical_yaml,
    deterministic_id,
    validate_document,
)
from if_benchmark.schema.serialization import canonical_data
from if_benchmark.articulation_capability import (
    ARTICULATION_CAPABILITY_VERSION,
    aggregate_articulation_capabilities,
    articulation_capability_errors,
    validated_articulation_arm_capability,
)
from if_benchmark.touch_capability import (
    TOUCH_CAPABILITY_VERSION,
    TOUCH_EXECUTABLE_USES,
    touch_capability_errors,
    validated_touch_arm_capability,
)
from if_benchmark.pair_recipe import (
    LEGACY_M5_REPORT_V3_PROJECTIONS,
    TERMINAL_PLACEMENT_RECIPE_VERSION,
    rebind_legacy_terminal_placement_recipe,
    validate_terminal_placement_recipe,
)
from if_benchmark.relation_contract import (
    ANCHOR_PAIR_DISTANCE_BANDS_M,
    ANCHOR_PAIR_TERMINAL_CLEARANCE_M,
)


SCHEMA_VERSION = 1
DOMAIN_VERSION = 5
DEFAULT_MAX_CANDIDATES = 2048
SIZE_RANK = {"small": 0, "medium": 1, "large": 2}

DEFAULT_ASSET_CATALOG_PATH = Path("if_benchmark/assets/asset_catalog.yml")
DEFAULT_ASSET_VALIDATION_PATH = Path(
    "if_benchmark/data/pipeline_runs/release/"
    "m04_asset_validation/frozen/validated_assets.yml"
)
DEFAULT_ATTRIBUTE_CATALOG_PATH = Path(
    "if_benchmark/data/attributes/reviewed_attribute_catalog.yml"
)
DEFAULT_GEOMETRY_PATH = Path("if_benchmark/data/inventory/geometry_summary.yml")
DEFAULT_INITIAL_POSE_PATH = Path(
    "if_benchmark/data/initial_pose/validated_initial_states.yml"
)
DEFAULT_SKILL_PAIRS_PATH = Path(
    "if_benchmark/data/pipeline_runs/release/m05_pair_validation/frozen/validated_skill_pairs.yml"
)
DEFAULT_TERMINAL_PAIRS_PATH = Path(
    "if_benchmark/data/pipeline_runs/release/m05_pair_validation/frozen/validated_terminal_pairs.yml"
)
DEFAULT_RECEIVER_PROFILES_PATH = Path(
    "if_benchmark/assets/articulated_receiver_profiles.yml"
)
DEFAULT_ARCHETYPES_PATH = Path("if_benchmark/data/families/archetypes.yml")
DEFAULT_FAMILIES_PATH = Path("if_benchmark/data/families/abstract_families.yml")
DEFAULT_OUTPUT_DIR = Path("if_benchmark/data/assignment_domains")
DEFAULT_PROCESS_WORKERS = min(32, os.cpu_count() or 1)

SUMMARY_FILE = "domain_summary.yml"
EMPTY_FILE = "empty_domain_report.yml"

CONFIG: dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "domain_version": DOMAIN_VERSION,
    "max_candidates": DEFAULT_MAX_CANDIDATES,
    "size_rank": SIZE_RANK,
    "candidate_to_asset_role": CANDIDATE_TO_ASSET_ROLE,
    "asset_validation_evidence_identity": (
        "asset_model_use_role_v4_worker_policy_manifest"
    ),
    "articulation_capability_policy": ARTICULATION_CAPABILITY_VERSION,
    "touch_capability_policy": TOUCH_CAPABILITY_VERSION,
    "pair_placement_recipe_policy": TERMINAL_PLACEMENT_RECIPE_VERSION,
    "selection": {
        "coverage": "deterministic_greedy_set_cover",
        "tie_break": "sha256_joint_binding",
        "fill": "lowest_sha256_joint_bindings",
        "overflow": "selection_capacity_exceeded",
        "exact_pair_evidence_keys": "audit_only",
        "pair_marginals": [
            "stage_relation",
            "subject_model",
            "receiver_model",
            "validated_band",
        ],
    },
    "assignment_seed": "sha256(family_id|domain_hash|seed) modulo candidate_count",
    "anchor_pair_distance_bands_m": ANCHOR_PAIR_DISTANCE_BANDS_M,
    "anchor_pair_terminal_clearance_m": ANCHOR_PAIR_TERMINAL_CLEARANCE_M,
}

CONTACT_VALIDATED_USES = frozenset(
    use
    for use in ("pick_left", "pick_right", "bimanual_grab_subject")
    if use in M4_EXECUTABLE_USES
)
CONTACT_REF_REQUIRED_FIELDS = (
    "policy_version",
    "contact_contract_id",
    "contact_contract_hash",
    "contact_projection_hash",
    "candidate_use",
    "use_profile",
    "contact_count",
    "contact_groups",
)
ARTICULATION_VALIDATED_USES = frozenset(
    ("open", "close", "switch_on", "switch_off")
)
_LEGACY_ARTICULATION_REPORT_CACHE: dict[
    tuple[str, str, str], Mapping[str, Any]
] = {}
ARTICULATION_REF_REQUIRED_FIELDS = (
    "profile_version",
    "profile_hash",
    "profile_source_hash",
    "motion_policy_version",
    "motion_policy_hash",
    "capability_version",
    "sampling_version",
    "requested_arms",
    "arm_capabilities",
)
TOUCH_REF_REQUIRED_FIELDS = (
    "capability_version",
    "sampling_version",
    "requested_arms",
    "arm_results",
    "arm_capabilities",
)


def _validated_m4_execution_evidence(
    evidence: Mapping[str, Any],
) -> bool:
    if str(evidence.get("stage") or "") != "canonical_skill":
        return False
    for field_name in ("evidence_id", "evidence_path", "evidence_hash"):
        if not evidence.get(field_name):
            raise ValueError(
                f"validated canonical evidence lacks {field_name}"
            )
    return True


def _validated_contact_ref(
    evidence: Mapping[str, Any],
) -> dict[str, Any] | None:
    use = str(evidence.get("use") or "")
    binding = evidence.get("contact_contract_binding")
    if use not in CONTACT_VALIDATED_USES:
        return None
    if not isinstance(binding, Mapping):
        raise ValueError(f"validated {use} evidence lacks Contact contract")
    missing = [
        field
        for field in CONTACT_REF_REQUIRED_FIELDS
        if binding.get(field) is None
    ]
    if missing:
        raise ValueError(
            f"validated {use} Contact contract lacks {', '.join(missing)}"
        )
    if evidence.get("contact_contract_hash") != binding.get(
        "contact_contract_hash"
    ):
        raise ValueError(f"validated {use} Contact contract hash mismatch")
    for field_name in ("evidence_id", "evidence_path", "evidence_hash"):
        if not evidence.get(field_name):
            raise ValueError(
                f"validated {use} evidence lacks {field_name}"
            )
    return canonical_data(dict(binding))


def _validated_articulation_ref(
    evidence: Mapping[str, Any],
) -> dict[str, Any] | None:
    use = str(evidence.get("use") or "")
    if use not in ARTICULATION_VALIDATED_USES:
        return None
    binding = evidence.get("articulation_execution_binding")
    if not isinstance(binding, Mapping):
        raise ValueError(
            f"validated {use} evidence lacks articulation execution binding"
        )
    if any(
        not binding.get(field)
        for field in ("capability_version", "arm_capabilities")
    ):
        binding = _legacy_articulation_binding_from_report(evidence, binding)
    missing = [
        field
        for field in ARTICULATION_REF_REQUIRED_FIELDS
        if not binding.get(field)
    ]
    if missing:
        raise ValueError(
            f"validated {use} articulation binding lacks {', '.join(missing)}"
        )
    if binding.get("capability_version") != ARTICULATION_CAPABILITY_VERSION:
        raise ValueError(f"validated {use} articulation capability is stale")
    capability_failures = articulation_capability_errors(binding, use=use)
    if capability_failures:
        raise ValueError(
            f"validated {use} articulation capability is invalid: "
            f"{', '.join(capability_failures)}"
        )
    for field_name in ("evidence_id", "evidence_path", "evidence_hash"):
        if not evidence.get(field_name):
            raise ValueError(
                f"validated {use} evidence lacks {field_name}"
            )
    return canonical_data(dict(binding))


def _validated_touch_ref(
    evidence: Mapping[str, Any],
) -> dict[str, Any] | None:
    use = str(evidence.get("use") or "")
    if use not in TOUCH_EXECUTABLE_USES:
        return None
    binding = evidence.get("touch_execution_binding")
    if not isinstance(binding, Mapping):
        raise ValueError(
            f"validated {use} evidence lacks touch execution binding"
        )
    missing = [
        field for field in TOUCH_REF_REQUIRED_FIELDS if not binding.get(field)
    ]
    if missing:
        raise ValueError(
            f"validated {use} touch binding lacks {', '.join(missing)}"
        )
    failures = touch_capability_errors(binding, use=use)
    if failures:
        raise ValueError(
            f"validated {use} touch capability is invalid: "
            f"{', '.join(failures)}"
        )
    for field_name in ("evidence_id", "evidence_path", "evidence_hash"):
        if not evidence.get(field_name):
            raise ValueError(f"validated {use} evidence lacks {field_name}")
    return canonical_data(dict(binding))


def _legacy_articulation_binding_from_report(
    evidence: Mapping[str, Any], binding: Mapping[str, Any]
) -> dict[str, Any]:
    """Recover exact arm/candidate evidence from a self-hashed legacy M4 report.

    Compatibility is intentionally evidence-backed: a profile-only ledger row
    is never upgraded by assumption.  If the original report is unavailable,
    stale, ambiguous, or lacks a successful runtime trace, the row remains
    unusable and the caller rejects it.
    """

    use = str(evidence.get("use") or "")
    path = Path(str(evidence.get("evidence_path") or ""))
    report = _verified_legacy_articulation_report(
        path,
        evidence_hash=str(evidence.get("evidence_hash") or ""),
        evidence_id=str(evidence.get("evidence_id") or ""),
    )
    if report is None:
        return canonical_data(dict(binding))
    asset = str(evidence.get("asset") or "")
    model_id = str(evidence.get("model_id") or "")
    records = [
        row
        for row in report.get("records") or []
        if str(row.get("asset") or "") == asset
        and str(row.get("model_id") or "") == model_id
        and str(row.get("use") or "") == use
        and row.get("pass") is True
    ]
    if len(records) != 1:
        return canonical_data(dict(binding))
    record = records[0]
    attempts = [
        row
        for row in report.get("attempts") or []
        if str(row.get("asset") or "") == asset
        and str(row.get("model_id") or "") == model_id
        and str(row.get("use") or "") == use
        and row.get("warmup") is not True
    ]
    capability = aggregate_articulation_capabilities(
        attempts,
        use=use,
        denominator=int(record.get("denominator") or 0),
        threshold=float(record.get("threshold") or 0.0),
    )
    if capability.get("failures") or not capability.get("arm_capabilities"):
        return canonical_data(dict(binding))
    return canonical_data(
        {
            **dict(binding),
            "capability_version": capability["capability_version"],
            "arm_capabilities": capability["arm_capabilities"],
        }
    )


def _verified_legacy_articulation_report(
    path: Path, *, evidence_hash: str, evidence_id: str
) -> Mapping[str, Any] | None:
    """Load one legacy report once, caching only after full identity proof."""

    key = (str(path), str(evidence_hash), str(evidence_id))
    cached = _LEGACY_ARTICULATION_REPORT_CACHE.get(key)
    if cached is not None:
        return cached
    if not path.is_file() or not evidence_hash or not evidence_id:
        return None
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if (
        str(report.get("report_hash") or "") != evidence_hash
        or evidence_hash != artifact_hash(report, omit_keys=("report_hash",))
        or str(report.get("evidence_id") or "") != evidence_id
    ):
        return None
    _LEGACY_ARTICULATION_REPORT_CACHE[key] = report
    return report


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


def _model_key(asset: str, model_id: str) -> str:
    return f"{asset}|{model_id}"


@dataclass(frozen=True)
class AssetRecord:
    key: str
    asset: str
    model_id: str
    category: str
    asset_roles: tuple[str, ...]
    attributes: Mapping[str, str]
    geometry: Mapping[str, Any]
    validations: Mapping[str, Mapping[str, Any]]
    role_evidence: Mapping[str, Mapping[str, Any]]
    initial_state: Mapping[str, Any] = field(default_factory=dict)

    @property
    def noun(self) -> str | None:
        value = self.attributes.get("type")
        return str(value) if value else None


@dataclass(frozen=True)
class AssetIndex:
    records: Mapping[str, AssetRecord]
    by_role: Mapping[str, tuple[str, ...]]
    by_use: Mapping[str, tuple[str, ...]]
    by_category: Mapping[str, tuple[str, ...]]
    by_attribute: Mapping[tuple[str, str, str], tuple[str, ...]]
    source_hashes: Mapping[str, str]


@dataclass(frozen=True)
class PairRecord:
    pair_key: str
    relation: str
    subject_key: str
    receiver_key: str
    stage: str
    evidence_id: str | None
    evidence_path: str | None
    recipe_id: str | None
    placement_recipe: Mapping[str, Any]
    placement_recipe_source: str
    band_ids: tuple[str, ...]
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class PairIndex:
    terminal: Mapping[tuple[str, str, str], PairRecord]
    skill: Mapping[tuple[str, str, str], PairRecord]
    by_relation_terminal: Mapping[str, tuple[str, ...]]
    by_relation_skill: Mapping[str, tuple[str, ...]]
    source_hashes: Mapping[str, str]


class SelectionCapacityExceeded(ValueError):
    """Carries a complete diagnostic domain when bounded coverage is impossible."""

    def __init__(self, message: str, domain: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.domain = dict(domain)


def _validated_asset_projection_rows(
    validation: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Reject ambiguous or miscounted M4 projections before M8 indexing."""

    raw_rows = validation.get("records")
    if not isinstance(raw_rows, list) or not all(
        isinstance(row, Mapping) for row in raw_rows
    ):
        raise ValueError("validated asset projection records are invalid")
    rows = [dict(row) for row in raw_rows]
    identities = [
        (
            str(row.get("asset")),
            str(row.get("model_id")),
            str(row.get("use")),
        )
        for row in rows
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("validated asset projection contains duplicate identity")
    if any(row.get("status") != "validated" for row in rows):
        raise ValueError("validated asset projection contains non-validated row")
    if any(str(row.get("use")) not in M4_ASSET_SCOPE_USES for row in rows):
        raise ValueError("validated asset projection contains unsupported use")

    summary = validation.get("summary")
    if not isinstance(summary, Mapping):
        raise ValueError("validated asset projection summary is missing")
    record_count = summary.get("record_count")
    if (
        not isinstance(record_count, int)
        or isinstance(record_count, bool)
        or record_count != len(rows)
    ):
        raise ValueError("validated asset projection record count mismatch")
    raw_counts = summary.get("record_count_by_use")
    if not isinstance(raw_counts, Mapping) or any(
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
        for value in raw_counts.values()
    ):
        raise ValueError("validated asset projection use counts are invalid")
    if set(map(str, raw_counts)) != set(M4_ASSET_SCOPE_USES):
        raise ValueError("validated asset projection use count keys are stale")
    observed = Counter(str(row.get("use")) for row in rows)
    if sum(int(value) for value in raw_counts.values()) != len(rows) or any(
        int(raw_counts.get(use, -1)) != count
        for use, count in observed.items()
    ):
        raise ValueError("validated asset projection use counts mismatch")
    return rows


def build_asset_index(
    *,
    asset_catalog: Mapping[str, Any] | None = None,
    asset_validation: Mapping[str, Any] | None = None,
    attribute_catalog: Mapping[str, Any] | None = None,
    geometry_summary: Mapping[str, Any] | None = None,
    initial_pose_set: Mapping[str, Any] | None = None,
    initial_state_selection: Mapping[str, Any] | None = None,
    semantic_exclusions: Mapping[str, Any] | None = None,
    asset_catalog_path: Path = DEFAULT_ASSET_CATALOG_PATH,
    asset_validation_path: Path = DEFAULT_ASSET_VALIDATION_PATH,
    attribute_catalog_path: Path = DEFAULT_ATTRIBUTE_CATALOG_PATH,
    geometry_path: Path = DEFAULT_GEOMETRY_PATH,
    initial_pose_path: Path = DEFAULT_INITIAL_POSE_PATH,
    initial_state_selection_path: Path | None = None,
    semantic_exclusions_path: Path = DEFAULT_SEMANTIC_EXCLUSIONS_PATH,
) -> AssetIndex:
    """Build the canonical M8 asset query index from frozen artifacts."""

    catalog = dict(asset_catalog or _load_yaml(asset_catalog_path))
    validation = dict(asset_validation or _load_yaml(asset_validation_path))
    if validation.get("schema_version") != ASSET_VALIDATION_SCHEMA_VERSION:
        raise ValueError("validated asset projection schema is stale")
    if validation.get("status") != "validated":
        raise ValueError("asset validation input is not the validated projection")
    if validation.get("projection_hash") != artifact_hash(
        validation, omit_keys=("projection_hash",)
    ):
        raise ValueError("validated asset projection hash mismatch")
    validation_rows = _validated_asset_projection_rows(validation)
    attributes = dict(attribute_catalog or _load_yaml(attribute_catalog_path))
    geometry = dict(geometry_summary or _load_yaml(geometry_path))
    poses = dict(initial_pose_set or _load_yaml(initial_pose_path))
    semantic_policy = (
        dict(semantic_exclusions)
        if semantic_exclusions is not None
        else load_asset_semantic_exclusions(
            semantic_exclusions_path,
            asset_catalog_path=asset_catalog_path,
        )
    )
    excluded_asset_uses = set(expand_asset_semantic_exclusions(semantic_policy))
    validate_initial_pose_set(poses, require_support_validated=True)
    if initial_state_selection is not None:
        selection = dict(initial_state_selection)
        selection_source = "in_memory_canonical_assignment_state_selection"
    elif initial_state_selection_path is not None:
        selection = load_assignment_state_selection(initial_state_selection_path)
        selection_source = str(initial_state_selection_path)
    elif initial_pose_set is None and initial_pose_path == DEFAULT_INITIAL_POSE_PATH:
        selection = load_assignment_state_selection()
        selection_source = str(DEFAULT_ASSIGNMENT_STATE_SELECTION_PATH)
    else:
        selection = None
        selection_source = "legacy_default_profile_selection"
    pose_by_key = {
        _model_key(asset, model_id): row
        for (asset, model_id), row in canonical_assignment_states(
            poses, selection
        ).items()
    }
    selection_identity = canonical_assignment_selection_identity(selection)

    validation_by_key: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in validation_rows:
        _validated_m4_execution_evidence(row)
        _validated_contact_ref(row)
        _validated_articulation_ref(row)
        if (
            str(row["asset"]),
            str(row["model_id"]),
            str(row["use"]),
        ) in excluded_asset_uses:
            continue
        key = _model_key(str(row["asset"]), str(row["model_id"]))
        validation_by_key[key][str(row["use"])] = dict(row)

    attributes_by_key: dict[str, dict[str, str]] = {}
    for asset_name, asset in (attributes.get("assets") or {}).items():
        for model_id, model in (asset.get("models") or {}).items():
            values = {
                str(kind): str(descriptor["value"])
                for kind, descriptor in (model.get("descriptors") or {}).items()
                if descriptor.get("status") == "reviewed"
                and descriptor.get("value") is not None
            }
            attributes_by_key[_model_key(str(asset_name), str(model_id))] = values

    records: dict[str, AssetRecord] = {}
    for asset_name, asset in sorted((catalog.get("assets") or {}).items()):
        for model_id, model in sorted(
            (asset.get("models") or {}).items(), key=lambda item: str(item[0])
        ):
            key = _model_key(str(asset_name), str(model_id))
            initial_state = pose_by_key.get(key)
            if initial_state is None:
                continue
            roles: set[str] = set()
            role_evidence: dict[str, Mapping[str, Any]] = {}
            for candidate_use, candidate in (model.get("candidates") or {}).items():
                asset_role = CANDIDATE_TO_ASSET_ROLE.get(str(candidate_use))
                if asset_role is None:
                    continue
                roles.add(asset_role)
                role_evidence[asset_role] = dict(candidate)
            bounds = initial_state.get("origin_relative_bounds") or {}
            extents = bounds.get("extents_xyz") or []
            initial_geometry = dict((geometry.get("models") or {}).get(key) or {})
            if len(extents) == 3:
                initial_geometry.update(
                    {
                        "xy_footprint": {
                            "x": float(extents[0]),
                            "y": float(extents[1]),
                            "area": float(extents[0]) * float(extents[1]),
                        },
                        "height": float(extents[2]),
                        "origin_relative_bounds": bounds,
                        "geometry_hash": initial_state.get("geometry_hash"),
                        "initial_state_id": initial_state.get("initial_state_id"),
                        "initial_state_revision": initial_state.get("revision"),
                        "support": initial_state.get("support"),
                    }
                )
            record = AssetRecord(
                key=key,
                asset=str(asset_name),
                model_id=str(model_id),
                category=str(model.get("category") or ""),
                asset_roles=tuple(sorted(roles)),
                attributes=attributes_by_key.get(key, {}),
                geometry=initial_geometry,
                validations=validation_by_key.get(key, {}),
                role_evidence=role_evidence,
                initial_state=dict(initial_state),
            )
            records[key] = record

    by_role: dict[str, list[str]] = defaultdict(list)
    by_use: dict[str, list[str]] = defaultdict(list)
    by_category: dict[str, list[str]] = defaultdict(list)
    by_attribute: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for key, record in records.items():
        for role in record.asset_roles:
            by_role[role].append(key)
        for use in record.validations:
            by_use[use].append(key)
        by_category[record.category].append(key)
        for kind, value in record.attributes.items():
            by_attribute[(record.category, kind, value)].append(key)

    source_hashes = {
        str(asset_catalog_path): artifact_hash(catalog),
        str(asset_validation_path): artifact_hash(validation),
        str(attribute_catalog_path): artifact_hash(attributes),
        str(geometry_path): artifact_hash(geometry),
        str(initial_pose_path): str(poses["validated_pose_set_hash"]),
        selection_source: selection_identity["selection_hash"],
        str(semantic_exclusions_path): str(semantic_policy["policy_hash"]),
    }
    return AssetIndex(
        records=records,
        by_role={key: tuple(sorted(value)) for key, value in by_role.items()},
        by_use={key: tuple(sorted(value)) for key, value in by_use.items()},
        by_category={
            key: tuple(sorted(value)) for key, value in by_category.items()
        },
        by_attribute={
            key: tuple(sorted(value)) for key, value in by_attribute.items()
        },
        source_hashes=source_hashes,
    )


def _pair_record(
    row: Mapping[str, Any],
    *,
    stage: str,
    projection_schema_version: int,
    projection_stage: str,
    projection_sha256: str | None,
    source_ledger_hash: str | None,
    receiver_profiles: Mapping[str, Any],
) -> PairRecord:
    subject = row.get("subject") or {}
    receiver = row.get("receiver") or {}
    if stage == "canonical_skill":
        recipe_id = row.get("validated_skill_recipe_id")
        placement_recipe = row.get("validated_skill_placement_recipe")
        band_ids = row.get("validated_skill_band_ids") or []
        evidence_id = row.get("evidence_id")
        evidence_path = row.get("evidence_path")
    else:
        preconditions = row.get("skill_preconditions") or {}
        recipe_id = row.get("validated_recipe_id") or preconditions.get(
            "terminal_recipe_id"
        )
        placement_recipe = row.get("validated_placement_recipe")
        band_ids = row.get("validated_band_ids") or []
        if not band_ids and preconditions.get("terminal_band_id"):
            band_ids = [preconditions["terminal_band_id"]]
        evidence_id = preconditions.get("terminal_evidence_id") or row.get(
            "evidence_id"
        )
        evidence_path = row.get("evidence_path") if row.get("validated_recipe_id") else None
    relation = str(row["relation"])
    if not recipe_id:
        raise ValueError(
            f"validated pair {row.get('pair_key')} lacks recipe identity"
        )
    if isinstance(placement_recipe, Mapping):
        reviewed_profile = None
        if placement_recipe.get("constructor") is not None:
            profile_id = str(
                placement_recipe.get("receiver_state_profile_id") or ""
            )
            reviewed_profile = receiver_profiles.get(profile_id)
            if not isinstance(reviewed_profile, Mapping):
                raise ValueError(
                    f"validated pair {row.get('pair_key')} lacks reviewed "
                    f"receiver profile {profile_id!r}"
                )
        normalized_recipe = validate_terminal_placement_recipe(
            placement_recipe,
            relation,
            expected_recipe_id=str(recipe_id),
            reviewed_receiver_profile=reviewed_profile,
        )
        recipe_source = "frozen_m5_projection"
    else:
        normalized_recipe = rebind_legacy_terminal_placement_recipe(
            row,
            relation,
            stage=stage,
            projection_schema_version=projection_schema_version,
            projection_stage=projection_stage,
            projection_sha256=projection_sha256,
            source_ledger_hash=source_ledger_hash,
            receiver_profiles=receiver_profiles,
        )
        recipe_source = "deterministic_m5_report_v3_recipe_rebind_v2"
    return PairRecord(
        pair_key=str(row["pair_key"]),
        relation=relation,
        subject_key=_model_key(str(subject["asset"]), str(subject["model_id"])),
        receiver_key=_model_key(
            str(receiver["asset"]), str(receiver["model_id"])
        ),
        stage=stage,
        evidence_id=str(evidence_id) if evidence_id else None,
        evidence_path=str(evidence_path) if evidence_path else None,
        recipe_id=str(recipe_id) if recipe_id else None,
        placement_recipe=normalized_recipe,
        placement_recipe_source=recipe_source,
        band_ids=tuple(str(value) for value in band_ids),
        raw=canonical_data(dict(row)),
    )


def _validate_pair_projection_document(
    document: Mapping[str, Any],
    *,
    stage: str,
    artifact_name: str,
    source_path: Path | None,
) -> None:
    """Reject stale or content-tampered M5 pair projections before indexing."""

    schema_version = int(document.get("schema_version") or 0)
    if str(document.get("stage") or "") != stage:
        raise ValueError(f"M5 projection stage mismatch for {stage}")
    if str(document.get("artifact") or "") != artifact_name:
        raise ValueError(f"M5 projection artifact mismatch for {stage}")
    if not str(document.get("source_ledger_hash") or ""):
        raise ValueError(f"M5 projection lacks source ledger hash for {stage}")
    records = document.get("records")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError(f"M5 projection records are invalid for {stage}")
    if schema_version == 3:
        declared = str(document.get("projection_hash") or "")
        observed = artifact_hash(
            {
                key: value
                for key, value in document.items()
                if key != "projection_hash"
            }
        )
        if not declared or declared != observed:
            raise ValueError(f"M5 projection content hash mismatch for {stage}")
        return
    allowed = LEGACY_M5_REPORT_V3_PROJECTIONS.get(stage)
    if (
        schema_version != 2
        or source_path is None
        or not source_path.is_file()
        or not allowed
        or _file_sha256(source_path) != str(allowed["sha256"])
        or str(document.get("source_ledger_hash") or "")
        != str(allowed["source_ledger_hash"])
    ):
        raise ValueError(f"M5 projection is stale or unsupported for {stage}")


def build_pair_index(
    *,
    terminal_pairs: Mapping[str, Any] | None = None,
    skill_pairs: Mapping[str, Any] | None = None,
    terminal_pairs_path: Path = DEFAULT_TERMINAL_PAIRS_PATH,
    skill_pairs_path: Path = DEFAULT_SKILL_PAIRS_PATH,
    receiver_profiles: Mapping[str, Any] | None = None,
    receiver_profiles_path: Path = DEFAULT_RECEIVER_PROFILES_PATH,
) -> PairIndex:
    """Build terminal and canonical-skill pair lookup tables."""

    skill_doc = dict(skill_pairs or _load_yaml(skill_pairs_path))
    if terminal_pairs is not None:
        terminal_doc = dict(terminal_pairs)
    elif terminal_pairs_path == skill_pairs_path:
        terminal_doc = skill_doc
    else:
        terminal_doc = dict(_load_yaml(terminal_pairs_path))
    profile_doc = dict(
        receiver_profiles or _load_yaml(receiver_profiles_path)
    )
    _validate_pair_projection_document(
        terminal_doc,
        stage="terminal_settle",
        artifact_name="validated_terminal_pairs.yml",
        source_path=terminal_pairs_path if terminal_pairs is None else None,
    )
    _validate_pair_projection_document(
        skill_doc,
        stage="canonical_skill",
        artifact_name="validated_skill_pairs.yml",
        source_path=skill_pairs_path if skill_pairs is None else None,
    )
    profiles = profile_doc.get("profiles") or {}
    if not isinstance(profiles, Mapping):
        raise ValueError("articulated receiver profiles must be a mapping")
    terminal: dict[tuple[str, str, str], PairRecord] = {}
    skill: dict[tuple[str, str, str], PairRecord] = {}
    by_relation_terminal: dict[str, list[str]] = defaultdict(list)
    by_relation_skill: dict[str, list[str]] = defaultdict(list)
    terminal_schema_version = int(terminal_doc.get("schema_version") or 0)
    terminal_stage = str(terminal_doc.get("stage") or "")
    skill_schema_version = int(skill_doc.get("schema_version") or 0)
    skill_stage = str(skill_doc.get("stage") or "")
    terminal_projection_sha256 = (
        _file_sha256(terminal_pairs_path) if terminal_pairs is None else None
    )
    skill_projection_sha256 = (
        _file_sha256(skill_pairs_path) if skill_pairs is None else None
    )
    for row in terminal_doc.get("records", []):
        record = _pair_record(
            row,
            stage="terminal_settle",
            projection_schema_version=terminal_schema_version,
            projection_stage=terminal_stage,
            projection_sha256=terminal_projection_sha256,
            source_ledger_hash=str(
                terminal_doc.get("source_ledger_hash") or ""
            ),
            receiver_profiles=profiles,
        )
        key = (record.relation, record.subject_key, record.receiver_key)
        if key in terminal:
            raise ValueError(f"duplicate terminal M5 pair projection key: {key}")
        terminal[key] = record
        by_relation_terminal[record.relation].append(record.pair_key)
    for row in skill_doc.get("records", []):
        record = _pair_record(
            row,
            stage="canonical_skill",
            projection_schema_version=skill_schema_version,
            projection_stage=skill_stage,
            projection_sha256=skill_projection_sha256,
            source_ledger_hash=str(skill_doc.get("source_ledger_hash") or ""),
            receiver_profiles=profiles,
        )
        key = (record.relation, record.subject_key, record.receiver_key)
        if key in skill:
            raise ValueError(f"duplicate skill M5 pair projection key: {key}")
        skill[key] = record
        by_relation_skill[record.relation].append(record.pair_key)
    terminal_hash = (
        artifact_hash(terminal_doc)
        if terminal_pairs is not None
        else _file_sha256(terminal_pairs_path)
    )
    skill_hash = (
        artifact_hash(skill_doc)
        if skill_pairs is not None
        else _file_sha256(skill_pairs_path)
    )
    profile_hash = (
        artifact_hash(profile_doc)
        if receiver_profiles is not None
        else _file_sha256(receiver_profiles_path)
    )
    return PairIndex(
        terminal=terminal,
        skill=skill,
        by_relation_terminal={
            key: tuple(sorted(value))
            for key, value in by_relation_terminal.items()
        },
        by_relation_skill={
            key: tuple(sorted(value))
            for key, value in by_relation_skill.items()
        },
        source_hashes={
            str(terminal_pairs_path): terminal_hash,
            str(skill_pairs_path): skill_hash,
            str(receiver_profiles_path): profile_hash,
        },
    )


class _DisjointSet:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def _has_geometry(record: AssetRecord) -> bool:
    geometry = record.geometry
    footprint = geometry.get("xy_footprint") or {}
    return bool(
        geometry
        and footprint.get("x") is not None
        and footprint.get("y") is not None
        and geometry.get("height") is not None
    )


def _has_articulation_arm_capabilities(
    record: AssetRecord, *, use: str, arms: Sequence[str]
) -> bool:
    evidence = record.validations.get(str(use))
    if not isinstance(evidence, Mapping):
        return False
    binding = evidence.get("articulation_execution_binding")
    if not isinstance(binding, Mapping):
        return False
    return all(
        validated_articulation_arm_capability(
            binding, use=str(use), arm=str(arm)
        )
        is not None
        for arm in arms
    )


def _has_touch_arm_capabilities(
    record: AssetRecord, *, use: str, arms: Sequence[str]
) -> bool:
    evidence = record.validations.get(str(use))
    if not isinstance(evidence, Mapping):
        return False
    binding = evidence.get("touch_execution_binding")
    if not isinstance(binding, Mapping):
        return False
    return all(
        validated_touch_arm_capability(binding, use=str(use), arm=str(arm))
        is not None
        for arm in arms
    )


def _role_domains(
    archetype: Mapping[str, Any],
    asset_index: AssetIndex,
) -> tuple[dict[str, tuple[str, ...]], dict[str, Any]]:
    constraints = list(archetype.get("constraints", []))
    uses_by_role: dict[str, set[str]] = defaultdict(set)
    any_uses_by_role: dict[str, list[tuple[str, ...]]] = defaultdict(list)
    attributes_by_role: dict[str, set[str]] = defaultdict(set)
    articulation_arms_by_role: dict[
        str, dict[str, set[str]]
    ] = defaultdict(lambda: defaultdict(set))
    touch_arms_by_role: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for constraint in constraints:
        if constraint.get("phase") != "assignment":
            continue
        if constraint.get("type") == "use_validated":
            for role in constraint.get("roles", []):
                uses_by_role[str(role)].add(str(constraint["params"]["use"]))
        elif constraint.get("type") == "any_use_validated":
            uses = tuple(
                sorted(
                    {
                        str(use)
                        for use in constraint.get("params", {}).get("uses", [])
                        if str(use)
                    }
                )
            )
            if not uses:
                raise ValueError("any_use_validated requires non-empty uses")
            for role in constraint.get("roles", []):
                any_uses_by_role[str(role)].append(uses)
        elif constraint.get("type") == "articulation_arm_capability":
            use = str(constraint.get("params", {}).get("use") or "")
            arms = {
                str(arm)
                for arm in constraint.get("params", {}).get("arms", [])
            }
            for role in constraint.get("roles", []):
                articulation_arms_by_role[str(role)][use].update(arms)
        elif constraint.get("type") == "touch_arm_capability":
            use = str(constraint.get("params", {}).get("use") or "")
            arms = {
                str(arm)
                for arm in constraint.get("params", {}).get("arms", [])
            }
            for role in constraint.get("roles", []):
                touch_arms_by_role[str(role)][use].update(arms)
        elif constraint.get("type") == "reviewed_attribute":
            for role in constraint.get("roles", []):
                attributes_by_role[str(role)].add(
                    str(constraint["params"]["kind"])
                )
        elif (
            constraint.get("type") == "ordinal_comparable"
            and constraint.get("params", {}).get("basis") == "size"
        ):
            for role in constraint.get("roles", []):
                attributes_by_role[str(role)].add("size")

    domains: dict[str, tuple[str, ...]] = {}
    diagnostics: dict[str, Any] = {}
    for role in archetype.get("roles", []):
        role_id = str(role["id"])
        allowed = set(str(value) for value in role.get("allowed_asset_roles", []))
        required_uses = set(
            str(value) for value in role.get("required_uses", [])
        ) | uses_by_role[role_id]
        required_attributes = {"type"} | attributes_by_role[role_id]
        accepted: list[str] = []
        rejected = Counter()
        for key, record in asset_index.records.items():
            if not (allowed & set(record.asset_roles)):
                rejected["role_membership"] += 1
                continue
            if not required_uses.issubset(record.validations):
                rejected["validated_use"] += 1
                continue
            if any(
                not (set(uses) & set(record.validations))
                for uses in any_uses_by_role[role_id]
            ):
                rejected["any_validated_use"] += 1
                continue
            if any(
                not _has_articulation_arm_capabilities(
                    record, use=use, arms=sorted(arms)
                )
                for use, arms in articulation_arms_by_role[role_id].items()
            ):
                rejected["articulation_arm_capability_missing"] += 1
                continue
            if any(
                not _has_touch_arm_capabilities(
                    record, use=use, arms=sorted(arms)
                )
                for use, arms in touch_arms_by_role[role_id].items()
            ):
                rejected["touch_arm_capability_missing"] += 1
                continue
            if not required_attributes.issubset(record.attributes):
                rejected["reviewed_attribute"] += 1
                continue
            if not _has_geometry(record):
                rejected["geometry_precondition"] += 1
                continue
            accepted.append(key)
        domains[role_id] = tuple(sorted(accepted))
        diagnostics[role_id] = {
            "allowed_asset_roles": sorted(allowed),
            "required_uses": sorted(required_uses),
            "required_any_uses": [
                list(uses) for uses in any_uses_by_role[role_id]
            ],
            "required_attributes": sorted(required_attributes),
            "required_articulation_arm_capabilities": {
                use: sorted(arms)
                for use, arms in sorted(
                    articulation_arms_by_role[role_id].items()
                )
            },
            "required_touch_arm_capabilities": {
                use: sorted(arms)
                for use, arms in sorted(touch_arms_by_role[role_id].items())
            },
            "candidate_count": len(accepted),
            "rejected": dict(sorted(rejected.items())),
        }
    return domains, diagnostics


def _same_asset_groups(
    archetype: Mapping[str, Any],
    role_domains: Mapping[str, Sequence[str]],
) -> tuple[dict[str, tuple[str, ...]], dict[str, str]]:
    roles = list(role_domains)
    dsu = _DisjointSet(roles)
    for constraint in archetype.get("constraints", []):
        if (
            constraint.get("phase") == "assignment"
            and constraint.get("type") == "same_asset_model"
        ):
            constraint_roles = [str(role) for role in constraint.get("roles", [])]
            for role in constraint_roles[1:]:
                dsu.union(constraint_roles[0], role)
    grouped_roles: dict[str, list[str]] = defaultdict(list)
    for role in roles:
        grouped_roles[dsu.find(role)].append(role)
    group_domains: dict[str, tuple[str, ...]] = {}
    role_to_group: dict[str, str] = {}
    for root, members in sorted(grouped_roles.items()):
        group_id = min(members)
        common = set(role_domains[members[0]])
        for member in members[1:]:
            common.intersection_update(role_domains[member])
        group_domains[group_id] = tuple(sorted(common))
        for member in members:
            role_to_group[member] = group_id
    return group_domains, role_to_group


def _constraint_pair(
    constraint: Mapping[str, Any],
    binding: Mapping[str, str],
    pair_index: PairIndex,
    *,
    stage: str,
) -> PairRecord | None:
    roles = [str(role) for role in constraint.get("roles", [])]
    if len(roles) < 2:
        return None
    relation = str(constraint.get("params", {}).get("relation") or "")
    table = pair_index.skill if stage == "canonical_skill" else pair_index.terminal
    return table.get((relation, binding[roles[0]], binding[roles[1]]))


def _evaluate_constraint(
    constraint: Mapping[str, Any],
    binding: Mapping[str, str],
    asset_index: AssetIndex,
    pair_index: PairIndex,
) -> bool:
    roles = [str(role) for role in constraint.get("roles", [])]
    if any(role not in binding for role in roles):
        return True
    records = [asset_index.records[binding[role]] for role in roles]
    constraint_type = str(constraint.get("type"))
    params = constraint.get("params") or {}
    if constraint_type == "all_distinct_entities":
        # Roles reserve different future entity keys.  Reusing one asset/model
        # across those entity instances is intentionally legal.
        return len(roles) == len(set(roles))
    if constraint_type == "same_asset_model":
        return len({record.key for record in records}) == 1
    if constraint_type == "same_category":
        return len({record.category for record in records}) == 1
    if constraint_type == "distinct_category":
        return len({record.category for record in records}) == len(records)
    if constraint_type == "reviewed_attribute":
        kind = str(params.get("kind"))
        return all(kind in record.attributes for record in records)
    if constraint_type == "unique_descriptor":
        kind = str(params.get("kind"))
        peer_roles = [str(role) for role in params.get("peers", roles)]
        if any(role not in binding for role in peer_roles):
            return True
        values = [
            asset_index.records[binding[role]].attributes.get(kind)
            for role in peer_roles
        ]
        return all(value is not None for value in values) and len(set(values)) == len(
            values
        )
    if constraint_type == "ordinal_comparable":
        basis = str(params.get("basis"))
        if basis != "size":
            return True
        values = [record.attributes.get("size") for record in records]
        if any(value not in SIZE_RANK for value in values):
            return False
        return all(
            SIZE_RANK[str(left)] < SIZE_RANK[str(right)]
            for left, right in zip(values, values[1:])
        )
    if constraint_type == "distinct_reference_noun":
        reference_role = str(params.get("reference_role"))
        peer_roles = [str(role) for role in params.get("peers", [])]
        if reference_role not in binding or any(
            role not in binding for role in peer_roles
        ):
            return True
        reference_noun = asset_index.records[binding[reference_role]].noun
        return reference_noun is not None and all(
            asset_index.records[binding[role]].noun != reference_noun
            for role in peer_roles
        )
    if constraint_type == "use_validated":
        use = str(params.get("use"))
        return all(use in record.validations for record in records)
    if constraint_type == "any_use_validated":
        uses = {
            str(use) for use in params.get("uses", []) if str(use)
        }
        return bool(uses) and all(
            bool(uses & set(record.validations)) for record in records
        )
    if constraint_type == "articulation_arm_capability":
        use = str(params.get("use") or "")
        arms = [str(arm) for arm in params.get("arms", [])]
        return all(
            _has_articulation_arm_capabilities(record, use=use, arms=arms)
            for record in records
        )
    if constraint_type == "touch_arm_capability":
        use = str(params.get("use") or "")
        arms = [str(arm) for arm in params.get("arms", [])]
        return all(
            _has_touch_arm_capabilities(record, use=use, arms=arms)
            for record in records
        )
    if constraint_type == "pair_validated":
        return (
            _constraint_pair(
                constraint, binding, pair_index, stage="canonical_skill"
            )
            is not None
        )
    return True


def _unit_preconditions_satisfied(
    archetype: Mapping[str, Any],
    binding: Mapping[str, str],
    asset_index: AssetIndex,
    pair_index: PairIndex,
) -> tuple[bool, str | None]:
    for constraint in archetype.get("constraints", []):
        constraint_type = str(constraint.get("type") or "")
        if constraint_type == "terminal_relation_supported":
            relations = {
                str(value)
                for value in (constraint.get("params") or {}).get(
                    "relations", []
                )
            }
            pair_side_relations = relations & set(
                ANCHOR_PAIR_DISTANCE_BANDS_M
            )
            if pair_side_relations:
                roles = [str(role) for role in constraint.get("roles", [])]
                if len(roles) != 3 or any(role not in binding for role in roles):
                    return False, str(constraint["id"])
                subject_role, left_role, right_role = roles

                def half_x(role: str) -> float | None:
                    record = asset_index.records[binding[role]]
                    footprint = record.geometry.get("xy_footprint") or {}
                    extent = footprint.get("x")
                    if (
                        not isinstance(extent, (int, float))
                        or isinstance(extent, bool)
                        or float(extent) <= 0.0
                    ):
                        return None
                    return float(extent) / 2.0

                subject_half = half_x(subject_role)
                receiver_halves = {
                    "left_of_pair": half_x(left_role),
                    "right_of_pair": half_x(right_role),
                }
                if subject_half is None or any(
                    receiver_halves[relation] is None
                    for relation in pair_side_relations
                ):
                    return False, str(constraint["id"])
                for relation in pair_side_relations:
                    required_distance = (
                        float(receiver_halves[relation])
                        + subject_half
                        + ANCHOR_PAIR_TERMINAL_CLEARANCE_M
                    )
                    if required_distance > ANCHOR_PAIR_DISTANCE_BANDS_M[
                        relation
                    ][1]:
                        return False, str(constraint["id"])
            continue
        if constraint_type != "local_relation":
            continue
        predicates = set(str(value) for value in constraint["params"].get("predicates", []))
        relation = None
        if "in" in predicates:
            relation = "place_in"
        elif "on" in predicates:
            relation = "place_on"
        if relation is None:
            continue
        roles = [str(role) for role in constraint.get("roles", [])]
        key = (relation, binding[roles[0]], binding[roles[1]])
        if key not in pair_index.terminal:
            return False, str(constraint["id"])
    return True, None


def _pair_records_for_binding(
    archetype: Mapping[str, Any],
    binding: Mapping[str, str],
    pair_index: PairIndex,
) -> dict[str, PairRecord]:
    result: dict[str, PairRecord] = {}
    for constraint in archetype.get("constraints", []):
        constraint_id = str(constraint["id"])
        if constraint.get("type") == "pair_validated":
            record = _constraint_pair(
                constraint, binding, pair_index, stage="canonical_skill"
            )
            if record is not None:
                result[constraint_id] = record
        elif constraint.get("type") == "local_relation":
            predicates = set(
                str(value)
                for value in constraint.get("params", {}).get("predicates", [])
            )
            relation = "place_in" if "in" in predicates else "place_on" if "on" in predicates else None
            roles = [str(role) for role in constraint.get("roles", [])]
            if relation is not None:
                record = pair_index.terminal.get(
                    (relation, binding[roles[0]], binding[roles[1]])
                )
                if record is not None:
                    result[constraint_id] = record
    return result


def _binding_digest(family_id: str, binding: Mapping[str, str]) -> str:
    payload = family_id + "|" + "|".join(
        f"{role}={binding[role]}" for role in sorted(binding)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _coverage_tokens(
    family: Mapping[str, Any],
    archetype: Mapping[str, Any],
    binding: Mapping[str, str],
    asset_index: AssetIndex,
    pair_records: Mapping[str, PairRecord],
) -> frozenset[str]:
    tokens: set[str] = set()
    for role, key in binding.items():
        record = asset_index.records[key]
        tokens.add(f"role_model:{role}:{key}")
        tokens.add(f"role_category:{role}:{record.category}")
    for axis in family.get("axis_plan", []):
        if axis.get("axis") != "D1a":
            continue
        kind = str(axis.get("subtype"))
        attribute_roles = {
            str(role)
            for constraint in archetype.get("constraints", [])
            if constraint.get("type") == "reviewed_attribute"
            and str(constraint.get("params", {}).get("kind")) == kind
            for role in constraint.get("roles", [])
        }
        for role in sorted(attribute_roles):
            value = asset_index.records[binding[role]].attributes.get(kind)
            if value is not None:
                tokens.add(f"attribute:{kind}:{value}")
    for record in pair_records.values():
        prefix = f"pair_marginal:{record.stage}:{record.relation}"
        tokens.add(prefix)
        tokens.add(f"{prefix}:subject:{record.subject_key}")
        tokens.add(f"{prefix}:receiver:{record.receiver_key}")
        for band_id in record.band_ids:
            tokens.add(f"{prefix}:band:{band_id}")
    return frozenset(tokens)


def _pair_evidence_keys(
    pair_records: Mapping[str, PairRecord],
) -> frozenset[str]:
    return frozenset(
        f"{record.stage}:{record.pair_key}"
        for record in pair_records.values()
    )


def _greedy_coverage(
    best_by_token: Mapping[str, tuple[str, tuple[tuple[str, str], ...], frozenset[str]]],
    *,
    max_candidates: int,
) -> tuple[list[tuple[tuple[str, str], ...]], set[str], bool]:
    all_tokens = set(best_by_token)
    candidates: dict[
        tuple[tuple[str, str], ...], tuple[str, frozenset[str]]
    ] = {}
    for digest, skeleton, tokens in best_by_token.values():
        existing = candidates.get(skeleton)
        if existing is None or digest < existing[0]:
            candidates[skeleton] = (digest, tokens)
    selected: list[tuple[tuple[str, str], ...]] = []
    uncovered = set(all_tokens)
    while uncovered:
        if len(selected) >= max_candidates:
            return selected, uncovered, False
        ranked = sorted(
            (
                (-len(tokens & uncovered), digest, skeleton, tokens)
                for skeleton, (digest, tokens) in candidates.items()
                if tokens & uncovered
            )
        )
        if not ranked:
            break
        _, _, skeleton, tokens = ranked[0]
        selected.append(skeleton)
        uncovered.difference_update(tokens)
        candidates.pop(skeleton, None)
    return selected, uncovered, not uncovered


def _fact_for_constraint(
    constraint: Mapping[str, Any],
    binding: Mapping[str, str],
    asset_index: AssetIndex,
    pair_records: Mapping[str, PairRecord],
) -> str:
    constraint_id = str(constraint["id"])
    constraint_type = str(constraint["type"])
    roles = [str(role) for role in constraint.get("roles", [])]
    if constraint_type == "all_distinct_entities":
        return f"distinct_entity_instances({','.join(roles)})"
    if constraint_type == "same_asset_model":
        return f"same_asset_model({','.join(roles)},{binding[roles[0]]})"
    if constraint_type in {"same_category", "distinct_category"}:
        categories = [asset_index.records[binding[role]].category for role in roles]
        return f"{constraint_type}({','.join(roles)},{','.join(categories)})"
    if constraint_type in {"reviewed_attribute", "unique_descriptor"}:
        kind = str(constraint.get("params", {}).get("kind"))
        values = [
            f"{role}:{asset_index.records[binding[role]].attributes.get(kind)}"
            for role in roles
        ]
        return f"{constraint_type}({kind},{','.join(values)})"
    if constraint_type == "use_validated":
        return f"validated_use({roles[0]},{constraint['params']['use']})"
    if constraint_type == "any_use_validated":
        options = sorted(
            {
                str(use)
                for use in constraint.get("params", {}).get("uses", [])
                if str(use)
            }
        )
        selected = sorted(
            set(options)
            & set(asset_index.records[binding[roles[0]]].validations)
        )
        return (
            f"validated_any_use({roles[0]},"
            f"options={','.join(options)},available={','.join(selected)})"
        )
    if constraint_type == "touch_arm_capability":
        use = str(constraint["params"]["use"])
        arms = ",".join(str(arm) for arm in constraint["params"]["arms"])
        return f"validated_touch_arms({roles[0]},{use},{arms})"
    if constraint_type == "pair_validated":
        pair = pair_records[constraint_id]
        return f"validated_pair({pair.pair_key},canonical_skill)"
    if constraint_type == "ordinal_comparable":
        basis = str(constraint["params"]["basis"])
        return f"ordinal_comparable({basis},{','.join(roles)})"
    if constraint_type == "distinct_reference_noun":
        reference = str(constraint["params"]["reference_role"])
        noun = asset_index.records[binding[reference]].noun
        return f"distinct_reference_noun({reference}:{noun})"
    return f"{constraint_type}({','.join(roles)})"


def _evidence_refs(
    archetype: Mapping[str, Any],
    binding: Mapping[str, str],
    asset_index: AssetIndex,
    pair_records: Mapping[str, PairRecord],
) -> list[dict[str, Any]]:
    refs: dict[str, dict[str, Any]] = {}
    uses_by_role: dict[str, set[str]] = defaultdict(set)
    attrs_by_role: dict[str, set[str]] = defaultdict(lambda: {"type"})
    for constraint in archetype.get("constraints", []):
        roles = [str(role) for role in constraint.get("roles", [])]
        if constraint.get("type") == "use_validated":
            for role in roles:
                uses_by_role[role].add(str(constraint["params"]["use"]))
        elif constraint.get("type") == "any_use_validated":
            options = {
                str(use)
                for use in constraint.get("params", {}).get("uses", [])
                if str(use)
            }
            for role in roles:
                available = options & set(
                    asset_index.records[binding[role]].validations
                )
                if not available:
                    raise ValueError(
                        f"role {role} has no evidence for any_use_validated"
                    )
                # Preserve every available arm proof. M12 may choose either
                # arm after the object is placed in the complete scene.
                uses_by_role[role].update(available)
        elif constraint.get("type") in {"reviewed_attribute", "unique_descriptor"}:
            kind = str(constraint["params"].get("kind"))
            for role in roles:
                attrs_by_role[role].add(kind)
        elif (
            constraint.get("type") == "ordinal_comparable"
            and constraint.get("params", {}).get("basis") == "size"
        ):
            for role in roles:
                attrs_by_role[role].add("size")
    for role, key in sorted(binding.items()):
        record = asset_index.records[key]
        for asset_role in record.asset_roles:
            evidence = record.role_evidence.get(asset_role) or {}
            candidate_id = str(evidence.get("candidate_id") or f"{key}|{asset_role}")
            ref_key = f"catalog:{candidate_id}"
            refs[ref_key] = {
                "kind": "asset_catalog",
                "record_id": candidate_id,
                "asset": record.asset,
                "model_id": record.model_id,
                "metadata": {"role": role, "asset_role": asset_role},
            }
        refs[f"geometry:{key}"] = {
            "kind": "geometry",
            "path": str(DEFAULT_GEOMETRY_PATH),
            "record_id": key,
            "asset": record.asset,
            "model_id": record.model_id,
        }
        initial_state = record.initial_state
        if initial_state:
            refs[f"initial_pose:{key}"] = {
                "kind": "initial_pose",
                "path": str(DEFAULT_INITIAL_POSE_PATH),
                "record_id": initial_state.get("initial_state_id"),
                "asset": record.asset,
                "model_id": record.model_id,
                "status": "support_validated_human_approved",
                "metadata": {
                    "revision": initial_state.get("revision"),
                    "revision_hash": initial_state.get("revision_hash"),
                    "geometry_hash": initial_state.get("geometry_hash"),
                    "support_validation_hash": initial_state.get(
                        "support_validation_hash"
                    ),
                },
            }
        for kind in sorted(attrs_by_role[role]):
            if kind not in record.attributes:
                continue
            refs[f"attribute:{key}:{kind}"] = {
                "kind": "attribute_review",
                "path": str(DEFAULT_ATTRIBUTE_CATALOG_PATH),
                "record_id": f"{key}|{kind}",
                "asset": record.asset,
                "model_id": record.model_id,
                "field": kind,
                "status": "reviewed",
            }
        for use in sorted(uses_by_role[role]):
            evidence = record.validations[use]
            ref = {
                "kind": "asset_validation",
                "path": evidence.get("evidence_path"),
                "record_id": evidence.get("evidence_id"),
                "asset": record.asset,
                "model_id": record.model_id,
                "use": use,
                "stage": evidence.get("stage"),
                "status": evidence.get("status"),
            }
            metadata: dict[str, Any] = {"role": role}
            if _validated_m4_execution_evidence(evidence):
                ref["hash"] = evidence["evidence_hash"]
            contact_ref = _validated_contact_ref(evidence)
            if contact_ref is not None:
                if not evidence.get("evidence_hash"):
                    raise ValueError(
                        f"validated {use} evidence lacks report hash"
                    )
                ref["hash"] = evidence["evidence_hash"]
                metadata["contact_contract_binding"] = contact_ref
            articulation_ref = _validated_articulation_ref(evidence)
            if articulation_ref is not None:
                if not evidence.get("evidence_hash"):
                    raise ValueError(
                        f"validated {use} evidence lacks report hash"
                    )
                ref["hash"] = evidence["evidence_hash"]
                metadata["articulation_execution_binding"] = articulation_ref
            touch_ref = _validated_touch_ref(evidence)
            if touch_ref is not None:
                if not evidence.get("evidence_hash"):
                    raise ValueError(
                        f"validated {use} evidence lacks report hash"
                    )
                ref["hash"] = evidence["evidence_hash"]
                metadata["touch_execution_binding"] = touch_ref
            ref["metadata"] = metadata
            # One model may instantiate several distinct entity roles.  The
            # M4 proof is reusable, but its lineage remains role-scoped so M10
            # and M11 can prove every operated entity independently.
            refs[f"use:{key}:{use}:{role}"] = ref
    for constraint_id, pair in pair_records.items():
        refs[f"pair:{constraint_id}:{pair.pair_key}"] = {
            "kind": "pair_validation",
            "path": pair.evidence_path,
            "record_id": pair.pair_key,
            "relation": pair.relation,
            "stage": pair.stage,
            "status": "validated",
            "metadata": {
                "constraint_id": constraint_id,
                "evidence_id": pair.evidence_id,
                "recipe_id": pair.recipe_id,
                "placement_recipe": canonical_data(
                    dict(pair.placement_recipe)
                ),
                "placement_recipe_source": pair.placement_recipe_source,
                "band_ids": list(pair.band_ids),
            },
        }
    return [
        {
            field: value
            for field, value in refs[key].items()
            if value is not None
        }
        for key in sorted(refs)
    ]


def _materialize_candidate(
    family: Mapping[str, Any],
    archetype: Mapping[str, Any],
    binding: Mapping[str, str],
    asset_index: AssetIndex,
    pair_index: PairIndex,
    *,
    input_hash: str,
    config_hash: str,
) -> dict[str, Any]:
    pair_records = _pair_records_for_binding(archetype, binding, pair_index)
    identity = {"abstract_family_id": family["id"], "roles": dict(sorted(binding.items()))}
    candidate_id = deterministic_id("ac", identity)
    roles = {}
    resolved_values = {}
    entity_keys = {}
    for role, key in sorted(binding.items()):
        record = asset_index.records[key]
        roles[role] = {
            "asset": record.asset,
            "model_id": record.model_id,
            "category": record.category,
            "roles": list(record.asset_roles),
        }
        resolved_values[role] = {
            "noun": record.noun,
            "attributes": dict(sorted(record.attributes.items())),
            "geometry_ref": key,
        }
        if record.initial_state:
            resolved_values[role]["initial_state_ref"] = deepcopy(
                dict(record.initial_state)
            )
        entity_keys[role] = f"obj_{role}"
    facts = []
    for constraint in archetype.get("constraints", []):
        if constraint.get("phase") != "assignment":
            continue
        facts.append(
            {
                "constraint": constraint["id"],
                "fact": _fact_for_constraint(
                    constraint, binding, asset_index, pair_records
                ),
                "stage": "assignment",
                "status": "proven",
            }
        )
    for constraint_id, pair in pair_records.items():
        constraint = next(
            row
            for row in archetype["constraints"]
            if row["id"] == constraint_id
        )
        if constraint.get("phase") == "unit_build":
            facts.append(
                {
                    "constraint": constraint_id,
                    "fact": f"terminal_pair_precondition({pair.pair_key})",
                    "stage": "unit_build_precondition",
                    "status": "proven",
                }
            )
    layout_requirements = [
        deepcopy(constraint)
        for constraint in archetype.get("constraints", [])
        if constraint.get("phase") in {"unit_build", "layout"}
    ]
    pair_params = {
        constraint_id: {
            "pair_key": pair.pair_key,
            "stage": pair.stage,
            "evidence_id": pair.evidence_id,
            "recipe_id": pair.recipe_id,
            "placement_recipe": canonical_data(dict(pair.placement_recipe)),
            "placement_recipe_source": pair.placement_recipe_source,
            "band_ids": list(pair.band_ids),
        }
        for constraint_id, pair in sorted(pair_records.items())
    }
    return {
        "id": candidate_id,
        "abstract_family_id": family["id"],
        "archetype_ids": list(family["archetypes"]),
        "roles": roles,
        "resolved_values": resolved_values,
        "proven_facts": facts,
        "evidence_refs": _evidence_refs(
            archetype, binding, asset_index, pair_records
        ),
        "layout_requirements": layout_requirements,
        "materialization_params": {
            "entity_key_by_role": entity_keys,
            "pair_preconditions": pair_params,
        },
        "provenance": {
            "generator": "if_benchmark.assignments",
            "config_hash": config_hash,
            "sources": [
                {
                    "kind": "assignment_solver_inputs",
                    "hash": input_hash,
                    "metadata": {"family_id": family["id"]},
                }
            ],
        },
    }


def solve_assignment_domain(
    family: Mapping[str, Any],
    archetype: Mapping[str, Any],
    asset_index: AssetIndex,
    pair_index: PairIndex,
    *,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    source_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Solve, count, select, and materialize one bounded joint domain."""

    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive")
    role_domains, unary_diagnostics = _role_domains(archetype, asset_index)
    group_domains, role_to_group = _same_asset_groups(archetype, role_domains)
    empty_groups = [
        group for group, domain in group_domains.items() if not domain
    ]
    combined_sources = dict(source_hashes or {})
    combined_sources.update(asset_index.source_hashes)
    combined_sources.update(pair_index.source_hashes)
    input_hash = artifact_hash(combined_sources)
    config = {**CONFIG, "max_candidates": max_candidates}
    config_hash = artifact_hash(config)

    if empty_groups:
        result = {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": "assignment_domain",
            "domain_version": DOMAIN_VERSION,
            "abstract_family_id": family["id"],
            "archetype_ids": list(family["archetypes"]),
            "source_hashes": dict(sorted(combined_sources.items())),
            "config_hash": config_hash,
            "feasible_candidate_count": 0,
            "published_candidate_count": 0,
            "truncated": False,
            "selection": {
                "max_candidates": max_candidates,
                "coverage_token_count": 0,
                "covered_token_count": 0,
                "coverage_complete": True,
                "policy": config["selection"],
                "pair_evidence_audit": {
                    "exact_key_coverage_required": False,
                    "feasible_key_count": 0,
                    "feasible_key_universe_hash": artifact_hash([]),
                    "feasible_keys": [],
                    "published_key_count": 0,
                    "published_key_set_hash": artifact_hash([]),
                    "published_keys": [],
                    "unpublished_key_count": 0,
                },
            },
            "constraint_audit": {
                "unary_domains": unary_diagnostics,
                "same_asset_groups": {
                    group: {
                        "roles": sorted(
                            role for role, owner in role_to_group.items() if owner == group
                        ),
                        "candidate_count": len(group_domains[group]),
                    }
                    for group in sorted(group_domains)
                },
                "blocking_groups": empty_groups,
                "rejection_counts": {},
            },
            "candidates": [],
        }
        result["domain_hash"] = artifact_hash(result)
        return result

    assignment_constraints = [
        constraint
        for constraint in archetype.get("constraints", [])
        if constraint.get("phase") == "assignment"
    ]
    constraint_groups: dict[str, set[str]] = {}
    for constraint in assignment_constraints:
        constraint_groups[str(constraint["id"])] = {
            role_to_group[str(role)] for role in constraint.get("roles", [])
        }
    group_order = sorted(
        group_domains,
        key=lambda group: (
            len(group_domains[group]),
            -sum(group in groups for groups in constraint_groups.values()),
            group,
        ),
    )
    rejection_counts: Counter[str] = Counter()
    partial_count = 0
    feasible_count = 0
    all_skeletons: list[tuple[tuple[str, str], ...]] = []
    reservoir: list[tuple[int, str, tuple[tuple[str, str], ...]]] = []
    best_by_token: dict[
        str, tuple[str, tuple[tuple[str, str], ...], frozenset[str]]
    ] = {}
    pair_evidence_universe: set[str] = set()

    def visit(index: int, group_binding: dict[str, str]) -> None:
        nonlocal partial_count, feasible_count
        if index == len(group_order):
            binding = {
                role: group_binding[group]
                for role, group in role_to_group.items()
            }
            unit_ok, blocking_constraint = _unit_preconditions_satisfied(
                archetype, binding, asset_index, pair_index
            )
            if not unit_ok:
                rejection_counts[str(blocking_constraint)] += 1
                return
            feasible_count += 1
            skeleton = tuple(sorted(binding.items()))
            digest = _binding_digest(str(family["id"]), binding)
            pair_records = _pair_records_for_binding(
                archetype, binding, pair_index
            )
            pair_evidence_universe.update(
                _pair_evidence_keys(pair_records)
            )
            tokens = _coverage_tokens(
                family, archetype, binding, asset_index, pair_records
            )
            for token in tokens:
                existing = best_by_token.get(token)
                if existing is None or digest < existing[0]:
                    best_by_token[token] = (digest, skeleton, tokens)
            if feasible_count <= max_candidates:
                all_skeletons.append(skeleton)
            else:
                if feasible_count == max_candidates + 1:
                    for existing_skeleton in all_skeletons:
                        existing_binding = dict(existing_skeleton)
                        existing_digest = _binding_digest(
                            str(family["id"]), existing_binding
                        )
                        heapq.heappush(
                            reservoir,
                            (
                                -int(existing_digest, 16),
                                existing_digest,
                                existing_skeleton,
                            ),
                        )
                    all_skeletons.clear()
                item = (-int(digest, 16), digest, skeleton)
                if len(reservoir) < max_candidates:
                    heapq.heappush(reservoir, item)
                elif item[0] > reservoir[0][0]:
                    heapq.heapreplace(reservoir, item)
            return

        group = group_order[index]
        for key in group_domains[group]:
            group_binding[group] = key
            partial_count += 1
            role_binding = {
                role: group_binding[owner]
                for role, owner in role_to_group.items()
                if owner in group_binding
            }
            failed = None
            for constraint in assignment_constraints:
                groups = constraint_groups[str(constraint["id"])]
                if not groups.issubset(group_binding):
                    continue
                if not _evaluate_constraint(
                    constraint, role_binding, asset_index, pair_index
                ):
                    failed = str(constraint["id"])
                    break
            if failed is not None:
                rejection_counts[failed] += 1
            else:
                visit(index + 1, group_binding)
        group_binding.pop(group, None)

    visit(0, {})

    coverage_complete = True
    capacity_failure: dict[str, Any] | None = None
    uncovered: set[str] = set()
    if feasible_count <= max_candidates:
        selected_skeletons = all_skeletons
    else:
        selected_skeletons, uncovered, coverage_complete = _greedy_coverage(
            best_by_token, max_candidates=max_candidates
        )
        if not coverage_complete:
            capacity_failure = {
                "code": "selection_capacity_exceeded",
                "required_token_count": len(best_by_token),
                "covered_token_count": len(best_by_token) - len(uncovered),
                "uncovered_token_count": len(uncovered),
                "max_candidates": max_candidates,
                "representative_uncovered_tokens": sorted(uncovered)[:25],
            }
        selected_set = set(selected_skeletons)
        for _, _, skeleton in sorted(
            reservoir, key=lambda item: item[1]
        ):
            if len(selected_skeletons) >= max_candidates:
                break
            if skeleton in selected_set:
                continue
            selected_skeletons.append(skeleton)
            selected_set.add(skeleton)

    selected_skeletons = sorted(
        set(selected_skeletons),
        key=lambda skeleton: _binding_digest(str(family["id"]), dict(skeleton)),
    )
    candidates = [
        _materialize_candidate(
            family,
            archetype,
            dict(skeleton),
            asset_index,
            pair_index,
            input_hash=input_hash,
            config_hash=config_hash,
        )
        for skeleton in selected_skeletons
    ]
    covered_tokens: set[str] = set()
    published_pair_evidence_keys: set[str] = set()
    for skeleton in selected_skeletons:
        binding = dict(skeleton)
        selected_pair_records = _pair_records_for_binding(
            archetype, binding, pair_index
        )
        covered_tokens.update(
            _coverage_tokens(
                family,
                archetype,
                binding,
                asset_index,
                selected_pair_records,
            )
        )
        published_pair_evidence_keys.update(
            _pair_evidence_keys(selected_pair_records)
        )
    feasible_pair_keys = sorted(pair_evidence_universe)
    published_pair_keys = sorted(published_pair_evidence_keys)
    result = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "assignment_domain",
        "domain_version": DOMAIN_VERSION,
        "abstract_family_id": family["id"],
        "archetype_ids": list(family["archetypes"]),
        "source_hashes": dict(sorted(combined_sources.items())),
        "config_hash": config_hash,
        "feasible_candidate_count": feasible_count,
        "published_candidate_count": len(candidates),
        "truncated": feasible_count > len(candidates),
        "selection": {
            "max_candidates": max_candidates,
            "coverage_token_count": len(best_by_token),
            "covered_token_count": len(covered_tokens),
            "coverage_complete": coverage_complete
            and len(covered_tokens) == len(best_by_token),
            "uncovered_tokens": sorted(uncovered),
            "policy": config["selection"],
            "pair_evidence_audit": {
                "exact_key_coverage_required": False,
                "feasible_key_count": len(feasible_pair_keys),
                "feasible_key_universe_hash": artifact_hash(
                    feasible_pair_keys
                ),
                "feasible_keys": feasible_pair_keys,
                "published_key_count": len(published_pair_keys),
                "published_key_set_hash": artifact_hash(
                    published_pair_keys
                ),
                "published_keys": published_pair_keys,
                "unpublished_key_count": len(
                    pair_evidence_universe - published_pair_evidence_keys
                ),
            },
            **(
                {"failure": capacity_failure}
                if capacity_failure is not None
                else {}
            ),
        },
        "constraint_audit": {
            "unary_domains": unary_diagnostics,
            "same_asset_groups": {
                group: {
                    "roles": sorted(
                        role for role, owner in role_to_group.items() if owner == group
                    ),
                    "candidate_count": len(group_domains[group]),
                }
                for group in sorted(group_domains)
            },
            "group_order": group_order,
            "partial_assignment_count": partial_count,
            "rejection_counts": dict(sorted(rejection_counts.items())),
        },
        "candidates": candidates,
    }
    result["domain_hash"] = artifact_hash(result)
    if capacity_failure is not None:
        raise SelectionCapacityExceeded(
            f"selection_capacity_exceeded for {family['id']}: "
            f"{len(best_by_token)} tokens cannot fit in "
            f"{max_candidates} candidates",
            result,
        )
    return result


def _family_sources(
    *,
    archetypes_doc: Mapping[str, Any],
    families_doc: Mapping[str, Any],
    archetypes_path: Path,
    families_path: Path,
) -> dict[str, str]:
    return {
        str(archetypes_path): artifact_hash(archetypes_doc),
        str(families_path): artifact_hash(families_doc),
    }


def build_assignment_artifacts(
    *,
    asset_index: AssetIndex | None = None,
    pair_index: PairIndex | None = None,
    archetypes_doc: Mapping[str, Any] | None = None,
    families_doc: Mapping[str, Any] | None = None,
    archetypes_path: Path = DEFAULT_ARCHETYPES_PATH,
    families_path: Path = DEFAULT_FAMILIES_PATH,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> dict[str, Any]:
    """Build all 39 domains plus the summary and empty-domain audit."""

    assets = asset_index or build_asset_index()
    pairs = pair_index or build_pair_index()
    archetypes_value = dict(archetypes_doc or _load_yaml(archetypes_path))
    families_value = dict(families_doc or _load_yaml(families_path))
    archetype_by_id = {
        str(row["id"]): row for row in archetypes_value.get("archetypes", [])
    }
    source_hashes = _family_sources(
        archetypes_doc=archetypes_value,
        families_doc=families_value,
        archetypes_path=archetypes_path,
        families_path=families_path,
    )
    domains: dict[str, dict[str, Any]] = {}
    for family in families_value.get("families", []):
        archetype_id = str(family["archetypes"][0])
        domains[str(family["id"])] = solve_assignment_domain(
            family,
            archetype_by_id[archetype_id],
            assets,
            pairs,
            max_candidates=max_candidates,
            source_hashes=source_hashes,
        )

    rows = [
        _domain_summary_row(domain)
        for _, domain in sorted(domains.items())
    ]
    empty_rows = [
        row
        for _, domain in sorted(domains.items())
        if (row := _empty_domain_row(domain)) is not None
    ]
    summary = _build_domain_summary(
        rows,
        empty_rows,
        max_candidates=max_candidates,
    )
    empty_report = _build_empty_report(
        empty_rows,
        config_hash=summary["config_hash"],
    )
    return {
        "domains": domains,
        SUMMARY_FILE: summary,
        EMPTY_FILE: empty_report,
    }


def _build_domain_summary(
    rows: Sequence[Mapping[str, Any]],
    empty_rows: Sequence[Mapping[str, Any]],
    *,
    max_candidates: int,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "assignment_domain_summary",
        "domain_version": DOMAIN_VERSION,
        "config_hash": artifact_hash({**CONFIG, "max_candidates": max_candidates}),
        "acceptance_ready": all(
            bool(row["coverage_complete"]) for row in rows
        ),
        "summary": {
            "family_count": len(rows),
            "nonempty_family_count": sum(
                bool(row["feasible_candidate_count"]) for row in rows
            ),
            "empty_family_count": len(empty_rows),
            "feasible_candidate_count": sum(
                int(row["feasible_candidate_count"]) for row in rows
            ),
            "published_candidate_count": sum(
                int(row["published_candidate_count"]) for row in rows
            ),
            "truncated_family_count": sum(
                bool(row["truncated"]) for row in rows
            ),
            "coverage_token_count": sum(
                int(row["marginal_coverage"]["coverage_token_count"])
                for row in rows
            ),
            "covered_token_count": sum(
                int(row["marginal_coverage"]["covered_token_count"])
                for row in rows
            ),
            "constraint_rejection_counts": dict(
                sorted(
                    sum(
                        (
                            Counter(row["constraint_rejection_counts"])
                            for row in rows
                        ),
                        Counter(),
                    ).items()
                )
            ),
            "selection_capacity_failure_count": sum(
                "selection_failure" in row for row in rows
            ),
            "selection_capacity_failure_families": [
                row["abstract_family_id"]
                for row in rows
                if "selection_failure" in row
            ],
            "feasible_pair_evidence_key_occurrences": sum(
                int(row["pair_evidence_audit"]["feasible_key_count"])
                for row in rows
            ),
            "published_pair_evidence_key_occurrences": sum(
                int(row["pair_evidence_audit"]["published_key_count"])
                for row in rows
            ),
            "unpublished_pair_evidence_key_occurrences": sum(
                int(row["pair_evidence_audit"]["unpublished_key_count"])
                for row in rows
            ),
        },
        "domains": [dict(row) for row in rows],
        "artifact_hashes": {
            f"{row['abstract_family_id']}.yml": row["domain_hash"]
            for row in rows
        },
    }
    summary["summary_hash"] = artifact_hash(summary)
    return summary


def _build_empty_report(
    empty_rows: Sequence[Mapping[str, Any]],
    *,
    config_hash: str,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "empty_assignment_domain_report",
        "domain_version": DOMAIN_VERSION,
        "config_hash": config_hash,
        "summary": {"empty_family_count": len(empty_rows)},
        "empty_domains": [dict(row) for row in empty_rows],
    }
    report["report_hash"] = artifact_hash(report)
    return report


def _domain_summary_row(domain: Mapping[str, Any]) -> dict[str, Any]:
    row = {
        "abstract_family_id": domain["abstract_family_id"],
        "feasible_candidate_count": domain["feasible_candidate_count"],
        "published_candidate_count": domain["published_candidate_count"],
        "truncated": domain["truncated"],
        "coverage_complete": domain["selection"]["coverage_complete"],
        "marginal_coverage": {
            "coverage_token_count": domain["selection"][
                "coverage_token_count"
            ],
            "covered_token_count": domain["selection"][
                "covered_token_count"
            ],
        },
        "constraint_rejection_counts": domain["constraint_audit"].get(
            "rejection_counts", {}
        ),
        "pair_evidence_audit": {
            key: value
            for key, value in domain["selection"][
                "pair_evidence_audit"
            ].items()
            if key not in {"feasible_keys", "published_keys"}
        },
        "domain_hash": domain["domain_hash"],
    }
    if domain["selection"].get("failure") is not None:
        row["selection_failure"] = domain["selection"]["failure"]
    return row


def _empty_domain_row(domain: Mapping[str, Any]) -> dict[str, Any] | None:
    if domain["feasible_candidate_count"]:
        return None
    audit = domain["constraint_audit"]
    blocking_groups = list(audit.get("blocking_groups", []))
    rejection_counts = dict(audit.get("rejection_counts", {}))
    if blocking_groups:
        stage = "unary"
        blocking_constraint_ids = [
            f"unary_domain:{group}" for group in blocking_groups
        ]
    elif rejection_counts:
        stage = "binary"
        blocking_constraint_ids = sorted(rejection_counts)
    else:
        stage = "complete"
        blocking_constraint_ids = ["complete_assignment_precondition"]
    representative_missing_evidence: list[dict[str, Any]] = []
    for group in blocking_groups[:3]:
        diagnostic = audit.get("unary_domains", {}).get(group, {})
        representative_missing_evidence.append(
            {
                "group": group,
                "required_uses": diagnostic.get("required_uses", []),
                "required_attributes": diagnostic.get(
                    "required_attributes", []
                ),
                "rejected": diagnostic.get("rejected", {}),
            }
        )
    for constraint_id, count in sorted(
        rejection_counts.items(), key=lambda item: (-item[1], item[0])
    )[:3]:
        representative_missing_evidence.append(
            {
                "constraint_id": constraint_id,
                "rejected_assignment_count": count,
            }
        )
    return {
        "abstract_family_id": domain["abstract_family_id"],
        "stage": stage,
        "blocking_constraint_ids": blocking_constraint_ids,
        "blocking_groups": blocking_groups,
        "domain_sizes": {
            "before": {
                role: value.get("candidate_count", 0)
                for role, value in sorted(
                    audit.get("unary_domains", {}).items()
                )
            },
            "after": 0,
        },
        "unary_domains": audit.get("unary_domains", {}),
        "rejection_counts": rejection_counts,
        "representative_missing_evidence": representative_missing_evidence,
    }


_ASSIGNMENT_WORKER_CONTEXT: Mapping[str, Any] = {}


def _init_assignment_worker(context: Mapping[str, Any]) -> None:
    global _ASSIGNMENT_WORKER_CONTEXT
    _ASSIGNMENT_WORKER_CONTEXT = context


def _validated_worker_count(workers: int) -> int:
    worker_count = int(workers)
    if worker_count < 1:
        raise ValueError("workers must be a positive integer")
    return worker_count


def _run_assignment_family_workers(
    function: Callable[[tuple[Mapping[str, Any], str]], dict[str, Any]],
    tasks: Sequence[tuple[Mapping[str, Any], str]],
    *,
    workers: int,
    context: Mapping[str, Any],
) -> Iterable[dict[str, Any]]:
    """Run independent family artifacts and yield source-order results."""

    worker_count = min(_validated_worker_count(workers), max(1, len(tasks)))
    if worker_count == 1:
        _init_assignment_worker(context)
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
        initializer=_init_assignment_worker,
        initargs=(context,),
    ) as executor:
        # M8 families vary greatly in solve cost, so one family per scheduling
        # chunk avoids pinning several expensive families to the same worker.
        yield from executor.map(function, tasks, chunksize=1)


def _freeze_assignment_family(
    task: tuple[Mapping[str, Any], str],
) -> dict[str, Any]:
    family, output_dir_text = task
    family_id = str(family["id"])
    try:
        output_path = Path(output_dir_text) / f"{family_id}.yml"
        expected_sources = _ASSIGNMENT_WORKER_CONTEXT["expected_sources"]
        expected_config_hash = _ASSIGNMENT_WORKER_CONTEXT[
            "expected_config_hash"
        ]
        domain: dict[str, Any] | None = None
        reused = False
        if output_path.is_file():
            existing = _load_yaml(output_path)
            if (
                existing.get("schema_version") == SCHEMA_VERSION
                and existing.get("artifact_kind") == "assignment_domain"
                and existing.get("domain_version") == DOMAIN_VERSION
                and existing.get("abstract_family_id") == family_id
                and existing.get("archetype_ids") == family["archetypes"]
                and existing.get("source_hashes") == expected_sources
                and existing.get("config_hash") == expected_config_hash
                and isinstance(existing.get("domain_hash"), str)
                and existing.get("domain_hash")
                == artifact_hash(existing, omit_keys=("domain_hash",))
            ):
                domain = existing
                reused = True
        if domain is None:
            try:
                domain = solve_assignment_domain(
                    family,
                    _ASSIGNMENT_WORKER_CONTEXT["archetype_by_id"][
                        str(family["archetypes"][0])
                    ],
                    _ASSIGNMENT_WORKER_CONTEXT["assets"],
                    _ASSIGNMENT_WORKER_CONTEXT["pairs"],
                    max_candidates=int(
                        _ASSIGNMENT_WORKER_CONTEXT["max_candidates"]
                    ),
                    source_hashes=_ASSIGNMENT_WORKER_CONTEXT[
                        "family_source_hashes"
                    ],
                )
            except SelectionCapacityExceeded as error:
                domain = error.domain
        if not reused:
            _write_yaml(output_path, domain)
        return {
            "family_id": family_id,
            "row": _domain_summary_row(domain),
            "empty_row": _empty_domain_row(domain),
            "capacity_failure": domain["selection"].get("failure") is not None,
        }
    except Exception as error:
        raise RuntimeError(
            f"M8 freeze failed for family {family_id}: "
            f"{type(error).__name__}: {error}"
        ) from error


def _check_assignment_family(
    task: tuple[Mapping[str, Any], str],
) -> dict[str, Any]:
    family, output_dir_text = task
    family_id = str(family["id"])
    try:
        path = Path(output_dir_text) / f"{family_id}.yml"
        if not path.is_file():
            return {
                "family_id": family_id,
                "row": None,
                "empty_row": None,
                "stale_paths": [str(path)],
            }
        domain = _load_yaml(path)
        archetype = _ASSIGNMENT_WORKER_CONTEXT["archetype_by_id"].get(
            str(family["archetypes"][0])
        )
        if archetype is None:
            return {
                "family_id": family_id,
                "row": None,
                "empty_row": None,
                "stale_paths": [
                    f"missing archetype:{family['archetypes'][0]}"
                ],
            }
        try:
            validate_assignment_domain(
                domain,
                family=family,
                archetype=archetype,
            )
        except ValueError:
            return {
                "family_id": family_id,
                "row": None,
                "empty_row": None,
                "stale_paths": [str(path)],
            }
        stale_paths: list[str] = []
        if (
            domain.get("abstract_family_id") != family_id
            or domain.get("archetype_ids") != family.get("archetypes")
            or domain.get("source_hashes")
            != _ASSIGNMENT_WORKER_CONTEXT["source_hashes"]
            or domain.get("config_hash")
            != _ASSIGNMENT_WORKER_CONTEXT["config_hash"]
            or not isinstance(domain.get("domain_hash"), str)
        ):
            stale_paths.append(str(path))
        return {
            "family_id": family_id,
            "row": _domain_summary_row(domain),
            "empty_row": _empty_domain_row(domain),
            "stale_paths": stale_paths,
        }
    except Exception as error:
        raise RuntimeError(
            f"M8 check failed for family {family_id}: "
            f"{type(error).__name__}: {error}"
        ) from error


def _validate_assignment_family(
    task: tuple[Mapping[str, Any], str],
) -> dict[str, Any]:
    family, output_dir_text = task
    family_id = str(family["id"])
    try:
        path = Path(output_dir_text) / f"{family_id}.yml"
        if not path.is_file():
            return {
                "family_id": family_id,
                "domain_count": 0,
                "candidate_count": 0,
                "row": None,
                "empty_row": None,
                "errors": [f"missing domain artifact: {path}"],
            }
        domain = _load_yaml(path)
        errors: list[str] = []
        if path.read_text(encoding="utf-8") != canonical_yaml(domain):
            errors.append(f"domain is not canonical YAML: {path}")
        errors.extend(
            _validate_domain_envelope(
                domain,
                family=family,
                archetype=_ASSIGNMENT_WORKER_CONTEXT["archetype_by_id"][
                    str(family["archetypes"][0])
                ],
            )
        )
        return {
            "family_id": family_id,
            "domain_count": 1,
            "candidate_count": len(domain.get("candidates", [])),
            "row": _domain_summary_row(domain),
            "empty_row": _empty_domain_row(domain),
            "errors": errors,
        }
    except Exception as error:
        raise RuntimeError(
            f"M8 validate failed for family {family_id}: "
            f"{type(error).__name__}: {error}"
        ) from error


def freeze_assignment_artifacts(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    workers: int = 1,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    assets = build_asset_index()
    pairs = build_pair_index()
    archetypes_doc = _load_yaml(DEFAULT_ARCHETYPES_PATH)
    families_doc = _load_yaml(DEFAULT_FAMILIES_PATH)
    archetype_by_id = {
        str(row["id"]): row for row in archetypes_doc.get("archetypes", [])
    }
    source_hashes = _family_sources(
        archetypes_doc=archetypes_doc,
        families_doc=families_doc,
        archetypes_path=DEFAULT_ARCHETYPES_PATH,
        families_path=DEFAULT_FAMILIES_PATH,
    )
    expected_sources = dict(source_hashes)
    expected_sources.update(assets.source_hashes)
    expected_sources.update(pairs.source_hashes)
    expected_sources = dict(sorted(expected_sources.items()))
    expected_config_hash = artifact_hash(
        {**CONFIG, "max_candidates": max_candidates}
    )
    families = list(families_doc.get("families", []))
    rows: list[dict[str, Any]] = []
    empty_rows: list[dict[str, Any]] = []
    capacity_failures: list[str] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(family, str(output_dir)) for family in families]
    results = _run_assignment_family_workers(
        _freeze_assignment_family,
        tasks,
        workers=workers,
        context={
            "assets": assets,
            "pairs": pairs,
            "archetype_by_id": archetype_by_id,
            "family_source_hashes": source_hashes,
            "expected_sources": expected_sources,
            "expected_config_hash": expected_config_hash,
            "max_candidates": max_candidates,
        },
    )
    for index, result in enumerate(results, start=1):
        family_id = str(result["family_id"])
        rows.append(result["row"])
        if result["empty_row"] is not None:
            empty_rows.append(result["empty_row"])
        if result["capacity_failure"]:
            capacity_failures.append(family_id)
        if progress is not None:
            progress(index, len(families), family_id)
    expected_names = {f"{family['id']}.yml" for family in families}
    for extra in sorted(output_dir.glob("asf_*.yml")):
        if extra.name not in expected_names:
            extra.unlink()
    summary = _build_domain_summary(
        rows, empty_rows, max_candidates=max_candidates
    )
    empty_report = _build_empty_report(
        empty_rows, config_hash=summary["config_hash"]
    )
    _write_yaml(output_dir / SUMMARY_FILE, summary)
    _write_yaml(output_dir / EMPTY_FILE, empty_report)
    return {
        "domains": {},
        SUMMARY_FILE: summary,
        EMPTY_FILE: empty_report,
        "capacity_failures": capacity_failures,
    }


def refresh_assignment_reports(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> dict[str, Any]:
    """Rebuild aggregate reports from already frozen per-family envelopes."""

    families_doc = _load_yaml(DEFAULT_FAMILIES_PATH)
    rows: list[dict[str, Any]] = []
    empty_rows: list[dict[str, Any]] = []
    for family in families_doc.get("families", []):
        path = output_dir / f"{family['id']}.yml"
        if not path.is_file():
            raise ValueError(f"missing assignment domain: {path}")
        domain = _load_yaml(path)
        rows.append(_domain_summary_row(domain))
        empty_row = _empty_domain_row(domain)
        if empty_row is not None:
            empty_rows.append(empty_row)
    summary = _build_domain_summary(
        rows, empty_rows, max_candidates=max_candidates
    )
    empty_report = _build_empty_report(
        empty_rows, config_hash=summary["config_hash"]
    )
    _write_yaml(output_dir / SUMMARY_FILE, summary)
    _write_yaml(output_dir / EMPTY_FILE, empty_report)
    return {SUMMARY_FILE: summary, EMPTY_FILE: empty_report}


def check_assignment_artifacts(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    workers: int = 1,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Check frozen freshness without repeating the full finite-domain solve."""

    assets = build_asset_index()
    pairs = build_pair_index()
    archetypes_doc = _load_yaml(DEFAULT_ARCHETYPES_PATH)
    families_doc = _load_yaml(DEFAULT_FAMILIES_PATH)
    archetype_by_id = {
        str(row["id"]): row
        for row in archetypes_doc.get("archetypes", [])
    }
    source_hashes = _family_sources(
        archetypes_doc=archetypes_doc,
        families_doc=families_doc,
        archetypes_path=DEFAULT_ARCHETYPES_PATH,
        families_path=DEFAULT_FAMILIES_PATH,
    )
    source_hashes.update(assets.source_hashes)
    source_hashes.update(pairs.source_hashes)
    source_hashes = dict(sorted(source_hashes.items()))
    config_hash = artifact_hash(
        {**CONFIG, "max_candidates": max_candidates}
    )
    families = list(families_doc.get("families", []))
    stale_paths: list[str] = []
    rows: list[dict[str, Any]] = []
    empty_rows: list[dict[str, Any]] = []
    tasks = [(family, str(output_dir)) for family in families]
    results = _run_assignment_family_workers(
        _check_assignment_family,
        tasks,
        workers=workers,
        context={
            "archetype_by_id": archetype_by_id,
            "source_hashes": source_hashes,
            "config_hash": config_hash,
        },
    )
    for index, result in enumerate(results, start=1):
        family_id = str(result["family_id"])
        stale_paths.extend(result["stale_paths"])
        if result["row"] is not None:
            rows.append(result["row"])
        if result["empty_row"] is not None:
            empty_rows.append(result["empty_row"])
        if progress is not None:
            progress(index, len(families), family_id)
    expected_names = {f"{family['id']}.yml" for family in families}
    actual_names = {
        path.name
        for path in output_dir.glob("asf_*.yml")
        if path.is_file()
    }
    stale_paths.extend(
        str(output_dir / name) for name in sorted(actual_names - expected_names)
    )
    expected_summary = _build_domain_summary(
        rows, empty_rows, max_candidates=max_candidates
    )
    expected_empty = _build_empty_report(
        empty_rows, config_hash=expected_summary["config_hash"]
    )
    for name, expected in (
        (SUMMARY_FILE, expected_summary),
        (EMPTY_FILE, expected_empty),
    ):
        path = output_dir / name
        if not path.is_file() or canonical_data(_load_yaml(path)) != canonical_data(
            expected
        ):
            stale_paths.append(str(path))
    return {"stale": bool(stale_paths), "stale_paths": stale_paths}


def _validate_domain_envelope(
    domain: Mapping[str, Any],
    *,
    family: Mapping[str, Any],
    archetype: Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []
    required = {
        "schema_version",
        "artifact_kind",
        "domain_version",
        "abstract_family_id",
        "archetype_ids",
        "source_hashes",
        "config_hash",
        "feasible_candidate_count",
        "published_candidate_count",
        "truncated",
        "selection",
        "constraint_audit",
        "candidates",
        "domain_hash",
    }
    missing = sorted(required - set(domain))
    if missing:
        errors.append(f"missing domain fields: {missing}")
        return errors
    if domain["schema_version"] != SCHEMA_VERSION:
        errors.append("domain schema_version is stale")
    if domain["artifact_kind"] != "assignment_domain":
        errors.append("domain artifact_kind is invalid")
    if domain["domain_version"] != DOMAIN_VERSION:
        errors.append("domain_version is stale")
    if domain["abstract_family_id"] != family["id"]:
        errors.append("domain abstract_family_id does not match family")
    if domain["archetype_ids"] != family["archetypes"]:
        errors.append("domain archetype_ids do not match family")
    expected_hash = artifact_hash(
        {key: value for key, value in domain.items() if key != "domain_hash"}
    )
    if domain["domain_hash"] != expected_hash:
        errors.append("domain self hash is invalid")
    if domain["published_candidate_count"] != len(domain["candidates"]):
        errors.append("published candidate count mismatch")
    if domain["feasible_candidate_count"] < domain["published_candidate_count"]:
        errors.append("feasible candidate count is below published count")
    if bool(domain["truncated"]) != (
        domain["feasible_candidate_count"] > domain["published_candidate_count"]
    ):
        errors.append("truncation flag is inconsistent with candidate counts")
    if len(domain["candidates"]) > domain["selection"]["max_candidates"]:
        errors.append("domain exceeds max_candidates")
    if domain["selection"].get("coverage_complete") is not True:
        errors.append("domain has incomplete required coverage")
    pair_audit = domain["selection"].get("pair_evidence_audit") or {}
    feasible_pair_keys = list(pair_audit.get("feasible_keys") or [])
    published_pair_keys = list(pair_audit.get("published_keys") or [])
    if pair_audit.get("exact_key_coverage_required") is not False:
        errors.append("exact pair evidence keys must be audit-only")
    if feasible_pair_keys != sorted(set(feasible_pair_keys)):
        errors.append("feasible pair evidence keys are not canonical")
    if published_pair_keys != sorted(set(published_pair_keys)):
        errors.append("published pair evidence keys are not canonical")
    if not set(published_pair_keys).issubset(feasible_pair_keys):
        errors.append("published pair evidence keys exceed feasible universe")
    if pair_audit.get("feasible_key_count") != len(feasible_pair_keys):
        errors.append("feasible pair evidence key count mismatch")
    if pair_audit.get("published_key_count") != len(published_pair_keys):
        errors.append("published pair evidence key count mismatch")
    if pair_audit.get("unpublished_key_count") != (
        len(feasible_pair_keys) - len(published_pair_keys)
    ):
        errors.append("unpublished pair evidence key count mismatch")
    if pair_audit.get("feasible_key_universe_hash") != artifact_hash(
        feasible_pair_keys
    ):
        errors.append("feasible pair evidence universe hash mismatch")
    if pair_audit.get("published_key_set_hash") != artifact_hash(
        published_pair_keys
    ):
        errors.append("published pair evidence set hash mismatch")
    role_ids = {str(role["id"]) for role in archetype.get("roles", [])}
    constraint_ids = {
        str(constraint["id"]) for constraint in archetype.get("constraints", [])
    }
    candidate_ids: set[str] = set()
    candidate_pair_keys: set[str] = set()
    for index, candidate in enumerate(domain["candidates"]):
        report = validate_document(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": "assignment_candidate",
                "assignment_candidate": candidate,
            }
        )
        if not report["valid"]:
            errors.append(f"candidate {index} schema invalid: {report['errors']}")
            continue
        if candidate["id"] in candidate_ids:
            errors.append(f"duplicate candidate id: {candidate['id']}")
        candidate_ids.add(candidate["id"])
        if candidate["abstract_family_id"] != family["id"]:
            errors.append(
                f"candidate {candidate['id']} family id mismatch"
            )
        if candidate["archetype_ids"] != family["archetypes"]:
            errors.append(
                f"candidate {candidate['id']} archetype ids mismatch"
            )
        if set(candidate["roles"]) != role_ids:
            errors.append(f"candidate {candidate['id']} role set mismatch")
        for fact in candidate["proven_facts"]:
            if fact["constraint"] not in constraint_ids:
                errors.append(
                    f"candidate {candidate['id']} references unknown constraint "
                    f"{fact['constraint']}"
                )
        for params in candidate.get("materialization_params", {}).get(
            "pair_preconditions", {}
        ).values():
            candidate_pair_keys.add(
                f"{params['stage']}:{params['pair_key']}"
            )
    if candidate_pair_keys != set(published_pair_keys):
        errors.append(
            "published pair evidence keys do not match candidate payloads"
        )
    return errors


def validate_assignment_domain(
    domain: Mapping[str, Any],
    *,
    family: Mapping[str, Any],
    archetype: Mapping[str, Any],
) -> None:
    """Reject a stale or tampered M8 envelope before downstream materialization."""

    errors = _validate_domain_envelope(
        domain,
        family=family,
        archetype=archetype,
    )
    if errors:
        raise ValueError("; ".join(errors))


def validate_assignment_artifacts(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    workers: int = 1,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    if not output_dir.is_dir():
        return {"valid": False, "errors": [f"missing artifact dir: {output_dir}"]}
    archetypes_doc = _load_yaml(DEFAULT_ARCHETYPES_PATH)
    families_doc = _load_yaml(DEFAULT_FAMILIES_PATH)
    archetype_by_id = {
        str(row["id"]): row for row in archetypes_doc.get("archetypes", [])
    }
    errors: list[str] = []
    domain_count = 0
    candidate_count = 0
    domain_rows: list[dict[str, Any]] = []
    empty_rows: list[dict[str, Any]] = []
    families = list(families_doc.get("families", []))
    tasks = [(family, str(output_dir)) for family in families]
    results = _run_assignment_family_workers(
        _validate_assignment_family,
        tasks,
        workers=workers,
        context={"archetype_by_id": archetype_by_id},
    )
    for index, result in enumerate(results, start=1):
        family_id = str(result["family_id"])
        domain_count += int(result["domain_count"])
        candidate_count += int(result["candidate_count"])
        if result["row"] is not None:
            domain_rows.append(result["row"])
        if result["empty_row"] is not None:
            empty_rows.append(result["empty_row"])
        errors.extend(result["errors"])
        if progress is not None:
            progress(index, len(families), family_id)
    for name in (SUMMARY_FILE, EMPTY_FILE):
        if not (output_dir / name).is_file():
            errors.append(f"missing artifact: {output_dir / name}")
    if not errors:
        summary = _load_yaml(output_dir / SUMMARY_FILE)
        empty_report = _load_yaml(output_dir / EMPTY_FILE)
        if summary.get("acceptance_ready") is not True:
            errors.append("domain summary is not acceptance_ready")
        expected_summary_hash = artifact_hash(
            {key: value for key, value in summary.items() if key != "summary_hash"}
        )
        if summary.get("summary_hash") != expected_summary_hash:
            errors.append("domain summary self hash is invalid")
        expected_report_hash = artifact_hash(
            {
                key: value
                for key, value in empty_report.items()
                if key != "report_hash"
            }
        )
        if empty_report.get("report_hash") != expected_report_hash:
            errors.append("empty-domain report self hash is invalid")
        if summary.get("domains") != domain_rows:
            errors.append("domain summary rows do not match domain artifacts")
        if summary.get("artifact_hashes") != {
            f"{row['abstract_family_id']}.yml": row["domain_hash"]
            for row in domain_rows
        }:
            errors.append(
                "domain summary artifact hashes do not match domain artifacts"
            )
        if empty_report.get("empty_domains") != empty_rows:
            errors.append(
                "empty-domain report rows do not match domain artifacts"
            )
    return {
        "valid": not errors,
        "errors": errors,
        "domain_count": domain_count,
        "candidate_count": candidate_count,
    }


def select_assignment_candidate(
    domain: Mapping[str, Any],
    assignment_seed: int | str,
) -> dict[str, Any]:
    """Atomically select one complete candidate from a frozen domain."""

    if domain.get("selection", {}).get("coverage_complete") is not True:
        raise ValueError(
            f"assignment domain {domain.get('abstract_family_id')} is not "
            "coverage-complete"
        )
    candidates = list(domain.get("candidates") or [])
    if not candidates:
        raise ValueError(
            f"assignment domain {domain.get('abstract_family_id')} is empty"
        )
    material = (
        f"{domain['abstract_family_id']}|{domain['domain_hash']}|{assignment_seed}"
    )
    index = int(hashlib.sha256(material.encode("utf-8")).hexdigest(), 16) % len(
        candidates
    )
    return deepcopy(candidates[index])
