"""BenchmarkV1 asset-scope validation ledger freezer.

Milestone 4 consumes the Milestone 3 candidate catalog and canonical
asset-scope probe reports.  Static gates are ledger-visible, but executable
uses are promoted to validated only by terminal-independent canonical skill
reports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import yaml

from if_benchmark.asset_semantic_exclusions import (
    DEFAULT_SEMANTIC_EXCLUSIONS_PATH,
    expand_asset_semantic_exclusions,
    load_asset_semantic_exclusions,
    semantic_exclusion_identity,
)
from if_benchmark.asset_validation_contract import (
    ASSET_VALIDATION_REPORT_SCHEMA_VERSION,
    ASSET_VALIDATION_SCHEMA_VERSION,
    LEGACY_CONTACT_REPORT_SCHEMA_VERSION,
    M4_ASSET_SCOPE_USES,
    M4_DISABLED_USES,
    M4_EXECUTABLE_USES,
)
from if_benchmark.asset_probe_evidence import (
    m4_attempt_is_blocking,
    m4_report_attempt_accounting_errors,
    m4_row_attempt_accounting_errors,
)
from if_benchmark.asset_probe_runtime import contact_groups_from_candidate
from if_benchmark.grasp_validation.contact_contract import (
    EXACT_GRASP_POLICY_VERSION,
    GraspContractError,
    build_contact_contract,
    resolved_contact_preflight_policy_version,
)
from if_benchmark.initial_pose.core import (
    DEFAULT_ASSIGNMENT_STATE_SELECTION_PATH,
    InitialPoseContractError,
    assignment_approval_for_state,
    canonical_assignment_selection_identity,
    canonical_assignment_states,
    load_assignment_state_selection,
    validate_canonical_initial_state,
    validate_initial_pose_set,
    validate_pose_binding,
)
from if_benchmark.operation_placement import (
    M4_TOUCH_PLACEMENT_POLICY_HASH,
    M4_TOUCH_PLACEMENT_POLICY_VERSION,
    PICK_OPERATION_PLACEMENT_POLICY_HASH,
    PICK_OPERATION_PLACEMENT_POLICY_VERSION,
    m4_canonical_placement_witness_errors,
)
from if_benchmark.execution_policy import (
    m4_worker_execution_policy_manifest,
    m4_worker_execution_policy_manifest_errors,
)
from if_benchmark.schema.serialization import canonical_data
from if_benchmark.success_contract import SuccessContract
from if_benchmark.articulation_capability import (
    ARTICULATION_ARM_SAMPLING_VERSION,
    ARTICULATION_CAPABILITY_VERSION,
    ARTICULATION_REQUESTED_ARMS,
    LEGACY_ARTICULATION_CAPABILITY_VERSION,
    aggregate_articulation_capabilities,
)
from if_benchmark.touch_capability import (
    TOUCH_ARM_SAMPLING_VERSION,
    TOUCH_CAPABILITY_VERSION,
    TOUCH_EXECUTABLE_USES,
    TOUCH_REQUESTED_ARMS,
    aggregate_touch_capabilities,
    touch_capability_errors,
)
from if_benchmark.scene_validation.articulation_execution import (
    load_articulation_execution_catalog,
    resolve_articulation_execution_profile,
)
from .asset_probe_runner import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_PROBE_SEEDS,
    DEFAULT_PROBE_THRESHOLDS,
    parse_cuda_devices,
    parse_seed_list,
    parse_thresholds,
    run_probe_equivalence_check,
    run_probe_plan,
    DEFAULT_WARMUP_TIMEOUT_SECONDS,
    _summary,
    _success_evidence_failures,
)

SCHEMA_VERSION = ASSET_VALIDATION_SCHEMA_VERSION
REPORT_SCHEMA_VERSION = ASSET_VALIDATION_REPORT_SCHEMA_VERSION
DEFAULT_ASSET_CATALOG_PATH = Path("if_benchmark/assets/asset_catalog.yml")
DEFAULT_LEDGER_PATH = Path("if_benchmark/assets/asset_validation.yml")
DEFAULT_OUTPUT_DIR = Path("if_benchmark/data/asset_validation")
DEFAULT_REPORTS_DIR = DEFAULT_OUTPUT_DIR / "reports"
DEFAULT_INITIAL_POSE_SET_PATH = Path(
    "if_benchmark/data/initial_pose/validated_initial_states.yml"
)
VALIDATED_FILE = "validated_assets.yml"
REJECTED_FILE = "rejected_assets.yml"
MISSING_EVIDENCE_FILE = "missing_evidence_report.yml"
PROBE_PLAN_FILE = "probe_plan.yml"


def _paths_refer_to_same_file(left: Path, right: Path) -> bool:
    """Compare CLI paths by identity instead of relative/absolute spelling."""

    try:
        return left.samefile(right)
    except OSError:
        return left.resolve(strict=False) == right.resolve(strict=False)


STATUS_CANDIDATE = "candidate"
STATUS_VALIDATED = "validated"
STATUS_REJECTED = "rejected"
STATUSES = (STATUS_CANDIDATE, STATUS_VALIDATED, STATUS_REJECTED)

STAGE_STATIC = "static_geometry"
STAGE_ARTICULATED = "articulated_metadata"
STAGE_CANONICAL = "canonical_skill"

USE_PICK_LEFT = "pick_left"
USE_PICK_RIGHT = "pick_right"
USE_HANDOVER = "handover"
USE_OPEN = "open"
USE_CLOSE = "close"
USE_RELATION_ANCHOR = "relation_anchor"
USE_CLICK = "click_subject"
USE_PRESS = "press_subject"
USE_ACTIVATE = "activate_subject"
USE_SWITCH_ON = "switch_on"
USE_SWITCH_OFF = "switch_off"
USE_ROTATE_ORIENTATION = "rotate_to_orientation_subject"
USE_ADJUST_ORIENTATION = "adjust_orientation_subject"
USE_BIMANUAL_GRAB = "bimanual_grab_subject"
EXECUTABLE_USES = M4_EXECUTABLE_USES
ASSET_SCOPE_USES = M4_ASSET_SCOPE_USES

CANDIDATE_USE_MAP: dict[str, tuple[str, ...]] = {
    "pick_subject": (USE_PICK_LEFT, USE_PICK_RIGHT),
    "articulated_open_close": (USE_OPEN, USE_CLOSE),
    "relation_anchor": (USE_RELATION_ANCHOR,),
    USE_CLICK: (USE_CLICK,),
    USE_PRESS: (USE_PRESS,),
    USE_ACTIVATE: (USE_SWITCH_ON, USE_SWITCH_OFF),
    USE_ROTATE_ORIENTATION: (USE_ROTATE_ORIENTATION,),
    USE_ADJUST_ORIENTATION: (USE_ADJUST_ORIENTATION,),
    USE_BIMANUAL_GRAB: (USE_BIMANUAL_GRAB,),
}
M4_NON_EVALUATED_CANDIDATE_USES: dict[str, str] = {
    "handover_subject": USE_HANDOVER,
    **{use: use for use in M4_DISABLED_USES},
}
M4_NON_EVALUATED_USES: dict[str, dict[str, str]] = {
    USE_HANDOVER: {
        "status": "not_evaluated",
        "reason": "m4_capability_explicitly_not_evaluated",
        "candidate_use": "handover_subject",
    },
    **{
        use: {
            "status": "not_evaluated",
            "reason": "m4_capability_explicitly_disabled",
            "candidate_use": use,
        }
        for use in M4_DISABLED_USES
    },
}
HISTORICAL_CANONICAL_USES = (*EXECUTABLE_USES, *M4_NON_EVALUATED_USES)
PAIR_SCOPE_CANDIDATES = (
    "place_in_receiver",
    "place_on_receiver",
    "stack_base",
    "hang_on_pair",
    "dump_into_pair",
    "stamp_on_pair",
)

FRESH_CANONICAL_SOURCE_KIND = "m4_fresh_canonical_skill_probe"
UNIFIED_FROZEN_CANONICAL_SOURCE_KIND = (
    "m4_unified_frozen_canonical_release"
)
CANONICAL_REPORT_SOURCE_KINDS = frozenset(
    {
        FRESH_CANONICAL_SOURCE_KIND,
        UNIFIED_FROZEN_CANONICAL_SOURCE_KIND,
    }
)

M4_EXACT_GRASP_USES: dict[str, tuple[str, str]] = {
    USE_PICK_LEFT: ("pick_subject", "single_arm_pick"),
    USE_PICK_RIGHT: ("pick_subject", "single_arm_pick"),
    USE_BIMANUAL_GRAB: ("bimanual_grab_subject", "bimanual"),
}
M4_ARTICULATION_USES = frozenset(
    (USE_OPEN, USE_CLOSE, USE_SWITCH_ON, USE_SWITCH_OFF)
)

CONFIG: dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "report_schema_version": REPORT_SCHEMA_VERSION,
    "asset_scope_uses": list(ASSET_SCOPE_USES),
    "executable_uses": list(EXECUTABLE_USES),
    "disabled_uses": list(M4_DISABLED_USES),
    "candidate_use_map": {key: list(value) for key, value in CANDIDATE_USE_MAP.items()},
    "non_evaluated_uses": M4_NON_EVALUATED_USES,
    "pair_scope_candidates_deferred_to_m5": list(PAIR_SCOPE_CANDIDATES),
    "static_gates": {
        "pick_left": "contact_points_count > 0 and usable model geometry",
        "pick_right": "contact_points_count > 0 and usable model geometry",
        "relation_anchor": "contact sheet/category evidence and stable geometry",
        "click_subject": "contact_points_count > 0",
        "press_subject": "contact_points_count > 0",
        "switch_on": "articulation metadata plus reviewed switch joint drive annotation",
        "switch_off": "articulation metadata plus reviewed switch joint drive annotation",
        "rotate_to_orientation_subject": "contact points plus usable geometry and resolved orientation evidence",
        "adjust_orientation_subject": "contact points plus usable geometry and resolved orientation evidence",
        "bimanual_grab_subject": "contact_group_count >= 2 and usable model geometry",
    },
    "articulated_metadata_gate": {
        "open": "articulation metadata plus selected joint/contact drive annotation",
        "close": "articulation metadata plus selected joint/contact drive annotation",
        "switch_on": "articulation metadata plus reviewed switch joint/contact drive annotation",
        "switch_off": "articulation metadata plus reviewed switch joint/contact drive annotation",
    },
    "promotion": "executable uses require stage=canonical_skill reports that meet threshold",
    "canonical_probe_source_kind": FRESH_CANONICAL_SOURCE_KIND,
    "canonical_frozen_release_source_kind": (
        UNIFIED_FROZEN_CANONICAL_SOURCE_KIND
    ),
    "pick_operation_placement_policy_version": (
        PICK_OPERATION_PLACEMENT_POLICY_VERSION
    ),
    "pick_operation_placement_policy_hash": (
        PICK_OPERATION_PLACEMENT_POLICY_HASH
    ),
    "worker_execution_policy_manifest_hash": (
        m4_worker_execution_policy_manifest()["manifest_hash"]
    ),
}


def config_hash() -> str:
    return _sha256_bytes(_canonical_json(CONFIG).encode("utf-8"))


def _require_single_m4_release_report(reports_dir: Path) -> Path | None:
    reports = sorted(Path(reports_dir).glob("*.json"))
    if len(reports) > 1:
        raise ValueError(
            "formal M4 freeze/check permits at most one outward canonical "
            "report; publish source runs with "
            "publish_m4_unified_report.py first"
        )
    if not reports:
        # Static-only catalogs can legitimately have no runtime report.  Any
        # runtime-bearing formal release, however, must pass the unified
        # self-contained report gate below.
        return None
    path = reports[0]
    report = _load_json_required(path)
    validate_normalized_report(report, expected_path=path)
    if (
        report.get("schema_version") != REPORT_SCHEMA_VERSION
        or report.get("source_kind")
        != UNIFIED_FROZEN_CANONICAL_SOURCE_KIND
    ):
        raise ValueError(
            "formal M4 freeze/check requires its sole report to be a "
            "schema-v3 self-contained unified release; publish it with "
            "publish_m4_unified_report.py first"
        )
    return path


def freeze_asset_validation(
    asset_catalog_path: Path = DEFAULT_ASSET_CATALOG_PATH,
    ledger_path: Path = DEFAULT_LEDGER_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    semantic_exclusions_path: Path | None = None,
) -> dict[str, Any]:
    _require_single_m4_release_report(reports_dir)
    artifacts = freeze_asset_validation_to_memory(
        asset_catalog_path,
        reports_dir,
        semantic_exclusions_path=semantic_exclusions_path,
    )
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_yaml(ledger_path, artifacts["ledger"])
    _write_yaml(output_dir / VALIDATED_FILE, artifacts["validated_assets"])
    _write_yaml(output_dir / REJECTED_FILE, artifacts["rejected_assets"])
    _write_yaml(output_dir / MISSING_EVIDENCE_FILE, artifacts["missing_evidence_report"])
    return artifacts


def check_asset_validation(
    asset_catalog_path: Path = DEFAULT_ASSET_CATALOG_PATH,
    ledger_path: Path = DEFAULT_LEDGER_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    semantic_exclusions_path: Path | None = None,
) -> dict[str, Any]:
    _require_single_m4_release_report(reports_dir)
    generated = freeze_asset_validation_to_memory(
        asset_catalog_path,
        reports_dir,
        semantic_exclusions_path=semantic_exclusions_path,
    )
    current_ledger = _load_yaml(ledger_path)
    current_validated = _load_yaml(output_dir / VALIDATED_FILE)
    current_rejected = _load_yaml(output_dir / REJECTED_FILE)
    current_missing = _load_yaml(output_dir / MISSING_EVIDENCE_FILE)
    missing = [
        name
        for name, data in (
            ("asset_validation.yml", current_ledger),
            (VALIDATED_FILE, current_validated),
            (REJECTED_FILE, current_rejected),
            (MISSING_EVIDENCE_FILE, current_missing),
        )
        if data is None
    ]
    stale = bool(
        missing
        or _canonical_json(current_ledger) != _canonical_json(generated["ledger"])
        or _canonical_json(current_validated) != _canonical_json(generated["validated_assets"])
        or _canonical_json(current_rejected) != _canonical_json(generated["rejected_assets"])
        or _canonical_json(current_missing) != _canonical_json(generated["missing_evidence_report"])
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "stale": stale,
        "missing_artifacts": missing,
        "current_ledger_hash": current_ledger.get("ledger_hash") if current_ledger else None,
        "generated_ledger_hash": generated["ledger"].get("ledger_hash"),
        "current_validated_hash": current_validated.get("projection_hash") if current_validated else None,
        "generated_validated_hash": generated["validated_assets"].get("projection_hash"),
        "current_rejected_hash": current_rejected.get("projection_hash") if current_rejected else None,
        "generated_rejected_hash": generated["rejected_assets"].get("projection_hash"),
        "current_missing_hash": current_missing.get("report_hash") if current_missing else None,
        "generated_missing_hash": generated["missing_evidence_report"].get("report_hash"),
    }


def freeze_asset_validation_to_memory(
    asset_catalog_path: Path = DEFAULT_ASSET_CATALOG_PATH,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    *,
    semantic_exclusions_path: Path | None = None,
) -> dict[str, Any]:
    catalog = _load_yaml_required(asset_catalog_path)
    records, deferred, non_evaluated = build_static_records(catalog)
    records, contact_contract_blocked = apply_m4_contact_contract_gates(
        catalog,
        records,
        catalog_root=asset_catalog_path.parent,
    )
    semantic_policy = _load_semantic_exclusion_policy(
        asset_catalog_path, semantic_exclusions_path
    )
    records, semantic_exclusions = apply_semantic_exclusions(
        records, semantic_policy
    )
    report_rows, report_index = load_canonical_reports(reports_dir)
    records, report_application = apply_validation(records, report_rows, scope="asset")
    ledger = build_ledger(
        catalog,
        records,
        report_index,
        report_application,
        asset_catalog_path,
        semantic_policy=semantic_policy,
        semantic_exclusions=semantic_exclusions,
        contact_contract_blocked=contact_contract_blocked,
        non_evaluated_capabilities=non_evaluated,
    )
    validated_assets = build_status_projection(ledger, STATUS_VALIDATED, VALIDATED_FILE)
    rejected_assets = build_status_projection(ledger, STATUS_REJECTED, REJECTED_FILE)
    missing_evidence_report = build_missing_evidence_report(
        ledger,
        deferred,
        report_application,
        non_evaluated_capabilities=non_evaluated,
    )
    return {
        "ledger": ledger,
        "validated_assets": validated_assets,
        "rejected_assets": rejected_assets,
        "missing_evidence_report": missing_evidence_report,
    }


def _m4_contact_contract_binding(
    catalog: dict[str, Any],
    *,
    catalog_root: Path,
    asset: str,
    model_id: str,
    use: str,
) -> dict[str, Any] | None:
    profile = M4_EXACT_GRASP_USES.get(str(use))
    if profile is None:
        return None
    candidate_use, use_profile = profile
    model = (
        ((catalog.get("assets") or {}).get(str(asset)) or {})
        .get("models", {})
        .get(str(model_id))
    )
    if not isinstance(model, dict):
        raise ValueError(f"missing catalog model for Contact contract: {asset}:{model_id}")
    candidate = (model.get("candidates") or {}).get(candidate_use)
    if not isinstance(candidate, dict) or not candidate.get("candidate"):
        raise ValueError(
            f"missing {candidate_use} candidate for Contact contract: {asset}:{model_id}"
        )
    metadata_path_value = (
        ((candidate.get("evidence") or {}).get("metadata") or {}).get("path")
    )
    if not metadata_path_value:
        raise ValueError(f"missing Contact metadata path: {asset}:{model_id}")
    metadata_path = Path(str(metadata_path_value))
    if not metadata_path.is_file():
        metadata_path = catalog_root / metadata_path
    metadata = _load_json_required(metadata_path)
    matrices = metadata.get("contact_points_pose")
    if not isinstance(matrices, list) or not matrices:
        raise ValueError(f"missing Contact matrices: {metadata_path}")
    groups = contact_groups_from_candidate(candidate, model_data=metadata)
    contract = build_contact_contract(
        asset=str(asset),
        model_id=str(model_id),
        use=candidate_use,
        contact_matrices=matrices,
        contact_groups=groups,
        use_profile=use_profile,
    )
    return {
        "policy_version": EXACT_GRASP_POLICY_VERSION,
        "contact_contract_id": contract["contact_contract_id"],
        "contact_contract_hash": contract["contact_contract_hash"],
        "contact_projection_hash": contract["contact_projection_hash"],
        "candidate_use": candidate_use,
        "use_profile": use_profile,
        "contact_count": len(matrices),
        "contact_groups": groups,
    }


def plan_probes(
    asset_catalog_path: Path = DEFAULT_ASSET_CATALOG_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    *,
    retry_failed: bool = False,
    include_validated: bool = False,
    asset_models: set[tuple[str, str]] | None = None,
    uses: set[str] | None = None,
    initial_pose_set_path: Path = DEFAULT_INITIAL_POSE_SET_PATH,
    initial_state_selection_path: Path | None = None,
    semantic_exclusions_path: Path | None = None,
) -> dict[str, Any]:
    pose_set = _load_yaml(initial_pose_set_path)
    if not isinstance(pose_set, dict):
        raise InitialPoseContractError(
            "formal M4 planning requires a support_validated_initial_pose_set"
        )
    validate_initial_pose_set(pose_set, require_support_validated=True)
    catalog = _load_yaml(asset_catalog_path)
    if not isinstance(catalog, dict):
        raise ValueError(f"expected asset catalog mapping: {asset_catalog_path}")
    selection = (
        load_assignment_state_selection(initial_state_selection_path)
        if initial_state_selection_path is not None
        else (
            load_assignment_state_selection()
            if _paths_refer_to_same_file(
                initial_pose_set_path, DEFAULT_INITIAL_POSE_SET_PATH
            )
            else None
        )
    )
    states_by_model = canonical_assignment_states(pose_set, selection)
    selection_identity = canonical_assignment_selection_identity(selection)
    selection_path = (
        initial_state_selection_path
        if initial_state_selection_path is not None
        else (
            DEFAULT_ASSIGNMENT_STATE_SELECTION_PATH
            if selection is not None
            else None
        )
    )
    ledger = freeze_asset_validation_to_memory(
        asset_catalog_path,
        reports_dir,
        semantic_exclusions_path=semantic_exclusions_path,
    )["ledger"]
    jobs: list[dict[str, Any]] = []
    skipped = Counter()
    semantic_exclusions = [
        row
        for row in ledger.get("semantic_exclusions") or []
        if (uses is None or row.get("use") in uses)
        and (
            asset_models is None
            or (str(row.get("asset")), str(row.get("model_id")))
            in asset_models
        )
    ]
    contact_contract_blocked = [
        row
        for row in ledger.get("contact_contract_blocked") or []
        if (uses is None or row.get("use") in uses)
        and (
            asset_models is None
            or (str(row.get("asset")), str(row.get("model_id")))
            in asset_models
        )
    ]
    non_evaluated_capabilities = [
        row
        for row in ledger.get("non_evaluated_capabilities") or []
        if (uses is None or row.get("use") in uses)
        and (
            asset_models is None
            or (str(row.get("asset")), str(row.get("model_id")))
            in asset_models
        )
    ]
    skipped["semantic_unsupported"] = len(semantic_exclusions)
    skipped["contact_contract_blocked"] = len(contact_contract_blocked)
    for asset_name, asset in sorted((ledger.get("assets") or {}).items()):
        for model_id, model in sorted((asset.get("models") or {}).items(), key=lambda item: _natural_key(item[0])):
            for use, record in sorted((model.get("uses") or {}).items(), key=lambda item: _natural_key(item[0])):
                if use not in EXECUTABLE_USES:
                    continue
                if uses is not None and use not in uses:
                    continue
                if asset_models is not None and (asset_name, str(model_id)) not in asset_models:
                    continue
                if not _should_plan_probe(
                    record,
                    retry_failed=retry_failed,
                    include_validated=include_validated,
                ):
                    continue
                initial_state = states_by_model.get((asset_name, str(model_id)))
                if initial_state is None:
                    skipped["outside_initial_pose_role_universe"] += 1
                    continue
                if initial_state.get("status") == "unsupported":
                    skipped["human_unsupported"] += 1
                    continue
                support = initial_state.get("support_validation") or {}
                if support.get("status") != "validated":
                    raise InitialPoseContractError(
                        f"Initial Pose support evidence missing: {asset_name}|{model_id}"
                    )
                validate_canonical_initial_state(initial_state)
                approval = assignment_approval_for_state(
                    pose_set,
                    initial_state,
                    (selection.get("selections") or {}).get(
                        f"{asset_name}|{model_id}"
                    )
                    if selection is not None
                    else None,
                )
                binding = {
                    "initial_state_id": initial_state["initial_state_id"],
                    "revision": initial_state["revision"],
                    "approval_hash": approval["approval_hash"],
                    "geometry_hash": initial_state["geometry_hash"],
                }
                validate_pose_binding(binding, initial_state, approval)
                contact_binding = _m4_contact_contract_binding(
                    catalog,
                    catalog_root=asset_catalog_path.parent,
                    asset=asset_name,
                    model_id=str(model_id),
                    use=use,
                )
                candidate_use = next(
                    candidate_name
                    for candidate_name, mapped_uses in CANDIDATE_USE_MAP.items()
                    if use in mapped_uses
                )
                job = {
                        "job_id": _probe_job_id(asset_name, str(model_id), use),
                        "scope": "asset",
                        "stage": STAGE_CANONICAL,
                        "asset": asset_name,
                        "model_id": str(model_id),
                        "use": use,
                        "candidate_use": candidate_use,
                        "minimal_scene": _minimal_scene_kind(use),
                        "candidate_id": record.get("source_candidate_id"),
                        "expected_report_schema": REPORT_SCHEMA_VERSION,
                        "initial_pose_binding": binding,
                        "canonical_initial_state": initial_state,
                        "initial_state_selection_version": selection_identity[
                            "selection_version"
                        ],
                        "initial_state_selection_hash": selection_identity[
                            "selection_hash"
                        ],
                    }
                if use in M4_ARTICULATION_USES:
                    job["articulation_arm_sampling_version"] = (
                        ARTICULATION_ARM_SAMPLING_VERSION
                    )
                    job["requested_arms"] = list(ARTICULATION_REQUESTED_ARMS)
                elif use in TOUCH_EXECUTABLE_USES:
                    job["touch_arm_sampling_version"] = (
                        TOUCH_ARM_SAMPLING_VERSION
                    )
                    job["touch_placement_policy_version"] = (
                        M4_TOUCH_PLACEMENT_POLICY_VERSION
                    )
                    job["touch_placement_policy_hash"] = (
                        M4_TOUCH_PLACEMENT_POLICY_HASH
                    )
                    job["requested_arms"] = list(TOUCH_REQUESTED_ARMS)
                if contact_binding is not None:
                    job["contact_contract_binding"] = contact_binding
                    job["contact_contract_hash"] = contact_binding[
                        "contact_contract_hash"
                    ]
                jobs.append(job)
    contact_contract_hashes = sorted(
        {
            str(job["contact_contract_hash"])
            for job in jobs
            if job.get("contact_contract_hash")
        }
    )
    plan = {
        "schema_version": SCHEMA_VERSION,
        "config_hash": config_hash(),
        "pick_operation_placement_policy_version": (
            PICK_OPERATION_PLACEMENT_POLICY_VERSION
        ),
        "pick_operation_placement_policy_hash": (
            PICK_OPERATION_PLACEMENT_POLICY_HASH
        ),
        "source_ledger_hash": ledger.get("ledger_hash"),
        "semantic_exclusion_policy": ledger.get("semantic_exclusion_policy"),
        "semantic_exclusions": semantic_exclusions,
        "contact_contract_blocked": contact_contract_blocked,
        "non_evaluated_capabilities": non_evaluated_capabilities,
        "source_validated_pose_set_path": str(initial_pose_set_path),
        "source_validated_pose_set_hash": pose_set["validated_pose_set_hash"],
        "source_initial_state_selection_path": (
            str(selection_path) if selection_path is not None else None
        ),
        "initial_state_selection_version": selection_identity["selection_version"],
        "initial_state_selection_hash": selection_identity["selection_hash"],
        "contact_contract_hashes": contact_contract_hashes,
        "articulation_arm_sampling_version": (
            ARTICULATION_ARM_SAMPLING_VERSION
        ),
        "touch_arm_sampling_version": TOUCH_ARM_SAMPLING_VERSION,
        "touch_placement_policy_version": M4_TOUCH_PLACEMENT_POLICY_VERSION,
        "touch_placement_policy_hash": M4_TOUCH_PLACEMENT_POLICY_HASH,
        "summary": {
            "probe_job_count": len(jobs),
            "probe_job_count_by_use": dict(sorted(Counter(job["use"] for job in jobs).items())),
            "probe_attempt_count": sum(
                len(DEFAULT_PROBE_SEEDS)
                * (
                    len(ARTICULATION_REQUESTED_ARMS)
                    if job["use"] in M4_ARTICULATION_USES
                    else len(TOUCH_REQUESTED_ARMS)
                    if job["use"] in TOUCH_EXECUTABLE_USES
                    else 1
                )
                for job in jobs
            ),
            "retry_failed": bool(retry_failed),
            "include_validated": bool(include_validated),
            "exact_contact_job_count": sum(
                bool(job.get("contact_contract_hash")) for job in jobs
            ),
            "contact_contract_count": len(contact_contract_hashes),
            "skipped_initial_pose_counts": dict(sorted(skipped.items())),
            "semantic_unsupported_count": len(semantic_exclusions),
            "contact_contract_blocked_count": len(contact_contract_blocked),
            "non_evaluated_capability_count": len(
                non_evaluated_capabilities
            ),
            "non_evaluated_uses": sorted(M4_NON_EVALUATED_USES),
        },
        "jobs": jobs,
    }
    plan["probe_plan_hash"] = _artifact_hash(plan, "probe_plan_hash")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_yaml(output_dir / PROBE_PLAN_FILE, plan)
    return plan


def apply_report_file(
    report_path: Path,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    *,
    copy_source: bool = False,
) -> dict[str, Any]:
    report = normalize_report_file(report_path)
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / f"{report['evidence_id']}.json"
    if not out_path.exists():
        _write_json(out_path, report)
    if copy_source:
        source_dir = reports_dir / "sources"
        source_dir.mkdir(parents=True, exist_ok=True)
        target = source_dir / report_path.name
        if not target.exists() and report_path.resolve() != target.resolve():
            shutil.copyfile(report_path, target)
    return {
        "evidence_id": report["evidence_id"],
        "evidence_path": _path_string(out_path),
        "record_count": len(report.get("records") or []),
        "created": True,
    }


def clear_canonical_reports(reports_dir: Path = DEFAULT_REPORTS_DIR) -> int:
    """Delete normalized canonical report artifacts from the M4 reports dir."""

    if not reports_dir.is_dir():
        return 0
    count = 0
    for path in sorted(reports_dir.glob("*.json")):
        path.unlink()
        count += 1
    return count


def build_static_records(
    catalog: dict[str, Any],
) -> tuple[
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    records: dict[str, dict[str, Any]] = {}
    deferred: list[dict[str, Any]] = []
    non_evaluated: list[dict[str, Any]] = []
    for candidate in catalog.get("pair_candidates") or []:
        deferred.append(_deferred_pair_candidate(candidate, reason="pair_scope_deferred_to_milestone_5"))
    for asset_name, asset in sorted((catalog.get("assets") or {}).items()):
        for model_id, model in sorted((asset.get("models") or {}).items(), key=lambda item: _natural_key(item[0])):
            for candidate_use, candidate in sorted((model.get("candidates") or {}).items(), key=lambda item: _natural_key(item[0])):
                if candidate_use in M4_NON_EVALUATED_CANDIDATE_USES:
                    use = M4_NON_EVALUATED_CANDIDATE_USES[candidate_use]
                    non_evaluated.append(
                        {
                            "asset": asset_name,
                            "model_id": str(model_id),
                            "use": use,
                            "candidate_use": candidate_use,
                            "candidate_id": candidate.get("candidate_id"),
                            **M4_NON_EVALUATED_USES[use],
                        }
                    )
                    continue
                if candidate_use in PAIR_SCOPE_CANDIDATES:
                    deferred.append(
                        {
                            "candidate_id": candidate.get("candidate_id"),
                            "asset": asset_name,
                            "model_id": str(model_id),
                            "candidate_use": candidate_use,
                            "reason": "pair_scope_deferred_to_milestone_5",
                        }
                    )
                    continue
                for use in CANDIDATE_USE_MAP.get(candidate_use, ()):
                    record = _static_record(asset_name, str(model_id), use, candidate)
                    records[_record_key(asset_name, str(model_id), use)] = record
    return records, deferred, non_evaluated


def apply_semantic_exclusions(
    records: dict[str, dict[str, Any]],
    policy: dict[str, Any] | None,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Mark reviewed semantic exclusions without treating them as skill failures."""

    out = {key: dict(value) for key, value in records.items()}
    applied: list[dict[str, Any]] = []
    if policy is None:
        return out, applied
    for (asset, model_id, use), exclusion in sorted(
        expand_asset_semantic_exclusions(policy).items(),
        key=lambda item: _record_sort_key("|".join(item[0])),
    ):
        key = _record_key(asset, model_id, use)
        current = out.get(key)
        if current is None:
            raise ValueError(
                f"semantic exclusion has no asset-scope candidate: {key}"
            )
        record = dict(current)
        record.update(
            {
                "status": STATUS_REJECTED,
                "disposition": "unsupported",
                "reason": f"semantic_unsupported: {exclusion['reason_code']}",
                "semantic_exclusion": dict(exclusion),
            }
        )
        out[key] = record
        applied.append(dict(exclusion))
    return out, applied


def apply_m4_contact_contract_gates(
    catalog: dict[str, Any],
    records: dict[str, dict[str, Any]],
    *,
    catalog_root: Path,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Reject exact-grasp uses whose immutable R02 evidence is malformed.

    One bad Contact Group must not abort construction of the complete M4 plan.
    The affected asset/model/use remains explicitly accounted for as a source
    evidence blocker and cannot be promoted by an old runtime report.
    """

    out = {key: dict(value) for key, value in records.items()}
    blocked: list[dict[str, Any]] = []
    for key, current in sorted(out.items(), key=lambda item: _record_sort_key(item[0])):
        asset, model_id, use = key.split("|", 2)
        if use not in M4_EXACT_GRASP_USES:
            continue
        if current.get("status") != STATUS_CANDIDATE:
            continue
        try:
            binding = _m4_contact_contract_binding(
                catalog,
                catalog_root=catalog_root,
                asset=asset,
                model_id=model_id,
                use=use,
            )
        except (GraspContractError, ValueError) as exc:
            failure = {
                "asset": asset,
                "model_id": model_id,
                "use": use,
                "disposition": "source_blocked",
                "reason_code": "contact_contract_invalid",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            record = dict(current)
            record.update(
                {
                    "status": STATUS_REJECTED,
                    "disposition": "source_blocked",
                    "reason": f"contact_contract_invalid: {exc}",
                    "contact_contract_gate": failure,
                }
            )
            out[key] = record
            blocked.append(failure)
            continue
        if binding is not None:
            record = dict(current)
            record["contact_contract_binding"] = binding
            record["contact_contract_hash"] = binding[
                "contact_contract_hash"
            ]
            out[key] = record
    return out, blocked


def _source_probe_plan(report: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact self-hashed plan committed by one M4 report.

    A live probe report joins its on-disk run plan.  A frozen unified release
    embeds the original self-hashed plan partitions and an explicit key
    selection, so the release JSON is the sole runtime dependency for
    M5--M15 without pretending the source runs were one execution.
    """

    source_kind = str(report.get("source_kind") or "")
    if source_kind == UNIFIED_FROZEN_CANONICAL_SOURCE_KIND:
        if report.get("source_probe_plan_path"):
            raise ValueError(
                "unified canonical report must not depend on an external "
                "source probe plan"
            )
        partitions = report.get("source_probe_plan_partitions")
        if not isinstance(partitions, list) or not partitions:
            raise ValueError(
                "unified canonical report lacks embedded source plan "
                "partitions"
            )
        selected_jobs: list[dict[str, Any]] = []
        selected_keys: set[str] = set()
        partition_index: list[dict[str, Any]] = []
        for index, partition in enumerate(partitions):
            if not isinstance(partition, dict):
                raise ValueError(
                    "unified canonical source plan partition is invalid"
                )
            embedded = partition.get("source_probe_plan")
            if not isinstance(embedded, dict):
                raise ValueError(
                    "unified canonical source plan partition lacks plan"
                )
            observed_source_hash = _artifact_hash(
                embedded, "probe_plan_hash"
            )
            if (
                embedded.get("probe_plan_hash") != observed_source_hash
                or partition.get("source_probe_plan_hash")
                != observed_source_hash
            ):
                raise ValueError(
                    "unified canonical source plan partition hash mismatch"
                )
            if (
                embedded.get("pick_operation_placement_policy_version")
                != PICK_OPERATION_PLACEMENT_POLICY_VERSION
                or embedded.get("pick_operation_placement_policy_hash")
                != PICK_OPERATION_PLACEMENT_POLICY_HASH
            ):
                raise ValueError(
                    "unified canonical source plan partition placement is "
                    "stale"
                )
            raw_keys = partition.get("selected_record_keys")
            if not isinstance(raw_keys, list) or not raw_keys:
                raise ValueError(
                    "unified canonical source plan partition selected no "
                    "records"
                )
            keys = [str(key) for key in raw_keys]
            if (
                partition.get("selection_order") != index
                or keys != sorted(keys, key=_record_sort_key)
                or len(keys) != len(set(keys))
                or partition.get("selected_record_count") != len(keys)
                or partition.get("selected_record_key_hash")
                != _m4_identity_digest(keys)
            ):
                raise ValueError(
                    "unified canonical source plan partition selection is "
                    "invalid"
                )
            overlap = selected_keys.intersection(keys)
            if overlap:
                raise ValueError(
                    "unified canonical source plan partitions overlap"
                )
            selected_keys.update(keys)
            jobs_by_key: dict[str, list[dict[str, Any]]] = {}
            for job in embedded.get("jobs") or []:
                if not isinstance(job, dict):
                    raise ValueError(
                        "unified canonical source plan contains invalid job"
                    )
                jobs_by_key.setdefault(
                    _m4_identity_string(job), []
                ).append(job)
            for key in keys:
                jobs = jobs_by_key.get(key) or []
                if len(jobs) != 1:
                    raise ValueError(
                        "unified canonical source plan selection requires "
                        f"one source job: {key}"
                    )
                selected_jobs.append(jobs[0])
            partition_index.append(
                {
                    "selection_order": index,
                    "source_evidence_id": partition.get(
                        "source_evidence_id"
                    ),
                    "source_report_hash": partition.get(
                        "source_report_hash"
                    ),
                    "source_probe_plan_hash": observed_source_hash,
                    "selected_record_count": len(keys),
                    "selected_record_key_hash": _m4_identity_digest(keys),
                }
            )
        selected_jobs.sort(
            key=lambda job: _record_sort_key(_m4_identity_string(job))
        )
        plan = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "kind": "m4_partitioned_source_probe_plan_index",
            "pick_operation_placement_policy_version": (
                PICK_OPERATION_PLACEMENT_POLICY_VERSION
            ),
            "pick_operation_placement_policy_hash": (
                PICK_OPERATION_PLACEMENT_POLICY_HASH
            ),
            "partitions": partition_index,
            "jobs": selected_jobs,
        }
        plan["probe_plan_hash"] = _artifact_hash(
            plan, "probe_plan_hash"
        )
        if report.get("source_probe_plan_hash") != plan[
            "probe_plan_hash"
        ]:
            raise ValueError(
                "unified canonical report/source plan partition join "
                "mismatch"
            )
        return plan
    elif source_kind == FRESH_CANONICAL_SOURCE_KIND:
        raw_path = report.get("source_probe_plan_path")
        if not raw_path:
            raise ValueError("canonical report lacks source probe plan path")
        path = Path(str(raw_path))
        loaded = _load_yaml(path)
        if not isinstance(loaded, dict):
            raise ValueError(f"canonical source probe plan is missing: {path}")
        plan = loaded
    else:
        raise ValueError("canonical report source kind is invalid")
    observed_hash = _artifact_hash(plan, "probe_plan_hash")
    if plan.get("probe_plan_hash") != observed_hash:
        raise ValueError("canonical source probe plan hash mismatch")
    if report.get("source_probe_plan_hash") != observed_hash:
        raise ValueError("canonical report/source probe plan join mismatch")
    if (
        plan.get("pick_operation_placement_policy_version")
        != PICK_OPERATION_PLACEMENT_POLICY_VERSION
        or plan.get("pick_operation_placement_policy_hash")
        != PICK_OPERATION_PLACEMENT_POLICY_HASH
    ):
        raise ValueError("canonical source probe plan placement is stale")
    return plan


def _source_probe_plan_jobs(
    report: Mapping[str, Any],
) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    """Load the exact self-hashed M4 source plan and index its jobs."""

    plan = _source_probe_plan(report)
    by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for job in plan.get("jobs") or []:
        if not isinstance(job, dict):
            raise ValueError("legacy source probe plan contains invalid job")
        key = (
            str(job.get("asset")),
            str(job.get("model_id")),
            str(job.get("use")),
        )
        by_key.setdefault(key, []).append(job)
    return by_key


def _m4_identity_string(row: Mapping[str, Any]) -> str:
    return (
        f"{row.get('asset')}|{row.get('model_id')}|{row.get('use')}"
    )


def _m4_identity_digest(keys: list[str]) -> str:
    return _sha256_bytes(_canonical_json(keys).encode("utf-8"))


def _validate_unified_release_metadata(
    report: Mapping[str, Any],
    *,
    records: list[dict[str, Any]],
    attempts: list[dict[str, Any]],
    source_plan_jobs: Mapping[
        tuple[str, str, str], list[dict[str, Any]]
    ],
) -> None:
    """Validate the self-contained single-report M4 release envelope."""

    release = report.get("unified_release")
    if not isinstance(release, dict):
        raise ValueError("unified canonical report lacks release metadata")
    record_keys = sorted(
        (_m4_identity_string(row) for row in records),
        key=_record_sort_key,
    )
    if len(record_keys) != len(set(record_keys)):
        raise ValueError("unified canonical report has duplicate record keys")
    if any(
        str(row.get("use") or "") not in EXECUTABLE_USES
        for row in records
    ):
        raise ValueError(
            "unified canonical report may contain only enabled M4 uses"
        )
    stale_touch_keys = sorted(
        _m4_identity_string(row)
        for row in records
        if str(row.get("use") or "") in TOUCH_EXECUTABLE_USES
        and not _touch_row_uses_current_contract(row)
    )
    if stale_touch_keys:
        raise ValueError(
            "unified canonical report contains stale touch capability rows: "
            + ", ".join(stale_touch_keys[:3])
        )
    stale_articulation_keys = sorted(
        _m4_identity_string(row)
        for row in records
        if str(row.get("use") or "") in M4_ARTICULATION_USES
        and row.get("articulation_capability_version")
        != ARTICULATION_CAPABILITY_VERSION
    )
    if stale_articulation_keys:
        raise ValueError(
            "unified canonical report contains stale articulation "
            "capability rows: "
            + ", ".join(stale_articulation_keys[:3])
        )
    plan_keys: list[str] = []
    for key, jobs in source_plan_jobs.items():
        if len(jobs) != 1:
            raise ValueError(
                "unified source plan requires one job per record key"
            )
        plan_keys.append(f"{key[0]}|{key[1]}|{key[2]}")
    plan_keys.sort(key=_record_sort_key)
    if plan_keys != record_keys:
        raise ValueError(
            "unified source plan jobs do not exactly match report records"
        )

    attempt_count_by_key = Counter(
        _m4_identity_string(attempt) for attempt in attempts
    )
    attempt_ids = [str(attempt.get("attempt_id") or "") for attempt in attempts]
    if (
        any(not attempt_id for attempt_id in attempt_ids)
        or len(attempt_ids) != len(set(attempt_ids))
    ):
        raise ValueError(
            "unified canonical report attempt ids must be present and unique"
        )
    if report.get("summary") != _summary(records, attempts):
        raise ValueError("unified canonical report summary is inconsistent")
    expected_release = {
        "enabled_uses": list(EXECUTABLE_USES),
        "record_count": len(records),
        "attempt_count": len(attempts),
        "source_plan_job_count": len(plan_keys),
        "record_key_hash": _m4_identity_digest(record_keys),
    }
    for field, expected in expected_release.items():
        if release.get(field) != expected:
            raise ValueError(
                f"unified canonical release {field} is inconsistent"
            )
    if release.get("merge_policy") != "last_source_record_key_wins":
        raise ValueError(
            "unified canonical release merge_policy is inconsistent"
        )

    lineage = report.get("source_report_lineage")
    if not isinstance(lineage, list) or not lineage:
        raise ValueError("unified canonical report lacks source lineage")
    if release.get("component_count") != len(lineage):
        raise ValueError(
            "unified canonical release component count is inconsistent"
        )
    partitions = report.get("source_probe_plan_partitions")
    if not isinstance(partitions, list) or len(partitions) != len(lineage):
        raise ValueError(
            "unified canonical source partitions do not match lineage"
        )
    selected_keys: set[str] = set()
    selected_attempt_count = 0
    source_identities: set[tuple[str, str]] = set()
    for index, component in enumerate(lineage):
        if not isinstance(component, dict):
            raise ValueError("unified canonical lineage entry is invalid")
        source_evidence_id = str(
            component.get("source_evidence_id") or ""
        )
        source_report_hash = str(
            component.get("source_report_hash") or ""
        )
        if not source_evidence_id.startswith("canonical_skill_") or not re.fullmatch(
            r"[0-9a-f]{64}", source_report_hash
        ):
            raise ValueError(
                "unified canonical lineage identity is invalid"
            )
        source_identity = (source_evidence_id, source_report_hash)
        if source_identity in source_identities:
            raise ValueError(
                "unified canonical lineage contains duplicate source"
            )
        source_identities.add(source_identity)
        if component.get("selection_order") != index:
            raise ValueError(
                "unified canonical lineage selection order is invalid"
            )
        component_keys = component.get("selected_record_keys")
        if not isinstance(component_keys, list) or not component_keys:
            raise ValueError(
                "unified canonical lineage component selected no records"
            )
        normalized_keys = [str(key) for key in component_keys]
        if normalized_keys != sorted(
            normalized_keys, key=_record_sort_key
        ) or len(normalized_keys) != len(set(normalized_keys)):
            raise ValueError(
                "unified canonical lineage record keys are not canonical"
            )
        overlap = selected_keys.intersection(normalized_keys)
        if overlap:
            raise ValueError(
                "unified canonical lineage record ownership overlaps"
            )
        selected_keys.update(normalized_keys)
        component_attempt_count = sum(
            attempt_count_by_key.get(key, 0) for key in normalized_keys
        )
        selected_attempt_count += component_attempt_count
        expected_uses = sorted(
            {key.rsplit("|", 1)[-1] for key in normalized_keys},
            key=_natural_key,
        )
        if (
            component.get("selected_record_count")
            != len(normalized_keys)
            or component.get("selected_attempt_count")
            != component_attempt_count
            or component.get("selected_record_key_hash")
            != _m4_identity_digest(normalized_keys)
            or component.get("selected_uses") != expected_uses
        ):
            raise ValueError(
                "unified canonical lineage accounting is inconsistent"
            )
        partition = partitions[index]
        if not isinstance(partition, dict) or any(
            partition.get(field) != component.get(field)
            for field in (
                "selection_order",
                "source_kind",
                "source_schema_version",
                "source_report_path",
                "source_evidence_id",
                "source_report_hash",
                "source_probe_plan_path",
                "source_probe_plan_hash",
                "source_runner_hash",
                "selected_record_count",
                "selected_attempt_count",
                "selected_record_key_hash",
                "selected_record_keys",
            )
        ):
            raise ValueError(
                "unified canonical source partition/lineage join mismatch"
            )
    if selected_keys != set(record_keys) or selected_attempt_count != len(
        attempts
    ):
        raise ValueError(
            "unified canonical lineage does not cover the complete report"
        )


def _legacy_report_contact_bindings(
    report: dict[str, Any],
    *,
    source_plan_jobs: Mapping[
        tuple[str, str, str], list[dict[str, Any]]
    ] | None = None,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Recover immutable R02 refs from one self-hashed v2 probe plan.

    Version-2 reports did not repeat the Contact ref in each aggregate row,
    but they did bind the exact self-hashed plan whose unique job carried it.
    This is a deterministic lineage rebind, not a recreation of runtime
    evidence.
    """

    by_key = dict(source_plan_jobs or _source_probe_plan_jobs(report))
    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in report.get("records") or []:
        use = str(row.get("use"))
        if use not in M4_EXACT_GRASP_USES:
            continue
        key = (
            str(row.get("asset")),
            str(row.get("model_id")),
            use,
        )
        jobs = by_key.get(key) or []
        if len(jobs) != 1:
            raise ValueError(
                "legacy source probe plan must contain exactly one matching "
                f"Contact job: {key}"
            )
        binding = jobs[0].get("contact_contract_binding")
        if not isinstance(binding, dict):
            raise ValueError(f"legacy Contact job lacks contract binding: {key}")
        result[key] = dict(binding)
    return result


def _contact_binding_matches(
    current: dict[str, Any], evidence: dict[str, Any]
) -> bool:
    required = (
        "contact_contract_id",
        "contact_contract_hash",
        "candidate_use",
        "use_profile",
        "contact_count",
        "contact_groups",
    )
    if any(evidence.get(field) != current.get(field) for field in required):
        return False
    for field in ("policy_version", "contact_projection_hash"):
        if evidence.get(field) is not None and evidence.get(field) != current.get(
            field
        ):
            return False
    return True


def _touch_row_uses_current_contract(row: Mapping[str, Any]) -> bool:
    return bool(
        str(row.get("use") or "") in TOUCH_EXECUTABLE_USES
        and row.get("touch_capability_version") == TOUCH_CAPABILITY_VERSION
        and row.get("touch_arm_sampling_version")
        == TOUCH_ARM_SAMPLING_VERSION
        and row.get("touch_placement_policy_version")
        == M4_TOUCH_PLACEMENT_POLICY_VERSION
        and row.get("touch_placement_policy_hash")
        == M4_TOUCH_PLACEMENT_POLICY_HASH
        and row.get("touch_requested_arms") == list(TOUCH_REQUESTED_ARMS)
    )


def load_canonical_reports(reports_dir: Path = DEFAULT_REPORTS_DIR) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    row_by_key: dict[str, tuple[tuple[str, str], dict[str, Any]]] = {}
    reports: dict[str, Any] = {}
    if not reports_dir.is_dir():
        return [], reports
    for path in sorted(reports_dir.glob("*.json"), key=lambda p: p.name):
        report = _load_json_required(path)
        try:
            validate_normalized_report(report, expected_path=path)
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            # Stale reports remain immutable audit artifacts but are not
            # allowed to prevent a focused replacement plan or promote any
            # row. This is fail-closed reuse: only independently valid current
            # reports enter row precedence below.
            reports[f"stale:{path.name}"] = {
                "path": _path_string(path),
                "status": "stale",
                "reason": str(exc),
            }
            continue
        legacy_bindings = (
            _legacy_report_contact_bindings(report)
            if report.get("schema_version")
            == LEGACY_CONTACT_REPORT_SCHEMA_VERSION
            else {}
        )
        report_precedence = _report_precedence_key(report, path)
        reports[report["evidence_id"]] = {
            "path": _path_string(path),
            "report_hash": report.get("report_hash"),
            "source_kind": report.get("source_kind"),
            "worker_execution_policy_manifest_hash": report[
                "worker_execution_policy_manifest"
            ]["manifest_hash"],
            "record_count": len(report.get("records") or []),
            "started_at": (report.get("runner") or {}).get("started_at"),
            "finished_at": (report.get("runner") or {}).get("finished_at"),
        }
        for row in report.get("records") or []:
            key = _record_key(row.get("asset"), str(row.get("model_id")), row.get("use"))
            enriched = {
                **row,
                "evidence_id": report["evidence_id"],
                "evidence_path": _path_string(path),
                "evidence_hash": report["report_hash"],
                "worker_execution_policy_manifest_hash": report[
                    "worker_execution_policy_manifest"
                ]["manifest_hash"],
            }
            contact_key = (
                str(row.get("asset")),
                str(row.get("model_id")),
                str(row.get("use")),
            )
            use = contact_key[2]
            if use in TOUCH_EXECUTABLE_USES and not (
                _touch_row_uses_current_contract(row)
            ):
                # The enclosing immutable report can still own valid rows for
                # other uses. Old Click/Press rows are row-scoped stale and
                # are never rebound or promoted.
                continue
            if contact_key in legacy_bindings:
                enriched["contact_contract_binding"] = legacy_bindings[
                    contact_key
                ]
                enriched["contact_contract_hash"] = legacy_bindings[
                    contact_key
                ].get("contact_contract_hash")
            if (
                use in M4_ARTICULATION_USES
                and row.get("articulation_capability_version")
                != LEGACY_ARTICULATION_CAPABILITY_VERSION
            ):
                row_attempts = [
                    attempt
                    for attempt in report.get("attempts") or []
                    if str(attempt.get("asset")) == contact_key[0]
                    and str(attempt.get("model_id")) == contact_key[1]
                    and str(attempt.get("use")) == contact_key[2]
                    and not m4_attempt_is_blocking(attempt)
                ]
                capability = aggregate_articulation_capabilities(
                    row_attempts,
                    use=use,
                    denominator=len(row_attempts),
                    threshold=float(row.get("threshold") or 0.0),
                    require_explicit_runtime_capability=(
                        report.get("schema_version") == REPORT_SCHEMA_VERSION
                    ),
                )
                enriched["articulation_capability_version"] = capability[
                    "capability_version"
                ]
                enriched["articulation_arm_capabilities"] = capability[
                    "arm_capabilities"
                ]
            existing = row_by_key.get(key)
            if existing is None or report_precedence > existing[0]:
                row_by_key[key] = (report_precedence, enriched)
    rows = [item[1] for key, item in sorted(row_by_key.items(), key=lambda item: _record_sort_key(item[0]))]
    return rows, reports


def apply_validation(
    records: dict[str, dict[str, Any]],
    report_rows: list[dict[str, Any]],
    *,
    scope: str = "asset",
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if scope != "asset":
        raise ValueError("Milestone 4 only accepts scope='asset'")
    applied: list[dict[str, Any]] = []
    ignored: list[dict[str, Any]] = []
    report_cache: dict[str, dict[str, Any] | None] = {}
    out = {key: dict(value) for key, value in records.items()}
    for row in report_rows:
        key = _record_key(row.get("asset"), str(row.get("model_id")), row.get("use"))
        current = out.get(key)
        if current is None:
            ignored.append({**_report_row_identity(row), "reason": "report_row_has_no_asset_scope_candidate"})
            continue
        if current.get("disposition") == "unsupported":
            ignored.append(
                {
                    **_report_row_identity(row),
                    "reason": "semantic_unsupported_precedes_runtime_evidence",
                    "semantic_exclusion": current.get("semantic_exclusion"),
                }
            )
            continue
        if current.get("status") == STATUS_REJECTED and current.get("stage") != STAGE_CANONICAL:
            ignored.append({**_report_row_identity(row), "reason": "static_or_metadata_gate_rejected_candidate"})
            continue
        if (
            not row.get("evidence_id")
            or not row.get("evidence_path")
            or not row.get("evidence_hash")
        ):
            ignored.append(
                {
                    **_report_row_identity(row),
                    "reason": "stale_execution_policy_evidence",
                }
            )
            continue
        if not _row_has_live_validated_report_lineage(row, report_cache):
            ignored.append(
                {
                    **_report_row_identity(row),
                    "reason": "stale_canonical_report_evidence",
                }
            )
            continue
        current_contact = current.get("contact_contract_binding")
        if isinstance(current_contact, dict):
            evidence_contact = row.get("contact_contract_binding")
            if not isinstance(evidence_contact, dict) or not _contact_binding_matches(
                current_contact, evidence_contact
            ):
                ignored.append(
                    {
                        **_report_row_identity(row),
                        "reason": "stale_contact_contract_evidence",
                    }
                )
                continue
        if (
            str(row.get("use")) in M4_ARTICULATION_USES
            and row.get("articulation_capability_version")
            != ARTICULATION_CAPABILITY_VERSION
        ):
            ignored.append(
                {
                    **_report_row_identity(row),
                    "reason": "stale_articulation_capability_evidence",
                }
            )
            continue
        if (
            str(row.get("use")) in TOUCH_EXECUTABLE_USES
            and not _touch_row_uses_current_contract(row)
        ):
            ignored.append(
                {
                    **_report_row_identity(row),
                    "reason": "stale_touch_capability_evidence",
                }
            )
            continue
        promoted = _canonical_skill_record(current, row)
        out[key] = promoted
        applied.append({**_report_row_identity(row), "status": promoted["status"], "reason": promoted["reason"]})
    return out, {"applied": applied, "ignored": ignored}


def _row_has_live_validated_report_lineage(
    row: Mapping[str, Any],
    cache: dict[str, dict[str, Any] | None],
) -> bool:
    """Join an enriched row back to the validated report that owns it."""

    raw_path = row.get("evidence_path")
    if not isinstance(raw_path, str) or not raw_path:
        return False
    if raw_path not in cache:
        path = Path(raw_path)
        try:
            report = _load_json_required(path)
            validate_normalized_report(report, expected_path=path)
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            IndexError,
            json.JSONDecodeError,
        ):
            cache[raw_path] = None
        else:
            cache[raw_path] = report
    report = cache[raw_path]
    if not isinstance(report, dict):
        return False
    if (
        row.get("evidence_id") != report.get("evidence_id")
        or row.get("evidence_hash") != report.get("report_hash")
        or row.get("worker_execution_policy_manifest_hash")
        != (report.get("worker_execution_policy_manifest") or {}).get(
            "manifest_hash"
        )
    ):
        return False
    identity = (
        str(row.get("asset")),
        str(row.get("model_id")),
        str(row.get("use")),
    )
    matches = [
        source_row
        for source_row in report.get("records") or []
        if (
            str(source_row.get("asset")),
            str(source_row.get("model_id")),
            str(source_row.get("use")),
        )
        == identity
    ]
    if len(matches) != 1:
        return False
    source_row = matches[0]
    return all(
        canonical_data(row.get(field)) == canonical_data(value)
        for field, value in source_row.items()
    )


def asset_eligibility(
    ledger: dict[str, Any],
    asset: str,
    model_id: str,
    use: str,
    *,
    stage: str | None = None,
    require_validated: bool = True,
) -> dict[str, Any]:
    if str(use) in M4_NON_EVALUATED_USES:
        policy = M4_NON_EVALUATED_USES[str(use)]
        return {
            "eligible": False,
            "status": policy["status"],
            "disposition": "not_evaluated",
            "reason": policy["reason"],
            "record": None,
        }
    record = ((((ledger.get("assets") or {}).get(asset) or {}).get("models") or {}).get(str(model_id)) or {}).get("uses", {}).get(use)
    if record is None:
        return {"eligible": False, "status": None, "reason": "no asset-scope validation record", "record": None}
    if record.get("disposition") == "unsupported":
        return {
            "eligible": False,
            "status": record.get("status"),
            "disposition": "unsupported",
            "reason": record.get("reason"),
            "record": record,
        }
    if stage is not None and record.get("stage") != stage:
        return {"eligible": False, "status": record.get("status"), "reason": f"record stage is {record.get('stage')}", "record": record}
    eligible = record.get("status") == STATUS_VALIDATED if require_validated else record.get("status") in STATUSES
    return {"eligible": bool(eligible), "status": record.get("status"), "reason": record.get("reason"), "record": record}


def pair_eligibility(*args: Any, **kwargs: Any) -> dict[str, Any]:
    if not args and not kwargs:
        return {
            "eligible": False,
            "status": None,
            "reason": "pair validation ledger required",
            "record": None,
        }
    from .pair_validation import pair_eligibility as _pair_eligibility

    return _pair_eligibility(*args, **kwargs)


def build_ledger(
    catalog: dict[str, Any],
    records: dict[str, dict[str, Any]],
    report_index: dict[str, Any],
    report_application: dict[str, Any],
    asset_catalog_path: Path = DEFAULT_ASSET_CATALOG_PATH,
    *,
    semantic_policy: dict[str, Any] | None = None,
    semantic_exclusions: list[dict[str, Any]] | None = None,
    contact_contract_blocked: list[dict[str, Any]] | None = None,
    non_evaluated_capabilities: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    assets: dict[str, Any] = {}
    by_status = Counter()
    by_use = Counter()
    by_stage = Counter()
    for key, record in sorted(records.items(), key=lambda item: _record_sort_key(item[0])):
        asset_name, model_id, use = key.split("|", 2)
        model = assets.setdefault(asset_name, {"models": {}})["models"].setdefault(str(model_id), {"uses": {}})
        model["uses"][use] = record
        by_status[record["status"]] += 1
        by_use[use] += 1
        by_stage[record["stage"]] += 1
    for asset_name, asset in assets.items():
        asset["model_ids"] = sorted(asset["models"], key=_natural_key)
    ledger = {
        "version": SCHEMA_VERSION,
        "schema_version": SCHEMA_VERSION,
        "config_hash": config_hash(),
        "source_asset_catalog_path": _path_string(asset_catalog_path),
        "source_asset_catalog_hash": catalog.get("catalog_hash"),
        "source_inventory_hash": catalog.get("source_inventory_hash"),
        "source_geometry_inventory_hash": catalog.get("source_geometry_inventory_hash"),
        "source_attribute_catalog_hash": catalog.get("source_attribute_catalog_hash"),
        "semantic_exclusion_policy": semantic_exclusion_identity(
            semantic_policy
        ),
        "semantic_exclusions": list(semantic_exclusions or []),
        "contact_contract_blocked": list(contact_contract_blocked or []),
        "non_evaluated_capabilities": list(
            non_evaluated_capabilities or []
        ),
        "reports": report_index,
        "report_application": report_application,
        "summary": {
            "asset_count": len(assets),
            "model_count": sum(len(asset["models"]) for asset in assets.values()),
            "record_count": len(records),
            "record_count_by_status": {status: by_status.get(status, 0) for status in STATUSES},
            "record_count_by_use": {use: by_use.get(use, 0) for use in ASSET_SCOPE_USES},
            "record_count_by_stage": dict(sorted(by_stage.items())),
            "semantic_unsupported_count": len(semantic_exclusions or []),
            "contact_contract_blocked_count": len(
                contact_contract_blocked or []
            ),
            "non_evaluated_capability_count": len(
                non_evaluated_capabilities or []
            ),
            "non_evaluated_uses": sorted(M4_NON_EVALUATED_USES),
        },
        "assets": assets,
        "pairs": {},
    }
    ledger["ledger_hash"] = _artifact_hash(ledger, "ledger_hash")
    return ledger


def build_status_projection(ledger: dict[str, Any], status: str, artifact_name: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    by_use = Counter()
    for asset_name, asset in sorted((ledger.get("assets") or {}).items()):
        for model_id, model in sorted((asset.get("models") or {}).items(), key=lambda item: _natural_key(item[0])):
            for use, record in sorted((model.get("uses") or {}).items(), key=lambda item: _natural_key(item[0])):
                if record.get("status") != status:
                    continue
                rows.append(_projection_row(asset_name, str(model_id), use, record))
                by_use[use] += 1
    projection = {
        "schema_version": SCHEMA_VERSION,
        "source_ledger_hash": ledger.get("ledger_hash"),
        "status": status,
        "summary": {
            "record_count": len(rows),
            "record_count_by_use": {use: by_use.get(use, 0) for use in ASSET_SCOPE_USES},
        },
        "records": rows,
    }
    projection["projection_hash"] = _artifact_hash(projection, "projection_hash")
    return projection


def build_missing_evidence_report(
    ledger: dict[str, Any],
    deferred_pair_candidates: list[dict[str, Any]],
    report_application: dict[str, Any],
    *,
    non_evaluated_capabilities: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    missing: list[dict[str, Any]] = []
    for asset_name, asset in sorted((ledger.get("assets") or {}).items()):
        for model_id, model in sorted((asset.get("models") or {}).items(), key=lambda item: _natural_key(item[0])):
            for use, record in sorted((model.get("uses") or {}).items(), key=lambda item: _natural_key(item[0])):
                if use not in EXECUTABLE_USES:
                    continue
                if record.get("status") != STATUS_CANDIDATE:
                    continue
                missing.append(
                    {
                        "asset": asset_name,
                        "model_id": str(model_id),
                        "use": use,
                        "candidate_id": record.get("source_candidate_id"),
                        "readiness_stage": record.get("stage"),
                        "reason": "missing_canonical_skill_evidence",
                    }
                )
    report = {
        "schema_version": SCHEMA_VERSION,
        "source_ledger_hash": ledger.get("ledger_hash"),
        "summary": {
            "missing_canonical_skill_evidence_count": len(missing),
            "deferred_pair_candidate_count": len(deferred_pair_candidates),
            "ignored_report_row_count": len(report_application.get("ignored") or []),
            "non_evaluated_capability_count": len(
                non_evaluated_capabilities or []
            ),
        },
        "missing_canonical_skill_evidence": missing,
        "deferred_pair_scope_candidates": deferred_pair_candidates,
        "non_evaluated_capabilities": list(
            non_evaluated_capabilities or []
        ),
        "ignored_report_rows": report_application.get("ignored") or [],
    }
    report["report_hash"] = _artifact_hash(report, "report_hash")
    return report


def normalize_report_file(path: Path) -> dict[str, Any]:
    raw = _load_json_required(path)
    if _is_normalized_report(raw):
        validate_normalized_report(raw)
        return raw
    raise ValueError(
        "Milestone 4 apply-report requires canonical_skill_asset_report schema v2; "
        "legacy/runtime normalization fallback is forbidden"
    )


def records_from_report(raw: dict[str, Any], path: Path | None = None) -> list[dict[str, Any]]:
    payload = raw.get("payload") if isinstance(raw.get("payload"), dict) else raw
    kind = str(raw.get("kind") or payload.get("kind") or "")
    if _looks_like_generated_scene_report(payload):
        raise ValueError("generated-scene/layout/perception reports cannot promote asset validation")
    if kind == "canonical_skill":
        source_records = payload.get("records") or raw.get("records") or []
        rows = [_normalize_explicit_row(item, payload) for item in source_records]
        if not rows:
            raise ValueError("canonical_skill report contains no asset-scope records")
        return rows
    raise ValueError(f"unsupported canonical asset report kind: {kind or '<missing>'}; legacy smoke reports are refused")
    return rows


def validate_normalized_report(report: dict[str, Any], expected_path: Path | None = None) -> None:
    if not _is_normalized_report(report):
        raise ValueError("expected normalized canonical asset report")
    report_schema_version = report.get("schema_version")
    if report_schema_version not in {
        REPORT_SCHEMA_VERSION,
        LEGACY_CONTACT_REPORT_SCHEMA_VERSION,
    }:
        raise ValueError("canonical report schema is stale")
    source_kind = str(report.get("source_kind") or "")
    if source_kind not in CANONICAL_REPORT_SOURCE_KINDS:
        raise ValueError("canonical report source kind is invalid")
    if (
        source_kind == UNIFIED_FROZEN_CANONICAL_SOURCE_KIND
        and report_schema_version != REPORT_SCHEMA_VERSION
    ):
        raise ValueError(
            "unified canonical report must use the current schema version"
        )
    source_plan_jobs = _source_probe_plan_jobs(report)
    source_plan_contact_bindings = _legacy_report_contact_bindings(
        report, source_plan_jobs=source_plan_jobs
    )
    legacy_contact_bindings = (
        source_plan_contact_bindings
        if report_schema_version == LEGACY_CONTACT_REPORT_SCHEMA_VERSION
        else {}
    )
    if report_schema_version == REPORT_SCHEMA_VERSION and (
        report.get("pick_operation_placement_policy_version")
        != PICK_OPERATION_PLACEMENT_POLICY_VERSION
        or report.get("pick_operation_placement_policy_hash")
        != PICK_OPERATION_PLACEMENT_POLICY_HASH
    ):
        raise ValueError("canonical report Pick placement policy is stale")
    expected_hash = _artifact_hash(report, "report_hash")
    if report.get("report_hash") != expected_hash:
        raise ValueError("canonical report hash mismatch")
    expected_evidence = _evidence_id(report)
    if report.get("evidence_id") != expected_evidence:
        raise ValueError("canonical report evidence id mismatch")
    contract = SuccessContract.load()
    if report.get("success_contract_version") != contract.version:
        raise ValueError("canonical report success contract version is stale")
    if report.get("success_contract_hash") != contract.hash:
        raise ValueError("canonical report success contract hash is stale")
    manifest_errors = m4_worker_execution_policy_manifest_errors(
        report.get("worker_execution_policy_manifest")
    )
    if manifest_errors:
        raise ValueError(
            "canonical report worker execution policy manifest is stale: "
            + ", ".join(manifest_errors)
        )
    attempts = report.get("attempts")
    if not isinstance(attempts, list):
        raise ValueError("canonical report attempts missing")
    attempts_by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for attempt in attempts:
        if not isinstance(attempt, dict):
            raise ValueError("canonical report attempt must be a mapping")
        key = (
            str(attempt.get("asset")),
            str(attempt.get("model_id")),
            str(attempt.get("use")),
        )
        attempts_by_key.setdefault(key, []).append(attempt)
    records = report.get("records")
    if not isinstance(records, list) or not all(
        isinstance(row, dict) for row in records
    ):
        raise ValueError("canonical report records must be a list of mappings")
    report_accounting_errors = m4_report_attempt_accounting_errors(
        records, attempts, require_seed_coverage=True
    )
    if report_accounting_errors:
        raise ValueError(
            "canonical report attempt accounting mismatch: "
            + ", ".join(report_accounting_errors)
        )
    if source_kind == UNIFIED_FROZEN_CANONICAL_SOURCE_KIND:
        _validate_unified_release_metadata(
            report,
            records=records,
            attempts=attempts,
            source_plan_jobs=source_plan_jobs,
        )
    articulation_rows = [
        row
        for row in records
        if row.get("use") in {"open", "close", "switch_on", "switch_off"}
    ]
    if articulation_rows:
        if any(
            row.get("articulation_capability_version")
            == ARTICULATION_CAPABILITY_VERSION
            for row in articulation_rows
        ) and report.get("articulation_arm_sampling_version") != (
            ARTICULATION_ARM_SAMPLING_VERSION
        ):
            raise ValueError("canonical articulation sampling contract is stale")
        catalog = load_articulation_execution_catalog()
        if (
            report.get("articulation_execution_catalog_hash")
            != catalog["profile_hash"]
        ):
            raise ValueError("canonical articulation execution profile is stale")
        for row in articulation_rows:
            profile = resolve_articulation_execution_profile(
                str(row.get("asset")),
                str(row.get("model_id")),
                str(row.get("use")),
                contract=contract,
                catalog=catalog,
            )
            if row.get("articulation_execution_profile_versions") != [
                profile["profile_version"]
            ]:
                raise ValueError("canonical articulation profile version is stale")
            if row.get("articulation_execution_profile_hashes") != [
                profile["profile_hash"]
            ]:
                raise ValueError("canonical articulation profile evidence is stale")
            if row.get("articulation_execution_profile_source_hashes") != [
                profile["source_asset_catalog_sha256"]
            ]:
                raise ValueError("canonical articulation profile source is stale")
            if row.get("articulation_motion_policy_versions") != [
                profile["motion_policy_version"]
            ]:
                raise ValueError("canonical articulation motion policy is stale")
            if row.get("articulation_motion_policy_hashes") != [
                profile["motion_policy_hash"]
            ]:
                raise ValueError(
                    "canonical articulation motion policy hash is stale"
                )
    touch_rows = [
        row for row in records if row.get("use") in TOUCH_EXECUTABLE_USES
    ]
    current_touch_rows = [
        row for row in touch_rows if _touch_row_uses_current_contract(row)
    ]
    if current_touch_rows and (
        report.get("touch_arm_sampling_version")
        != TOUCH_ARM_SAMPLING_VERSION
        or report.get("touch_placement_policy_version")
        != M4_TOUCH_PLACEMENT_POLICY_VERSION
        or report.get("touch_placement_policy_hash")
        != M4_TOUCH_PLACEMENT_POLICY_HASH
    ):
        raise ValueError("canonical touch contract is stale")
    if expected_path is not None and expected_path.stem != expected_evidence:
        raise ValueError("canonical report filename does not match evidence id")
    for row in records:
        if row.get("scope") != "asset" or row.get("stage") != STAGE_CANONICAL:
            raise ValueError("canonical report rows must be asset-scope canonical_skill rows")
        if row.get("use") not in HISTORICAL_CANONICAL_USES:
            raise ValueError(f"unsupported canonical asset use: {row.get('use')}")
        threshold = row.get("threshold")
        min_rate = row.get("min_rate")
        if threshold is None or min_rate is None:
            raise ValueError("canonical report rows require threshold and min_rate")
        if row.get("asset") is None or row.get("model_id") is None:
            raise ValueError("canonical report rows require asset and model_id")
        key = (str(row["asset"]), str(row["model_id"]), str(row["use"]))
        current_touch_row = _touch_row_uses_current_contract(row)
        row_attempts = attempts_by_key.get(key) or []
        accounting_errors = m4_row_attempt_accounting_errors(
            row, row_attempts, require_seed_coverage=True
        )
        if accounting_errors:
            raise ValueError(
                "canonical record attempt accounting mismatch: "
                + ", ".join(accounting_errors)
            )
        denominator_attempts = [
            attempt
            for attempt in row_attempts
            if not m4_attempt_is_blocking(attempt)
        ]
        matching_plan_jobs = source_plan_jobs.get(key) or []
        if len(matching_plan_jobs) != 1:
            raise ValueError(
                "canonical placement requires one unique source plan job: "
                f"{key}"
            )
        canonical_state_hash = (
            (
                matching_plan_jobs[0].get("canonical_initial_state")
                or {}
            ).get("canonical_state_hash")
        )
        if current_touch_row and (
            matching_plan_jobs[0].get("touch_arm_sampling_version")
            != TOUCH_ARM_SAMPLING_VERSION
            or matching_plan_jobs[0].get("touch_placement_policy_version")
            != M4_TOUCH_PLACEMENT_POLICY_VERSION
            or matching_plan_jobs[0].get("touch_placement_policy_hash")
            != M4_TOUCH_PLACEMENT_POLICY_HASH
            or matching_plan_jobs[0].get("requested_arms")
            != list(TOUCH_REQUESTED_ARMS)
        ):
            raise ValueError("canonical touch source plan binding is stale")
        if report_schema_version == REPORT_SCHEMA_VERSION and not canonical_state_hash:
            raise ValueError(
                "canonical placement source plan lacks canonical state hash"
            )
        placement_failures = sorted(
            {
                failure
                for attempt in denominator_attempts
                for failure in m4_canonical_placement_witness_errors(
                    attempt.get("canonical_placement_witness"),
                    expected_use=str(row["use"]),
                    expected_canonical_state_hash=(
                        str(canonical_state_hash)
                        if canonical_state_hash is not None
                        else None
                    ),
                    require_current_touch_policy=(
                        row.get("use") not in TOUCH_EXECUTABLE_USES
                        or current_touch_row
                    ),
                )
            }
        )
        placement_hashes = sorted(
            {
                str(
                    (attempt.get("canonical_placement_witness") or {}).get(
                        "witness_hash"
                    )
                )
                for attempt in denominator_attempts
                if (attempt.get("canonical_placement_witness") or {}).get(
                    "witness_hash"
                )
            }
        )
        if denominator_attempts and len(placement_hashes) != 1:
            placement_failures = sorted(
                {
                    *placement_failures,
                    "canonical_placement_witness_inconsistent",
                }
            )
        if placement_failures:
            raise ValueError(
                "canonical placement evidence is stale: "
                + ", ".join(placement_failures)
            )
        if report_schema_version == REPORT_SCHEMA_VERSION and (
            row.get("canonical_placement_evidence_failures") != []
            or row.get("canonical_placement_policy_version")
            != PICK_OPERATION_PLACEMENT_POLICY_VERSION
            or row.get("canonical_placement_policy_hash")
            != PICK_OPERATION_PLACEMENT_POLICY_HASH
            or row.get("canonical_placement_witness_hashes")
            != placement_hashes
        ):
            raise ValueError(
                "canonical placement evidence accounting mismatch"
            )
        attempt_policy_failures = sorted(
            {
                failure
                for attempt in denominator_attempts
                for failure in m4_worker_execution_policy_manifest_errors(
                    attempt.get("execution_policy_manifest"),
                    use=str(row.get("use") or ""),
                )
            }
        )
        if attempt_policy_failures:
            raise ValueError(
                "canonical attempt worker execution policy is stale: "
                + ", ".join(attempt_policy_failures)
            )
        if denominator_attempts:
            expected_manifest_hashes = sorted(
                {
                    str(
                        (attempt.get("execution_policy_manifest") or {}).get(
                            "manifest_hash"
                        )
                    )
                    for attempt in row_attempts
                    if (attempt.get("execution_policy_manifest") or {}).get(
                        "manifest_hash"
                    )
                }
            )
            if row.get("execution_policy_evidence_failures") != []:
                raise ValueError(
                    "canonical record worker execution policy failures changed"
                )
            if row.get("worker_execution_policy_manifest_hashes") != (
                expected_manifest_hashes
            ):
                raise ValueError(
                    "canonical record worker execution policy hash is stale"
                )
        if row.get("use") in M4_ARTICULATION_USES:
            expected_action = {
                "execution_profile_version": (
                    row.get("articulation_execution_profile_versions")
                    or [None]
                )[0],
                "execution_profile_hash": (
                    row.get("articulation_execution_profile_hashes")
                    or [None]
                )[0],
                "execution_profile_source_hash": (
                    row.get(
                        "articulation_execution_profile_source_hashes"
                    )
                    or [None]
                )[0],
                "motion_policy_version": (
                    row.get("articulation_motion_policy_versions")
                    or [None]
                )[0],
                "motion_policy_hash": (
                    row.get("articulation_motion_policy_hashes")
                    or [None]
                )[0],
            }
            expected_articulation_failures: set[str] = set()
            for attempt in denominator_attempts:
                observed = (attempt.get("diagnostics") or {}).get(
                    "articulation_action"
                )
                if not isinstance(observed, dict):
                    expected_articulation_failures.add(
                        "articulation_execution_preflight_missing"
                    )
                    continue
                for field, expected in expected_action.items():
                    if observed.get(field) != expected:
                        expected_articulation_failures.add(
                            f"articulation_execution_{field}_mismatch"
                        )
            legacy_articulation_capability = (
                row.get("articulation_capability_version")
                == LEGACY_ARTICULATION_CAPABILITY_VERSION
            )
            capability = None
            if not legacy_articulation_capability:
                capability = aggregate_articulation_capabilities(
                    denominator_attempts,
                    use=str(row["use"]),
                    denominator=len(denominator_attempts),
                    threshold=float(row["threshold"]),
                    require_explicit_runtime_capability=True,
                )
                expected_articulation_failures.update(capability["failures"])
            expected_articulation_failure_list = sorted(
                expected_articulation_failures
            )
            if (
                report_schema_version == REPORT_SCHEMA_VERSION
                and row.get("articulation_evidence_failures")
                != expected_articulation_failure_list
            ):
                raise ValueError(
                    "canonical articulation evidence accounting mismatch"
                )
            if (
                expected_articulation_failure_list
                and row.get("outcome") != "blocked"
            ):
                raise ValueError(
                    "canonical articulation runtime profile evidence is stale"
                )
            if (
                report_schema_version == REPORT_SCHEMA_VERSION
                and not legacy_articulation_capability
                and capability is not None
                and (
                row.get("articulation_capability_version")
                != capability["capability_version"]
                or canonical_data(
                    row.get("articulation_arm_capabilities") or []
                )
                != capability["arm_capabilities"]
                )
            ):
                raise ValueError(
                    "canonical articulation capability accounting mismatch"
                )
        if current_touch_row:
            capability = aggregate_touch_capabilities(
                denominator_attempts,
                use=str(row["use"]),
                threshold=float(row["threshold"]),
            )
            capable_arms = {
                str(item["arm"])
                for item in capability["arm_capabilities"]
            }
            expected_arm_results: dict[str, dict[str, Any]] = {}
            for arm in TOUCH_REQUESTED_ARMS:
                arm_attempts = [
                    attempt
                    for attempt in row_attempts
                    if str(attempt.get("requested_arm") or "") == arm
                ]
                blocked_attempts = [
                    attempt
                    for attempt in arm_attempts
                    if m4_attempt_is_blocking(attempt)
                ]
                arm_denominator = [
                    attempt
                    for attempt in arm_attempts
                    if not m4_attempt_is_blocking(attempt)
                ]
                arm_success = sum(
                    1 for attempt in arm_denominator if attempt.get("success")
                )
                arm_rate = (
                    arm_success / len(arm_denominator)
                    if arm_denominator
                    else 0.0
                )
                arm_passed = arm in capable_arms and not blocked_attempts
                expected_arm_results[arm] = {
                    "attempts": len(arm_attempts),
                    "denominator": len(arm_denominator),
                    "excluded_blocking_attempts": len(blocked_attempts),
                    "plan_success": sum(
                        1
                        for attempt in arm_denominator
                        if attempt.get("plan_success")
                    ),
                    "success": arm_success,
                    "rate": _clean_float(arm_rate),
                    "min_rate": _clean_float(arm_rate),
                    "threshold": _clean_float(float(row["threshold"])),
                    "outcome": (
                        "blocked"
                        if blocked_attempts
                        else "pass"
                        if arm_passed
                        else "fail"
                    ),
                    "pass": bool(arm_passed),
                }
            passing_arms = {
                arm
                for arm, result in expected_arm_results.items()
                if result["pass"]
            }
            expected_capabilities = [
                item
                for item in capability["arm_capabilities"]
                if str(item["arm"]) in passing_arms
            ]
            expected_binding = {
                "capability_version": TOUCH_CAPABILITY_VERSION,
                "sampling_version": TOUCH_ARM_SAMPLING_VERSION,
                "placement_policy_version": (
                    M4_TOUCH_PLACEMENT_POLICY_VERSION
                ),
                "placement_policy_hash": M4_TOUCH_PLACEMENT_POLICY_HASH,
                "requested_arms": list(TOUCH_REQUESTED_ARMS),
                "canonical_placement_witness_hash": capability[
                    "canonical_placement_witness_hash"
                ],
                "arm_results": expected_arm_results,
                "failures": capability["failures"],
                "arm_capabilities": expected_capabilities,
            }
            if (
                row.get("touch_capability_version")
                != TOUCH_CAPABILITY_VERSION
                or row.get("touch_arm_sampling_version")
                != TOUCH_ARM_SAMPLING_VERSION
                or row.get("touch_placement_policy_version")
                != M4_TOUCH_PLACEMENT_POLICY_VERSION
                or row.get("touch_placement_policy_hash")
                != M4_TOUCH_PLACEMENT_POLICY_HASH
                or row.get("touch_requested_arms")
                != list(TOUCH_REQUESTED_ARMS)
                or canonical_data(row.get("touch_arm_results") or {})
                != canonical_data(expected_arm_results)
                or canonical_data(row.get("touch_arm_capabilities") or [])
                != canonical_data(expected_capabilities)
                or row.get("touch_evidence_failures")
                != capability["failures"]
                or row.get("touch_canonical_placement_witness_hash")
                != capability["canonical_placement_witness_hash"]
            ):
                raise ValueError(
                    "canonical touch capability accounting mismatch"
                )
            if capability["failures"] and row.get("outcome") != "blocked":
                raise ValueError("canonical touch runtime evidence is stale")
            if row.get("pass") is True and touch_capability_errors(
                expected_binding, use=str(row["use"])
            ):
                raise ValueError("canonical touch execution binding is invalid")
        if row.get("use") in M4_EXACT_GRASP_USES:
            binding = (
                legacy_contact_bindings.get(key)
                if report_schema_version
                == LEGACY_CONTACT_REPORT_SCHEMA_VERSION
                else row.get("contact_contract_binding")
            )
            if not isinstance(binding, dict):
                raise ValueError("canonical Contact record lacks contract binding")
            if (
                report_schema_version == REPORT_SCHEMA_VERSION
                and source_plan_contact_bindings.get(key) != binding
            ):
                raise ValueError(
                    "canonical Contact record/source plan binding mismatch"
                )
            if row.get("contact_contract_hash") not in (
                None,
                binding.get("contact_contract_hash"),
            ):
                raise ValueError("canonical Contact record hash mismatch")
            for attempt in denominator_attempts:
                observed = (attempt.get("diagnostics") or {}).get(
                    "contact_contract_preflight"
                )
                if not isinstance(observed, dict):
                    raise ValueError(
                        "canonical Contact attempt lacks runtime preflight"
                    )
                if observed.get("contact_contract_hash") != binding.get(
                    "contact_contract_hash"
                ):
                    raise ValueError(
                        "canonical Contact attempt contract hash mismatch"
                    )
                if resolved_contact_preflight_policy_version(
                    observed, binding
                ) != binding.get("policy_version"):
                    raise ValueError(
                        "canonical Contact attempt policy version mismatch"
                    )
                if (
                    report_schema_version == REPORT_SCHEMA_VERSION
                    and observed.get("contact_projection_hash")
                    != binding.get("contact_projection_hash")
                ):
                    raise ValueError(
                        "canonical Contact attempt projection hash mismatch"
                    )
            if report_schema_version == REPORT_SCHEMA_VERSION:
                expected_contact_failures = sorted(
                    {
                        f"contact_contract_{field}_mismatch"
                        for attempt in denominator_attempts
                        for field in (
                            "policy_version",
                            "contact_contract_id",
                            "contact_contract_hash",
                            "contact_projection_hash",
                            "candidate_use",
                            "use_profile",
                            "contact_count",
                            "contact_groups",
                        )
                        if (
                            (
                                resolved_contact_preflight_policy_version(
                                    (
                                        (attempt.get("diagnostics") or {}).get(
                                            "contact_contract_preflight"
                                        )
                                        or {}
                                    ),
                                    binding,
                                )
                                if field == "policy_version"
                                else (
                                    (attempt.get("diagnostics") or {}).get(
                                        "contact_contract_preflight"
                                    )
                                    or {}
                                ).get(field)
                            )
                            != binding.get(field)
                        )
                    }
                )
                if row.get("contact_evidence_failures") != expected_contact_failures:
                    raise ValueError(
                        "canonical Contact record evidence accounting mismatch"
                    )
                if expected_contact_failures and (
                    row.get("outcome") != "blocked"
                    or row.get("pass") is not False
                ):
                    raise ValueError(
                        "canonical Contact evidence failures must block record"
                    )
        expected_predicate_blocked = sum(
            1
            for attempt in denominator_attempts
            if isinstance(attempt.get("predicate_result"), dict)
            and attempt["predicate_result"].get("status") == "blocked"
        )
        if row.get("predicate_blocked_attempts") != expected_predicate_blocked:
            raise ValueError("canonical record blocked predicate accounting mismatch")
        expected_failures = sorted(
            {
                failure
                for attempt in denominator_attempts
                for failure in _success_evidence_failures(attempt, contract)
            }
        )
        if row.get("success_evidence_failures") != expected_failures:
            raise ValueError("canonical record success evidence audit mismatch")
        expected_versions = sorted(
            {
                str(attempt.get("success_contract_version"))
                for attempt in denominator_attempts
                if attempt.get("success_contract_version") is not None
            }
        )
        expected_hashes = sorted(
            {
                str(attempt.get("success_contract_hash"))
                for attempt in denominator_attempts
                if attempt.get("success_contract_hash") is not None
            }
        )
        expected_snapshots = sorted(
            {
                str(attempt.get("success_snapshot_hash"))
                for attempt in denominator_attempts
                if attempt.get("success_snapshot_hash") is not None
            }
        )
        if row.get("success_contract_versions") != expected_versions:
            raise ValueError("canonical record contract version accounting mismatch")
        if row.get("success_contract_hashes") != expected_hashes:
            raise ValueError("canonical record contract hash accounting mismatch")
        if row.get("success_snapshot_hashes") != expected_snapshots:
            raise ValueError("canonical record snapshot accounting mismatch")
        if expected_failures:
            if row.get("outcome") != "blocked" or row.get("pass"):
                raise ValueError("stale success evidence must block canonical record")
        elif expected_predicate_blocked:
            if row.get("outcome") != "blocked" or row.get("pass"):
                raise ValueError("blocked predicate must block canonical record")
        elif denominator_attempts:
            if expected_versions != [contract.version]:
                raise ValueError("canonical record success contract version missing")
            if expected_hashes != [contract.hash]:
                raise ValueError("canonical record success contract hash missing")
            if not expected_snapshots:
                raise ValueError("canonical record success snapshots missing")


def _static_record(asset_name: str, model_id: str, use: str, candidate: dict[str, Any]) -> dict[str, Any]:
    stage = STAGE_ARTICULATED if use in {USE_OPEN, USE_CLOSE, USE_SWITCH_ON, USE_SWITCH_OFF} else STAGE_STATIC
    gate = _gate_candidate(use, candidate)
    review = candidate.get("review") or {}
    unresolved = [str(item) for item in review.get("unresolved") or []]
    base = {
        "scope": "asset",
        "stage": stage,
        "status": STATUS_CANDIDATE,
        "reason": "ready_for_canonical_skill_probe" if use in EXECUTABLE_USES else "static_anchor_validated",
        "asset": asset_name,
        "model_id": model_id,
        "use": use,
        "source_candidate_id": candidate.get("candidate_id"),
        "source_candidate_use": candidate.get("use"),
        "inventory_id": candidate.get("inventory_id"),
        "category": candidate.get("category"),
        "evidence_id": _static_evidence_id(asset_name, model_id, use, gate),
        "evidence_path": None,
        "config_hash": config_hash(),
        "preconditions": [gate],
    }
    if unresolved:
        base["review"] = {
            "requires_review": True,
            "unresolved": unresolved,
            "fact_status": review.get("fact_status"),
        }
    if not gate["passed"]:
        base["status"] = STATUS_REJECTED
        base["reason"] = gate["reason"]
        return base
    if unresolved:
        base["status"] = STATUS_REJECTED
        base["reason"] = "review_required: " + ", ".join(unresolved)
        return base
    if use not in EXECUTABLE_USES and use != USE_RELATION_ANCHOR:
        base["status"] = STATUS_REJECTED
        base["reason"] = f"canonical_skill_not_implemented: {use}"
        return base
    if use == USE_RELATION_ANCHOR:
        base["status"] = STATUS_VALIDATED
    return base


def _gate_candidate(use: str, candidate: dict[str, Any]) -> dict[str, Any]:
    metadata = candidate.get("metadata") or {}
    if use in {USE_PICK_LEFT, USE_PICK_RIGHT}:
        checks = {
            "contact_points": int(metadata.get("contact_points_count") or 0) > 0,
            "usable_geometry": _has_usable_geometry(metadata),
        }
        return _gate_result(STAGE_STATIC, checks)
    if use == USE_HANDOVER:
        checks = {
            "contact_groups": int(metadata.get("contact_group_count") or 0) >= 2,
            "usable_geometry": _has_usable_geometry(metadata),
        }
        return _gate_result(STAGE_STATIC, checks)
    if use == USE_RELATION_ANCHOR:
        contact_sheet = ((candidate.get("evidence") or {}).get("contact_sheet") or {})
        checks = {
            "contact_sheet": bool(contact_sheet.get("present")),
            "category": bool(candidate.get("category")),
            "stable": metadata.get("stable") is True,
        }
        return _gate_result(STAGE_STATIC, checks)
    if use in {USE_OPEN, USE_CLOSE, USE_SWITCH_ON, USE_SWITCH_OFF}:
        drive = candidate.get("articulation_drive") or {}
        checks = {
            "articulation_metadata": bool(metadata.get("has_articulation")) and int(metadata.get("joint_count") or 0) > 0,
            "joint_index": drive.get("joint_index") is not None,
            "open_contact_ids": bool(drive.get("open_contact_ids")),
            "drive_mode": bool(drive.get("drive_mode")),
        }
        return _gate_result(STAGE_ARTICULATED, checks)
    if use in {USE_CLICK, USE_PRESS}:
        checks = {
            "contact_points": int(metadata.get("contact_points_count") or 0) > 0,
        }
        return _gate_result(STAGE_STATIC, checks)
    if use in {USE_ROTATE_ORIENTATION, USE_ADJUST_ORIENTATION}:
        checks = {
            "contact_points": int(metadata.get("contact_points_count") or 0) > 0,
            "usable_geometry": _has_usable_geometry(metadata),
        }
        return _gate_result(STAGE_STATIC, checks)
    if use == USE_BIMANUAL_GRAB:
        checks = {
            "contact_groups": int(metadata.get("contact_group_count") or 0) >= 2,
            "usable_geometry": _has_usable_geometry(metadata),
        }
        return _gate_result(STAGE_STATIC, checks)
    return {"stage": STAGE_STATIC, "passed": False, "checks": {}, "reason": "unknown_asset_scope_use"}


def _gate_result(stage: str, checks: dict[str, bool]) -> dict[str, Any]:
    failed = [name for name, ok in checks.items() if not ok]
    return {
        "stage": stage,
        "passed": not failed,
        "checks": checks,
        "reason": None if not failed else f"{stage}_fail: missing " + ", ".join(failed),
    }


def _canonical_skill_record(current: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    threshold = float(row["threshold"])
    min_rate = float(row["min_rate"])
    excluded_count = int(
        row.get("excluded_source_or_internal_attempts") or 0
    )
    excluded_blocking_count = int(
        row.get("excluded_blocking_attempts") or 0
    )
    contact_evidence_blocked = bool(row.get("contact_evidence_failures"))
    requested_arm_sampling = bool(
        row.get("articulation_arm_sampling_version")
        == ARTICULATION_ARM_SAMPLING_VERSION
        or row.get("touch_arm_sampling_version")
        == TOUCH_ARM_SAMPLING_VERSION
    )
    blocked = (
        row.get("outcome") == "blocked"
        or (excluded_blocking_count > 0 and not requested_arm_sampling)
        or contact_evidence_blocked
    )
    passed = bool(not blocked and row.get("pass")) and min_rate >= threshold
    record = dict(current)
    record.update(
        {
            "stage": STAGE_CANONICAL,
            "status": (
                STATUS_CANDIDATE
                if blocked
                else STATUS_VALIDATED
                if passed
                else STATUS_REJECTED
            ),
            "reason": (
                f"canonical_probe_blocked: {row.get('verdict') or 'source_or_internal_failure'}"
                if blocked
                else "ok"
                if passed
                else _canonical_fail_reason(row)
            ),
            "evidence_id": row.get("evidence_id"),
            "evidence_path": row.get("evidence_path"),
            "evidence_hash": row.get("evidence_hash"),
            "worker_execution_policy_manifest_hash": row.get(
                "worker_execution_policy_manifest_hash"
            ),
            "canonical_placement_policy_version": row.get(
                "canonical_placement_policy_version"
            ),
            "canonical_placement_policy_hash": row.get(
                "canonical_placement_policy_hash"
            ),
            "canonical_placement_witness_hash": (
                (row.get("canonical_placement_witness_hashes") or [None])[0]
            ),
            "config_hash": row.get("config_hash"),
            "min_rate": _clean_float(min_rate),
            "threshold": _clean_float(threshold),
            "seeds": row.get("seeds") or [],
            "attempts": int(row.get("attempts") or 0),
            "denominator": int(
                row.get("denominator")
                if row.get("denominator") is not None
                else row.get("attempts")
                or 0
            ),
            "excluded_source_or_internal_attempts": excluded_count,
            "excluded_blocking_attempts": excluded_blocking_count,
            "plan_success": int(row.get("plan_success") or 0),
            "success": int(row.get("success") or 0),
            "per_variant": row.get("per_variant") or {},
            "fail_reasons": row.get("fail_reasons") or {},
        }
    )
    if str(row.get("use")) in M4_ARTICULATION_USES:
        profile_versions = row.get(
            "articulation_execution_profile_versions"
        ) or []
        profile_hashes = row.get(
            "articulation_execution_profile_hashes"
        ) or []
        source_hashes = row.get(
            "articulation_execution_profile_source_hashes"
        ) or []
        motion_policy_versions = row.get(
            "articulation_motion_policy_versions"
        ) or []
        motion_policy_hashes = row.get(
            "articulation_motion_policy_hashes"
        ) or []
        if not all(
            len(values) == 1
            for values in (
                profile_versions,
                profile_hashes,
                source_hashes,
                motion_policy_versions,
                motion_policy_hashes,
            )
        ):
            raise ValueError("canonical articulation profile binding is ambiguous")
        record["articulation_execution_binding"] = {
            "profile_version": profile_versions[0],
            "profile_hash": profile_hashes[0],
            "profile_source_hash": source_hashes[0],
            "motion_policy_version": motion_policy_versions[0],
            "motion_policy_hash": motion_policy_hashes[0],
            "capability_version": str(
                row.get("articulation_capability_version") or ""
            ),
            "sampling_version": str(
                row.get("articulation_arm_sampling_version") or ""
            ),
            "requested_arms": canonical_data(
                row.get("articulation_requested_arms") or []
            ),
            "arm_results": canonical_data(
                row.get("articulation_arm_results") or {}
            ),
            "arm_capabilities": canonical_data(
                row.get("articulation_arm_capabilities") or []
            ),
        }
        if (
            record["articulation_execution_binding"]["capability_version"]
            != ARTICULATION_CAPABILITY_VERSION
            or (
                record.get("status") == STATUS_VALIDATED
                and not record["articulation_execution_binding"][
                    "arm_capabilities"
                ]
            )
        ):
            raise ValueError(
                "canonical articulation arm capability binding is missing"
            )
    if str(row.get("use")) in TOUCH_EXECUTABLE_USES:
        touch_binding = canonical_data(
            {
                "capability_version": str(
                    row.get("touch_capability_version") or ""
                ),
                "sampling_version": str(
                    row.get("touch_arm_sampling_version") or ""
                ),
                "placement_policy_version": str(
                    row.get("touch_placement_policy_version") or ""
                ),
                "placement_policy_hash": str(
                    row.get("touch_placement_policy_hash") or ""
                ),
                "requested_arms": row.get("touch_requested_arms") or [],
                "canonical_placement_witness_hash": row.get(
                    "touch_canonical_placement_witness_hash"
                ),
                "arm_results": row.get("touch_arm_results") or {},
                "failures": row.get("touch_evidence_failures") or [],
                "arm_capabilities": row.get("touch_arm_capabilities") or [],
            }
        )
        record["touch_execution_binding"] = touch_binding
        if record.get("status") == STATUS_VALIDATED:
            binding_errors = touch_capability_errors(
                touch_binding, use=str(row["use"])
            )
            if binding_errors:
                raise ValueError(
                    "canonical touch execution binding is invalid: "
                    + ", ".join(binding_errors)
                )
    return record


def _normalize_explicit_row(item: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    use = str(item.get("use"))
    if use not in HISTORICAL_CANONICAL_USES:
        raise ValueError(f"unsupported canonical asset use: {use}")
    threshold = item.get(
        "threshold", payload.get("threshold", DEFAULT_PROBE_THRESHOLDS.get(use, 0.6))
    )
    seeds = item.get("seeds", payload.get("seeds") or [])
    row = {
        "scope": "asset",
        "stage": STAGE_CANONICAL,
        "asset": str(item.get("asset")),
        "model_id": str(item.get("model_id")),
        "use": use,
        "pass": bool(item.get("pass")),
        "verdict": item.get("verdict") or item.get("reason"),
        "min_rate": _clean_float(float(item.get("min_rate", 0.0))),
        "threshold": _clean_float(float(threshold)),
        "seeds": [int(seed) for seed in seeds],
        "attempts": int(item.get("attempts") or 0),
        "denominator": int(
            item.get("denominator")
            if item.get("denominator") is not None
            else item.get("attempts")
            or 0
        ),
        "excluded_source_or_internal_attempts": int(
            item.get("excluded_source_or_internal_attempts") or 0
        ),
        "excluded_blocking_attempts": int(
            item.get("excluded_blocking_attempts") or 0
        ),
        "outcome": str(item.get("outcome") or ""),
        "plan_success": int(item.get("plan_success") or 0),
        "success": int(item.get("success") or 0),
        "per_variant": item.get("per_variant") or {},
        "fail_reasons": item.get("fail_reasons") or {},
    }
    row["config_hash"] = _row_config_hash(row)
    return row


def _projection_row(asset_name: str, model_id: str, use: str, record: dict[str, Any]) -> dict[str, Any]:
    row = {
        "asset": asset_name,
        "model_id": model_id,
        "use": use,
        "stage": record.get("stage"),
        "status": record.get("status"),
        "reason": record.get("reason"),
        "evidence_id": record.get("evidence_id"),
        "evidence_path": record.get("evidence_path"),
        "evidence_hash": record.get("evidence_hash"),
        "worker_execution_policy_manifest_hash": record.get(
            "worker_execution_policy_manifest_hash"
        ),
        "config_hash": record.get("config_hash"),
        "source_candidate_id": record.get("source_candidate_id"),
        "source_candidate_use": record.get("source_candidate_use"),
    }
    if record.get("disposition"):
        row["disposition"] = record.get("disposition")
    if record.get("semantic_exclusion"):
        row["semantic_exclusion"] = record.get("semantic_exclusion")
    if record.get("contact_contract_gate"):
        row["contact_contract_gate"] = record.get("contact_contract_gate")
    if record.get("contact_contract_binding"):
        row["contact_contract_binding"] = record.get(
            "contact_contract_binding"
        )
        row["contact_contract_hash"] = record.get(
            "contact_contract_hash"
        )
    if record.get("articulation_execution_binding"):
        row["articulation_execution_binding"] = record.get(
            "articulation_execution_binding"
        )
    if record.get("touch_execution_binding"):
        row["touch_execution_binding"] = record.get(
            "touch_execution_binding"
        )
    if record.get("review"):
        row["review"] = record.get("review")
    for key in ("min_rate", "threshold", "seeds", "attempts", "plan_success", "success"):
        if key in record:
            row[key] = record[key]
    return row


def _has_usable_geometry(metadata: dict[str, Any]) -> bool:
    extents = metadata.get("scaled_extents")
    footprint = metadata.get("xy_footprint") or {}
    return (
        isinstance(extents, list)
        and len(extents) == 3
        and all(_is_positive_number(value) for value in extents)
        and _is_positive_number(metadata.get("height"))
        and _is_positive_number(footprint.get("area"))
    )


def _looks_like_generated_scene_report(payload: dict[str, Any]) -> bool:
    forbidden_keys = {"layout_fail", "scene_rollout_fail", "descriptor_visibility", "scene_spec", "family_id"}
    if any(key in payload for key in forbidden_keys):
        return True
    kind = str(payload.get("kind") or "")
    return kind in {"layout_fail", "scene_rollout_fail", "perception_gate", "descriptor_gate"}


def _is_normalized_report(raw: dict[str, Any]) -> bool:
    return raw.get("kind") == "canonical_skill_asset_report" and raw.get("scope") == "asset"


def _canonical_fail_reason(row: dict[str, Any]) -> str:
    verdict = row.get("verdict") or "threshold not met"
    return f"canonical_skill_fail: {verdict}"


def _report_row_identity(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "asset": row.get("asset"),
        "model_id": str(row.get("model_id")),
        "use": row.get("use"),
        "evidence_id": row.get("evidence_id"),
    }


def _deferred_pair_candidate(candidate: dict[str, Any], *, reason: str) -> dict[str, Any]:
    row = {
        "candidate_id": candidate.get("candidate_id"),
        "candidate_use": candidate.get("use"),
        "scope": candidate.get("scope", "pair"),
        "reason": reason,
    }
    if "subject" in candidate:
        row["subject"] = candidate.get("subject")
    if "receiver" in candidate:
        row["receiver"] = candidate.get("receiver")
    if "review" in candidate:
        row["review"] = candidate.get("review")
    if "warnings" in candidate:
        row["warnings"] = candidate.get("warnings")
    return row


def _should_plan_probe(
    record: dict[str, Any],
    *,
    retry_failed: bool,
    include_validated: bool = False,
) -> bool:
    if record.get("status") == STATUS_CANDIDATE:
        return True
    if include_validated and record.get("status") == STATUS_VALIDATED and record.get("stage") == STAGE_CANONICAL:
        return True
    return bool(retry_failed and record.get("status") == STATUS_REJECTED and record.get("stage") == STAGE_CANONICAL)


def _report_precedence_key(report: dict[str, Any], path: Path) -> tuple[str, str]:
    runner = report.get("runner") or {}
    timestamp = str(runner.get("finished_at") or runner.get("started_at") or "")
    return (timestamp, path.name)


def _record_key(asset: Any, model_id: Any, use: Any) -> str:
    return f"{asset}|{model_id}|{use}"


def _record_sort_key(key: str) -> tuple[Any, ...]:
    asset, model_id, use = key.split("|", 2)
    return (_natural_key(asset), _natural_key(model_id), _natural_key(use))


def _probe_job_id(asset_name: str, model_id: str, use: str) -> str:
    return f"probe|{asset_name}|{model_id}|{use}"


def _minimal_scene_kind(use: str) -> str:
    if use in {USE_PICK_LEFT, USE_PICK_RIGHT}:
        return "single_subject_clean_table"
    if use == USE_HANDOVER:
        return "single_subject_two_arm_handover"
    if use in {USE_OPEN, USE_CLOSE, USE_SWITCH_ON, USE_SWITCH_OFF}:
        return "single_articulated_subject_clean_table"
    if use in {USE_CLICK, USE_PRESS}:
        return "single_subject_touch_clean_table"
    if use in {USE_ROTATE_ORIENTATION, USE_ADJUST_ORIENTATION}:
        return "single_subject_orientation_clean_table"
    if use == USE_BIMANUAL_GRAB:
        return "single_subject_two_arm_grab"
    return "asset_scope_minimal_scene"


def _static_evidence_id(asset_name: str, model_id: str, use: str, gate: dict[str, Any]) -> str:
    digest = _sha256_bytes(_canonical_json({"asset": asset_name, "model_id": model_id, "use": use, "gate": gate, "config_hash": config_hash()}).encode("utf-8"))[:20]
    return f"{gate['stage']}_{digest}"


def _row_config_hash(row: dict[str, Any]) -> str:
    keys = (
        "scope",
        "stage",
        "use",
        "threshold",
        "seeds",
        "articulation_arm_sampling_version",
        "articulation_requested_arms",
        "touch_arm_sampling_version",
        "touch_requested_arms",
        "touch_placement_policy_version",
        "touch_placement_policy_hash",
    )
    return _sha256_bytes(_canonical_json({key: row.get(key) for key in keys}).encode("utf-8"))


def _evidence_id(report: dict[str, Any]) -> str:
    payload = {key: value for key, value in report.items() if key not in {"evidence_id", "report_hash"}}
    digest = _sha256_bytes(_canonical_json(payload).encode("utf-8"))[:20]
    return f"canonical_skill_{digest}"


def _is_positive_number(value: Any) -> bool:
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False


def _clean_float(value: float) -> float:
    return round(float(value), 12)


def _artifact_hash(data: dict[str, Any], hash_key: str) -> str:
    payload = {key: value for key, value in data.items() if key != hash_key}
    return _sha256_bytes(_canonical_json(payload).encode("utf-8"))


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _natural_key(value: str) -> tuple[Any, ...]:
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", str(value)))


def _load_yaml_required(path: Path) -> dict[str, Any]:
    data = _load_yaml(path)
    if data is None:
        raise FileNotFoundError(path)
    return data


def _load_yaml(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return None
    if not isinstance(data, dict):
        raise ValueError(f"expected YAML mapping in {path}")
    return data


def _load_json_required(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True, default_flow_style=False, width=100)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, sort_keys=True, ensure_ascii=False, indent=2)
        f.write("\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_semantic_exclusion_policy(
    asset_catalog_path: Path,
    semantic_exclusions_path: Path | None,
) -> dict[str, Any] | None:
    policy_path = semantic_exclusions_path
    if policy_path is None:
        try:
            uses_default_catalog = (
                asset_catalog_path.resolve() == DEFAULT_ASSET_CATALOG_PATH.resolve()
            )
        except OSError:
            uses_default_catalog = asset_catalog_path == DEFAULT_ASSET_CATALOG_PATH
        if not uses_default_catalog:
            return None
        policy_path = DEFAULT_SEMANTIC_EXCLUSIONS_PATH
    return load_asset_semantic_exclusions(
        policy_path,
        asset_catalog_path=asset_catalog_path,
    )


def _path_string(path: Path | None) -> str | None:
    if path is None:
        return None
    return str(path).replace("\\", "/")


def _status_line(command: str, artifacts: dict[str, Any]) -> str:
    summary = artifacts["ledger"].get("summary", {})
    return (
        f"{command}: wrote {summary.get('record_count', 0)} asset validation records; "
        f"validated {summary.get('record_count_by_status', {}).get(STATUS_VALIDATED, 0)}, "
        f"rejected {summary.get('record_count_by_status', {}).get(STATUS_REJECTED, 0)}, "
        f"candidate {summary.get('record_count_by_status', {}).get(STATUS_CANDIDATE, 0)}"
    )


def _check_status_line(report: dict[str, Any]) -> str:
    if report.get("stale"):
        missing = report.get("missing_artifacts", [])
        suffix = f"; missing {missing}" if missing else ""
        return f"check: asset validation artifacts are stale{suffix}"
    return "check: asset validation artifacts are current"


def _parallel_worker_count(parallel_workers: int | None, workers_per_gpu: int, cuda_devices: list[str] | None) -> int:
    if parallel_workers is not None:
        return max(1, int(parallel_workers))
    visible_device_count = len(cuda_devices) if cuda_devices else 1
    return max(1, visible_device_count * max(1, int(workers_per_gpu)))


def _parse_asset_models(values: list[str] | None) -> set[tuple[str, str]] | None:
    if not values:
        return None
    parsed = set()
    for value in values:
        if ":" not in value:
            raise ValueError(f"asset-model selector must be asset:model, got {value!r}")
        asset, model_id = value.rsplit(":", 1)
        parsed.add((asset, model_id))
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze or check BenchmarkV1 asset validation artifacts.")
    parser.add_argument("--asset-catalog", type=Path, default=DEFAULT_ASSET_CATALOG_PATH)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("freeze")
    subparsers.add_parser("check")
    plan_parser = subparsers.add_parser("plan-probes")
    plan_parser.add_argument("--retry-failed", action="store_true", help="Include canonical_skill rejected executable rows in the probe plan.")
    plan_parser.add_argument("--include-validated", action="store_true")
    plan_parser.add_argument("--asset-model", action="append", help="Exact asset:model selector; repeatable.")
    plan_parser.add_argument("--use", action="append", choices=EXECUTABLE_USES)
    plan_parser.add_argument("--initial-pose-set", type=Path, default=DEFAULT_INITIAL_POSE_SET_PATH)
    plan_parser.add_argument("--initial-state-selection", type=Path)
    run_parser = subparsers.add_parser("run-probes")
    run_parser.add_argument("--plan", type=Path)
    run_parser.add_argument("--initial-pose-set", type=Path, default=DEFAULT_INITIAL_POSE_SET_PATH)
    run_parser.add_argument("--initial-state-selection", type=Path)
    run_parser.add_argument("--python", dest="python_executable")
    run_parser.add_argument("--task-config", default="demo_clean")
    run_parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_PROBE_SEEDS))
    run_parser.add_argument("--use", action="append", choices=EXECUTABLE_USES)
    run_parser.add_argument("--threshold", action="append", help="Per-use threshold override, e.g. pick_left=0.8")
    run_parser.add_argument("--start-index", type=int, default=0)
    run_parser.add_argument("--max-jobs", type=int)
    run_parser.add_argument("--timeout-seconds", type=int, default=240)
    run_parser.add_argument("--warmup-timeout-seconds", type=int, default=DEFAULT_WARMUP_TIMEOUT_SECONDS)
    run_parser.add_argument("--torch-extensions-dir", type=Path)
    run_parser.add_argument("--clear-existing-reports", action="store_true")
    run_parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    run_parser.add_argument("--reuse-task-env", action="store_true")
    run_parser.add_argument("--cuda-devices", help="Comma-separated device ids assigned round-robin, e.g. 0,1")
    run_parser.add_argument("--parallel-workers", type=int, help="Total concurrent batch workers. Defaults to len(cuda-devices) * workers-per-gpu.")
    run_parser.add_argument("--workers-per-gpu", type=int, default=1)
    run_parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume only from a checkpoint bound to the exact current M4 "
            "plan, inputs, execution policy, seeds, thresholds, and scheduling config."
        ),
    )
    run_parser.add_argument(
        "--job-cost-history",
        type=Path,
        help=(
            "Optional prior attempts.jsonl used only for deterministic GPU "
            "load balancing; its file hash is bound into resume metadata."
        ),
    )
    equivalence_parser = subparsers.add_parser("probe-equivalence")
    equivalence_parser.add_argument("--plan", type=Path)
    equivalence_parser.add_argument("--python", dest="python_executable")
    equivalence_parser.add_argument("--task-config", default="demo_clean")
    equivalence_parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_PROBE_SEEDS))
    equivalence_parser.add_argument("--use", action="append", choices=EXECUTABLE_USES)
    equivalence_parser.add_argument("--threshold", action="append", help="Per-use threshold override, e.g. pick_left=0.8")
    equivalence_parser.add_argument("--start-index", type=int, default=0)
    equivalence_parser.add_argument("--max-jobs", type=int, default=1)
    equivalence_parser.add_argument("--timeout-seconds", type=int, default=240)
    equivalence_parser.add_argument("--warmup-timeout-seconds", type=int, default=DEFAULT_WARMUP_TIMEOUT_SECONDS)
    equivalence_parser.add_argument("--torch-extensions-dir", type=Path)
    equivalence_parser.add_argument("--batch-size", type=int, default=2)
    dashboard_parser = subparsers.add_parser("dashboard")
    dashboard_parser.add_argument("--report", type=Path)
    dashboard_parser.add_argument("--output", type=Path)
    dashboard_parser.add_argument("--records", type=Path)
    dashboard_parser.add_argument("--attempts", type=Path)
    dashboard_parser.add_argument("--timing-summary", type=Path)
    apply_parser = subparsers.add_parser("apply-report")
    apply_parser.add_argument("report", type=Path)
    apply_parser.add_argument("--copy-source", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "freeze":
        artifacts = freeze_asset_validation(args.asset_catalog, args.ledger, args.output_dir, args.reports_dir)
        print(_status_line("freeze", artifacts))
        return 0
    if args.command == "check":
        report = check_asset_validation(args.asset_catalog, args.ledger, args.output_dir, args.reports_dir)
        print(_check_status_line(report))
        return 1 if report.get("stale") else 0
    if args.command == "plan-probes":
        plan = plan_probes(
            args.asset_catalog,
            args.output_dir,
            args.reports_dir,
            retry_failed=args.retry_failed,
            include_validated=args.include_validated,
            asset_models=_parse_asset_models(args.asset_model),
            uses=set(args.use) if args.use else None,
            initial_pose_set_path=args.initial_pose_set,
            initial_state_selection_path=args.initial_state_selection,
        )
        print(f"plan-probes: wrote {plan.get('summary', {}).get('probe_job_count', 0)} probe jobs")
        return 0
    if args.command == "run-probes":
        if args.resume and args.clear_existing_reports:
            parser.error("run-probes --resume cannot be combined with --clear-existing-reports")
        if args.clear_existing_reports:
            removed = clear_canonical_reports(args.reports_dir)
            print(f"run-probes: removed {removed} existing canonical report files")
        plan_path = args.plan
        if plan_path is None:
            plan_probes(
                args.asset_catalog,
                args.output_dir,
                args.reports_dir,
                initial_pose_set_path=args.initial_pose_set,
                initial_state_selection_path=args.initial_state_selection,
            )
            plan_path = args.output_dir / PROBE_PLAN_FILE
        cuda_devices = parse_cuda_devices(args.cuda_devices)
        parallel_workers = _parallel_worker_count(args.parallel_workers, args.workers_per_gpu, cuda_devices)
        result = run_probe_plan(
            asset_catalog_path=args.asset_catalog,
            plan_path=plan_path,
            reports_dir=args.reports_dir,
            run_dir=args.output_dir / "runs",
            python_executable=args.python_executable,
            task_config=args.task_config,
            seeds=parse_seed_list(args.seeds),
            thresholds=parse_thresholds(args.threshold),
            uses=set(args.use) if args.use else None,
            start_index=args.start_index,
            max_jobs=args.max_jobs,
            timeout_seconds=args.timeout_seconds,
            warmup_timeout_seconds=args.warmup_timeout_seconds,
            torch_extensions_dir=args.torch_extensions_dir,
            clear_existing_reports=False,
            batch_size=args.batch_size,
            reuse_task_env=args.reuse_task_env,
            parallel_workers=parallel_workers,
            cuda_devices=cuda_devices,
            resume=args.resume,
            job_cost_history_path=args.job_cost_history,
        )
        print(
            f"run-probes: wrote {result['record_count']} records / {result['attempt_count']} attempts "
            f"to {result['evidence_path']}; timing {result['timing_summary_path']}"
        )
        return 0
    if args.command == "probe-equivalence":
        plan_path = args.plan
        if plan_path is None:
            plan_probes(args.asset_catalog, args.output_dir, args.reports_dir)
            plan_path = args.output_dir / PROBE_PLAN_FILE
        result = run_probe_equivalence_check(
            asset_catalog_path=args.asset_catalog,
            plan_path=plan_path,
            output_dir=args.output_dir,
            python_executable=args.python_executable,
            task_config=args.task_config,
            seeds=parse_seed_list(args.seeds),
            thresholds=parse_thresholds(args.threshold),
            uses=set(args.use) if args.use else None,
            start_index=args.start_index,
            max_jobs=args.max_jobs,
            timeout_seconds=args.timeout_seconds,
            warmup_timeout_seconds=args.warmup_timeout_seconds,
            torch_extensions_dir=args.torch_extensions_dir,
            batch_size=args.batch_size,
        )
        status = "passed" if result["pass"] else "failed"
        print(
            f"probe-equivalence: {status}; compared {result['attempt_count']} attempts; "
            f"mismatches {result['mismatch_count']}; report {result['report_path']}"
        )
        return 0 if result["pass"] else 1
    if args.command == "dashboard":
        from .validation_dashboard import write_validation_dashboard

        output_path = args.output or (args.output_dir / "validation_dashboard.html")
        result = write_validation_dashboard(
            report_path=args.report,
            reports_dir=args.reports_dir,
            data_dir=args.output_dir,
            output_path=output_path,
            records_path=args.records,
            attempts_path=args.attempts,
            timing_summary_path=args.timing_summary,
        )
        stale = " stale projections" if result["projection_stale"] else ""
        print(
            f"dashboard: wrote {result['output_path']} from {result['report_path']} "
            f"({result['record_count']} records / {result['attempt_count']} attempts){stale}"
        )
        return 0
    if args.command == "apply-report":
        result = apply_report_file(args.report, args.reports_dir, copy_source=args.copy_source)
        print(f"apply-report: wrote {result['record_count']} records to {result['evidence_path']}")
        return 0
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
