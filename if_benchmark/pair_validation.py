"""BenchmarkV1 pair-scope validation ledger freezer.

Milestone 5 consumes the Milestone 3 asset catalog and the Milestone 4
asset-scope ledger.  Rule-based geometry is used only for triage: it can reject
known impossible pairs or identify clearly plausible geometry, but terminal and
canonical validation facts come only from normalized pair reports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .asset_probe_pose import canonical_pose_policy_hash, resolve_canonical_pose
from .initial_pose.core import (
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
from .schema import artifact_hash
from .schema.serialization import canonical_data
from .pair_skill_contract import (
    HANG_ENTRY_POLICY_ID,
    HANG_RELATION_POLICY_VERSION,
)
from .pair_yaw_geometry import (
    DEFAULT_ANGLE_STEP_DEG,
    DEFAULT_BAND_EROSION_DEG,
    DEFAULT_BOUNDARY_TOLERANCE_DEG,
    DEFAULT_MIN_CONTINUOUS_BAND_DEG,
    pose_aware_box_geometry,
    solve_relative_yaw_domain,
    yaw_solver_policy_hash,
)
from .pair_recipe import (
    TERMINAL_PLACEMENT_RECIPE_VERSION,
    TERMINAL_PROBE_POLICY_VERSION,
    canonical_terminal_placement_recipe,
)

SCHEMA_VERSION = 3
REPORT_SCHEMA_VERSION = 5

DEFAULT_ASSET_CATALOG_PATH = Path("if_benchmark/assets/asset_catalog.yml")
DEFAULT_ASSET_LEDGER_PATH = Path("if_benchmark/assets/asset_validation.yml")
DEFAULT_LEDGER_PATH = Path("if_benchmark/assets/pair_validation.yml")
DEFAULT_OUTPUT_DIR = Path("if_benchmark/data/pair_validation")
DEFAULT_REPORTS_DIR = DEFAULT_OUTPUT_DIR / "reports"
DEFAULT_RUBRIC_PATH = DEFAULT_OUTPUT_DIR / "pair_rubric_review.yml"
DEFAULT_RECEIVER_PROFILE_PATH = Path("if_benchmark/assets/articulated_receiver_profiles.yml")
DEFAULT_INITIAL_POSE_SET_PATH = Path(
    "if_benchmark/data/initial_pose/validated_initial_states.yml"
)

SKILL_PROBE_POLICY_VERSION = (
    "m5_canonical_pair_skill_v22_proportional_asset_mass_inertia"
)
SKILL_RELATION_POLICY_VERSIONS = {
    relation: (
        HANG_RELATION_POLICY_VERSION
        if relation == "hang_on"
        else SKILL_PROBE_POLICY_VERSION
    )
    for relation in (
        "place_in",
        "place_on",
        "stack",
        "hang_on",
        "dump_into",
        "stamp_on",
    )
}

GEOMETRY_MATRIX_FILE = "pair_geometry_matrix.yml"
VALIDATED_FILE = "validated_pairs.yml"
VALIDATED_TERMINAL_FILE = "validated_terminal_pairs.yml"
VALIDATED_SKILL_FILE = "validated_skill_pairs.yml"
GAP_REPORT_FILE = "pair_gap_report.yml"
DECISION_DIFF_FILE = "runs/pair_decision_diff.yml"
TERMINAL_PLAN_FILE = "terminal_settle_plan.yml"
SKILL_PLAN_FILE = "canonical_skill_pair_plan.yml"

STATUS_REJECTED = "rejected"
STATUS_PRUNED = "pruned"
STATUS_GEOMETRY_PASS = "geometry_pass"
STATUS_AMBIGUOUS = "ambiguous_needs_probe"
STATUS_PROBE_BLOCKED = "probe_spec_blocked"
STATUS_PROBE_RUN_BLOCKED = "probe_run_blocked"
STATUS_VALIDATED_TERMINAL = "validated_terminal"
STATUS_VALIDATED_SKILL = "validated_skill"
STATUSES = (
    STATUS_REJECTED,
    STATUS_PRUNED,
    STATUS_GEOMETRY_PASS,
    STATUS_AMBIGUOUS,
    STATUS_PROBE_BLOCKED,
    STATUS_PROBE_RUN_BLOCKED,
    STATUS_VALIDATED_TERMINAL,
    STATUS_VALIDATED_SKILL,
)

STAGE_GEOMETRY = "geometry_matrix"
STAGE_PROBE_SPEC = "probe_spec"
STAGE_TERMINAL = "terminal_settle"
STAGE_CANONICAL = "canonical_skill"

CORE_RELATIONS = ("place_in", "place_on", "stack")
EXTRA_RELATIONS = ("hang_on", "dump_into", "stamp_on")
# ``RELATIONS`` is the historical/serialization vocabulary.  Keep disabled
# values here so old reports remain readable and their original policy hashes
# remain verifiable.  Enablement is a publication concern, not a rewrite of
# already-collected physical evidence.
RELATIONS = CORE_RELATIONS + EXTRA_RELATIONS

RELATION_ENABLEMENT_POLICY_VERSION = "m5_relation_enablement_v2"
DISABLED_RELATIONS: dict[str, dict[str, str]] = {
    "hang_on": {
        "status": "disabled",
        "reason": "explicitly_disabled_no_m6_m15_consumer",
        "disposition": "historical_evidence_only",
    },
    "dump_into": {
        "status": "disabled",
        "reason": "explicitly_disabled_no_m6_m15_consumer",
        "disposition": "historical_evidence_only",
    },
    "stamp_on": {
        "status": "disabled",
        "reason": "explicitly_disabled_no_m6_m15_consumer",
        "disposition": "historical_evidence_only",
    },
}
ACTIVE_RELATIONS = tuple(
    relation for relation in RELATIONS if relation not in DISABLED_RELATIONS
)

RECEIVER_CANDIDATE_TO_RELATION = {
    "place_in_receiver": "place_in",
    "place_on_receiver": "place_on",
    "stack_base": "stack",
}
PAIR_CANDIDATE_TO_RELATION = {
    "hang_on_pair": "hang_on",
    "dump_into_pair": "dump_into",
    "stamp_on_pair": "stamp_on",
}

TERMINAL_SOURCE_KIND = "m5_terminal_settle_pair_probe"
CANONICAL_SOURCE_KIND = "m5_canonical_skill_pair_probe"

PAIR_GEOMETRY_CONFIG: dict[str, Any] = {
    "edge_clearance_m": 0.015,
    "geometry_pass_margin_m": 0.025,
    "place_in_cavity_fraction": 0.70,
    "place_on_support_fraction": 0.75,
    "stack_support_fraction": 0.65,
    "geometry_pass_max_area_ratio": 0.70,
    "relative_yaw": {
        "angle_step_deg": DEFAULT_ANGLE_STEP_DEG,
        "boundary_tolerance_deg": DEFAULT_BOUNDARY_TOLERANCE_DEG,
        "band_erosion_deg": DEFAULT_BAND_EROSION_DEG,
        "min_continuous_band_deg": DEFAULT_MIN_CONTINUOUS_BAND_DEG,
    },
    "extra_pair_semantics": {
        "hang_on": "039_mug functional point 0 to 040_rack functional point 0 midpoint",
        "dump_into": "063_tabletrashbin proxy contents into basket receiver target",
        "stamp_on": "100_seal contact ids [4, 5, 6, 7] to receiver top target",
    },
}

CONFIG: dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "report_schema_version": REPORT_SCHEMA_VERSION,
    "relations": list(RELATIONS),
    "core_relations": list(CORE_RELATIONS),
    "extra_relations": list(EXTRA_RELATIONS),
    "receiver_candidate_to_relation": RECEIVER_CANDIDATE_TO_RELATION,
    "pair_candidate_to_relation": PAIR_CANDIDATE_TO_RELATION,
    "states": list(STATUSES),
    "geometry": PAIR_GEOMETRY_CONFIG,
    "canonical_pose_policy_hash": canonical_pose_policy_hash(),
    "yaw_solver_policy_hash": yaw_solver_policy_hash(),
    "promotion": {
        "geometry_pass": "queryable geometry evidence only, never terminal/canonical validation",
        "terminal": "terminal_settle_pair_report promotes terminal readiness",
        "canonical": "canonical_skill_pair_report promotes actor rollout readiness",
    },
}


def config_hash() -> str:
    return _sha256_bytes(_canonical_json(CONFIG).encode("utf-8"))


def relation_enablement_policy_hash() -> str:
    """Hash the active/publication relation set without staling probe evidence."""

    payload = {
        "version": RELATION_ENABLEMENT_POLICY_VERSION,
        "recognized_relations": list(RELATIONS),
        "active_relations": list(ACTIVE_RELATIONS),
        "disabled_relations": DISABLED_RELATIONS,
    }
    return _sha256_bytes(_canonical_json(payload).encode("utf-8"))


def require_active_relation_selection(
    *,
    relations: set[str] | None = None,
    pair_keys: set[str] | None = None,
) -> None:
    """Reject an explicit request to execute a disabled relation."""

    requested = {str(relation) for relation in (relations or set())}
    requested.update(
        str(pair_key).split("|", 1)[0]
        for pair_key in (pair_keys or set())
    )
    disabled = sorted(requested.intersection(DISABLED_RELATIONS))
    if disabled:
        raise ValueError(f"relation_disabled:{','.join(disabled)}")


def terminal_probe_policy_hash() -> str:
    payload = {
        "terminal": TERMINAL_PROBE_POLICY_VERSION,
        "placement_recipe": TERMINAL_PLACEMENT_RECIPE_VERSION,
        "canonical_pose": canonical_pose_policy_hash(),
        "relative_yaw": yaw_solver_policy_hash(),
    }
    return _sha256_bytes(_canonical_json(payload).encode("utf-8"))


def skill_probe_policy_hash() -> str:
    """Hash only the canonical-skill contract.

    This deliberately does not participate in ``config_hash``.  Updating the
    skill implementation must not stale already-frozen rigid Terminal
    evidence.
    """

    payload = {
        "version": SKILL_PROBE_POLICY_VERSION,
        "placement_recipe": TERMINAL_PLACEMENT_RECIPE_VERSION,
        "relations": list(RELATIONS),
        "scene": {
            "receiver_xy": {"left": [-0.05, -0.10], "right": [0.05, -0.10]},
            "place_in_receiver_xy_by_pick_arm": {
                "left": [0.05, -0.10],
                "right": [-0.05, -0.10],
            },
            "subject_xy": {
                "default": {"left": [-0.20, -0.10], "right": [0.20, -0.10]},
                "rigid_place_in_lanes": {
                    "classifier": "terminal_aabb_projected_through_exact_m4_validated_pick_quaternion",
                    "threshold_pick_footprint_axis_m": 0.18,
                    "clearance_m": 0.025,
                    "side_lane_min_abs_x": 0.20,
                    "front_lane_abs_x": 0.16,
                    "side_table_limit_abs_x": 0.46,
                    "front_table_limit_min_y": -0.36,
                    "selection": "larger_aabb_table_margin",
                },
                "articulated_place_in": {"left": [-0.20, -0.10], "right": [0.20, -0.10]},
            },
            "hang_receiver_xy": [0.18, 0.15],
            "receiver_random_yaw": False,
            "seed_randomness": "subject_small_xy_translation_only",
        },
        "witness_selection": "highest_terminal_margin_then_band_id",
        "pick_precondition": (
            "exact_m4_validated_q_and_root_z_with_fixture_xy_as_actual_aabb_center"
            "_then_terminal_subject_mass_after_lift"
        ),
        "setup_stability": {
            "m4_validated_subject": "restore_recorded_pose_and_delegate_to_m4_then_run_skill",
            "receiver_or_other_actor": "probe_run_blocked",
        },
        "articulated_place_in": "reviewed_open_state_pick_insert_release",
        "articulated_terminal_witness": {
            "canonical_pose": "copied_from_m5_ar_terminal_probe_job",
            "placement_recipe": "exact_selected_m5_ar_band_recipe",
        },
        "relation_adapters": {
            "place_on": "reviewed_functional_point_skill",
            "stack": "terminal_root_top_down_0.05m_reverse_path_retract",
            "hang_on": "right_arm_functional_axis_insert",
            "dump_into": "reviewed_contact_1_arm_lift_receiver_relative_pour_5x5mm_proxy",
            "stamp_on": "contact_ids_4_5_6_7_reverse_approach_retract",
        },
        "success": "plan_success_and_release_and_60_plus_25_iteration_settle_and_canonical_checker",
    }
    return _sha256_bytes(_canonical_json(payload).encode("utf-8"))


def skill_relation_policy_hash(relation: str) -> str:
    if relation not in RELATIONS:
        raise ValueError(f"unsupported pair relation: {relation}")
    payload = {
        "base_skill_probe_policy_hash": skill_probe_policy_hash(),
        "relation": relation,
        "version": SKILL_RELATION_POLICY_VERSIONS[relation],
    }
    return _sha256_bytes(_canonical_json(payload).encode("utf-8"))


def skill_relation_policy_hashes() -> dict[str, str]:
    return {
        relation: skill_relation_policy_hash(relation)
        for relation in RELATIONS
    }


def _requires_explicit_skill_relation_hash(relation: str) -> bool:
    return (
        SKILL_RELATION_POLICY_VERSIONS.get(relation)
        != SKILL_PROBE_POLICY_VERSION
    )


def freeze_pair_validation(
    asset_catalog_path: Path = DEFAULT_ASSET_CATALOG_PATH,
    asset_ledger_path: Path = DEFAULT_ASSET_LEDGER_PATH,
    ledger_path: Path = DEFAULT_LEDGER_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    rubric_path: Path = DEFAULT_RUBRIC_PATH,
) -> dict[str, Any]:
    artifacts = freeze_pair_validation_to_memory(asset_catalog_path, asset_ledger_path, reports_dir, rubric_path)
    previous_ledger = _load_yaml(ledger_path) if ledger_path.is_file() else None
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_yaml(ledger_path, artifacts["ledger"])
    _write_yaml(output_dir / GEOMETRY_MATRIX_FILE, artifacts["geometry_matrix"])
    _write_yaml(output_dir / VALIDATED_FILE, artifacts["validated_pairs"])
    _write_yaml(output_dir / VALIDATED_TERMINAL_FILE, artifacts["validated_terminal_pairs"])
    _write_yaml(output_dir / VALIDATED_SKILL_FILE, artifacts["validated_skill_pairs"])
    _write_yaml(output_dir / GAP_REPORT_FILE, artifacts["pair_gap_report"])
    if previous_ledger:
        decision_diff = build_pair_decision_diff(previous_ledger, artifacts["ledger"])
        if decision_diff["summary"]["changed_pair_count"] > 0:
            _write_yaml(output_dir / DECISION_DIFF_FILE, decision_diff)
        artifacts["pair_decision_diff"] = decision_diff
    return artifacts


def build_pair_decision_diff(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    previous_pairs = previous.get("pairs") or {}
    current_pairs = current.get("pairs") or {}
    transitions: Counter[str] = Counter()
    records: list[dict[str, Any]] = []
    for key in sorted(set(previous_pairs) | set(current_pairs), key=_record_sort_key):
        before = previous_pairs.get(key) or {}
        after = current_pairs.get(key) or {}
        before_status = before.get("status")
        after_status = after.get("status")
        before_reason = before.get("reason")
        after_reason = after.get("reason")
        if before_status == after_status and before_reason == after_reason:
            continue
        transition = f"{before_status or '<missing>'}->{after_status or '<missing>'}"
        transitions[transition] += 1
        records.append(
            {
                "pair_key": key,
                "relation": after.get("relation") or before.get("relation"),
                "subject": after.get("subject") or before.get("subject"),
                "receiver": after.get("receiver") or before.get("receiver"),
                "before_status": before_status,
                "after_status": after_status,
                "before_reason": before_reason,
                "after_reason": after_reason,
            }
        )
    diff = {
        "schema_version": 1,
        "kind": "pair_decision_diff",
        "source_previous_ledger_hash": previous.get("ledger_hash"),
        "source_current_ledger_hash": current.get("ledger_hash"),
        "summary": {
            "changed_pair_count": len(records),
            "transition_counts": dict(sorted(transitions.items())),
        },
        "records": records,
    }
    diff["report_hash"] = _artifact_hash(diff, "report_hash")
    return diff


def check_pair_validation(
    asset_catalog_path: Path = DEFAULT_ASSET_CATALOG_PATH,
    asset_ledger_path: Path = DEFAULT_ASSET_LEDGER_PATH,
    ledger_path: Path = DEFAULT_LEDGER_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    rubric_path: Path = DEFAULT_RUBRIC_PATH,
) -> dict[str, Any]:
    generated = freeze_pair_validation_to_memory(asset_catalog_path, asset_ledger_path, reports_dir, rubric_path)
    artifact_specs = (
        ("pair_validation.yml", ledger_path, "ledger_hash"),
        (GEOMETRY_MATRIX_FILE, output_dir / GEOMETRY_MATRIX_FILE, "matrix_hash"),
        (VALIDATED_FILE, output_dir / VALIDATED_FILE, "projection_hash"),
        (VALIDATED_TERMINAL_FILE, output_dir / VALIDATED_TERMINAL_FILE, "projection_hash"),
        (VALIDATED_SKILL_FILE, output_dir / VALIDATED_SKILL_FILE, "projection_hash"),
        (GAP_REPORT_FILE, output_dir / GAP_REPORT_FILE, "report_hash"),
    )
    missing = [
        name
        for name, path, _ in artifact_specs
        if not path.is_file()
    ]
    current_hashes = {
        name: _load_top_level_scalar(path, hash_key)
        for name, path, hash_key in artifact_specs
    }
    invalid_self_hashes: list[str] = []
    for name, path, hash_key in artifact_specs:
        if not path.is_file():
            continue
        current_document = _load_yaml_required(path)
        if current_document.get(hash_key) != _artifact_hash(
            current_document, hash_key
        ):
            invalid_self_hashes.append(name)
    current_ledger_hash = current_hashes["pair_validation.yml"]
    current_matrix_hash = current_hashes[GEOMETRY_MATRIX_FILE]
    current_validated_hash = current_hashes[VALIDATED_FILE]
    current_terminal_hash = current_hashes[VALIDATED_TERMINAL_FILE]
    current_skill_hash = current_hashes[VALIDATED_SKILL_FILE]
    current_gap_hash = current_hashes[GAP_REPORT_FILE]
    generated_ledger_hash = generated["ledger"].get("ledger_hash")
    generated_matrix_hash = generated["geometry_matrix"].get("matrix_hash")
    generated_validated_hash = generated["validated_pairs"].get("projection_hash")
    generated_terminal_hash = generated["validated_terminal_pairs"].get("projection_hash")
    generated_skill_hash = generated["validated_skill_pairs"].get("projection_hash")
    generated_gap_hash = generated["pair_gap_report"].get("report_hash")
    stale = bool(
        missing
        or invalid_self_hashes
        or current_ledger_hash != generated_ledger_hash
        or current_matrix_hash != generated_matrix_hash
        or current_validated_hash != generated_validated_hash
        or current_terminal_hash != generated_terminal_hash
        or current_skill_hash != generated_skill_hash
        or current_gap_hash != generated_gap_hash
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "stale": stale,
        "missing_artifacts": missing,
        "invalid_self_hashes": invalid_self_hashes,
        "current_ledger_hash": current_ledger_hash,
        "generated_ledger_hash": generated_ledger_hash,
        "current_matrix_hash": current_matrix_hash,
        "generated_matrix_hash": generated_matrix_hash,
        "current_validated_hash": current_validated_hash,
        "generated_validated_hash": generated_validated_hash,
        "current_terminal_hash": current_terminal_hash,
        "generated_terminal_hash": generated_terminal_hash,
        "current_skill_hash": current_skill_hash,
        "generated_skill_hash": generated_skill_hash,
        "current_gap_hash": current_gap_hash,
        "generated_gap_hash": generated_gap_hash,
    }


def freeze_pair_validation_to_memory(
    asset_catalog_path: Path = DEFAULT_ASSET_CATALOG_PATH,
    asset_ledger_path: Path = DEFAULT_ASSET_LEDGER_PATH,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    rubric_path: Path = DEFAULT_RUBRIC_PATH,
    receiver_profile_path: Path | None = None,
) -> dict[str, Any]:
    catalog = _load_yaml_required(asset_catalog_path)
    asset_ledger = _load_yaml_required(asset_ledger_path)
    rubric = _load_yaml_required(rubric_path)
    rubric_hash = _artifact_hash(rubric, "rubric_hash")
    if receiver_profile_path is None and asset_catalog_path == DEFAULT_ASSET_CATALOG_PATH:
        receiver_profile_path = DEFAULT_RECEIVER_PROFILE_PATH
    receiver_profiles = (
        _load_yaml(receiver_profile_path)
        if receiver_profile_path is not None and receiver_profile_path.is_file()
        else None
    )
    records = build_pair_records(catalog, asset_ledger, rubric, receiver_profiles=receiver_profiles)
    report_rows, report_index = load_pair_reports(reports_dir)
    records, report_application = apply_pair_validation(
        records,
        report_rows,
        expected_config_hash=config_hash(),
        expected_rubric_hash=rubric_hash,
        expected_policy_hash=terminal_probe_policy_hash(),
        expected_skill_policy_hash=skill_probe_policy_hash(),
    )
    ledger = build_ledger(
        catalog,
        asset_ledger,
        records,
        report_index,
        report_application,
        asset_catalog_path,
        asset_ledger_path,
        rubric_path=rubric_path,
        rubric_hash=rubric_hash,
    )
    geometry_matrix = build_geometry_matrix(ledger)
    terminal_projection = build_validated_projection(ledger, STAGE_TERMINAL, VALIDATED_TERMINAL_FILE)
    skill_projection = build_validated_projection(ledger, STAGE_CANONICAL, VALIDATED_SKILL_FILE)
    validated_pairs = build_combined_projection(terminal_projection, skill_projection)
    gap_report = build_gap_report(ledger, report_application)
    return {
        "ledger": ledger,
        "geometry_matrix": geometry_matrix,
        "validated_pairs": validated_pairs,
        "validated_terminal_pairs": terminal_projection,
        "validated_skill_pairs": skill_projection,
        "pair_gap_report": gap_report,
    }


def build_pair_records(
    catalog: dict[str, Any],
    asset_ledger: dict[str, Any],
    rubric: dict[str, Any] | None = None,
    *,
    receiver_profiles: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    subjects = _pick_validated_subjects(asset_ledger, catalog)
    receivers = _receiver_candidates(catalog)
    for relation in CORE_RELATIONS:
        for subject in subjects:
            for receiver in receivers.get(relation, []):
                key = pair_key(relation, subject["asset"], subject["model_id"], receiver["asset"], receiver["model_id"])
                rubric_review = _resolve_rubric_review(rubric or {}, relation, subject, receiver)
                probe_spec = _resolve_core_probe_spec(relation, subject, receiver, rubric_review)
                records[key] = _triage_pair(
                    relation,
                    subject,
                    receiver,
                    probe_spec,
                    source_candidate_id=receiver.get("candidate_id"),
                    source_candidate_use=receiver.get("candidate_use"),
                    rubric_review=rubric_review,
                )
    for candidate in catalog.get("pair_candidates") or []:
        relation = PAIR_CANDIDATE_TO_RELATION.get(str(candidate.get("use")))
        if relation is None or relation in DISABLED_RELATIONS:
            continue
        subject = _pair_endpoint(candidate.get("subject") or {})
        receiver = _pair_endpoint(candidate.get("receiver") or {})
        rubric_review = _resolve_rubric_review(rubric or {}, relation, subject, receiver)
        key = pair_key(relation, subject["asset"], subject["model_id"], receiver["asset"], receiver["model_id"])
        probe_spec = _resolve_extra_probe_spec(relation, subject, receiver, candidate)
        records[key] = _triage_pair(
            relation,
            subject,
            receiver,
            probe_spec,
            source_candidate_id=candidate.get("candidate_id"),
            source_candidate_use=candidate.get("use"),
            review=candidate.get("review"),
            rubric_review=rubric_review,
        )
    if receiver_profiles:
        records.update(_build_state_conditioned_pair_records(catalog, asset_ledger, receiver_profiles))
    return records


def _build_state_conditioned_pair_records(
    catalog: dict[str, Any],
    asset_ledger: dict[str, Any],
    receiver_profiles: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Project reviewed M5-AR terminal jobs into the common pair ledger."""

    from .articulated_pair_validation import build_articulated_probe_plan

    plan = build_articulated_probe_plan(catalog, receiver_profiles)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for job in plan.get("jobs") or []:
        if not _subject_has_validated_pick(asset_ledger, job.get("subject") or {}):
            continue
        grouped.setdefault(str(job["pair_key"]), []).append(job)
    records: dict[str, dict[str, Any]] = {}
    for key, jobs in sorted(grouped.items(), key=lambda item: _record_sort_key(item[0])):
        first = jobs[0]
        profile = first.get("receiver_state_profile") or {}
        profile_id = str(first["receiver_state_profile_id"])
        yaw_domain = first.get("relative_yaw") or {}
        terminal_recipes_by_band = {
            str(job["band_id"]): {
                "placement_recipe": job.get("placement_recipe") or {},
                "recipe_hash": job.get("recipe_hash"),
            }
            for job in jobs
        }
        probe_spec = {
            "status": "ready",
            "constructor": "articulated_place_in_cavity_link",
            "canonical_poses": first.get("canonical_poses") or {},
            "receiver_state_profile_id": profile_id,
            "receiver_state_profile_hash": first.get("receiver_state_profile_hash"),
            "fixture": profile.get("fixture") or {},
            "state": profile.get("state") or {},
            "cavity": profile.get("cavity") or {},
            "skill": profile.get("skill") or {},
            "terminal_recipes_by_band": terminal_recipes_by_band,
        }
        records[key] = {
            "pair_key": key,
            "relation": "place_in",
            "subject": first["subject"],
            "receiver": first["receiver"],
            "receiver_state_profile_id": profile_id,
            "receiver_state_profile_hash": first.get("receiver_state_profile_hash"),
            "receiver_state_profile": profile,
            "stage": STAGE_GEOMETRY,
            "status": STATUS_AMBIGUOUS,
            "reason": "reviewed_open_state_needs_terminal_settle_probe",
            "evidence_class": "heuristic",
            "probe_spec": probe_spec,
            "metrics": {
                "geometry_complete": True,
                "relative_yaw": yaw_domain,
                "best_rotated_fit_margin": yaw_domain.get("best_margin_m"),
            },
            "validations": {},
            "source_candidate_id": f"receiver_state_profile:{profile_id}",
        }
    return records


def _subject_has_validated_pick(asset_ledger: dict[str, Any], subject: dict[str, Any]) -> bool:
    uses = _asset_model_uses(asset_ledger, subject)
    return any((uses.get(use) or {}).get("status") == "validated" for use in ("pick_left", "pick_right"))


def load_pair_reports(reports_dir: Path = DEFAULT_REPORTS_DIR) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    row_by_key: dict[str, tuple[tuple[str, str], dict[str, Any]]] = {}
    reports: dict[str, Any] = {}
    if not reports_dir.is_dir():
        return [], reports
    for path in sorted(reports_dir.glob("*.json"), key=lambda p: p.name):
        report = _load_json_required(path)
        validate_normalized_report(report, expected_path=path)
        precedence = _report_precedence_key(report, path)
        reports[report["evidence_id"]] = {
            "path": _path_string(path),
            "report_hash": report.get("report_hash"),
            "kind": report.get("kind"),
            "stage": report.get("stage"),
            "source_kind": report.get("source_kind"),
            "record_count": len(report.get("records") or []),
            "started_at": (report.get("runner") or {}).get("started_at"),
            "finished_at": (report.get("runner") or {}).get("finished_at"),
            "config_hash": report.get("config_hash"),
            "source_rubric_hash": report.get("source_rubric_hash"),
            "terminal_probe_policy_hash": report.get("terminal_probe_policy_hash"),
            "skill_probe_policy_hash": report.get("skill_probe_policy_hash"),
            "skill_relation_policy_hashes": report.get(
                "skill_relation_policy_hashes"
            ),
            "canonical_pose_policy_hash": report.get("canonical_pose_policy_hash"),
            "yaw_solver_policy_hash": report.get("yaw_solver_policy_hash"),
            "articulated_policy_hash": report.get("articulated_policy_hash"),
        }
        for row in report.get("records") or []:
            key = pair_key_from_row(row)
            enriched = {
                **row,
                "evidence_id": report["evidence_id"],
                "evidence_path": _path_string(path),
                "report_config_hash": report.get("config_hash"),
                "source_rubric_hash": report.get("source_rubric_hash"),
                "terminal_probe_policy_hash": report.get("terminal_probe_policy_hash"),
                "skill_probe_policy_hash": report.get("skill_probe_policy_hash"),
                "skill_relation_policy_hash": row.get(
                    "skill_relation_policy_hash"
                ),
                "canonical_pose_policy_hash": report.get("canonical_pose_policy_hash"),
                "yaw_solver_policy_hash": report.get("yaw_solver_policy_hash"),
                "articulated_policy_hash": report.get("articulated_policy_hash"),
            }
            existing = row_by_key.get(f"{row.get('stage')}|{key}")
            if existing is None or precedence > existing[0]:
                row_by_key[f"{row.get('stage')}|{key}"] = (precedence, enriched)
    rows = [item[1] for item in sorted(row_by_key.values(), key=lambda item: _pair_row_sort_key(item[1]))]
    return rows, reports


def apply_pair_validation(
    records: dict[str, dict[str, Any]],
    report_rows: list[dict[str, Any]],
    *,
    expected_config_hash: str | None = None,
    expected_rubric_hash: str | None = None,
    expected_policy_hash: str | None = None,
    expected_skill_policy_hash: str | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    out = {key: dict(value) for key, value in records.items()}
    applied: list[dict[str, Any]] = []
    ignored: list[dict[str, Any]] = []
    for row in report_rows:
        freshness_mismatch = _report_freshness_mismatch(
            row,
            expected_config_hash=expected_config_hash,
            expected_rubric_hash=expected_rubric_hash,
            expected_policy_hash=expected_policy_hash,
            expected_skill_policy_hash=expected_skill_policy_hash,
        )
        if freshness_mismatch is not None:
            ignored.append({**_report_row_identity(row), "reason": freshness_mismatch})
            continue
        key = pair_key_from_row(row)
        relation = str(row.get("relation") or "")
        if relation in DISABLED_RELATIONS:
            ignored.append(
                {
                    **_report_row_identity(row),
                    "reason": f"relation_disabled:{relation}",
                }
            )
            continue
        current = out.get(key)
        if current is None:
            ignored.append({**_report_row_identity(row), "reason": "report_row_has_no_pair_candidate"})
            continue
        if row.get("receiver_state_profile_id") is not None:
            if row.get("receiver_state_profile_hash") != current.get("receiver_state_profile_hash"):
                ignored.append(
                    {
                        **_report_row_identity(row),
                        "reason": "stale_pair_report: receiver_state_profile_hash mismatch",
                    }
                )
                continue
        if current.get("status") in {STATUS_PRUNED, STATUS_PROBE_BLOCKED}:
            ignored.append({**_report_row_identity(row), "reason": f"pair_state_blocks_report_application: {current.get('status')}"})
            continue
        stage = str(row.get("stage"))
        if stage == STAGE_TERMINAL:
            promoted = _apply_stage_report(current, row, stage=STAGE_TERMINAL)
        elif stage == STAGE_CANONICAL:
            if not _has_stage_validation(current, STAGE_TERMINAL):
                ignored.append({**_report_row_identity(row), "reason": "canonical_skill_requires_terminal_settle_validation"})
                continue
            promoted = _apply_stage_report(current, row, stage=STAGE_CANONICAL)
        else:
            ignored.append({**_report_row_identity(row), "reason": f"unsupported_pair_report_stage: {stage}"})
            continue
        out[key] = promoted
        applied.append({**_report_row_identity(row), "status": promoted["status"], "reason": promoted["reason"]})
    return out, {"applied": applied, "ignored": ignored}


def pair_eligibility(
    ledger: dict[str, Any],
    subject_asset: str,
    subject_model_id: str,
    receiver_asset: str,
    receiver_model_id: str,
    relation: str,
    *,
    stage: str = STAGE_TERMINAL,
    require_validated: bool = True,
    receiver_state_profile_id: str | None = None,
) -> dict[str, Any]:
    if relation in DISABLED_RELATIONS:
        return {
            "eligible": False,
            "status": "disabled",
            "reason": f"relation_disabled:{relation}",
            "record": None,
        }
    key = pair_key(
        relation,
        subject_asset,
        subject_model_id,
        receiver_asset,
        receiver_model_id,
        receiver_state_profile_id=receiver_state_profile_id,
    )
    record = (ledger.get("pairs") or {}).get(key)
    if record is None:
        return {"eligible": False, "status": None, "reason": "no pair validation record", "record": None}
    if not require_validated:
        return {"eligible": True, "status": record.get("status"), "reason": record.get("reason"), "record": record}
    if stage == STAGE_GEOMETRY:
        eligible = record.get("status") == STATUS_GEOMETRY_PASS
    elif stage == STAGE_TERMINAL:
        eligible = _has_stage_validation(record, STAGE_TERMINAL) or _has_stage_validation(record, STAGE_CANONICAL)
    elif stage == STAGE_CANONICAL:
        eligible = _has_stage_validation(record, STAGE_CANONICAL)
    else:
        return {"eligible": False, "status": record.get("status"), "reason": f"unsupported pair eligibility stage: {stage}", "record": record}
    return {"eligible": bool(eligible), "status": record.get("status"), "reason": record.get("reason"), "record": record}


def plan_terminal_settle(
    asset_catalog_path: Path = DEFAULT_ASSET_CATALOG_PATH,
    asset_ledger_path: Path = DEFAULT_ASSET_LEDGER_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    rubric_path: Path = DEFAULT_RUBRIC_PATH,
    *,
    retry_failed: bool = False,
    initial_pose_set_path: Path = DEFAULT_INITIAL_POSE_SET_PATH,
) -> dict[str, Any]:
    ledger = freeze_pair_validation_to_memory(asset_catalog_path, asset_ledger_path, reports_dir, rubric_path)["ledger"]
    jobs: list[dict[str, Any]] = []
    for record in _iter_pair_records(ledger):
        if _should_plan_terminal_record(record, retry_failed=retry_failed):
            jobs.extend(_terminal_probe_jobs(record))
    jobs, pose_blockers, pose_set_hash, selection_identity = _bind_initial_pose_jobs(
        jobs, initial_pose_set_path
    )
    plan = _build_probe_plan(ledger, jobs, stage=STAGE_TERMINAL)
    plan["source_validated_pose_set_path"] = str(initial_pose_set_path)
    plan["source_validated_pose_set_hash"] = pose_set_hash
    plan["initial_state_selection_version"] = selection_identity[
        "selection_version"
    ]
    plan["initial_state_selection_hash"] = selection_identity["selection_hash"]
    plan["source_initial_state_selection_path"] = (
        str(DEFAULT_ASSIGNMENT_STATE_SELECTION_PATH)
        if initial_pose_set_path == DEFAULT_INITIAL_POSE_SET_PATH
        else None
    )
    plan["initial_pose_blockers"] = pose_blockers
    plan["summary"]["initial_pose_blocked_job_count"] = len(pose_blockers)
    plan["probe_plan_hash"] = _artifact_hash(plan, "probe_plan_hash")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_yaml(output_dir / TERMINAL_PLAN_FILE, plan)
    return plan


def plan_skill_probes(
    asset_catalog_path: Path = DEFAULT_ASSET_CATALOG_PATH,
    asset_ledger_path: Path = DEFAULT_ASSET_LEDGER_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    rubric_path: Path = DEFAULT_RUBRIC_PATH,
    *,
    initial_pose_set_path: Path = DEFAULT_INITIAL_POSE_SET_PATH,
) -> dict[str, Any]:
    artifacts = freeze_pair_validation_to_memory(asset_catalog_path, asset_ledger_path, reports_dir, rubric_path)
    ledger = artifacts["ledger"]
    asset_ledger = _load_yaml_required(asset_ledger_path)
    jobs = []
    blocked_jobs = []
    for record in _iter_pair_records(ledger):
        if not _has_stage_validation(record, STAGE_TERMINAL) or _has_stage_validation(record, STAGE_CANONICAL):
            continue
        job, blockers = _canonical_skill_probe_job(record, asset_ledger)
        if blockers:
            blocked_jobs.append(
                {
                    "pair_key": record["pair_key"],
                    "relation": record["relation"],
                    "status": STATUS_PROBE_RUN_BLOCKED,
                    "blocked_reasons": blockers,
                }
            )
        else:
            jobs.append(job)
    jobs, pose_blockers, pose_set_hash, selection_identity = _bind_initial_pose_jobs(
        jobs, initial_pose_set_path
    )
    blocked_jobs.extend(pose_blockers)
    plan = _build_probe_plan(ledger, jobs, stage=STAGE_CANONICAL)
    plan["source_validated_pose_set_path"] = str(initial_pose_set_path)
    plan["source_validated_pose_set_hash"] = pose_set_hash
    plan["initial_state_selection_version"] = selection_identity[
        "selection_version"
    ]
    plan["initial_state_selection_hash"] = selection_identity["selection_hash"]
    plan["source_initial_state_selection_path"] = (
        str(DEFAULT_ASSIGNMENT_STATE_SELECTION_PATH)
        if initial_pose_set_path == DEFAULT_INITIAL_POSE_SET_PATH
        else None
    )
    plan["blocked_jobs"] = blocked_jobs
    plan["summary"]["blocked_job_count"] = len(blocked_jobs)
    plan["probe_plan_hash"] = _artifact_hash(plan, "probe_plan_hash")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_yaml(output_dir / SKILL_PLAN_FILE, plan)
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


def normalize_report_file(path: Path) -> dict[str, Any]:
    raw = _load_json_required(path)
    if _is_normalized_report(raw):
        validate_normalized_report(raw)
        return raw
    if _looks_like_generated_scene_report(raw):
        raise ValueError("generated-scene/layout/perception reports cannot promote pair validation")
    source_kind = str(raw.get("source_kind") or "")
    if source_kind not in {TERMINAL_SOURCE_KIND, CANONICAL_SOURCE_KIND}:
        raise ValueError("Milestone 5 apply-report accepts only fresh pair probe reports")
    stage = STAGE_TERMINAL if source_kind == TERMINAL_SOURCE_KIND else STAGE_CANONICAL
    kind = "terminal_settle_pair_report" if stage == STAGE_TERMINAL else "canonical_skill_pair_report"
    records = [_normalize_report_row(item, stage=stage) for item in raw.get("records") or []]
    if not records:
        raise ValueError(f"report contains no pair-scope rows: {path}")
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "kind": kind,
        "scope": "pair",
        "stage": stage,
        "source_kind": source_kind,
        "source_path": _path_string(path),
        "source_sha256": _sha256_file(path),
        "source_probe_plan_path": raw.get("source_probe_plan_path"),
        "source_probe_plan_hash": raw.get("source_probe_plan_hash"),
        "config_hash": raw.get("config_hash"),
        "source_rubric_hash": raw.get("source_rubric_hash"),
        "terminal_probe_policy_hash": raw.get("terminal_probe_policy_hash"),
        "skill_probe_policy_hash": raw.get("skill_probe_policy_hash"),
        "skill_relation_policy_hashes": raw.get(
            "skill_relation_policy_hashes"
        ),
        "canonical_pose_policy_hash": raw.get("canonical_pose_policy_hash"),
        "yaw_solver_policy_hash": raw.get("yaw_solver_policy_hash"),
        "articulated_policy_hash": raw.get("articulated_policy_hash"),
        "source_validated_pose_set_path": raw.get(
            "source_validated_pose_set_path"
        ),
        "source_validated_pose_set_hash": raw.get(
            "source_validated_pose_set_hash"
        ),
        "initial_state_selection_version": raw.get(
            "initial_state_selection_version"
        ),
        "initial_state_selection_hash": raw.get(
            "initial_state_selection_hash"
        ),
        "runner": raw.get("runner") or {},
        "records": records,
    }
    report["evidence_id"] = _evidence_id(report)
    report["report_hash"] = _artifact_hash(report, "report_hash")
    return report


def validate_normalized_report(report: dict[str, Any], expected_path: Path | None = None) -> None:
    if not _is_normalized_report(report):
        raise ValueError("expected normalized pair report")
    if report.get("source_kind") not in {TERMINAL_SOURCE_KIND, CANONICAL_SOURCE_KIND}:
        raise ValueError("Milestone 5 accepts only fresh pair probe reports")
    if int(report.get("schema_version") or 0) != REPORT_SCHEMA_VERSION:
        raise ValueError("pair report schema is stale")
    expected_hash = _artifact_hash(report, "report_hash")
    if report.get("report_hash") != expected_hash:
        raise ValueError("pair report hash mismatch")
    expected_evidence = _evidence_id(report)
    if report.get("evidence_id") != expected_evidence:
        raise ValueError("pair report evidence id mismatch")
    if expected_path is not None and expected_path.stem != expected_evidence:
        raise ValueError("pair report filename does not match evidence id")
    stage = report.get("stage")
    if stage not in {STAGE_TERMINAL, STAGE_CANONICAL}:
        raise ValueError(f"unsupported pair report stage: {stage}")
    expected_kind = "terminal_settle_pair_report" if stage == STAGE_TERMINAL else "canonical_skill_pair_report"
    if report.get("kind") != expected_kind or report.get("scope") != "pair":
        raise ValueError("pair report kind/scope mismatch")
    for row in report.get("records") or []:
        if row.get("scope") != "pair" or row.get("stage") != stage:
            raise ValueError("pair report rows must match report scope and stage")
        if row.get("relation") not in RELATIONS:
            raise ValueError(f"unsupported pair relation: {row.get('relation')}")
        if row.get("outcome") is None and int(report.get("schema_version") or 1) < 3:
            pass
        elif row.get("outcome") not in {"pass", "fail", "blocked"}:
            raise ValueError(f"unsupported pair report outcome: {row.get('outcome')}")
        pair_key_from_row(row)
    _validate_report_probe_plan_binding(report)


def build_ledger(
    catalog: dict[str, Any],
    asset_ledger: dict[str, Any],
    records: dict[str, dict[str, Any]],
    report_index: dict[str, Any],
    report_application: dict[str, Any],
    asset_catalog_path: Path = DEFAULT_ASSET_CATALOG_PATH,
    asset_ledger_path: Path = DEFAULT_ASSET_LEDGER_PATH,
    *,
    rubric_path: Path = DEFAULT_RUBRIC_PATH,
    rubric_hash: str | None = None,
) -> dict[str, Any]:
    by_status = Counter()
    by_relation = Counter()
    by_stage = Counter()
    for record in records.values():
        by_status[record["status"]] += 1
        by_relation[record["relation"]] += 1
        by_stage[record["stage"]] += 1
    ledger = {
        "version": SCHEMA_VERSION,
        "schema_version": SCHEMA_VERSION,
        "config_hash": config_hash(),
        "source_asset_catalog_path": _path_string(asset_catalog_path),
        "source_asset_catalog_hash": catalog.get("catalog_hash"),
        "source_asset_validation_path": _path_string(asset_ledger_path),
        "source_asset_validation_hash": asset_ledger.get("ledger_hash"),
        "source_inventory_hash": catalog.get("source_inventory_hash"),
        "source_geometry_inventory_hash": catalog.get("source_geometry_inventory_hash"),
        "source_rubric_path": _path_string(rubric_path),
        "source_rubric_hash": rubric_hash,
        "terminal_probe_policy_hash": terminal_probe_policy_hash(),
        "skill_probe_policy_hash": skill_probe_policy_hash(),
        "skill_relation_policy_hashes": skill_relation_policy_hashes(),
        "relation_enablement_policy_version": RELATION_ENABLEMENT_POLICY_VERSION,
        "relation_enablement_policy_hash": relation_enablement_policy_hash(),
        "active_relations": list(ACTIVE_RELATIONS),
        "disabled_relations": DISABLED_RELATIONS,
        "canonical_pose_policy_hash": canonical_pose_policy_hash(),
        "yaw_solver_policy_hash": yaw_solver_policy_hash(),
        "reports": report_index,
        "report_application": report_application,
        "summary": {
            "pair_count": len(records),
            "pair_count_by_status": {status: by_status.get(status, 0) for status in STATUSES},
            "pair_count_by_relation": {relation: by_relation.get(relation, 0) for relation in ACTIVE_RELATIONS},
            "pair_count_by_stage": dict(sorted(by_stage.items())),
        },
        "pairs": {key: records[key] for key in sorted(records, key=_record_sort_key)},
    }
    ledger["ledger_hash"] = _artifact_hash(ledger, "ledger_hash")
    return ledger


def build_geometry_matrix(ledger: dict[str, Any]) -> dict[str, Any]:
    rows = [_matrix_row(record) for record in _iter_pair_records(ledger)]
    by_status = Counter(row["status"] for row in rows)
    by_relation = Counter(row["relation"] for row in rows)
    matrix = {
        "schema_version": SCHEMA_VERSION,
        "source_ledger_hash": ledger.get("ledger_hash"),
        "config_hash": ledger.get("config_hash"),
        "summary": {
            "pair_count": len(rows),
            "pair_count_by_status": {status: by_status.get(status, 0) for status in STATUSES},
            "pair_count_by_relation": {relation: by_relation.get(relation, 0) for relation in ACTIVE_RELATIONS},
        },
        "records": rows,
    }
    matrix["matrix_hash"] = _artifact_hash(matrix, "matrix_hash")
    return matrix


def build_validated_projection(ledger: dict[str, Any], stage: str, artifact_name: str) -> dict[str, Any]:
    rows = []
    for record in _iter_pair_records(ledger):
        stage_record = _stage_record(record, stage)
        if stage_record is None:
            continue
        rows.append(_projection_row(record, stage, stage_record))
    by_relation = Counter(row["relation"] for row in rows)
    projection = {
        "schema_version": SCHEMA_VERSION,
        "source_ledger_hash": ledger.get("ledger_hash"),
        "artifact": artifact_name,
        "stage": stage,
        "summary": {
            "pair_count": len(rows),
            "pair_count_by_relation": {relation: by_relation.get(relation, 0) for relation in ACTIVE_RELATIONS},
        },
        "records": rows,
    }
    projection["projection_hash"] = _artifact_hash(projection, "projection_hash")
    return projection


def build_combined_projection(terminal_projection: dict[str, Any], skill_projection: dict[str, Any]) -> dict[str, Any]:
    records = [
        {**row, "validated_stage": STAGE_TERMINAL}
        for row in terminal_projection.get("records") or []
    ] + [
        {**row, "validated_stage": STAGE_CANONICAL}
        for row in skill_projection.get("records") or []
    ]
    projection = {
        "schema_version": SCHEMA_VERSION,
        "terminal_projection_hash": terminal_projection.get("projection_hash"),
        "skill_projection_hash": skill_projection.get("projection_hash"),
        "summary": {
            "pair_count": len(records),
            "terminal_pair_count": len(terminal_projection.get("records") or []),
            "skill_pair_count": len(skill_projection.get("records") or []),
        },
        "records": records,
    }
    projection["projection_hash"] = _artifact_hash(projection, "projection_hash")
    return projection


def build_gap_report(ledger: dict[str, Any], report_application: dict[str, Any]) -> dict[str, Any]:
    rows = []
    reason_counts = Counter()
    for record in _iter_pair_records(ledger):
        if record.get("status") not in {
            STATUS_REJECTED,
            STATUS_PRUNED,
            STATUS_AMBIGUOUS,
            STATUS_PROBE_BLOCKED,
            STATUS_PROBE_RUN_BLOCKED,
        }:
            continue
        row = {
            "pair_key": record["pair_key"],
            "relation": record["relation"],
            "status": record["status"],
            "reason": record.get("reason"),
            "subject": record["subject"],
            "receiver": record["receiver"],
            "probe_spec_status": (record.get("probe_spec") or {}).get("status"),
        }
        if record.get("source_candidate_id"):
            row["source_candidate_id"] = record.get("source_candidate_id")
        if record.get("missing_inputs"):
            row["missing_inputs"] = record.get("missing_inputs")
        if record.get("review_unresolved"):
            row["review_unresolved"] = record.get("review_unresolved")
        rows.append(row)
        reason_counts[str(record.get("reason"))] += 1
    report = {
        "schema_version": SCHEMA_VERSION,
        "source_ledger_hash": ledger.get("ledger_hash"),
        "summary": {
            "gap_count": len(rows),
            "gap_count_by_status": dict(sorted(Counter(row["status"] for row in rows).items())),
            "gap_count_by_reason": dict(sorted(reason_counts.items())),
            "ignored_report_row_count": len(report_application.get("ignored") or []),
        },
        "records": rows,
        "ignored_report_rows": report_application.get("ignored") or [],
    }
    report["report_hash"] = _artifact_hash(report, "report_hash")
    return report


def pair_key(
    relation: str,
    subject_asset: str,
    subject_model_id: str,
    receiver_asset: str,
    receiver_model_id: str,
    receiver_state_profile_id: str | None = None,
) -> str:
    key = f"{relation}|{subject_asset}:{subject_model_id}|{receiver_asset}:{receiver_model_id}"
    if receiver_state_profile_id is not None:
        profile_id = str(receiver_state_profile_id).strip()
        if not profile_id:
            raise ValueError("receiver_state_profile_id must be non-empty when provided")
        key += f"|state:{profile_id}"
    return key


def pair_key_from_row(row: dict[str, Any]) -> str:
    subject = row.get("subject") or {}
    receiver = row.get("receiver") or {}
    return pair_key(
        str(row.get("relation")),
        str(subject.get("asset")),
        str(subject.get("model_id")),
        str(receiver.get("asset")),
        str(receiver.get("model_id")),
        receiver_state_profile_id=row.get("receiver_state_profile_id"),
    )


def _pick_validated_subjects(asset_ledger: dict[str, Any], catalog: dict[str, Any]) -> list[dict[str, Any]]:
    subjects = []
    for asset_name, asset in sorted((asset_ledger.get("assets") or {}).items(), key=lambda item: _natural_key(item[0])):
        for model_id, model in sorted((asset.get("models") or {}).items(), key=lambda item: _natural_key(item[0])):
            uses = model.get("uses") or {}
            pick_left = (uses.get("pick_left") or {}).get("status") == "validated"
            pick_right = (uses.get("pick_right") or {}).get("status") == "validated"
            if not (pick_left or pick_right):
                continue
            catalog_model = ((catalog.get("assets") or {}).get(asset_name) or {}).get("models", {}).get(str(model_id)) or {}
            candidate = (catalog_model.get("candidates") or {}).get("pick_subject") or {}
            if not candidate:
                candidate = next(iter((catalog_model.get("candidates") or {}).values()), {})
            endpoint = _candidate_endpoint(asset_name, str(model_id), candidate)
            endpoint["category"] = _first_non_empty(
                endpoint.get("category"),
                (uses.get("pick_left") or {}).get("category"),
                (uses.get("pick_right") or {}).get("category"),
            )
            endpoint["validated_uses"] = [use for use, record in sorted(uses.items()) if record.get("status") == "validated"]
            subjects.append(endpoint)
    return subjects


def _receiver_candidates(catalog: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    receivers: dict[str, list[dict[str, Any]]] = {relation: [] for relation in CORE_RELATIONS}
    for asset_name, asset in sorted((catalog.get("assets") or {}).items(), key=lambda item: _natural_key(item[0])):
        for model_id, model in sorted((asset.get("models") or {}).items(), key=lambda item: _natural_key(item[0])):
            for candidate_use, candidate in sorted((model.get("candidates") or {}).items(), key=lambda item: _natural_key(item[0])):
                relation = RECEIVER_CANDIDATE_TO_RELATION.get(candidate_use)
                if relation is None:
                    continue
                receivers[relation].append(_candidate_endpoint(asset_name, str(model_id), candidate))
    return receivers


def _candidate_endpoint(asset_name: str, model_id: str, candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "asset": asset_name,
        "model_id": str(model_id),
        "category": candidate.get("category"),
        "inventory_id": candidate.get("inventory_id"),
        "candidate_id": candidate.get("candidate_id"),
        "candidate_use": candidate.get("use"),
        "metadata": candidate.get("metadata") or {},
        "evidence": candidate.get("evidence") or {},
        "review": candidate.get("review") or {},
        "anchors": candidate.get("anchors") or {},
        "support_proxy": candidate.get("support_proxy") or {},
        "physical_defaults": candidate.get("physical_defaults") or {},
        "articulation_drive": candidate.get("articulation_drive") or {},
    }


def _pair_endpoint(endpoint: dict[str, Any]) -> dict[str, Any]:
    return {
        "asset": str(endpoint.get("asset")),
        "model_id": str(endpoint.get("model_id")),
        "category": endpoint.get("category"),
        "inventory_id": endpoint.get("inventory_id"),
        "metadata": endpoint.get("metadata") or {},
    }


def _resolve_core_probe_spec(
    relation: str,
    subject: dict[str, Any],
    receiver: dict[str, Any],
    rubric_review: dict[str, Any] | None = None,
) -> dict[str, Any]:
    review = rubric_review or {}
    if review.get("verdict") == "reject":
        return _probe_blocked("rubric_rejected")
    if review.get("verdict") == "review_required":
        return _probe_blocked("rubric_review_required")
    subject_pose = resolve_canonical_pose(subject, "pick_subject", relation=relation, role="subject")
    receiver_pose = resolve_canonical_pose(
        receiver,
        str(receiver.get("candidate_use") or "receiver"),
        relation=relation,
        role="receiver",
    )
    unresolved = [
        name
        for name, pose in (("subject_canonical_pose", subject_pose), ("receiver_canonical_pose", receiver_pose))
        if not pose.get("resolved")
    ]
    if unresolved:
        return _probe_blocked("missing_trusted_canonical_pose", unresolved)
    receiver_anchor = review.get("terminal_anchor") or _placement_anchor(relation, receiver)
    if receiver_anchor is None:
        return _probe_blocked("missing_receiver_terminal_anchor")
    if relation == "place_in":
        checker = "containment_contact_and_xy_proximity"
    elif relation == "place_on":
        checker = "support_contact_and_above_receiver"
    else:
        checker = "stack_support_contact_and_above_base"
    return {
        "status": "resolved",
        "relation": relation,
        "terminal_anchor": receiver_anchor,
        "rubric_verdict": review.get("verdict"),
        "checker": checker,
        "scene": "subject_receiver_clean_table",
        "canonical_poses": {
            "subject": subject_pose,
            "receiver": receiver_pose,
        },
        "canonical_pose_policy_hash": canonical_pose_policy_hash(),
        "yaw_solver_policy_hash": yaw_solver_policy_hash(),
    }


def _resolve_extra_probe_spec(
    relation: str,
    subject: dict[str, Any],
    receiver: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    if relation == "hang_on":
        missing = []
        if _count(subject, "functional_point_count") <= 0:
            missing.append("subject_functional_point_0")
        if _count(receiver, "functional_point_count") <= 0:
            missing.append("receiver_functional_point_0")
        if missing:
            return _probe_blocked("missing_hang_anchor", missing)
        return {
            "status": "resolved",
            "relation": relation,
            "subject_functional_point_id": 0,
            "receiver_functional_point_id": 0,
            "terminal_target": "midpoint(receiver_root, receiver_functional_point_0)",
            "checker": "subject_fp_near_receiver_midpoint_and_height",
            "source_task": "envs/hanging_mug.py",
        }
    if relation == "dump_into":
        receiver_anchor = _placement_anchor("place_in", receiver)
        if receiver_anchor is None:
            return _probe_blocked("missing_dump_receiver_target", ["receiver_target_or_cavity"])
        subject_md = subject.get("metadata") or {}
        if not _has_instantiable_metadata(subject_md):
            return _probe_blocked("missing_dump_subject_metadata", ["subject_instantiation_metadata"])
        return {
            "status": "resolved",
            "relation": relation,
            "proxy_contents": {"shape": "sphere", "count": 5, "radius_m": 0.008},
            "receiver_target": receiver_anchor,
            "pour_axis": "canonical_left_arm_pour_axis",
            "checker": "proxy_contents_inside_receiver",
            "source_task": "envs/dump_bin_bigbin.py",
        }
    if relation == "stamp_on":
        contact_points = _count(subject, "contact_points_count")
        if contact_points < 8:
            return _probe_blocked("missing_stamp_contact_face", ["subject_contact_ids_4_5_6_7"])
        receiver_anchor = _placement_anchor("place_on", receiver)
        if receiver_anchor is None:
            return _probe_blocked("missing_stamp_target_anchor", ["receiver_top_or_functional_target"])
        return {
            "status": "resolved",
            "relation": relation,
            "stamp_contact_ids": [4, 5, 6, 7],
            "receiver_target": receiver_anchor,
            "checker": "stamp_xy_alignment_and_grippers_open",
            "source_task": "envs/stamp_seal.py",
        }
    return _probe_blocked("unsupported_extra_relation")


def _triage_pair(
    relation: str,
    subject: dict[str, Any],
    receiver: dict[str, Any],
    probe_spec: dict[str, Any],
    *,
    source_candidate_id: Any = None,
    source_candidate_use: Any = None,
    review: Any = None,
    rubric_review: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = _geometry_metrics(relation, subject, receiver, probe_spec)
    key = pair_key(relation, subject["asset"], subject["model_id"], receiver["asset"], receiver["model_id"])
    base = {
        "scope": "pair",
        "pair_key": key,
        "relation": relation,
        "subject": _endpoint_identity(subject),
        "receiver": _endpoint_identity(receiver),
        "stage": STAGE_GEOMETRY,
        "status": STATUS_AMBIGUOUS,
        "reason": "needs_terminal_settle_probe",
        "metrics": metrics,
        "probe_spec": probe_spec,
        "source_candidate_id": source_candidate_id,
        "source_candidate_use": source_candidate_use,
        "config_hash": config_hash(),
        "evidence_id": _static_evidence_id(key, metrics, probe_spec),
        "evidence_path": None,
    }
    if rubric_review:
        base["rubric_review"] = rubric_review
    review_unresolved = ((review or {}).get("unresolved") if isinstance(review, dict) else None) or []
    if review_unresolved:
        base["review_unresolved"] = [str(item) for item in review_unresolved]
    if (rubric_review or {}).get("verdict") == "reject":
        base["status"] = STATUS_PRUNED
        base["reason"] = f"rubric_reject: {(rubric_review or {}).get('reason') or 'reviewed incompatible receiver use'}"
        base["evidence_class"] = "heuristic"
        return base
    if probe_spec.get("status") != "resolved":
        base["stage"] = STAGE_PROBE_SPEC
        base["status"] = STATUS_PROBE_BLOCKED
        base["reason"] = probe_spec.get("reason") or "probe_spec_blocked"
        base["missing_inputs"] = probe_spec.get("missing_inputs") or []
        return base
    if relation in CORE_RELATIONS:
        rejection = _core_rejection_reason(relation, receiver, metrics)
        if rejection is not None and (rubric_review or {}).get("pair_expectation") == "positive":
            base["status"] = STATUS_AMBIGUOUS
            base["reason"] = "rubric_positive_expectation_needs_terminal_settle_probe"
            return base
        if rejection is not None:
            base["status"] = STATUS_PRUNED
            base["reason"] = rejection
            base["evidence_class"] = "heuristic"
            return base
        if (
            (rubric_review or {}).get("pair_expectation") == "positive"
            and isinstance(metrics.get("best_rotated_fit_margin"), (int, float))
            and float(metrics["best_rotated_fit_margin"]) < 0
        ):
            base["status"] = STATUS_AMBIGUOUS
            base["reason"] = "rubric_positive_expectation_needs_terminal_settle_probe"
            return base
        if metrics.get("geometry_complete") is not True:
            base["status"] = STATUS_AMBIGUOUS
            base["reason"] = "missing_geometry_needs_terminal_settle_probe"
            return base
        if _geometry_passes(relation, subject, metrics):
            base["status"] = STATUS_GEOMETRY_PASS
            base["reason"] = "geometry_rule_pass: clearly plausible"
            return base
    base["status"] = STATUS_AMBIGUOUS
    base["reason"] = "needs_terminal_settle_probe"
    return base


def _geometry_metrics(
    relation: str,
    subject: dict[str, Any],
    receiver: dict[str, Any],
    probe_spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    canonical_poses = (probe_spec or {}).get("canonical_poses") or {}
    subject_pose = canonical_poses.get("subject")
    receiver_pose = canonical_poses.get("receiver")
    subject_geometry = pose_aware_box_geometry(subject, subject_pose) if subject_pose else None
    receiver_geometry = pose_aware_box_geometry(receiver, receiver_pose) if receiver_pose else None
    subject_fp = (
        {
            "x": subject_geometry["x"],
            "y": subject_geometry["y"],
            "area": subject_geometry["area"],
        }
        if subject_geometry
        else _footprint(subject)
    )
    receiver_fp = (
        {
            "x": receiver_geometry["x"],
            "y": receiver_geometry["y"],
            "area": receiver_geometry["area"],
        }
        if receiver_geometry
        else _footprint(receiver)
    )
    metrics: dict[str, Any] = {
        "subject_footprint": subject_fp,
        "receiver_outer_footprint": receiver_fp,
        "subject_local_footprint": _footprint(subject),
        "receiver_local_footprint": _footprint(receiver),
        "subject_height": subject_geometry.get("height") if subject_geometry else _height(subject),
        "receiver_height": receiver_geometry.get("height") if receiver_geometry else _height(receiver),
        "receiver_stable": (receiver.get("metadata") or {}).get("stable"),
        "subject_stable": (subject.get("metadata") or {}).get("stable"),
        "canonical_poses": canonical_poses,
    }
    if subject_fp is None or receiver_fp is None:
        metrics["geometry_complete"] = False
        metrics["missing_geometry"] = [
            name
            for name, value in (("subject_footprint", subject_fp), ("receiver_footprint", receiver_fp))
            if value is None
        ]
        return metrics
    factor = _usable_factor(relation)
    usable = {
        "x": _clean_float(receiver_fp["x"] * factor),
        "y": _clean_float(receiver_fp["y"] * factor),
        "area": _clean_float(receiver_fp["area"] * factor * factor),
        "factor": factor,
    }
    relative_yaw = None
    if relation in CORE_RELATIONS and subject_geometry and receiver_geometry:
        yaw_config = PAIR_GEOMETRY_CONFIG["relative_yaw"]
        relative_yaw = solve_relative_yaw_domain(
            relation,
            subject_geometry,
            receiver_geometry,
            usable_factor=factor,
            edge_clearance_m=float(PAIR_GEOMETRY_CONFIG["edge_clearance_m"]),
            receiver_support_proxy=receiver.get("support_proxy") or {},
            angle_step_deg=float(yaw_config["angle_step_deg"]),
            boundary_tolerance_deg=float(yaw_config["boundary_tolerance_deg"]),
            erosion_deg=float(yaw_config["band_erosion_deg"]),
            min_continuous_band_deg=float(yaw_config["min_continuous_band_deg"]),
        )
    best_margin = relative_yaw.get("best_margin_m") if relative_yaw else None
    best_rotation_deg = relative_yaw.get("best_delta_deg") if relative_yaw else 0.0
    area_ratio = subject_fp["area"] / usable["area"] if usable["area"] > 0 else None
    metrics.update(
        {
            "geometry_complete": True,
            "usable_footprint_proxy": usable,
            "best_rotated_fit_margin": _clean_float(best_margin) if best_margin is not None else None,
            "best_rotation_deg": _clean_float(best_rotation_deg),
            "area_ratio": _clean_float(area_ratio) if area_ratio is not None else None,
            "height_delta": _clean_float(
                float(metrics.get("receiver_height") or 0.0) - float(metrics.get("subject_height") or 0.0)
            ),
            "relative_yaw": relative_yaw,
        }
    )
    return metrics


def _core_rejection_reason(relation: str, receiver: dict[str, Any], metrics: dict[str, Any]) -> str | None:
    if relation in CORE_RELATIONS and (receiver.get("metadata") or {}).get("stable") is False:
        return "geometry_prune: receiver_or_base_known_unstable"
    relative_yaw = metrics.get("relative_yaw") or {}
    if relation == "place_in" and relative_yaw.get("outer_impossible") is True:
        return "geometry_prune: canonical_footprint_exceeds_receiver_outer_bound"
    return None


def _geometry_passes(relation: str, subject: dict[str, Any], metrics: dict[str, Any]) -> bool:
    margin = metrics.get("best_rotated_fit_margin")
    area_ratio = metrics.get("area_ratio")
    if not isinstance(margin, (int, float)) or not isinstance(area_ratio, (int, float)):
        return False
    if margin < float(PAIR_GEOMETRY_CONFIG["geometry_pass_margin_m"]):
        return False
    if area_ratio > float(PAIR_GEOMETRY_CONFIG["geometry_pass_max_area_ratio"]):
        return False
    if relation == "stack" and (subject.get("metadata") or {}).get("stable") is not True:
        return False
    return True


def _apply_stage_report(current: dict[str, Any], row: dict[str, Any], *, stage: str) -> dict[str, Any]:
    outcome = str(row.get("outcome") or ("pass" if row.get("pass") else "fail"))
    passed = outcome == "pass"
    stage_status = STATUS_VALIDATED_TERMINAL if stage == STAGE_TERMINAL else STATUS_VALIDATED_SKILL
    failed_status = STATUS_PROBE_RUN_BLOCKED if outcome == "blocked" else STATUS_REJECTED
    validation = {
        "stage": stage,
        "status": stage_status if passed else failed_status,
        "outcome": outcome,
        "reason": "ok" if passed else _report_fail_reason(row, stage),
        "evidence_id": row.get("evidence_id"),
        "evidence_path": row.get("evidence_path"),
        "config_hash": row.get("config_hash"),
        "metrics": row.get("metrics") or {},
        "verdict": row.get("verdict"),
    }
    for key in (
        "threshold",
        "min_rate",
        "seeds",
        "attempts",
        "valid_attempts",
        "plan_success",
        "success",
        "fail_reasons",
        "validated_recipe_id",
        "validated_placement_recipe",
        "validated_band_ids",
        "relative_yaw",
        "canonical_pose_policy_hash",
        "yaw_solver_policy_hash",
        "validated_skill_recipe_id",
        "validated_skill_placement_recipe",
        "validated_skill_band_ids",
        "skill_preconditions",
        "m4_evidence_refs",
        "receiver_state_profile_id",
        "receiver_state_profile_hash",
        "skill_probe_policy_hash",
        "skill_relation_policy_hash",
        "initial_state_selection_version",
        "initial_state_selection_hash",
    ):
        if key in row:
            validation[key] = row[key]
    record = dict(current)
    validations = dict(record.get("validations") or {})
    validations[stage] = validation
    record["validations"] = validations
    if passed:
        record["stage"] = stage
        record["status"] = stage_status
        record["reason"] = "ok"
        record["evidence_id"] = row.get("evidence_id")
        record["evidence_path"] = row.get("evidence_path")
        record["config_hash"] = row.get("config_hash")
    else:
        record["stage"] = stage
        record["status"] = failed_status
        record["reason"] = validation["reason"]
    if stage == STAGE_TERMINAL:
        record["relative_yaw"] = row.get("relative_yaw") or {}
        record["validated_recipe_id"] = row.get("validated_recipe_id")
        record["validated_placement_recipe"] = row.get(
            "validated_placement_recipe"
        )
    elif stage == STAGE_CANONICAL:
        record["validated_skill_recipe_id"] = row.get("validated_skill_recipe_id")
        record["validated_skill_placement_recipe"] = row.get(
            "validated_skill_placement_recipe"
        )
        record["validated_skill_band_ids"] = row.get("validated_skill_band_ids") or []
    return record


def _normalize_report_row(item: dict[str, Any], *, stage: str) -> dict[str, Any]:
    relation = str(item.get("relation"))
    if relation not in RELATIONS:
        raise ValueError(f"unsupported pair relation: {relation}")
    row = {
        "scope": "pair",
        "stage": stage,
        "relation": relation,
        "subject": _endpoint_identity(item.get("subject") or {}),
        "receiver": _endpoint_identity(item.get("receiver") or {}),
        "outcome": str(item.get("outcome") or ("pass" if item.get("pass") else "fail")),
        "pass": item.get("pass") if item.get("pass") is None else bool(item.get("pass")),
        "verdict": item.get("verdict") or item.get("reason"),
        "metrics": item.get("metrics") or {},
    }
    computed_pair_key = pair_key_from_row(row)
    supplied_pair_key = item.get("pair_key")
    if supplied_pair_key is not None and str(supplied_pair_key) != computed_pair_key:
        raise ValueError("pair report row pair_key does not match its endpoints")
    row["pair_key"] = computed_pair_key
    if item.get("job_id") is not None:
        row["job_id"] = str(item["job_id"])
    if stage in {STAGE_TERMINAL, STAGE_CANONICAL} and any(
        key in item
        for key in ("min_rate", "threshold", "seeds", "attempts", "valid_attempts", "plan_success", "success", "fail_reasons")
    ):
        row.update(
            {
                "min_rate": _clean_float(float(item.get("min_rate", 0.0))),
                "threshold": _clean_float(float(item.get("threshold", 0.6))),
                "seeds": [int(seed) for seed in item.get("seeds") or []],
                "attempts": int(item.get("attempts") or 0),
                "valid_attempts": int(item.get("valid_attempts") or item.get("plan_success") or 0),
                "plan_success": int(item.get("plan_success") or 0),
                "success": int(item.get("success") or 0),
                "fail_reasons": item.get("fail_reasons") or {},
            }
        )
    for key in (
        "validated_recipe_id",
        "validated_placement_recipe",
        "validated_band_ids",
        "relative_yaw",
        "canonical_pose_policy_hash",
        "yaw_solver_policy_hash",
        "validated_skill_recipe_id",
        "validated_skill_placement_recipe",
        "validated_skill_band_ids",
        "skill_preconditions",
        "m4_evidence_refs",
        "receiver_state_profile_id",
        "receiver_state_profile_hash",
        "skill_probe_policy_hash",
        "skill_relation_policy_hash",
        "initial_state_selection_version",
        "initial_state_selection_hash",
    ):
        if key in item:
            row[key] = item.get(key)
    row["config_hash"] = _row_config_hash(row)
    return row


def _is_normalized_report(report: dict[str, Any]) -> bool:
    return report.get("scope") == "pair" and report.get("kind") in {"terminal_settle_pair_report", "canonical_skill_pair_report"}


def _looks_like_generated_scene_report(payload: dict[str, Any]) -> bool:
    forbidden_keys = {"layout_fail", "scene_rollout_fail", "descriptor_visibility", "scene_spec", "family_id"}
    if any(key in payload for key in forbidden_keys):
        return True
    return str(payload.get("kind") or "") in {"layout_fail", "scene_rollout_fail", "perception_gate", "descriptor_gate"}


def _has_stage_validation(record: dict[str, Any], stage: str) -> bool:
    validation = _stage_record(record, stage)
    if validation is not None:
        return True
    return stage == STAGE_TERMINAL and (record.get("validations") or {}).get(STAGE_CANONICAL, {}).get("status") == STATUS_VALIDATED_SKILL


def _stage_record(record: dict[str, Any], stage: str) -> dict[str, Any] | None:
    validation = (record.get("validations") or {}).get(stage)
    if not isinstance(validation, dict):
        return None
    expected = STATUS_VALIDATED_TERMINAL if stage == STAGE_TERMINAL else STATUS_VALIDATED_SKILL
    return validation if validation.get("status") == expected else None


def _probe_job(
    record: dict[str, Any],
    *,
    stage: str,
    yaw_band: dict[str, Any] | None = None,
) -> dict[str, Any]:
    band_id = str((yaw_band or {}).get("band_id") or "fixed")
    job = {
        "job_id": f"{stage}|{record['pair_key']}|{band_id}" if stage == STAGE_TERMINAL else f"{stage}|{record['pair_key']}",
        "scope": "pair",
        "stage": stage,
        "relation": record["relation"],
        "subject": record["subject"],
        "receiver": record["receiver"],
        "pair_key": record["pair_key"],
        "probe_spec": record.get("probe_spec") or {},
        "geometry_metrics": _probe_geometry_metrics(record.get("metrics") or {}, yaw_band=yaw_band),
        "rubric_review": record.get("rubric_review") or {},
        "source_candidate_id": record.get("source_candidate_id"),
        "expected_report_schema": REPORT_SCHEMA_VERSION,
    }
    for key in (
        "receiver_state_profile_id",
        "receiver_state_profile_hash",
        "receiver_state_profile",
    ):
        if record.get(key) is not None:
            job[key] = record[key]
    if stage == STAGE_TERMINAL:
        job["yaw_witness"] = {
            **(yaw_band or {}),
            "band_id": band_id,
            "witness_delta_deg": float((yaw_band or {}).get("witness_delta_deg") or 0.0),
        }
        job["placement_recipe"] = canonical_terminal_placement_recipe(
            record["relation"]
        )
        job["recipe_id"] = _terminal_recipe_id(record, job["yaw_witness"], job["placement_recipe"])
    return job


def _canonical_skill_probe_job(
    record: dict[str, Any],
    asset_ledger: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Create one executable skill job from the strongest Terminal witness."""

    blockers: list[str] = []
    terminal = _stage_record(record, STAGE_TERMINAL) or {}
    witness = _select_terminal_skill_witness(terminal)
    if witness is None:
        blockers.append("missing_passed_terminal_witness")

    selected_arm, pick_refs, pick_blockers = _select_skill_arm(record, asset_ledger)
    blockers.extend(pick_blockers)
    m4_refs = list(pick_refs)
    receiver_state_profile_id = record.get("receiver_state_profile_id")
    receiver_state_profile_hash = record.get("receiver_state_profile_hash")
    receiver_profile = record.get("receiver_state_profile")
    if receiver_state_profile_id is not None:
        receiver_uses = _asset_model_uses(asset_ledger, record["receiver"])
        open_evidence = _validated_m4_use_ref(receiver_uses.get("open"), "open")
        if open_evidence is None:
            blockers.append("missing_validated_m4_open_evidence")
        else:
            m4_refs.append(open_evidence)
        close_evidence = _validated_m4_use_ref(receiver_uses.get("close"), "close")
        if close_evidence is not None:
            m4_refs.append(close_evidence)
        if not receiver_state_profile_hash:
            blockers.append("missing_receiver_state_profile_hash")
        if not isinstance(receiver_profile, dict):
            blockers.append("missing_receiver_state_profile")

    selected_witness = witness or {}
    band_id = str(
        selected_witness.get("band_id")
        or (selected_witness.get("yaw_witness") or {}).get("band_id")
        or "fixed"
    )
    recipe_id = str(selected_witness.get("recipe_id") or terminal.get("validated_recipe_id") or "")
    placement_recipe = dict(
        terminal.get("validated_placement_recipe")
        or canonical_terminal_placement_recipe(record["relation"])
    )
    if receiver_state_profile_id is not None and witness is not None:
        terminal_recipes = (record.get("probe_spec") or {}).get("terminal_recipes_by_band") or {}
        selected_recipe = terminal_recipes.get(band_id)
        if not isinstance(selected_recipe, dict) or not isinstance(
            selected_recipe.get("placement_recipe"), dict
        ):
            blockers.append("missing_articulated_terminal_recipe")
        else:
            placement_recipe = dict(selected_recipe["placement_recipe"])
            if recipe_id and str(placement_recipe.get("recipe_id") or "") != recipe_id:
                blockers.append("articulated_terminal_recipe_mismatch")
    skill_preconditions = {
        "terminal_evidence_id": terminal.get("evidence_id"),
        "terminal_recipe_id": recipe_id or None,
        "terminal_band_id": band_id,
        "selected_arm": selected_arm,
        "receiver_fixture": "receiver_state_profile.fixture" if receiver_state_profile_id else "canonical_clean_table",
        "articulated_entry_state": "reviewed_open_state" if receiver_state_profile_id else None,
    }
    job = _probe_job(record, stage=STAGE_CANONICAL)
    job.update(
        {
            "terminal_evidence": {
                "evidence_id": terminal.get("evidence_id"),
                "evidence_path": terminal.get("evidence_path"),
                "validated_recipe_id": terminal.get("validated_recipe_id"),
                "validated_band_ids": terminal.get("validated_band_ids") or [],
            },
            "yaw_witness": selected_witness.get("yaw_witness") or selected_witness,
            "placement_recipe": placement_recipe,
            "recipe_id": recipe_id or None,
            "band_id": band_id,
            "selected_arm": selected_arm,
            "skill_preconditions": skill_preconditions,
            "m4_evidence_refs": m4_refs,
            "skill_probe_policy_hash": skill_probe_policy_hash(),
            "skill_relation_policy_hash": skill_relation_policy_hash(
                str(record["relation"])
            ),
        }
    )
    job["job_id"] = (
        f"{job['job_id']}|policy:"
        f"{job['skill_relation_policy_hash'][:12]}"
    )
    job["probe_spec"] = dict(job.get("probe_spec") or {})
    if record["relation"] == "hang_on":
        job["probe_spec"]["hang_entry_policy"] = HANG_ENTRY_POLICY_ID
    if receiver_state_profile_id is not None:
        job.update(
            {
                "receiver_state_profile_id": receiver_state_profile_id,
                "receiver_state_profile_hash": receiver_state_profile_hash,
                "receiver_state_profile": receiver_profile,
            }
        )
    return job, list(dict.fromkeys(blockers))


def _select_terminal_skill_witness(terminal: dict[str, Any]) -> dict[str, Any] | None:
    band_results = list((terminal.get("metrics") or {}).get("band_results") or [])
    passed = [
        row
        for row in band_results
        if row.get("outcome") == "pass" or row.get("pass") is True
    ]
    if not passed:
        return None

    def sort_key(row: dict[str, Any]) -> tuple[float, str]:
        witness = row.get("yaw_witness") or {}
        margin = witness.get("max_margin_m", row.get("max_margin_m"))
        try:
            score = float(margin)
        except (TypeError, ValueError):
            score = float("-inf")
        band_id = str(row.get("band_id") or witness.get("band_id") or "fixed")
        return (-score, band_id)

    return dict(sorted(passed, key=sort_key)[0])


def _select_skill_arm(
    record: dict[str, Any],
    asset_ledger: dict[str, Any],
) -> tuple[str | None, list[dict[str, Any]], list[str]]:
    required = {"hang_on": "right", "dump_into": "left"}.get(str(record.get("relation")))
    uses = _asset_model_uses(asset_ledger, record.get("subject") or {})
    candidates: list[tuple[float, int, str, dict[str, Any]]] = []
    for arm in ("left", "right"):
        use = uses.get(f"pick_{arm}") or {}
        if use.get("status") != "validated":
            continue
        # ``left`` wins exact ties.
        candidates.append((-_m4_success_rate(use), 0 if arm == "left" else 1, arm, use))
    if required is not None:
        candidates = [item for item in candidates if item[2] == required]
    if not candidates:
        reason = f"missing_validated_m4_pick_{required}_evidence" if required else "missing_validated_m4_pick_evidence"
        return required, [], [reason]
    _, _, arm, use = sorted(candidates)[0]
    ref = _validated_m4_use_ref(use, f"pick_{arm}")
    return arm, [ref] if ref else [], []


def _asset_model_uses(asset_ledger: dict[str, Any], endpoint: dict[str, Any]) -> dict[str, Any]:
    asset = ((asset_ledger.get("assets") or {}).get(str(endpoint.get("asset"))) or {})
    model = ((asset.get("models") or {}).get(str(endpoint.get("model_id"))) or {})
    return model.get("uses") or {}


def _m4_success_rate(use: dict[str, Any]) -> float:
    metrics = use.get("metrics") or {}
    for key in ("rate", "success_rate", "min_rate"):
        try:
            return float(metrics.get(key))
        except (TypeError, ValueError):
            pass
    try:
        attempts = float(metrics.get("attempts") or use.get("attempts") or 0)
        success = float(metrics.get("success") or use.get("success") or 0)
        return success / attempts if attempts > 0 else 0.0
    except (TypeError, ValueError):
        return 0.0


def _validated_m4_use_ref(use: dict[str, Any] | None, use_name: str) -> dict[str, Any] | None:
    if not isinstance(use, dict) or use.get("status") != "validated":
        return None
    ref = {
        "scope": "asset",
        "use": use_name,
        "evidence_id": use.get("evidence_id"),
        "evidence_path": use.get("evidence_path"),
        "status": use.get("status"),
        "success_rate": _m4_success_rate(use),
    }
    if use_name.startswith("pick_"):
        validated_pose = _validated_m4_pick_pose(use, use_name)
        if validated_pose is not None:
            ref["validated_pose"] = validated_pose
    return ref


def _validated_m4_pick_pose(use: dict[str, Any], use_name: str) -> dict[str, Any] | None:
    evidence_path = use.get("evidence_path")
    if not evidence_path:
        return None
    report = _load_m4_report(str(evidence_path))
    attempts = report.get("attempts") if isinstance(report, dict) else None
    if not isinstance(attempts, list):
        return None
    matching: list[tuple[tuple[float, ...], dict[str, Any], str]] = []
    for attempt in attempts:
        if (
            not isinstance(attempt, dict)
            or attempt.get("success") is not True
            or str(attempt.get("asset")) != str(use.get("asset"))
            or str(attempt.get("model_id")) != str(use.get("model_id"))
            or str(attempt.get("use")) != str(use_name)
        ):
            continue
        pose = ((attempt.get("diagnostics") or {}).get("root_pose") or {})
        qpos = pose.get("qpos")
        if not isinstance(qpos, list) or len(qpos) != 4:
            continue
        try:
            q_key = tuple(round(float(value), 9) for value in qpos)
            normalized = {
                "q_wxyz": [float(value) for value in qpos],
                "z": float(pose.get("z", 0.741)),
                "source": str(pose.get("source") or "m4_pick_evidence"),
            }
        except (TypeError, ValueError):
            continue
        matching.append((q_key, normalized, str(attempt.get("attempt_id") or "")))
    if not matching:
        return None
    counts = Counter(row[0] for row in matching)
    selected_key = sorted(counts, key=lambda key: (-counts[key], key))[0]
    selected_rows = [row for row in matching if row[0] == selected_key]
    selected_pose = dict(selected_rows[0][1])
    selected_pose["success_count"] = len(selected_rows)
    selected_pose["attempt_ids"] = sorted(row[2] for row in selected_rows if row[2])
    return selected_pose


@lru_cache(maxsize=None)
def _load_m4_report(path_string: str) -> dict[str, Any]:
    path = Path(path_string)
    if not path.is_file():
        return {}
    try:
        return _load_json_required(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def _probe_geometry_metrics(metrics: dict[str, Any], *, yaw_band: dict[str, Any] | None) -> dict[str, Any]:
    """Keep plans diagnostic but avoid repeating a full pair domain per witness."""

    compact = {
        key: value
        for key, value in metrics.items()
        if key not in {"canonical_poses", "relative_yaw"}
    }
    relative_yaw = dict(metrics.get("relative_yaw") or {})
    if relative_yaw:
        covered = set(str(value) for value in (yaw_band or {}).get("covered_band_ids") or [])
        source_bands = list(relative_yaw.get("geometric_bands") or [])
        relative_yaw["geometric_bands"] = [
            band for band in source_bands if not covered or str(band.get("band_id")) in covered
        ]
        if (yaw_band or {}).get("band_id") != "fallback_witness":
            relative_yaw["fallback_witness"] = None
        compact["relative_yaw"] = relative_yaw
    return compact


def _terminal_probe_jobs(record: dict[str, Any]) -> list[dict[str, Any]]:
    relative_yaw = ((record.get("metrics") or {}).get("relative_yaw") or {})
    bands = list(relative_yaw.get("geometric_bands") or [])
    if not bands and relative_yaw.get("fallback_witness"):
        bands = [relative_yaw["fallback_witness"]]
    if not bands:
        bands = [
            {
                "band_id": "fixed",
                "connected_component_id": "fixed",
                "start_deg": 0.0,
                "end_deg": 0.0,
                "interval": "point",
                "continuous": False,
                "geometry_source": "relation_specific_fixed_contract",
                "max_margin_m": None,
                "witness_delta_deg": 0.0,
                "witness_status": "pending_terminal",
                "sampling_eligible": False,
            }
        ]
    # A circular connected component may be split into two non-wrapping bands
    # for storage/display.  It still owns exactly one pointwise Terminal
    # witness, so group the split segments before constructing jobs.
    components: dict[str, dict[str, Any]] = {}
    for band in bands:
        component_id = str(band.get("connected_component_id") or band.get("band_id") or "fixed")
        band_id = str(band.get("band_id") or component_id)
        if component_id not in components:
            components[component_id] = {
                **band,
                "band_id": component_id,
                "connected_component_id": component_id,
                "covered_band_ids": [],
            }
        components[component_id]["covered_band_ids"].append(band_id)
    jobs: list[dict[str, Any]] = []
    articulated_recipes = (
        (record.get("probe_spec") or {}).get("terminal_recipes_by_band")
        or {}
    )
    for component_id in sorted(components):
        witness = components[component_id]
        job = _probe_job(
            record,
            stage=STAGE_TERMINAL,
            yaw_band=witness,
        )
        covered_band_ids = list(
            witness.get("covered_band_ids") or [component_id]
        )
        recipe_rows = [
            articulated_recipes.get(str(band_id))
            for band_id in covered_band_ids
        ]
        if any(row is not None for row in recipe_rows):
            if not all(isinstance(row, dict) for row in recipe_rows):
                raise ValueError(
                    "articulated terminal component has incomplete recipes"
                )
            placement_recipes = {
                _canonical_json(row.get("placement_recipe") or {})
                for row in recipe_rows
            }
            if len(placement_recipes) != 1:
                raise ValueError(
                    "articulated terminal component mixes placement recipes"
                )
            placement_recipe = dict(
                recipe_rows[0].get("placement_recipe") or {}
            )
            recipe_id = str(placement_recipe.get("recipe_id") or "")
            if not recipe_id:
                raise ValueError(
                    "articulated terminal placement recipe lacks recipe_id"
                )
            job["placement_recipe"] = placement_recipe
            job["recipe_id"] = recipe_id
        jobs.append(job)
    return jobs


def _terminal_recipe_id(
    record: dict[str, Any],
    yaw_witness: dict[str, Any],
    placement_recipe: dict[str, Any],
) -> str:
    payload = {
        "pair_key": record.get("pair_key"),
        "probe_spec": record.get("probe_spec") or {},
        "yaw_witness": yaw_witness,
        "placement_recipe": placement_recipe,
        "terminal_probe_policy_hash": terminal_probe_policy_hash(),
    }
    return "m5_recipe_" + _sha256_bytes(_canonical_json(payload).encode("utf-8"))[:20]


def _expected_terminal_job_recipe_id(job: Mapping[str, Any]) -> str:
    """Return the exact recipe id after all initial-state bindings exist."""

    placement_recipe = dict(job.get("placement_recipe") or {})
    if placement_recipe.get("constructor") is not None:
        recipe_id = str(placement_recipe.get("recipe_id") or "")
        if not recipe_id:
            raise ValueError(
                "state-conditioned terminal placement recipe lacks recipe_id"
            )
        return recipe_id
    return _terminal_recipe_id(
        dict(job),
        dict(job.get("yaw_witness") or {}),
        placement_recipe,
    )


def _build_probe_plan(ledger: dict[str, Any], jobs: list[dict[str, Any]], *, stage: str) -> dict[str, Any]:
    plan = {
        "schema_version": SCHEMA_VERSION,
        "config_hash": config_hash(),
        "source_ledger_hash": ledger.get("ledger_hash"),
        "source_rubric_hash": ledger.get("source_rubric_hash"),
        "terminal_probe_policy_hash": ledger.get("terminal_probe_policy_hash"),
        "skill_probe_policy_hash": ledger.get("skill_probe_policy_hash") if stage == STAGE_CANONICAL else None,
        "skill_relation_policy_hashes": (
            ledger.get("skill_relation_policy_hashes")
            if stage == STAGE_CANONICAL
            else None
        ),
        "relation_enablement_policy_version": ledger.get(
            "relation_enablement_policy_version"
        ),
        "relation_enablement_policy_hash": ledger.get(
            "relation_enablement_policy_hash"
        ),
        "active_relations": list(ledger.get("active_relations") or ACTIVE_RELATIONS),
        "disabled_relations": dict(
            ledger.get("disabled_relations") or DISABLED_RELATIONS
        ),
        "canonical_pose_policy_hash": ledger.get("canonical_pose_policy_hash"),
        "yaw_solver_policy_hash": ledger.get("yaw_solver_policy_hash"),
        "stage": stage,
        "summary": {
            "probe_job_count": len(jobs),
            "probe_pair_count": len({str(job["pair_key"]) for job in jobs}),
            "probe_job_count_by_relation": dict(sorted(Counter(job["relation"] for job in jobs).items())),
        },
        "jobs": jobs,
    }
    plan["probe_plan_hash"] = _artifact_hash(plan, "probe_plan_hash")
    return plan


@lru_cache(maxsize=16)
def _load_probe_plan_cached(
    path_string: str,
    stat_identity: tuple[int, int, int, int, int],
) -> dict[str, Any]:
    del stat_identity
    return _load_yaml_required(Path(path_string))


def _load_bound_probe_plan(report: Mapping[str, Any]) -> dict[str, Any]:
    path_text = str(report.get("source_probe_plan_path") or "")
    declared_hash = str(report.get("source_probe_plan_hash") or "")
    if not path_text or not declared_hash:
        raise ValueError("pair report lacks exact source probe-plan lineage")
    path = Path(path_text)
    try:
        stat = path.stat()
    except OSError as exc:
        raise ValueError(f"pair report source probe plan is unavailable: {path}") from exc
    identity = (
        int(stat.st_dev),
        int(stat.st_ino),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    )
    plan = _load_probe_plan_cached(str(path), identity)
    observed_hash = _artifact_hash(plan, "probe_plan_hash")
    if (
        int(plan.get("schema_version") or 0) != SCHEMA_VERSION
        or str(plan.get("probe_plan_hash") or "") != observed_hash
        or declared_hash != observed_hash
        or str(plan.get("stage") or "") != str(report.get("stage") or "")
    ):
        raise ValueError("pair report source probe-plan binding is stale")
    return plan


def _same_endpoint(left: Any, right: Any) -> bool:
    return _endpoint_identity(dict(left or {})) == _endpoint_identity(
        dict(right or {})
    )


def _validate_job_identity(row: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if (
        str(row.get("relation") or "") != str(job.get("relation") or "")
        or str(row.get("pair_key") or pair_key_from_row(dict(row)))
        != str(job.get("pair_key") or "")
        or not _same_endpoint(row.get("subject"), job.get("subject"))
        or not _same_endpoint(row.get("receiver"), job.get("receiver"))
        or pair_key_from_row(dict(job)) != str(job.get("pair_key") or "")
    ):
        raise ValueError("pair report row does not match its source probe job")


def _validate_terminal_report_row(
    row: Mapping[str, Any],
    jobs_by_id: Mapping[str, Mapping[str, Any]],
    used_job_ids: set[str],
) -> None:
    band_results = (row.get("metrics") or {}).get("band_results")
    if (
        not isinstance(band_results, Sequence)
        or isinstance(band_results, (str, bytes))
        or not band_results
    ):
        raise ValueError("terminal pair report lacks per-job band results")
    normalized_bands: list[Mapping[str, Any]] = []
    for band in band_results:
        if not isinstance(band, Mapping):
            raise ValueError("terminal pair report band result is invalid")
        job_id = str(band.get("job_id") or "")
        job = jobs_by_id.get(job_id)
        if not job_id or job is None or job_id in used_job_ids:
            raise ValueError("terminal pair report has an unknown or duplicate job_id")
        used_job_ids.add(job_id)
        _validate_job_identity(band, job)
        _validate_job_identity(row, job)
        expected_band_id = str(
            (job.get("yaw_witness") or {}).get("band_id") or "fixed"
        )
        expected_recipe_id = _expected_terminal_job_recipe_id(job)
        if (
            str(job.get("job_id") or "") != job_id
            or int(job.get("expected_report_schema") or 0)
            != REPORT_SCHEMA_VERSION
            or str(band.get("band_id") or "") != expected_band_id
            or canonical_data(band.get("yaw_witness") or {})
            != canonical_data(job.get("yaw_witness") or {})
            or str(job.get("recipe_id") or "") != expected_recipe_id
            or str(band.get("recipe_id") or "") != expected_recipe_id
            or canonical_data(band.get("placement_recipe") or {})
            != canonical_data(job.get("placement_recipe") or {})
            or band.get("initial_state_selection_version")
            != job.get("initial_state_selection_version")
            or band.get("initial_state_selection_hash")
            != job.get("initial_state_selection_hash")
        ):
            raise ValueError("terminal pair report recipe/job binding mismatch")
        normalized_bands.append(band)

    passed_bands = sorted(
        (band for band in normalized_bands if band.get("outcome") == "pass"),
        key=lambda band: str(band.get("band_id") or ""),
    )
    if passed_bands:
        selected = passed_bands[0]
        expected_band_ids = sorted(
            {
                str(band_id)
                for band in passed_bands
                for band_id in (
                    list((band.get("yaw_witness") or {}).get("covered_band_ids") or [])
                    or [str(band.get("band_id") or "")]
                )
            }
        )
        if (
            row.get("pass") is not True
            or row.get("outcome") != "pass"
            or row.get("validated_recipe_id") != selected.get("recipe_id")
            or canonical_data(row.get("validated_placement_recipe") or {})
            != canonical_data(selected.get("placement_recipe") or {})
            or sorted(str(value) for value in row.get("validated_band_ids") or [])
            != expected_band_ids
        ):
            raise ValueError("terminal pair report validated witness mismatch")
    elif (
        row.get("pass") is True
        or row.get("validated_recipe_id") is not None
        or bool(row.get("validated_placement_recipe"))
        or bool(row.get("validated_band_ids"))
    ):
        raise ValueError("terminal pair report promotes a non-passing job")


def _validate_canonical_report_row(
    row: Mapping[str, Any],
    jobs_by_id: Mapping[str, Mapping[str, Any]],
    used_job_ids: set[str],
) -> None:
    job_id = str(row.get("job_id") or "")
    job = jobs_by_id.get(job_id)
    if not job_id or job is None or job_id in used_job_ids:
        raise ValueError("canonical pair report has an unknown or duplicate job_id")
    used_job_ids.add(job_id)
    _validate_job_identity(row, job)
    band_id = str(
        job.get("band_id")
        or (job.get("yaw_witness") or {}).get("band_id")
        or "fixed"
    )
    expected_bands = [band_id] if row.get("pass") is True else []
    if (
        int(job.get("expected_report_schema") or 0) != REPORT_SCHEMA_VERSION
        or row.get("validated_skill_recipe_id") != job.get("recipe_id")
        or canonical_data(row.get("validated_skill_placement_recipe") or {})
        != canonical_data(job.get("placement_recipe") or {})
        or list(row.get("validated_skill_band_ids") or []) != expected_bands
        or canonical_data(row.get("skill_preconditions") or {})
        != canonical_data(job.get("skill_preconditions") or {})
        or canonical_data(row.get("m4_evidence_refs") or [])
        != canonical_data(job.get("m4_evidence_refs") or [])
        or row.get("skill_probe_policy_hash")
        != job.get("skill_probe_policy_hash")
        or row.get("skill_relation_policy_hash")
        != job.get("skill_relation_policy_hash")
        or row.get("initial_state_selection_version")
        != job.get("initial_state_selection_version")
        or row.get("initial_state_selection_hash")
        != job.get("initial_state_selection_hash")
    ):
        raise ValueError("canonical pair report recipe/job binding mismatch")


def _validate_report_probe_plan_binding(report: Mapping[str, Any]) -> None:
    plan = _load_bound_probe_plan(report)
    stage = str(report.get("stage") or "")
    top_level_fields = (
        (
            "config_hash",
            "source_rubric_hash",
            "terminal_probe_policy_hash",
            "canonical_pose_policy_hash",
            "yaw_solver_policy_hash",
        )
        if stage == STAGE_TERMINAL
        else (
            "config_hash",
            "source_rubric_hash",
            "skill_probe_policy_hash",
            "skill_relation_policy_hashes",
            "source_validated_pose_set_hash",
            "initial_state_selection_version",
            "initial_state_selection_hash",
        )
    )
    for field in top_level_fields:
        if canonical_data(report.get(field)) != canonical_data(plan.get(field)):
            raise ValueError(f"pair report probe-plan {field} mismatch")
    jobs_by_id: dict[str, Mapping[str, Any]] = {}
    for job in plan.get("jobs") or []:
        if not isinstance(job, Mapping):
            raise ValueError("pair probe plan contains an invalid job")
        job_id = str(job.get("job_id") or "")
        if not job_id or job_id in jobs_by_id:
            raise ValueError("pair probe plan contains duplicate job ids")
        _validate_job_identity(job, job)
        jobs_by_id[job_id] = job
    used_job_ids: set[str] = set()
    rows = report.get("records") or []
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("pair report contains invalid records")
    selected_job_count = (report.get("runner") or {}).get(
        "selected_job_count"
    )
    if not rows:
        if selected_job_count is not None and int(selected_job_count) == 0:
            return
        raise ValueError("pair report contains no records")
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("pair report contains an invalid row")
        if stage == STAGE_TERMINAL:
            _validate_terminal_report_row(row, jobs_by_id, used_job_ids)
        else:
            _validate_canonical_report_row(row, jobs_by_id, used_job_ids)
    if selected_job_count is not None and int(selected_job_count) != len(used_job_ids):
        raise ValueError("pair report selected_job_count does not match bound jobs")


def _bind_initial_pose_jobs(
    jobs: list[dict[str, Any]], initial_pose_set_path: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, dict[str, str]]:
    """Bind both M5 endpoints to support-validated human-approved states."""

    pose_set = _load_yaml_required(initial_pose_set_path)
    validate_initial_pose_set(pose_set, require_support_validated=True)
    pose_set_hash = str(pose_set.get("validated_pose_set_hash") or "")
    if pose_set_hash != artifact_hash(
        pose_set, omit_keys=("validated_pose_set_hash",)
    ):
        raise InitialPoseContractError("Initial Pose Set self-hash mismatch")
    states = {
        (
            str(row["asset"]),
            str(row["model_id"]),
            str(row.get("initial_state_profile_id") or "default"),
        ): row
        for row in pose_set.get("canonical_initial_states") or []
    }
    selection = (
        load_assignment_state_selection()
        if initial_pose_set_path == DEFAULT_INITIAL_POSE_SET_PATH
        else None
    )
    assignment_states = canonical_assignment_states(pose_set, selection)
    selection_identity = canonical_assignment_selection_identity(selection)
    bound = []
    blockers = []
    for original in jobs:
        job = dict(original)
        endpoint_states: dict[str, dict[str, Any]] = {}
        endpoint_bindings: dict[str, dict[str, Any]] = {}
        reasons = []
        for endpoint_name in ("subject", "receiver"):
            endpoint = job.get(endpoint_name) or {}
            profile_id = "default"
            if endpoint_name == "receiver":
                profile_id = str(
                    job.get("receiver_state_profile_id")
                    or (job.get("probe_spec") or {}).get(
                        "receiver_state_profile_id"
                    )
                    or "default"
                )
            asset = str(endpoint.get("asset"))
            model_id = str(endpoint.get("model_id"))
            if profile_id == "default":
                state = assignment_states.get((asset, model_id))
            else:
                state = states.get((asset, model_id, profile_id))
            if state is None:
                reasons.append(
                    f"{endpoint_name}:missing_initial_pose_state:{profile_id}"
                )
                continue
            if state.get("status") == "unsupported":
                reasons.append(f"{endpoint_name}:human_unsupported")
                continue
            if (state.get("support_validation") or {}).get("status") != "validated":
                reasons.append(f"{endpoint_name}:support_not_validated")
                continue
            validate_canonical_initial_state(state)
            selection_entry = (
                (selection.get("selections") or {}).get(f"{asset}|{model_id}")
                if selection is not None
                and state.get("assignment_state_revision") is not None
                else None
            )
            approval = assignment_approval_for_state(
                pose_set,
                state,
                selection_entry,
            )
            endpoint_states[endpoint_name] = state
            binding = {
                "initial_state_id": state["initial_state_id"],
                "revision": state["revision"],
                "approval_hash": approval["approval_hash"],
                "geometry_hash": state["geometry_hash"],
            }
            validate_pose_binding(binding, state, approval)
            endpoint_bindings[endpoint_name] = binding
        if reasons:
            blockers.append(
                {
                    "pair_key": job.get("pair_key"),
                    "relation": job.get("relation"),
                    "status": STATUS_PROBE_RUN_BLOCKED,
                    "blocked_reasons": reasons,
                }
            )
            continue
        probe_spec = dict(job.get("probe_spec") or {})
        probe_spec["canonical_poses"] = {
            endpoint_name: {
                "initial_state_id": state["initial_state_id"],
                "revision": state["revision"],
                "revision_hash": state["revision_hash"],
                "geometry_hash": state["geometry_hash"],
                "q0_wxyz": state["q0_wxyz"],
                "q_wxyz": state["q0_wxyz"],
                "z": state["support"]["origin_z"],
                "support": state["support"],
                "yaw_policy": state["yaw_policy"],
                "origin_relative_bounds": state["origin_relative_bounds"],
                "articulated_joint_state": state["articulated_joint_state"],
                "source": "support_validated_human_approved_initial_pose",
            }
            for endpoint_name, state in endpoint_states.items()
        }
        job["probe_spec"] = probe_spec
        job["canonical_initial_states"] = endpoint_states
        job["initial_pose_bindings"] = endpoint_bindings
        job["validated_pose_set_hash"] = pose_set_hash
        job["initial_state_selection_version"] = selection_identity[
            "selection_version"
        ]
        job["initial_state_selection_hash"] = selection_identity["selection_hash"]
        if str(job.get("stage") or "") == STAGE_TERMINAL:
            job["recipe_id"] = _expected_terminal_job_recipe_id(job)
        bound.append(job)
    return bound, blockers, pose_set_hash, selection_identity


def _matrix_row(record: dict[str, Any]) -> dict[str, Any]:
    row = {
        "pair_key": record["pair_key"],
        "relation": record["relation"],
        "subject": record["subject"],
        "receiver": record["receiver"],
        "status": record["status"],
        "stage": record["stage"],
        "reason": record["reason"],
        "metrics": record.get("metrics") or {},
        "probe_spec_status": (record.get("probe_spec") or {}).get("status"),
        "evidence_id": record.get("evidence_id"),
    }
    if record.get("missing_inputs"):
        row["missing_inputs"] = record.get("missing_inputs")
    return row


def _projection_row(record: dict[str, Any], stage: str, stage_record: dict[str, Any]) -> dict[str, Any]:
    row = {
        "pair_key": record["pair_key"],
        "relation": record["relation"],
        "subject": record["subject"],
        "receiver": record["receiver"],
        "stage": stage,
        "status": stage_record.get("status"),
        "reason": stage_record.get("reason"),
        "evidence_id": stage_record.get("evidence_id"),
        "evidence_path": stage_record.get("evidence_path"),
        "config_hash": stage_record.get("config_hash"),
        "metrics": stage_record.get("metrics") or {},
    }
    if stage == STAGE_TERMINAL:
        row["validated_recipe_id"] = stage_record.get("validated_recipe_id")
        row["validated_placement_recipe"] = stage_record.get(
            "validated_placement_recipe"
        )
        row["validated_band_ids"] = stage_record.get("validated_band_ids") or []
        row["relative_yaw"] = stage_record.get("relative_yaw") or {}
        for key in (
            "receiver_state_profile_id",
            "receiver_state_profile_hash",
        ):
            if stage_record.get(key) is not None:
                row[key] = stage_record[key]
    elif stage == STAGE_CANONICAL:
        for key, default in (
            ("validated_skill_recipe_id", None),
            ("validated_skill_placement_recipe", None),
            ("validated_skill_band_ids", []),
            ("skill_preconditions", {}),
            ("m4_evidence_refs", []),
            ("receiver_state_profile_id", None),
            ("receiver_state_profile_hash", None),
            ("skill_probe_policy_hash", None),
            ("skill_relation_policy_hash", None),
        ):
            value = stage_record.get(key)
            if value is not None or default not in (None, [], {}):
                row[key] = value if value is not None else default
    return row


def _placement_anchor(relation: str, endpoint: dict[str, Any]) -> dict[str, Any] | None:
    metadata = endpoint.get("metadata") or {}
    anchors = endpoint.get("anchors") or {}
    selected = anchors.get("selected")
    if selected is not None:
        return {
            "type": "functional_point",
            "id": int(selected),
            "source": anchors.get("selected_source") or "candidate_review",
        }
    if _count(endpoint, "functional_point_count") > 0:
        return {"type": "functional_point", "id": 0}
    if _count(endpoint, "target_pose_count") > 0:
        return {"type": "target_pose", "id": 0}
    if relation == "stack" and _footprint(endpoint) is not None and _height(endpoint) is not None:
        return {"type": "derived_top_center", "source": "metadata_footprint_height"}
    if relation in {"place_on", "place_in"} and _has_instantiable_metadata(metadata):
        return {"type": "derived_receiver_center", "source": "metadata_footprint_height"}
    return None


def _resolve_rubric_review(
    rubric: dict[str, Any],
    relation: str,
    subject: dict[str, Any],
    receiver: dict[str, Any],
) -> dict[str, Any]:
    matches: list[tuple[int, dict[str, Any]]] = []
    receiver_asset = str(receiver.get("asset"))
    receiver_model = str(receiver.get("model_id"))
    candidate_use = str(receiver.get("candidate_use") or "")
    for row in rubric.get("receiver_reviews") or []:
        if str(row.get("asset")) != receiver_asset or str(row.get("use")) != candidate_use:
            continue
        model_id = str(row.get("model_id", "*"))
        if model_id not in {"*", receiver_model}:
            continue
        matches.append((1 if model_id == receiver_model else 0, row))
    review = dict(max(matches, key=lambda item: item[0])[1]) if matches else {}
    if review.get("verdict") == "specialized":
        allowed = {str(value) for value in review.get("allowed_subject_categories") or []}
        subject_category = str(subject.get("category") or "")
        if subject_category not in allowed:
            review["verdict"] = "reject"
            review["reason"] = (
                f"specialized receiver allows subject categories {sorted(allowed)}; "
                f"got {subject_category or '<missing>'}"
            )
        else:
            review["verdict"] = "accept"
    expectation = next(
        (
            row
            for row in rubric.get("expectations") or []
            if str(row.get("relation")) == relation
            and str(row.get("subject_asset")) == str(subject.get("asset"))
            and str(row.get("receiver_asset")) == str(receiver.get("asset"))
            and str(row.get("subject_model_id", "*")) in {"*", str(subject.get("model_id"))}
            and str(row.get("receiver_model_id", "*")) in {"*", str(receiver.get("model_id"))}
        ),
        None,
    )
    if expectation is not None:
        subject_model_selector = str(expectation.get("subject_model_id", "*"))
        receiver_model_selector = str(expectation.get("receiver_model_id", "*"))
        review["pair_expectation"] = str(expectation.get("expected"))
        review["pair_expectation_evidence"] = expectation.get("evidence")
        review["pair_expectation_scope"] = (
            "model_pair"
            if subject_model_selector != "*" and receiver_model_selector != "*"
            else "asset_family"
        )
        if expectation.get("expected") == "negative":
            review["verdict"] = "reject"
            review["reason"] = str(expectation.get("reason") or "reviewed negative pair expectation")
    return review


def _should_plan_terminal_record(record: dict[str, Any], *, retry_failed: bool) -> bool:
    # State-conditioned Terminal probes retain their dedicated M5-AR runner
    # and checkpoints; their normalized reports are projected into this
    # ledger, but the rigid terminal worker must never receive them.
    if record.get("receiver_state_profile_id") is not None:
        return False
    if record.get("status") in {STATUS_GEOMETRY_PASS, STATUS_AMBIGUOUS}:
        return not _has_stage_validation(record, STAGE_TERMINAL)
    if retry_failed:
        validation = (record.get("validations") or {}).get(STAGE_TERMINAL) or {}
        return validation.get("status") in {STATUS_REJECTED, STATUS_PROBE_RUN_BLOCKED}
    return False


def _report_freshness_mismatch(
    row: dict[str, Any],
    *,
    expected_config_hash: str | None,
    expected_rubric_hash: str | None,
    expected_policy_hash: str | None,
    expected_skill_policy_hash: str | None,
) -> str | None:
    expected: tuple[tuple[str, str | None], ...] = (
        ("report_config_hash", expected_config_hash),
        ("source_rubric_hash", expected_rubric_hash),
    )
    if row.get("stage") == STAGE_TERMINAL:
        if row.get("receiver_state_profile_id") is not None:
            from .articulated_pair_validation import articulated_policy_hash

            expected += (("articulated_policy_hash", articulated_policy_hash()),)
        else:
            expected += (
                ("terminal_probe_policy_hash", expected_policy_hash),
                ("canonical_pose_policy_hash", canonical_pose_policy_hash()),
                ("yaw_solver_policy_hash", yaw_solver_policy_hash()),
            )
    elif row.get("stage") == STAGE_CANONICAL:
        expected += (("skill_probe_policy_hash", expected_skill_policy_hash),)
        relation = str(row.get("relation") or "")
        actual_relation_hash = row.get("skill_relation_policy_hash")
        if actual_relation_hash is not None:
            expected += (
                (
                    "skill_relation_policy_hash",
                    skill_relation_policy_hash(relation),
                ),
            )
        elif (
            expected_skill_policy_hash == skill_probe_policy_hash()
            and _requires_explicit_skill_relation_hash(relation)
        ):
            return (
                "stale_pair_report: skill_relation_policy_hash "
                f"missing for {relation}"
            )
    for key, value in expected:
        if value is not None and row.get(key) != value:
            return f"stale_pair_report: {key} mismatch"
    return None


def _probe_blocked(reason: str, missing_inputs: list[str] | None = None) -> dict[str, Any]:
    return {
        "status": "blocked",
        "reason": reason,
        "missing_inputs": list(missing_inputs or []),
    }


def _footprint(endpoint: dict[str, Any]) -> dict[str, float] | None:
    fp = (endpoint.get("metadata") or {}).get("xy_footprint") or {}
    try:
        x = float(fp.get("x"))
        y = float(fp.get("y"))
        area = float(fp.get("area"))
    except (TypeError, ValueError):
        return None
    if x <= 0 or y <= 0 or area <= 0:
        return None
    return {"x": _clean_float(x), "y": _clean_float(y), "area": _clean_float(area)}


def _height(endpoint: dict[str, Any]) -> float | None:
    value = (endpoint.get("metadata") or {}).get("height")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return _clean_float(number) if number > 0 else None


def _has_instantiable_metadata(metadata: dict[str, Any]) -> bool:
    if not metadata:
        return False
    visual = metadata.get("has_visual_mesh")
    collision = metadata.get("has_collision_mesh")
    return visual is not False and collision is not False


def _usable_factor(relation: str) -> float:
    if relation == "place_in":
        return float(PAIR_GEOMETRY_CONFIG["place_in_cavity_fraction"])
    if relation == "stack":
        return float(PAIR_GEOMETRY_CONFIG["stack_support_fraction"])
    return float(PAIR_GEOMETRY_CONFIG["place_on_support_fraction"])


def _endpoint_identity(endpoint: dict[str, Any]) -> dict[str, Any]:
    row = {
        "asset": str(endpoint.get("asset")),
        "model_id": str(endpoint.get("model_id")),
    }
    if endpoint.get("category") is not None:
        row["category"] = endpoint.get("category")
    if endpoint.get("inventory_id") is not None:
        row["inventory_id"] = endpoint.get("inventory_id")
    return row


def _count(endpoint: dict[str, Any], key: str) -> int:
    try:
        return int((endpoint.get("metadata") or {}).get(key) or 0)
    except (TypeError, ValueError):
        return 0


def _first_non_empty(*values: Any) -> Any:
    for value in values:
        if value:
            return value
    return None


def _iter_pair_records(ledger: dict[str, Any]) -> list[dict[str, Any]]:
    return [record for _, record in sorted((ledger.get("pairs") or {}).items(), key=lambda item: _record_sort_key(item[0]))]


def _report_fail_reason(row: dict[str, Any], stage: str) -> str:
    verdict = row.get("verdict") or "failed"
    suffix = "blocked" if row.get("outcome") == "blocked" else "fail"
    return f"{stage}_{suffix}: {verdict}"


def _report_row_identity(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "relation": row.get("relation"),
        "subject": row.get("subject"),
        "receiver": row.get("receiver"),
        "stage": row.get("stage"),
        "evidence_id": row.get("evidence_id"),
        "receiver_state_profile_id": row.get("receiver_state_profile_id"),
    }


def _static_evidence_id(pair_key_value: str, metrics: dict[str, Any], probe_spec: dict[str, Any]) -> str:
    payload = {
        "pair_key": pair_key_value,
        "metrics": metrics,
        "probe_spec": probe_spec,
        "config_hash": config_hash(),
    }
    return "pair_geometry_" + _sha256_bytes(_canonical_json(payload).encode("utf-8"))[:20]


def _row_config_hash(row: dict[str, Any]) -> str:
    keys = (
        "scope",
        "stage",
        "relation",
        "threshold",
        "seeds",
        "receiver_state_profile_id",
        "receiver_state_profile_hash",
        "skill_probe_policy_hash",
        "skill_relation_policy_hash",
    )
    return _sha256_bytes(_canonical_json({key: row.get(key) for key in keys}).encode("utf-8"))


def _evidence_id(report: dict[str, Any]) -> str:
    payload = {key: value for key, value in report.items() if key not in {"evidence_id", "report_hash"}}
    prefix = "terminal_settle_pair" if report.get("stage") == STAGE_TERMINAL else "canonical_skill_pair"
    return f"{prefix}_{_sha256_bytes(_canonical_json(payload).encode('utf-8'))[:20]}"


def _artifact_hash(data: dict[str, Any], hash_key: str) -> str:
    payload = {key: value for key, value in data.items() if key != hash_key}
    return _sha256_bytes(_canonical_json(payload).encode("utf-8"))


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _natural_key(value: Any) -> tuple[Any, ...]:
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", str(value)))


def _record_sort_key(key: str) -> tuple[Any, ...]:
    parts = key.split("|")
    if len(parts) not in {3, 4}:
        raise ValueError(f"invalid pair key: {key}")
    relation, subject, receiver = parts[:3]
    state = ""
    if len(parts) == 4:
        if not parts[3].startswith("state:"):
            raise ValueError(f"invalid pair state suffix: {key}")
        state = parts[3][len("state:") :]
    subject_asset, subject_model = subject.rsplit(":", 1)
    receiver_asset, receiver_model = receiver.rsplit(":", 1)
    return (
        _natural_key(relation),
        _natural_key(subject_asset),
        _natural_key(subject_model),
        _natural_key(receiver_asset),
        _natural_key(receiver_model),
        _natural_key(state),
    )


def _pair_row_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    stage_order = {STAGE_TERMINAL: 0, STAGE_CANONICAL: 1}
    return (stage_order.get(str(row.get("stage")), 99), _record_sort_key(pair_key_from_row(row)))


def _report_precedence_key(report: dict[str, Any], path: Path) -> tuple[str, str]:
    runner = report.get("runner") or {}
    timestamp = str(runner.get("finished_at") or runner.get("started_at") or "")
    return (timestamp, path.name)


def _clean_float(value: float) -> float:
    return round(float(value), 12)


def _load_yaml_required(path: Path) -> dict[str, Any]:
    data = _load_yaml(path)
    if data is None:
        raise FileNotFoundError(path)
    return data


def _load_yaml(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=yaml.CSafeLoader)
    if data is None:
        return None
    if not isinstance(data, dict):
        raise ValueError(f"expected YAML mapping in {path}")
    return data


def _load_top_level_scalar(path: Path, key: str) -> Any:
    if not path.is_file():
        return None
    prefix = f"{key}:"
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith(prefix):
                return yaml.safe_load(line[len(prefix):].strip())
    return None


def _load_json_required(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.parent / f".{path.name}.tmp"
    with temporary_path.open("w", encoding="utf-8") as f:
        yaml.dump(
            data,
            f,
            Dumper=yaml.CSafeDumper,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
            width=100,
        )
    temporary_path.replace(path)


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


def _path_string(path: Path | None) -> str | None:
    if path is None:
        return None
    return str(path).replace("\\", "/")


def _status_line(command: str, artifacts: dict[str, Any]) -> str:
    summary = artifacts["ledger"].get("summary", {})
    return (
        f"{command}: wrote {summary.get('pair_count', 0)} pair validation records; "
        f"terminal {artifacts['validated_terminal_pairs'].get('summary', {}).get('pair_count', 0)}, "
        f"skill {artifacts['validated_skill_pairs'].get('summary', {}).get('pair_count', 0)}"
    )


def _check_status_line(report: dict[str, Any]) -> str:
    if report.get("stale"):
        missing = report.get("missing_artifacts", [])
        suffix = f"; missing {missing}" if missing else ""
        return f"check: pair validation artifacts are stale{suffix}"
    return "check: pair validation artifacts are current"


def _parallel_worker_count(parallel_workers: int | None, workers_per_gpu: int, cuda_devices: list[str] | None) -> int:
    if parallel_workers is not None:
        return max(1, int(parallel_workers))
    visible_device_count = len(cuda_devices) if cuda_devices else 1
    return max(1, visible_device_count * max(1, int(workers_per_gpu)))


def _terminal_cpu_capacity() -> int:
    """Return the CPU capacity visible to the canonical headless M5 gate."""

    try:
        affinity_count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity_count = int(os.cpu_count() or 1)

    quota_count: int | None = None
    cpu_max = Path("/sys/fs/cgroup/cpu.max")
    try:
        quota_text, period_text = cpu_max.read_text(encoding="utf-8").split()[:2]
        if quota_text != "max":
            quota = int(quota_text)
            period = int(period_text)
            if quota > 0 and period > 0:
                quota_count = max(1, quota // period)
    except (OSError, ValueError):
        pass

    return max(
        1,
        min(affinity_count, quota_count)
        if quota_count is not None
        else affinity_count,
    )


def _terminal_parallel_worker_count(parallel_workers: int | None) -> int:
    if parallel_workers is not None:
        return max(1, int(parallel_workers))
    # CPU PhysX plus process/checkpoint overhead peaked near 80% of the cgroup
    # quota in the canonical 165-attempt sweep.  Leave headroom for the runner
    # and filesystem while bounding cold-process storms on large hosts.
    return max(1, min(64, int(_terminal_cpu_capacity() * 0.8)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze or check BenchmarkV1 pair validation artifacts.")
    parser.add_argument("--asset-catalog", type=Path, default=DEFAULT_ASSET_CATALOG_PATH)
    parser.add_argument("--asset-ledger", type=Path, default=DEFAULT_ASSET_LEDGER_PATH)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--rubric", type=Path, default=DEFAULT_RUBRIC_PATH)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("freeze")
    subparsers.add_parser("check")
    dashboard_parser = subparsers.add_parser("dashboard")
    dashboard_parser.add_argument("--report", type=Path)
    dashboard_parser.add_argument("--output", type=Path)
    dashboard_parser.add_argument("--attempts", type=Path)
    dashboard_parser.add_argument("--progress", type=Path)
    dashboard_parser.add_argument("--audit", type=Path)
    dashboard_parser.add_argument("--manifest", type=Path)
    plan_terminal_parser = subparsers.add_parser("plan-terminal-settle")
    plan_terminal_parser.add_argument("--retry-failed", action="store_true")
    plan_terminal_parser.add_argument("--initial-pose-set", type=Path, default=DEFAULT_INITIAL_POSE_SET_PATH)
    plan_skill_parser = subparsers.add_parser("plan-skill-probes")
    plan_skill_parser.add_argument("--initial-pose-set", type=Path, default=DEFAULT_INITIAL_POSE_SET_PATH)
    run_skill_parser = subparsers.add_parser("run-skill-probes")
    run_skill_parser.add_argument("--plan", type=Path)
    run_skill_parser.add_argument(
        "--run-dir",
        type=Path,
        help="Dedicated checkpoint/progress directory; defaults to output-dir/runs/canonical_skill.",
    )
    run_skill_parser.add_argument("--python", dest="python_executable")
    run_skill_parser.add_argument("--task-config", default="demo_clean")
    run_skill_parser.add_argument("--seeds", default="0,1,2")
    run_skill_parser.add_argument("--relation", action="append", choices=RELATIONS)
    run_skill_parser.add_argument("--pair-key", action="append")
    run_skill_parser.add_argument("--profile", action="append")
    run_skill_parser.add_argument("--threshold", action="append")
    run_skill_parser.add_argument("--start-index", type=int, default=0)
    run_skill_parser.add_argument("--max-jobs", type=int)
    run_skill_parser.add_argument("--timeout-seconds", type=int, default=300)
    run_skill_parser.add_argument("--warmup-timeout-seconds", type=int, default=1800)
    run_skill_parser.add_argument("--torch-extensions-dir", type=Path)
    run_skill_parser.add_argument("--batch-size", type=int, default=96)
    run_skill_parser.add_argument(
        "--reuse-task-env",
        dest="reuse_task_env",
        action="store_true",
        default=True,
        help="Reuse the process-local task and CuRobo planners across attempts (default).",
    )
    run_skill_parser.add_argument(
        "--no-reuse-task-env",
        dest="reuse_task_env",
        action="store_false",
        help="Disable process-local reuse for isolation diagnostics.",
    )
    run_skill_parser.add_argument("--cuda-devices")
    run_skill_parser.add_argument("--parallel-workers", type=int)
    run_skill_parser.add_argument("--workers-per-gpu", type=int, default=1)
    run_skill_parser.add_argument("--resume", action="store_true")
    run_skill_parser.add_argument("--clear-existing-reports", action="store_true")
    render_skill_parser = subparsers.add_parser("render-skill-probe")
    render_skill_parser.add_argument("--plan", type=Path)
    render_skill_parser.add_argument("--pair-key", required=True)
    render_skill_parser.add_argument("--profile")
    render_skill_parser.add_argument("--seed", type=int, default=0)
    render_skill_parser.add_argument("--task-config", default="demo_clean")
    render_skill_parser.add_argument("--render-dir", type=Path)
    audit_skill_parser = subparsers.add_parser("audit-skill-probes")
    audit_skill_parser.add_argument("--plan", type=Path)
    audit_skill_parser.add_argument("--checkpoint-dir", type=Path)
    audit_skill_parser.add_argument("--audit-output", type=Path)
    audit_skill_parser.add_argument(
        "--relation",
        action="append",
        choices=RELATIONS,
    )
    audit_skill_parser.add_argument("--pair-key", action="append")
    audit_skill_parser.add_argument("--profile", action="append")
    audit_skill_parser.add_argument(
        "--receiver-profiles",
        type=Path,
        default=DEFAULT_RECEIVER_PROFILE_PATH,
    )
    run_terminal_parser = subparsers.add_parser("run-terminal-settle")
    run_terminal_parser.add_argument("--plan", type=Path)
    run_terminal_parser.add_argument("--python", dest="python_executable")
    run_terminal_parser.add_argument("--seeds", default="0,1,2")
    run_terminal_parser.add_argument("--relation", action="append", choices=RELATIONS)
    run_terminal_parser.add_argument("--pair-key", action="append", help="Run only the exact pair key; repeatable.")
    run_terminal_parser.add_argument("--threshold", action="append", help="Per-relation threshold override, e.g. stack=0.8")
    run_terminal_parser.add_argument("--start-index", type=int, default=0)
    run_terminal_parser.add_argument("--max-jobs", type=int)
    run_terminal_parser.add_argument("--timeout-seconds", type=int, default=180)
    run_terminal_parser.add_argument("--warmup-timeout-seconds", type=int, default=1800)
    run_terminal_parser.add_argument("--torch-extensions-dir", type=Path)
    run_terminal_parser.add_argument("--clear-existing-reports", action="store_true")
    run_terminal_parser.add_argument("--batch-size", type=int, default=64)
    run_terminal_parser.add_argument(
        "--reuse-task-env",
        dest="reuse_task_env",
        action="store_true",
        default=True,
        help="Reuse process-local imports/resources across attempts (default).",
    )
    run_terminal_parser.add_argument(
        "--no-reuse-task-env",
        dest="reuse_task_env",
        action="store_false",
        help="Disable process-local reuse for diagnostics.",
    )
    run_terminal_parser.add_argument("--cuda-devices", help="Operational compatibility only; canonical terminal settle is CPU-only.")
    run_terminal_parser.add_argument("--parallel-workers", type=int, help="Concurrent CPU batch workers. Defaults to 80% of the visible CPU quota (max 64).")
    run_terminal_parser.add_argument("--workers-per-gpu", type=int, default=1)
    run_terminal_parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse compatible attempts.jsonl entries and run only missing or infrastructure-blocked attempts.",
    )
    migrate_terminal_parser = subparsers.add_parser(
        "migrate-terminal-settle",
        help="Rebind a complete legacy checkpoint to the current policy and replace changed-relation attempts.",
    )
    migrate_terminal_parser.add_argument("--plan", type=Path)
    migrate_terminal_parser.add_argument("--source-attempts", type=Path, required=True)
    migrate_terminal_parser.add_argument(
        "--replacement-attempts",
        type=Path,
        action="append",
        default=[],
        help="Fresh current-policy attempts.jsonl; repeat for each targeted replacement run.",
    )
    migrate_terminal_parser.add_argument("--seeds", default="0,1,2")
    migrate_terminal_parser.add_argument(
        "--threshold",
        action="append",
        help="Per-relation threshold override, e.g. stack=0.8",
    )
    render_terminal_parser = subparsers.add_parser("render-terminal-settle")
    render_terminal_parser.add_argument("--plan", type=Path)
    render_terminal_parser.add_argument("--pair-key", required=True)
    render_terminal_parser.add_argument("--band-id")
    render_terminal_parser.add_argument("--seed", type=int, default=0)
    render_terminal_parser.add_argument("--clearance", type=float)
    render_terminal_parser.add_argument("--render-dir", type=Path)
    audit_terminal_parser = subparsers.add_parser("audit-terminal-settle")
    audit_terminal_parser.add_argument("--plan", type=Path)
    audit_terminal_parser.add_argument("--checkpoint-dir", type=Path)
    audit_terminal_parser.add_argument("--audit-output", type=Path)
    apply_parser = subparsers.add_parser("apply-report")
    apply_parser.add_argument("report", type=Path)
    apply_parser.add_argument("--copy-source", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "freeze":
        artifacts = freeze_pair_validation(args.asset_catalog, args.asset_ledger, args.ledger, args.output_dir, args.reports_dir, args.rubric)
        print(_status_line("freeze", artifacts))
        return 0
    if args.command == "check":
        report = check_pair_validation(args.asset_catalog, args.asset_ledger, args.ledger, args.output_dir, args.reports_dir, args.rubric)
        print(_check_status_line(report))
        return 1 if report.get("stale") else 0
    if args.command == "dashboard":
        from .pair_validation_dashboard import write_pair_validation_dashboard

        result = write_pair_validation_dashboard(
            report_path=args.report,
            reports_dir=args.reports_dir,
            data_dir=args.output_dir,
            output_path=args.output or (args.output_dir / "pair_validation_dashboard.html"),
            attempts_path=args.attempts,
            progress_path=args.progress,
            audit_path=args.audit,
            manifest_path=args.manifest,
        )
        print(
            f"dashboard: wrote {result['output_path']} from {result['report_path']} "
            f"({result['record_count']} records / {result['attempt_count']} attempts)"
        )
        return 0
    if args.command == "plan-terminal-settle":
        plan = plan_terminal_settle(
            args.asset_catalog,
            args.asset_ledger,
            args.output_dir,
            args.reports_dir,
            args.rubric,
            retry_failed=args.retry_failed,
            initial_pose_set_path=args.initial_pose_set,
        )
        print(f"plan-terminal-settle: wrote {plan.get('summary', {}).get('probe_job_count', 0)} probe jobs")
        return 0
    if args.command == "plan-skill-probes":
        plan = plan_skill_probes(
            args.asset_catalog,
            args.asset_ledger,
            args.output_dir,
            args.reports_dir,
            args.rubric,
            initial_pose_set_path=args.initial_pose_set,
        )
        print(f"plan-skill-probes: wrote {plan.get('summary', {}).get('probe_job_count', 0)} probe jobs")
        return 0
    if args.command == "run-skill-probes":
        from .pair_skill_runner import run_skill_probe_plan
        from .pair_terminal_settle_runner import parse_cuda_devices, parse_seed_list, parse_thresholds

        plan_path = args.plan
        if plan_path is None:
            plan_skill_probes(args.asset_catalog, args.asset_ledger, args.output_dir, args.reports_dir, args.rubric)
            plan_path = args.output_dir / SKILL_PLAN_FILE
        cuda_devices = parse_cuda_devices(args.cuda_devices)
        parallel_workers = _parallel_worker_count(
            args.parallel_workers,
            args.workers_per_gpu,
            cuda_devices,
        )
        result = run_skill_probe_plan(
            plan_path=plan_path,
            reports_dir=args.reports_dir,
            run_dir=args.run_dir or (args.output_dir / "runs" / "canonical_skill"),
            python_executable=args.python_executable,
            task_config=args.task_config,
            seeds=parse_seed_list(args.seeds),
            thresholds=parse_thresholds(args.threshold),
            relations=set(args.relation) if args.relation else None,
            pair_keys=set(args.pair_key) if args.pair_key else None,
            profile_ids=set(args.profile) if args.profile else None,
            start_index=args.start_index,
            max_jobs=args.max_jobs,
            timeout_seconds=args.timeout_seconds,
            warmup_timeout_seconds=args.warmup_timeout_seconds,
            torch_extensions_dir=args.torch_extensions_dir,
            batch_size=args.batch_size,
            reuse_task_env=args.reuse_task_env,
            parallel_workers=parallel_workers,
            cuda_devices=cuda_devices,
            resume=args.resume,
            clear_existing_reports=args.clear_existing_reports,
        )
        print(
            f"run-skill-probes: wrote {result['record_count']} records / "
            f"{result['attempt_count']} attempts to {result['evidence_path']}"
        )
        blocked_count = int(
            ((result.get("summary") or {}).get("outcomes") or {}).get(
                "blocked", 0
            )
        )
        if blocked_count:
            print(
                "run-skill-probes: infrastructure-blocked records remain: "
                f"{blocked_count}; resume after resolving the worker failure",
                file=sys.stderr,
            )
            return 2
        return 0
    if args.command == "render-skill-probe":
        from .pair_skill_runner import render_skill_probe_job

        plan_path = args.plan or (args.output_dir / SKILL_PLAN_FILE)
        if not plan_path.is_file():
            plan_skill_probes(args.asset_catalog, args.asset_ledger, args.output_dir, args.reports_dir, args.rubric)
        result = render_skill_probe_job(
            plan_path=plan_path,
            pair_key=args.pair_key,
            profile_id=args.profile,
            seed=args.seed,
            task_config=args.task_config,
            output_dir=args.render_dir or (args.output_dir / "renders" / "canonical_skill"),
        )
        print(
            f"render-skill-probe: wrote {len(result['frames'])} frames to "
            f"{result['manifest_path']}; success={result['result'].get('success')}"
        )
        return 0
    if args.command == "audit-skill-probes":
        from .pair_skill_runner import audit_skill_probe_run

        plan_path = args.plan or (args.output_dir / SKILL_PLAN_FILE)
        checkpoint_dir = args.checkpoint_dir or (args.output_dir / "runs" / "canonical_skill" / "checkpoints")
        output_path = args.audit_output or (args.output_dir / "runs" / "canonical_skill_failure_audit.yml")
        audit = audit_skill_probe_run(
            plan_path=plan_path,
            checkpoint_dir=checkpoint_dir,
            output_path=output_path,
            receiver_profile_path=args.receiver_profiles,
            relations=set(args.relation) if args.relation else None,
            pair_keys=set(args.pair_key) if args.pair_key else None,
            profile_ids=set(args.profile) if args.profile else None,
        )
        print(
            f"audit-skill-probes: wrote {audit['summary']['failure_count']} failures "
            f"to {output_path}; acceptance_ready={audit['summary']['acceptance_ready']}"
        )
        return 0 if audit["summary"]["acceptance_ready"] else 1
    if args.command == "run-terminal-settle":
        from .pair_terminal_settle_runner import parse_cuda_devices, parse_seed_list, parse_thresholds, run_terminal_settle_plan

        plan_path = args.plan or (args.output_dir / TERMINAL_PLAN_FILE)
        if args.plan is None and (not args.resume or not plan_path.is_file()):
            plan_terminal_settle(args.asset_catalog, args.asset_ledger, args.output_dir, args.reports_dir, args.rubric)
        cuda_devices = parse_cuda_devices(args.cuda_devices)
        parallel_workers = _terminal_parallel_worker_count(args.parallel_workers)
        result = run_terminal_settle_plan(
            asset_catalog_path=args.asset_catalog,
            plan_path=plan_path,
            reports_dir=args.reports_dir,
            run_dir=args.output_dir / "runs",
            python_executable=args.python_executable,
            seeds=parse_seed_list(args.seeds),
            thresholds=parse_thresholds(args.threshold),
            relations=set(args.relation) if args.relation else None,
            pair_keys=set(args.pair_key) if args.pair_key else None,
            start_index=args.start_index,
            max_jobs=args.max_jobs,
            timeout_seconds=args.timeout_seconds,
            warmup_timeout_seconds=args.warmup_timeout_seconds,
            torch_extensions_dir=args.torch_extensions_dir,
            clear_existing_reports=args.clear_existing_reports,
            batch_size=args.batch_size,
            reuse_task_env=args.reuse_task_env,
            parallel_workers=parallel_workers,
            cuda_devices=cuda_devices,
            resume=args.resume,
        )
        print(
            f"run-terminal-settle: wrote {result['record_count']} records / {result['attempt_count']} attempts "
            f"to {result['evidence_path']}; timing {result['timing_summary_path']}"
        )
        return 0
    if args.command == "migrate-terminal-settle":
        from .pair_terminal_settle_runner import (
            migrate_terminal_settle_evidence,
            parse_seed_list,
            parse_thresholds,
        )

        plan_path = args.plan or (args.output_dir / TERMINAL_PLAN_FILE)
        result = migrate_terminal_settle_evidence(
            asset_catalog_path=args.asset_catalog,
            plan_path=plan_path,
            source_attempts_path=args.source_attempts,
            replacement_attempts_paths=list(args.replacement_attempts),
            reports_dir=args.reports_dir,
            run_dir=args.output_dir / "runs",
            seeds=parse_seed_list(args.seeds),
            thresholds=parse_thresholds(args.threshold),
        )
        print(
            f"migrate-terminal-settle: wrote {result['record_count']} records / "
            f"{result['attempt_count']} attempts to {result['evidence_path']}; "
            f"coverage_complete={result['migration']['coverage_complete']}"
        )
        return 0
    if args.command == "render-terminal-settle":
        from .pair_terminal_settle_worker import render_terminal_settle_job

        plan_path = args.plan or (args.output_dir / TERMINAL_PLAN_FILE)
        if not plan_path.is_file():
            plan_terminal_settle(args.asset_catalog, args.asset_ledger, args.output_dir, args.reports_dir, args.rubric, retry_failed=True)
        result = render_terminal_settle_job(
            asset_catalog_path=args.asset_catalog,
            plan_path=plan_path,
            pair_key=args.pair_key,
            seed=args.seed,
            output_dir=args.render_dir or (args.output_dir / "renders" / "terminal_settle"),
            clearance_m=args.clearance,
            band_id=args.band_id,
        )
        print(
            f"render-terminal-settle: wrote {len(result['frames'])} frames to {result['manifest_path']}; "
            f"success={result['result'].get('success')} reason={result['result'].get('failure_reason')}"
        )
        return 0
    if args.command == "audit-terminal-settle":
        from .pair_terminal_settle_runner import audit_terminal_settle_run

        plan_path = args.plan or (args.output_dir / TERMINAL_PLAN_FILE)
        checkpoint_dir = args.checkpoint_dir or (args.output_dir / "runs" / "checkpoints")
        output_path = args.audit_output or (args.output_dir / "runs" / "terminal_settle_failure_audit.yml")
        audit = audit_terminal_settle_run(plan_path=plan_path, checkpoint_dir=checkpoint_dir, output_path=output_path)
        print(
            f"audit-terminal-settle: wrote {audit['summary']['failure_count']} failures "
            f"to {output_path}"
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
