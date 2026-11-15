"""Pure contract for the shared M4/M12/M15 Pick planning oracle.

This module intentionally imports neither the simulator nor the motion-planner
adapter.  M12 can therefore construct and validate planning-only evidence while
the simulator worker owns the actual oracle invocation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import math
from typing import Any

from if_benchmark.schema.serialization import artifact_hash, canonical_data
from if_benchmark.planner_batch import normalize_candidate_search_budget
from if_benchmark.success_contract import (
    PICK_ACQUISITION_REQUIRED_CLEARANCE_M,
    PICK_ACQUISITION_TRACKING_GUARD_M,
)

from .contact_contract import EXACT_GRASP_POLICY


PICK_PLANNING_ORACLE_VERSION = "m4_m12_m15_pick_planning_oracle_v6_operation_followthrough"
PICK_PLANNING_ADAPTER_ID = (
    "if_benchmark.grasp_validation.runtime.plan_exact_contact_candidates"
)
PICK_PLANNING_ADAPTER_VERSION = (
    "exact_mesh_all_waypoints_approach_attached_lift_orientation_world_v9"
)
PICK_PLANNING_WITNESS_VERSION = "m12_goal_variant_single_arm_witness_v4_operation_followthrough"
# Compatibility name for callers that have not yet migrated their Python API.
# Newly built artifacts use the generic Pick contract/version above.
D4_PLANNING_WITNESS_VERSION = PICK_PLANNING_WITNESS_VERSION
ARM_INDEPENDENT_PICK_VALIDATED_USES = frozenset(
    {"adjust_orientation_subject"}
)

DEFAULT_PICK_PRE_GRASP_M = 0.06
DEFAULT_PICK_USE = "pick_subject"
DEFAULT_PICK_USE_PROFILE = "single_arm_pick"

PICK_RESAMPLE_PLANNING_REJECTIONS = frozenset(
    {
        "initial_scene_unstable",
        "no_pre_grasp_path",
        "no_grasp_path",
        "unsafe_joint_margin",
        "physical_mesh_pre_grasp_collision",
        "physical_mesh_fixed_orientation_approach_collision",
        "no_attached_held_lift_path",
        "no_attached_post_lift_orientation_path",
        "no_goal_followthrough_in_budget",
    }
)
PICK_INFRASTRUCTURE_PLANNING_REJECTIONS = frozenset(
    {
        "collision_world_update_failed",
        "pre_grasp_collision_world_update_failed",
        "grasp_collision_world_update_failed",
        "pre_grasp_planner_exception",
        "grasp_planner_exception",
        "attached_held_lift_planner_exception",
        "attached_post_lift_orientation_planner_exception",
    }
)
PICK_SOURCE_PLANNING_REJECTIONS = frozenset(
    {
        "invalid_contact_matrix",
        "no_permitted_contacts",
        "support_projection_outside_inner_pad",
    }
)
PICK_BENIGN_PLANNING_REJECTIONS = frozenset(
    {
        "upward_from_below_table_approach",
        # These rows make proof-preserving Boolean short-circuiting explicit.
        # They are not independent planner failures and must not override the
        # concrete retryable reason that already made the group/layout false.
        "pruned_after_any_group_satisfied",
        "pruned_after_all_group_failed",
        "pruned_after_required_group_failed",
    }
)


class PickPlanningProviderError(RuntimeError):
    """Typed failure raised before a provider can return a valid witness."""

    disposition = "contract_failure"


class PickPlanningProviderInfrastructureError(PickPlanningProviderError):
    disposition = "infrastructure_failure"


class PickPlanningProviderSourceError(PickPlanningProviderError):
    disposition = "source_failure"


class PickPlanningProviderContractError(PickPlanningProviderError):
    disposition = "contract_failure"


# Compatibility aliases for callers that still import the former D4-only API.
D4_RESAMPLE_PLANNING_REJECTIONS = PICK_RESAMPLE_PLANNING_REJECTIONS
D4_INFRASTRUCTURE_PLANNING_REJECTIONS = (
    PICK_INFRASTRUCTURE_PLANNING_REJECTIONS
)
D4_SOURCE_PLANNING_REJECTIONS = PICK_SOURCE_PLANNING_REJECTIONS
D4_BENIGN_PLANNING_REJECTIONS = PICK_BENIGN_PLANNING_REJECTIONS
D4PlanningProviderError = PickPlanningProviderError
D4PlanningProviderInfrastructureError = PickPlanningProviderInfrastructureError
D4PlanningProviderSourceError = PickPlanningProviderSourceError
D4PlanningProviderContractError = PickPlanningProviderContractError


def _blocking_pick_result_keys(
    witness: Mapping[str, Any],
) -> set[tuple[str, str]]:
    """Return failed targets that make at least one requirement unsatisfied."""

    status_by_key = {
        (str(row.get("entity_key")), str(row.get("arm"))): str(
            row.get("status") or ""
        )
        for row in witness.get("results") or []
        if isinstance(row, Mapping)
    }
    blocking: set[tuple[str, str]] = set()
    groups = witness.get("requirement_results") or []
    if not groups:
        return {
            key for key, status in status_by_key.items() if status == "failed"
        }
    for group in groups:
        if not isinstance(group, Mapping) or group.get("passed") is True:
            continue
        entity = str(group.get("entity_key") or "")
        for arm in group.get("arms") or []:
            key = (entity, str(arm))
            if status_by_key.get(key) == "failed":
                blocking.add(key)
    return blocking


def classify_pick_planning_failure(
    witness: Mapping[str, Any],
) -> dict[str, Any]:
    """Classify failed planning evidence without hiding infra/source errors.

    Only explicit deterministic path/joint/collision rejections are eligible
    for layout-seed resampling.  An unknown reason is contract-invalid and
    therefore blocking by default.
    """

    blocking_keys = _blocking_pick_result_keys(witness)
    failed_rows = [
        row
        for row in witness.get("results") or []
        if isinstance(row, Mapping)
        and row.get("status") == "failed"
        and (
            not blocking_keys
            or (str(row.get("entity_key")), str(row.get("arm")))
            in blocking_keys
        )
    ]
    reasons = {
        str(reason)
        for row in failed_rows
        for reason, count in dict(row.get("rejection_counts") or {}).items()
        if int(count) > 0
    }
    reasons.update(
        str(row.get("failure_reason") or "")
        for row in failed_rows
        if row.get("failure_reason")
    )
    infrastructure = sorted(reasons & PICK_INFRASTRUCTURE_PLANNING_REJECTIONS)
    source = sorted(reasons & PICK_SOURCE_PLANNING_REJECTIONS)
    benign = sorted(reasons & PICK_BENIGN_PLANNING_REJECTIONS)
    unknown = sorted(
        reasons
        - PICK_RESAMPLE_PLANNING_REJECTIONS
        - PICK_INFRASTRUCTURE_PLANNING_REJECTIONS
        - PICK_SOURCE_PLANNING_REJECTIONS
        - PICK_BENIGN_PLANNING_REJECTIONS
    )
    retryable = sorted(reasons & PICK_RESAMPLE_PLANNING_REJECTIONS)
    if infrastructure:
        disposition, owner = "infrastructure_failure", "planning_oracle"
    elif source:
        disposition, owner = "source_failure", "unit_builder"
    elif unknown or not retryable:
        disposition, owner = (
            ("source_failure", "unit_builder")
            if benign and not unknown
            else ("contract_failure", "planning_oracle")
        )
    else:
        disposition, owner = "layout_unsat", "layout"
        if "no_goal_followthrough_in_budget" in reasons:
            disposition = "operation_search_exhausted"
    return {
        "disposition": disposition,
        "owner": owner,
        "reasons": sorted(reasons),
        "retryable_reasons": retryable,
        "infrastructure_reasons": infrastructure,
        "source_reasons": source,
        "benign_ignored_reasons": benign if retryable else [],
        "unknown_reasons": unknown,
    }


def classify_d4_planning_failure(
    witness: Mapping[str, Any],
) -> dict[str, Any]:
    """Compatibility wrapper for the generic Pick failure classifier."""

    return classify_pick_planning_failure(witness)


def pick_planning_identity() -> dict[str, Any]:
    """Return the current policy/oracle/adapter identity without sim imports."""

    policy = {
        "version": str(EXACT_GRASP_POLICY["version"]),
        "hash": artifact_hash(EXACT_GRASP_POLICY),
    }
    adapter = {
        "id": PICK_PLANNING_ADAPTER_ID,
        "version": PICK_PLANNING_ADAPTER_VERSION,
    }
    adapter["hash"] = artifact_hash(adapter)
    return {
        "oracle_version": PICK_PLANNING_ORACLE_VERSION,
        "planning_policy": policy,
        "adapter": adapter,
    }


def _value(source: Mapping[str, Any] | Any, field: str) -> Any:
    return source.get(field) if isinstance(source, Mapping) else getattr(source, field)


def pick_planning_request_record(
    request: Mapping[str, Any] | Any,
    *,
    resolved_contact_ids: Sequence[int] | None = None,
    resolved_maximum_attempts: int | None = None,
) -> dict[str, Any]:
    """Normalize every input that can select or plan one Pick candidate."""

    contact_ids = (
        resolved_contact_ids
        if resolved_contact_ids is not None
        else _value(request, "contact_ids")
    )
    if contact_ids is None:
        raise ValueError("Pick planning request requires resolved contact_ids")
    raw_priority_contact_ids = (
        request.get("priority_contact_ids")
        if isinstance(request, Mapping)
        else getattr(request, "priority_contact_ids", None)
    )
    priority_contact_ids = (
        []
        if raw_priority_contact_ids is None
        else [int(value) for value in raw_priority_contact_ids]
    )
    maximum_attempts = (
        resolved_maximum_attempts
        if resolved_maximum_attempts is not None
        else _value(request, "maximum_attempts")
    )
    if maximum_attempts is None:
        maximum_attempts = EXACT_GRASP_POLICY["maximum_physical_contact_attempts"]
    arm = str(_value(request, "arm"))
    if arm not in {"left", "right"}:
        raise ValueError(f"invalid Pick planning arm: {arm!r}")
    groups = _value(request, "contact_groups") or ()
    group_index = _value(request, "group_index")
    support_plane = _value(request, "support_plane_z_m")
    required_clearance = (
        request.get("required_pick_clearance_m")
        if isinstance(request, Mapping)
        else getattr(request, "required_pick_clearance_m", None)
    )
    tracking_guard = (
        request.get("lift_tracking_guard_m")
        if isinstance(request, Mapping)
        else getattr(request, "lift_tracking_guard_m", None)
    )
    raw_orientation_targets = (
        request.get("post_lift_target_quats_wxyz")
        if isinstance(request, Mapping)
        else getattr(request, "post_lift_target_quats_wxyz", ())
    ) or ()
    orientation_targets: list[list[float]] = []
    for index, raw_target in enumerate(raw_orientation_targets):
        if not isinstance(raw_target, Sequence) or len(raw_target) != 4:
            raise ValueError(
                f"post-lift orientation target {index} is not quat_wxyz"
            )
        values = [float(value) for value in raw_target]
        norm = sum(value * value for value in values) ** 0.5
        if not all(math.isfinite(value) for value in values) or norm <= 1e-12:
            raise ValueError(
                f"post-lift orientation target {index} is invalid"
            )
        values = [value / norm for value in values]
        if values[0] < 0.0:
            values = [-value for value in values]
        orientation_targets.append(values)
    record = {
        "oracle_version": PICK_PLANNING_ORACLE_VERSION,
        "subject": str(_value(request, "subject")),
        "arm": arm,
        "asset": str(_value(request, "asset")),
        "model_id": str(_value(request, "model_id")),
        "use": str(_value(request, "use")),
        "use_profile": str(_value(request, "use_profile")),
        "requested_pre_grasp_m": float(
            _value(request, "requested_pre_grasp_m")
        ),
        "table_acquisition": bool(_value(request, "table_acquisition")),
        "support_plane_z_m": (
            None if support_plane is None else float(support_plane)
        ),
        "contact_groups": [
            [int(value) for value in group] for group in groups
        ],
        "group_index": None if group_index is None else int(group_index),
        "contact_ids": [int(value) for value in contact_ids],
        "maximum_attempts": int(maximum_attempts),
        "minimum_normalized_joint_margin": float(
            _value(request, "minimum_normalized_joint_margin")
        ),
        "world_yaw_deg": float(_value(request, "world_yaw_deg")),
        "reach_zone": str(_value(request, "reach_zone")),
        "required_pick_clearance_m": float(
            required_clearance
            if required_clearance is not None
            else PICK_ACQUISITION_REQUIRED_CLEARANCE_M
        ),
        "lift_tracking_guard_m": float(
            tracking_guard
            if tracking_guard is not None
            else PICK_ACQUISITION_TRACKING_GUARD_M
        ),
        "post_lift_target_quats_wxyz": orientation_targets,
    }
    if priority_contact_ids:
        record["priority_contact_ids"] = priority_contact_ids
    if not record["subject"] or not record["asset"] or not record["model_id"]:
        raise ValueError("Pick planning request has an empty subject/asset/model_id")
    if not record["contact_ids"]:
        raise ValueError("Pick planning request has no permitted Contact")
    if len(record["contact_ids"]) != len(set(record["contact_ids"])):
        raise ValueError("Pick planning request has duplicate contact_ids")
    if any(value < 0 for value in record["contact_ids"]):
        raise ValueError("Pick planning request has a negative contact_id")
    if len(priority_contact_ids) != len(set(priority_contact_ids)):
        raise ValueError(
            "Pick planning request has duplicate priority_contact_ids"
        )
    if not set(priority_contact_ids).issubset(record["contact_ids"]):
        raise ValueError(
            "Pick planning request priority_contact_ids are outside contact_ids"
        )
    if record["maximum_attempts"] < 1:
        raise ValueError("Pick planning request maximum_attempts must be positive")
    if record["required_pick_clearance_m"] < 0.0:
        raise ValueError("Pick planning request clearance must be non-negative")
    if not 0.0 <= record["lift_tracking_guard_m"] <= PICK_ACQUISITION_TRACKING_GUARD_M:
        raise ValueError("Pick planning request lift guard is outside policy")
    if len({tuple(value) for value in orientation_targets}) != len(
        orientation_targets
    ):
        raise ValueError("Pick planning request has duplicate orientation targets")
    return canonical_data(record)


def _contact_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "policy_version",
        "contact_contract_id",
        "contact_contract_hash",
        "contact_projection_hash",
        "candidate_use",
        "use_profile",
        "contact_count",
        "contact_groups",
    }
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"Contact binding is missing {missing}")
    result = canonical_data(dict(value))
    if int(result["contact_count"]) < 1:
        raise ValueError("Contact binding contact_count must be positive")
    return result


def _normalized_requirement_groups(
    requirement_groups: Sequence[Mapping[str, Any]],
    *,
    target_keys: set[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Normalize one direct, singleton-arm requirement per GoalVariant."""

    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    covered: set[tuple[str, str]] = set()
    for raw in requirement_groups:
        goal_variant_id = str(raw.get("goal_variant_id") or "")
        entity = str(raw.get("entity_key") or "")
        mode = str(raw.get("mode") or "")
        arms = sorted({str(arm) for arm in raw.get("arms") or []})
        selected_arm = str(raw.get("selected_arm") or "")
        route = str(raw.get("route") or "")
        if (
            not goal_variant_id
            or not entity
            or mode != "all"
            or selected_arm not in {"left", "right"}
            or arms != [selected_arm]
            or route != "direct"
            or goal_variant_id in seen
        ):
            raise ValueError(
                "invalid or duplicate GoalVariant Pick requirement group: "
                f"{goal_variant_id!r}"
            )
        seen.add(goal_variant_id)
        group_keys = {(entity, arm) for arm in arms}
        if target_keys is not None and not group_keys <= target_keys:
            missing = sorted(group_keys - target_keys)
            raise ValueError(f"Pick requirement group has no targets for {missing}")
        covered.update(group_keys)
        normalized.append(
            {
                "goal_variant_id": goal_variant_id,
                "entity_key": entity,
                "selected_arm": selected_arm,
                "mode": "all",
                "arms": arms,
                "route": "direct",
            }
        )
    if not normalized:
        raise ValueError("Pick planning request has no requirement groups")
    if target_keys is not None and covered != target_keys:
        unused = sorted(target_keys - covered)
        raise ValueError(f"Pick planning request has unrequired targets {unused}")
    return sorted(
        normalized,
        key=lambda row: row["goal_variant_id"],
    )


def _normalized_evaluation_order(
    raw_order: Sequence[Mapping[str, Any]] | None,
    *,
    target_keys: set[tuple[str, str]],
) -> list[dict[str, str]]:
    """Normalize the deterministic, proof-neutral target evaluation order."""

    if raw_order is None:
        keys = sorted(target_keys)
    else:
        if any(not isinstance(row, Mapping) for row in raw_order):
            raise ValueError(
                "Pick planning evaluation order rows must be mappings"
            )
        keys = [
            (str(row.get("entity_key") or ""), str(row.get("arm") or ""))
            for row in raw_order
        ]
    if (
        len(keys) != len(target_keys)
        or len(keys) != len(set(keys))
        or set(keys) != target_keys
    ):
        raise ValueError(
            "Pick planning evaluation order must cover every target exactly once"
        )
    return [
        {"entity_key": entity, "arm": arm}
        for entity, arm in keys
    ]


def build_pick_planning_request(
    *,
    tentative_scene_hash: str,
    realized_family_id: str,
    layout_seed: int,
    source_realization_hash: str,
    source_unit_hash: str,
    config_hash: str,
    targets: Sequence[Mapping[str, Any]],
    requirement_groups: Sequence[Mapping[str, Any]],
    evaluation_order: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Freeze exact Pick queries and task-local singleton-arm requirements."""

    normalized_targets: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for raw in targets:
        entity = str(raw.get("entity_key") or "")
        arm = str(raw.get("arm") or "")
        key = (entity, arm)
        if not entity or arm not in {"left", "right"} or key in seen:
            raise ValueError(f"invalid or duplicate Pick planning target: {key!r}")
        seen.add(key)
        validated_use = str(raw["validated_use"])
        binding = _contact_binding(raw["contact_contract_binding"])
        if validated_use in {"pick_left", "pick_right"}:
            if validated_use != f"pick_{arm}":
                raise ValueError(
                    f"Pick planning target validated use mismatch: {key!r}"
                )
        elif validated_use in ARM_INDEPENDENT_PICK_VALIDATED_USES:
            if validated_use != str(binding["candidate_use"]):
                raise ValueError(
                    "Pick planning target validated use/Contact mismatch: "
                    f"{key!r}"
                )
        else:
            raise ValueError(
                f"unsupported Pick planning validated use: {validated_use!r}"
            )
        raw_planning_request = raw["planning_request"]
        unresolved_contact_ids = _value(raw_planning_request, "contact_ids")
        planning_request = pick_planning_request_record(
            raw_planning_request,
            resolved_contact_ids=(
                list(range(int(binding["contact_count"])))
                if unresolved_contact_ids is None
                else None
            ),
        )
        if planning_request["subject"] != entity or planning_request["arm"] != arm:
            raise ValueError(f"Pick planning target/request mismatch: {key!r}")
        if planning_request["use"] != str(binding["candidate_use"]):
            raise ValueError(f"Pick planning target Contact use mismatch: {key!r}")
        if planning_request["use_profile"] != str(binding["use_profile"]):
            raise ValueError(f"Pick planning target Contact profile mismatch: {key!r}")
        if planning_request["contact_groups"] != binding["contact_groups"]:
            raise ValueError(f"Pick planning target Contact groups mismatch: {key!r}")
        all_contacts = list(range(int(binding["contact_count"])))
        operation = raw.get("operation_constraint")
        if operation is not None:
            if (not isinstance(operation, Mapping) or not operation.get("constraint_hash")
                    or not operation.get("goal_variant_ids")
                    or not operation.get("allowed_contact_ids")
                    or sorted(set(operation["allowed_contact_ids"])) != planning_request["contact_ids"]
                    or not set(planning_request["contact_ids"]).issubset(all_contacts)):
                raise ValueError(f"invalid operation Contact domain: {key!r}")
        elif planning_request["contact_ids"] != all_contacts:
            raise ValueError(f"Pick planning target Contact ids mismatch: {key!r}")
        priority_contact_ids = planning_request.get(
            "priority_contact_ids", []
        )
        if not set(priority_contact_ids).issubset(
            planning_request["contact_ids"]
        ):
            raise ValueError(
                f"Pick planning target priority Contact ids mismatch: {key!r}"
            )
        normalized_targets.append(
            {
                "entity_key": entity,
                "arm": arm,
                "validated_use": validated_use,
                "entity_pose": canonical_data(dict(raw["entity_pose"])),
                "operation_anchor_contract_hash": str(
                    raw["operation_anchor_contract_hash"]
                ),
                "contact_contract_binding": binding,
                "planning_request": planning_request,
                "planning_request_hash": artifact_hash(planning_request),
                **({"operation_constraint": canonical_data(dict(operation))} if operation is not None else {}),
            }
        )
    if not normalized_targets:
        raise ValueError("Pick planning request has no targets")
    normalized_groups = _normalized_requirement_groups(
        requirement_groups,
        target_keys=seen,
    )
    normalized_evaluation_order = _normalized_evaluation_order(
        evaluation_order,
        target_keys=seen,
    )
    payload = {
        "schema_version": 1,
        "artifact_kind": "pick_planning_request",
        "witness_contract_version": PICK_PLANNING_WITNESS_VERSION,
        "oracle_version": PICK_PLANNING_ORACLE_VERSION,
        "tentative_scene_hash": str(tentative_scene_hash),
        "realized_family_id": str(realized_family_id),
        "layout_seed": int(layout_seed),
        "source_realization_hash": str(source_realization_hash),
        "source_unit_hash": str(source_unit_hash),
        "config_hash": str(config_hash),
        "requirement_groups": normalized_groups,
        "evaluation_order": normalized_evaluation_order,
        "targets": sorted(
            normalized_targets,
            key=lambda row: (row["entity_key"], row["arm"]),
        ),
    }
    return {**payload, "request_hash": artifact_hash(payload)}


def build_d4_planning_request(
    *,
    tentative_scene_hash: str,
    realized_family_id: str,
    layout_seed: int,
    source_realization_hash: str,
    source_unit_hash: str,
    config_hash: str,
    targets: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compatibility wrapper producing one singleton task per D4 arm."""

    arms_by_entity: dict[str, set[str]] = {}
    for target in targets:
        entity = str(target.get("entity_key") or "")
        arm = str(target.get("arm") or "")
        arms_by_entity.setdefault(entity, set()).add(arm)
    incomplete = sorted(
        entity
        for entity, arms in arms_by_entity.items()
        if not entity or arms != {"left", "right"}
    )
    if incomplete:
        raise ValueError(f"D4 planning request lacks both arms for {incomplete}")
    return build_pick_planning_request(
        tentative_scene_hash=tentative_scene_hash,
        realized_family_id=realized_family_id,
        layout_seed=layout_seed,
        source_realization_hash=source_realization_hash,
        source_unit_hash=source_unit_hash,
        config_hash=config_hash,
        targets=targets,
        requirement_groups=[
            {
                "goal_variant_id": f"compat_d4:{entity}:{arm}",
                "entity_key": entity,
                "selected_arm": arm,
                "mode": "all",
                "arms": [arm],
                "route": "direct",
            }
            for entity in sorted(arms_by_entity)
            for arm in ("left", "right")
        ],
    )


def _result_row(
    target: Mapping[str, Any], result: Mapping[str, Any]
) -> dict[str, Any]:
    planning = result.get("last_planning") or {}
    compiled = result.get("compiled_plan")
    passed = bool(result.get("plannable"))
    if isinstance(compiled, Mapping):
        compiled_payload = {
            key: value
            for key, value in compiled.items()
            if key != "compiled_plan_hash"
        }
        if compiled.get("compiled_plan_hash") != artifact_hash(compiled_payload):
            raise ValueError("compiled Pick plan self-hash mismatch")
        if str(compiled.get("arm") or "") != str(target["arm"]):
            raise ValueError("compiled Pick plan arm mismatch")
        if str(compiled.get("contact_contract_hash") or "") != str(
            target["contact_contract_binding"]["contact_contract_hash"]
        ):
            raise ValueError("compiled Pick plan Contact contract mismatch")
    rejection_counts = {
        str(reason): int(count)
        for reason, count in sorted(
            dict(planning.get("rejection_counts") or {}).items()
        )
    }
    candidate_count = int(planning.get("candidate_variant_count") or 0)
    reachable_count = int(planning.get("reachable_variant_count") or 0)
    row = {
        "entity_key": str(target["entity_key"]),
        "arm": str(target["arm"]),
        "validated_use": str(target["validated_use"]),
        "planning_request_hash": str(target["planning_request_hash"]),
        "oracle_result_request_hash": str(result.get("request_hash") or ""),
        "status": "passed" if passed else "failed",
        "failure_code": None if passed else str(result.get("failure_code") or ""),
        "failure_reason": None if passed else str(result.get("failure_reason") or ""),
        "selected_candidate_id": (
            str(compiled.get("candidate_id") or "")
            if isinstance(compiled, Mapping)
            else None
        ),
        "compiled_plan_hash": (
            str(compiled.get("compiled_plan_hash") or "")
            if isinstance(compiled, Mapping)
            else None
        ),
        "selected_candidate": (
            canonical_data(dict(compiled))
            if isinstance(compiled, Mapping)
            else None
        ),
        "contact_contract_binding": canonical_data(
            dict(target["contact_contract_binding"])
        ),
        "candidate_variant_count": candidate_count,
        "reachable_variant_count": reachable_count,
        "rejection_counts": rejection_counts,
    }
    priority_execution = planning.get("contact_priority_execution")
    if isinstance(priority_execution, Mapping):
        row["contact_priority_execution"] = canonical_data(
            dict(priority_execution)
        )
    if planning.get("candidate_search_budget") is not None:
        row["candidate_search_budget"] = normalize_candidate_search_budget(
            planning["candidate_search_budget"]
        )
        row["candidate_search_budget_applied"] = bool(
            planning.get("candidate_search_budget_applied", False)
        )
    return row


def _requirement_results(
    requirement_groups: Sequence[Mapping[str, Any]],
    result_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    status_by_key = {
        (str(row.get("entity_key")), str(row.get("arm"))): str(
            row.get("status") or ""
        )
        for row in result_rows
        if isinstance(row, Mapping)
    }
    results: list[dict[str, Any]] = []
    for group in requirement_groups:
        goal_variant_id = str(group["goal_variant_id"])
        entity = str(group["entity_key"])
        mode = str(group["mode"])
        arms = [str(arm) for arm in group["arms"]]
        passed_arms = [
            arm for arm in arms if status_by_key.get((entity, arm)) == "passed"
        ]
        failed_arms = [arm for arm in arms if arm not in passed_arms]
        passed = (
            not failed_arms if mode == "all" else bool(passed_arms)
        )
        results.append(
            {
                "goal_variant_id": goal_variant_id,
                "entity_key": entity,
                "selected_arm": str(group["selected_arm"]),
                "mode": mode,
                "arms": arms,
                "route": "direct",
                "passed_arms": passed_arms,
                "failed_arms": failed_arms,
                "passed": bool(passed),
            }
        )
    return canonical_data(results)


def build_pick_planning_witness(
    request: Mapping[str, Any],
    results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Strip runtime arrays and self-hash one generic Pick certificate."""

    identity = pick_planning_identity()
    rows = []
    for target in request.get("targets") or []:
        key = f"{target['entity_key']}:{target['arm']}"
        result = results.get(key)
        if not isinstance(result, Mapping):
            raise ValueError(f"Pick planning oracle omitted result {key}")
        rows.append(_result_row(target, result))
    requirement_results = _requirement_results(
        request.get("requirement_groups") or [],
        rows,
    )
    payload = {
        "schema_version": 1,
        "artifact_kind": "pick_planning_witness",
        "witness_contract_version": PICK_PLANNING_WITNESS_VERSION,
        **identity,
        "planning_only": True,
        "request_hash": str(request.get("request_hash") or ""),
        "tentative_scene_hash": str(request.get("tentative_scene_hash") or ""),
        "source_realization_hash": str(request.get("source_realization_hash") or ""),
        "source_unit_hash": str(request.get("source_unit_hash") or ""),
        "config_hash": str(request.get("config_hash") or ""),
        "results": rows,
        "requirement_results": requirement_results,
        "passed": bool(requirement_results)
        and all(row["passed"] is True for row in requirement_results),
    }
    witness = {**payload, "witness_hash": artifact_hash(payload)}
    errors = pick_planning_witness_errors(request, witness)
    if errors:
        raise ValueError("; ".join(errors))
    return canonical_data(witness)


def build_d4_planning_witness(
    request: Mapping[str, Any],
    results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Compatibility wrapper for ``build_pick_planning_witness``."""

    return build_pick_planning_witness(request, results)


def pick_planning_witness_errors(
    request: Mapping[str, Any], witness: Mapping[str, Any]
) -> list[str]:
    """Return stable structural/source errors for one planning witness."""

    errors: list[str] = []
    identity = pick_planning_identity()
    request_payload = {key: value for key, value in request.items() if key != "request_hash"}
    if request.get("request_hash") != artifact_hash(request_payload):
        errors.append("request_self_hash_mismatch")
    if request.get("schema_version") != 1:
        errors.append("request_schema_version_mismatch")
    if request.get("artifact_kind") != "pick_planning_request":
        errors.append("request_artifact_kind_mismatch")
    if request.get("witness_contract_version") != PICK_PLANNING_WITNESS_VERSION:
        errors.append("request_witness_contract_version_mismatch")
    if request.get("oracle_version") != PICK_PLANNING_ORACLE_VERSION:
        errors.append("request_oracle_version_mismatch")
    expected_header = {
        "schema_version": 1,
        "artifact_kind": "pick_planning_witness",
        "witness_contract_version": PICK_PLANNING_WITNESS_VERSION,
        **identity,
        "planning_only": True,
        "request_hash": request.get("request_hash"),
        "tentative_scene_hash": request.get("tentative_scene_hash"),
        "source_realization_hash": request.get("source_realization_hash"),
        "source_unit_hash": request.get("source_unit_hash"),
        "config_hash": request.get("config_hash"),
    }
    for field, expected in expected_header.items():
        if witness.get(field) != expected:
            errors.append(f"witness_{field}_mismatch")
    witness_payload = {key: value for key, value in witness.items() if key != "witness_hash"}
    if witness.get("witness_hash") != artifact_hash(witness_payload):
        errors.append("witness_self_hash_mismatch")

    targets = {
        (str(row.get("entity_key")), str(row.get("arm"))): row
        for row in request.get("targets") or []
    }
    try:
        evaluation_order = _normalized_evaluation_order(
            request.get("evaluation_order"),
            target_keys=set(targets),
        )
        if canonical_data(request.get("evaluation_order")) != canonical_data(
            evaluation_order
        ):
            errors.append("request_evaluation_order_invalid")
    except (TypeError, ValueError) as exc:
        errors.append(f"request_evaluation_order_invalid:{exc}")
    rows = witness.get("results") or []
    observed: set[tuple[str, str]] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            errors.append(f"result_{index}_not_mapping")
            continue
        key = (str(row.get("entity_key")), str(row.get("arm")))
        target = targets.get(key)
        if target is None or key in observed:
            errors.append(f"result_{index}_unexpected_or_duplicate")
            continue
        observed.add(key)
        if row.get("validated_use") != target.get("validated_use"):
            errors.append(f"result_{index}_validated_use_mismatch")
        if row.get("planning_request_hash") != target.get("planning_request_hash"):
            errors.append(f"result_{index}_request_hash_mismatch")
        if row.get("oracle_result_request_hash") != target.get("planning_request_hash"):
            errors.append(f"result_{index}_oracle_request_hash_mismatch")
        if row.get("contact_contract_binding") != target.get("contact_contract_binding"):
            errors.append(f"result_{index}_contact_binding_mismatch")
        status = row.get("status")
        if status not in {"passed", "failed"}:
            errors.append(f"result_{index}_invalid_status")
        if "candidate_search_budget" in row or "candidate_search_budget_applied" in row:
            try:
                checked_budget = normalize_candidate_search_budget(
                    row.get("candidate_search_budget")
                )
                if checked_budget is None or checked_budget != row.get("candidate_search_budget"):
                    raise ValueError("missing or noncanonical candidate search budget")
                if not isinstance(row.get("candidate_search_budget_applied"), bool):
                    raise ValueError("missing budget application flag")
            except (TypeError, ValueError):
                errors.append(f"result_{index}_invalid_candidate_search_budget")
        candidate_count = row.get("candidate_variant_count")
        reachable_count = row.get("reachable_variant_count")
        if (
            not isinstance(candidate_count, int)
            or isinstance(candidate_count, bool)
            or candidate_count < 0
            or not isinstance(reachable_count, int)
            or isinstance(reachable_count, bool)
            or reachable_count < 0
            or reachable_count > candidate_count
        ):
            errors.append(f"result_{index}_invalid_candidate_counts")
        rejection_counts = row.get("rejection_counts")
        if not isinstance(rejection_counts, Mapping) or any(
            not isinstance(count, int)
            or isinstance(count, bool)
            or count < 0
            for count in (rejection_counts.values() if isinstance(rejection_counts, Mapping) else ())
        ):
            errors.append(f"result_{index}_invalid_rejection_counts")
        elif isinstance(candidate_count, int) and sum(
            int(count) for count in rejection_counts.values()
        ) > candidate_count:
            errors.append(f"result_{index}_rejection_count_exceeds_candidates")
        selected = row.get("selected_candidate")
        if status == "passed":
            if not row.get("selected_candidate_id") or not row.get("compiled_plan_hash"):
                errors.append(f"result_{index}_missing_selected_candidate")
            if not isinstance(selected, Mapping):
                errors.append(f"result_{index}_missing_compiled_candidate")
            else:
                selected_payload = {
                    field: value
                    for field, value in selected.items()
                    if field != "compiled_plan_hash"
                }
                if selected.get("compiled_plan_hash") != artifact_hash(
                    selected_payload
                ):
                    errors.append(f"result_{index}_compiled_plan_hash_mismatch")
                if selected.get("compiled_plan_hash") != row.get(
                    "compiled_plan_hash"
                ):
                    errors.append(f"result_{index}_compiled_plan_join_mismatch")
                if selected.get("candidate_id") != row.get(
                    "selected_candidate_id"
                ):
                    errors.append(f"result_{index}_candidate_join_mismatch")
                if str(selected.get("arm") or "") != key[1]:
                    errors.append(f"result_{index}_candidate_arm_mismatch")
                if targets[key].get("operation_constraint") is not None:
                    from if_benchmark.reach_validation.followthrough import validate_admission_receipt
                    try:
                        validate_admission_receipt(selected.get("goal_followthrough"),
                            targets[key]["operation_constraint"], arm=key[1],
                            contact_id=selected.get("contact_id"))
                    except (ValueError, KeyError, TypeError) as exc:
                        errors.append(f"result_{index}_invalid_goal_followthrough:{exc}")
                if str(selected.get("contact_contract_hash") or "") != str(
                    (row.get("contact_contract_binding") or {}).get(
                        "contact_contract_hash"
                    )
                ):
                    errors.append(f"result_{index}_candidate_contact_mismatch")
            if reachable_count == 0:
                errors.append(f"result_{index}_no_reachable_candidate")
            if row.get("failure_code") is not None or row.get("failure_reason") is not None:
                errors.append(f"result_{index}_passed_with_failure")
        elif status == "failed":
            if (
                row.get("selected_candidate_id") is not None
                or row.get("compiled_plan_hash") is not None
                or selected is not None
            ):
                errors.append(f"result_{index}_failed_with_candidate")
            if not row.get("failure_code") or not row.get("failure_reason"):
                errors.append(f"result_{index}_missing_failure")
    if observed != set(targets):
        errors.append("witness_target_coverage_mismatch")
    try:
        groups = _normalized_requirement_groups(
            request.get("requirement_groups") or [],
            target_keys=set(targets),
        )
    except ValueError as exc:
        errors.append(f"request_requirement_groups_invalid:{exc}")
        groups = []
    expected_requirement_results = _requirement_results(groups, rows)
    if witness.get("requirement_results") != expected_requirement_results:
        errors.append("witness_requirement_results_mismatch")
    passed = bool(expected_requirement_results) and all(
        row["passed"] is True for row in expected_requirement_results
    )
    if witness.get("passed") is not passed:
        errors.append("witness_passed_mismatch")
    return errors


def d4_planning_witness_errors(
    request: Mapping[str, Any], witness: Mapping[str, Any]
) -> list[str]:
    """Compatibility wrapper for generic Pick witness validation."""

    return pick_planning_witness_errors(request, witness)


def validate_pick_planning_witness(
    request: Mapping[str, Any], witness: Mapping[str, Any]
) -> dict[str, Any]:
    """Return a canonical valid witness or raise ``ValueError``."""

    errors = pick_planning_witness_errors(request, witness)
    if errors:
        raise ValueError("; ".join(errors))
    return canonical_data(dict(witness))


def validate_d4_planning_witness(
    request: Mapping[str, Any], witness: Mapping[str, Any]
) -> dict[str, Any]:
    """Compatibility wrapper for generic Pick witness validation."""

    return validate_pick_planning_witness(request, witness)


def _goal_requires_pick_planning(goal: Mapping[str, Any]) -> bool:
    return (
        str(goal.get("action_family") or "") == "O2O_relation"
        or str(goal.get("verb") or "") in {"pick", "adjust_orientation"}
    )


def _direct_terminal_arms(
    scene: Mapping[str, Any], entity: str, relation: str
) -> list[str]:
    """Return arms admitted by every matching direct terminal reservation."""

    arm_sets: list[set[str]] = []
    for row in (scene.get("terminal_reach") or {}).values():
        if (
            not isinstance(row, Mapping)
            or str(row.get("subject") or "") != entity
            or str(row.get("relation") or "") != relation
            or row.get("proxy_passed") is not True
        ):
            continue
        arm_sets.append(
            {
                str(arm)
                for arm in row.get("reachable_arms") or []
                if str(arm) in {"left", "right"}
            }
        )
    if not arm_sets:
        return []
    available = set.intersection(*arm_sets)
    return [arm for arm in ("left", "right") if arm in available]


def preferred_non_d4_arm(
    realized_id: str,
    selection_key: str,
    layout_seed: int,
    arms: Sequence[str],
) -> str:
    """Choose a stable tie-break arm without assuming consecutive seeds."""

    available = [arm for arm in ("left", "right") if arm in set(arms)]
    if not available:
        raise ValueError(f"pick_planning_no_direct_arm:{selection_key}")
    if len(available) == 1:
        return available[0]
    stable_key = f"{realized_id}|{selection_key}|{int(layout_seed)}"
    digest = hashlib.sha256(stable_key.encode("utf-8")).digest()
    return available[int.from_bytes(digest[:8], "big") % len(available)]


def _pick_requirement_candidates(
    realized: Mapping[str, Any], scene: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Return the exact per-GoalVariant direct-arm domains for one layout."""

    from if_benchmark.layout.manipulation import selected_realized
    realized = selected_realized(realized, scene)
    operation = ((scene.get("provenance") or {}).get("metadata") or {}).get("operation_space") or {}
    if operation and (operation.get("constraint_hash") != artifact_hash(operation, omit_keys=("constraint_hash",))
                      or operation.get("scene_state_hash") != scene.get("initial_observation", {}).get("scene_state_hash")):
        raise ValueError("invalid operation-space constraint receipt")
    scene_reach = scene.get("reach") or {}
    pending: list[dict[str, Any]] = []
    for variant in sorted(
        realized.get("goal_variants") or [],
        key=lambda row: str(row.get("id") or ""),
    ):
        goal = variant.get("goal") or {}
        if not _goal_requires_pick_planning(goal):
            continue
        goal_variant_id = str(variant.get("id") or "")
        entity = str(goal.get("subject") or "")
        if not goal_variant_id or not entity:
            raise ValueError("pick_planning_goal_variant_identity_missing")
        reachable = [
            arm
            for arm in ("left", "right")
            if arm
            in {
                str(value)
                for value in (scene_reach.get(entity) or {}).get(
                    "reachable_arms", []
                )
            }
        ]
        if str(goal.get("action_family") or "") == "O2O_relation":
            relation = str(goal.get("relation") or "")
            terminal_arms = set(_direct_terminal_arms(scene, entity, relation))
            reachable = [arm for arm in reachable if arm in terminal_arms]
        branches = (operation.get("goal_options") or {}).get(goal_variant_id)
        if branches is not None:
            reachable = [arm for arm in reachable if any(row["arm"] == arm for row in branches)]
        semantic_arm = goal.get("arm")
        if semantic_arm is not None:
            selected_arm = str(semantic_arm)
            if selected_arm not in reachable:
                raise ValueError(
                    "pick_planning_semantic_arm_unreachable:"
                    f"{goal_variant_id}:{entity}:{selected_arm}"
                )
            pending.append(
                {
                    "goal_variant_id": goal_variant_id,
                    "entity_key": entity,
                    "selected_arm": selected_arm,
                    "selection": "semantic_arm",
                    "eligible_arms": reachable,
                    "semantic": True,
                }
            )
        else:
            if not reachable:
                raise ValueError(f"pick_planning_no_direct_arm:{entity}")
            pending.append(
                {
                    "goal_variant_id": goal_variant_id,
                    "entity_key": entity,
                    "selected_arm": None,
                    "selection": (
                        "direct_terminal_reach"
                        if str(goal.get("action_family") or "")
                        == "O2O_relation"
                        else "concrete_scene_reach"
                    ),
                    "eligible_arms": reachable,
                    "semantic": False,
                }
            )
    return pending


def _scene_pick_arm_overrides(
    scene: Mapping[str, Any],
) -> dict[str, str] | None:
    """Read a complete final arm assignment already frozen in SceneSpec."""

    result: dict[str, str] = {}
    found = False
    for reach in (scene.get("reach") or {}).values():
        if not isinstance(reach, Mapping):
            continue
        for marker in reach.get("pick_planning_requirements") or []:
            if not isinstance(marker, Mapping):
                raise ValueError("pick_planning_scene_marker_invalid")
            if marker.get("planning_witness_required") is not True:
                continue
            found = True
            goal_variant_id = str(marker.get("goal_variant_id") or "")
            selected_arm = str(marker.get("selected_arm") or "")
            if (
                not goal_variant_id
                or selected_arm not in {"left", "right"}
                or goal_variant_id in result
            ):
                raise ValueError("pick_planning_scene_marker_invalid")
            result[goal_variant_id] = selected_arm
    return result if found else None


def single_arm_pick_requirement_groups(
    realized: Mapping[str, Any],
    scene: Mapping[str, Any],
    *,
    arm_overrides: Mapping[str, str] | None = None,
    respect_scene_assignment: bool = True,
) -> list[dict[str, Any]]:
    """Bind every Pick GoalVariant to exactly one direct execution arm.

    D4 remains a pair of left/right GoalVariants sharing one SceneSpec.  It is
    not represented as one two-arm requirement.  An arm-unspecified variant
    deterministically freezes one arm after the concrete layout is known.
    """

    pending = _pick_requirement_candidates(realized, scene)
    candidate_ids = {str(row["goal_variant_id"]) for row in pending}
    explicit_overrides = (
        {str(key): str(value) for key, value in arm_overrides.items()}
        if arm_overrides is not None
        else _scene_pick_arm_overrides(scene)
        if respect_scene_assignment
        else None
    )
    if explicit_overrides is not None and set(explicit_overrides) != candidate_ids:
        raise ValueError("pick_planning_arm_assignment_coverage_mismatch")

    # Forced singleton choices are counted first.  Then dual-reachable tasks
    # fill the less-used side, with a seed-stable hash only as the tie-break.
    # This balances tasks after layout without steering the layout search.
    counts = {"left": 0, "right": 0}
    for row in pending:
        if row["semantic"] or len(row["eligible_arms"]) != 1:
            continue
        selected_arm = str(row["eligible_arms"][0])
        row["selected_arm"] = selected_arm
        counts[selected_arm] += 1
    for row in pending:
        if row["semantic"] or row["selected_arm"] is not None:
            continue
        eligible = list(row["eligible_arms"])
        minimum = min(counts[arm] for arm in eligible)
        least_used = [arm for arm in eligible if counts[arm] == minimum]
        selected_arm = preferred_non_d4_arm(
            str(realized.get("id") or scene.get("realized_family_id") or ""),
            str(row["goal_variant_id"]),
            int(scene.get("layout_seed") or 0),
            least_used,
        )
        row["selected_arm"] = selected_arm
        counts[selected_arm] += 1

    if explicit_overrides is not None:
        for row in pending:
            goal_variant_id = str(row["goal_variant_id"])
            selected_arm = str(explicit_overrides[goal_variant_id])
            if selected_arm not in set(row["eligible_arms"]):
                raise ValueError(
                    "pick_planning_selected_arm_ineligible:"
                    f"{goal_variant_id}:{selected_arm}"
                )
            if row["semantic"] and selected_arm != str(row["selected_arm"]):
                raise ValueError(
                    "pick_planning_semantic_arm_override:"
                    f"{goal_variant_id}:{selected_arm}"
                )
            if len(row["eligible_arms"]) == 1 and selected_arm != str(
                row["selected_arm"]
            ):
                raise ValueError(
                    "pick_planning_forced_arm_override:"
                    f"{goal_variant_id}:{selected_arm}"
                )
            if selected_arm != str(row["selected_arm"]):
                row["selection"] = "exact_planning_fallback"
            row["selected_arm"] = selected_arm

    operation = ((scene.get("provenance") or {}).get("metadata") or {}).get("operation_space")
    if operation:
        from if_benchmark.layout.manipulation import compatible_arm_choices
        compatible_arm_choices(pending, operation, fixed=explicit_overrides is not None)

    groups = [
        {
            "goal_variant_id": str(row["goal_variant_id"]),
            "entity_key": str(row["entity_key"]),
            "selected_arm": str(row["selected_arm"]),
            "mode": "all",
            "arms": [str(row["selected_arm"])],
            "route": "direct",
            "selection": str(row["selection"]),
        }
        for row in pending
    ]
    if not groups:
        return []
    # ``selection`` is scene metadata, not part of the Boolean oracle group.
    return canonical_data(sorted(groups, key=lambda row: row["goal_variant_id"]))


def next_non_d4_pick_arm_assignment(
    realized: Mapping[str, Any],
    scene: Mapping[str, Any],
    request: Mapping[str, Any],
    witness: Mapping[str, Any],
    *,
    prior_requests: Sequence[Mapping[str, Any]] = (),
) -> dict[str, str] | None:
    """Flip only concretely failed, dual-eligible non-D4 requirements.

    The first request keeps the balanced deterministic assignment.  A failed
    exact path may then move the affected arm-unspecified task to its other
    statically eligible arm.  Semantic D4 and forced singleton tasks are never
    changed.  Pruned targets are left untouched until they are actually
    evaluated by a later request.
    """

    checked = validate_pick_planning_witness(request, witness)
    if checked.get("passed") is True:
        return None
    if classify_pick_planning_failure(checked).get("disposition") not in {"layout_unsat", "operation_search_exhausted"}:
        return None
    candidates = {
        str(row["goal_variant_id"]): row
        for row in _pick_requirement_candidates(realized, scene)
    }
    current = {
        str(group["goal_variant_id"]): str(group["selected_arm"])
        for group in _normalized_requirement_groups(
            request.get("requirement_groups") or []
        )
    }
    if set(current) != set(candidates):
        raise ValueError("pick_planning_arm_assignment_coverage_mismatch")
    tried: dict[str, set[str]] = {key: set() for key in current}
    for previous in [*prior_requests, request]:
        for group in previous.get("requirement_groups") or []:
            goal_variant_id = str(group.get("goal_variant_id") or "")
            arm = str(group.get("selected_arm") or "")
            if goal_variant_id in tried and arm in {"left", "right"}:
                tried[goal_variant_id].add(arm)

    concrete_failed_keys: set[tuple[str, str]] = set()
    for row in checked.get("results") or []:
        if not isinstance(row, Mapping) or row.get("status") != "failed":
            continue
        reasons = {
            str(reason)
            for reason, count in dict(row.get("rejection_counts") or {}).items()
            if int(count) > 0
        }
        failure_reason = str(row.get("failure_reason") or "")
        if failure_reason:
            reasons.add(failure_reason)
        if (
            reasons & (PICK_RESAMPLE_PLANNING_REJECTIONS - {"initial_scene_unstable"})
            and not reasons <= PICK_BENIGN_PLANNING_REJECTIONS
        ):
            concrete_failed_keys.add(
                (str(row.get("entity_key") or ""), str(row.get("arm") or ""))
            )
    if not concrete_failed_keys:
        return None

    result = dict(current)
    changed = False
    for goal_variant_id in sorted(current):
        row = candidates[goal_variant_id]
        selected_arm = current[goal_variant_id]
        if (str(row["entity_key"]), selected_arm) not in concrete_failed_keys:
            continue
        eligible = [str(arm) for arm in row["eligible_arms"]]
        alternatives = [
            arm
            for arm in eligible
            if arm != selected_arm and arm not in tried[goal_variant_id]
        ]
        if bool(row["semantic"]) or len(eligible) != 2 or not alternatives:
            # This concrete failed target remains mandatory, so changing other
            # groups could not make the complete singleton contract pass.
            return None
        result[goal_variant_id] = alternatives[0]
        changed = True
    return canonical_data(result) if changed else None


def _semantic_pick_requirement_groups(
    realized: Mapping[str, Any],
    scene: Mapping[str, Any],
) -> list[dict[str, Any]]:
    groups = single_arm_pick_requirement_groups(realized, scene)
    return _normalized_requirement_groups(groups) if groups else []


def _validated_arm_selection_retry_rows(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        raw = []
    if not isinstance(raw, list):
        raise ValueError("pick_planning_arm_selection_retries_not_list")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in raw:
        if not isinstance(row, Mapping):
            raise ValueError("pick_planning_arm_selection_retry_not_mapping")
        request = row.get("request")
        witness = row.get("witness")
        if not isinstance(request, Mapping) or not isinstance(witness, Mapping):
            raise ValueError("pick_planning_arm_selection_retry_incomplete")
        checked = validate_pick_planning_witness(request, witness)
        request_hash = str(request.get("request_hash") or "")
        if not request_hash or request_hash in seen:
            raise ValueError("pick_planning_arm_selection_retry_duplicate")
        seen.add(request_hash)
        if checked.get("passed") is True or classify_pick_planning_failure(
            checked
        ).get("disposition") not in {"layout_unsat", "operation_search_exhausted"}:
            raise ValueError(
                "pick_planning_arm_selection_retry_not_deterministic_no_path"
            )
        result.append(
            canonical_data(
                {"request": dict(request), "witness": dict(checked)}
            )
        )
    return result


def validated_scene_pick_planning_contract(
    scene_artifact: Mapping[str, Any],
    *,
    realized_artifact: Mapping[str, Any] | None = None,
    unit_artifact: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the generic M12 Pick planning proof and exact source joins."""

    scene = scene_artifact.get("scene_spec")
    if not isinstance(scene, Mapping):
        raise ValueError("pick_scene_spec_missing")
    request = scene_artifact.get("pick_planning_request")
    witness = scene_artifact.get("pick_planning_witness")
    observed_required = scene_artifact.get("pick_planning_witness_required")
    observed_status = scene_artifact.get("pick_planning_witness_status")
    try:
        arm_selection_retries = _validated_arm_selection_retry_rows(
            scene_artifact.get("pick_planning_arm_selection_retries") or []
        )
    except (TypeError, ValueError) as exc:
        arm_selection_retries = []
        retry_validation_error = str(exc)
    else:
        retry_validation_error = None
    realized: Mapping[str, Any] | None = None
    if realized_artifact is not None:
        realized = (
            realized_artifact.get("realized_scene_family")
            or realized_artifact
        )
        expected_groups = _semantic_pick_requirement_groups(realized, scene)
    elif isinstance(request, Mapping):
        expected_groups = _normalized_requirement_groups(
            request.get("requirement_groups") or [],
        )
    else:
        expected_groups = []
    required = bool(expected_groups)
    errors: list[str] = []
    if retry_validation_error is not None:
        errors.append(
            "pick_planning_arm_selection_retries_invalid:"
            f"{retry_validation_error}"
        )
    scene_groups = canonical_data(
        sorted(
            [
                {
                    "goal_variant_id": str(marker.get("goal_variant_id") or ""),
                    "entity_key": str(entity),
                    "selected_arm": str(marker.get("selected_arm") or ""),
                    "mode": str(marker.get("requirement_mode") or ""),
                    "arms": sorted(
                        str(arm) for arm in marker.get("required_arms") or []
                    ),
                    "route": str(marker.get("route") or ""),
                }
                for entity, reach in (scene.get("reach") or {}).items()
                if isinstance(reach, Mapping)
                for marker in reach.get("pick_planning_requirements") or []
                if isinstance(marker, Mapping)
                if marker.get("planning_witness_required") is True
            ],
            key=lambda row: row["goal_variant_id"],
        )
    )
    if scene_groups != canonical_data(expected_groups):
        errors.append("pick_planning_scene_semantic_marker_mismatch")
    if observed_required is not required:
        errors.append("pick_planning_required_mismatch")
    if not required:
        if observed_status != "not_applicable":
            errors.append("pick_planning_non_pick_status_invalid")
        if request is not None:
            errors.append("pick_planning_non_pick_request_present")
        if witness is not None:
            errors.append("pick_planning_non_pick_witness_present")
        if arm_selection_retries:
            errors.append("pick_planning_non_pick_arm_selection_retry_present")
        if errors:
            raise ValueError("; ".join(errors))
        return {
            "required": False,
            "status": "not_applicable",
            "required_entities": [],
            "requirement_groups": [],
            "required_target_keys": [],
            "passed_target_keys": [],
            "selected_arm_by_goal_variant": {},
            "arm_selection_retries": [],
            "request": None,
            "witness": None,
        }

    if observed_status != "passed":
        errors.append("pick_planning_required_status_not_passed")
    if not isinstance(request, Mapping):
        errors.append("pick_planning_request_missing")
    if not isinstance(witness, Mapping):
        errors.append("pick_planning_witness_missing")
    if errors:
        raise ValueError("; ".join(errors))
    assert isinstance(request, Mapping)
    assert isinstance(witness, Mapping)

    if realized is not None and retry_validation_error is None:
        try:
            expected_retry_groups = _normalized_requirement_groups(
                single_arm_pick_requirement_groups(
                    realized,
                    scene,
                    respect_scene_assignment=False,
                )
            )
            previous_retry_requests: list[Mapping[str, Any]] = []
            for index, retry in enumerate(arm_selection_retries):
                retry_request = retry["request"]
                if canonical_data(
                    retry_request.get("requirement_groups") or []
                ) != canonical_data(expected_retry_groups):
                    errors.append(
                        "pick_planning_arm_selection_retry_"
                        f"{index}_assignment_mismatch"
                    )
                    break
                next_assignment = next_non_d4_pick_arm_assignment(
                    realized,
                    scene,
                    retry_request,
                    retry["witness"],
                    prior_requests=previous_retry_requests,
                )
                if next_assignment is None:
                    errors.append(
                        "pick_planning_arm_selection_retry_"
                        f"{index}_has_no_legal_fallback"
                    )
                    break
                expected_retry_groups = _normalized_requirement_groups(
                    single_arm_pick_requirement_groups(
                        realized,
                        scene,
                        arm_overrides=next_assignment,
                        respect_scene_assignment=False,
                    )
                )
                previous_retry_requests.append(retry_request)
            if canonical_data(
                request.get("requirement_groups") or []
            ) != canonical_data(expected_retry_groups):
                errors.append(
                    "pick_planning_final_arm_assignment_retry_trace_mismatch"
                )
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"pick_planning_arm_selection_retry_invalid:{exc}")

    expected_header = {
        "tentative_scene_hash": artifact_hash(scene),
        "realized_family_id": scene.get("realized_family_id"),
        "layout_seed": scene.get("layout_seed"),
        "source_realization_hash": scene_artifact.get(
            "source_realization_hash"
        ),
        "source_unit_hash": scene_artifact.get("source_unit_hash"),
        "config_hash": scene_artifact.get("config_hash"),
    }
    for field, expected in expected_header.items():
        if request.get(field) != expected:
            errors.append(f"pick_planning_request_{field}_mismatch")

    targets = request.get("targets") or []
    target_keys = {
        (str(row.get("entity_key")), str(row.get("arm")))
        for row in targets
        if isinstance(row, Mapping)
    }
    expected_target_keys = {
        (str(group["entity_key"]), str(arm))
        for group in expected_groups
        for arm in group["arms"]
    }
    if len(targets) != len(target_keys) or target_keys != expected_target_keys:
        errors.append("pick_planning_request_target_coverage_mismatch")
    if canonical_data(request.get("requirement_groups")) != canonical_data(
        expected_groups
    ):
        errors.append("pick_planning_request_requirement_groups_mismatch")
    try:
        rebuilt_request = build_pick_planning_request(
            tentative_scene_hash=str(request.get("tentative_scene_hash") or ""),
            realized_family_id=str(request.get("realized_family_id") or ""),
            layout_seed=int(request.get("layout_seed")),
            source_realization_hash=str(
                request.get("source_realization_hash") or ""
            ),
            source_unit_hash=str(request.get("source_unit_hash") or ""),
            config_hash=str(request.get("config_hash") or ""),
            targets=targets,
            requirement_groups=request.get("requirement_groups") or [],
            evaluation_order=request.get("evaluation_order"),
        )
        if rebuilt_request != canonical_data(dict(request)):
            errors.append("pick_planning_request_projection_mismatch")
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"pick_planning_request_invalid:{exc}")

    scene_entities = scene.get("entities") or {}
    scene_reach = scene.get("reach") or {}
    for index, target in enumerate(targets):
        if not isinstance(target, Mapping):
            continue
        entity = str(target.get("entity_key") or "")
        entity_spec = scene_entities.get(entity) or {}
        planning_request = target.get("planning_request") or {}
        if target.get("entity_pose") != entity_spec.get("pose"):
            errors.append(f"pick_planning_target_{index}_entity_pose_mismatch")
        if (
            planning_request.get("subject") != entity
            or planning_request.get("arm") != target.get("arm")
            or planning_request.get("asset") != entity_spec.get("asset")
            or str(planning_request.get("model_id"))
            != str(entity_spec.get("model_id"))
            or planning_request.get("reach_zone")
            != (scene_reach.get(entity) or {}).get("zone")
        ):
            errors.append(f"pick_planning_target_{index}_scene_binding_mismatch")
    try:
        validated_witness = validate_pick_planning_witness(request, witness)
    except ValueError as exc:
        errors.append(f"pick_planning_witness_invalid:{exc}")
        validated_witness = None
    if (
        validated_witness is not None
        and validated_witness.get("passed") is not True
    ):
        errors.append("pick_planning_witness_not_passed")

    if realized_artifact is not None and realized is not None:
        if request.get("source_realization_hash") != realized_artifact.get(
            "artifact_hash"
        ):
            errors.append("pick_planning_realization_join_mismatch")
        if request.get("realized_family_id") != realized.get("id"):
            errors.append("pick_planning_realization_id_mismatch")

    if unit_artifact is not None:
        if request.get("source_unit_hash") != unit_artifact.get("artifact_hash"):
            errors.append("pick_planning_unit_join_mismatch")
        contracts: dict[str, Mapping[str, Any]] = {}
        duplicate_contracts: set[str] = set()
        for unit in unit_artifact.get("units") or []:
            for entity, contract in (
                (unit.get("params") or {}).get("operation_anchors") or {}
            ).items():
                key = str(entity)
                if key in contracts:
                    duplicate_contracts.add(key)
                elif isinstance(contract, Mapping):
                    contracts[key] = contract
        if duplicate_contracts:
            errors.append("pick_planning_duplicate_operation_anchor_contract")
        for index, target in enumerate(targets):
            if not isinstance(target, Mapping):
                errors.append(f"pick_planning_target_{index}_not_mapping")
                continue
            entity = str(target.get("entity_key") or "")
            contract = contracts.get(entity)
            if not isinstance(contract, Mapping):
                errors.append(f"pick_planning_target_{index}_contract_missing")
                continue
            if target.get("operation_anchor_contract_hash") != contract.get(
                "contract_hash"
            ):
                errors.append(f"pick_planning_target_{index}_contract_hash_mismatch")
            use = str(target.get("validated_use") or "")
            evidence = [
                row
                for row in (
                    (contract.get("source") or {}).get(
                        "validated_contact_contracts"
                    )
                    or []
                )
                if str(row.get("use") or "") == use
                and isinstance(row.get("binding"), Mapping)
            ]
            if len(evidence) != 1:
                errors.append(
                    f"pick_planning_target_{index}_contact_evidence_missing"
                )
            elif target.get("contact_contract_binding") != evidence[0].get(
                "binding"
            ):
                errors.append(
                    f"pick_planning_target_{index}_contact_binding_mismatch"
                )

    if errors:
        raise ValueError("; ".join(errors))
    passed_arms: dict[str, list[str]] = {}
    assert validated_witness is not None
    for row in validated_witness.get("results") or []:
        if isinstance(row, Mapping) and row.get("status") == "passed":
            passed_arms.setdefault(str(row["entity_key"]), []).append(
                str(row["arm"])
            )
    passed_arms = {
        entity: sorted(set(arms))
        for entity, arms in sorted(passed_arms.items())
    }
    selected_arm_by_goal_variant = {
        str(group["goal_variant_id"]): str(group["selected_arm"])
        for group in expected_groups
    }
    required_target_keys = sorted(
        f"{entity}:{arm}" for entity, arm in expected_target_keys
    )
    passed_target_keys = sorted(
        f"{entity}:{arm}"
        for entity, arms in passed_arms.items()
        for arm in arms
    )
    return {
        "required": True,
        "status": "passed",
        "required_entities": sorted(
            {str(group["entity_key"]) for group in expected_groups}
        ),
        "requirement_groups": canonical_data(expected_groups),
        "required_target_keys": required_target_keys,
        "passed_target_keys": passed_target_keys,
        "selected_arm_by_goal_variant": canonical_data(
            selected_arm_by_goal_variant
        ),
        "arm_selection_retries": canonical_data(arm_selection_retries),
        "request": canonical_data(dict(request)),
        "witness": canonical_data(dict(validated_witness)),
    }


def validated_scene_d4_planning_contract(
    scene_artifact: Mapping[str, Any],
    *,
    realized_artifact: Mapping[str, Any] | None = None,
    unit_artifact: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compatibility wrapper for the generic scene Pick planning contract."""

    return validated_scene_pick_planning_contract(
        scene_artifact,
        realized_artifact=realized_artifact,
        unit_artifact=unit_artifact,
    )
