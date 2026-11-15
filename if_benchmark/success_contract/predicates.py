"""Global BenchmarkV1 success predicate registry.

Every function in this module is pure: it consumes a frozen SuccessSnapshot
and canonical contract, and never imports or controls a simulator.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Sequence

from if_benchmark.relation_contract import (
    relation_anchor_pair_acceptance,
    relation_surface_gap_acceptance,
)
from if_benchmark.schema.serialization import canonical_data
from if_benchmark.articulated_terminal_contract import (
    articulated_place_in_success_metrics,
)
from if_benchmark.pair_recipe import validate_terminal_placement_recipe
from if_benchmark.terminal_placement import terminal_observation_key
from if_benchmark.orientation_adjustment import (
    ORIENTATION_ADJUSTMENT_VERSION,
    normalize_orientation_adjustment_params,
    target_quaternion_wxyz,
)

from .articulation import resolve_articulation_goal
from .contract import SuccessContract
from .pick_execution import hold_drift_policy
from .relation import RelationContract
from .snapshot import SuccessSnapshotError, normalize_success_snapshot


Predicate = Callable[
    [Mapping[str, Any], Mapping[str, Any], SuccessContract], dict[str, Any]
]


def evaluate_success(
    predicate: str,
    arguments: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    contract: SuccessContract,
) -> dict[str, Any]:
    """Evaluate one named predicate with stable tri-state diagnostics."""

    name = str(predicate)
    evaluator = _REGISTRY.get(name)
    if evaluator is None:
        return _blocked(name, contract, "unsupported_success_predicate")
    try:
        value = normalize_success_snapshot(snapshot, contract)
    except SuccessSnapshotError as exc:
        return _blocked(
            name,
            contract,
            "success_snapshot_invalid",
            metrics={"error": str(exc)},
        )
    try:
        result = evaluator(arguments, value, contract)
    except (KeyError, TypeError, ValueError) as exc:
        result = _blocked(
            name,
            contract,
            "success_observation_incomplete",
            metrics={"error": str(exc)},
        )
    result.update(
        {
            "predicate": name,
            **contract.identity,
            "success_snapshot_hash": value["success_snapshot_hash"],
            "profile_hashes": value["profile_hashes"],
            "source_hashes": value["source_hashes"],
        }
    )
    if name in {"relation_holds", "placement_completed"}:
        result.update(RelationContract.from_success_contract(contract).identity)
    return canonical_data(result)


def _active_contact(
    args: Mapping[str, Any], snapshot: Mapping[str, Any], contract: SuccessContract
) -> dict[str, Any]:
    terminal = _terminal(snapshot)
    contacts = terminal.get("active_contacts")
    if not isinstance(contacts, Sequence):
        return _blocked("active_contact", contract, "active_contacts_missing")
    left = str(args.get("entity_a") or args.get("left") or "")
    right = str(args.get("entity_b") or args.get("right") or "")
    arm = str(args.get("arm") or "")
    regions = {str(value) for value in args.get("regions", []) or []}
    epsilon = float(contract.section("active_contact")["impulse_epsilon"])
    matched: list[dict[str, Any]] = []
    for row in contacts:
        if not isinstance(row, Mapping):
            continue
        endpoints = {str(row.get("entity_a") or ""), str(row.get("entity_b") or "")}
        if left and right and endpoints != {left, right}:
            continue
        if arm and str(row.get("arm") or "") != arm:
            continue
        if regions and str(row.get("region") or "") not in regions:
            continue
        impulse = float(row.get("summed_impulse_norm", row.get("impulse_norm", 0.0)))
        if impulse > epsilon:
            matched.append(canonical_data(dict(row)))
    return _result(
        bool(matched),
        failed=[] if matched else ["active_contact_missing"],
        metrics={"matched_contacts": matched, "impulse_epsilon": epsilon},
    )


def _closure_evidence(
    args: Mapping[str, Any], snapshot: Mapping[str, Any], contract: SuccessContract
) -> dict[str, Any]:
    arm = _arm(args)
    subject = str(args["subject"])
    gripper = _gripper(snapshot, arm)
    closure = gripper.get("closure_evidence")
    if not isinstance(closure, Mapping):
        return _blocked("closure_evidence", contract, "closure_observation_missing")
    if str(closure.get("subject") or "") != subject:
        return _result(
            False,
            failed=["closure_subject_mismatch"],
            metrics={"observed_subject": closure.get("subject")},
        )
    required = ("valid", "path", "actual_width_m")
    missing = [key for key in required if key not in closure]
    if missing:
        return _blocked(
            "closure_evidence",
            contract,
            "closure_observation_incomplete",
            metrics={"missing_fields": missing},
        )
    gripper_width = float(gripper["actual_width_m"])
    closure_width = float(closure["actual_width_m"])
    width_matches = closure_width == gripper_width
    passed = bool(closure["valid"] and width_matches)
    metrics = canonical_data(dict(closure))
    metrics["actual_gripper_width_m"] = gripper_width
    metrics["closure_width_matches_actual"] = width_matches
    return _result(
        passed,
        failed=[]
        if passed
        else [
            "closure_width_not_actual"
            if not width_matches
            else str(closure.get("failure_code") or "closure_not_verified")
        ],
        metrics=metrics,
    )


def _hold_consistency(
    args: Mapping[str, Any], snapshot: Mapping[str, Any], contract: SuccessContract
) -> dict[str, Any]:
    arm = _arm(args)
    subject = str(args["subject"])
    hold = _gripper(snapshot, arm).get("hold_consistency")
    if not isinstance(hold, Mapping):
        return _blocked("hold_consistency", contract, "hold_observation_missing")
    if str(hold.get("subject") or "") != subject:
        return _result(False, failed=["hold_subject_mismatch"], metrics=dict(hold))
    policy = hold_drift_policy(
        phase=str(args.get("_hold_phase") or "carry"), contract=contract
    )
    required_fields = ["translation_drift_m"]
    if bool(policy["rotation_constraint_enabled"]):
        required_fields.append("rotation_drift_deg")
    for key in required_fields:
        if key not in hold:
            return _blocked(
                "hold_consistency",
                contract,
                "hold_observation_incomplete",
                metrics={"missing_field": key},
            )
    translation = float(hold["translation_drift_m"])
    translation_limit = float(policy["translation_limit_m"])
    translation_passed = (
        translation <= translation_limit
        if bool(policy["translation_limit_inclusive"])
        else translation < translation_limit
    )
    checks = {
        "hold_translation_stable": translation_passed,
    }
    if bool(policy["rotation_constraint_enabled"]):
        checks["hold_rotation_stable"] = float(hold["rotation_drift_deg"]) <= float(
            policy["rotation_limit_deg"]
        )
    return _checks(
        checks,
        metrics={
            **canonical_data(dict(hold)),
            "translation_limit_m": translation_limit,
            "translation_limit_inclusive": bool(policy["translation_limit_inclusive"]),
            "rotation_constraint_enabled": bool(policy["rotation_constraint_enabled"]),
            "rotation_limit_deg": policy["rotation_limit_deg"],
        },
    )


def _held_by(
    args: Mapping[str, Any], snapshot: Mapping[str, Any], contract: SuccessContract
) -> dict[str, Any]:
    arm = _arm(args)
    subject = str(args["subject"])
    gripper = _gripper(snapshot, arm)
    if "engaged" not in gripper or "actual_width_m" not in gripper:
        return _blocked("held_by", contract, "actual_gripper_state_missing")
    contact = _active_contact(
        {
            "entity_a": subject,
            "entity_b": f"gripper:{arm}",
            "arm": arm,
            "regions": ["inner_pad"],
        },
        snapshot,
        contract,
    )
    attachment = gripper.get("post_close_attachment")
    attachment_passed = bool(
        isinstance(attachment, Mapping)
        and attachment.get("passed") is True
        and attachment.get("subject") == subject
        and attachment.get("certified_close") is True
        and (attachment.get("checks") or {}).get("attachment_active") is True
        and (attachment.get("metrics") or {}).get("policy", {}).get("mode")
        == "fixed_constraint"
    )
    if contact.get("status") == "blocked":
        return _blocked(
            "held_by",
            contract,
            "held_by_contact_observation_incomplete",
            metrics={"contact": contact},
        )
    closure = _closure_evidence(args, snapshot, contract)
    physically_engaged = bool(
        gripper["engaged"]
        or attachment_passed
        or closure.get("passed") is True
    )
    if (
        not physically_engaged
        and closure.get("status") == "blocked"
        and bool(contact["passed"])
    ):
        return _blocked(
            "held_by",
            contract,
            "held_by_closure_observation_incomplete",
            metrics={"contact": contact, "closure": closure},
        )
    # A complete negative physical observation is enough to prove NOT held.
    if not physically_engaged or not (
        bool(contact["passed"]) or attachment_passed
    ):
        return _result(
            False,
            failed=[
                "actual_gripper_not_engaged"
                if not physically_engaged
                else "active_inner_contact_or_fixed_attachment"
            ],
            metrics={
                "arm": arm,
                "subject": subject,
                "actual_width_m": float(gripper["actual_width_m"]),
                "contact": contact,
                "post_close_attachment": attachment,
            },
        )
    hold = _hold_consistency(args, snapshot, contract)
    blocked = [
        row for row in (closure, hold, contact) if row.get("status") == "blocked"
    ]
    if blocked:
        return _blocked(
            "held_by",
            contract,
            "held_by_observation_incomplete",
            metrics={"blocked_components": blocked},
        )
    checks = {
        "actual_gripper_engaged": physically_engaged,
        "closure_evidence": bool(closure["passed"]),
        "active_inner_contact_or_fixed_attachment": bool(
            contact["passed"] or attachment_passed
        ),
        "hold_consistency": bool(hold["passed"]),
    }
    return _checks(
        checks,
        metrics={
            "arm": arm,
            "subject": subject,
            "actual_width_m": float(gripper["actual_width_m"]),
            "coarse_opening_engaged": bool(gripper["engaged"]),
            "closure": closure,
            "contact": contact,
            "post_close_attachment": attachment,
            "hold": hold,
        },
    )


def _pick_success(
    args: Mapping[str, Any], snapshot: Mapping[str, Any], contract: SuccessContract
) -> dict[str, Any]:
    subject = str(args["subject"])
    held = _held_by({**dict(args), "_hold_phase": "pick"}, snapshot, contract)
    if held.get("status") == "blocked":
        return held
    if held.get("status") == "failed":
        return _checks(
            {"held_by_requested_arm": False},
            metrics={"held_by": held},
        )
    entity = _entity(snapshot, subject)
    aabb = _aabb(entity)
    support_plane_z = _support_plane_z(snapshot, subject)
    clearance = float(aabb["min"][2]) - support_plane_z
    table_contact = _active_contact(
        {"entity_a": subject, "entity_b": "support:table"}, snapshot, contract
    )
    if table_contact.get("status") == "blocked":
        return table_contact
    # Pick stability is the measured actor-to-arm_ee hold relation checked by
    # _held_by. A securely held object may move together with the robot.
    threshold = float(contract.section("hold")["pick_support_clearance_m"])
    return _checks(
        {
            "held_by_requested_arm": bool(held["passed"]),
            "support_clearance": clearance >= threshold,
            "no_subject_table_contact": not bool(table_contact["passed"]),
        },
        metrics={
            "support_clearance_m": clearance,
            "minimum_support_clearance_m": threshold,
            "stability_frame": "arm_ee",
            "held_by": held,
            "subject_table_contact": table_contact,
        },
    )


def _air_handover_success(args, snapshot, contract):
    subject = str(args["subject"])
    target = str(args["target_arm"])
    source = str(args["source_arm"])
    target_hold = _held_by({"subject": subject, "arm": target}, snapshot, contract)
    source_hold = _held_by({"subject": subject, "arm": source}, snapshot, contract)
    if any(row.get("status") == "blocked" for row in (target_hold, source_hold)):
        return _blocked(
            "air_handover_success",
            contract,
            "handover_hold_observation_incomplete",
            metrics={"target_hold": target_hold, "source_hold": source_hold},
        )
    source_contact = _active_contact(
        {"entity_a": subject, "entity_b": f"gripper:{source}", "arm": source},
        snapshot,
        contract,
    )
    entity = _entity(snapshot, subject)
    clearance = float(_aabb(entity)["min"][2]) - _support_plane_z(snapshot, subject)
    stable = _entity_stable(snapshot, subject)
    if stable is None:
        return _blocked("air_handover_success", contract, "subject_stability_missing")
    return _checks(
        {
            "held_by_target_arm": bool(target_hold["passed"]),
            "not_held_by_source_arm": not bool(source_hold["passed"]),
            "no_source_gripper_contact": not bool(source_contact.get("passed")),
            "clear_of_support": clearance > 0.0,
            "subject_stable": stable,
        },
        metrics={
            "target_hold": target_hold,
            "source_hold": source_hold,
            "source_contact": source_contact,
            "support_clearance_m": clearance,
        },
    )


def _bimanual_success(args, snapshot, contract):
    subject = str(args["subject"])
    left = _held_by({"subject": subject, "arm": "left"}, snapshot, contract)
    right = _held_by({"subject": subject, "arm": "right"}, snapshot, contract)
    if any(row.get("status") == "blocked" for row in (left, right)):
        return _blocked(
            "bimanual_success",
            contract,
            "bimanual_hold_observation_incomplete",
            metrics={"left": left, "right": right},
        )
    clearance = float(_aabb(_entity(snapshot, subject))["min"][2]) - _support_plane_z(
        snapshot, subject
    )
    table = _active_contact(
        {"entity_a": subject, "entity_b": "support:table"}, snapshot, contract
    )
    if table.get("status") == "blocked":
        return table
    stable = _entity_stable(snapshot, subject)
    if stable is None:
        return _blocked("bimanual_success", contract, "subject_stability_missing")
    threshold = float(contract.section("hold")["pick_support_clearance_m"])
    return _checks(
        {
            "held_by_left": bool(left["passed"]),
            "held_by_right": bool(right["passed"]),
            "support_clearance": clearance >= threshold,
            "no_subject_table_contact": not bool(table.get("passed")),
            "subject_stable": stable,
        },
        metrics={"left": left, "right": right, "support_clearance_m": clearance},
    )


def _articulation_state_success(args, snapshot, contract):
    subject = str(args["subject"])
    state = str(args["state"])
    initial = (_initial(snapshot).get("articulations") or {}).get(subject)
    terminal = (_terminal(snapshot).get("articulations") or {}).get(subject)
    if not isinstance(initial, Mapping) or not isinstance(terminal, Mapping):
        return _blocked(
            "articulation_state_success", contract, "articulation_observation_missing"
        )
    for key in ("qpos", "lower", "upper"):
        if key not in terminal:
            return _blocked(
                "articulation_state_success",
                contract,
                "articulation_observation_incomplete",
                metrics={"missing_field": key},
            )
    if "qpos" not in initial:
        return _blocked(
            "articulation_state_success", contract, "initial_articulation_qpos_missing"
        )
    lower = float(terminal["lower"])
    upper = float(terminal["upper"])
    span = upper - lower
    if span <= 0.0:
        return _blocked("articulation_state_success", contract, "invalid_joint_limits")
    q_initial = float(initial["qpos"])
    q_terminal = float(terminal["qpos"])
    profile = terminal.get("state_profile") or args.get("state_profile") or {}
    if not isinstance(profile, Mapping):
        profile = {}
    profile_hash = terminal.get("state_profile_hash")
    if not profile_hash:
        return _blocked(
            "articulation_state_success", contract, "state_profile_hash_missing"
        )
    declared_hashes = snapshot.get("profile_hashes") or {}
    if profile_hash not in set(str(value) for value in declared_hashes.values()):
        return _blocked(
            "articulation_state_success", contract, "state_profile_hash_mismatch"
        )
    goal = resolve_articulation_goal(
        state=state,
        initial_qpos=q_initial,
        lower=lower,
        upper=upper,
        state_profile=profile,
        contract=contract,
    )
    if goal["status"] != "resolved":
        return _blocked(
            "articulation_state_success",
            contract,
            str(goal["failure_code"]),
            metrics={"articulation_goal": goal},
        )
    direction = int(goal["direction_sign"])
    interval = goal["target_interval"]
    ratio = (q_terminal - lower) / span
    progress = direction * (q_terminal - q_initial) / span
    minimum = float(goal["minimum_signed_normalized_progress"])
    progress_passed = progress >= minimum
    interval_passed = float(interval[0]) <= ratio <= float(interval[1])
    stable = _entity_stable(snapshot, subject)
    if stable is None and progress_passed and interval_passed:
        return _blocked(
            "articulation_state_success", contract, "articulation_stability_missing"
        )
    return _checks(
        {
            "signed_normalized_progress": progress_passed,
            "terminal_ratio_in_target_interval": interval_passed,
            "articulation_terminal_stable": bool(stable),
        },
        metrics={
            "initial_qpos": q_initial,
            "terminal_qpos": q_terminal,
            "lower": lower,
            "upper": upper,
            "normalized_ratio": ratio,
            "signed_normalized_progress": progress,
            "minimum_signed_normalized_progress": minimum,
            "target_interval": [float(interval[0]), float(interval[1])],
            "direction_sign": direction,
            "feasible_target_interval": goal["feasible_target_interval"],
            "resolved_target_ratio": goal["target_ratio"],
            "resolved_target_qpos": goal["target_qpos"],
            "state_profile_hash": terminal.get("state_profile_hash"),
        },
    )


def _articulated_place_in_relation(
    *,
    relation: str,
    subject: str,
    target: str,
    snapshot: Mapping[str, Any],
    contract: SuccessContract,
    stable: bool | None,
) -> dict[str, Any] | None:
    """Evaluate a recipe-bound articulated cavity terminal when declared."""

    if relation != "in" or not target:
        return None
    terminal = _terminal(snapshot)
    key = terminal_observation_key(subject, relation, [target])
    required_keys = terminal.get("required_terminal_placement_keys")
    explicitly_required = (
        isinstance(required_keys, Sequence)
        and not isinstance(required_keys, (str, bytes))
        and key in {str(value) for value in required_keys}
    )
    articulations = terminal.get("articulations")
    articulated_target = (
        isinstance(articulations, Mapping) and target in articulations
    )
    observations = terminal.get("terminal_placements")
    if not isinstance(observations, Mapping):
        if explicitly_required or articulated_target:
            return _blocked(
                "relation_holds",
                contract,
                "terminal_recipe_observation_missing",
            )
        return None
    observation = observations.get(key)
    if observation is None:
        if explicitly_required or articulated_target:
            return _blocked(
                "relation_holds",
                contract,
                "terminal_recipe_observation_missing",
            )
        return None
    if not isinstance(observation, Mapping):
        return _blocked(
            "relation_holds", contract, "terminal_recipe_observation_invalid"
        )
    binding = observation.get("placement_recipe_binding")
    if not isinstance(binding, Mapping):
        return _blocked(
            "relation_holds", contract, "terminal_recipe_binding_missing"
        )
    recipe = binding.get("placement_recipe")
    if not isinstance(recipe, Mapping) or recipe.get("constructor") != (
        "articulated_place_in_cavity_link"
    ):
        return _blocked(
            "relation_holds", contract, "terminal_recipe_binding_invalid"
        )
    profile_id = str(binding.get("receiver_state_profile_id") or "")
    profile_row = (snapshot.get("profiles") or {}).get(
        f"terminal_receiver:{profile_id}"
    )
    if not isinstance(profile_row, Mapping):
        return _blocked(
            "relation_holds", contract, "terminal_receiver_profile_missing"
        )
    expected_profile_hash = str(
        binding.get("receiver_state_profile_hash") or ""
    )
    if (
        not expected_profile_hash
        or str(profile_row.get("profile_hash") or "")
        != expected_profile_hash
        or expected_profile_hash
        not in {
            str(value)
            for value in (snapshot.get("profile_hashes") or {}).values()
        }
    ):
        return _blocked(
            "relation_holds", contract, "terminal_receiver_profile_hash_mismatch"
        )
    reviewed_profile = {
        name: value
        for name, value in profile_row.items()
        if name != "profile_hash"
    }
    try:
        validated_recipe = validate_terminal_placement_recipe(
            recipe,
            "place_in",
            expected_recipe_id=str(recipe.get("recipe_id") or ""),
            reviewed_receiver_profile=reviewed_profile,
        )
    except ValueError as exc:
        return _blocked(
            "relation_holds",
            contract,
            "terminal_recipe_binding_invalid",
            metrics={"error": str(exc)},
        )
    if (
        str(binding.get("placement_recipe_source") or "") == ""
        or str(binding.get("placement_recipe_hash") or "")
        != str(validated_recipe.get("recipe_hash") or "")
        or observation.get("cavity") != reviewed_profile.get("cavity")
    ):
        return _blocked(
            "relation_holds", contract, "terminal_recipe_lineage_mismatch"
        )

    contacts = terminal.get("active_contacts")
    if not isinstance(contacts, Sequence):
        return _blocked(
            "relation_holds", contract, "relation_contact_observation_missing"
        )
    cavity = observation.get("cavity") or {}
    link_names = sorted(
        {
            str(value)
            for field in ("allowed_support_links", "forbidden_support_links")
            for value in cavity.get(field, [])
        }
    )
    epsilon = float(contract.section("active_contact")["impulse_epsilon"])
    contact_by_link: dict[str, Any] = {}
    table_contact = False
    world_contact = False
    for link_name in link_names:
        matched: list[dict[str, Any]] = []
        for row in contacts:
            if not isinstance(row, Mapping) or {
                str(row.get("entity_a") or ""),
                str(row.get("entity_b") or ""),
            } != {subject, target}:
                continue
            receiver_link = (
                row.get("link_a")
                if str(row.get("entity_a") or "") == target
                else row.get("link_b")
            )
            impulse = float(
                row.get(
                    "summed_impulse_norm",
                    row.get("impulse_norm", 0.0),
                )
            )
            if str(receiver_link or "") == link_name and impulse > epsilon:
                matched.append(canonical_data(dict(row)))
        contact_by_link[link_name] = {
            "active": bool(matched),
            "matched_contacts": matched,
            "impulse_epsilon": epsilon,
        }
    for row in contacts:
        if not isinstance(row, Mapping):
            continue
        endpoints = {
            str(row.get("entity_a") or ""),
            str(row.get("entity_b") or ""),
        }
        impulse = float(
            row.get("summed_impulse_norm", row.get("impulse_norm", 0.0))
        )
        if impulse <= epsilon or subject not in endpoints:
            continue
        table_contact = table_contact or "support:table" in endpoints
        world_contact = world_contact or bool(
            endpoints & {"support:world", "support:ground"}
        )
    articulation_state = observation.get("articulation_state")
    if not isinstance(articulation_state, Mapping):
        return _blocked(
            "relation_holds", contract, "terminal_articulation_state_missing"
        )
    metrics = articulated_place_in_success_metrics(
        subject_state={"aabb": _aabb(_entity(snapshot, subject))},
        cavity_world_matrix=observation.get("cavity_world_matrix"),
        cavity=dict(cavity),
        contact_by_link=contact_by_link,
        subject_table_contact=table_contact,
        subject_world_contact=world_contact,
        terminal_stability={"stable": bool(stable)},
        articulation_state=dict(articulation_state),
    )
    metrics.update(
        {
            "terminal_observation_key": key,
            "placement_recipe_hash": binding["placement_recipe_hash"],
            "placement_recipe_source": binding["placement_recipe_source"],
            "receiver_state_profile_id": profile_id,
            "receiver_state_profile_hash": expected_profile_hash,
            "contact_by_link": contact_by_link,
            "subject_table_contact": table_contact,
            "subject_world_contact": world_contact,
        }
    )
    if stable is None and all(
        value
        for name, value in metrics["checks"].items()
        if name != "terminal_stable"
    ):
        return _blocked(
            "relation_holds", contract, "relation_stability_missing"
        )
    return _checks(metrics["checks"], metrics=metrics)


def _relation_holds(args, snapshot, contract):
    relation = str(args["relation"])
    subject = str(args["subject"])
    target = str(args.get("target") or args.get("receiver") or "")
    target2 = str(args.get("target2") or "")
    relation_contract = RelationContract.from_success_contract(contract)
    subject_entity = _entity(snapshot, subject)
    target_entity = _entity(snapshot, target) if target else {}
    stable = _relation_stable(
        snapshot, [value for value in (subject, target, target2) if value]
    )
    articulated = _articulated_place_in_relation(
        relation=relation,
        subject=subject,
        target=target,
        snapshot=snapshot,
        contract=contract,
        stable=stable,
    )
    if articulated is not None:
        return articulated
    if relation in {"in", "on", "stack"}:
        contact = _active_contact(
            {"entity_a": subject, "entity_b": target}, snapshot, contract
        )
        table = _active_contact(
            {"entity_a": subject, "entity_b": "support:table"}, snapshot, contract
        )
        if any(row.get("status") == "blocked" for row in (contact, table)):
            return _blocked(
                "relation_holds", contract, "relation_contact_observation_missing"
            )
        metrics = _support_relation_metrics(
            relation,
            _aabb(subject_entity),
            _aabb(target_entity),
            subject_receiver_contact=bool(contact["passed"]),
            subject_table_contact=bool(table["passed"]),
            stable=bool(stable),
            relation_contract=relation_contract,
        )
        if stable is None and all(
            value
            for key, value in metrics["checks"].items()
            if key != "relation_entities_stable"
        ):
            return _blocked("relation_holds", contract, "relation_stability_missing")
        return _checks(metrics["checks"], metrics=metrics)
    subject_position = _position(subject_entity)
    target_position = _position(target_entity)
    dx = subject_position[0] - target_position[0]
    dy = subject_position[1] - target_position[1]
    if relation in {"left_of", "right_of", "in_front_of", "behind", "next_to"}:
        band = relation_contract.surface_gap_band(relation)
        if band is None:
            return _blocked(
                "relation_holds", contract, "relation_surface_gap_band_missing"
            )
        acceptance = relation_surface_gap_acceptance(
            _aabb(subject_entity),
            _aabb(target_entity),
            relation,
            surface_gap_band_m=band,
            directional_orthogonal_gap_maximum_m=float(
                relation_contract.data[
                    "directional_orthogonal_gap_maximum_m"
                ]
            ),
        )
        geometry_checks = acceptance["checks"]
        surface_metrics = acceptance["metrics"]
        if stable is None and all(geometry_checks.values()):
            return _blocked("relation_holds", contract, "relation_stability_missing")
        return _checks(
            {**geometry_checks, "relation_entities_stable": bool(stable)},
            metrics={
                "relation": relation,
                "dx_m": dx,
                "dy_m": dy,
                **surface_metrics,
                **relation_contract.identity,
            },
        )
    if relation == "between":
        acceptance = relation_anchor_pair_acceptance(
            _aabb(subject_entity),
            _aabb(target_entity),
            _aabb(_entity(snapshot, target2)),
            relation,
            between_maximum_perpendicular_m=float(
                relation_contract.data["between_maximum_perpendicular_m"]
            ),
        )
        geometry_checks = acceptance["checks"]
        if stable is None and all(geometry_checks.values()):
            return _blocked("relation_holds", contract, "relation_stability_missing")
        return _checks(
            {**geometry_checks, "relation_entities_stable": bool(stable)},
            metrics={
                **acceptance["metrics"],
                **relation_contract.identity,
            },
        )
    if relation in {"left_of_pair", "right_of_pair"}:
        band = relation_contract.distance_band(relation)
        if band is None:
            return _blocked(
                "relation_holds", contract, "relation_distance_band_missing"
            )
        acceptance = relation_anchor_pair_acceptance(
            _aabb(subject_entity),
            _aabb(target_entity),
            _aabb(_entity(snapshot, target2)),
            relation,
            distance_band_m=band,
        )
        geometry_checks = acceptance["checks"]
        if stable is None and all(geometry_checks.values()):
            return _blocked("relation_holds", contract, "relation_stability_missing")
        return _checks(
            {**geometry_checks, "relation_entities_stable": bool(stable)},
            metrics={
                **acceptance["metrics"],
                **relation_contract.identity,
            },
        )
    return _blocked("relation_holds", contract, "unsupported_relation")


def _placement_completed(args, snapshot, contract):
    relation = _relation_holds(args, snapshot, contract)
    if relation.get("status") == "blocked":
        return relation
    if relation.get("status") == "failed":
        return _checks({"relation_holds": False}, metrics={"relation_holds": relation})
    subject = str(args["subject"])
    holds = {
        arm: _held_by({"subject": subject, "arm": arm}, snapshot, contract)
        for arm in ("left", "right")
    }
    if any(row.get("status") == "blocked" for row in holds.values()):
        return _blocked(
            "placement_completed",
            contract,
            "placement_hold_observation_incomplete",
            metrics={"relation_holds": relation, "held_by": holds},
        )
    return _checks(
        {
            "relation_holds": bool(relation["passed"]),
            "not_held_by_left": not bool(holds["left"]["passed"]),
            "not_held_by_right": not bool(holds["right"]["passed"]),
        },
        metrics={"relation_holds": relation, "held_by": holds},
    )


def _click_press_success(args, snapshot, contract):
    events = (
        _terminal(snapshot).get("events") or snapshot.get("execution_events") or {}
    ).get("contacts")
    if not isinstance(events, Sequence):
        return _blocked(
            "click_press_success", contract, "contact_event_observation_missing"
        )
    arm = _arm(args)
    subject = str(args["subject"])
    contact_id = args.get("contact_id")
    # M15 freezes the semantic event and exact authored operation anchor in
    # CheckerConfig.  Match those fields on the very same physical contact
    # event consumed by this predicate.  M4's older narrow invocation omits
    # them and remains valid because its runner separately binds use/Contact.
    exact_event_fields = {
        field: str(args.get(field) or "")
        for field in ("use", "event", "operation_anchor_id")
        if field in args
    }
    matched = [
        row
        for row in events
        if isinstance(row, Mapping)
        and str(row.get("arm")) == arm
        and str(row.get("subject")) == subject
        and (contact_id is None or row.get("contact_id") == contact_id)
        and all(
            str(row.get(field) or "") == expected
            for field, expected in exact_event_fields.items()
        )
    ]
    if not matched:
        return _result(
            False, failed=["authored_contact_event_missing"], metrics={"events": []}
        )
    row = matched[-1]
    config = contract.section("click_press")
    extent = float(row.get("target_local_extent_m", 0.0))
    tolerance = min(
        float(config["maximum_contact_tolerance_m"]),
        max(
            float(config["minimum_contact_tolerance_m"]),
            float(config["target_extent_fraction"]) * extent,
        ),
    )
    distance = float(row.get("distance_to_authored_center_m", math.inf))
    checks = {
        # Click/Press records an execution-time PhysX manifold.  A settled
        # manifold may have zero impulse on the sampled terminal step, so its
        # point count is the primary existence proof; impulse remains a
        # backward-compatible path for older snapshots.
        "active_contact": int(row.get("contact_point_count", 0)) > 0
        or float(row.get("summed_impulse_norm", 0.0))
        > float(contract.section("active_contact")["impulse_epsilon"]),
        "contact_in_authored_region": distance <= tolerance,
        "subject_not_displaced": not bool(
            row.get("subject_displacement_failed", False)
        ),
    }
    return _checks(
        checks, metrics={**canonical_data(dict(row)), "tolerance_m": tolerance}
    )


def _orientation_success(args, snapshot, contract):
    subject = str(args["subject"])
    entity = _entity(snapshot, subject)
    relative_params = None
    if "rotation_delta_deg" in args:
        if args.get("orientation_adjustment_version") != (
            ORIENTATION_ADJUSTMENT_VERSION
        ):
            return _blocked(
                "orientation_success",
                contract,
                "orientation_adjustment_version_stale",
            )
        try:
            relative_params = normalize_orientation_adjustment_params(
                {
                    key: args.get(key)
                    for key in (
                        "orientation_axis",
                        "orientation_reference",
                        "rotation_delta_deg",
                    )
                }
            )
        except ValueError as exc:
            return _blocked(
                "orientation_success",
                contract,
                "orientation_adjustment_params_invalid",
                metrics={"error": str(exc)},
            )
        initial_entity = ((_initial(snapshot).get("entities") or {}).get(subject) or {})
        initial_quaternion = (initial_entity.get("pose") or {}).get(
            "quat_wxyz"
        ) or initial_entity.get("quat_wxyz")
        if not isinstance(initial_quaternion, Sequence):
            return _blocked(
                "orientation_success",
                contract,
                "initial_orientation_missing",
            )
        target = target_quaternion_wxyz(initial_quaternion, relative_params)
    else:
        target = args.get("target_quat_wxyz") or entity.get("target_quat_wxyz")
    current = (entity.get("pose") or {}).get("quat_wxyz") or entity.get("quat_wxyz")
    if not isinstance(target, Sequence) or not isinstance(current, Sequence):
        return _blocked("orientation_success", contract, "orientation_target_missing")
    equivalent_targets = [target]
    for value in args.get("symmetry_quaternions_wxyz", []) or []:
        if isinstance(value, Sequence):
            equivalent_targets.append(value)
    errors = [_quat_error_deg(current, value) for value in equivalent_targets]
    error = min(errors)
    initial_xy = _position((_initial(snapshot).get("entities") or {}).get(subject, {}))[
        :2
    ]
    terminal_xy = _position(entity)[:2]
    drift = math.dist(initial_xy, terminal_xy)
    config = contract.section("orientation")
    not_held = []
    for arm in ("left", "right"):
        result = _held_by({"subject": subject, "arm": arm}, snapshot, contract)
        if result.get("status") == "blocked":
            return result
        not_held.append(not bool(result["passed"]))
    support = _active_contact(
        {"entity_a": subject, "entity_b": "support:table"}, snapshot, contract
    )
    if support.get("status") == "blocked":
        return support
    stable = _entity_stable(snapshot, subject)
    if stable is None:
        return _blocked("orientation_success", contract, "subject_stability_missing")
    return _checks(
        {
            "orientation_error": error <= float(config["maximum_error_deg"]),
            "subject_stable": stable,
            "subject_not_held": all(not_held),
            "legal_support_contact": bool(support.get("passed")),
            "xy_drift": drift <= float(config["maximum_xy_drift_m"]),
        },
        metrics={
            "orientation_error_deg": error,
            "equivalent_orientation_errors_deg": errors,
            "maximum_error_deg": float(config["maximum_error_deg"]),
            "xy_drift_m": drift,
            "maximum_xy_drift_m": float(config["maximum_xy_drift_m"]),
            "orientation_adjustment": relative_params,
            "target_quat_wxyz": [float(value) for value in target],
        },
    )


_REGISTRY: dict[str, Predicate] = {
    "active_contact": _active_contact,
    "closure_evidence": _closure_evidence,
    "hold_consistency": _hold_consistency,
    "held_by": _held_by,
    "pick_success": _pick_success,
    "air_handover_success": _air_handover_success,
    "bimanual_success": _bimanual_success,
    "articulation_state_success": _articulation_state_success,
    "relation_holds": _relation_holds,
    "placement_completed": _placement_completed,
    "click_press_success": _click_press_success,
    "orientation_success": _orientation_success,
}


def _terminal(snapshot):
    return snapshot["terminal"]


def _initial(snapshot):
    return snapshot["initial"]


def _entity(snapshot, key):
    entities = _terminal(snapshot).get("entities")
    if not isinstance(entities, Mapping) or key not in entities:
        raise KeyError(f"terminal entity missing: {key}")
    if not isinstance(entities[key], Mapping):
        raise TypeError(f"terminal entity malformed: {key}")
    return entities[key]


def _gripper(snapshot, arm):
    grippers = _terminal(snapshot).get("grippers")
    if not isinstance(grippers, Mapping) or arm not in grippers:
        raise KeyError(f"terminal gripper missing: {arm}")
    if not isinstance(grippers[arm], Mapping):
        raise TypeError(f"terminal gripper malformed: {arm}")
    return grippers[arm]


def _arm(args):
    arm = str(args.get("arm") or args.get("requested_arm") or "")
    if arm not in {"left", "right"}:
        raise ValueError("arm must be left or right")
    return arm


def _aabb(entity):
    value = entity.get("aabb")
    if not isinstance(value, Mapping) or not all(
        key in value for key in ("min", "max")
    ):
        raise KeyError("entity collision AABB missing")
    minimum = [float(item) for item in value["min"]]
    maximum = [float(item) for item in value["max"]]
    center = value.get("center") or [(a + b) * 0.5 for a, b in zip(minimum, maximum)]
    return {"min": minimum, "max": maximum, "center": [float(item) for item in center]}


def _position(entity):
    raw = (entity.get("pose") or {}).get("position", entity.get("position"))
    if isinstance(raw, Mapping):
        return tuple(float(raw.get(axis, 0.0)) for axis in ("x", "y", "z"))
    if isinstance(raw, Sequence) and len(raw) >= 3:
        return tuple(float(raw[index]) for index in range(3))
    raise KeyError("entity position missing")


def _support_plane_z(snapshot, subject):
    entity = _entity(snapshot, subject)
    value = entity.get("support_plane_z")
    if value is None:
        value = ((_initial(snapshot).get("entities") or {}).get(subject) or {}).get(
            "support_plane_z"
        )
    if value is None:
        raise KeyError("support plane missing")
    return float(value)


def _entity_stable(snapshot, subject):
    rows = (snapshot.get("stability") or {}).get("entities")
    if isinstance(rows, Mapping) and subject in rows:
        row = rows[subject]
        return bool(row.get("stable") if isinstance(row, Mapping) else row)
    entity = (_terminal(snapshot).get("entities") or {}).get(subject)
    if isinstance(entity, Mapping) and "stable" in entity:
        return bool(entity["stable"])
    return None


def _relation_stable(snapshot, entities):
    values = [_entity_stable(snapshot, entity) for entity in entities]
    return None if any(value is None for value in values) else all(values)


def _result(passed, *, failed, metrics):
    return {
        "status": "passed" if passed else "failed",
        "passed": bool(passed),
        "failed_predicates": list(failed),
        "metrics": canonical_data(dict(metrics)),
    }


def _checks(checks, *, metrics):
    normalized = {str(key): bool(value) for key, value in checks.items()}
    failed = [key for key, value in normalized.items() if not value]
    return _result(not failed, failed=failed, metrics={**metrics, "checks": normalized})


def _blocked(predicate, contract, code, metrics=None):
    return {
        "status": "blocked",
        "passed": None,
        "predicate": predicate,
        "failed_predicates": [code],
        "metrics": canonical_data(dict(metrics or {})),
        **contract.identity,
    }


def _support_relation_metrics(
    relation,
    subject,
    receiver,
    *,
    subject_receiver_contact,
    subject_table_contact,
    stable,
    relation_contract,
):
    subject_center = subject["center"]
    receiver_center = receiver["center"]
    receiver_min = receiver["min"]
    receiver_max = receiver["max"]
    containment = relation_contract.data["containment"]
    support = relation_contract.data["support"]
    if relation == "in":
        margin = [
            min(
                subject_center[index] - receiver_min[index],
                receiver_max[index] - subject_center[index],
            )
            for index in (0, 1)
        ]
        center_rim = subject_center[2] - receiver_max[2]
        bottom_rim = subject["min"][2] - receiver_max[2]
        depth = max(0.0, -bottom_rim)
        checks = {
            "relation_entities_stable": stable,
            "subject_receiver_contact": subject_receiver_contact,
            "subject_not_table_contact": not subject_table_contact,
            "xy_inside_receiver": min(margin)
            >= -float(containment["xy_outside_tolerance_m"]),
            "subject_meaningfully_inside_receiver": center_rim
            <= float(containment["center_rim_tolerance_m"])
            or depth
            >= float(containment["depth_split_m"])
            - float(containment["depth_tolerance_m"]),
        }
        return {
            "relation": relation,
            "checks": checks,
            "xy_inside_margin_m": margin,
            "subject_center_rim_delta_m": center_rim,
            "containment_depth_m": depth,
            **relation_contract.identity,
        }
    fraction = float(
        support["stack_receiver_footprint_fraction"]
        if relation == "stack"
        else support["on_receiver_footprint_fraction"]
    )
    minimum = float(
        support["stack_minimum_xy_limit_m"]
        if relation == "stack"
        else support["on_minimum_xy_limit_m"]
    )
    footprint = [receiver_max[index] - receiver_min[index] for index in (0, 1)]
    xy_distance = math.dist(subject_center[:2], receiver_center[:2])
    xy_limit = max(minimum, min(footprint) * fraction)
    bottom_top_gap = subject["min"][2] - receiver_max[2]
    depth = max(0.0, -bottom_top_gap)
    center_top_delta = subject_center[2] - receiver_max[2]
    center_threshold = (
        float(containment["center_rim_tolerance_m"]) if relation == "on" else -0.005
    )
    checks = {
        "relation_entities_stable": stable,
        "subject_receiver_contact": subject_receiver_contact,
        "subject_not_table_contact": not subject_table_contact,
        "xy_aligned": xy_distance <= xy_limit,
        "subject_center_above_receiver_top": center_top_delta > center_threshold,
        "subject_not_meaningfully_inside_receiver": depth
        < float(containment["depth_split_m"]) - float(containment["depth_tolerance_m"]),
    }
    return {
        "relation": relation,
        "checks": checks,
        "xy_distance_m": xy_distance,
        "xy_limit_m": xy_limit,
        "bottom_top_gap_m": bottom_top_gap,
        "containment_depth_m": depth,
        **relation_contract.identity,
    }


def support_relation_metrics(
    relation: str,
    subject_aabb: Mapping[str, Any],
    receiver_aabb: Mapping[str, Any],
    *,
    subject_receiver_contact: bool,
    subject_table_contact: bool,
    stable: bool,
    contract: SuccessContract,
) -> dict[str, Any]:
    """Canonical M5/M15 geometry-contact-stability implementation."""

    normalized_relation = {
        "place_in": "in",
        "place_on": "on",
        "stack": "stack",
    }.get(str(relation), str(relation))
    if normalized_relation not in {"in", "on", "stack"}:
        raise ValueError(f"unsupported support relation {relation!r}")
    metrics = _support_relation_metrics(
        normalized_relation,
        _aabb({"aabb": subject_aabb}),
        _aabb({"aabb": receiver_aabb}),
        subject_receiver_contact=bool(subject_receiver_contact),
        subject_table_contact=bool(subject_table_contact),
        stable=bool(stable),
        relation_contract=RelationContract.from_success_contract(contract),
    )
    checks = metrics["checks"]
    failed = [name for name, passed in checks.items() if not passed]
    return canonical_data(
        {
            **metrics,
            "success": not failed,
            "status": "passed" if not failed else "failed",
            "failed_predicates": failed,
            "failure_reason": "ok" if not failed else failed[0],
            **contract.identity,
        }
    )


def _quat_error_deg(current, target):
    left = [float(value) for value in current]
    right = [float(value) for value in target]
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        raise ValueError("orientation quaternion cannot be zero")
    dot = abs(sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm))
    return math.degrees(2.0 * math.acos(max(-1.0, min(1.0, dot))))


__all__ = ["evaluate_success", "support_relation_metrics"]
