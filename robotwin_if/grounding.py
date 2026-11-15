"""Score the Mixed 50 Grounding protocol from the simulator's physical trace."""

import gzip
import json
import math

_RELATION_CHECKS = {
    "between": {"between_projection", "between_perpendicular"},
    "left_of_pair": {"relative_to_anchor_pair", "anchor_pair_distance_band"},
    "right_of_pair": {"relative_to_anchor_pair", "anchor_pair_distance_band"},
    **{r: {"relation_surface_gap_band", "relation_directional_orthogonal_gap"}
       for r in ("left_of", "right_of", "in_front_of", "behind")},
}


def _acquired(grippers, subject):
    for gripper in grippers.values():
        closure = gripper.get("closure_evidence", {})
        if (closure.get("valid") is True and closure.get("subject") == subject
                and closure.get("path") and "actual_width_m" in closure
                and float(closure["actual_width_m"]) == float(gripper["actual_width_m"])):
            return True
    return False


def _contact(contacts, subject):
    for contact in contacts:
        endpoints = {contact["entity_a"], contact["entity_b"]}
        arm = contact.get("arm")
        if (arm in ("left", "right") and endpoints == {f"gripper:{arm}", subject}
                and float(contact["summed_impulse_norm"]) > 1e-6):
            return True
    return False


def _in_destination(row, subject, relation, support):
    result = row["result"]["metrics"]["relation_holds"]
    checks = result["metrics"]["checks"]
    expected = _RELATION_CHECKS[relation]
    if (result["status"] not in ("passed", "failed")
            or set(checks) != expected | {"relation_entities_stable"}
            or any(type(value) is not bool for value in checks.values())):
        raise ValueError("Unexpected relation checker output")
    entity = row["snapshot"]["terminal"]["entities"][subject]
    clearance = float(entity["aabb"]["min"][2]) - support
    if not math.isfinite(clearance) or float(entity["support_plane_z"]) != support:
        raise ValueError("Invalid subject support plane")
    return abs(clearance) <= .02 and all(checks[key] for key in expected)


def score_trace(path, node, result, *, check_stride=20):
    goal = node["task"]["goal"]
    subject = goal["subject"]
    pick = node["pool"] == "pick"
    first_hit = acquisition_step = window_start = None
    step = last_check = 0
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        initial = json.loads(next(stream))
        dt = float(initial["dt"])
        if (initial["kind"] != "initial" or not math.isfinite(dt) or dt <= 0
                or initial["physical_state_hash"] != result["initial_state_hash"]
                or initial["observation_hash"] != result["initial_observation_hash"]):
            raise ValueError("Invalid initial physical evidence")
        support = float(initial["success_initial"]["entities"][subject]["support_plane_z"])
        for line in stream:
            row = json.loads(line)
            if row["kind"] == "physics":
                if row["step"] != step + 1 or (step and step % check_stride == 0 and last_check != step):
                    raise ValueError("Missing or out-of-order physical evidence")
                step += 1
                if pick and first_hit is None and _contact(row["contacts"], subject):
                    first_hit = step
                if not pick and acquisition_step is None and _acquired(row["actual_grippers"], subject):
                    acquisition_step = step
                continue
            if (row["kind"] != "success_check" or row["step"] != step or step <= last_check
                    or (step % check_stride and step != result["physics_steps"])):
                raise ValueError("Invalid checker cadence")
            last_check = step
            if not pick:
                qualifies = acquisition_step is not None and _in_destination(row, subject, goal["relation"], support)
                if qualifies:
                    if window_start is None:
                        window_start = step
                    if first_hit is None and (step - window_start) * dt >= .5:
                        first_hit = step
                else:
                    window_start = None
    if step != result["physics_steps"] or last_check != step or not step:
        raise ValueError("Incomplete physical trace")
    return {"passed": first_hit is not None, "first_step": first_hit, "acquisition_step": acquisition_step}
