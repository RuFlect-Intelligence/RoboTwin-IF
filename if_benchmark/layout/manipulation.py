"""Production CPU manipulation constraints shared by M11, M12 and route search.

Local envelopes are proposal/filter evidence, never an IK or physical success
certificate. Candidate branches preserve exact arm/Contact identity throughout
acquisition and placement. All semantic and full-motion gates remain live.
"""
from __future__ import annotations

from copy import deepcopy
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from itertools import product
import math

import numpy as np
from transforms3d.euler import euler2mat
from transforms3d.quaternions import quat2mat

from if_benchmark.schema import artifact_hash
from .operation_space import TemplateCache, convex_piece, overlap, translated

from if_benchmark.manipulation_policy import POLICY

_TEMPLATES = OrderedDict()


class OperationSpaceSearchExhausted(RuntimeError):
    """Finite proposal budget; never a geometric impossibility certificate."""


def local_poses(unit):
    p = unit["params"]
    if unit["type"] == "ObjectUnit":
        return {p["object_key"]: p["local_pose"]}
    return p["internal_layout" if unit["type"] == "LandmarkRelationUnit" else "receiver_layout"]


def resources(units):
    result = {}
    for unit in units:
        for key, row in unit["params"].get("operation_spaces", {}).items():
            if key in result:
                raise ValueError("duplicate operation-space subject")
            if row["content_hash"] != artifact_hash(row, omit_keys=("content_hash",)):
                raise ValueError("operation-space resource hash mismatch")
            if row.get("policy") != POLICY:
                raise ValueError("operation-space policy is stale; rebuild M11")
            result[key] = row
    return result


@lru_cache(maxsize=256)
def _source_digest(path, size, mtime_ns):
    from .operation_catalog import digest
    return digest(path)


def verify_operation_sources(units):
    for unit in units:
        for path, expected in unit["params"].get("operation_sources", {}).items():
            stat = Path(path).stat()
            if _source_digest(path, stat.st_size, stat.st_mtime_ns) != expected:
                raise ValueError(f"operation-space source is stale; rebuild M11: {path}")


def materialize_operation_spaces(realized, units, *, catalog):
    """Freeze cooked geometry once in M11, before the layout hot loop."""
    goals = [v["goal"] for v in realized.get("goal_variants", [])
             if v.get("goal", {}).get("action_family") == "O2O_relation"]
    if not goals:
        return units
    bindings = {v["entity_key"]: v for v in realized["role_bindings"].values()}
    robot = catalog.robot()
    for unit in units:
        from .operation_catalog import digest
        root = Path(__file__).resolve().parents[1]
        implementation = {str(root / p): digest(root / p) for p in (
            "manipulation_policy.py", "layout/manipulation.py", "layout/operation_catalog.py",
            "layout/operation_space.py", "layout/operation_space_live.py",
            "reach_validation/candidates.py", "reach_validation/contract.py")}
        unit["params"].setdefault("operation_sources", {}).update(
            {**implementation, **robot["identity"]["files"]})
        for entity, local in local_poses(unit).items():
            wanted = [g for g in goals if g["subject"] == entity]
            cooked = catalog.obstacle(bindings[entity], local)
            unit["params"]["operation_sources"].update(cooked["identity"]["files"])
            unit["params"].setdefault("operation_obstacles", {})[entity] = {
                "pieces": cooked["geometry"]["pieces"], "geometry_key": cooked["key"],
                "link_matrices": cooked["geometry"].get("link_matrices", {})}
            if not wanted:
                continue
            cooked = catalog.subject(bindings[entity], local)
            row = {"policy": POLICY, "geometry": cooked["geometry"],
                   "geometry_identity": cooked["identity"], "geometry_key": cooked["key"],
                   "robots": robot["geometry"], "robot_geometry_key": robot["key"],
                   "support_plane_z_m": -float(local["translation_xyz"][2]),
                   "subject_identity": {"asset": bindings[entity]["asset"],
                                        "model_id": bindings[entity]["model_id"],
                                        "canonical_state_hash": local["canonical_state_hash"]}}
            row["content_hash"] = artifact_hash(row)
            unit["params"].setdefault("operation_spaces", {})[entity] = row
    return units


class OperationTemplates:
    """Per-family memory, not rehashed/copied per XY placement."""
    def __init__(self, units, *, cache_dir=None):
        self.disk = TemplateCache(cache_dir or Path(__file__).resolve().parents[2] / ".cache/if_benchmark/operation_templates")
        self.resources = resources(units)
        self.cache = {}

    def get(self, subject, arm, cid, relation, *, yaw_deg=0., retreat=.02):
        from if_benchmark.reach_validation.contract import ROUTE_EXECUTION_CONTRACT, route_lift_delta_m
        # Relation names share identical geometry when their motion offsets do.
        # Keep stack's different lift and every arm/Contact/yaw distinct.
        lift = route_lift_delta_m(relation)
        key = subject, arm, int(cid), lift, float(yaw_deg), float(retreat)
        if key in self.cache:
            return self.cache[key]
        row = self.resources[subject]
        shared_key = (row["content_hash"], *key[1:])
        if shared_key in _TEMPLATES:
            template = _TEMPLATES.pop(shared_key)
            _TEMPLATES[shared_key] = template
            self.cache[key] = template
            return template
        g = row["geometry"]
        rot = euler2mat(0, 0, math.radians(yaw_deg))
        contact = np.asarray(g["contacts"][str(cid)], float).copy()
        contact[:3, :3] = rot @ contact[:3, :3]
        contact[:3, 3] = rot @ contact[:3, 3]
        spec = {"arm": arm, "contact_id": int(cid), "motion_model": "fixed_orientation_translation",
                "subject_identity": row["subject_identity"], "geometry_key": row["geometry_key"],
                "robot_geometry_key": row["robot_geometry_key"],
                "subject_pieces": [(np.asarray(p) @ rot.T).tolist() for p in g["pieces"]],
                "subject_shape_sources": g["sources"], "robot_pieces": row["robots"][arm]["pieces"],
                "contact_matrix": contact.tolist(), "support_plane_z_m": row["support_plane_z_m"],
                "approach_retreat_m": retreat, "lift_m": lift,
                "descent_m": ROUTE_EXECUTION_CONTRACT["phase_offsets_m"]["approach"],
                "release_retreat_m": ROUTE_EXECUTION_CONTRACT["release_retreat"]["delta_m"],
                "robot_padding_m": POLICY["robot_padding_m"],
                "held_world_box_padding_m": POLICY["held_world_box_padding_m"]}
        try:
            template, _ = self.disk.get(spec)
        except ValueError as exc:
            if "Contact outside inner pad" not in str(exc):
                raise
            template = None
        self.cache[key] = template
        _TEMPLATES[shared_key] = template
        if len(_TEMPLATES) > 512:
            _TEMPLATES.popitem(last=False)
        return template


def _box(lo, hi, label):
    return convex_piece(list(product(*zip(lo, hi))), label=label)


def scene_boxes(scene, units=()):
    from if_benchmark.reach_validation.contract import _entity_world_aabb
    cooked = {k: v for u in units for k, v in u["params"].get("operation_obstacles", {}).items()}
    result = {}
    for key, entity in scene["entities"].items():
        if key in cooked:
            for i, p in enumerate(cooked[key]["pieces"]):
                name = f"{key}:{i}"
                result[name] = convex_piece(np.asarray(p) + entity["pose"]["position"], label=name)
                result[name]["entity"] = key
        else:
            a = _entity_world_aabb(entity)
            result[key] = {**_box(a["min"], a["max"], key), "entity": key}
    return result


def phase_clear(template, phase, boxes, offset, *, excluded=()):
    if template is None:
        return False
    # AABB rejection is only broad-phase; preserve convex piece gaps and height.
    for a in template["phases"][phase]:
        amin, amax = np.asarray(a["min"]) + offset, np.asarray(a["max"]) + offset
        moved = None
        for key, b in boxes.items():
            if b.get("entity", key) in excluded or np.any(amax < b["min"]) or np.any(np.asarray(b["max"]) < amin):
                continue
            if moved is None:
                moved = translated(a, offset)
            if overlap(moved, b):
                return False
    return True


def initial_clear(template, boxes, subject, position):
    return all(phase_clear(template, phase, boxes, np.asarray(position), excluded=(subject,))
               for phase in ("approach_open", "close", "lift_held"))


def placement_clear(template, boxes, subject, terminal, relation, receivers):
    if template is None:
        return False
    position = np.asarray(terminal["actor_pose"][:3])
    vertical = relation in {"in", "on", "stack"}
    if vertical:
        # At approach the intended receiver still blocks the complete robot and
        # object. Only descent/closed release may use intentional contact.
        approach = position + [0., 0., float(template["spec"]["descent_m"])]
        if not phase_clear(template, "close", boxes, approach, excluded=(subject,)):
            return False
    for phase in ("descent_held", "release_open", "retreat_open"):
        excluded = (subject, *receivers) if vertical and phase != "retreat_open" else (subject,)
        test = template
        if phase == "retreat_open":
            # The released object is no longer carried. Do not classify its
            # intended support contact as robot retreat collision.
            test = {**template, "phases": {phase: [p for p in template["phases"][phase]
                                                   if not p["label"].startswith("subject:")]}}
        if not phase_clear(test, phase, boxes, position, excluded=excluded):
            return False
    return True


def goal_groups(realized):
    """Fixed non-D2b context; never combine different subjects or semantic arms."""
    groups = {}
    for v in realized.get("goal_variants", []):
        g = v.get("goal", {})
        if g.get("action_family") != "O2O_relation":
            continue
        fixed = {k: value for k, value in g.items() if k not in {"verb", "relation"}}
        groups.setdefault(artifact_hash(fixed), []).append(str(v["id"]))
    from itertools import combinations
    from if_benchmark.minimal_pairs.core import edge_axis
    variants = {str(v["id"]): v for v in realized.get("goal_variants", [])}
    declared = {frozenset((e["from"], e["to"])) for e in realized.get("minimal_pair_edges", [])
                if e.get("axis") == "D2b"}
    valid = []
    for ids in groups.values():
        if len(ids) < 2:
            continue
        for a, b in combinations(ids, 2):
            left, right = variants[a], variants[b]
            edge = edge_axis(left["goal"], right["goal"],
                             left_intended_axes=left.get("intended_axes"),
                             right_intended_axes=right.get("intended_axes"))
            if (frozenset((a, b)) not in declared or not edge.is_single_axis
                    or edge.actual_axes != ("D2b",)):
                break
        else:
            valid.append(sorted(ids))
    return valid


TASK_DOMAIN_POLICY = "d2b_complete_subsets_balanced_contexts_v2"


def goal_domains(realized):
    """Full domain first, then bounded, balanced reductions of every context.

    A four-way directional context may retain four, three, or two goals.
    Try synchronized reductions before the mixed frontier: a lexicographic
    product can otherwise spend the entire budget varying only the last context.
    """
    from heapq import heappop, heappush
    from itertools import combinations
    all_ids = sorted(v["id"] for v in realized.get("goal_variants", []))
    groups = sorted(goal_groups(realized))
    if not groups:
        return [all_ids]
    fixed = set(all_ids) - set().union(*(set(g) for g in groups))
    choices = [[list(p) for size in range(len(g), 1, -1)
                for p in combinations(g, size)] for g in groups]
    result, emitted = [], set()

    def append(indices):
        ids = tuple(sorted(fixed.union(*(set(rows[i]) for rows, i in zip(choices, indices)))))
        if ids not in emitted:
            emitted.add(ids)
            result.append(list(ids))
        return len(result) >= POLICY["maximum_goal_domains"]

    for rank in range(max(map(len, choices))):
        if append(tuple(min(rank, len(rows) - 1) for rows in choices)):
            return result
    origin = (0,) * len(choices)
    frontier, visited = [(0, 0, origin)], {origin}
    while frontier:
        _, _, indices = heappop(frontier)
        if append(indices):
            break
        for axis, rows in enumerate(choices):
            if indices[axis] + 1 >= len(rows):
                continue
            next_indices = (*indices[:axis], indices[axis] + 1, *indices[axis + 1:])
            if next_indices not in visited:
                visited.add(next_indices)
                heappush(frontier, (max(next_indices), sum(next_indices), next_indices))
    return result


def selected_layout_units(realized, units):
    """Transient M12 reservation view; preserve all physical M11 geometry.

    Excluded goals need neither workspace reservations nor not-success checks.
    The frozen Unit artifact, actor poses, collision resources and source hash
    remain unchanged. Endpoints are part of the key, not just the relation name.
    """
    from if_benchmark.units.core import _envelope
    active = {(str(g["subject"]), str(g["relation"]),
               tuple(str(g[k]) for k in ("target", "target2") if g.get(k)))
              for v in realized.get("goal_variants", [])
              for g in [v.get("goal") or {}]
              if g.get("action_family") == "O2O_relation"}
    result = []
    for unit in units:
        params = unit.get("params") or {}
        regions = params.get("reserved_regions")
        if unit.get("type") != "ReceiverReservationUnit" or regions is None:
            result.append(unit)
            continue
        selected = [r for r in regions if
                    (str(r["subject_key"]), str(r["relation"]),
                     tuple(str(k) for k in (r.get("receiver_keys") or params.get("receiver_keys") or []))) in active]
        if len(selected) == len(regions):
            result.append(unit)
            continue
        footprint = _envelope([*params["receiver_footprints"].values(),
                               *(r["footprint"] for r in selected)],
                              source="selected_receiver_and_terminal_region_envelope")
        result.append({**unit, "params": {**params, "reserved_regions": selected,
                      "relations": sorted({r["relation"] for r in selected}), "footprint": footprint}})
    return result


def domain_receipt(realized, selected):
    original = sorted(v["id"] for v in realized.get("goal_variants", []))
    selected = sorted(selected)
    if selected not in goal_domains(realized):
        raise ValueError("invalid D2b counterfactual task domain")
    payload = {"policy": POLICY["version"], "domain_policy": TASK_DOMAIN_POLICY,
               "minimum_tasks_per_D2b_context": 2,
               "original_goal_variant_ids": original, "selected_goal_variant_ids": selected,
               "excluded_goal_variant_ids": sorted(set(original) - set(selected)),
               "contexts": goal_groups(realized), "source_goals_hash": artifact_hash(realized["goal_variants"])}
    return {**payload, "selection_hash": artifact_hash(payload)}


def valid_domain(realized, scene):
    ids = sorted(scene.get("goal_variant_ids", []))
    expected = sorted(v["id"] for v in realized.get("goal_variants", []))
    metadata = (scene.get("provenance") or {}).get("metadata") or {}
    if ids == expected and "d2b_task_domain" not in metadata:
        return True
    try:
        return metadata.get("d2b_task_domain") == domain_receipt(realized, ids)
    except ValueError:
        return False


def selected_realized(realized, scene):
    if "goal_variant_ids" not in scene and not ((scene.get("provenance") or {}).get("metadata") or {}).get("d2b_task_domain"):
        return realized
    if not valid_domain(realized, scene):
        raise ValueError("invalid scene task-domain provenance")
    ids = set(scene["goal_variant_ids"])
    return {**realized, "goal_variants": [v for v in realized["goal_variants"] if v["id"] in ids],
            "minimal_pair_edges": [e for e in realized.get("minimal_pair_edges", [])
                                   if e["from"] in ids and e["to"] in ids]}


def validate_operation_request(request, scene, units):
    """Rejoin a worker's filtered Contact request to its frozen local proof."""
    op = ((scene.get("provenance") or {}).get("metadata") or {}).get("operation_space")
    if op is None:
        if resources(units):
            raise ValueError("M11 operation resources require a scene proof")
        if any(t.get("operation_constraint") is not None for t in request["targets"]):
            raise ValueError("operation target has no scene proof")
        return
    verify_operation_sources(units)
    if (op.get("constraint_hash") != artifact_hash(op, omit_keys=("constraint_hash",))
            or op.get("scene_state_hash") != scene.get("initial_observation", {}).get("scene_state_hash")
            or op.get("resource_hashes") != {k: v["content_hash"] for k, v in resources(units).items()}):
        raise ValueError("operation request scene/resource proof mismatch")
    options = op["goal_options"]
    for target in request["targets"]:
        ids = sorted(g["goal_variant_id"] for g in request["requirement_groups"]
                     if g["entity_key"] == target["entity_key"] and g["selected_arm"] == target["arm"]
                     and g["goal_variant_id"] in options)
        expected = None
        if ids:
            allowed = set.intersection(*[{b["contact_id"] for b in options[gid]
                                          if b["arm"] == target["arm"]} for gid in ids])
            expected = {"constraint_hash": op["constraint_hash"], "goal_variant_ids": ids,
                        "allowed_contact_ids": sorted(allowed)}
            if not allowed or target["planning_request"]["contact_ids"] != sorted(allowed):
                raise ValueError("operation target Contact domain mismatch")
        if target.get("operation_constraint") != expected:
            raise ValueError("operation target assigned-goal proof mismatch")


def compatible_arm_choices(pending, operation, *, fixed=False):
    """Same (subject, arm) must have a common Contact for every assigned goal."""
    options = operation.get("goal_options") or {}
    domains = [[r["selected_arm"]] if fixed or r["semantic"] else
               [r["selected_arm"], *[a for a in r["eligible_arms"] if a != r["selected_arm"]]]
               for r in pending]
    from itertools import islice
    for assignment in islice(product(*domains), POLICY["maximum_arm_assignments"]):
        shared = {}
        valid = True
        for row, arm in zip(pending, assignment):
            if row["goal_variant_id"] not in options:
                continue
            contacts = {b["contact_id"] for b in options[row["goal_variant_id"]] if b["arm"] == arm}
            key = row["entity_key"], arm
            shared[key] = contacts if key not in shared else shared[key] & contacts
            if not shared[key]:
                valid = False
                break
        if valid:
            for row, arm in zip(pending, assignment):
                if row["selected_arm"] != arm:
                    row["selection"] = "operation_space_common_contact"
                row["selected_arm"] = arm
            return
    raise ValueError("pick_planning_no_direct_arm:operation_space_common_contact")


def reserve_pair_operation_space(realized, units):
    """Center the subject in the free gap; size both sides for its operation.

    The chosen spacing is a proposal. M12 still checks its actual arm/Contact
    branches; a narrow template is never substituted for a different grasp.
    """
    from if_benchmark.units.core import _clean, _envelope, _support_plane_position
    templates = OperationTemplates(units)
    for unit in units:
        p = unit["params"]
        if unit["type"] != "ReceiverReservationUnit" or len(p["receiver_keys"]) != 2:
            continue
        regions = [r for r in p["reserved_regions"] if r["relation"] == "between"]
        if not regions:
            continue
        left, right = sorted(p["receiver_keys"], key=lambda k: p["receiver_footprints"][k]["center_xy"][0])
        left_fp, right_fp = (p["receiver_footprints"][k] for k in (left, right))
        left_edge = left_fp["center_xy"][0] + left_fp["half_extents_xy"][0]
        right_edge = right_fp["center_xy"][0] - right_fp["half_extents_xy"][0]
        old_gap = right_edge - left_edge
        branches = []
        for region in regions:
            subject = region["subject_key"]
            if subject not in templates.resources:
                raise ValueError("D2b operation resource missing")
            for arm in ("left", "right"):
                for raw_cid in templates.resources[subject]["geometry"]["contacts"]:
                    template = templates.get(subject, arm, int(raw_cid), "between")
                    if template is None:
                        continue
                    lows, highs = [], []
                    # Reserve the whole moving gripper, fixed wrist camera and
                    # carried shape. No enclosing sphere or object-only width.
                    for phase in ("descent_held", "release_open", "retreat_open"):
                        lows.append(template["phase_bounds"][phase]["min"][0])
                        highs.append(template["phase_bounds"][phase]["max"][0])
                    vertices = np.concatenate(templates.resources[subject]["geometry"]["pieces"])
                    subject_center_x = float((vertices[:, 0].min() + vertices[:, 0].max()) / 2)
                    # Do not shift every terminal toward one landmark to center
                    # one asymmetric wrist/camera envelope. Keep the subject's
                    # collision AABB centered, and reserve enough room on BOTH
                    # sides for this same arm/Contact operation envelope.
                    half = max(subject_center_x - min(lows), max(highs) - subject_center_x,
                               float(region["footprint"]["half_extents_xy"][0]))
                    width = 2 * (half + POLICY["layout_clearance_m"])
                    branches.append({"subject": subject, "arm": arm, "contact_id": int(raw_cid),
                                     "width_m": width, "envelope_center_x": (max(highs) + min(lows)) / 2,
                                     "subject_center_x": subject_center_x,
                                     "centering": "subject_aabb_at_free_gap_midpoint"})
        if not branches:
            raise ValueError("no support-compatible operation-space branch")
        by_subject = {subject: min((b for b in branches if b["subject"] == subject),
                                  key=lambda r: (r["width_m"], r["arm"], r["contact_id"]))
                      for subject in {b["subject"] for b in branches}}
        branch = max(by_subject.values(), key=lambda r: r["width_m"])
        half_delta = max(0., (branch["width_m"] - old_gap) / 2)
        moves = {left: -half_delta, right: half_delta}
        for key, delta in moves.items():
            local = p["receiver_layout"][key]
            local["translation_xyz"][0] = _clean(local["translation_xyz"][0] + delta)
            footprint = p["receiver_footprints"][key]
            footprint["center_xy"][0] = _clean(footprint["center_xy"][0] + delta)
            for name, value in p["anchors"].items():
                if name.startswith(key + ":"):
                    value[0] = _clean(value[0] + delta)
            # M11 support anchors use normalized coordinates and exact equality.
            # Recompute from the final pose instead of retaining float additions
            # at a different precision from the original Unit builder.
            p["anchors"][key + ":origin"] = list(local["translation_xyz"])
            p["anchors"][key + ":support"] = _support_plane_position(
                local, p["receiver_z_bounds"][key])
        gap_center = (left_edge + right_edge) / 2
        for r in p["reserved_regions"]:
            if r["relation"] == "between":
                # Actor origins need not coincide with their collision centers.
                r["footprint"]["center_xy"][0] = _clean(gap_center - by_subject[r["subject_key"]]["subject_center_x"])
            elif r["relation"] == "left_of_pair":
                r["footprint"]["center_xy"][0] = _clean(r["footprint"]["center_xy"][0] - half_delta)
            elif r["relation"] == "right_of_pair":
                r["footprint"]["center_xy"][0] = _clean(r["footprint"]["center_xy"][0] + half_delta)
        p["footprint"] = _envelope([*p["receiver_footprints"].values(),
                                    *[r["footprint"] for r in p["reserved_regions"]]],
                                   source="operation_space_pair_reservation")
        p["operation_spacing"] = {"policy": POLICY["version"], "original_gap_m": old_gap,
                                   "gap_m": old_gap + 2 * half_delta, "proposal_branch": branch,
                                   "proposal_branches_by_subject": by_subject,
                                   "branch_count": len(branches), "not_a_grasp_certificate": True}


def geometry_scene(realized, units, placements, table_z):
    """Exact pure M11 pose projection needed by the shared terminal contract."""
    from transforms3d.quaternions import axangle2quat, qmult
    bindings = {b["entity_key"]: (role, b) for role, b in realized["role_bindings"].items()}
    entities = {}
    for unit in units:
        if unit["id"] not in placements:
            continue
        for key, local in local_poses(unit).items():
            role, binding = bindings[key]
            position = (np.r_[placements[unit["id"]], table_z] + local["translation_xyz"]).tolist()
            yaw = float(local.get("relative_yaw_deg", 0.))
            quat = qmult(axangle2quat([0, 0, 1], math.radians(yaw)), local["q0_wxyz"]).tolist()
            support = deepcopy(local["support"])
            support["origin_z"] = position[2]
            support["plane_z"] = position[2] + local["origin_relative_bounds"]["min_xyz"][2] - support["clearance_m"]
            entities[key] = {"asset": binding["asset"], "model_id": binding["model_id"],
                "role_ref": role, "unit_id": unit["id"], "pose": {"position": position, "quat_wxyz": quat},
                "initial_pose": {**local, "support": support, "world_yaw_deg": yaw}}
    return {"entities": entities, "unit_poses": [{"unit_id": key, "position": [*p, table_z], "yaw_deg": 0.}
                                                for key, p in placements.items()]}


class ReplayableCandidates:
    """Generate only the terminal prefix actually needed; replay it per Contact.

    Iteration order, rejected terminals and the complete fallback domain stay
    unchanged. Appending before yielding also preserves a partially read prefix.
    """
    def __init__(self, source):
        self.source = iter(source)
        self.rows = []

    def __iter__(self):
        index = 0
        while True:
            if index == len(self.rows):
                try:
                    self.rows.append(next(self.source))
                except StopIteration:
                    return
            yield self.rows[index]
            index += 1


class OperationLayoutGate:
    def __init__(self, realized, units, config):
        self.realized, self.units, self.config = realized, units, config
        verify_operation_sources(units)
        self.templates = OperationTemplates(units)
        self.goals = [v for v in realized.get("goal_variants", [])
                      if v.get("goal", {}).get("action_family") == "O2O_relation"]
        self.enabled = bool(self.goals)
        if self.enabled and any(v["goal"]["subject"] not in self.templates.resources for v in self.goals):
            raise ValueError("M11 operation-space resources missing; rebuild M11 before M12")
        self.cache = {}
        self.last_options = {}

    def evaluate(self, placements):
        if not self.enabled:
            return True
        key = tuple((k, tuple(p)) for k, p in sorted(placements.items()))
        if key in self.cache:
            self.last_options = self.cache[key]
            return bool(self.last_options)
        if len(self.cache) >= POLICY["maximum_cpu_layout_evaluations_per_goal_domain"]:
            raise OperationSpaceSearchExhausted("CPU operation-space proposal budget exhausted")
        from .core import _in_reach
        from if_benchmark.reach_validation.candidates import terminal_candidates
        scene = geometry_scene(self.realized, self.units, placements, self.config.table_z)
        boxes = scene_boxes(scene, self.units)
        options = {}
        for variant in self.goals:
            goal = variant["goal"]
            subject, relation = goal["subject"], goal["relation"]
            initial = scene["entities"][subject]["pose"]["position"]
            receivers = [goal[k] for k in ("target", "target2") if goal.get(k)]
            arms = [goal["arm"]] if goal.get("arm") else ["left", "right"]
            # Enumerate semantics once per goal, not once per Contact or arm.
            candidates = None
            branches = []
            for arm in arms:
                for cid in self.templates.resources[subject]["geometry"]["contacts"]:
                    template = self.templates.get(subject, arm, int(cid), relation)
                    if template is None:
                        continue
                    contact = np.asarray(template["spec"]["contact_matrix"])[:2, 3] + initial[:2]
                    if not _in_reach(contact, arm, self.config) or not initial_clear(template, boxes, subject, initial):
                        continue
                    if candidates is None:
                        candidates = ReplayableCandidates(terminal_candidates({"units": self.units}, scene, goal))
                    for terminal in candidates:
                        # The terminal frame may include an M5 relative yaw.
                        r = quat2mat(terminal["actor_pose"][3:]) @ quat2mat(
                            self.templates.resources[subject]["geometry"]["quat_wxyz"]).T
                        yaw = math.degrees(math.atan2(r[1, 0], r[0, 0]))
                        held = self.templates.get(subject, arm, int(cid), relation, yaw_deg=round(yaw, 8))
                        if held is None:
                            continue
                        final_contact = np.asarray(held["spec"]["contact_matrix"])[:2, 3] + terminal["actor_pose"][:2]
                        if not _in_reach(final_contact, arm, self.config):
                            continue
                        if placement_clear(held, boxes, subject, terminal, relation, receivers):
                            branches.append({"arm": arm, "contact_id": int(cid),
                                "terminal_example": {k: terminal[k] for k in ("offset_xy_m", "yaw_delta_deg")}})
                            break
            if not branches:
                self.last_options = {}
                self.cache[key] = {}
                return False
            options[variant["id"]] = branches
        self.last_options = options
        self.cache[key] = options
        return True

    def proposal_points(self, unit, placements, original, seed):
        """Analytic operation-envelope tangencies, not a centimetre scan."""
        if not self.enabled or not placements:
            return original
        active = [(k, p) for k, p in local_poses(unit).items() if k in self.templates.resources]
        if not active:
            return original
        scene = geometry_scene(self.realized, self.units, placements, self.config.table_z)
        boxes = scene_boxes(scene, self.units)
        result = set()
        for subject, local in active:
            relation = next(v["goal"]["relation"] for v in self.goals if v["goal"]["subject"] == subject)
            for arm in ("left", "right"):
                for cid in self.templates.resources[subject]["geometry"]["contacts"]:
                    template = self.templates.get(subject, arm, int(cid), relation)
                    if template is None:
                        continue
                    for phase in ("approach_open", "close", "lift_held"):
                        b = template["phase_bounds"][phase]
                        lo = np.asarray(b["min"]) + local["translation_xyz"]
                        hi = np.asarray(b["max"]) + local["translation_xyz"]
                        for box in boxes.values():
                            if hi[2] + self.config.table_z < box["min"][2] or lo[2] + self.config.table_z > box["max"][2]:
                                continue
                            xs = [box["min"][0] - hi[0] - POLICY["layout_clearance_m"],
                                  box["max"][0] - lo[0] + POLICY["layout_clearance_m"]]
                            ys = [box["min"][1] - hi[1] - POLICY["layout_clearance_m"],
                                  box["max"][1] - lo[1] + POLICY["layout_clearance_m"]]
                            for p in original[:8]:
                                result.update((round(x, 12), p[1]) for x in xs)
                                result.update((p[0], round(y, 12)) for y in ys)
                            result.update((round(x, 12), round(y, 12)) for x in xs for y in ys)
        ordered = sorted(result, key=lambda p: artifact_hash({"family": self.realized["id"], "unit": unit["id"], "seed": seed, "point": p}))
        return list(dict.fromkeys([*ordered[:64], *original]))
