"""Cheap M12 proposal priorities; no IK, collision exemption or grasp proof.

Keep physical footprints separate from alternative future reservations. Ranking
uses a same-arm, same-Contact rectangular reach proxy, never arm assignment.
The complete operation-space and native planning gates still decide admission.
"""
from itertools import product
import math

from if_benchmark.schema import artifact_hash

POLICY = {
    "version": "component_reservations_joint_reach_v1",
    "feasible_portfolio_size": 4,
    "reach_margin_saturation_m": .06,
    "score_resolution_m": .005,
}


def _bounds(footprint):
    x, y = footprint["center_xy"]
    hx, hy = footprint["half_extents_xy"]
    return x - hx, x + hx, y - hy, y + hy


def components(unit):
    """(kind, owner, bounds) in Unit-local XY; virtual regions are alternatives."""
    from .core import _entity_footprints
    physical = [("physical", key, _bounds(fp)) for key, fp in
                _entity_footprints([unit]).items()]
    return physical + [("reservation", r["subject_key"], _bounds(r["footprint"]))
                       for r in unit["params"].get("reserved_regions", [])]


def blocking_pairs(first, second):
    for a in first:
        for b in second:
            if a[0] == b[0] == "reservation":
                continue  # Different counterfactual executions, not co-occupancy.
            if a[0] != b[0] and a[1] == b[1]:
                continue  # This exact subject vacates its initial pose in this task.
            yield a[2], b[2]


def component_overlap(first, first_xy, second, second_xy, margin):
    from .core import _global_bounds, _overlap_with_margin
    return any(_overlap_with_margin(_global_bounds(a, first_xy),
                                    _global_bounds(b, second_xy), margin)
               for a, b in blocking_pairs(components(first), components(second)))


class InitialCandidatePolicy:
    def __init__(self, operation_gate):
        from .manipulation import local_poses
        self.gate = operation_gate
        self.config = operation_gate.config
        self.units = {u["id"]: u for u in operation_gate.units}
        self.parts = {k: components(u) for k, u in self.units.items()}
        self.locals = {k: (u["id"], p) for u in self.units.values()
                       for k, p in local_poses(u).items()}
        self.contacts = {}
        for subject, resource in operation_gate.templates.resources.items():
            local = self.locals[subject][1]
            angle = math.radians(local.get("relative_yaw_deg", 0.))
            c, s = math.cos(angle), math.sin(angle)
            self.contacts[subject] = []
            for matrix in resource["geometry"]["contacts"].values():
                x, y = matrix[0][3], matrix[1][3]
                self.contacts[subject].append((c*x-s*y, s*x+c*y))
        self.regions = {}
        for unit in self.units.values():
            for r in unit["params"].get("reserved_regions", []):
                key = (r["subject_key"], r["relation"], tuple(r.get("receiver_keys") or
                       unit["params"].get("receiver_keys") or []))
                self.regions[key] = (unit["id"], r)

    def overlaps(self, unit, point, placements):
        from .core import _global_bounds, _overlap_with_margin
        return any(_overlap_with_margin(_global_bounds(a, point),
                                        _global_bounds(b, position), self.config.margin)
                   for other, position in placements.items()
                   for a, b in blocking_pairs(self.parts[unit["id"]], self.parts[other]))

    def _margin(self, point, arm):
        reach = self.config.data["reach"][arm]
        return min(point[0]-reach["x"][0], reach["x"][1]-point[0],
                   point[1]-reach["y"][0], reach["y"][1]-point[1])

    def _points(self, goal, placements):
        subject = goal["subject"]
        uid, local = self.locals[subject]
        initial = None if uid not in placements else tuple(
            placements[uid][i] + local["translation_xyz"][i] for i in (0, 1))
        key = (subject, goal["relation"], tuple(goal[k] for k in ("target", "target2") if goal.get(k)))
        uid, region = self.regions[key]
        terminal = None if uid not in placements else tuple(
            placements[uid][i] + region["footprint"]["center_xy"][i] for i in (0, 1))
        # Reservation center is a cheap proxy, not the admitted terminal pose.
        yaw = math.radians(region.get("relative_yaw_deg", 0.))
        return initial, terminal, yaw

    def score(self, placements):
        margins, distances = [], []
        for variant in self.gate.goals:
            g = variant["goal"]
            initial, terminal, yaw = self._points(g, placements)
            if initial is None and terminal is None:
                continue
            arms = [g["arm"]] if g.get("arm") else ["left", "right"]
            best = -1.e6
            for dx, dy in self.contacts[g["subject"]]:
                tx, ty = math.cos(yaw)*dx-math.sin(yaw)*dy, math.sin(yaw)*dx+math.cos(yaw)*dy
                for arm in arms:
                    values = []
                    if initial is not None:
                        values.append(self._margin((initial[0]+dx, initial[1]+dy), arm))
                    if terminal is not None:
                        values.append(self._margin((terminal[0]+tx, terminal[1]+ty), arm))
                    best = max(best, min(values))
            margins.append(best)
            if initial is not None and terminal is not None:
                distances.append(math.dist(initial, terminal))
        if not margins:
            return (0, 0, 0, 0)
        # Saturate useful slack; do not chase extra reach depth at any distance.
        resolution = POLICY["score_resolution_m"]
        capped = [min(x, POLICY["reach_margin_saturation_m"]) for x in margins]
        return (sum(x < -1e-10 for x in margins),
                -round(min(capped)/resolution),
                -round(sum(capped)/len(capped)/resolution),
                round(sum(distances)/max(1, len(distances))/resolution))

    def tie(self, unit, point, seed):
        return artifact_hash(dict(family=self.gate.realized["id"], unit=unit["id"],
                                  seed=seed, point=point))

    def order(self, unit, placements, original, seed, *, anchor_domains=()):
        """Add component tangencies and joint reach centers, retain every input."""
        from .manipulation import local_poses
        from .core import (_unit_bounds_local, _camera_anchor_halfplanes,
                           _rectangle_halfplanes, _halfplane_domain_candidates,
                           _global_bounds, _bounds_in_workspace, _bounds_in_polygon)
        local_bounds = _unit_bounds_local(unit)
        camera = _camera_anchor_halfplanes(local_bounds, self.config.camera_polygon)
        result = set(original)
        xs, ys = set(), set()
        for other, position in placements.items():
            for a, b in blocking_pairs(self.parts[unit["id"]], self.parts[other]):
                m = self.config.margin
                xs.update((position[0]+b[0]-a[1]-m, position[0]+b[1]-a[0]+m))
                ys.update((position[1]+b[2]-a[3]-m, position[1]+b[3]-a[2]+m))
        # Include component-boundary intersections with camera/reach domains,
        # including legal components that collapse to a line or single point.
        for axis, values in enumerate((xs, ys)):
            for value in values:
                equality = ([(1., 0., value), (-1., 0., -value)] if axis == 0
                            else [(0., 1., value), (0., -1., -value)])
                for domain in anchor_domains:
                    points, _ = _halfplane_domain_candidates(
                        [*_rectangle_halfplanes(domain), *camera, *equality])
                    result.update(points)
        # Includes cross-obstacle tangencies. No deletion of legal inputs.
        result.update(product(xs, ys))
        for p in original:
            result.update((x, p[1]) for x in xs)
            result.update((p[0], y) for y in ys)
        for subject, local in local_poses(unit).items():
            if subject not in self.contacts:
                continue
            goals = [v["goal"] for v in self.gate.goals if v["goal"]["subject"] == subject]
            allowed = {g.get("arm") for g in goals}
            arms = ("left", "right") if None in allowed else tuple(allowed)
            centers = []
            for arm in arms:
                reach = self.config.data["reach"][arm]
                centers.append(tuple(sum(reach[k])/2 for k in ("x", "y")))
            if len(arms) == 2:
                left, right = (self.config.data["reach"][a] for a in ("left", "right"))
                lo = [max(left[k][0], right[k][0]) for k in ("x", "y")]
                hi = [min(left[k][1], right[k][1]) for k in ("x", "y")]
                if all(a <= b for a, b in zip(lo, hi)):
                    centers.append(tuple((a+b)/2 for a, b in zip(lo, hi)))
            targets = [p for g in goals for p in [self._points(g, placements)[1]] if p is not None]
            if targets:
                centers.append(tuple(sum(p[i] for p in targets)/len(targets) for i in (0, 1)))
            for offset in self.contacts[subject]:
                for center in centers:
                    p = tuple(center[i]-offset[i]-local["translation_xyz"][i] for i in (0, 1))
                    result.add(p)
                    result.update((x, p[1]) for x in xs)
                    result.update((p[0], y) for y in ys)
        result = {tuple(round(float(x), 12) for x in p) for p in result}
        # Drop only already-proven partial overlap before expensive template work.
        result = [p for p in result
                  if _bounds_in_workspace(_global_bounds(local_bounds, p), self.config)
                  and _bounds_in_polygon(_global_bounds(local_bounds, p), self.config.camera_polygon)
                  and not self.overlaps(unit, p, placements)]
        return self.rank(unit, placements, result, seed)

    def rank(self, unit, placements, points, seed):
        return sorted(points, key=lambda p: (self.score({**placements, unit["id"]: p}),
                                             self.tie(unit, p, seed)))
