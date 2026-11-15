"""CPU proposal geometry for the actual complete held-object sphere cover.

These are predictions from frozen M11 geometry. Settling and native FK still
require exact-scene validation; a negative proxy gap never proves unsat.
"""
import numpy as np
from transforms3d.quaternions import quat2mat
from itertools import product

from if_benchmark.layout.initial_candidates import InitialCandidatePolicy, POLICY as INITIAL_POLICY
from if_benchmark.reach_validation.contract import ROUTE_EXECUTION_CONTRACT

from if_benchmark.grasp_validation.convex_cover import (
    CONVEX_GEOMETRY_VERSION, _geometry_hash, conservative_piece_cover,
)

POLICY = {
    **INITIAL_POLICY,
    'version': 'component_reservations_native_held_proxy_v2',
    'held_proxy_clearance_m': .005,
    'held_proxy_score_resolution_m': .001,
    'maximum_extra_boundary_proposals': 64,
    'proxy_scope': 'acquisition_vertical_held_spheres_vs_piece_aabbs',
    'proxy_is_planning_proof': False,
}


def held_spheres(template):
    spec = template['spec']
    pieces = [np.asarray(p, dtype=float) for p in spec['subject_pieces']]
    grasp = template['grasp']['grasp_pose']
    rotation = quat2mat(grasp[3:])
    origin = np.asarray(grasp[:3])
    transform = np.eye(4)
    transform[:3, :3] = rotation.T
    transform[:3, 3] = -rotation.T @ origin
    geometry = dict(version=CONVEX_GEOMETRY_VERSION, frame='world',
                    geometry_sha256=_geometry_hash(pieces), pieces=pieces)
    collision = ROUTE_EXECUTION_CONTRACT['held_collision']
    cover = conservative_piece_cover(
        geometry, world_to_parent=transform,
        dimension_envelope_m=float(collision['dimension_envelope_m']),
        padding_m=float(collision['planner_padding_m']), slot_count=32)
    spheres = np.asarray(cover['spheres'], dtype=float)
    spheres = spheres[spheres[:, 3] > 0].copy()
    spheres[:, :3] = spheres[:, :3] @ rotation.T + origin
    return spheres


def vertical_sweep_gap(spheres, position, lift, boxes, *, excluded=()):
    """Signed clearance against the swept obstacle in proposal coordinates.

    Positive values are clearance. Negative values describe how far the entire
    proxy sweep overlaps its configuration-space obstacle, not single-point
    physical penetration depth.
    """
    centers = spheres[:, :3] + np.asarray(position)
    retained = [box for key, box in boxes.items() if box.get('entity', key) not in excluded]
    if not retained:
        return float('inf')
    lo = np.asarray([box['min'] for box in retained], float)
    hi = np.asarray([box['max'] for box in retained], float)
    lo[:, 2] -= lift
    # Extruding the box downward tests the complete upward sphere-center segment.
    d = np.maximum(lo[:, None, :] - centers, centers - hi[:, None, :])
    gap = (np.linalg.norm(np.maximum(d, 0.), axis=2)
           + np.minimum(np.max(d, axis=2), 0.) - spheres[:, 3])
    return float(np.min(gap))


class NativeProxyCandidatePolicy(InitialCandidatePolicy):
    """Prefer clear held-lift proposals before rectangular reach/distance ties.

    Keep every legal previous candidate. Neither proxy overlap nor an empty
    sampled domain is used to declare geometric impossibility.
    """
    def __init__(self, operation_gate):
        super().__init__(operation_gate)
        from if_benchmark.layout.manipulation import geometry_scene, scene_boxes
        from if_benchmark.reach_validation.contract import route_lift_delta_m
        scene = geometry_scene(self.gate.realized, self.gate.units,
                               {uid: (0., 0.) for uid in self.units}, self.config.table_z)
        boxes = scene_boxes(scene, self.gate.units)
        self.boxes = [(key, self.locals[box.get('entity', key)][0], box)
                      for key, box in boxes.items()]
        self.spheres = {}
        self.lifts = {v['id']: route_lift_delta_m(v['goal']['relation']) for v in self.gate.goals}
        for subject, resource in self.gate.templates.resources.items():
            relation = next(v['goal']['relation'] for v in self.gate.goals
                            if v['goal']['subject'] == subject)
            for arm in ('left', 'right'):
                for cid in resource['geometry']['contacts']:
                    template = self.gate.templates.get(subject, arm, int(cid), relation)
                    if template is not None:
                        self.spheres[subject, arm, int(cid)] = (
                            held_spheres(template), np.asarray(template['spec']['contact_matrix'])[:2, 3])
        self.score_cache = {}

    def _boxes(self, placements):
        return {key: {'min': np.asarray(box['min']) + [*placements[uid], 0.],
                      'max': np.asarray(box['max']) + [*placements[uid], 0.],
                      'entity': box.get('entity', key)}
                for key, uid, box in self.boxes if uid in placements}

    def proxy_gaps(self, placements):
        boxes = self._boxes(placements)
        result = []
        gap_cache = {}
        for variant in self.gate.goals:
            goal = variant['goal']
            subject = goal['subject']
            uid, local = self.locals[subject]
            if uid not in placements:
                continue
            initial, terminal, yaw = self._points(goal, placements)
            position = np.asarray([*placements[uid], self.config.table_z]) + local['translation_xyz']
            gaps = []
            arms = (goal['arm'],) if goal.get('arm') else ('left', 'right')
            rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            for (owner, arm, cid), (spheres, contact) in self.spheres.items():
                if owner != subject or arm not in arms or self._margin(np.asarray(initial) + contact, arm) < -1e-10:
                    continue
                if terminal is not None and self._margin(np.asarray(terminal) + rotation @ contact, arm) < -1e-10:
                    continue
                key = subject, arm, cid, self.lifts[variant['id']]
                if key not in gap_cache:
                    gap_cache[key] = vertical_sweep_gap(spheres, position, key[-1], boxes, excluded=(subject,))
                gaps.append(gap_cache[key])
            result.append(max(gaps, default=-1.e6))
        return result

    def score(self, placements):
        key = tuple((k, tuple(p)) for k, p in sorted(placements.items()))
        if key in self.score_cache:
            return self.score_cache[key]
        base = super().score(placements)
        gaps = [min(gap, POLICY['held_proxy_clearance_m']) for gap in self.proxy_gaps(placements)]
        resolution = POLICY['held_proxy_score_resolution_m']
        proxy = (sum(gap < 0 for gap in gaps), -round(min(gaps, default=0.) / resolution))
        result = (base[0], *proxy, *base[1:])
        self.score_cache[key] = result
        return result

    def order(self, unit, placements, original, seed, *, anchor_domains=()):
        from if_benchmark.layout.manipulation import local_poses
        boxes = self._boxes(placements)
        xs, ys = set(), set()
        margin = POLICY['held_proxy_clearance_m']
        for subject, local in local_poses(unit).items():
            lifts = [self.lifts[v['id']] for v in self.gate.goals if v['goal']['subject'] == subject]
            if not lifts:
                continue
            lift = max(lifts)
            for (owner, _, _), (spheres, _) in self.spheres.items():
                if owner != subject:
                    continue
                centers = spheres[:, :3] + local['translation_xyz']
                lo = np.min(centers - spheres[:, 3, None], axis=0)
                hi = np.max(centers + spheres[:, 3, None], axis=0) + [0., 0., lift]
                for box in boxes.values():
                    if box['entity'] == subject or hi[2] + self.config.table_z < box['min'][2] or lo[2] + self.config.table_z > box['max'][2]:
                        continue
                    xs.update((box['min'][0] - hi[0] - margin, box['max'][0] - lo[0] + margin))
                    ys.update((box['min'][1] - hi[1] - margin, box['max'][1] - lo[1] + margin))
        extra = set(product(xs, ys))
        for p in original:
            extra.update((x, p[1]) for x in xs)
            extra.update((p[0], y) for y in ys)
        # These are additional proposals, not a substitute for the inherited
        # complete AABB domain. Stable tie ordering remains seed-dependent.
        extras = sorted(extra, key=lambda p: self.tie(unit, p, seed))[:POLICY['maximum_extra_boundary_proposals']]
        return super().order(unit, placements, [*original, *extras], seed,
                             anchor_domains=anchor_domains)
