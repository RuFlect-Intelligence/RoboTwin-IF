"""Small, spatially distinct M12 proposal batches with a complete lazy fallback.

This bounds expensive proxy scoring in the first batch. It does not remove a
layout for being near another layout, or turn finite search into unsat proof.
"""
from math import dist

from .initial_candidates import InitialCandidatePolicy
from .planner_proxy_candidates import NativeProxyCandidatePolicy, POLICY as PROXY_POLICY

POLICY = {
    **PROXY_POLICY,
    'version': 'component_reservations_sparse_held_proxy_v3',
    'primary_candidate_limit': 32,
    'proxy_scoring_batch_limit': 128,
    'cheap_priority_candidates': 64,
    'primary_separation_m': .02,
    'dense_fallback': 'lazy_same_legal_domain',
}


def separated_prefix(points, limit, separation):
    """Retain exact coordinates, including boundary-only candidates."""
    chosen = []
    for point in points:
        if all(dist(point, other) >= separation for other in chosen):
            chosen.append(point)
            if len(chosen) >= limit:
                break
    return chosen


class CandidateBatches:
    """The solver stops iteration on success; deferred scoring is then skipped."""
    def __init__(self, primary, deferred, rank, receipt):
        self.primary, self.deferred = primary, deferred
        self.rank, self.receipt = rank, receipt

    def __iter__(self):
        for point in self.primary:
            self.receipt['visited'] += 1
            yield point
        width = POLICY['proxy_scoring_batch_limit']
        for start in range(0, len(self.deferred), width):
            batch = self.deferred[start:start + width]
            self.receipt['fallback_batches'] += 1
            for point in self.rank(batch):
                self.receipt['visited'] += 1
                yield point


class SparseNativeProxyCandidatePolicy(NativeProxyCandidatePolicy):
    def __init__(self, operation_gate):
        super().__init__(operation_gate)
        self.sampling_batches = []

    def rank(self, unit, placements, points, seed):
        from .core import _not_success_holds
        uid = unit['id']
        count = len(points)
        # The same exact, cheap t0 gate otherwise runs after sorting in core.
        # Apply it early only when the whole selected task domain is placed.
        if len(placements) + 1 == len(self.units):
            points = [p for p in points if _not_success_holds(
                self.gate.units, {**placements, uid: p})]
        tie = {p: self.tie(unit, p, seed) for p in points}
        cheap = sorted(points, key=lambda p: (
            InitialCandidatePolicy.score(self, {**placements, uid: p}), tie[p]))
        # Keep promising close neighbors until AFTER proxy scoring: two points
        # a few millimetres apart can straddle the held-sphere collision edge.
        priority = cheap[:POLICY['cheap_priority_candidates']]
        spread = separated_prefix(cheap, POLICY['proxy_scoring_batch_limit']-len(priority),
                                  POLICY['primary_separation_m'])
        coarse = list(dict.fromkeys([*priority, *spread]))
        scored = set()
        receipt = dict(unit_id=uid, generated_legal=count,
                       initial_success_rejected=count-len(points),
                       proxy_scored=0, visited=0, fallback_batches=0)

        def rank_proxy(rows):
            ordered = sorted(rows, key=lambda p: (self.score({**placements, uid: p}), tie[p]))
            scored.update(rows)
            receipt['proxy_scored'] = len(scored)
            return ordered

        ordered = rank_proxy(coarse)
        primary = separated_prefix(ordered, POLICY['primary_candidate_limit'],
                                   POLICY['primary_separation_m'])
        primary_set, coarse_set = set(primary), set(coarse)
        # Coarse reserves precede near duplicates. No rounding or permanent
        # distance exclusion: a narrow feasible opening remains available.
        deferred = [p for p in ordered if p not in primary_set]
        deferred.extend(p for p in cheap if p not in coarse_set)
        receipt.update(primary=len(primary), deferred=len(deferred))
        self.sampling_batches.append(receipt)
        return CandidateBatches(primary, deferred, rank_proxy, receipt)

    def sampling_receipt(self):
        return dict(batches=[dict(row) for row in self.sampling_batches],
                    primary_limit=POLICY['primary_candidate_limit'],
                    primary_separation_m=POLICY['primary_separation_m'],
                    complete_legal_fallback_retained=True)
