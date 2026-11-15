"""Read-only commitment extractors, independent of Success and policy internals."""
import math


class PickIntent:
    def __init__(self, candidates, links, *, impulse=1e-6, persistence=2):
        self.candidates, self.links = set(candidates), set(links)
        self.impulse, self.persistence = impulse, persistence
        self.streaks = {}
        self.result = None

    def observe(self, contacts, step, time_seconds):
        if self.result is not None:
            return
        active = set()
        for c in contacts:
            if c["summed_impulse_norm"] <= self.impulse:
                continue
            for entity, link in ((c["entity_a"], c.get("link_b")), (c["entity_b"], c.get("link_a"))):
                if entity in self.candidates and link in self.links:
                    active.add(entity)
        self.streaks = {k: self.streaks.get(k, 0) + 1 for k in active}
        committed = sorted(k for k, n in self.streaks.items() if n >= self.persistence)
        if committed:
            self.result = {"classification": "unique" if len(committed) == 1 else "ambiguous",
                           "value": committed[0] if len(committed) == 1 else None,
                           "candidates": committed, "step": step, "time_seconds": time_seconds}

    def finish(self):
        return self.result or {"classification": "no_commitment", "value": None}


class ArmIntent:
    def __init__(self, baseline, scales, *, window=2., activity=.005, dominance=2.):
        if set(baseline) != {"left", "right"} or set(scales) != set(baseline):
            raise ValueError("both arm channels required")
        for arm in baseline:
            if not baseline[arm] or len(baseline[arm]) != len(scales[arm]):
                raise ValueError("arm channel shape mismatch")
            if not all(math.isfinite(s) and s > 0 for s in scales[arm]):
                raise ValueError("finite positive action scales required")
        self.baseline, self.scales = baseline, scales
        self.window, self.activity, self.dominance = window, activity, dominance
        self.integral = {arm: 0. for arm in baseline}
        self.elapsed = 0.

    def observe(self, command, dt):
        duration = min(dt, max(0., self.window - self.elapsed))
        if duration <= 0:
            return
        for arm, base in self.baseline.items():
            values = command[arm]
            if len(values) != len(base) or not all(math.isfinite(v) for v in values):
                raise ValueError("invalid physical control command")
            self.integral[arm] += duration * sum(abs(v-b)/s for v,b,s in
                zip(values, base, self.scales[arm])) / len(base)
        self.elapsed += duration

    def finish(self):
        if self.elapsed <= 0:
            raise ValueError("D4 requires observed control time")
        magnitude = {k: v / self.elapsed for k,v in self.integral.items()}
        arm = max(magnitude, key=magnitude.get)
        other = "right" if arm == "left" else "left"
        if magnitude[arm] <= self.activity:
            classification, value = "no_commitment", None
        elif (magnitude[arm] >= self.dominance * magnitude[other]
              or math.isclose(magnitude[arm], self.dominance * magnitude[other], rel_tol=1e-9, abs_tol=1e-12)):
            classification, value = "unique", arm
        else:
            classification, value = "ambiguous", None
        return {"classification": classification, "value": value,
                "magnitude": magnitude, "observed_seconds": self.elapsed}


def project_intent(event, members):
    """Map one shared event to a context column, retaining out-of-set mass."""
    if event["classification"] != "unique":
        return {"column": None, "residual": event["classification"]}
    matches = [m["matrix_index"] for m in members if m["intent_value"] == event["value"]]
    if len(matches) > 1:
        raise ValueError("context has indistinguishable intent values")
    return {"column": matches[0] if matches else None, "residual": None if matches else "other"}


def shared_done(*, success, time_seconds, pick_required, pick_result, arm_required, policy, action_count):
    if time_seconds + 1e-8 >= policy.success_seconds or action_count >= policy.max_actions:
        return True
    pick_done = not pick_required or pick_result is not None or time_seconds + 1e-8 >= policy.pick_seconds
    arm_done = not arm_required or time_seconds + 1e-8 >= policy.arm_seconds
    return bool(success and pick_done and arm_done)
