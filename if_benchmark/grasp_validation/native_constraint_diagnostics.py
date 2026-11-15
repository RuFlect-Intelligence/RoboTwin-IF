"""Observe native constraint terms without changing their evaluation or verdict.

CuRobo calls each cost's ``forward`` directly, so Module forward hooks do not
observe these calls. The temporary wrappers return the identical native tensors
and restore the original instance state even when evaluation raises.
"""
from contextlib import contextmanager
from functools import wraps

import numpy as np


CONSTRAINTS = {
    "joint_bounds": "bound_constraint",
    "world_collision": "primitive_collision_constraint",
    "self_collision": "robot_self_collision_constraint",
}


@contextmanager
def observe_constraint_terms(rollout):
    """Capture the three terms from one unmodified non-graph native call."""
    captured, originals = {}, []
    try:
        for label, name in CONSTRAINTS.items():
            constraint = getattr(rollout, name)
            original = constraint.forward
            owned = "forward" in vars(constraint)
            originals.append((constraint, original, owned))

            @wraps(original)
            def forward(*args, _original=original, _label=label, **kwargs):
                value = _original(*args, **kwargs)
                captured.setdefault(_label, []).append(
                    value.detach().to("cpu").numpy().copy())
                return value

            constraint.forward = forward
        yield captured
    finally:
        for constraint, original, owned in reversed(originals):
            if owned:
                constraint.forward = original
            else:
                del constraint.forward


class NativeConstraintDiagnostics:
    """Diagnostic partition of the native failure mask, never admission input."""
    def __init__(self):
        self.failed = {label: [] for label in CONSTRAINTS}
        self.checked = 0
        self.errors = []

    def add(self, captured, feasible, *, offset):
        feasible = np.asarray(feasible, dtype=bool).reshape(-1)
        union = np.zeros(len(feasible), dtype=bool)
        errors = []
        failed = {}
        for label in CONSTRAINTS:
            calls = captured.get(label, [])
            if len(calls) != 1:
                errors.append(f"{label}:expected_one_native_evaluation")
                continue
            values = np.asarray(calls[0]).reshape(-1)
            if len(values) != len(feasible) or not np.isfinite(values).all():
                errors.append(f"{label}:invalid_native_term")
                continue
            mask = values != 0
            failed[label] = (offset + np.flatnonzero(mask)).tolist()
            union |= mask
        if offset != self.checked:
            errors.append("noncontiguous_waypoint_batch")
        if not errors and not np.array_equal(union, ~feasible):
            errors.append("component_mask_differs_from_native_verdict")
        self.checked += len(feasible)
        if errors:
            self.errors.extend({"offset": offset, "reason": reason} for reason in errors)
        else:
            for label, indices in failed.items():
                self.failed[label].extend(indices)

    def receipt(self):
        return {
            "version": "native_constraint_component_observation_v1",
            "diagnostic_only": True,
            "complete": not self.errors,
            "checked_waypoint_count": self.checked,
            "failed_waypoint_indices_by_constraint": self.failed,
            "errors": self.errors,
        }
