"""Shared mass-property updates for manipulable dynamic assets.

RoboTwin intentionally overrides many automatically computed or authored
asset masses.  For unchanged geometry, mass and principal inertia must scale
by the same density ratio.  Updating only ``mass`` leaves ``I / m``
inconsistent and produces non-physical settling and manipulation dynamics.

This module is dependency-light and duck-types the SAPIEN body/link API so the
invariant can be unit-tested without starting a simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import numpy as np


class MassInertiaError(ValueError):
    """Raised before an invalid mass-property update reaches the simulator."""


@dataclass(frozen=True)
class _MassInertiaUpdate:
    body: Any
    old_mass: float
    old_inertia: np.ndarray
    new_mass: float
    new_inertia: np.ndarray


def _read_body_property(body: Any, name: str) -> Any:
    try:
        return getattr(body, name)
    except AttributeError:
        getter = getattr(body, f"get_{name}", None)
        if not callable(getter):
            raise MassInertiaError(
                f"dynamic body does not expose {name!r}"
            )
        return getter()


def _write_body_property(body: Any, name: str, value: Any) -> None:
    setter = getattr(body, f"set_{name}", None)
    if callable(setter):
        setter(value)
    else:
        setattr(body, name, value)


def _prepare_mass_inertia_update(
    body: Any,
    target_mass: float,
) -> _MassInertiaUpdate:
    try:
        new_mass = float(target_mass)
    except (TypeError, ValueError) as exc:
        raise MassInertiaError(
            f"target mass must be a finite positive scalar: {target_mass!r}"
        ) from exc
    if not math.isfinite(new_mass) or new_mass <= 0.0:
        raise MassInertiaError(
            f"target mass must be finite and positive: {new_mass!r}"
        )

    try:
        old_mass = float(_read_body_property(body, "mass"))
    except (TypeError, ValueError) as exc:
        raise MassInertiaError("dynamic body mass is not a scalar") from exc
    if not math.isfinite(old_mass) or old_mass <= 0.0:
        raise MassInertiaError(
            f"dynamic body mass must be finite and positive: {old_mass!r}"
        )

    try:
        old_inertia = np.asarray(
            _read_body_property(body, "inertia"), dtype=float
        ).copy()
    except (TypeError, ValueError) as exc:
        raise MassInertiaError("dynamic body inertia is not numeric") from exc
    if (
        old_inertia.shape != (3,)
        or not np.all(np.isfinite(old_inertia))
        or np.any(old_inertia <= 0.0)
    ):
        raise MassInertiaError(
            "dynamic body inertia must contain three finite positive "
            f"principal moments: {old_inertia!r}"
        )

    ratio = new_mass / old_mass
    new_inertia = old_inertia * ratio
    if not np.all(np.isfinite(new_inertia)) or np.any(new_inertia <= 0.0):
        raise MassInertiaError(
            f"scaled inertia is invalid for mass ratio {ratio!r}"
        )
    return _MassInertiaUpdate(
        body=body,
        old_mass=old_mass,
        old_inertia=old_inertia,
        new_mass=new_mass,
        new_inertia=new_inertia,
    )


def set_bodies_mass_and_scale_inertia(
    body_mass_pairs: Iterable[tuple[Any, float]],
) -> None:
    """Apply a validated, rollback-safe batch of coupled mass updates.

    Each body or articulation link is scaled from its own current mass and
    inertia.  The center-of-mass local pose is deliberately untouched.
    """

    updates = [
        _prepare_mass_inertia_update(body, mass)
        for body, mass in body_mass_pairs
    ]
    applied: list[_MassInertiaUpdate] = []
    try:
        for update in updates:
            # Add the update before the first write so a failure in either
            # setter rolls this body back as well as earlier bodies.
            applied.append(update)
            _write_body_property(update.body, "mass", update.new_mass)
            _write_body_property(
                update.body, "inertia", update.new_inertia.copy()
            )
    except Exception:
        for update in reversed(applied):
            try:
                _write_body_property(
                    update.body, "inertia", update.old_inertia.copy()
                )
                _write_body_property(
                    update.body, "mass", update.old_mass
                )
            except Exception:
                # Preserve the original simulator error.  A failed rollback
                # means this runtime must be discarded by its caller.
                pass
        raise


def set_mass_and_scale_inertia(body: Any, target_mass: float) -> None:
    """Set one dynamic body's mass while preserving its inertia/mass ratio."""

    set_bodies_mass_and_scale_inertia(((body, target_mass),))


__all__ = [
    "MassInertiaError",
    "set_bodies_mass_and_scale_inertia",
    "set_mass_and_scale_inertia",
]
