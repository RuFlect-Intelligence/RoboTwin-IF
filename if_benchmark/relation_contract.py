"""Shared scale-invariant geometry for M11 reservations and M13 checkers."""

from __future__ import annotations

import math
from typing import Any, Final, Mapping, Sequence


RELATION_GEOMETRY_CONTRACT_VERSION: Final = (
    "relation_geometry_v3_surface_gap_and_anchor_pair"
)
# Kept as a public compatibility name because existing M11/M12 consumers use
# it as the identity of the complete shared relation-geometry contract.
RELATION_SURFACE_GAP_CONTRACT_VERSION: Final = (
    RELATION_GEOMETRY_CONTRACT_VERSION
)
DIRECTIONAL_ORTHOGONAL_GAP_MAX_M: Final = 0.03
RELATION_SURFACE_GAP_BANDS_M: Final[dict[str, tuple[float, float]]] = {
    "next_to": (0.03, 0.09),
    "left_of": (0.03, 0.09),
    "right_of": (0.03, 0.09),
    "in_front_of": (0.03, 0.09),
    "behind": (0.03, 0.09),
}
ANCHOR_PAIR_DISTANCE_BANDS_M: Final[dict[str, tuple[float, float]]] = {
    "left_of_pair": (0.04, 0.18),
    "right_of_pair": (0.04, 0.18),
}
ANCHOR_PAIR_TERMINAL_CLEARANCE_M: Final[float] = 0.015
BETWEEN_MAXIMUM_PERPENDICULAR_M: Final = 0.08


def relation_surface_gap_band(relation: str) -> tuple[float, float] | None:
    """Return the scale-invariant AABB surface-gap band for one relation."""

    return RELATION_SURFACE_GAP_BANDS_M.get(str(relation))


def relation_surface_gap_metrics(
    subject_aabb: Mapping[str, Any],
    receiver_aabb: Mapping[str, Any],
    relation: str,
) -> dict[str, float | str]:
    """Measure the signed directional or nearest XY AABB surface gap."""

    subject_min, subject_max = _xy_bounds(subject_aabb)
    receiver_min, receiver_max = _xy_bounds(receiver_aabb)
    relation = str(relation)
    if relation == "left_of":
        gap = receiver_min[0] - subject_max[0]
        axis = "x"
        orthogonal_gap = _axis_gap(
            subject_min[1], subject_max[1], receiver_min[1], receiver_max[1]
        )
    elif relation == "right_of":
        gap = subject_min[0] - receiver_max[0]
        axis = "x"
        orthogonal_gap = _axis_gap(
            subject_min[1], subject_max[1], receiver_min[1], receiver_max[1]
        )
    elif relation == "in_front_of":
        gap = receiver_min[1] - subject_max[1]
        axis = "y"
        orthogonal_gap = _axis_gap(
            subject_min[0], subject_max[0], receiver_min[0], receiver_max[0]
        )
    elif relation == "behind":
        gap = subject_min[1] - receiver_max[1]
        axis = "y"
        orthogonal_gap = _axis_gap(
            subject_min[0], subject_max[0], receiver_min[0], receiver_max[0]
        )
    elif relation == "next_to":
        separation_x = max(
            subject_min[0] - receiver_max[0],
            receiver_min[0] - subject_max[0],
            0.0,
        )
        separation_y = max(
            subject_min[1] - receiver_max[1],
            receiver_min[1] - subject_max[1],
            0.0,
        )
        gap = math.hypot(separation_x, separation_y)
        axis = "xy_nearest"
        orthogonal_gap = None
    else:
        raise ValueError(f"surface-gap relation is unsupported: {relation}")
    result: dict[str, float | str] = {
        "relation": relation,
        "surface_gap_axis": axis,
        "surface_gap_m": round(float(gap), 12),
    }
    if orthogonal_gap is not None:
        result["directional_orthogonal_gap_m"] = round(
            float(orthogonal_gap), 12
        )
    return result


def relation_surface_gap_acceptance(
    subject_aabb: Mapping[str, Any],
    receiver_aabb: Mapping[str, Any],
    relation: str,
    *,
    surface_gap_band_m: Sequence[float] | None = None,
    directional_orthogonal_gap_maximum_m: float = (
        DIRECTIONAL_ORTHOGONAL_GAP_MAX_M
    ),
) -> dict[str, Any]:
    """Evaluate the one surface-gap predicate shared by M11/M14/M15.

    Callers may pass the band frozen into a SuccessContract; otherwise the
    canonical BenchmarkV1 band is used.  The returned check names are the
    exact names emitted by the final SuccessContract checker.
    """

    normalized_relation = str(relation)
    raw_band = (
        relation_surface_gap_band(normalized_relation)
        if surface_gap_band_m is None
        else surface_gap_band_m
    )
    if raw_band is None or len(raw_band) != 2:
        raise ValueError(
            f"surface-gap relation has no acceptance band: {normalized_relation}"
        )
    minimum, maximum = (float(raw_band[0]), float(raw_band[1]))
    orthogonal_maximum = float(
        directional_orthogonal_gap_maximum_m
    )
    if not all(
        math.isfinite(value)
        for value in (minimum, maximum, orthogonal_maximum)
    ) or not (0.0 <= minimum <= maximum and orthogonal_maximum >= 0.0):
        raise ValueError("surface-gap acceptance limits are invalid")

    metrics = relation_surface_gap_metrics(
        subject_aabb, receiver_aabb, normalized_relation
    )
    surface_gap = float(metrics["surface_gap_m"])
    checks = {
        "relation_surface_gap_band": minimum
        <= surface_gap
        <= maximum,
    }
    orthogonal_gap = metrics.get("directional_orthogonal_gap_m")
    if orthogonal_gap is not None:
        checks["relation_directional_orthogonal_gap"] = (
            float(orthogonal_gap) <= orthogonal_maximum
        )
        metrics["maximum_directional_orthogonal_gap_m"] = (
            orthogonal_maximum
        )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            **metrics,
            "minimum_surface_gap_m": minimum,
            "maximum_surface_gap_m": maximum,
            "lower_surface_gap_margin_m": surface_gap - minimum,
            "upper_surface_gap_margin_m": maximum - surface_gap,
        },
    }


def relation_anchor_pair_acceptance(
    subject_aabb: Mapping[str, Any],
    receiver_aabb: Mapping[str, Any],
    receiver2_aabb: Mapping[str, Any],
    relation: str,
    *,
    distance_band_m: Sequence[float] | None = None,
    between_maximum_perpendicular_m: float = (
        BETWEEN_MAXIMUM_PERPENDICULAR_M
    ),
) -> dict[str, Any]:
    """Evaluate the shared collision-AABB-center anchor-pair predicate.

    This is the one pure geometry kernel consumed by M11 reservation checks,
    M13 route search, M15 live terminal admissibility, and the final Success
    Contract checker.  Receiver ordering is deliberately irrelevant: between
    uses the undirected segment and pair-side relations use its X envelope.
    """

    normalized_relation = str(relation)
    if normalized_relation not in {
        "between",
        "left_of_pair",
        "right_of_pair",
    }:
        raise ValueError(
            f"anchor-pair relation is unsupported: {normalized_relation}"
        )
    subject = _aabb_center_xyz(subject_aabb)
    receiver = _aabb_center_xyz(receiver_aabb)
    receiver2 = _aabb_center_xyz(receiver2_aabb)

    if normalized_relation == "between":
        maximum = float(between_maximum_perpendicular_m)
        if not math.isfinite(maximum) or maximum < 0.0:
            raise ValueError("between perpendicular limit is invalid")
        vx = receiver2[0] - receiver[0]
        vy = receiver2[1] - receiver[1]
        wx = subject[0] - receiver[0]
        wy = subject[1] - receiver[1]
        denominator = vx * vx + vy * vy
        projection = (
            (wx * vx + wy * vy) / denominator
            if denominator > 0.0
            else -1.0
        )
        closest_x = receiver[0] + projection * vx
        closest_y = receiver[1] + projection * vy
        perpendicular = math.hypot(
            subject[0] - closest_x, subject[1] - closest_y
        )
        checks = {
            "between_projection": 0.0 < projection < 1.0,
            "between_perpendicular": perpendicular <= maximum,
        }
        return {
            "passed": all(checks.values()),
            "checks": checks,
            "metrics": {
                "projection": projection,
                "perpendicular_m": perpendicular,
                "maximum_perpendicular_m": maximum,
                "projection_lower_margin": projection,
                "projection_upper_margin": 1.0 - projection,
                "perpendicular_margin_m": maximum - perpendicular,
            },
        }

    raw_band = (
        ANCHOR_PAIR_DISTANCE_BANDS_M[normalized_relation]
        if distance_band_m is None
        else distance_band_m
    )
    if len(raw_band) != 2:
        raise ValueError("anchor-pair distance band must contain two values")
    minimum, maximum = float(raw_band[0]), float(raw_band[1])
    if not all(math.isfinite(value) for value in (minimum, maximum)) or not (
        0.0 <= minimum <= maximum
    ):
        raise ValueError("anchor-pair distance band is invalid")
    boundary = (
        min(receiver[0], receiver2[0])
        if normalized_relation == "left_of_pair"
        else max(receiver[0], receiver2[0])
    )
    signed_delta = (
        boundary - subject[0]
        if normalized_relation == "left_of_pair"
        else subject[0] - boundary
    )
    checks = {
        "relative_to_anchor_pair": signed_delta > 0.0,
        "anchor_pair_distance_band": minimum <= signed_delta <= maximum,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "subject_x_m": subject[0],
            "boundary_x_m": boundary,
            "signed_delta_m": signed_delta,
            "minimum_m": minimum,
            "maximum_m": maximum,
            "lower_margin_m": signed_delta - minimum,
            "upper_margin_m": maximum - signed_delta,
        },
    }


def _axis_gap(
    subject_minimum: float,
    subject_maximum: float,
    receiver_minimum: float,
    receiver_maximum: float,
) -> float:
    return max(
        subject_minimum - receiver_maximum,
        receiver_minimum - subject_maximum,
        0.0,
    )


def _xy_bounds(
    aabb: Mapping[str, Any],
) -> tuple[tuple[float, float], tuple[float, float]]:
    minimum = aabb.get("min")
    maximum = aabb.get("max")
    if (
        not isinstance(minimum, Sequence)
        or isinstance(minimum, (str, bytes))
        or len(minimum) < 2
        or not isinstance(maximum, Sequence)
        or isinstance(maximum, (str, bytes))
        or len(maximum) < 2
    ):
        raise ValueError("relation surface-gap check requires an XY AABB")
    lower = (float(minimum[0]), float(minimum[1]))
    upper = (float(maximum[0]), float(maximum[1]))
    if any(lower[index] > upper[index] for index in (0, 1)):
        raise ValueError("relation surface-gap AABB bounds are inverted")
    return lower, upper


def _aabb_center_xyz(aabb: Mapping[str, Any]) -> tuple[float, float, float]:
    minimum = aabb.get("min")
    maximum = aabb.get("max")
    if (
        not isinstance(minimum, Sequence)
        or isinstance(minimum, (str, bytes))
        or len(minimum) < 3
        or not isinstance(maximum, Sequence)
        or isinstance(maximum, (str, bytes))
        or len(maximum) < 3
    ):
        raise ValueError("anchor-pair relation requires a 3D collision AABB")
    lower = tuple(float(minimum[index]) for index in range(3))
    upper = tuple(float(maximum[index]) for index in range(3))
    if not all(math.isfinite(value) for value in (*lower, *upper)) or any(
        lower[index] > upper[index] for index in range(3)
    ):
        raise ValueError("anchor-pair relation AABB is invalid")
    return tuple((lower[index] + upper[index]) * 0.5 for index in range(3))


__all__ = [
    "ANCHOR_PAIR_DISTANCE_BANDS_M",
    "ANCHOR_PAIR_TERMINAL_CLEARANCE_M",
    "BETWEEN_MAXIMUM_PERPENDICULAR_M",
    "DIRECTIONAL_ORTHOGONAL_GAP_MAX_M",
    "RELATION_GEOMETRY_CONTRACT_VERSION",
    "RELATION_SURFACE_GAP_BANDS_M",
    "RELATION_SURFACE_GAP_CONTRACT_VERSION",
    "relation_anchor_pair_acceptance",
    "relation_surface_gap_acceptance",
    "relation_surface_gap_band",
    "relation_surface_gap_metrics",
]
