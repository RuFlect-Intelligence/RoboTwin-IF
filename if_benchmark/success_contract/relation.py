"""Shared D2b semantic relation contract and Reservation conformance checks."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
from typing import Any, Iterable, Mapping, Sequence

from if_benchmark.relation_contract import (
    ANCHOR_PAIR_DISTANCE_BANDS_M,
    BETWEEN_MAXIMUM_PERPENDICULAR_M,
    DIRECTIONAL_ORTHOGONAL_GAP_MAX_M,
    RELATION_SURFACE_GAP_BANDS_M,
    RELATION_SURFACE_GAP_CONTRACT_VERSION,
    relation_anchor_pair_acceptance,
    relation_surface_gap_acceptance,
)
from if_benchmark.schema.serialization import artifact_hash, canonical_data

from .contract import SuccessContract


@dataclass(frozen=True)
class RelationContract:
    data: dict[str, Any]
    hash: str

    @classmethod
    def from_success_contract(cls, contract: SuccessContract) -> "RelationContract":
        data = contract.section("relation")
        data["contract_version"] = RELATION_SURFACE_GAP_CONTRACT_VERSION
        data["surface_gap_bands_m"] = {
            relation: list(band)
            for relation, band in sorted(
                RELATION_SURFACE_GAP_BANDS_M.items()
            )
        }
        data["directional_orthogonal_gap_maximum_m"] = (
            DIRECTIONAL_ORTHOGONAL_GAP_MAX_M
        )
        data["distance_bands_m"].update(
            {
                relation: list(band)
                for relation, band in sorted(
                    ANCHOR_PAIR_DISTANCE_BANDS_M.items()
                )
            }
        )
        data["between_maximum_perpendicular_m"] = (
            BETWEEN_MAXIMUM_PERPENDICULAR_M
        )
        for anchor_group in ("directional", "distance"):
            data["anchors"][anchor_group] = {
                "subject": "collision_aabb",
                "receiver": "collision_aabb",
            }
        return cls(data=data, hash=artifact_hash(data))

    @property
    def version(self) -> str:
        return str(self.data["contract_version"])

    @property
    def relation_contract_hash(self) -> str:
        return self.hash

    def distance_band(self, relation: str) -> tuple[float, float] | None:
        value = (self.data.get("distance_bands_m") or {}).get(str(relation))
        if value is None:
            return None
        return float(value[0]), float(value[1])

    def surface_gap_band(self, relation: str) -> tuple[float, float] | None:
        value = (self.data.get("surface_gap_bands_m") or {}).get(
            str(relation)
        )
        if value is None:
            return None
        return float(value[0]), float(value[1])

    @property
    def identity(self) -> dict[str, str]:
        return {
            "relation_contract_version": self.version,
            "relation_contract_hash": self.hash,
        }


def static_relation_geometry_evidence(
    relation: str,
    subject: Mapping[str, Any],
    receiver: Mapping[str, Any],
    *,
    target2: Mapping[str, Any] | None = None,
    contract: RelationContract,
) -> dict[str, Any]:
    """Evaluate the Checker's static relation geometry without simulator state.

    M12 cannot observe terminal contact or settle stability, but it must use the
    same geometric acceptance region as M15 when excluding goals that already
    hold at the initial layout.  Required contact/stability guarantees are set
    true only to expose the shared geometric predicate; callers must not treat
    this result as terminal execution evidence.
    """

    receiver_value = canonical_data(dict(receiver))
    if target2 is not None:
        receiver_value["target2"] = canonical_data(dict(target2))
    passed, metrics = _static_relation_sample(
        str(relation),
        canonical_data(dict(subject)),
        subject_initial=canonical_data(dict(subject)),
        receiver=receiver_value,
        guarantees={
            "subject_receiver_contact": True,
            "subject_not_table_contact": True,
            "participating_entities_stable": True,
        },
        contract=contract,
    )
    return canonical_data(
        {
            "relation": str(relation),
            "geometry_holds": bool(passed),
            "metrics": metrics,
            **contract.identity,
        }
    )


def validate_reservation_subset(
    reservation: Mapping[str, Any],
    *,
    relation: str,
    subject_initial: Mapping[str, Any] | None,
    receiver: Mapping[str, Any] | None,
    contract: RelationContract,
) -> dict[str, Any]:
    """Validate declared terminal samples as a subset of semantic success.

    This is intentionally not a Reservation generator.  Execution-only fields
    such as transit/release clearance and approach corridors are ignored.
    """

    failures: list[dict[str, Any]] = []
    if reservation.get("relation_contract_hash") != contract.hash:
        failures.append({"code": "reservation_relation_contract_hash_mismatch"})
    if str(reservation.get("relation") or relation) != str(relation):
        failures.append({"code": "reservation_relation_mismatch"})
    anchor_group = _anchor_group(relation)
    anchors = (contract.data.get("anchors") or {}).get(anchor_group) or {}
    for key in ("subject", "receiver"):
        declared = reservation.get(f"{key}_anchor")
        if declared is not None and str(declared) != str(anchors.get(key) or ""):
            failures.append({"code": f"reservation_{key}_anchor_mismatch"})
    samples = list(_reservation_samples(reservation))
    if not samples:
        failures.append({"code": "reservation_terminal_region_empty"})
    checks: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        passed, metrics = _static_relation_sample(
            relation,
            sample,
            subject_initial=subject_initial,
            receiver=receiver,
            guarantees=reservation.get("guarantees") or {},
            contract=contract,
        )
        checks.append({"index": index, "passed": passed, "metrics": metrics})
        if not passed:
            failures.append(
                {"code": "reservation_sample_outside_relation_acceptance", "index": index}
            )
    return {
        "status": "passed" if not failures else "failed",
        "passed": not failures,
        **contract.identity,
        "sample_count": len(samples),
        "checks": checks,
        "failures": failures,
    }


def _reservation_samples(reservation: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    if isinstance(reservation.get("terminal_pose"), Mapping):
        yield canonical_data(dict(reservation["terminal_pose"]))
    for value in reservation.get("terminal_samples", []) or []:
        if isinstance(value, Mapping):
            yield canonical_data(dict(value))
    region = reservation.get("terminal_region")
    if not isinstance(region, Mapping):
        return
    if isinstance(region.get("samples"), Sequence):
        for value in region["samples"]:
            if isinstance(value, Mapping):
                yield canonical_data(dict(value))
        return
    bounds = region.get("position_bounds")
    if isinstance(bounds, Sequence) and len(bounds) >= 3:
        axes = [tuple(float(v) for v in axis) for axis in bounds[:3]]
        if all(len(axis) == 2 for axis in axes):
            center = [(axis[0] + axis[1]) * 0.5 for axis in axes]
            yield {"position": center}
            for values in product(*[(axis[0], axis[1]) for axis in axes]):
                yield {"position": values}


def _static_relation_sample(
    relation: str,
    sample: Mapping[str, Any],
    *,
    subject_initial: Mapping[str, Any] | None,
    receiver: Mapping[str, Any] | None,
    guarantees: Mapping[str, Any],
    contract: RelationContract,
) -> tuple[bool, dict[str, Any]]:
    position = _position(sample)
    receiver_position = _position(receiver or {})
    dx = position[0] - receiver_position[0]
    dy = position[1] - receiver_position[1]
    distance = math.hypot(dx, dy)
    metrics: dict[str, Any] = {"dx_m": dx, "dy_m": dy, "distance_m": distance}
    if relation in {
        "left_of",
        "right_of",
        "in_front_of",
        "behind",
        "next_to",
    }:
        surface_band = contract.surface_gap_band(relation)
        subject_aabb = _sample_subject_aabb(sample, subject_initial)
        receiver_aabb = _extract_aabb(receiver or {})
        if surface_band is None or subject_aabb is None or receiver_aabb is None:
            metrics["surface_gap_observation_complete"] = False
            return False, metrics
        acceptance = relation_surface_gap_acceptance(
            subject_aabb,
            receiver_aabb,
            relation,
            surface_gap_band_m=surface_band,
            directional_orthogonal_gap_maximum_m=float(
                contract.data["directional_orthogonal_gap_maximum_m"]
            ),
        )
        metrics.update(
            {
                **acceptance["metrics"],
                "surface_gap_observation_complete": True,
            }
        )
        return bool(acceptance["passed"]), metrics
    if relation in {"left_of_pair", "right_of_pair", "between"}:
        subject_aabb = _sample_subject_aabb(sample, subject_initial)
        receiver_aabb = _extract_aabb(receiver or {})
        receiver2_aabb = _extract_aabb(
            (receiver or {}).get("target2") or {}
        )
        pair_band = contract.distance_band(relation)
        if (
            subject_aabb is None
            or receiver_aabb is None
            or receiver2_aabb is None
            or (relation != "between" and pair_band is None)
        ):
            metrics["anchor_pair_observation_complete"] = False
            return False, metrics
        acceptance = relation_anchor_pair_acceptance(
            subject_aabb,
            receiver_aabb,
            receiver2_aabb,
            relation,
            distance_band_m=pair_band,
            between_maximum_perpendicular_m=float(
                contract.data["between_maximum_perpendicular_m"]
            ),
        )
        metrics.update(
            {
                **acceptance["metrics"],
                "anchor_pair_observation_complete": True,
            }
        )
        return bool(acceptance["passed"]), metrics
    # A Reservation cannot observe terminal contact/stability.  It may only
    # claim a subset when those constraints are explicit guarantees and every
    # static geometry sample is inside the Checker's acceptance region.
    required = contract.data.get("requirements", {}).get(relation) or {}
    required_guarantees = {
        "subject_receiver_contact": required.get("contact") == "subject_receiver",
        "subject_not_table_contact": bool(required.get("no_table_contact")),
        "participating_entities_stable": bool(required.get("stability")),
    }
    guarantees_passed = all(
        not needed or guarantees.get(name) is True
        for name, needed in required_guarantees.items()
    )
    geometry_passed = _static_support_geometry(
        relation, sample.get("aabb"), (receiver or {}).get("aabb"), contract
    )
    metrics.update(
        {
            "required_guarantees": required_guarantees,
            "declared_guarantees": canonical_data(dict(guarantees)),
            "guarantees_passed": guarantees_passed,
            "geometry_passed": geometry_passed,
        }
    )
    return bool(guarantees_passed and geometry_passed), metrics


def _anchor_group(relation: str) -> str:
    if relation in {"left_of", "right_of", "in_front_of", "behind"}:
        return "directional"
    if relation == "next_to":
        return "distance"
    if relation in {"between", "left_of_pair", "right_of_pair"}:
        return "between"
    return relation


def _static_support_geometry(
    relation: str,
    subject_raw: Any,
    receiver_raw: Any,
    contract: RelationContract,
) -> bool:
    if not isinstance(subject_raw, Mapping) or not isinstance(receiver_raw, Mapping):
        return False
    try:
        subject_min = [float(v) for v in subject_raw["min"]]
        subject_max = [float(v) for v in subject_raw["max"]]
        receiver_min = [float(v) for v in receiver_raw["min"]]
        receiver_max = [float(v) for v in receiver_raw["max"]]
    except (KeyError, TypeError, ValueError):
        return False
    subject_center = [(a + b) * 0.5 for a, b in zip(subject_min, subject_max)]
    receiver_center = [(a + b) * 0.5 for a, b in zip(receiver_min, receiver_max)]
    containment = contract.data["containment"]
    if relation == "in":
        xy_margin = min(
            min(
                subject_center[index] - receiver_min[index],
                receiver_max[index] - subject_center[index],
            )
            for index in (0, 1)
        )
        center_rim = subject_center[2] - receiver_max[2]
        depth = max(0.0, receiver_max[2] - subject_min[2])
        return bool(
            xy_margin >= -float(containment["xy_outside_tolerance_m"])
            and (
                center_rim <= float(containment["center_rim_tolerance_m"])
                or depth
                >= float(containment["depth_split_m"])
                - float(containment["depth_tolerance_m"])
            )
        )
    support = contract.data["support"]
    prefix = "stack" if relation == "stack" else "on"
    footprint = [receiver_max[index] - receiver_min[index] for index in (0, 1)]
    limit = max(
        float(support[f"{prefix}_minimum_xy_limit_m"]),
        min(footprint) * float(support[f"{prefix}_receiver_footprint_fraction"]),
    )
    xy_distance = math.dist(subject_center[:2], receiver_center[:2])
    depth = max(0.0, receiver_max[2] - subject_min[2])
    center_delta = subject_center[2] - receiver_max[2]
    center_threshold = (
        float(containment["center_rim_tolerance_m"])
        if relation == "on"
        else -0.005
    )
    return bool(
        xy_distance <= limit
        and center_delta > center_threshold
        and depth
        < float(containment["depth_split_m"])
        - float(containment["depth_tolerance_m"])
    )


def _position(value: Mapping[str, Any]) -> tuple[float, float, float]:
    raw = value.get("position", value)
    if isinstance(raw, Mapping):
        return tuple(float(raw.get(axis, 0.0)) for axis in ("x", "y", "z"))
    if isinstance(raw, Sequence) and len(raw) >= 3:
        return tuple(float(raw[index]) for index in range(3))
    return 0.0, 0.0, 0.0


def _extract_aabb(value: Mapping[str, Any]) -> dict[str, list[float]] | None:
    raw = value.get("aabb", value)
    if not isinstance(raw, Mapping):
        return None
    minimum = raw.get("min")
    maximum = raw.get("max")
    if (
        not isinstance(minimum, Sequence)
        or isinstance(minimum, (str, bytes))
        or len(minimum) < 3
        or not isinstance(maximum, Sequence)
        or isinstance(maximum, (str, bytes))
        or len(maximum) < 3
    ):
        return None
    return {
        "min": [float(minimum[index]) for index in range(3)],
        "max": [float(maximum[index]) for index in range(3)],
    }


def _sample_subject_aabb(
    sample: Mapping[str, Any],
    subject_initial: Mapping[str, Any] | None,
) -> dict[str, list[float]] | None:
    direct = _extract_aabb(sample)
    if direct is not None:
        return direct
    source = _extract_aabb(subject_initial or {})
    if source is None:
        return None
    source_center = [
        (source["min"][index] + source["max"][index]) * 0.5
        for index in range(3)
    ]
    target_center = _position(sample)
    offset = [target_center[index] - source_center[index] for index in range(3)]
    return {
        "min": [
            source["min"][index] + offset[index] for index in range(3)
        ],
        "max": [
            source["max"][index] + offset[index] for index in range(3)
        ],
    }


__all__ = [
    "RelationContract",
    "static_relation_geometry_evidence",
    "validate_reservation_subset",
]
