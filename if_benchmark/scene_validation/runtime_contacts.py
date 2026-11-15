"""M15 live contact observation with one validation and ordered aggregation.

The generic public SuccessContract observer remains the reference. This adapter
consumes live SAPIEN points directly, retaining its endpoint filters, vector
validation, point order, per-key impulse addition order and canonical output.
No contact history or physical observation is cached.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

from if_benchmark.schema.serialization import canonical_data
from if_benchmark.success_contract.contract import SuccessContract
from if_benchmark.success_contract.observation import (
    SuccessObservationError,
    _body_name,
    _vector,
)


def observe_current_sapien_contacts(
    scene: Any,
    *,
    endpoint_by_body_name: Mapping[str, str],
    embodiment_profile: Mapping[str, Any],
    grippers: Mapping[str, Any],
    contract: SuccessContract,
) -> list[dict[str, Any]]:
    """Read each live manifold once and normalize each vector exactly once."""
    endpoints = {str(key): str(value) for key, value in endpoint_by_body_name.items()}
    links: dict[str, dict[str, Any]] = {}
    for arm, arm_profile in (embodiment_profile.get("arms") or {}).items():
        for row in arm_profile.get("fingers") or []:
            links[str(row["link"])] = {
                **canonical_data(dict(row)), "arm": str(arm), "region": "finger",
            }
        for name in arm_profile.get("palm_wrist_links") or []:
            links[str(name)] = {"arm": str(arm), "region": "palm_wrist"}
    centers = {
        arm: [
            sum(float(pose["position"][index]) for pose in state["link_poses"].values())
            / len(state["link_poses"])
            for index in range(3)
        ]
        for arm, state in grippers.items()
    }
    grouped: dict[tuple[str, ...], dict[str, Any]] = {}
    vectors: dict[tuple[str, ...], list[float]] = {}
    for contact in scene.get_contacts():
        names = [_body_name(body) for body in getattr(contact, "bodies", [])]
        if len(names) != 2 or any(name not in endpoints for name in names):
            continue
        if endpoints[names[0]] == endpoints[names[1]]:
            continue
        gripper_index = next(
            (index for index, name in enumerate(names) if name in links), None
        )
        link = None if gripper_index is None else links[names[gripper_index]]
        region = "entity_contact" if link is None else str(link["region"])
        origin = inward = None
        # A contact can contain both inner/outer points. Local lookup avoids
        # rebuilding global keys per point while preserving each key's order.
        contact_groups = {}
        for point in contact.points:
            position = _vector(point.position, "position")
            point_region = region
            if region == "finger":
                if origin is None:
                    arm = str(link["arm"])
                    origin = grippers[arm]["link_poses"][names[gripper_index]]["position"]
                    inward = [centers[arm][i] - float(origin[i]) for i in range(3)]
                local = [position[i] - float(origin[i]) for i in range(3)]
                point_region = (
                    "inner_pad"
                    if sum(a * b for a, b in zip(inward, local)) >= 0.0
                    else "outer_finger"
                )
            normal = _vector(point.normal, "normal")
            impulse = _vector(point.impulse, "impulse")
            separation = float(point.separation)
            target = contact_groups.get(point_region)
            if target is None:
                left, right = endpoints[names[0]], endpoints[names[1]]
                if not left or not right:
                    raise SuccessObservationError("contact manifold endpoints missing")
                key = (
                    left, right, names[0], names[1],
                    "" if link is None else str(link.get("arm") or ""),
                    "" if link is None else str(link.get("finger") or ""),
                    str(point_region or ""),
                )
                row = grouped.get(key)
                if row is None:
                    # All scalar/vector values below are already normalized
                    # and newly owned; spelling canonical key order avoids a
                    # redundant JSON encode/decode of every contact point.
                    row = {
                        "arm": key[4] or None, "entity_a": left,
                        "entity_b": right, "finger": key[5] or None,
                        "link_a": key[2] or None, "link_b": key[3] or None,
                        "point_count": 0, "points": [],
                        "region": key[6] or None, "summed_impulse": None,
                        "summed_impulse_norm": 0.0,
                    }
                    grouped[key] = row
                    vectors[key] = [0.0, 0.0, 0.0]
                target = (row["points"], vectors[key])
                contact_groups[point_region] = target
            points, summed = target
            # Identical point iteration and scalar additions to the reference;
            # do not replace this with a NumPy/batched reduction.
            for index, value in enumerate(impulse):
                summed[index] += value
            points.append({"impulse": impulse, "normal": normal,
                           "position": position, "separation": separation})
    epsilon = float(contract.section("active_contact")["impulse_epsilon"])
    active = []
    for key in sorted(grouped):
        summed = vectors[key]
        norm = math.sqrt(sum(value * value for value in summed))
        if norm <= epsilon:
            continue
        row = grouped[key]
        row["summed_impulse"] = summed
        row["summed_impulse_norm"] = norm
        row["point_count"] = len(row["points"])
        active.append(row)
    return active
