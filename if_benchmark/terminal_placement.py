"""Shared frozen-recipe terminal construction and runtime bindings.

M5, the post-M13 route gate, M15 execution, and the terminal SuccessContract
must agree on the same placement recipe.  This module is the deliberately
small boundary between the simulator-independent recipe/profile contract and
the live articulated receiver link frame.
"""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from if_benchmark.articulated_pair_validation import (
    load_receiver_profiles,
    validate_receiver_profile,
)
from if_benchmark.articulated_terminal_contract import cavity_witness_world_pose
from if_benchmark.pair_recipe import validate_terminal_placement_recipe
from if_benchmark.schema import artifact_hash
from if_benchmark.schema.serialization import canonical_data


DEFAULT_RECEIVER_PROFILES_PATH = (
    Path(__file__).resolve().parent
    / "assets"
    / "articulated_receiver_profiles.yml"
)
PAIR_RECIPE_SOURCES = frozenset(
    {
        "frozen_m5_projection",
        "deterministic_m5_report_v3_recipe_rebind_v2",
    }
)


def load_reviewed_receiver_profiles(
    path: Path = DEFAULT_RECEIVER_PROFILES_PATH,
) -> dict[str, Any]:
    """Load the reviewed profile mapping, excluding catalog metadata."""

    catalog = load_receiver_profiles(path)
    profiles = catalog.get("profiles")
    if not isinstance(profiles, Mapping):
        raise ValueError("articulated receiver profile catalog is malformed")
    return canonical_data(dict(profiles))


def terminal_recipe_binding(
    region: Mapping[str, Any],
    relation: str,
    *,
    receiver_profiles: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate one M11 recipe and return its exact immutable lineage."""

    recipe = region.get("placement_recipe")
    source = str(region.get("placement_recipe_source") or "")
    if not isinstance(recipe, Mapping) or not source:
        raise ValueError("terminal reservation lacks frozen placement recipe")
    raw = canonical_data(dict(recipe))
    if (
        str(relation) not in {"in", "on", "stack"}
        and raw.get("constructor") is None
    ):
        clearance = raw.get("clearance_m")
        if (
            source != "relation_default"
            or raw.get("policy") != "m11_non_pair_terminal_clearance_v1"
            or isinstance(clearance, bool)
            or not isinstance(clearance, (int, float))
            or not math.isfinite(float(clearance))
            or float(clearance) < 0.0
        ):
            raise ValueError("non-pair terminal recipe is invalid")
        return canonical_data(
            {
                "placement_recipe": raw,
                "placement_recipe_source": source,
                "placement_recipe_hash": artifact_hash(raw),
                "receiver_state_profile": None,
                "receiver_state_profile_id": None,
                "receiver_state_profile_hash": None,
            }
        )
    if source not in PAIR_RECIPE_SOURCES:
        raise ValueError("pair terminal recipe source is not trusted")
    profile: Mapping[str, Any] | None = None
    if raw.get("constructor") == "articulated_place_in_cavity_link":
        profiles = (
            load_reviewed_receiver_profiles()
            if receiver_profiles is None
            else receiver_profiles
        )
        profile_id = str(raw.get("receiver_state_profile_id") or "")
        candidate = profiles.get(profile_id)
        if not isinstance(candidate, Mapping):
            raise ValueError(
                f"reviewed articulated receiver profile is missing: {profile_id}"
            )
        blockers = validate_receiver_profile(profile_id, dict(candidate))
        if blockers:
            raise ValueError(
                "reviewed articulated receiver profile is invalid: "
                + ",".join(blockers)
            )
        profile = canonical_data(dict(candidate))
    recipe_relation = {
        "in": "place_in",
        "on": "place_on",
        "stack": "stack",
    }.get(str(relation), str(relation))
    validated = validate_terminal_placement_recipe(
        raw,
        recipe_relation,
        expected_recipe_id=(
            str(raw["recipe_id"]) if raw.get("recipe_id") else None
        ),
        reviewed_receiver_profile=profile,
    )
    recipe_hash = str(validated.get("recipe_hash") or artifact_hash(validated))
    return canonical_data(
        {
            "placement_recipe": validated,
            "placement_recipe_source": source,
            "placement_recipe_hash": recipe_hash,
            "receiver_state_profile": profile,
            "receiver_state_profile_id": (
                str(validated.get("receiver_state_profile_id") or "") or None
            ),
            "receiver_state_profile_hash": (
                str(validated.get("receiver_state_profile_hash") or "") or None
            ),
        }
    )


def articulated_cavity_world_matrix(
    receiver: Any,
    reviewed_receiver_profile: Mapping[str, Any],
) -> np.ndarray:
    """Resolve the reviewed cavity frame from the receiver's live moving link."""

    cavity = reviewed_receiver_profile.get("cavity") or {}
    link_name = str(cavity.get("frame_link") or "")
    link_dict = getattr(receiver, "link_dict", None)
    if not isinstance(link_dict, Mapping) or link_name not in link_dict:
        raise ValueError(f"missing_cavity_frame_link:{link_name}")
    link = link_dict[link_name]
    pose = link.get_pose()
    matrix = np.asarray(pose.to_transformation_matrix(), dtype=float)
    frame_transform = cavity.get("frame_transform")
    if frame_transform is not None:
        matrix = matrix @ np.asarray(frame_transform, dtype=float)
    validate_se3(matrix, "articulated cavity world matrix")
    return matrix


def articulated_terminal_actor_pose(
    *,
    cavity_world_matrix: Sequence[Sequence[float]],
    reviewed_receiver_profile: Mapping[str, Any],
    placement_recipe: Mapping[str, Any],
    subject_q0_wxyz: Sequence[float],
    subject_local_bottom_z: float,
    witness_delta_deg: float,
    offset_xy_m: Sequence[float] = (0.0, 0.0),
) -> list[float]:
    """Construct the exact cavity-local terminal root pose used by every stage."""

    matrix = validate_se3(
        cavity_world_matrix, "articulated cavity world matrix"
    )
    offset = np.asarray(offset_xy_m, dtype=float)
    if offset.shape != (2,) or not np.isfinite(offset).all():
        raise ValueError("articulated terminal offset must be finite xy")
    q0 = np.asarray(subject_q0_wxyz, dtype=float)
    if q0.shape != (4,) or not np.isfinite(q0).all():
        raise ValueError("articulated terminal q0 must be a finite quaternion")
    profile = deepcopy(dict(reviewed_receiver_profile))
    recipe = validate_terminal_placement_recipe(
        placement_recipe,
        "place_in",
        expected_recipe_id=str(placement_recipe.get("recipe_id") or ""),
        reviewed_receiver_profile=profile,
    )
    pose = cavity_witness_world_pose(
        cavity_world_matrix=matrix,
        cavity=profile["cavity"],
        subject_q0_wxyz=q0.tolist(),
        witness_delta_deg=float(witness_delta_deg),
        subject_local_bottom_z=float(subject_local_bottom_z),
        clearance_m=float(recipe["clearance_m"]),
    )
    position = np.asarray(pose["p"], dtype=float)
    position += matrix[:3, :2] @ offset
    return [*position.tolist(), *[float(value) for value in pose["q_wxyz"]]]


def terminal_frame_binding(
    cavity_world_matrix: Sequence[Sequence[float]],
) -> dict[str, Any]:
    """Freeze a self-hashed live cavity frame into one route witness."""

    matrix = validate_se3(cavity_world_matrix, "terminal frame")
    payload = {"matrix": matrix.round(12).tolist()}
    return {**payload, "hash": artifact_hash(payload)}


def terminal_observation_key(
    subject: str, relation: str, receiver_keys: Sequence[str]
) -> str:
    """Return the stable SuccessSnapshot key for one reserved terminal."""

    receivers = ",".join(sorted(str(value) for value in receiver_keys))
    return f"{str(subject)}|{str(relation)}|{receivers}"


def validate_terminal_frame_binding(value: Mapping[str, Any]) -> np.ndarray:
    payload = {"matrix": value.get("matrix")}
    if str(value.get("hash") or "") != artifact_hash(payload):
        raise ValueError("terminal frame binding hash mismatch")
    return validate_se3(payload["matrix"], "terminal frame binding")


def validate_se3(value: Any, field: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{field} must be a finite 4x4 matrix")
    if not np.allclose(
        matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-10, rtol=0.0
    ):
        raise ValueError(f"{field} homogeneous row is invalid")
    rotation = matrix[:3, :3]
    if not np.allclose(
        rotation.T @ rotation, np.eye(3), atol=1e-8, rtol=0.0
    ) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-8, rtol=0.0):
        raise ValueError(f"{field} rotation is not SO(3)")
    return matrix


__all__ = [
    "DEFAULT_RECEIVER_PROFILES_PATH",
    "PAIR_RECIPE_SOURCES",
    "articulated_cavity_world_matrix",
    "articulated_terminal_actor_pose",
    "load_reviewed_receiver_profiles",
    "terminal_frame_binding",
    "terminal_observation_key",
    "terminal_recipe_binding",
    "validate_se3",
    "validate_terminal_frame_binding",
]
