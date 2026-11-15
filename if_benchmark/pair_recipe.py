"""Shared, simulator-independent M5 terminal placement recipe contract."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Final, Mapping

from if_benchmark.schema.serialization import canonical_data


TERMINAL_PROBE_POLICY_VERSION: Final = (
    "m5_terminal_settle_v11_proportional_asset_mass_inertia"
)
ARTICULATED_TERMINAL_SETTLE_POLICY_VERSION: Final = (
    "m5_ar_v3_proportional_asset_mass_inertia"
)
TERMINAL_PLACEMENT_RECIPE_VERSION: Final = "m5_terminal_placement_recipe_v2"

# The former schema-2 projection allowlist is intentionally empty.  Those
# projections were produced before the proportional dynamic-asset mass/inertia
# policy and therefore cannot prove the current M5 physical contract.  Keeping
# the admission set empty also makes Assignments and Units reject the old
# projection envelopes before they can mint a current placement recipe.
LEGACY_M5_REPORT_V3_PROJECTIONS: Final = {}


def canonical_terminal_placement_recipe(relation: str) -> dict[str, Any]:
    """Return the exact deterministic Terminal recipe for one relation."""

    normalized = str(relation)
    clearance_m = (
        0.03 if normalized in {"place_in", "dump_into"} else 0.008
    )
    return {
        "clearance_m": clearance_m,
        "yaw_convention": "subject_yaw_minus_receiver_yaw",
        "settle_policy": TERMINAL_PROBE_POLICY_VERSION,
    }


def _payload_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        canonical_data(dict(value)),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def receiver_state_profile_hash(
    profile_id: str, profile: Mapping[str, Any]
) -> str:
    return _payload_hash(
        {
            "receiver_state_profile_id": str(profile_id),
            "profile": dict(profile),
        }
    )


def canonical_articulated_terminal_placement_recipe(
    *,
    profile_id: str,
    profile: Mapping[str, Any],
    band_id: str,
) -> dict[str, Any]:
    cavity = profile.get("cavity") or {}
    payload = {
        "recipe_id": (
            "m5_ar_place_in_cavity_link_v1:"
            f"{profile_id}:{band_id}"
        ),
        "constructor": "articulated_place_in_cavity_link",
        "clearance_m": float(profile.get("terminal_clearance_m") or 0.008),
        "yaw_convention": "subject_yaw_minus_receiver_yaw",
        "settle_policy": ARTICULATED_TERMINAL_SETTLE_POLICY_VERSION,
        "receiver_state_profile_id": str(profile_id),
        "receiver_state_profile_hash": receiver_state_profile_hash(
            profile_id, profile
        ),
        "cavity_frame_link": str(cavity["frame_link"]),
        "support_plane_z": float(cavity["support_plane_z"]),
    }
    return canonical_data({**payload, "recipe_hash": _payload_hash(payload)})


def validate_terminal_placement_recipe(
    recipe: Mapping[str, Any],
    relation: str,
    *,
    expected_recipe_id: str | None = None,
    reviewed_receiver_profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and normalize one frozen M5 placement recipe.

    Ordinary M5 relations are deterministic and therefore require the exact
    canonical payload.  State-conditioned articulated receivers carry an
    explicit constructor payload instead; it is accepted only when all fields
    that determine the terminal pose and settling semantics are present.
    """

    normalized = canonical_data(dict(recipe))
    clearance = normalized.get("clearance_m")
    if (
        isinstance(clearance, bool)
        or not isinstance(clearance, (int, float))
        or not math.isfinite(float(clearance))
        or float(clearance) < 0.0
    ):
        raise ValueError("terminal placement recipe has invalid clearance_m")
    if normalized.get("yaw_convention") != (
        "subject_yaw_minus_receiver_yaw"
    ):
        raise ValueError("terminal placement recipe has invalid yaw convention")
    constructor = normalized.get("constructor")
    if constructor is None:
        if expected_recipe_id is not None and not str(
            expected_recipe_id
        ).startswith("m5_recipe_"):
            raise ValueError("terminal placement recipe has invalid recipe id")
        expected = canonical_terminal_placement_recipe(relation)
        if normalized != expected:
            raise ValueError(
                "terminal placement recipe differs from canonical relation policy"
            )
        return canonical_data(expected)

    if str(relation) != "place_in" or constructor != (
        "articulated_place_in_cavity_link"
    ):
        raise ValueError("unsupported state-conditioned terminal recipe")
    recipe_id = normalized.get("recipe_id")
    if not isinstance(recipe_id, str) or not recipe_id.startswith(
        "m5_ar_place_in_cavity_link_v1:"
    ):
        raise ValueError("articulated terminal recipe has invalid recipe id")
    if expected_recipe_id is not None and recipe_id != str(expected_recipe_id):
        raise ValueError("articulated terminal recipe id mismatch")
    if normalized.get("settle_policy") != (
        ARTICULATED_TERMINAL_SETTLE_POLICY_VERSION
    ):
        raise ValueError("articulated terminal recipe has stale settle policy")
    for field in (
        "receiver_state_profile_id",
        "receiver_state_profile_hash",
        "cavity_frame_link",
        "recipe_hash",
    ):
        if not isinstance(normalized.get(field), str) or not normalized[field]:
            raise ValueError(f"articulated terminal recipe lacks {field}")
    support_plane_z = normalized.get("support_plane_z")
    if (
        isinstance(support_plane_z, bool)
        or not isinstance(support_plane_z, (int, float))
        or not math.isfinite(float(support_plane_z))
    ):
        raise ValueError(
            "articulated terminal recipe has invalid support_plane_z"
        )
    unhashed = {
        key: value for key, value in normalized.items() if key != "recipe_hash"
    }
    if normalized["recipe_hash"] != _payload_hash(unhashed):
        raise ValueError("articulated terminal recipe hash mismatch")
    if reviewed_receiver_profile is not None:
        profile_id = str(normalized["receiver_state_profile_id"])
        band_id = recipe_id.rsplit(":", 1)[-1]
        expected = canonical_articulated_terminal_placement_recipe(
            profile_id=profile_id,
            profile=reviewed_receiver_profile,
            band_id=band_id,
        )
        if normalized != expected:
            raise ValueError(
                "articulated terminal recipe differs from reviewed profile"
            )
    return normalized


def rebind_legacy_terminal_placement_recipe(
    row: Mapping[str, Any],
    relation: str,
    *,
    stage: str,
    projection_schema_version: int,
    projection_stage: str,
    projection_sha256: str | None,
    source_ledger_hash: str | None,
    receiver_profiles: Mapping[str, Any],
) -> dict[str, Any]:
    """Reject pre-policy M5 projections at the former migration boundary.

    Schema-2 projections predate the proportional mass/inertia physical
    contract.  Reconstructing a current recipe from them would relabel old
    physical evidence, so the compatibility boundary is now fail-closed.
    """

    if not LEGACY_M5_REPORT_V3_PROJECTIONS:
        raise ValueError(
            "legacy M5 physical projection rebind is disabled under the "
            "proportional mass/inertia policy"
        )

    expected_status = {
        "terminal_settle": "validated_terminal",
        "canonical_skill": "validated_skill",
    }.get(str(stage))
    allowed_projection = LEGACY_M5_REPORT_V3_PROJECTIONS.get(str(stage))
    if (
        not allowed_projection
        or str(projection_sha256 or "")
        != str(allowed_projection["sha256"])
        or str(source_ledger_hash or "")
        != str(allowed_projection["source_ledger_hash"])
        or projection_schema_version != 2
        or str(projection_stage) != str(stage)
        or str(row.get("stage") or "") != str(stage)
        or str(row.get("status") or "") != str(expected_status or "")
    ):
        raise ValueError("row is not an eligible legacy M5 projection")
    if not all(
        isinstance(row.get(field), str) and bool(row.get(field))
        for field in ("pair_key", "evidence_id", "evidence_path", "config_hash")
    ):
        raise ValueError("legacy M5 projection lacks evidence provenance")

    recipe_field = (
        "validated_skill_recipe_id"
        if stage == "canonical_skill"
        else "validated_recipe_id"
    )
    band_field = (
        "validated_skill_band_ids"
        if stage == "canonical_skill"
        else "validated_band_ids"
    )
    recipe_id = str(row.get(recipe_field) or "")
    band_ids = [str(value) for value in row.get(band_field) or []]
    articulated = recipe_id.startswith("m5_ar_place_in_cavity_link_v1:")
    rigid = recipe_id.startswith("m5_recipe_")
    if not (rigid or articulated) or not band_ids:
        raise ValueError(
            "legacy unidentified M5 recipe cannot be rebound"
        )
    profile_id = ""
    reviewed_profile: Mapping[str, Any] | None = None
    if articulated:
        parts = recipe_id.split(":")
        if len(parts) != 3 or str(relation) != "place_in":
            raise ValueError("legacy articulated M5 recipe id is malformed")
        profile_id = parts[1]
        if f"|state:{profile_id}" not in str(row.get("pair_key") or ""):
            raise ValueError("legacy articulated M5 pair/profile mismatch")
        reviewed_profile = receiver_profiles.get(profile_id)
        if not isinstance(reviewed_profile, Mapping):
            raise ValueError("legacy articulated M5 profile is unavailable")
        live_profile_hash = receiver_state_profile_hash(
            profile_id, reviewed_profile
        )
        row_profile_id = row.get("receiver_state_profile_id")
        row_profile_hash = row.get("receiver_state_profile_hash")
        if row_profile_id is not None and str(row_profile_id) != profile_id:
            raise ValueError("legacy articulated M5 profile id changed")
        if row_profile_hash is not None and str(row_profile_hash) != (
            live_profile_hash
        ):
            raise ValueError("legacy articulated M5 profile hash changed")
    elif "|state:" in str(row.get("pair_key") or ""):
        raise ValueError("legacy rigid recipe cannot bind a state-conditioned pair")

    if stage == "terminal_settle":
        passed_rows = [
            result
            for result in (row.get("metrics") or {}).get("band_results") or []
            if isinstance(result, Mapping)
            and result.get("pass") is True
            and str(result.get("recipe_id") or "") == recipe_id
            and bool(
                {
                    str(value)
                    for value in (
                        (result.get("yaw_witness") or {}).get(
                            "covered_band_ids"
                        )
                        or [result.get("band_id")]
                    )
                }
                & set(band_ids)
            )
        ]
        if not passed_rows:
            raise ValueError(
                "legacy M5 projection does not bind a passing recipe witness: "
                f"{row.get('pair_key')}"
            )
    else:
        preconditions = row.get("skill_preconditions") or {}
        if (
            str(preconditions.get("terminal_recipe_id") or "") != recipe_id
            or not preconditions.get("terminal_evidence_id")
            or not row.get("skill_probe_policy_hash")
        ):
            raise ValueError(
                "legacy M5 skill projection lacks terminal recipe lineage: "
                f"{row.get('pair_key')}"
            )

    if articulated:
        assert reviewed_profile is not None
        return validate_terminal_placement_recipe(
            canonical_articulated_terminal_placement_recipe(
                profile_id=profile_id,
                profile=reviewed_profile,
                band_id=recipe_id.rsplit(":", 1)[-1],
            ),
            relation,
            expected_recipe_id=recipe_id,
            reviewed_receiver_profile=reviewed_profile,
        )
    return validate_terminal_placement_recipe(
        canonical_terminal_placement_recipe(relation),
        relation,
        expected_recipe_id=recipe_id,
    )


__all__ = [
    "ARTICULATED_TERMINAL_SETTLE_POLICY_VERSION",
    "LEGACY_M5_REPORT_V3_PROJECTIONS",
    "TERMINAL_PLACEMENT_RECIPE_VERSION",
    "TERMINAL_PROBE_POLICY_VERSION",
    "canonical_articulated_terminal_placement_recipe",
    "canonical_terminal_placement_recipe",
    "rebind_legacy_terminal_placement_recipe",
    "receiver_state_profile_hash",
    "validate_terminal_placement_recipe",
]
