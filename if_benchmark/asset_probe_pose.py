from __future__ import annotations

import hashlib
import json
from typing import Any


ROOT_QPOS_IDENTITY = [1.0, 0.0, 0.0, 0.0]
ROOT_QPOS_ROBOTWIN_OBJECT = [0.5, 0.5, 0.5, 0.5]
ROOT_QPOS_X_UP = [0.707, 0.707, 0.0, 0.0]
ROOT_QPOS_SCANNER = [0.0, 0.0, 0.707, 0.707]
ROOT_QPOS_BOTTLE_SIDE = [0.707, 0.0, 0.0, 0.707]
CANONICAL_POSE_POLICY_VERSION = "canonical_pose_q0_yaw_v2_reviewed_bottle"

FALLBACK_ALLOWED_HARD_METADATA_USES = {
    "pick_left",
    "pick_right",
    "handover",
    "bimanual_grab_subject",
}


# These hints mirror the original RoboTwin task setup code under envs/. They
# provide a shared canonical q0 for asset probes, pair geometry, terminal
# settle, and future scene realization.
ASSET_POSE_HINTS: dict[str, dict[str, Any]] = {
    "001_bottle": {
        "qpos": ROOT_QPOS_ROBOTWIN_OBJECT,
        "source": "if_benchmark/data/attribute_catalog/001_bottle/contact_sheet.png",
    },
    "007_shoe-box": {"qpos": [0.5, 0.5, -0.5, -0.5], "source": "envs/place_dual_shoes.py"},
    "006_hamburg": {"qpos": ROOT_QPOS_ROBOTWIN_OBJECT, "source": "envs/place_burger_fries.py"},
    "018_microphone": {"qpos": ROOT_QPOS_X_UP, "source": "envs/handover_mic.py"},
    "020_hammer": {"qpos": [0.0, 0.0, 0.995, 0.105], "z": 0.783, "source": "envs/beat_block_hammer.py"},
    "021_cup": {"qpos": ROOT_QPOS_ROBOTWIN_OBJECT, "source": "envs/place_empty_cup.py"},
    "023_tissue-box": {"qpos": ROOT_QPOS_ROBOTWIN_OBJECT, "source": "box-like RoboTwin object convention"},
    "024_scanner": {"qpos": ROOT_QPOS_SCANNER, "source": "envs/scan_object.py"},
    "039_mug": {"qpos": ROOT_QPOS_X_UP, "source": "envs/hanging_mug.py"},
    "047_mouse": {"qpos": ROOT_QPOS_ROBOTWIN_OBJECT, "source": "envs/place_a2b_left.py"},
    "048_stapler": {"qpos": ROOT_QPOS_ROBOTWIN_OBJECT, "source": "envs/place_a2b_left.py"},
    "050_bell": {"qpos": ROOT_QPOS_ROBOTWIN_OBJECT, "source": "envs/place_a2b_left.py"},
    "057_toycar": {"qpos": ROOT_QPOS_ROBOTWIN_OBJECT, "source": "envs/place_a2b_left.py"},
    "062_plasticbox": {"qpos": ROOT_QPOS_ROBOTWIN_OBJECT, "source": "envs/place_cans_plasticbox.py"},
    "072_electronicscale": {"qpos": ROOT_QPOS_ROBOTWIN_OBJECT, "source": "envs/place_object_scale.py"},
    "073_rubikscube": {"qpos": ROOT_QPOS_ROBOTWIN_OBJECT, "source": "envs/place_a2b_left.py"},
    "075_bread": {"qpos": ROOT_QPOS_ROBOTWIN_OBJECT, "source": "envs/place_a2b_left.py"},
    "081_playingcards": {"qpos": ROOT_QPOS_ROBOTWIN_OBJECT, "source": "envs/place_a2b_left.py"},
    "086_woodenblock": {"qpos": ROOT_QPOS_ROBOTWIN_OBJECT, "source": "envs/place_a2b_left.py"},
    "107_soap": {"qpos": ROOT_QPOS_ROBOTWIN_OBJECT, "source": "envs/place_a2b_left.py"},
    "112_tea-box": {"qpos": ROOT_QPOS_ROBOTWIN_OBJECT, "source": "envs/scan_object.py"},
    "113_coffee-box": {"qpos": ROOT_QPOS_ROBOTWIN_OBJECT, "source": "envs/place_a2b_left.py"},
}


MODEL_POSE_HINTS: dict[str, dict[str, dict[str, Any]]] = {
    "077_phone": {
        "0": {"qpos": [0.707, 0.707, 0.0, 0.0], "source": "envs/place_phone_stand.py"},
        "1": {"qpos": ROOT_QPOS_ROBOTWIN_OBJECT, "source": "envs/place_phone_stand.py"},
        "2": {"qpos": [0.5, 0.5, -0.5, -0.5], "source": "envs/place_phone_stand.py"},
        "4": {"qpos": [0.5, -0.5, 0.5, -0.5], "source": "envs/place_phone_stand.py"},
    }
}


CATEGORY_POSE_HINTS: dict[str, dict[str, Any]] = {
    "cup": {"qpos": ROOT_QPOS_ROBOTWIN_OBJECT, "source": "category_hint:cup"},
    "package": {"qpos": ROOT_QPOS_ROBOTWIN_OBJECT, "source": "category_hint:package"},
    "tissue_box": {"qpos": ROOT_QPOS_ROBOTWIN_OBJECT, "source": "category_hint:tissue_box"},
}


def candidate_pose_defaults(candidate: dict[str, Any], use: str) -> dict[str, Any]:
    physical_defaults = candidate.get("physical_defaults") or {}
    if physical_defaults.get("upright_qpos"):
        return _pose_defaults(
            qpos=physical_defaults["upright_qpos"],
            z=physical_defaults.get("z"),
            source="physical_defaults.upright_qpos",
        )
    drive = candidate.get("articulation_drive") or {}
    if drive.get("root_qpos"):
        return _pose_defaults(qpos=drive["root_qpos"], z=drive.get("z"), source="articulation_drive.root_qpos")
    if use in {"open", "close", "switch_on", "switch_off"}:
        return _pose_defaults(qpos=ROOT_QPOS_IDENTITY, source="articulation_default")

    asset = str(candidate.get("asset") or "")
    model_id = str(candidate.get("model_id") or "")
    model_hint = (MODEL_POSE_HINTS.get(asset) or {}).get(model_id)
    if model_hint:
        return _pose_defaults(**model_hint)

    asset_hint = ASSET_POSE_HINTS.get(asset)
    if asset_hint:
        return _pose_defaults(**asset_hint)

    category_hint = CATEGORY_POSE_HINTS.get(str(candidate.get("category") or ""))
    if category_hint:
        return _pose_defaults(**category_hint)

    extents = ((candidate.get("metadata") or {}).get("scaled_extents") or [0.0, 0.0, 0.0])
    if (
        len(extents) == 3
        and float(extents[1]) > float(extents[0]) * 1.2
        and float(extents[1]) > float(extents[2]) * 1.2
    ):
        return _pose_defaults(qpos=ROOT_QPOS_ROBOTWIN_OBJECT, source="scaled_extents.y_major")
    if len(extents) == 3 and all(float(value) > 0 for value in extents):
        return _pose_defaults(qpos=ROOT_QPOS_IDENTITY, source="scaled_extents.canonical_identity")
    return _pose_defaults(qpos=ROOT_QPOS_IDENTITY, source="fallback_identity")


def resolve_canonical_pose(
    candidate: dict[str, Any],
    use: str,
    *,
    relation: str | None = None,
    role: str | None = None,
) -> dict[str, Any]:
    """Resolve the gravity-aligned q0 plus the allowed world-yaw contract."""

    defaults = candidate_pose_defaults(candidate, use)
    source = str(defaults.get("source") or "")
    resolved = source != "fallback_identity"
    fixed_relation = relation in {"hang_on", "dump_into", "stamp_on"}
    yaw = (
        {"mode": "fixed", "yaw_deg": 0.0}
        if fixed_relation
        else {"mode": "free", "period_deg": 360.0}
    )
    return {
        "q0_wxyz": [float(value) for value in defaults["qpos"]],
        "z": float(defaults["z"]),
        "source": source,
        "policy_hash": canonical_pose_policy_hash(),
        "resolved": bool(resolved),
        "yaw": yaw,
        "use": str(use),
        "role": role,
    }


def canonical_pose_policy_hash() -> str:
    payload = {
        "version": CANONICAL_POSE_POLICY_VERSION,
        "asset_hints": ASSET_POSE_HINTS,
        "model_hints": MODEL_POSE_HINTS,
        "category_hints": CATEGORY_POSE_HINTS,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def candidate_pose_options(
    candidate: dict[str, Any],
    use: str,
    *,
    purpose: str,
) -> list[dict[str, Any]]:
    """Return heuristic alternatives only for a new reviewed revision.

    BenchmarkV1 workers must consume one exact approved q0.  Requiring an
    explicit purpose prevents this legacy proposal helper from being called as
    an M4 runtime retry mechanism.
    """

    if purpose != "initial_pose_revision_proposal":
        raise ValueError(
            "candidate pose options are restricted to Initial Pose revision proposals"
        )
    primary = candidate_pose_defaults(candidate, use)
    options = [primary]
    if use in {"open", "close", "switch_on", "switch_off"}:
        return options
    if _pose_is_hard_metadata(primary) and use not in FALLBACK_ALLOWED_HARD_METADATA_USES:
        return options

    category = str(candidate.get("category") or "")
    asset = str(candidate.get("asset") or "")
    extents = ((candidate.get("metadata") or {}).get("scaled_extents") or [0.0, 0.0, 0.0])
    z_major = len(extents) == 3 and extents[2] >= extents[0] and extents[2] >= extents[1]

    if asset == "024_scanner" or category == "scanner":
        options.extend(
            [
                _pose_defaults(qpos=ROOT_QPOS_X_UP, source="fallback:x_up"),
                _pose_defaults(qpos=ROOT_QPOS_ROBOTWIN_OBJECT, source="fallback:robotwin_object"),
            ]
        )
    elif asset == "018_microphone" or category == "microphone":
        options.extend(
            [
                _pose_defaults(qpos=ROOT_QPOS_SCANNER, source="fallback:scanner_axis"),
                _pose_defaults(qpos=ROOT_QPOS_ROBOTWIN_OBJECT, source="fallback:robotwin_object"),
            ]
        )
    elif category == "bottle":
        options.extend(
            [
                _pose_defaults(qpos=ROOT_QPOS_BOTTLE_SIDE, source="fallback:bottle_side"),
                _pose_defaults(qpos=ROOT_QPOS_SCANNER, source="fallback:scanner_axis"),
                _pose_defaults(qpos=ROOT_QPOS_X_UP, source="fallback:x_up"),
            ]
        )
    elif z_major:
        options.extend(
            [
                _pose_defaults(qpos=ROOT_QPOS_SCANNER, source="fallback:z_major_scanner_axis"),
                _pose_defaults(qpos=ROOT_QPOS_X_UP, source="fallback:z_major_x_up"),
                _pose_defaults(qpos=ROOT_QPOS_ROBOTWIN_OBJECT, source="fallback:robotwin_object"),
            ]
        )
    else:
        options.extend(
            [
                _pose_defaults(qpos=ROOT_QPOS_ROBOTWIN_OBJECT, source="fallback:robotwin_object"),
                _pose_defaults(qpos=ROOT_QPOS_SCANNER, source="fallback:scanner_axis"),
                _pose_defaults(qpos=ROOT_QPOS_X_UP, source="fallback:x_up"),
            ]
        )
    return _dedupe_pose_options(options)


def candidate_qpos(candidate: dict[str, Any], use: str) -> list[float]:
    return candidate_pose_defaults(candidate, use)["qpos"]


def _pose_is_hard_metadata(pose: dict[str, Any]) -> bool:
    return str(pose.get("source") or "") in {
        "physical_defaults.upright_qpos",
        "articulation_drive.root_qpos",
        "articulation_default",
    }


def _dedupe_pose_options(options: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[float, ...]] = set()
    for option in options:
        key = tuple(round(float(value), 6) for value in [*option["qpos"], option["z"]])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(option)
    return deduped


def _pose_defaults(qpos: list[float], source: str, z: float | None = None) -> dict[str, Any]:
    return {
        "qpos": [float(value) for value in qpos],
        "z": 0.741 if z is None else float(z),
        "source": str(source),
    }
