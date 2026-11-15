"""Simulator-free helpers for exact-scene reach and route evidence."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ENVELOPE_VERSION = "aloha_generic_xy_z_pose_class_v1"
COLLISION_CASE_VERSION = "curobo_live_world_attached_subject_v1"
DEFAULT_OUTPUT_ROOT = Path("if_benchmark/data/reach_validation")
DEFAULT_LAYOUT_CONFIG = Path("if_benchmark/data/layout/layout_config.yml")

PILOT_TASK_IDS = (
    ("ss_020ed3bc77e05aae", "task_33558177a8e79258"),
    ("ss_020ed3bc77e05aae", "task_556e419422171684"),
    ("ss_020ed3bc77e05aae", "task_9d4ee8ba0bf8a11f"),
    ("ss_020ed3bc77e05aae", "task_f1027e575d7365f5"),
    ("ss_0ab73b0e3b867843", "task_7fecbe363ef5b454"),
    ("ss_0ab73b0e3b867843", "task_e02500f9f8ca5aa1"),
    ("ss_39d9a2d8314acc54", "task_20b8b76f4d814695"),
    ("ss_04308f87c5ec27e4", "task_43db2372225e9ffd"),
    ("ss_0ab73b0e3b867843", "task_4b7da8b3395b1e2e"),
    ("ss_0ab73b0e3b867843", "task_741d2fa2daf23583"),
    ("ss_0c8ae8ab2cb4695b", "task_3324960d74115628"),
    ("ss_0c8ae8ab2cb4695b", "task_d6f622246e0150fa"),
)
HIGH_CONFIDENCE_TASK_IDS = frozenset(PILOT_TASK_IDS[:7])


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def write_temporary_report(path: Path, value: Mapping[str, Any]) -> Path:
    payload = dict(value)
    payload.pop("report_hash", None)
    payload["report_hash"] = content_hash(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _quat_from_matrix(matrix: Sequence[Sequence[float]]) -> list[float]:
    m = matrix
    trace = float(m[0][0]) + float(m[1][1]) + float(m[2][2])
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        values = [
            0.25 * scale,
            (float(m[2][1]) - float(m[1][2])) / scale,
            (float(m[0][2]) - float(m[2][0])) / scale,
            (float(m[1][0]) - float(m[0][1])) / scale,
        ]
    else:
        axis = max(range(3), key=lambda index: float(m[index][index]))
        if axis == 0:
            scale = math.sqrt(
                1.0 + float(m[0][0]) - float(m[1][1]) - float(m[2][2])
            ) * 2.0
            values = [
                (float(m[2][1]) - float(m[1][2])) / scale,
                0.25 * scale,
                (float(m[0][1]) + float(m[1][0])) / scale,
                (float(m[0][2]) + float(m[2][0])) / scale,
            ]
        elif axis == 1:
            scale = math.sqrt(
                1.0 + float(m[1][1]) - float(m[0][0]) - float(m[2][2])
            ) * 2.0
            values = [
                (float(m[0][2]) - float(m[2][0])) / scale,
                (float(m[0][1]) + float(m[1][0])) / scale,
                0.25 * scale,
                (float(m[1][2]) + float(m[2][1])) / scale,
            ]
        else:
            scale = math.sqrt(
                1.0 + float(m[2][2]) - float(m[0][0]) - float(m[1][1])
            ) * 2.0
            values = [
                (float(m[1][0]) - float(m[0][1])) / scale,
                (float(m[0][2]) + float(m[2][0])) / scale,
                (float(m[1][2]) + float(m[2][1])) / scale,
                0.25 * scale,
            ]
    norm = math.sqrt(sum(value * value for value in values))
    return [round(value / norm, 12) for value in values]


def pose_class_quaternion(pose_class: str, yaw_deg: float) -> list[float]:
    """Return a gripper frame whose +X axis is the approach direction."""

    yaw = math.radians(float(yaw_deg))
    cosine, sine = math.cos(yaw), math.sin(yaw)
    if pose_class == "top_down":
        # columns are x(approach), y(wrist yaw), z
        matrix = (
            (0.0, cosine, sine),
            (0.0, sine, -cosine),
            (-1.0, 0.0, 0.0),
        )
    elif pose_class == "best_horizontal":
        matrix = (
            (cosine, -sine, 0.0),
            (sine, cosine, 0.0),
            (0.0, 0.0, 1.0),
        )
    else:
        raise ValueError(f"unknown pose class: {pose_class!r}")
    return _quat_from_matrix(matrix)


def inclusive_axis(lower: float, upper: float, step: float) -> list[float]:
    if step <= 0.0 or upper < lower:
        raise ValueError("invalid deterministic grid axis")
    count = int(math.floor((upper - lower) / step + 1e-9))
    values = [round(lower + index * step, 12) for index in range(count + 1)]
    if values[-1] < upper - 1e-9:
        values.append(round(upper, 12))
    return values


def normalized_joint_margin(
    solution: Sequence[float], limits: Sequence[Sequence[float]]
) -> float:
    margins = []
    for value, pair in zip(solution, limits):
        lower, upper = float(pair[0]), float(pair[1])
        span = upper - lower
        if span <= 1e-12:
            continue
        margins.append(min(float(value) - lower, upper - float(value)) / span)
    return round(min(margins, default=0.0), 12)


def deterministic_terminal_offsets() -> list[tuple[float, float]]:
    """Small bounded local domain; it is search evidence, not a reservation."""

    offsets = [(0.0, 0.0)]
    for radius in (0.02, 0.04, 0.05):
        offsets.extend(
            [
                (-radius, 0.0),
                (radius, 0.0),
                (0.0, -radius),
                (0.0, radius),
                (-radius, -radius),
                (-radius, radius),
                (radius, -radius),
                (radius, radius),
            ]
        )
    return offsets


def rejection_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        reason = str(row.get("rejection_reason") or "")
        if reason:
            counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))
