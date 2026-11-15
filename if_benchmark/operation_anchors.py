"""Simulator-independent operation-anchor geometry shared by M4 and M15.

Layout reachability is defined at the point the robot must manipulate, not at
an arbitrary asset origin.  Rigid metadata stores actor-local points, whereas
legacy articulated metadata stores link-local points.  This module normalizes
both formats and projects them through the exact frozen R01 pose/state while
keeping the R01 and operation-metadata hash domains explicit.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET

import numpy as np

from if_benchmark.schema import artifact_hash
from if_benchmark.schema.serialization import canonical_data


OPERATION_ANCHOR_POLICY_VERSION = "operation_anchor_projection_v2_link_fk"
_RIGID_POINT_FIELDS = {
    "grasp_contact": "contact_points_pose",
    "functional": "functional_matrix",
    "orientation": "orientation_point",
}
_ARTICULATED_POINT_FIELDS = {
    "grasp_contact": "contact_points",
    "functional": "functional_points",
    "orientation": "orientation_point",
}
_OPERATION_METADATA_FIELDS = tuple(
    sorted(
        set(_RIGID_POINT_FIELDS.values())
        | set(_ARTICULATED_POINT_FIELDS.values())
    )
)
_TABLE_APPROACH_ORDER = {
    "top_down": 0,
    "oblique_down": 1,
    "horizontal": 2,
    "upward": 3,
}
_CONTACT_TO_GRASP_FRAME = np.asarray(
    [
        [0.0, 0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=float,
)


def _approach_direction(contact_matrix: np.ndarray) -> np.ndarray:
    direction = (contact_matrix @ _CONTACT_TO_GRASP_FRAME)[:3, 0]
    norm = float(np.linalg.norm(direction))
    if not math.isclose(norm, 1.0, abs_tol=5e-3):
        raise ValueError("Contact approach axis must be unit length")
    return direction


def _classify_approach(direction: Sequence[float]) -> str:
    z = float(direction[2])
    if z <= -0.85:
        return "top_down"
    if z <= -0.30:
        return "oblique_down"
    if z < 0.30:
        return "horizontal"
    return "upward"


def _rotation_matrix(q_wxyz: Sequence[float]) -> np.ndarray:
    q = np.asarray(q_wxyz, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError("q0_wxyz must contain four finite values")
    norm = float(np.linalg.norm(q))
    if norm <= 0.0:
        raise ValueError("q0_wxyz must have non-zero norm")
    w, x, y, z = q / norm
    return np.asarray(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=float,
    )


def _scale_xyz(
    model_data: Mapping[str, Any],
    exact_scale_xyz: Sequence[float] | None = None,
) -> np.ndarray:
    """Return exact runtime scale, normalizing legacy uniform scalars."""

    value = (
        exact_scale_xyz
        if exact_scale_xyz is not None
        else model_data.get("scale")
    )
    if value is None:
        raise ValueError("operation-anchor scale is missing")
    scale = np.asarray(value, dtype=float)
    if scale.shape in {(), (1,)}:
        scale = np.repeat(scale.reshape(1), 3)
    if (
        scale.shape != (3,)
        or not np.isfinite(scale).all()
        or np.any(scale <= 0.0)
    ):
        raise ValueError(
            "operation-anchor scale must contain three positive finite values"
        )
    return scale


def _xyz_rpy_transform(element: ET.Element | None, scale: float) -> np.ndarray:
    matrix = np.eye(4, dtype=float)
    if element is None:
        return matrix
    xyz = np.asarray(
        [float(value) for value in element.get("xyz", "0 0 0").split()],
        dtype=float,
    )
    rpy = [float(value) for value in element.get("rpy", "0 0 0").split()]
    if xyz.shape != (3,) or len(rpy) != 3:
        raise ValueError("URDF joint origin must contain xyz/rpy triples")
    roll, pitch, yaw = rpy
    cx, sx = math.cos(roll), math.sin(roll)
    cy, sy = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(yaw), math.sin(yaw)
    matrix[:3, :3] = np.asarray(
        [
            [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
            [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
            [-sy, cy * sx, cy * cx],
        ],
        dtype=float,
    )
    matrix[:3, 3] = xyz * float(scale)
    return matrix


def _axis_motion(joint_type: str, axis: np.ndarray, value: float) -> np.ndarray:
    matrix = np.eye(4, dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12:
        raise ValueError("URDF joint axis must have non-zero norm")
    axis = axis / norm
    if joint_type == "prismatic":
        # R01 stores SAPIEN's runtime coordinate in metres.  URDF loader scale
        # has already been applied to this qpos; multiplying it again would
        # reproduce the Cabinet 46653 coordinate bug.
        matrix[:3, 3] = axis * float(value)
        return matrix
    if joint_type not in {"revolute", "continuous"}:
        return matrix
    x, y, z = axis
    cosine = math.cos(float(value))
    sine = math.sin(float(value))
    cross = 1.0 - cosine
    matrix[:3, :3] = np.asarray(
        [
            [
                cosine + x * x * cross,
                x * y * cross - z * sine,
                x * z * cross + y * sine,
            ],
            [
                y * x * cross + z * sine,
                cosine + y * y * cross,
                y * z * cross - x * sine,
            ],
            [
                z * x * cross - y * sine,
                z * y * cross + x * sine,
                cosine + z * z * cross,
            ],
        ],
        dtype=float,
    )
    return matrix


def _articulated_link_transforms(
    urdf_path: Path,
    articulated_joint_state: Mapping[str, Any],
    scale: np.ndarray,
) -> tuple[dict[str, np.ndarray], set[str], str]:
    """Resolve URDF link frames at the exact frozen SAPIEN joint state."""

    if not np.allclose(
        scale, np.repeat(scale[0], 3), atol=1e-12, rtol=0.0
    ):
        raise ValueError("articulated URDF scale must be uniform")
    try:
        payload = urdf_path.read_bytes()
        root = ET.fromstring(payload)
    except (OSError, ET.ParseError) as exc:
        raise ValueError(
            f"cannot parse articulated URDF {urdf_path}: {exc}"
        ) from exc

    qpos = np.asarray(articulated_joint_state.get("qpos") or [], dtype=float)
    state_joints = list(articulated_joint_state.get("joints") or [])
    if (
        qpos.ndim != 1
        or not np.isfinite(qpos).all()
        or len(state_joints) != len(qpos)
    ):
        raise ValueError("articulated R01 qpos/joint metadata is inconsistent")
    state_by_name: dict[str, tuple[int, str]] = {}
    for row in state_joints:
        index = int(row.get("joint_index", -1))
        name = str(row.get("joint_name") or "")
        joint_type = str(row.get("joint_type") or "")
        if (
            not name
            or index < 0
            or index >= len(qpos)
            or name in state_by_name
        ):
            raise ValueError("articulated R01 joint identity is invalid")
        state_by_name[name] = (index, joint_type)
    if {index for index, _ in state_by_name.values()} != set(range(len(qpos))):
        raise ValueError("articulated R01 joint indices must cover qpos exactly")

    links = {str(link.get("name") or "") for link in root.findall("link")}
    if not links or "" in links:
        raise ValueError("articulated URDF has invalid link identities")
    joints: list[dict[str, Any]] = []
    children: set[str] = set()
    active_urdf_names: set[str] = set()
    for node in root.findall("joint"):
        parent_node = node.find("parent")
        child_node = node.find("child")
        if parent_node is None or child_node is None:
            raise ValueError("articulated URDF joint lacks parent/child")
        name = str(node.get("name") or "")
        joint_type = str(node.get("type") or "fixed")
        parent = str(parent_node.get("link") or "")
        child = str(child_node.get("link") or "")
        if not name or parent not in links or child not in links:
            raise ValueError(
                "articulated URDF joint references an unknown link"
            )
        value = 0.0
        if joint_type != "fixed":
            if name not in state_by_name:
                raise ValueError(f"R01 state omits active URDF joint {name}")
            index, state_type = state_by_name[name]
            if state_type != joint_type:
                raise ValueError(f"R01/URDF joint type mismatch for {name}")
            value = float(qpos[index])
            active_urdf_names.add(name)
        axis_node = node.find("axis")
        axis = np.asarray(
            [
                float(item)
                for item in (
                    axis_node.get("xyz", "1 0 0")
                    if axis_node is not None
                    else "1 0 0"
                ).split()
            ],
            dtype=float,
        )
        if axis.shape != (3,):
            raise ValueError(f"URDF joint axis is invalid for {name}")
        joints.append(
            {
                "name": name,
                "type": joint_type,
                "parent": parent,
                "child": child,
                "origin": _xyz_rpy_transform(
                    node.find("origin"), float(scale[0])
                ),
                "axis": axis,
                "value": value,
            }
        )
        children.add(child)
    if active_urdf_names != set(state_by_name):
        extra = sorted(set(state_by_name) - active_urdf_names)
        raise ValueError(
            f"R01 state contains unknown active URDF joints: {extra}"
        )

    roots = sorted(links - children)
    if not roots:
        raise ValueError("articulated URDF has no root link")
    transforms = {name: np.eye(4, dtype=float) for name in roots}
    pending = list(joints)
    while pending:
        progressed = False
        for joint in list(pending):
            if joint["parent"] not in transforms:
                continue
            transforms[joint["child"]] = (
                transforms[joint["parent"]]
                @ joint["origin"]
                @ _axis_motion(
                    joint["type"], joint["axis"], joint["value"]
                )
            )
            pending.remove(joint)
            progressed = True
        if not progressed:
            unresolved = sorted(str(row["name"]) for row in pending)
            raise ValueError(
                f"articulated URDF link graph is unresolved: {unresolved}"
            )

    active_index = articulated_joint_state.get("joint_index")
    if active_index is None and len(qpos) == 1:
        active_index = 0
    if active_index is None:
        moving_roots = {
            str(row["child"])
            for row in joints
            if row["type"] != "fixed"
        }
    else:
        selected_names = {
            name
            for name, (index, _) in state_by_name.items()
            if index == int(active_index)
        }
        if len(selected_names) != 1:
            raise ValueError("articulated active joint index is invalid")
        moving_roots = {
            str(row["child"])
            for row in joints
            if row["name"] in selected_names
        }

    parent_by_link = {
        str(row["child"]): str(row["parent"]) for row in joints
    }

    def descends_from_moving_root(link: str) -> bool:
        current = str(link)
        while True:
            if current in moving_roots:
                return True
            if current not in parent_by_link:
                return False
            current = parent_by_link[current]

    eligible_links = {
        link for link in links if descends_from_moving_root(link)
    }
    return transforms, eligible_links, hashlib.sha256(payload).hexdigest()


def _matrix_rows(value: Any) -> list[np.ndarray]:
    if value is None or (
        isinstance(value, (list, tuple)) and len(value) == 0
    ):
        return []
    array = np.asarray(value, dtype=float)
    if array.shape == (4, 4):
        array = array.reshape(1, 4, 4)
    if (
        array.ndim != 3
        or array.shape[1:] != (4, 4)
        or not np.isfinite(array).all()
    ):
        raise ValueError(
            "operation-anchor matrices must have shape 4x4 or Nx4x4"
        )
    return [row.copy() for row in array]


def _point_rows(
    model_data: Mapping[str, Any],
    *,
    kind: str,
    root_rotation: np.ndarray,
    scale: np.ndarray,
    link_transforms: Mapping[str, np.ndarray] | None = None,
    eligible_links: set[str] | None = None,
) -> list[dict[str, Any]]:
    articulated = link_transforms is not None
    field = (
        _ARTICULATED_POINT_FIELDS[kind]
        if articulated
        else _RIGID_POINT_FIELDS[kind]
    )
    raw_rows = model_data.get(field) or []
    normalized: list[tuple[np.ndarray, str | None]] = []
    if articulated:
        if not isinstance(raw_rows, list):
            raise ValueError(f"articulated {field} must be a list")
        for raw in raw_rows:
            if not isinstance(raw, Mapping):
                raise ValueError(
                    f"articulated {field} rows must be mappings"
                )
            base_link = str(raw.get("base") or "")
            if base_link not in link_transforms:
                raise ValueError(
                    f"articulated {field} references unknown link "
                    f"{base_link!r}"
                )
            matrices = _matrix_rows(raw.get("matrix"))
            if len(matrices) != 1:
                raise ValueError(
                    f"articulated {field} row must contain one matrix"
                )
            normalized.append((matrices[0], base_link))
    else:
        normalized = [
            (matrix, None) for matrix in _matrix_rows(raw_rows)
        ]

    root_transform = np.eye(4, dtype=float)
    root_transform[:3, :3] = root_rotation
    rows: list[dict[str, Any]] = []
    for index, (matrix, base_link) in enumerate(normalized):
        scaled = matrix.copy()
        scaled[:3, 3] *= scale
        local_frame = (
            link_transforms[str(base_link)]
            if articulated and base_link is not None
            else np.eye(4, dtype=float)
        )
        canonical = root_transform @ local_frame @ scaled
        row: dict[str, Any] = {
            "id": f"{kind}:{index}",
            "kind": kind,
            "point_index": int(index),
            "position_xyz_m": canonical[:3, 3].round(12).tolist(),
            "source_field": field,
            "source_frame": (
                "link_local" if articulated else "actor_local"
            ),
            # Authored switch-press points may intentionally live on a fixed
            # reference link.  Preserve that valid policy and record motion
            # membership separately instead of guessing eligibility from
            # topology alone; execution profiles own exact Contact selection.
            "operation_eligible": True,
            "moves_with_active_joint": (
                str(base_link) in (eligible_links or set())
                if articulated
                else True
            ),
        }
        if base_link is not None:
            row["base_link"] = str(base_link)
        if kind == "grasp_contact":
            direction = _approach_direction(canonical)
            row.update(
                {
                    "approach_direction_xyz": direction.round(12).tolist(),
                    "approach_class": _classify_approach(direction),
                }
            )
        rows.append(row)
    return rows


def build_operation_anchor_contract(
    model_data: Mapping[str, Any],
    q0_wxyz: Sequence[float],
    *,
    scale_xyz: Sequence[float] | None = None,
    articulated_joint_state: Mapping[str, Any] | None = None,
    urdf_path: Path | str | None = None,
    canonical_state_hash: str | None = None,
) -> dict[str, Any]:
    """Project reviewed operation points into canonical actor-root frame.

    ``model_data.transform_matrix`` is deliberately not consumed. BenchmarkV1
    restores the approved actor root after load; applying that legacy loader
    transform here would double-transform the operation point.
    """

    root_rotation = _rotation_matrix(q0_wxyz)
    scale = _scale_xyz(model_data, scale_xyz)
    joint_state = dict(articulated_joint_state or {})
    articulated = (
        str(joint_state.get("mode") or "not_articulated")
        != "not_articulated"
    )
    link_transforms: dict[str, np.ndarray] | None = None
    eligible_links: set[str] | None = None
    urdf_sha256: str | None = None
    if articulated:
        if urdf_path is None:
            raise ValueError("articulated operation anchors require a URDF")
        (
            link_transforms,
            eligible_links,
            urdf_sha256,
        ) = _articulated_link_transforms(
            Path(urdf_path), joint_state, scale
        )
    anchors = [
        row
        for kind in ("grasp_contact", "functional", "orientation")
        for row in _point_rows(
            model_data,
            kind=kind,
            root_rotation=root_rotation,
            scale=scale,
            link_transforms=link_transforms,
            eligible_links=eligible_links,
        )
    ]
    metadata_projection = {
        field: model_data[field]
        for field in _OPERATION_METADATA_FIELDS
        if field in model_data
    }
    payload: dict[str, Any] = {
        "version": OPERATION_ANCHOR_POLICY_VERSION,
        "projection_mode": (
            "articulated_link_fk" if articulated else "rigid_actor_local"
        ),
        "q0_wxyz": [round(float(value), 12) for value in q0_wxyz],
        "scale_xyz": scale.round(12).tolist(),
        "operation_metadata_hash": artifact_hash(metadata_projection),
        "anchors": anchors,
    }
    if canonical_state_hash:
        payload["canonical_state_hash"] = str(canonical_state_hash)
    if articulated:
        payload.update(
            {
                "articulated_joint_state_hash": artifact_hash(joint_state),
                "articulated_qpos": [
                    round(float(value), 12) for value in joint_state["qpos"]
                ],
                "active_joint_index": joint_state.get("joint_index"),
                "urdf_sha256": str(urdf_sha256),
            }
        )
    return canonical_data(
        {
            **payload,
            "contract_hash": artifact_hash(payload),
        }
    )


def select_table_pick_anchor(
    contract: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Choose the same deterministic approach tier preferred by table Pick."""

    candidates = [
        dict(row)
        for row in contract.get("anchors", [])
        if row.get("kind") == "grasp_contact"
        and row.get("operation_eligible", True) is True
        and row.get("approach_class") != "upward"
    ]
    return min(
        candidates,
        key=lambda row: (
            _TABLE_APPROACH_ORDER.get(
                str(row.get("approach_class")), 99
            ),
            int(row.get("point_index", 10**9)),
        ),
        default=None,
    )


def origin_bounds_for_anchor_target(
    anchor_position_xyz_m: Sequence[float],
    *,
    target_x_range_m: Sequence[float],
    target_y_range_m: Sequence[float],
) -> dict[str, list[float]]:
    """Translate a shared operation-point workspace into actor-origin bounds."""

    anchor = np.asarray(anchor_position_xyz_m, dtype=float)
    x_range = np.asarray(target_x_range_m, dtype=float)
    y_range = np.asarray(target_y_range_m, dtype=float)
    if anchor.shape != (3,) or not np.isfinite(anchor).all():
        raise ValueError("anchor position must contain finite xyz")
    if x_range.shape != (2,) or y_range.shape != (2,):
        raise ValueError("target ranges must contain two values")
    if x_range[0] > x_range[1] or y_range[0] > y_range[1]:
        raise ValueError("target ranges must be ordered")
    return {
        "xlim": [
            float(x_range[0] - anchor[0]),
            float(x_range[1] - anchor[0]),
        ],
        "ylim": [
            float(y_range[0] - anchor[1]),
            float(y_range[1] - anchor[1]),
        ],
    }


def anchor_kind_positions(
    contract: Mapping[str, Any], kinds: Sequence[str]
) -> list[dict[str, Any]]:
    """Return eligible frozen positions for the requested operation kinds."""

    allowed = {str(value) for value in kinds}
    return [
        dict(row)
        for row in contract.get("anchors", [])
        if str(row.get("kind")) in allowed
        and row.get("operation_eligible", True) is True
        and len(row.get("position_xyz_m") or []) == 3
        and all(
            math.isfinite(float(value))
            for value in row["position_xyz_m"]
        )
    ]


__all__ = [
    "OPERATION_ANCHOR_POLICY_VERSION",
    "anchor_kind_positions",
    "build_operation_anchor_contract",
    "origin_bounds_for_anchor_target",
    "select_table_pick_anchor",
]
