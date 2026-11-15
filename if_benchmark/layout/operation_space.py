"""Cached local manipulation geometry; diagnostic API, not an M12 certificate.

Only fixed-orientation translation and direct prismatic fingers are supported.
Each convex piece stays separate. A clear result concerns this local envelope,
not arm IK, global motion, physics, intentional contacts or publication gates.
The module deliberately has no simulator, planner, CUDA or FCL dependency.
"""
from __future__ import annotations

from copy import deepcopy
from itertools import product
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
from transforms3d.quaternions import quat2mat

from if_benchmark.schema import artifact_hash
from if_benchmark.grasp_validation.core import support_safe_grasp_poses
from if_benchmark.grasp_validation.contact_contract import EXACT_GRASP_POLICY, validate_contact_matrix

VERSION = 'local_operation_space_translation_v1'
NUMERICAL_GUARD_M = 1e-5
PHASES = ('approach_open', 'close', 'lift_held', 'descent_held', 'release_open', 'retreat_open')


def points(value):
    p = np.asarray(value, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 4 or not np.isfinite(p).all():
        raise ValueError('expected at least four finite 3D vertices')
    return p


def vector(value):
    v = np.asarray(value, dtype=float)
    if v.shape != (3,) or not np.isfinite(v).all():
        raise ValueError('expected finite 3-vector')
    return v


def nonnegative(value):
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError('expected finite nonnegative distance')
    return value


def convex_piece(vertices, *, label, source=None):
    p = np.unique(points(vertices), axis=0)
    hull = ConvexHull(p)  # Degenerate geometry is an error, never a clear result.
    p = p[hull.vertices]
    hull = ConvexHull(p)
    # Coplanar triangulation duplicates are unnecessary in the four-variable LP.
    equations = np.unique(np.round(hull.equations, 12), axis=0)
    return {'label': str(label), 'vertices': p.tolist(),
            'halfspaces': equations.tolist(),
            'min': p.min(axis=0).tolist(), 'max': p.max(axis=0).tolist(),
            'source': deepcopy(source or {})}


def sweep(vertices, displacement, *, label, padding_m=0.0, source=None):
    """Exact convex-piece sweep for translation, with optional world-axis box margin."""
    p, d = points(vertices), vector(displacement)
    pad = nonnegative(padding_m)
    p = np.vstack((p, p + d))
    if pad:
        offsets = np.asarray(list(product((-pad, pad), repeat=3)))
        p = (p[:, None, :] + offsets).reshape(-1, 3)
    return convex_piece(p, label=label, source=source)


@lru_cache(maxsize=1)
def implementation_fingerprint():
    # Also invalidate when shared pose conversion changes without a policy rename.
    files = (Path(__file__), Path(support_safe_grasp_poses.__code__.co_filename),
             Path(validate_contact_matrix.__code__.co_filename))
    return [hashlib.sha256(path.read_bytes()).hexdigest() for path in files]


def template_key(spec):
    # Complete numeric inputs and provenance, not just asset names or Contact ids.
    return artifact_hash({'version': VERSION, 'implementation': implementation_fingerprint(),
                          'grasp_policy': EXACT_GRASP_POLICY, 'spec': spec})


def build_template(spec):
    """Build in world axes translated to subject origin; orientation is in the key.

    spec holds subject pieces, Contact matrix, support Z (all origin-relative),
    and robot pieces in physical EE coordinates at joint q=0. Finger q intervals
    bound uncertainty; independent extrusion can overestimate simultaneous closure.
    Held object margins are separate from raw geometry and are never implicit.
    """
    s = deepcopy(spec)
    if s.get('arm') not in ('left', 'right'):
        raise ValueError('arm must be left or right')
    if s.get('motion_model') != 'fixed_orientation_translation':
        raise ValueError('rotation and arbitrary joint trajectories require another template model')
    # Same authored-Contact validation and pose conversion as production Pick.
    # Asset rounding is accepted by that contract; no new orthonormalization.
    contact = validate_contact_matrix(s['contact_matrix'])
    retreat = nonnegative(s['approach_retreat_m'])
    lift = nonnegative(s['lift_m'])
    descent = nonnegative(s['descent_m'])
    release_retreat = nonnegative(s['release_retreat_m'])
    padding = nonnegative(s['robot_padding_m'])
    held_padding = nonnegative(s['held_world_box_padding_m'])
    support_z = float(s['support_plane_z_m'])
    if not np.isfinite(support_z):
        raise ValueError('nonfinite support plane')
    poses = support_safe_grasp_poses(contact, retreat, support_plane_z_m=support_z)
    if not poses['inner_pad_contains_authored_contact']:
        raise ValueError('support-adjusted Contact outside inner pad')
    grasp = np.asarray(poses['grasp_pose'], float)
    pre = np.asarray(poses['pre_grasp_pose'], float)
    rotation, origin = quat2mat(grasp[3:]), grasp[:3]
    phases = {name: [] for name in PHASES}
    robot = s['robot_pieces']
    if not robot or not s['subject_pieces']:
        raise ValueError('empty robot or subject geometry')
    for piece in robot:
        p = points(piece['vertices_ee_q0'])
        axis = vector(piece['joint_axis_ee'])
        lo, hi = map(float, piece['held_q_range_m'])
        opened = float(piece['open_q_m'])
        if not np.isfinite([lo, hi, opened]).all() or lo > hi or not lo <= opened <= hi:
            raise ValueError('invalid bounded finger opening')
        if piece['joint_type'] == 'fixed':
            if np.any(axis) or lo or hi or opened:
                raise ValueError('fixed piece cannot move')
        elif piece['joint_type'] == 'prismatic':
            if not np.isclose(np.linalg.norm(axis), 1., atol=1e-7):
                raise ValueError('prismatic axis must be unit length')
        else:
            raise ValueError('only fixed/direct prismatic robot pieces are supported')
        p0 = p @ rotation.T + origin
        a = axis @ rotation.T
        p_open = p0 + opened * a
        # Complete union over declared width interval, not guessed physical closure.
        p_held = np.vstack((p0 + lo * a, p0 + hi * a))
        for name, start, delta in (
            ('approach_open', p_open + pre[:3] - origin, origin - pre[:3]),
            ('close', p_held, [0, 0, 0]),
            ('lift_held', p_held, [0, 0, lift]),
            ('descent_held', p_held, [0, 0, descent]),
            ('release_open', p_held, [0, 0, 0]),
            ('retreat_open', p_open, -rotation[:, 0] * release_retreat),
        ):
            phases[name].append(sweep(start, delta, label=piece['label'], padding_m=padding,
                                      source=piece.get('source')))
    for i, piece in enumerate(s['subject_pieces']):
        for name, delta in (('close', [0, 0, 0]), ('lift_held', [0, 0, lift]),
                            ('descent_held', [0, 0, descent]), ('release_open', [0, 0, 0]),
                            ('retreat_open', [0, 0, 0])):
            phases[name].append(sweep(piece, delta, label=f'subject:{i}',
                                      padding_m=held_padding if name in ('lift_held', 'descent_held') else 0.))
    result = {'version': VERSION, 'key': template_key(s), 'spec': s, 'grasp': poses,
              'frame': 'world_axes_at_subject_origin', 'phases': phases,
              'scope': 'local_geometry_only_not_full_arm_or_physics',
              'held_width_policy': 'entire_declared_interval_conservative',
              'padding_metric': 'world_axis_box_per_side_not_euclidean_sphere',
              'phase_bounds': {name: {'min': np.min([p['min'] for p in ps], axis=0).tolist(),
                                      'max': np.max([p['max'] for p in ps], axis=0).tolist()}
                               for name, ps in phases.items()}}
    result['content_hash'] = artifact_hash(result)
    return result


class TemplateCache:
    """Content-addressed, verified JSON; disk and in-process reuse, atomic writes."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self._memory = {}

    def get(self, spec):
        key = template_key(spec)
        if key in self._memory:
            return deepcopy(self._memory[key]), 'memory'
        path = self.directory / (key + '.json')
        if path.exists():
            value = json.loads(path.read_text())
            if (value.get('key') != key or template_key(value['spec']) != key
                    or value.get('version') != VERSION
                    or artifact_hash(value, omit_keys=('content_hash',)) != value.get('content_hash')):
                raise ValueError('operation template cache identity/content mismatch')
            status = 'disk'
        else:
            value = build_template(spec)
            self.directory.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(dir=self.directory, suffix='.tmp')
            try:
                with os.fdopen(fd, 'w') as stream:
                    json.dump(value, stream, allow_nan=False, separators=(',', ':'))
                os.replace(temporary, path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            status = 'built'
        self._memory[key] = value
        return deepcopy(value), status


def _feasible(result):
    if result.status == 2:
        return False
    if not result.success:
        raise RuntimeError(f'geometry LP failed (never treated as clear): {result.message}')
    return True


def _constraints(a, b, displacement):
    """Intersection point x in A and B+t*d: four-variable linear feasibility."""
    ah, bh = np.asarray(a['halfspaces']), np.asarray(b['halfspaces'])
    matrix = np.vstack((np.c_[ah[:, :3], np.zeros(len(ah))],
                        np.c_[bh[:, :3], -bh[:, :3] @ displacement]))
    bound = -np.r_[ah[:, 3], bh[:, 3]]
    return matrix, bound


def _lp(objective, matrix, bound, bounds):
    return linprog(objective, A_ub=matrix, b_ub=bound, bounds=bounds, method='highs',
                   options={'primal_feasibility_tolerance': 1e-9,
                            'dual_feasibility_tolerance': 1e-9})


def overlap(a, b):
    """Closed-set intersection, including touching; geometry only."""
    if np.any(np.asarray(a['max']) < b['min']) or np.any(np.asarray(b['max']) < a['min']):
        return False
    ah, bh = np.asarray(a['halfspaces']), np.asarray(b['halfspaces'])
    return _feasible(_lp(np.zeros(3), np.vstack((ah[:, :3], bh[:, :3])),
                         -np.r_[ah[:, 3], bh[:, 3]], [(None, None)] * 3))


def collision_interval(a, b, direction):
    """All obstacle translations t*d causing convex intersection, in meters.

    No stepping scan: solve two 4-variable linear programs. The union of these
    intervals over piece pairs is the forbidden set for the complete template.
    """
    d = vector(direction)
    if not np.isclose(np.linalg.norm(d), 1., atol=1e-10):
        raise ValueError('direction must be unit length')
    # Project on axes perpendicular to movement for a cheap necessary condition.
    axis = int(np.argmax(np.abs(d)))
    if abs(d[axis]) == 1:
        other = [i for i in range(3) if i != axis]
        if (np.any(np.asarray(a['max'])[other] < np.asarray(b['min'])[other])
                or np.any(np.asarray(b['max'])[other] < np.asarray(a['min'])[other])):
            return None
    matrix, bound = _constraints(a, b, d)
    lower = _lp([0, 0, 0, 1], matrix, bound, [(None, None)] * 4)
    if not _feasible(lower):
        return None
    upper = _lp([0, 0, 0, -1], matrix, bound, [(None, None)] * 4)
    if not _feasible(upper):
        raise RuntimeError('inconsistent geometry LP interval')
    return [float(lower.x[3]), float(upper.x[3])]


def translated(piece, offset):
    out = deepcopy(piece)
    d = vector(offset)
    out['vertices'] = (np.asarray(out['vertices']) + d).tolist()
    out['min'] = (np.asarray(out['min']) + d).tolist()
    out['max'] = (np.asarray(out['max']) + d).tolist()
    h = np.asarray(out['halfspaces'])
    h[:, 3] -= h[:, :3] @ d
    out['halfspaces'] = h.tolist()
    return out


def check_phase(template, phase, obstacles, *, reference_offset=(0, 0, 0)):
    pieces = [translated(p, reference_offset) for p in template['phases'][phase]]
    hits = [{'operation_piece': a['label'], 'obstacle': b['label']}
            for a in pieces for b in obstacles if overlap(a, b)]
    return {'phase': phase, 'local_envelope_clear': not hits, 'intersections': hits,
            'scope': template['scope']}


def displacement_advice(template, phase, obstacles, direction, *, reference_offset=(0, 0, 0), guard_m=NUMERICAL_GUARD_M):
    """Move the supplied obstacle group, keeping its internal layout unchanged.

    Returns the first nonnegative clear component along one chosen direction.
    Guard is numerical only. Physical safety margin belongs in template inputs.
    This neither chooses the semantic direction nor validates table/camera/reach.
    """
    d = vector(direction)
    if not np.isclose(np.linalg.norm(d), 1., atol=1e-10):
        raise ValueError('direction must be unit length')
    guard = nonnegative(guard_m)
    if guard <= 0:
        raise ValueError('positive guard required to avoid closed contact boundary')
    intervals = []
    for p in template['phases'][phase]:
        a = translated(p, reference_offset)
        for b in obstacles:
            interval = collision_interval(a, b, direction)
            if interval is not None:
                intervals.append({'interval_m': interval, 'operation_piece': p['label'], 'obstacle': b['label']})
    merged = []
    for lo, hi in sorted(x['interval_m'] for x in intervals):
        lo, hi = lo - guard, hi + guard
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    shift = 0.
    for lo, hi in merged:
        if lo <= shift <= hi:
            shift = float(np.nextafter(hi, np.inf))
    return {'obstacle_direction': vector(direction).tolist(), 'minimum_positive_shift_m': shift,
            'forbidden_intervals_m': merged, 'pair_intervals': intervals,
            'algorithm': 'two_4_variable_LPs_per_broadphase_pair_no_grid_scan',
            'numerical_guard_m': guard}
