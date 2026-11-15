"""One-time read-only SAPIEN geometry adapter for operation-space templates."""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from transforms3d.euler import euler2mat

from if_benchmark.reach_validation.contract import ROUTE_EXECUTION_CONTRACT, route_lift_delta_m
from if_benchmark.schema import artifact_hash
from if_benchmark.grasp_validation.convex_cover import live_collision_pieces, _live_pose_matrix


def extract_robot_geometry(robot, arm):
    """Use every cooked shape of wrist, fingers and fixed camera, preserving pieces.

    Recover geometry in each actual link frame, then place it with the reviewed
    direct fixed/prismatic URDF chain at q=0. No robot pose/joint/drive is changed.
    """
    if arm not in ('left', 'right'):
        raise ValueError('invalid arm')
    urdf = Path(getattr(robot, arm + '_urdf_path'))
    root = ET.parse(urdf).getroot()
    prefix = 'fl' if arm == 'left' else 'fr'
    ee = prefix + '_link6'
    links = {link.name: link for link in getattr(robot, arm + '_entity').get_links()}
    joints = {j.find('child').get('link'): j for j in root.findall('joint')}
    grippers = {j.child_link.name: (mul, bias) for j, mul, bias in getattr(robot, arm + '_gripper')}
    scale = np.asarray(getattr(robot, arm + '_gripper_scale'), float)
    result = []
    for name in (ee, prefix + '_link7', prefix + '_link8', arm + '_camera'):
        link = links[name]
        geometry = live_collision_pieces(link)
        transform = np.eye(4)
        axis = np.zeros(3)
        opened = lo = hi = 0.
        kind = 'fixed'
        if name != ee:
            joint = joints[name]
            kind = joint.get('type')
            if joint.find('parent').get('link') != ee or kind not in ('fixed', 'prismatic'):
                raise ValueError('only direct fixed/prismatic EE children supported')
            origin = joint.find('origin')
            xyz = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
            rpy = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')
            transform[:3, :3] = euler2mat(*rpy)
            transform[:3, 3] = xyz
            if kind == 'prismatic':
                axis = transform[:3, :3] @ np.fromstring(joint.find('axis').get('xyz'), sep=' ')
                mul, bias = grippers[name]
                bounds = scale * mul + bias
                lo, hi = float(bounds.min()), float(bounds.max())
                opened = float(bounds[1])
                limits = joint.find('limit')
                # Closing drive may command beyond the hard stop (-10 mm here).
                # Actual unknown closure is bounded by physical joint travel.
                lo, hi = float(limits.get('lower')), float(limits.get('upper'))
                if not lo <= opened <= hi:
                    raise ValueError('open gripper command outside URDF joint limits')
        inv = np.linalg.inv(_live_pose_matrix(link.entity.pose))
        for i, (piece, source) in enumerate(zip(geometry['pieces'], geometry['sources'])):
            local = np.asarray(piece) @ inv[:3, :3].T + inv[:3, 3]
            q0 = local @ transform[:3, :3].T + transform[:3, 3]
            result.append({'label': f'{name}:{i}', 'vertices_ee_q0': q0.tolist(),
                           'joint_type': kind, 'joint_axis_ee': axis.tolist(),
                           'held_q_range_m': [lo, hi], 'open_q_m': opened, 'source': source})
    return {'arm': arm, 'pieces': result, 'urdf_sha256': hashlib.sha256(urdf.read_bytes()).hexdigest(),
            'frame': ee, 'width_policy': 'full_URDF_joint_range_including_unknown_physical_closure'}


def extract_scene_geometry(env, *, contact_ids, robot_geometry=None):
    """Plain JSON snapshot. Subtract origin only, never remove the frozen world yaw."""
    subject = env.entities['obj_subj']
    origin = np.asarray(subject.get_pose().p, float)
    entities = {}
    for name, actor in env.entities.items():
        geometry = live_collision_pieces(actor)
        geometry['pieces'] = [(np.asarray(p) - origin).tolist() for p in geometry['pieces']]
        geometry['frame'] = 'world_axes_at_subject_origin'
        # geometry_sha256 is explicitly the original live-world extraction hash.
        entities[name] = geometry
    contacts = {}
    for cid in contact_ids:
        matrix = np.asarray(subject.get_contact_point(int(cid), 'matrix'), float)
        matrix[:3, 3] -= origin
        contacts[str(cid)] = matrix.tolist()
    robots = robot_geometry or {a: extract_robot_geometry(env.robot, a) for a in ('left', 'right')}
    return {'entities': entities, 'contacts': contacts, 'robots': deepcopy(robots),
            'subject_origin_world': origin.tolist(),
            'support_plane_z_m': float(env._frozen_support_plane_z('obj_subj')) - float(origin[2]),
            'subject_identity': deepcopy(env.scene_spec['entities']['obj_subj']),
            'scope': 'geometry_snapshot_no_new_planning_or_skill_execution'}


def make_spec(snapshot, *, arm, contact_id, approach_retreat_m=.08,
              relation="between", lift_m=None, descent_m=None, release_retreat_m=None,
              robot_padding_m=0., held_world_box_padding_m=0.):
    identity = deepcopy(snapshot['subject_identity'])
    # An XY translation leaves this oriented template unchanged. Height above the
    # support, orientation, scale, joint state and complete geometry remain keyed.
    identity['pose']['position'] = [0., 0., 0.]
    return {'arm': arm, 'contact_id': int(contact_id), 'motion_model': 'fixed_orientation_translation',
            'subject_identity': identity, 'robot_urdf_sha256': snapshot['robots'][arm]['urdf_sha256'],
            'subject_pieces': snapshot['entities']['obj_subj']['pieces'],
            'robot_pieces': snapshot['robots'][arm]['pieces'],
            'contact_matrix': snapshot['contacts'][str(contact_id)],
            'support_plane_z_m': snapshot['support_plane_z_m'],
            'approach_retreat_m': approach_retreat_m, 'lift_m': route_lift_delta_m(relation) if lift_m is None else lift_m,
            'descent_m': ROUTE_EXECUTION_CONTRACT['phase_offsets_m']['approach'] if descent_m is None else descent_m,
            'release_retreat_m': ROUTE_EXECUTION_CONTRACT['release_retreat']['delta_m'] if release_retreat_m is None else release_retreat_m,
            'route_contract_hash': artifact_hash(ROUTE_EXECUTION_CONTRACT), 'relation': relation,
            'robot_padding_m': robot_padding_m, 'held_world_box_padding_m': held_world_box_padding_m,
            'subject_shape_sources': snapshot['entities']['obj_subj']['sources']}
