"""Collection-compatible physical grasp effects for continuous policy actions.

The simulator identifies whichever body the actual fingers close around. Task
goals, language, expert paths and planned contacts are never consulted here.
The collection's closure verifier, D6 factory and drive helpers remain shared.
"""
from collections import defaultdict, deque
from copy import deepcopy

import numpy as np

from if_benchmark.execution_policy import (
    HOLD_EE_FRAME_ARM, HELD_MOTION_ARM_DRIVE_STIFFNESS,
    HELD_MOTION_ARM_DRIVE_DAMPING, PICK_MANIPULAND_LINEAR_DAMPING,
    PICK_MANIPULAND_ANGULAR_DAMPING, PICK_REPLAY_ARM_DRIVE_STIFFNESS,
    PICK_REPLAY_ARM_DRIVE_DAMPING,
    GRIPPER_PLAN_STEPS, GRIPPER_TERMINAL_HOLD_RATIO,
)
from if_benchmark.success_contract import SuccessContract, hold_failure_code, pick_attachment_policy
from if_benchmark.success_contract.observation import (
    CONTACT_REGION_CLASSIFICATION_VERSION,
    DEFAULT_ALOHA_GRIPPER_PROFILE, load_embodiment_profile,
    build_hold_reference, summarize_hold_observation,
)
from .recording import MethodTap


def physics_contract():
    contract = SuccessContract.load()
    profile = load_embodiment_profile(DEFAULT_ALOHA_GRIPPER_PROFILE)
    return dict(version='m15_collection_grasp_physics_v2_pad_normal',
        success_contract_hash=contract.hash,
        contact_region_classifier=CONTACT_REGION_CLASSIFICATION_VERSION,
        embodiment_profile_hash=profile['profile_hash'],
        manipuland_damping=dict(linear=PICK_MANIPULAND_LINEAR_DAMPING,
            angular=PICK_MANIPULAND_ANGULAR_DAMPING),
        precontact_arm_drive=dict(stiffness=PICK_REPLAY_ARM_DRIVE_STIFFNESS,
            damping=PICK_REPLAY_ARM_DRIVE_DAMPING),
        held_arm_drive=dict(stiffness=HELD_MOTION_ARM_DRIVE_STIFFNESS,
            damping=HELD_MOTION_ARM_DRIVE_DAMPING),
        attachment=pick_attachment_policy(contract=contract),
        continuous_contact_min_steps=profile['classification']['continuous_contact_min_steps'],
        close_command_tolerance=0.01,
        close_terminal_hold_steps=int(GRIPPER_PLAN_STEPS * GRIPPER_TERMINAL_HOLD_RATIO),
        release_command_opening=profile['classification']['engaged_max_normalized_opening'],
        selection='unique actual bilateral inner-pad closure; all dynamic candidates',
        release='model open command or failed physical constraint; reset at episode end',
        model_goal_inputs=False, expert_action_generation=False,
        stage_bridge='continuous commands use physical closure; opening uses the reviewed open classifier')


class CollectionGraspPhysics:
    def __init__(self, env):
        self.env = env
        self.contract = SuccessContract.load()
        self.profile = load_embodiment_profile(DEFAULT_ALOHA_GRIPPER_PROFILE)
        self.settings = physics_contract()
        self.minimum = int(self.settings['continuous_contact_min_steps'])
        self.open_threshold = float(self.settings['release_command_opening'])
        self.subjects = {s for s, kind in env.dynamics.items() if kind == 'dynamic'}
        self.pending, self.attached, self.saved_drives = {}, {}, {}
        self.closed_since = {a:None for a in ('left','right')}
        self.commands = {a:float(getattr(env.robot, f'get_{a}_gripper_val')()) for a in ('left','right')}
        self.events, self.taps, self.step = [], [], 0
        self.initial_drives = {}
        for arm in ('left','right'):
            self.initial_drives[arm] = [dict(stiffness=float(j.get_stiffness()), damping=float(j.get_damping()))
                for j in getattr(env.robot, arm + '_arm_joints')]
        for subject in sorted(self.subjects):
            env._unified_configure_pick_dynamics(subject,
                linear_damping=PICK_MANIPULAND_LINEAR_DAMPING,
                angular_damping=PICK_MANIPULAND_ANGULAR_DAMPING)
        try:
            self.taps.append(MethodTap(env.robot, 'set_arm_joints', self._joint_wrapper))
            self.taps.append(MethodTap(env.robot, 'set_gripper', self._gripper_wrapper))
        except BaseException:
            self.close()
            raise

    def _activate_arm(self, arm):
        if arm not in self.saved_drives:
            from if_benchmark.scene_validation.articulation_attachment import boost_arm_drive
            self.saved_drives[arm] = boost_arm_drive(self.env.robot, arm,
                stiffness=PICK_REPLAY_ARM_DRIVE_STIFFNESS, damping=PICK_REPLAY_ARM_DRIVE_DAMPING)
            self.events.append(dict(kind='motion_drive_enabled', arm=arm, step=self.step,
                stiffness=PICK_REPLAY_ARM_DRIVE_STIFFNESS, damping=PICK_REPLAY_ARM_DRIVE_DAMPING))

    def _joint_wrapper(self, original):
        def command(target_position, target_velocity, arm_tag):
            arm = str(arm_tag)
            joints = getattr(self.env.robot, arm + '_arm_joints')
            previous = np.array([j.get_drive_target()[0] for j in joints])
            # A sampled IK feedforward velocity can contain a tiny initial
            # correction on an otherwise idle arm; the target change owns
            # acquisition of the collection's motion-drive profile.
            moving = np.any(np.abs(np.asarray(target_position) - previous) > 1e-7)
            if moving or arm in self.attached:
                self._activate_arm(arm)
            return original(target_position, target_velocity, arm_tag)
        return command

    def _gripper_wrapper(self, original):
        def command(gripper_val, arm_tag, gripper_eps=0.1):
            arm = str(arm_tag)
            self.commands[arm] = float(np.clip(gripper_val,0,1))
            if self.commands[arm] > self.open_threshold:
                self.release(arm, 'model_open_command')
                self.pending.pop(arm, None)
            return original(gripper_val, arm_tag, gripper_eps=gripper_eps)
        return command

    def release(self, arm, reason):
        held = self.attached.pop(arm, None)
        if held:
            self.env._remove_pick_attachment(self.env._arm(arm))
            self.env.hold_references.pop(arm, None)
            self.env.held_entities.pop(arm, None)
            self.env.pick_certifications.pop(arm, None)
            self.events.append(dict(kind='attachment_released', step=self.step, arm=arm,
                subject=held['subject'], reason=reason))
            saved = self.saved_drives.pop(arm, None)
            if saved is not None:
                from if_benchmark.scene_validation.articulation_attachment import restore_arm_drive
                restore_arm_drive(saved)
        self.pending.pop(arm, None)

    def after_step(self, grippers, contacts, step):
        self.step = step
        applied = self.env.robot.get_normal_real_gripper_val()
        for index, arm in enumerate(('left','right')):
            self.commands[arm] = float(getattr(self.env.robot, f'get_{arm}_gripper_val')())
            if applied[index] <= self.settings['close_command_tolerance']:
                if self.closed_since[arm] is None:
                    self.closed_since[arm] = step
            else:
                self.closed_since[arm] = None
            if self.commands[arm] > self.open_threshold:
                self.release(arm, 'model_open_command')
        fingers, touched = defaultdict(set), defaultdict(set)
        for row in contacts:
            arm = row.get('arm')
            if arm not in ('left','right') or row.get('region') != 'inner_pad':
                continue
            for subject in {row['entity_a'],row['entity_b']} & self.subjects:
                fingers[arm,subject].add(row.get('finger'))
                touched[arm].add(subject)
        for arm in ('left','right'):
            if arm in self.attached:
                result = self.env._observe_pick_attachment(self.env._arm(arm))
                if result['passed']:
                    continue
                self.release(arm, result.get('failure_code') or 'constraint_lost')
            qualified = [subject for (a,subject), pads in fingers.items()
                         if a == arm and len(pads - {None}) >= 2]
            settled = (self.closed_since[arm] is not None and
                step - self.closed_since[arm] >= self.settings['close_terminal_hold_steps'])
            if not settled or self.commands[arm] > self.open_threshold or len(qualified) != 1 or touched[arm] != set(qualified):
                self.pending.pop(arm, None)
                continue
            subject = qualified[0]
            pending = self.pending.get(arm)
            if pending is None or pending['subject'] != subject:
                reference = build_hold_reference(subject=subject, arm=arm,
                    actor_matrix=self.env._pose_matrix(subject),
                    ee_matrix=self.env._hold_ee_matrix(self.env._arm(arm)),
                    planning_ee_matrix=self.env._ee_matrix(self.env._arm(arm)),
                    actual_width_m=float(grippers[arm]['actual_width_m']), ee_frame=HOLD_EE_FRAME_ARM)
                pending = self.pending[arm] = dict(subject=subject, reference=reference,
                    samples=deque(maxlen=self.minimum))
            pending['samples'].append(dict(gripper=deepcopy(grippers[arm]), active_contacts=deepcopy(contacts),
                actor_matrix=self.env._pose_matrix(subject),
                ee_matrix=self.env._hold_ee_matrix(self.env._arm(arm)), ee_frame=HOLD_EE_FRAME_ARM))
            if len(pending['samples']) < self.minimum:
                continue
            summary = summarize_hold_observation(list(pending['samples']), reference=pending['reference'],
                subject=subject, arm=arm, embodiment_profile=self.profile,
                symmetry_axis=self.env._hold_symmetry_axis(subject))
            failure = hold_failure_code(summary, phase='pick', contract=self.contract)
            if failure is not None:
                continue
            self.env._unified_freeze_hold_reference(subject, arm)
            closure = {**summary['gripper']['closure_evidence'], 'reference_step':step}
            self.env._unified_store_certified_closure(arm, closure)
            result = self.env._unified_create_attachment(subject, arm)
            self.events.append(dict(kind='attachment_creation', arm=arm, subject=subject, step=step,
                result=deepcopy(result), closure=deepcopy(closure)))
            if not result['passed']:
                raise RuntimeError(f'collection_pick_attachment_failed:{arm}:{subject}:{result}')
            self.env.held_entities[arm] = subject
            self.attached[arm] = dict(subject=subject, reference_step=step, closure=closure,
                reference=deepcopy(self.env.hold_references[arm]))
            self._activate_arm(arm)
            self.pending.pop(arm, None)

    def observation(self, arm):
        held = self.attached.get(arm)
        if held is None:
            return None
        result = self.env._observe_pick_attachment(self.env._arm(arm))
        if not result['passed']:
            return None
        return {**held, 'attachment':{**result,'subject':held['subject'],'certified_close':True}}

    def report(self):
        return dict(contract=self.settings, initial_arm_drives=self.initial_drives,
            dynamic_subjects=sorted(self.subjects), events=deepcopy(self.events),
            active_attachments={a:h['subject'] for a,h in self.attached.items()})

    def close(self):
        from if_benchmark.scene_validation.articulation_attachment import restore_arm_drive
        for arm in list(self.attached):
            self.release(arm, 'episode_cleanup')
        for saved in self.saved_drives.values():
            restore_arm_drive(saved)
        self.saved_drives.clear()
        for tap in reversed(self.taps):
            tap.close()
        self.taps.clear()
        if getattr(self.env, '_if_collection_physics', None) is self:
            delattr(self.env, '_if_collection_physics')


def install(env):
    if getattr(env, '_if_collection_physics', None) is not None:
        raise ValueError('Collection physics already installed')
    controller = CollectionGraspPhysics(env)
    env._if_collection_physics = controller
    return controller
