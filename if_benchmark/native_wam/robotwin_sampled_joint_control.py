"""Joint14 sample transport on the collection robot's unchanged PD drives."""
import numpy as np


def sample_steps(period_s, physics_dt):
    if not np.isfinite([period_s, physics_dt]).all() or min(period_s, physics_dt) <= 0:
        raise ValueError('Control and physics periods must be finite and positive')
    ratio = period_s / physics_dt
    steps = round(ratio)
    if steps < 1 or not np.isclose(ratio, steps, rtol=1e-6, atol=1e-7):
        raise ValueError('Control period must contain a whole number of physics steps')
    return steps


def measured_arm_joints(robot, arm):
    entity = getattr(robot, arm + '_entity')
    active = list(entity.get_active_joints())
    values = np.asarray(entity.get_qpos(), dtype=np.float64)
    return np.array([values[active.index(j)] for j in getattr(robot, arm + '_arm_joints')])



def take_joint_action(env,command,*,period_s=.04):
    values=np.asarray(command,dtype=np.float64)
    if values.shape!=(14,) or not np.isfinite(values).all():raise ValueError('Expected finite joint14 command')
    if not np.all((values[[6, 13]] >= 0) & (values[[6, 13]] <= 1)):
        raise ValueError('Gripper commands must be in [0, 1]')
    if env.take_action_cnt>=env.step_lim:return
    steps=sample_steps(period_s,float(env.scene.get_timestep()))
    previous=getattr(env,'_if_sampled_joint_previous',None)
    if env.take_action_cnt==0 or previous is None:
        previous={arm:measured_arm_joints(env.robot,arm) for arm in ('left','right')}
        env._if_sampled_execution=[]
    targets={arm:values[start:start+6].copy() for arm,start in (('left',0),('right',7))}
    for arm,start in (('left',0),('right',7)):
        env.robot.set_arm_joints(targets[arm],(targets[arm]-previous[arm])/period_s,arm)
        env.robot.set_gripper(float(values[start+6]),arm,gripper_eps=0)
    env._if_sampled_joint_previous=targets
    env._if_sampled_execution.append(dict(version='robotwin_sampled_joint14_pd_v1',action_index=env.take_action_cnt,
        period_seconds=period_s,physics_steps=steps,command=values.tolist(),rejected_arms={}))
    env.take_action_cnt+=1
    for _ in range(steps):env.scene.step()
