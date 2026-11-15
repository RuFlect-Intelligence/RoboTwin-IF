"""GPU interpolation with exact constant intervals and the reviewed cuRobo clock.

The CPU device is supported for kernel regression tests. Production callers
use the original CUDA buffers/stream; no post-validation path repair occurs.
"""
import math

import torch
import warp as wp

wp.set_module_options({"fast_math": False, "enable_backward": False})


@wp.func
def _lerp(start: float, end: float, weight: float):
    if weight == 0.0:
        return start
    if weight == 1.0:
        return end
    if start == end:
        return start
    return start + weight * (end - start)


@wp.kernel
def _interpolate(
    position: wp.array(dtype=wp.float32), velocity: wp.array(dtype=wp.float32),
    acceleration: wp.array(dtype=wp.float32), jerk: wp.array(dtype=wp.float32),
    out_position: wp.array(dtype=wp.float32), out_velocity: wp.array(dtype=wp.float32),
    out_acceleration: wp.array(dtype=wp.float32), out_jerk: wp.array(dtype=wp.float32),
    opt_dt: wp.array(dtype=wp.float32), steps: wp.array(dtype=wp.int32),
    horizon: int, dof: int, out_horizon: int, raw_dt: float,
):
    tid = wp.tid()
    batch = tid // (out_horizon * dof)
    waypoint = (tid // dof) % out_horizon
    joint = tid % dof
    step_count = steps[batch]
    interval = float(step_count) / float(horizon - 1)
    upper = int(wp.ceil(float(waypoint) / interval))
    scale = raw_dt / opt_dt[batch]
    start_index = int(0)
    end_index = int(0)
    weight = float(0.0)
    if waypoint >= step_count or upper >= horizon:
        start_index = batch * horizon * dof + (horizon - 1) * dof + joint
        end_index = start_index
    else:
        upper = wp.max(upper, 1)
        lower = upper - 1
        weight = float(waypoint) / interval - float(lower)
        start_index = batch * horizon * dof + lower * dof + joint
        end_index = start_index + dof
    out_position[tid] = _lerp(position[start_index], position[end_index], weight)
    out_velocity[tid] = _lerp(velocity[start_index], velocity[end_index], weight) * scale
    out_acceleration[tid] = _lerp(acceleration[start_index], acceleration[end_index], weight) * scale * scale
    out_jerk[tid] = _lerp(jerk[start_index], jerk[end_index], weight) * scale * scale * scale


def stable_linear_interpolation(raw_traj, traj_tsteps, out_traj, opt_dt, raw_dt=0.5):
    """Keep the upstream signature, valid-step/tail policy and derivative scaling."""
    # CuRobo also supplies a one-element solver-dt Tensor. The original Warp
    # scalar argument marshals it to a host float; retain that ABI explicitly.
    if isinstance(raw_dt, torch.Tensor) and raw_dt.numel() != 1:
        raise ValueError("stable interpolation needs one solver time step")
    scalar_dt = float(raw_dt)
    names = ("position", "velocity", "acceleration", "jerk")
    raw = [getattr(raw_traj, name, None) for name in names]
    output = [getattr(out_traj, name, None) for name in names]
    if any(not isinstance(value, torch.Tensor) for value in [*raw, *output, traj_tsteps, opt_dt]):
        raise ValueError("stable interpolation needs all four trajectory channels and timing tensors")
    if raw[0].ndim != 3 or output[0].ndim != 3:
        raise ValueError("stable interpolation needs batch/horizon/joint arrays")
    batch, horizon, dof = raw[0].shape
    out_batch, out_horizon, out_dof = output[0].shape
    if (min(batch, dof, out_horizon) <= 0 or horizon < 2
            or (out_batch, out_dof) != (batch, dof)
            or any(value.shape != raw[0].shape for value in raw)
            or any(value.shape != output[0].shape for value in output)
            or traj_tsteps.numel() != batch or opt_dt.numel() != batch):
        raise ValueError("stable interpolation shape mismatch")
    if (any(value.dtype != torch.float32 for value in [*raw, *output, opt_dt])
            or traj_tsteps.dtype != torch.int32
            or any(value.device != raw[0].device or not value.is_contiguous()
                   for value in [*raw, *output, traj_tsteps, opt_dt])
            or not math.isfinite(scalar_dt) or scalar_dt <= 0):
        raise ValueError("stable interpolation requires contiguous float32 data and int32 steps on one device")
    # The shape/dtype guards do not synchronize CUDA. As with the upstream
    # scalar ABI, callers capturing this wrapper must pass raw_dt as a float.
    # Positive step counts/opt_dt are owned by cuRobo's original retiming helper.
    wp.init()
    stream = wp.stream_from_torch(raw[0].device) if raw[0].device.type == "cuda" else None
    wp.launch(kernel=_interpolate, dim=batch * out_horizon * dof,
              inputs=[*[wp.from_torch(value.view(-1)) for value in [*raw, *output]],
                      wp.from_torch(opt_dt.view(-1)), wp.from_torch(traj_tsteps.view(-1)),
                      horizon, dof, out_horizon, scalar_dt],
              device=wp.device_from_torch(raw[0].device), stream=stream)
    return out_traj
