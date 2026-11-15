"""Same-rollout M15 expert staging, with no simulator or training dependencies.

This is deliberately *not* a LeRobot dataset writer. A collector temporarily
wraps one Scene instance's step: observe the measured state and the targets
about to enter physics, then perform exactly the original step once. Video
observations use a fixed integer number of physics steps. A short final
interval and the genuine terminal observation are retained but never labeled
as full-rate training actions. The v3 exporter consumes this explicit staging
contract after M15, not the collector, has admitted an attempt.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np


STAGING_VERSION = "m15_expert_capture_staging_v1"
DEFAULT_CAMERAS = ("head_camera", "left_camera", "right_camera")
DEFAULT_FPS = 25
CUDA_RGB_TRANSFER = "sapien_cuda_alias_packed_uint8_v1"


class ExpertCaptureError(RuntimeError):
    """Incomplete or semantically invalid capture; never silently train it."""


def file_sidecar(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path.relative_to(relative_to) if relative_to else path.resolve()),
        "sha256": digest.hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def capture_options(options: Mapping[str, Any] | None) -> dict[str, Any]:
    value = dict(options or {})
    fps = value.get("fps", DEFAULT_FPS)
    if type(fps) is not int or fps <= 0:
        raise ExpertCaptureError("expert_capture_fps_invalid")
    cameras = value.get("cameras", DEFAULT_CAMERAS)
    if isinstance(cameras, (str, bytes)) or not isinstance(cameras, Sequence):
        raise ExpertCaptureError("expert_capture_cameras_invalid")
    cameras = list(cameras)
    if (not cameras or len(set(cameras)) != len(cameras)
            or any(not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_]+", name)
                   for name in cameras)):
        raise ExpertCaptureError("expert_capture_cameras_invalid")
    return {"fps": fps, "cameras": cameras}


def capture_selected_rgb(
    cameras: Any, camera_names: Sequence[str], *, readback: str = "cpu", verify: bool = False,
) -> dict[str, np.ndarray]:
    """Read only requested sensors, without stepping physics or updating poses.

    The caller owns wrist-pose synchronization and scene.update_render(). Keep
    the Camera wrapper's native sensor ordering and exact Color-to-RGB mapping;
    the unused alpha channel and unrequested views need no CPU readback.
    """
    if readback not in {"cpu", "cuda"}:
        raise ExpertCaptureError(f"expert_capture_readback_invalid:{readback}")
    selected = selected_rgb_sensors(cameras, camera_names)
    for sensor in selected.values():
        sensor.take_picture()
    if readback == "cuda":
        result = _cuda_rgb(selected)
        if verify:
            # Read these same rendered buffers, without another render, RNG
            # draw, camera-pose update, or physics step. Diagnostic opt-in only.
            for name, sensor in selected.items():
                expected = _cpu_rgb(sensor, name)
                if not np.array_equal(result[name], expected):
                    raise ExpertCaptureError(f"expert_capture_readback_mismatch:{name}")
        return result
    return {name: _cpu_rgb(sensor, name) for name, sensor in selected.items()}


def selected_rgb_sensors(cameras: Any, camera_names: Sequence[str]) -> dict[str, Any]:
    """Validate the complete selection before touching any renderer state."""
    names = list(camera_names)
    if not names or len(set(names)) != len(names):
        raise ExpertCaptureError("expert_capture_cameras_invalid")
    available: dict[str, Any] = {}
    if getattr(cameras, "collect_wrist_camera", False):
        for name in ("left_camera", "right_camera"):
            sensor = getattr(cameras, name, None)
            if sensor is not None:
                available[name] = sensor
    for name, sensor in zip(cameras.static_camera_name, cameras.static_camera_list):
        if name != "head_camera" or getattr(cameras, "collect_head_camera", False):
            if name in available:
                raise ExpertCaptureError(f"expert_capture_duplicate_camera:{name}")
            available[name] = sensor
    for name in names:
        if name not in available:
            raise ExpertCaptureError(f"expert_capture_camera_missing:{name}")
    return {name: sensor for name, sensor in available.items() if name in names}


def _cpu_rgb(sensor: Any, name: str) -> np.ndarray:
    rgba = np.asarray(sensor.get_picture("Color"))
    if rgba.ndim != 3 or rgba.shape[2] < 3:
        raise ExpertCaptureError(f"expert_capture_rgb_invalid:{name}")
    return np.ascontiguousarray((rgba[:, :, :3] * 255).clip(0, 255).astype(np.uint8))


def cuda_buffer_tensor(buffer: Any):
    """Alias the renderer allocation on its own GPU, never an implicit CPU copy.

    With the deployed Torch 2.4 runtime, as_tensor(CudaArray) uses the default
    CPU device. An explicit device is required even inside torch.cuda.device().
    The scene/camera owner must remain alive until the consumer has finished.
    """
    import torch

    device = int(buffer.cuda_id)
    if device < 0:
        raise ExpertCaptureError("expert_capture_cuda_buffer_device_invalid")
    tensor = torch.as_tensor(buffer, device=torch.device("cuda", device))
    if not tensor.is_cuda or tensor.device.index != device or tensor.data_ptr() != int(buffer.ptr):
        raise ExpertCaptureError("expert_capture_cuda_buffer_not_aliased")
    return tensor


def _cuda_rgb(selected: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Convert on the existing GPU, then perform one packed uint8 CPU transfer.

    Keep ordinary camera rendering and CPU-scene pose synchronization. SAPIEN
    RenderSystemGroup takes over pose updates; attaching it to these CPU
    physics scenes without a live pose buffer would freeze observations. Only the
    public per-camera CUDA readback API is used here; its RGBA buffers remain
    read-only. Torch is imported only for the explicitly requested GPU path.
    """
    import torch

    tensors, shapes = [], {}
    for name, sensor in selected.items():
        rgba = cuda_buffer_tensor(sensor.get_picture_cuda("Color"))
        if rgba.ndim != 3 or rgba.shape[2] < 3:
            raise ExpertCaptureError(f"expert_capture_rgb_invalid:{name}")
        shapes[name] = (*rgba.shape[:2], 3)
        tensors.append(rgba[:, :, :3].reshape(-1))
    # cat owns its storage even for a single camera, so in-place arithmetic
    # cannot modify a renderer-owned buffer. Different resolutions are legal.
    packed = torch.cat(tensors).mul_(255).clamp_(0, 255).to(torch.uint8).cpu().numpy()
    result: dict[str, np.ndarray] = {}
    offset = 0
    for name, shape in shapes.items():
        size = math.prod(shape)
        result[name] = packed[offset:offset + size].reshape(shape)
        offset += size
    return result


def _scalar(value: Any, label: str) -> float:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (1,) or not np.isfinite(array).all():
        raise ExpertCaptureError(f"expert_capture_scalar_invalid:{label}")
    return float(array[0])


def _joint_unit(joint: Any) -> str:
    kind = getattr(joint, "type", None)
    if kind is None and callable(getattr(joint, "get_type", None)):
        kind = joint.get_type()
    kind = str(kind).lower()
    if "revolute" in kind or "continuous" in kind:
        return "rad"
    if "prismatic" in kind:
        return "m"
    raise ExpertCaptureError(f"expert_capture_joint_type_unsupported:{kind}")


class RobotStateReader:
    """Read qpos/qvel, not Robot's misleading command-backed state getters.

    Main-vector order is left arm joints, left normalized gripper, right arm
    joints, right normalized gripper. Sizes and units come from the loaded
    robot. The normalized gripper action is the *actually applied* base finger
    drive target, not the possibly ahead-of-controller requested command.
    """

    def __init__(self, robot: Any) -> None:
        self.robot = robot
        self.arms: list[dict[str, Any]] = []
        names: list[str] = []
        units: list[str] = []
        finger_names: list[str] = []
        finger_units: list[str] = []
        arm_names: list[str] = []
        arm_units: list[str] = []
        for arm in ("left", "right"):
            entity = getattr(robot, f"{arm}_entity")
            active = list(entity.get_active_joints())
            joints = list(getattr(robot, f"{arm}_arm_joints"))
            fingers = list(getattr(robot, f"{arm}_gripper"))
            scale = np.asarray(getattr(robot, f"{arm}_gripper_scale"), dtype=float)
            if (not joints or not fingers or scale.shape != (2,)
                    or not np.isfinite(scale).all() or scale[0] == scale[1]):
                raise ExpertCaptureError(f"expert_capture_robot_contract_invalid:{arm}")
            selected = joints + [row[0] for row in fingers]
            indices = [active.index(joint) for joint in selected]
            if len(set(indices)) != len(indices):
                raise ExpertCaptureError(f"expert_capture_duplicate_joint:{arm}")
            joint_names = [f"{arm}.{joint.get_name()}" for joint in joints]
            joint_units = [_joint_unit(joint) for joint in joints]
            finger_names.extend(f"{arm}.{row[0].get_name()}" for row in fingers)
            finger_units.extend(_joint_unit(row[0]) for row in fingers)
            names.extend([*joint_names, f"{arm}.gripper_opening"])
            units.extend([*joint_units, "normalized_opening"])
            arm_names.extend(joint_names)
            arm_units.extend(joint_units)
            multiplier, offset = float(fingers[0][1]), float(fingers[0][2])
            if not math.isfinite(multiplier) or multiplier == 0 or not math.isfinite(offset):
                raise ExpertCaptureError(f"expert_capture_gripper_mapping_invalid:{arm}")
            self.arms.append({
                "arm": arm, "entity": entity, "joints": joints,
                "fingers": [row[0] for row in fingers], "indices": indices,
                "scale": scale.tolist(), "base_multiplier": multiplier,
                "base_offset": offset,
            })
        if len(set(names)) != len(names) or len(set(finger_names)) != len(finger_names):
            raise ExpertCaptureError("expert_capture_duplicate_feature_name")

        def feature(labels: list[str], measures: list[str]) -> dict[str, Any]:
            return {"dtype": "float32", "shape": [len(labels)], "names": labels, "units": measures}

        self.features = {
            "observation.state": feature(names, units),
            "action": feature(names, units),
            "observation.velocity": feature(names, [f"{unit}/s" for unit in units]),
            "observation.finger_position": feature(finger_names, finger_units),
            "observation.finger_velocity": feature(finger_names, [f"{unit}/s" for unit in finger_units]),
            "command.arm_position": feature(arm_names, arm_units),
            "command.arm_velocity": feature(arm_names, [f"{unit}/s" for unit in arm_units]),
            "command.finger_position": feature(finger_names, finger_units),
            "command.finger_velocity": feature(finger_names, [f"{unit}/s" for unit in finger_units]),
            "command.requested_gripper": feature(["left.gripper", "right.gripper"], ["normalized_opening"] * 2),
        }

    def read(self) -> dict[str, np.ndarray]:
        values: dict[str, list[float]] = {key: [] for key in self.features}
        for row in self.arms:
            qpos = np.asarray(row["entity"].get_qpos(), dtype=float).reshape(-1)
            qvel = np.asarray(row["entity"].get_qvel(), dtype=float).reshape(-1)
            indices = row["indices"]
            if qpos.shape != qvel.shape or max(indices) >= len(qpos):
                raise ExpertCaptureError("expert_capture_qpos_qvel_shape_changed")
            count = len(row["joints"])
            selected_qpos, selected_qvel = qpos[indices], qvel[indices]
            joint_targets = [_scalar(j.get_drive_target(), "joint_target") for j in row["joints"]]
            finger_targets = [_scalar(j.get_drive_target(), "finger_target") for j in row["fingers"]]
            span = row["scale"][1] - row["scale"][0]

            def opening(raw: float) -> float:
                return ((raw - row["base_offset"]) / row["base_multiplier"] - row["scale"][0]) / span

            # Do not clip physical overshoot out of measured state.
            values["observation.state"].extend([*selected_qpos[:count], opening(selected_qpos[count])])
            values["observation.velocity"].extend([
                *selected_qvel[:count], selected_qvel[count] / row["base_multiplier"] / span,
            ])
            commanded_opening = opening(finger_targets[0])
            if commanded_opening < -1e-5 or commanded_opening > 1 + 1e-5:
                raise ExpertCaptureError("expert_capture_gripper_target_out_of_range")
            values["action"].extend([*joint_targets, float(np.clip(commanded_opening, 0, 1))])
            values["observation.finger_position"].extend(selected_qpos[count:])
            values["observation.finger_velocity"].extend(selected_qvel[count:])
            values["command.arm_velocity"].extend(
                _scalar(j.get_drive_velocity_target(), "arm_velocity_target") for j in row["joints"])
            values["command.arm_position"].extend(joint_targets)
            values["command.finger_position"].extend(finger_targets)
            values["command.finger_velocity"].extend(
                _scalar(j.get_drive_velocity_target(), "finger_velocity_target") for j in row["fingers"])
            values["command.requested_gripper"].append(
                _scalar(getattr(self.robot, f"{row['arm']}_gripper_val"), "requested_gripper"))
        result = {key: np.asarray(value, dtype=np.float32) for key, value in values.items()}
        if any(not np.isfinite(value).all() for value in result.values()):
            raise ExpertCaptureError("expert_capture_nonfinite_robot_state")
        return result


class CleanVideoWriter:
    """Streaming ffmpeg writer: one input frame, one encoded frame, no labels."""

    def __init__(self, path: Path, *, width: int, height: int, fps: int) -> None:
        if not shutil.which("ffmpeg"):
            raise ExpertCaptureError("expert_capture_ffmpeg_missing")
        self.path, self.width, self.height, self.fps = path, width, height, fps
        self.frame_count = 0
        self._stderr = tempfile.TemporaryFile()
        self._closed = False
        try:
            self._process = subprocess.Popen([
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-n",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
                "-r", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
                "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv444p",
                "-threads", "1", "-movflags", "+faststart", str(path),
            ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self._stderr)
        except BaseException:
            self._stderr.close()
            raise

    def write(self, frame: np.ndarray) -> None:
        if self._closed or frame.dtype != np.uint8 or frame.shape != (self.height, self.width, 3):
            raise ExpertCaptureError("expert_capture_video_frame_invalid")
        try:
            self._process.stdin.write(memoryview(np.ascontiguousarray(frame)).cast("B"))
        except (BrokenPipeError, OSError) as exc:
            raise ExpertCaptureError("expert_capture_video_write_failed") from exc
        self.frame_count += 1

    def close(self) -> int:
        if self._closed:
            return self.frame_count
        self._closed = True
        try:
            try:
                self._process.stdin.close()
            except BrokenPipeError:
                pass
            try:
                code = self._process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)
                raise ExpertCaptureError("expert_capture_ffmpeg_close_timeout")
            if code:
                self._stderr.seek(0)
                detail = self._stderr.read(4096).decode("utf-8", errors="replace")
                raise ExpertCaptureError(f"expert_capture_ffmpeg_failed:{detail}")
            return self.frame_count
        finally:
            self._stderr.close()


class ExpertCapture:
    """Own one reversible Scene.step hook and one immutable attempt directory."""

    def __init__(
        self, scene: Any, *, output_dir: Path, state_reader: RobotStateReader,
        read_images: Callable[[], Mapping[str, np.ndarray]],
        metadata: Mapping[str, Any], options: Mapping[str, Any] | None = None,
        writer_factory: Callable[..., Any] = CleanVideoWriter,
        frame_observer: Callable[[Mapping[str, np.ndarray], int], None] | None = None,
    ) -> None:
        self.scene = scene
        self.output_dir = Path(output_dir).resolve()
        self.state_reader, self.read_images = state_reader, read_images
        self.metadata = deepcopy(dict(metadata))
        self.options = capture_options(options)
        self.fps = self.options["fps"]
        self.sim_dt = float(scene.get_timestep())
        if not math.isfinite(self.sim_dt) or self.sim_dt <= 0:
            raise ExpertCaptureError("expert_capture_sim_dt_invalid")
        ratio = 1 / (self.fps * self.sim_dt)
        self.stride = int(round(ratio))
        # PhysX stores dt as float32 (e.g. 0.00400000019), not exact decimal.
        if self.stride < 1 or not math.isclose(ratio, self.stride, rel_tol=1e-6, abs_tol=1e-6):
            raise ExpertCaptureError(f"expert_capture_fps_not_divisible:{self.fps}:{self.sim_dt}")
        self.writer_factory = writer_factory
        self.frame_observer = frame_observer
        self._writers: dict[str, Any] = {}
        self._frames: list[dict[str, Any]] = []
        self._controls: list[dict[str, Any]] = []
        self._steps = 0
        self._active = False
        self._closed = False
        self._inside = False
        self._failure: str | None = None
        self._result: dict[str, Any] | None = None
        self._finalization_interrupt: BaseException | None = None
        # Operational diagnostics stay out of capture metadata and its hashes.
        self._profile: dict[str, float] | None = (
            {} if os.environ.get("M15_PROFILE_TIMINGS") == "1" else None
        )

    def _record_timing(self, key: str, started: float | None) -> None:
        if started is not None and self._profile is not None:
            self._profile[key] = self._profile.get(key, 0.0) + time.perf_counter() - started

    def can_finalize_prefix(self) -> bool:
        """Whether an intact active capture contains a real full sample interval."""
        return bool(
            self._active and not self._closed and self._failure is None
            and self._steps >= self.stride
        )

    def begin(self) -> None:
        if self._active or self._closed:
            raise ExpertCaptureError("expert_capture_already_started")
        self.state_reader.read()  # Validate without advancing physics or rendering.
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self._old_step = self.scene.step
        instance_dict = getattr(self.scene, "__dict__", {})
        self._had_instance_step = "step" in instance_dict
        self._old_instance_step = instance_dict.get("step")
        self._hook = self._step
        try:
            self.scene.step = self._hook
            if self.scene.step is not self._hook:
                raise TypeError("instance step assignment was not retained")
        except (AttributeError, TypeError) as exc:
            raise ExpertCaptureError("expert_capture_scene_step_hook_unsupported") from exc
        self._active = True

    def _restore(self) -> None:
        if not self._active:
            return
        if self.scene.step is not self._hook:
            raise ExpertCaptureError("expert_capture_scene_step_hook_replaced")
        if self._had_instance_step:
            self.scene.step = self._old_instance_step
        else:
            del self.scene.step
        self._active = False

    def _append_frame(self, state: Mapping[str, Any], step: int) -> None:
        started = time.perf_counter() if self._profile is not None else None
        images = self.read_images()
        self._record_timing("image_read_seconds", started)
        frames: dict[str, np.ndarray] = {}
        for name in self.options["cameras"]:
            if name not in images:
                raise ExpertCaptureError(f"expert_capture_camera_missing:{name}")
            frame = np.asarray(images[name])
            if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 1:
                raise ExpertCaptureError(f"expert_capture_rgb_invalid:{name}")
            frames[name] = frame
        started = time.perf_counter() if self._profile is not None else None
        for name, frame in frames.items():
            if name not in self._writers:
                self._writers[name] = self.writer_factory(
                    self.output_dir / f"{name}.mp4", width=frame.shape[1], height=frame.shape[0], fps=self.fps)
            self._writers[name].write(frame)
        self._record_timing("video_write_seconds", started)
        self._frames.append({**{key: value.copy() for key, value in state.items()}, "step": step})
        if self.frame_observer is not None:
            started = time.perf_counter() if self._profile is not None else None
            self.frame_observer(frames, step)
            self._record_timing("review_write_seconds", started)

    def _step(self, *args: Any, **kwargs: Any) -> Any:
        if self._inside:
            raise ExpertCaptureError("expert_capture_recursive_scene_step")
        self._inside = True
        try:
            started = time.perf_counter() if self._profile is not None else None
            state = self.state_reader.read()
            self._record_timing("state_read_seconds", started)
            if self._steps % self.stride == 0:
                self._append_frame(state, self._steps)
            started = time.perf_counter() if self._profile is not None else None
            result = self._old_step(*args, **kwargs)
            self._record_timing("physics_step_seconds", started)
            self._controls.append({**state, "step": self._steps})
            self._steps += 1
            return result
        except BaseException as exc:
            self._failure = f"{type(exc).__name__}:{exc}"
            raise
        finally:
            self._inside = False

    def end(self, *, outcome: Any, completed: bool = True) -> dict[str, Any]:
        if self._closed:
            if self._result is None:
                raise ExpertCaptureError("expert_capture_finalization_failed")
            return deepcopy(self._result)
        try:
            return self._finish(outcome=outcome, completed=completed)
        finally:
            # Final-frame readback and encoder close can receive the attempt
            # alarm too. Finish best-effort cleanup, then preserve that exact
            # exception even if writing the diagnostic manifest also fails.
            if self._finalization_interrupt is not None:
                raise self._finalization_interrupt

    def finalized_result(self) -> dict[str, Any] | None:
        """Return persisted evidence even when finalization raised a deadline."""
        return deepcopy(self._result)

    def _finish(self, *, outcome: Any, completed: bool) -> dict[str, Any]:
        end_started = time.perf_counter() if self._profile is not None else None
        self._closed = True
        errors: list[str] = []

        def record_error(stage: str, exc: BaseException) -> None:
            errors.append(f"{stage}:{type(exc).__name__}:{exc}")
            if (isinstance(exc, (TimeoutError, KeyboardInterrupt, SystemExit))
                    and self._finalization_interrupt is None):
                self._finalization_interrupt = exc

        if self._failure:
            errors.append(self._failure)
        try:
            self._restore()
        except BaseException as exc:
            record_error("restore", exc)
        try:
            if self._finalization_interrupt is None:
                terminal = self.state_reader.read()
                # With zero completed steps there may already be a failed pre-step
                # frame at t=0. Avoid duplicate timestamps and never invent an action.
                if not self._frames or self._frames[-1]["step"] != self._steps:
                    self._append_frame(terminal, self._steps)
        except BaseException as exc:
            record_error("terminal", exc)
        cameras = {}
        for name, writer in self._writers.items():
            try:
                count = writer.close()
                cameras[name] = {
                    **file_sidecar(writer.path, relative_to=self.output_dir),
                    "width": writer.width, "height": writer.height, "channels": 3,
                    "color_space": "RGB", "fps": self.fps, "frame_count": count,
                }
                if count != len(self._frames):
                    errors.append(f"video_frame_count_mismatch:{name}")
            except BaseException as exc:
                record_error(f"video:{name}", exc)

        arrays: dict[str, np.ndarray] = {}
        for prefix, records in (("", self._frames), ("control.", self._controls)):
            for key, feature in self.state_reader.features.items():
                arrays[prefix + key] = np.asarray([row[key] for row in records], dtype=np.float32).reshape(
                    len(records), feature["shape"][0])
            indices = np.asarray([row["step"] for row in records], dtype=np.int64)
            arrays[prefix + "step_index"] = indices
            arrays[prefix + "timestamp"] = indices.astype(np.float64) * self.sim_dt
        arrays["video_timestamp"] = np.arange(len(self._frames), dtype=np.float64) / self.fps
        arrays["valid_action"] = arrays["step_index"] + self.stride <= self._steps
        arrays["control.valid_action"] = np.ones(len(self._controls), dtype=np.bool_)
        # Observation-only rows have no commanded action. The actual final
        # drive settings remain in the command logs, never a fabricated label.
        arrays["action"][~arrays["valid_action"]] = 0
        arrays_path = self.output_dir / "trajectory.npz"
        np.savez_compressed(arrays_path, **arrays)
        if self._steps == 0:
            errors.append("no_physics_steps")
        if not bool(arrays["valid_action"].any()):
            errors.append("no_complete_sample_interval")
        if set(cameras) != set(self.options["cameras"]):
            errors.append("camera_streams_incomplete")
        if isinstance(outcome, Mapping):
            outcome_details = deepcopy(dict(outcome))
            completed = completed and not bool(outcome.get("aborted"))
            outcome_name = outcome.get("classification")
        else:
            outcome_details = {"classification": outcome}
            outcome_name = outcome
        payload = {
            **self.metadata,
            "schema_version": STAGING_VERSION,
            "status": "complete" if completed and not errors else "incomplete",
            "outcome": outcome_name, "outcome_details": outcome_details, "errors": errors,
            "fps": self.fps, "sim_dt": self.sim_dt, "sample_step_stride": self.stride,
            "sample_period_s": self.stride * self.sim_dt,
            "frame_count": len(self._frames), "control_count": self._steps,
            "training_frame_count": int(arrays["valid_action"].sum()),
            "arrays": file_sidecar(arrays_path, relative_to=self.output_dir),
            "cameras": cameras, "features": deepcopy(self.state_reader.features),
            "timing": {
                "observation": "measured state and clean RGB immediately before the indexed physics step",
                "action": "absolute joint drive targets and normalized applied gripper targets for that step",
                "alignment": "observation at k, commands applied by original Scene.step k->k+1; no index shift",
                "control_log": "every completed physics step; includes intra-sample target changes",
                "policy_action_semantics": "instantaneous target at sampled step, not an interval-average or a held constant",
                "timestamp": "actual simulator elapsed seconds, zero at first execution step",
                "video_timestamp": "nominal frame_index/fps; final off-grid terminal duration is audit-only",
                "valid_action": "full stride remains; terminal observation and any short final interval are false",
                "capture_scope": "execution and terminal settle only; reset/setup/planning excluded",
            },
            "gripper": {
                "convention": "0=closed, 1=open, measured opening is not clipped",
                "mapping": [{key: row[key] for key in ("arm", "scale", "base_multiplier", "base_offset")}
                            for row in self.state_reader.arms],
                "requested_vs_applied": "action uses live finger drive target after rate limiting; requested command logged separately",
            },
            "observation_frame": "robot articulation joint coordinates; camera RGB in configured optical views",
            "ground_truth_policy_inputs": False,
            "canonical_dataset": False,
        }
        if outcome_details.get("capture_scope") == "skill_failure_prefix":
            # This is the terminal observation of the captured prefix, not the
            # later settled/checker state of the complete physical attempt.
            payload["timing"]["capture_scope"] = "skill_failure_prefix"
            payload["timing"]["capture_scope_description"] = (
                "execution prefix through skill failure; terminal settle, reset/setup/planning excluded"
            )
            payload["timing"]["control_log"] = (
                "every completed physics step in the captured prefix; terminal settle excluded"
            )
        payload["manifest_hash"] = hashlib.sha256(json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")).hexdigest()
        manifest_path = self.output_dir / "manifest.json"
        with manifest_path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        self._result = {**file_sidecar(manifest_path), "status": payload["status"],
                        "frame_count": payload["frame_count"], "training_frame_count": payload["training_frame_count"]}
        self._record_timing("end_total_seconds", end_started)
        if self._profile is not None:
            try:
                print("[m15-capture-timing] " + json.dumps({
                    "attempt_id": (self.metadata.get("identities") or {}).get("attempt_id"),
                    "control_count": self._steps, "frame_count": len(self._frames),
                    "camera_count": len(self._writers), "end_includes_terminal_frame": True,
                    **self._profile,
                }, sort_keys=True), file=sys.stderr, flush=True)
            except TimeoutError:
                raise
            except (OSError, ValueError):
                # Profiling is optional; a scheduler closing stderr must not
                # turn an already persisted capture into a failed attempt.
                pass
        return deepcopy(self._result)


def runtime_capture_metadata(joined: Mapping[str, Any], task: Mapping[str, Any],
                             spec: Mapping[str, Any]) -> dict[str, Any]:
    instruction = ((task.get("instruction") or {}).get("selected") or {}).get("text")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ExpertCaptureError("expert_capture_exact_instruction_missing")
    task_id = task.get("id")
    if task_id != spec.get("task_record_id"):
        raise ExpertCaptureError("expert_capture_task_id_mismatch")
    identities = {key: task.get(key) for key in ("scene_spec_id", "realized_family_id")}
    identities.update(task_record_id=task_id, attempt_id=spec.get("attempt_id"), seed=spec.get("seed"))
    for key in ("scene_spec_id", "realized_family_id", "task_record_id", "attempt_id"):
        if not isinstance(identities[key], str) or not identities[key]:
            raise ExpertCaptureError(f"expert_capture_identity_missing:{key}")
    if type(identities["seed"]) is not int or identities["seed"] not in (0, 1, 2):
        raise ExpertCaptureError("expert_capture_seed_invalid")
    sources = {key: str(joined[key]["artifact_hash"]) for key in ("scene", "unit", "realized")}
    tasks = [artifact for artifact in joined["tasks"] if artifact["task_record"]["id"] == task_id]
    if len(tasks) != 1:
        raise ExpertCaptureError("expert_capture_task_artifact_not_unique")
    sources["task"] = str(tasks[0]["artifact_hash"])
    for key in ("plan_hash", "config_hash", "source_scene_hash", "source_task_hash"):
        if key in spec:
            sources[key] = str(spec[key])
    route = (joined.get("routes") or {}).get(task_id)
    if route:
        route_hash = route.get("report_hash") or route.get("artifact_hash")
        if route_hash:
            sources["route"] = str(route_hash)
    return {"identities": identities, "source_hashes": sources, "instruction": instruction}
