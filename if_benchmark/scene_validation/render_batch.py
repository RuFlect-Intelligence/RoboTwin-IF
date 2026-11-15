"""Batch the existing cameras while retaining live CPU physics poses.

SAPIEN's camera groups disable automatic CPU pose uploads. Every render update,
including review images after expert capture ends, must therefore explicitly
upload all rigid/articulated visual and camera poses until this Scene closes.
The physics Scene, shaders, camera parameters and sampling clock are unchanged.
GPU transform arithmetic can differ from CPU rasterization at rounding edges;
the Color-to-RGB conversion itself remains identical.
"""
from __future__ import annotations

from collections import defaultdict
import json
import os
import time
from typing import Any, Mapping

import numpy as np

from .expert_capture import CUDA_RGB_TRANSFER, ExpertCaptureError, _cpu_rgb, cuda_buffer_tensor
from .render_gate import RenderGate


BACKEND = "sapien_camera_batch_live_cpu_poses_v1"


def _profile_log(label, payload):
    # A scheduler/tee can close its output pipe during teardown. Diagnostics
    # must neither invalidate a good capture nor prevent scene cleanup.
    try:
        print(label + " " + json.dumps(payload, sort_keys=True), flush=True)
    except TimeoutError:
        # TimeoutError is an OSError too; never consume the attempt alarm.
        raise
    except (OSError, ValueError):
        pass


def pose_rows(entities, destination):
    """Read each current entity pose, including each articulation's links."""
    for index, entity in enumerate(entities):
        pose = entity.get_pose()
        destination[index, :3] = pose.p
        destination[index, 3:] = pose.q
    if not np.isfinite(destination).all():
        raise ExpertCaptureError("expert_batch_nonfinite_pose")


class SceneCameraBatch:
    """Own one scene's render hook and camera groups through physical teardown."""

    def __init__(self, scene: Any, sensors: Mapping[str, Any]):
        if not sensors:
            raise ExpertCaptureError("expert_batch_no_cameras")
        self.scene, self.sensors = scene, dict(sensors)
        self._original_update = scene.update_render
        self._had_update_attribute = "update_render" in vars(scene)
        self._previous_attribute = vars(scene).get("update_render")
        self._closed = False
        self._initialized = False
        self._failed = False
        self._cuda_poses = None
        self._entities = []
        self._hook = self.update_render
        self._batches = []
        self._group = None
        self._render_gate = RenderGate(enabled=False, scope='', device='')
        self._profile = os.environ.get("M15_PROFILE_TIMINGS", "0") == "1"
        self._timings = dict(update_count=0, update_seconds=0.0, frame_count=0,
                             submit_seconds=0.0, readback_wait_convert_seconds=0.0,
                             cuda_alias_seconds=0.0, gpu_convert_submit_seconds=0.0,
                             host_wait_copy_seconds=0.0, host_rgb_bytes=0)
        self._device_info = {}

    def initialize(self):
        """Call only after the physical environment has registered this owner."""
        if self._initialized or self._closed or self._failed:
            raise ExpertCaptureError("expert_batch_invalid_initialization")
        try:
            self._initialize_resources()
            self._initialized = True
            self.update_render()
        except BaseException:
            self._failed = True
            raise

    def _initialize_resources(self):
        import sapien
        import torch

        scene, sensors = self.scene, self.sensors
        bodies = list(scene.render_system.render_bodies)
        all_cameras = list(scene.get_cameras())
        if any(sensor not in all_cameras for sensor in sensors.values()):
            raise ExpertCaptureError("expert_batch_camera_outside_scene")
        cameras = list(sensors.values())
        self._entities = [body.entity for body in bodies] + [camera.entity for camera in cameras]
        self._poses = np.empty((len(self._entities), 7), dtype=np.float32)
        pose_rows(self._entities, self._poses)
        device = int(scene.render_system.device.cuda_id)
        if device < 0:
            raise ExpertCaptureError("expert_batch_cuda_renderer_required")
        self._torch = torch
        self._device = torch.device("cuda", device)
        self._device_info = dict(
            backend=BACKEND, cuda_mask=os.environ.get("CUDA_VISIBLE_DEVICES"),
            rgb_transfer_backend=CUDA_RGB_TRANSFER,
            renderer_device=str(scene.render_system.device),
            renderer_pci=scene.render_system.device.pci_string,
            camera_shapes={name: [sensor.height, sensor.width, 3]
                           for name, sensor in sensors.items()},
        )
        serialized = os.environ.get('M15_SERIALIZE_RENDER', '0')
        if serialized not in {'0', '1'}:
            raise ExpertCaptureError('expert_capture_serialize_render_invalid')
        self._render_gate = RenderGate(enabled=serialized == '1',
                                       scope=os.environ.get('M15_RENDER_GATE_SCOPE', ''),
                                       device=str(scene.render_system.device.pci_string))
        self._device_info['serialized_render'] = serialized == '1'
        with torch.cuda.device(self._device):
            self._stream = torch.cuda.current_stream(self._device)
            self._cuda_poses = torch.as_tensor(self._poses, device=self._device).clone()
            self._original_update()
            for index, body in enumerate(bodies):
                for shape in body.render_shapes:
                    shape.set_gpu_pose_batch_index(index)
            for index, camera in enumerate(cameras, len(bodies)):
                camera.set_gpu_pose_batch_index(index)
            self._group = sapien.render.RenderSystemGroup([scene.render_system])
            self._group.set_cuda_stream(self._stream.cuda_stream)
            self._group.set_cuda_poses(sapien.CudaArray(self._cuda_poses))
            # Install before gpuInit can disable auto-upload on any camera.
            # A failed initialization remains fail-closed until physical teardown.
            scene.update_render = self._hook
            by_shape = defaultdict(list)
            for name, sensor in sensors.items():
                by_shape[(sensor.width, sensor.height)].append((name, sensor))
            for entries in by_shape.values():
                batch = self._group.create_camera_group([sensor for _, sensor in entries], ["Color"])
                self._batches.append((tuple(name for name, _ in entries), batch))
            # Unselected sensors retain their ordinary CPU auto-upload. Do not
            # gpuInit/render/allocate buffers for unused front/world cameras.
        if self._profile:
            _profile_log("[m15-render-device]", self._device_info)

    def update_render(self):
        if self._closed or self._failed or not self._initialized or self._group is None:
            raise ExpertCaptureError("expert_batch_render_closed")
        started = time.monotonic() if self._profile else None
        # Preserve CPU camera-model matrices and ordinary light updates too.
        self._original_update()
        pose_rows(self._entities, self._poses)
        with self._torch.cuda.device(self._device), self._torch.cuda.stream(self._stream):
            self._cuda_poses.copy_(self._torch.from_numpy(self._poses))
            # SAPIEN rejects additions/removals after group creation.
            self._group.update_render()
        if started is not None:
            self._timings['update_count'] += 1
            self._timings['update_seconds'] += time.monotonic() - started

    def read_rgb(self, *, verify=False):
        if self._closed or self._failed or not self._initialized:
            raise ExpertCaptureError("expert_batch_render_closed")
        result = {}
        with self._torch.cuda.device(self._device), self._torch.cuda.stream(self._stream):
            for names, batch in self._batches:
                started = time.monotonic() if self._profile else None
                batch.take_picture()
                submitted = time.monotonic() if self._profile else None
                rgba = cuda_buffer_tensor(batch.get_picture_cuda("Color"))
                aliased = time.monotonic() if self._profile else None
                # RGB conversion stays on this GPU. Only packed uint8 pixels
                # cross to the host; never download the float RGBA allocation.
                packed = (rgba[..., :3] * 255).clamp_(0, 255).to(self._torch.uint8).contiguous()
                converted = time.monotonic() if self._profile else None
                pixels = packed.cpu().numpy()
                if started is not None:
                    self._timings['submit_seconds'] += submitted - started
                    # Includes waiting for the submitted GPU work, not just
                    # conversion/memcpy. No extra synchronization is inserted.
                    self._timings['readback_wait_convert_seconds'] += time.monotonic() - submitted
                    self._timings['cuda_alias_seconds'] += aliased - submitted
                    self._timings['gpu_convert_submit_seconds'] += converted - aliased
                    self._timings['host_wait_copy_seconds'] += time.monotonic() - converted
                    self._timings['host_rgb_bytes'] += pixels.nbytes
                for index, name in enumerate(names):
                    result[name] = pixels[index]
        if self._profile:
            self._timings['frame_count'] += 1
        ordered = {name: result[name] for name in self.sensors}
        if verify:
            # Compare transfer/conversion of these exact rendered buffers.
            # CPU-vs-GPU pose math requires a separate dynamic reference test.
            for name, sensor in self.sensors.items():
                if not np.array_equal(ordered[name], _cpu_rgb(sensor, name)):
                    raise ExpertCaptureError(f"expert_capture_readback_mismatch:{name}")
        return ordered

    def frame_scope(self):
        """Hold across pose upload, render, conversion and completed CPU copy."""
        return self._render_gate.frame()

    def close(self):
        """Release only at physical Scene teardown, never at capture.end()."""
        if self._closed:
            return
        self._closed = True
        if self.scene.update_render is self._hook:
            if self._had_update_attribute:
                self.scene.update_render = self._previous_attribute
            else:
                del self.scene.update_render
        self._batches.clear()
        self._group = None
        self._cuda_poses = None
        self._entities.clear()
        self.sensors.clear()
        self._hook = None
        self._original_update = None
        self._previous_attribute = None
        self.scene = None
        self._timings['render_queue_wait_seconds'] = self._render_gate.wait_seconds
        self._timings['serialized_frame_count'] = self._render_gate.frames
        self._render_gate.close()
        if self._profile:
            _profile_log("[m15-render-timing]", dict(self._device_info, **self._timings))
