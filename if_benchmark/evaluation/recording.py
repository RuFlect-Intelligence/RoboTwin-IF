"""Passive rollout video and exact native action-boundary recording."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

from if_benchmark.manifest.io import atomic_write, file_hash

RECORDING = {
    'version': 'openwam_rollout_recording_v1',
    'video': {'fps': 25, 'cameras': ['head_camera', 'left_camera', 'right_camera'],
              'layout': 'horizontal', 'codec': 'libx264', 'crf': 23},
    'raw_actions': 'normalized80_physical20_dispatched20_command16_v1',
}
RECORDING_FILES = ('rollout.mp4', 'rollout.tmp.mp4', 'video_frames.jsonl',
                   'video_encoder.log', 'raw_actions.npz', 'recording.json')


def recording_contract():
    return deepcopy(RECORDING)


class MethodTap:
    """Restore the exact previous instance attribute, including absence."""
    def __init__(self, owner, name, wrapper):
        self.owner, self.name = owner, name
        self.existed = name in vars(owner)
        self.previous = vars(owner).get(name)
        self.original = getattr(owner, name)
        setattr(owner, name, wrapper(self.original))

    def close(self):
        if self.existed:
            setattr(self.owner, self.name, self.previous)
        else:
            delattr(self.owner, self.name)


class RawActionRecorder:
    def __init__(self, model, directory, metadata, get_step):
        self.path = Path(directory) / 'raw_actions.npz'
        self.metadata, self.get_step = metadata, get_step
        self.normalized, self.physical, self.chunk_steps, self.chunk_starts = [], [], [], []
        self.native, self.absolute, self.commands, self.steps, self.chunks, self.offsets, self.latencies = ([] for _ in range(7))
        self.offset, self.last_prediction = 0, None
        self.model, self.taps = model, []
        if hasattr(model, '_if_action_recorder'):
            raise ValueError('An action recorder is already attached')
        server = getattr(model, 'server', model)
        normalizer = server.engine.architecture.normalizer
        try:
            self.taps.append(MethodTap(normalizer, 'unnormalize', self._normalizer_wrapper))
            self.taps.append(MethodTap(server, 'predict', self._predict_wrapper))
            model._if_action_recorder = self
        except BaseException:
            for tap in reversed(self.taps):
                tap.close()
            raise

    def _normalizer_wrapper(self, original):
        def recorded(value, *args, **kwargs):
            raw = np.array(value, copy=True)
            result = original(value, *args, **kwargs)
            physical = np.array(result, copy=True)
            if raw.ndim != 2 or raw.shape[1] != 80 or physical.shape != (len(raw), 20):
                raise ValueError(f'Unexpected native action shapes: {raw.shape}, {physical.shape}')
            if not np.isfinite(raw).all() or not np.isfinite(physical).all():
                raise ValueError('Nonfinite native action chunk')
            self.normalized.append(raw); self.physical.append(physical)
            self.chunk_steps.append(self.get_step()); self.chunk_starts.append(len(self.commands))
            self.offset = 0
            return result
        return recorded

    def _predict_wrapper(self, original):
        def recorded(*args, **kwargs):
            result = original(*args, **kwargs)
            self.last_prediction = {'action': np.array(result['action'], dtype=np.float32, copy=True),
                                    'latency_ms': result.get('latency_ms', float('nan'))}
            return result
        return recorded

    def dispatched(self, absolute_eef20, command_ee16):
        if self.last_prediction is None or not self.normalized:
            raise ValueError('Dispatched action has no native generation evidence')
        self.native.append(self.last_prediction['action'].copy())
        self.absolute.append(np.array(absolute_eef20, copy=True))
        self.commands.append(np.array(command_ee16, copy=True))
        self.steps.append(self.get_step()); self.chunks.append(len(self.normalized) - 1)
        self.offsets.append(self.offset); self.offset += 1
        self.latencies.append(self.last_prediction['latency_ms'])

    def close(self):
        for tap in reversed(self.taps):
            tap.close()
        del self.model._if_action_recorder
        def stack(rows, shape):
            return np.stack(rows) if rows else np.empty(shape, dtype=np.float32)
        np.savez_compressed(self.path,
            normalized_chunks=stack(self.normalized, (0, 0, 80)),
            physical_chunks_eef20=stack(self.physical, (0, 0, 20)),
            chunk_start_physics_step=np.asarray(self.chunk_steps, dtype=np.int64),
            chunk_start_action_index=np.asarray(self.chunk_starts, dtype=np.int64),
            native_actions_eef20=stack(self.native, (0, 20)),
            absolute_actions_eef20=stack(self.absolute, (0, 20)),
            commands_ee16=stack(self.commands, (0, 16)),
            action_physics_step=np.asarray(self.steps, dtype=np.int64),
            action_chunk_index=np.asarray(self.chunks, dtype=np.int64),
            action_chunk_offset=np.asarray(self.offsets, dtype=np.int64),
            predict_latency_ms=np.asarray(self.latencies), metadata_json=json.dumps(self.metadata))
        return {'chunks': len(self.normalized), 'actions': len(self.commands)}


class VideoRecorder:
    def __init__(self, directory, dt, read_rgb, settings):
        self.directory, self.dt, self.read_rgb = Path(directory), dt, read_rgb
        self.settings = settings
        self.stride = round(1 / (dt * settings['fps']))
        if self.stride < 1 or abs(self.stride * dt * settings['fps'] - 1) > 1e-7:
            raise ValueError('Video FPS must divide the physics rate exactly')
        if shutil.which('ffmpeg') is None:
            raise RuntimeError('MP4 recording requires ffmpeg')
        self.proc, self.stderr = None, None
        self.frames = (self.directory / 'video_frames.jsonl').open('w')
        self.count, self.last_step, self.shape = 0, None, None

    def capture(self, step, *, force=False):
        if self.last_step == step or (not force and step % self.stride):
            return
        views = self.read_rgb()
        frames = [np.asarray(views[c]) for c in self.settings['cameras']]
        if any(x.dtype != np.uint8 or x.ndim != 3 or x.shape[2] != 3 for x in frames):
            raise ValueError('Video cameras must provide uint8 RGB')
        frame = np.ascontiguousarray(np.concatenate(frames, axis=1))
        height, width, _ = frame.shape
        if self.proc is None:
            if width % 2 or height % 2:
                raise ValueError('MP4 yuv420p dimensions must be even')
            self.shape = frame.shape
            self.stderr = (self.directory / 'video_encoder.log').open('wb')
            self.proc = subprocess.Popen([
                'ffmpeg', '-nostdin', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pixel_format', 'rgb24',
                '-video_size', f'{width}x{height}', '-framerate', str(self.settings['fps']), '-i', '-',
                '-an', '-c:v', 'libx264', '-threads', '1', '-preset', 'veryfast', '-crf', str(self.settings['crf']),
                '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(self.directory / 'rollout.tmp.mp4')],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.stderr)
        if frame.shape != self.shape:
            raise ValueError('Camera shape changed during recording')
        self.proc.stdin.write(frame.tobytes())
        self.frames.write(json.dumps({'frame': self.count, 'physics_step': step, 'simulation_seconds': step * self.dt}) + '\n')
        self.count += 1; self.last_step = step

    def close(self):
        self.frames.close()
        if self.proc is None:
            raise RuntimeError('No video frames captured')
        try:
            self.proc.stdin.close()
            code = self.proc.wait(timeout=30)
            if code != 0:
                raise RuntimeError(f'MP4 encoder exited {code}; see video_encoder.log')
            (self.directory / 'rollout.tmp.mp4').replace(self.directory / 'rollout.mp4')
        finally:
            if self.proc.poll() is None:
                self.proc.kill(); self.proc.wait()
            self.stderr.close()
        return {'frames': self.count, 'fps': self.settings['fps'], 'shape': list(self.shape),
                'last_physics_step': self.last_step, 'cameras': self.settings['cameras']}


class EpisodeRecorder:
    def __init__(self, directory, env, model, metadata, get_step, render_update, initial_obs, contract):
        if contract != RECORDING:
            raise ValueError('Unknown rollout recording contract')
        self.directory, self.contract, self.get_step = Path(directory), contract, get_step
        self.initial_obs = initial_obs
        self.raw = RawActionRecorder(model, directory, metadata, get_step)
        self.video = None
        try:
            def read_rgb():
                if self.initial_obs is not None:
                    obs = self.initial_obs; self.initial_obs = None
                    return {c: obs['observation'][c]['rgb'] for c in contract['video']['cameras']}
                from if_benchmark.scene_validation.expert_capture import capture_selected_rgb
                render_update()
                return capture_selected_rgb(env.cameras, contract['video']['cameras'])
            self.video = VideoRecorder(directory, float(env.scene.get_timestep()), read_rgb, contract['video'])
            self.video.capture(0)
        except BaseException:
            self.raw.close()
            if self.video is not None:
                try:
                    self.video.close()
                except Exception:
                    pass
            raise

    def close(self):
        errors = []
        details = {'contract': self.contract}
        try:
            self.video.capture(self.get_step(), force=True)
        except Exception as error:
            errors.append(str(error))
        for key, recorder in (('video', self.video), ('raw_actions', self.raw)):
            try:
                details[key] = recorder.close()
            except Exception as error:
                errors.append(str(error))
        details['complete'] = not errors
        details['errors'] = errors
        atomic_write(self.directory / 'recording.json', details)
        if errors:
            raise RuntimeError(f'Incomplete rollout recording: {errors}')
        return details


def recording_evidence(directory):
    return [{'path': name, 'sha256': file_hash(Path(directory) / name)}
            for name in RECORDING_FILES if (Path(directory) / name).is_file()]
