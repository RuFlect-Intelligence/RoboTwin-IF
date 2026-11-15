"""Coordinate complete camera submissions on one physical GPU in one job.

The lock is node-local and keyed by collection output scope and renderer PCI
identity. Kernel file locks release on process death; files must not be
unlinked while another worker can still open the same gate.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import os
from pathlib import Path
import stat
import time


class RenderGate:
    def __init__(self, *, enabled: bool, scope: str, device: str):
        self._fd = None
        self.wait_seconds = 0.0
        self.frames = 0
        if not enabled:
            return
        if not scope or not device:
            raise ValueError('render gate requires collection scope and physical device')
        identity = hashlib.sha256(scope.encode()).hexdigest()[:24]
        root = Path('/tmp') / f'm15-render-{os.getuid()}-{identity}'
        root.mkdir(mode=0o700, exist_ok=True)
        info = root.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ValueError('render gate directory is not private to this user')
        name = hashlib.sha256(device.encode()).hexdigest()[:24] + '.lock'
        self._fd = os.open(root / name, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)

    @contextmanager
    def frame(self):
        if self._fd is None:
            yield
            return
        started = time.monotonic()
        try:
            # Keep acquisition inside finally: an attempt alarm may interrupt
            # the blocking syscall or arrive immediately after it succeeds.
            fcntl.flock(self._fd, fcntl.LOCK_EX)
            self.wait_seconds += time.monotonic() - started
            self.frames += 1
            yield
        finally:
            fcntl.flock(self._fd, fcntl.LOCK_UN)

    def close(self):
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
