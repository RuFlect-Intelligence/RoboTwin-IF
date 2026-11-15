"""Small ffmpeg-backed writer for benchmark diagnostic RGB videos."""

from __future__ import annotations

from pathlib import Path
import subprocess

import numpy as np


class DiagnosticVideoWriter:
    def __init__(self, path: Path, *, width: int, height: int, fps: int = 8) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary_path = self.path.with_name(f"{self.path.stem}.tmp.mp4")
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.frame_count = 0
        self.process = subprocess.Popen(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pixel_format", "rgb24",
                "-video_size", f"{self.width}x{self.height}",
                "-framerate", str(self.fps), "-i", "-", "-an",
                "-vcodec", "libx264", "-preset", "veryfast", "-crf", "23",
                "-pix_fmt", "yuv420p", str(self.temporary_path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray) -> None:
        value = np.ascontiguousarray(frame, dtype=np.uint8)
        if value.shape != (self.height, self.width, 3):
            raise RuntimeError(
                f"diagnostic_video_frame_shape_mismatch:{value.shape}:"
                f"{self.height}x{self.width}"
            )
        if self.process.stdin is None:
            raise RuntimeError("diagnostic_video_stdin_missing")
        self.process.stdin.write(value.tobytes())
        self.frame_count += 1

    def close(self) -> dict[str, object]:
        if self.process.stdin is not None and not self.process.stdin.closed:
            self.process.stdin.close()
        return_code = self.process.wait(timeout=30)
        stderr = (
            self.process.stderr.read().decode("utf-8", errors="replace")
            if self.process.stderr is not None
            else ""
        )
        if return_code != 0:
            raise RuntimeError(
                f"diagnostic_video_ffmpeg_failed:{return_code}:{stderr[-1000:]}"
            )
        if self.frame_count <= 0 or not self.temporary_path.is_file():
            raise RuntimeError("diagnostic_video_empty")
        self.temporary_path.replace(self.path)
        return {
            "path": str(self.path),
            "view": "head_camera",
            "fps": self.fps,
            "frame_count": self.frame_count,
            "duration_sec": self.frame_count / float(self.fps),
        }
