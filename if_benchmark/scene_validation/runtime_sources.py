"""Episode-local, source-guarded copies of immutable observation contracts.

Observation loops must read current physical state on every step. Parsing the
same already validated YAML on each of those steps adds no physical evidence.
Each physical episode owns fresh copies; source changes fail that episode rather
than silently changing its checker thresholds midway through execution.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
from typing import Any, Callable

from if_benchmark.success_contract import (
    DEFAULT_ALOHA_GRIPPER_PROFILE,
    SuccessContract,
    load_embodiment_profile,
)
from if_benchmark.success_contract.contract import DEFAULT_SUCCESS_CONTRACT_PATH


class EpisodeSourceChangedError(RuntimeError):
    """An observation source changed during one physical episode."""


def _stamp(path: Path) -> tuple[int, int, int, int, int]:
    value = path.stat()
    return (value.st_dev, value.st_ino, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)


class _EpisodeSource:
    def __init__(self, path: Path | str, loader: Callable[[Path], Any]):
        self.path = Path(path)
        self._invalid = False
        before = _stamp(self.path)
        contents = self.path.read_bytes()
        self._value = loader(self.path)
        self._stamp = _stamp(self.path)
        if before != self._stamp:
            self._changed()
        self._digest = hashlib.sha256(contents).digest()

    def _changed(self) -> None:
        self._invalid = True
        raise EpisodeSourceChangedError(
            f"episode_observation_source_changed:{self.path}"
        )

    def verify(self) -> None:
        if self._invalid:
            self._changed()
        try:
            current = _stamp(self.path)
            if current == self._stamp:
                return
            contents = self.path.read_bytes()
            after = _stamp(self.path)
        except OSError as exc:
            self._invalid = True
            raise EpisodeSourceChangedError(
                f"episode_observation_source_unavailable:{self.path}"
            ) from exc
        if current != after or hashlib.sha256(contents).digest() != self._digest:
            self._changed()
        # A touch or an atomic replacement with identical bytes is harmless.
        self._stamp = after

    def copy(self) -> Any:
        # Callers may freely mutate returned dictionaries without contaminating
        # later observations. The validated private value never leaves here.
        return deepcopy(self._value)


class EpisodeObservationSources:
    def __init__(
        self,
        *,
        contract_path: Path | str = DEFAULT_SUCCESS_CONTRACT_PATH,
        profile_path: Path | str = DEFAULT_ALOHA_GRIPPER_PROFILE,
    ):
        self._contract = _EpisodeSource(contract_path, SuccessContract.load)
        self._profile = _EpisodeSource(profile_path, load_embodiment_profile)

    def _verify(self) -> None:
        # Both sources belong to one episode, even if a caller currently needs
        # only one. Never continue with half of a changed source set.
        self._contract.verify()
        self._profile.verify()

    def success_contract(self) -> SuccessContract:
        self._verify()
        return self._contract.copy()

    def embodiment_profile(self) -> dict[str, Any]:
        self._verify()
        return self._profile.copy()
