"""Canonical machine-readable BenchmarkV1 Success Contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from if_benchmark.schema.serialization import artifact_hash, canonical_data


DEFAULT_SUCCESS_CONTRACT_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "success_contract"
    / "success_contract_v1.yml"
)
SUCCESS_CONTRACT_ID = "benchmarkv1_shared_success_contract"
SUCCESS_CONTRACT_VERSION = "benchmarkv1_success_contract_v1"


class SuccessContractError(ValueError):
    """A canonical success contract or profile is missing or malformed."""


@dataclass(frozen=True)
class SuccessContract:
    """Validated immutable view over the canonical checker configuration."""

    data: dict[str, Any]
    path: str
    hash: str

    @classmethod
    def load(
        cls, path: Path | str = DEFAULT_SUCCESS_CONTRACT_PATH
    ) -> "SuccessContract":
        source = Path(path)
        with source.open("r", encoding="utf-8") as stream:
            value = yaml.safe_load(stream)
        if not isinstance(value, Mapping):
            raise SuccessContractError(f"success contract must be a mapping: {source}")
        data = canonical_data(dict(value))
        _validate_contract(data)
        return cls(data=data, path=str(source), hash=artifact_hash(data))

    @property
    def contract_id(self) -> str:
        return str(self.data["contract_id"])

    @property
    def version(self) -> str:
        return str(self.data["success_contract_version"])

    @property
    def success_contract_hash(self) -> str:
        return self.hash

    @property
    def snapshot_version(self) -> str:
        return str(self.data["snapshot_version"])

    @property
    def predicate_registry_version(self) -> str:
        return str(self.data["predicate_registry_version"])

    @property
    def identity(self) -> dict[str, str]:
        return {
            "contract_id": self.contract_id,
            "success_contract_version": self.version,
            "success_contract_hash": self.hash,
        }

    def section(self, name: str) -> dict[str, Any]:
        value = self.data.get(name)
        if not isinstance(value, Mapping):
            raise SuccessContractError(f"missing success contract section {name!r}")
        return canonical_data(dict(value))


def _validate_contract(data: Mapping[str, Any]) -> None:
    if data.get("contract_id") != SUCCESS_CONTRACT_ID:
        raise SuccessContractError("unexpected success contract id")
    if data.get("success_contract_version") != SUCCESS_CONTRACT_VERSION:
        raise SuccessContractError("unsupported success contract version")
    required = {
        "snapshot_version",
        "predicate_registry_version",
        "active_contact",
        "stability",
        "hold",
        "articulation",
        "relation",
        "click_press",
        "orientation",
        "d4",
    }
    missing = sorted(required - set(data))
    if missing:
        raise SuccessContractError(f"missing success contract sections: {missing}")
    if float(data["active_contact"]["impulse_epsilon"]) <= 0.0:
        raise SuccessContractError("active contact impulse epsilon must be positive")
    stability = data["stability"]
    if not (
        0
        < int(stability["observation_window_steps"])
        <= int(stability["maximum_settle_steps"])
    ):
        raise SuccessContractError("invalid stability window")
    hold = data["hold"]
    if float(hold["pick_relative_translation_drift_exclusive_max_m"]) <= 0.0:
        raise SuccessContractError("invalid Pick translation-drift limit")
    if hold["pick_relative_rotation_constraint_enabled"] is not False:
        raise SuccessContractError("Pick rotation constraint must be disabled")
    if str(hold["pick_post_close_attachment"]) != "fixed_constraint":
        raise SuccessContractError("Pick post-close attachment must be fixed_constraint")
    if hold["pick_attachment_rotation_constraint_enabled"] is not False:
        raise SuccessContractError(
            "Pick attachment rotation constraint must be disabled"
        )
    for key in (
        "pick_attachment_drive_stiffness",
        "pick_attachment_drive_damping",
        "pick_attachment_residual_translation_exclusive_max_m",
    ):
        if float(hold[key]) <= 0.0:
            raise SuccessContractError(f"invalid Pick attachment setting: {key}")
    for relation, band in data["relation"]["distance_bands_m"].items():
        if len(band) != 2 or float(band[0]) < 0 or float(band[0]) > float(band[1]):
            raise SuccessContractError(f"invalid distance band for {relation}")


__all__ = [
    "DEFAULT_SUCCESS_CONTRACT_PATH",
    "SUCCESS_CONTRACT_ID",
    "SUCCESS_CONTRACT_VERSION",
    "SuccessContract",
    "SuccessContractError",
]
