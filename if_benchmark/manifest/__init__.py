"""Milestone 16 publication graph and scene-group split APIs."""

from .contracts import ManifestConfig, SkillKey, TaskAdmission

__all__ = ["ManifestConfig", "SkillKey", "TaskAdmission", "build_manifest", "check_manifest", "freeze_manifest", "validate_manifest"]


def __getattr__(name):
    # Reading frozen JSON/YAML must not initialize SAPIEN or scene execution.
    if name in {"build_manifest", "check_manifest", "freeze_manifest", "validate_manifest"}:
        from . import core
        return getattr(core, name)
    raise AttributeError(name)
