"""Dependency-free runtime failure types shared by workers and adapters."""


class PhysicalCollisionValidationError(RuntimeError):
    """The exact-mesh validator failed before producing collision evidence."""
