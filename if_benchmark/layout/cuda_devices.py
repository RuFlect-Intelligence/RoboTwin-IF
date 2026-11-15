"""Operational CUDA device-selector validation shared by M12 entrypoints."""

from __future__ import annotations

import re
from typing import Any, Sequence


_SIGNED_DECIMAL = re.compile(r"[+-]?[0-9]+\Z")
_CANONICAL_NONNEGATIVE_DECIMAL = re.compile(r"(?:0|[1-9][0-9]*)\Z")


def normalize_cuda_device_selectors(
    values: Sequence[Any],
    *,
    label: str = "cuda_devices",
) -> tuple[str, ...]:
    """Trim and validate CUDA indexes/UUID selectors without rewriting UUIDs.

    Numeric selectors use an integer uniqueness key.  Non-canonical numeric
    spellings such as ``00``, signs, and negative indexes are rejected instead
    of silently naming the same CUDA device more than once.  CUDA UUID and MIG
    selectors remain opaque strings.
    """

    selectors: list[str] = []
    keys: set[tuple[str, int | str]] = set()
    for raw_value in values:
        selector = str(raw_value).strip()
        if not selector:
            raise ValueError(f"{label} must contain non-empty device ids")
        if any(character.isspace() for character in selector):
            raise ValueError(
                f"{label} device ids must not contain whitespace: {selector!r}"
            )
        if _SIGNED_DECIMAL.fullmatch(selector):
            if not _CANONICAL_NONNEGATIVE_DECIMAL.fullmatch(selector):
                raise ValueError(
                    f"{label} numeric device ids must be canonical "
                    "non-negative decimals without signs or leading zeros"
                )
            key: tuple[str, int | str] = ("index", int(selector))
        else:
            key = ("selector", selector)
        if key in keys:
            raise ValueError(f"{label} must contain unique device ids")
        keys.add(key)
        selectors.append(selector)
    return tuple(selectors)


__all__ = ["normalize_cuda_device_selectors"]
