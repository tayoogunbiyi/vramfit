"""Small config readers; defaults and feature semantics belong to adapters."""

from typing import Mapping

from vramfit.errors import InvalidModelConfigError


def positive_int(config: Mapping[str, object], name: str) -> int:
    value = config.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise InvalidModelConfigError(f"{name} must be a positive integer.")
    return value


def boolean(config: Mapping[str, object], name: str, default: bool) -> bool:
    value = config.get(name, default)
    if not isinstance(value, bool):
        raise InvalidModelConfigError(f"{name} must be a boolean.")
    return value
