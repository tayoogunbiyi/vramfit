"""Explicit adapter selection; no model-name guessing or plugin discovery."""

from collections.abc import Iterable

from vramfit.adapters.base import ModelAdapter
from vramfit.adapters.llama import LlamaAdapter
from vramfit.errors import UnsupportedArchitectureError


class DuplicateAdapterError(ValueError):
    """Registration would silently replace an existing family adapter."""


class AdapterRegistry:
    def __init__(self, adapters: Iterable[ModelAdapter] = ()) -> None:
        self._adapters: dict[str, ModelAdapter] = {}
        for adapter in adapters:
            self.register(adapter)

    def register(self, adapter: ModelAdapter) -> None:
        model_type = adapter.model_type
        if not isinstance(model_type, str) or not model_type.strip():
            raise ValueError("An adapter must declare a non-empty model_type.")
        if model_type in self._adapters:
            raise DuplicateAdapterError(
                f"Adapter already registered for {model_type!r}."
            )
        self._adapters[model_type] = adapter

    def get(self, model_type: str) -> ModelAdapter:
        try:
            return self._adapters[model_type]
        except KeyError:
            raise UnsupportedArchitectureError(
                model_type, ("No adapter is registered for this model type.",)
            ) from None


def default_registry() -> AdapterRegistry:
    """Build a fresh registry with the supported built-in families."""
    return AdapterRegistry([LlamaAdapter()])
