"""Coordinate pinned Hub evidence and pure adapter interpretation."""

from dataclasses import dataclass

from vramfit.adapters.registry import AdapterRegistry, default_registry
from vramfit.errors import InvalidModelConfigError
from vramfit.hub import ModelSnapshot, load_model_snapshot, load_tensor_inventory
from vramfit.models import DecoderSpec, ParameterEstimate


@dataclass(frozen=True)
class ModelInspection:
    snapshot: ModelSnapshot
    spec: DecoderSpec
    parameters: ParameterEstimate


def inspect_model(
    model_id: str, *, revision: str = "main", registry: AdapterRegistry | None = None,
) -> ModelInspection:
    snapshot = load_model_snapshot(model_id, revision=revision)
    model_type = snapshot.config.get("model_type")
    if not isinstance(model_type, str) or not model_type.strip():
        raise InvalidModelConfigError("config.json must declare a non-empty string model_type.")
    adapter = (default_registry() if registry is None else registry).get(model_type)
    spec = adapter.normalize(snapshot.config)
    headers = None
    if adapter.needs_headers(snapshot.config, snapshot.parameter_summary):
        headers = load_tensor_inventory(snapshot)
    parameters = adapter.resolve_parameters(snapshot.config, snapshot.parameter_summary, headers)
    return ModelInspection(snapshot, spec, parameters)
