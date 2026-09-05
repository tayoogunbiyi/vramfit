"""Pure family-specific interpretation of config and checkpoint evidence."""

from typing import Mapping, Protocol

from vramfit.models import (
    DecoderSpec,
    ParameterEstimate,
    ParameterSummary,
    TensorInventory,
)


class ModelAdapter(Protocol):
    """Implement this protocol to support a compatible model family.

    Methods must not mutate their inputs, perform I/O, or calculate workload
    memory. The orchestration layer calls normalize first, then needs_headers,
    fetches headers if needed, and finally calls resolve_parameters.
    """

    @property
    def model_type(self) -> str:
        """Exact config model_type handled by this adapter."""
        ...

    def normalize(self, config: Mapping[str, object]) -> DecoderSpec:
        """Validate topology/features and interpret fields and family defaults.

        Raise InvalidModelConfigError for malformed/inconsistent inputs and
        UnsupportedArchitectureError for unsupported semantics. Convert shared
        DecoderSpec validation errors into InvalidModelConfigError here.
        """
        ...

    def needs_headers(
        self, config: Mapping[str, object], summary: ParameterSummary | None
    ) -> bool:
        """Whether summary evidence needs a complete tensor inventory."""
        ...

    def resolve_parameters(
        self,
        config: Mapping[str, object],
        summary: ParameterSummary | None,
        headers: TensorInventory | None,
    ) -> ParameterEstimate:
        """Interpret evidence, account for sharing/buffers, or use a tested formula.

        None means unavailable, not zero. Return an explained unknown estimate
        when evidence is insufficient and no family fallback is implemented.
        Hub transport failures are handled by the caller, not by this method.
        """
        ...
