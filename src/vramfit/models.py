from dataclasses import dataclass, field
from typing import Literal, Mapping

ParameterSource = Literal[
    "hub_safetensors", "safetensors_headers", "architecture", "unknown"
]


@dataclass(frozen=True)
class DecoderSpec:
    """Validated geometry for a uniform, full-attention text decoder.

    Adapters establish that the model fits this support boundary before returning
    this type. Provenance maps normalized field names to source fields or the
    family-specific default/derivation used to obtain them.
    """

    model_type: str
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_context_length: int
    provenance: Mapping[str, str] = field(default_factory=dict)
    assumptions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "num_layers",
            "hidden_size",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "max_context_length",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("Attention heads must be divisible by KV heads.")


@dataclass(frozen=True)
class ParameterSummary:
    """Hub-reported counts, not necessarily unique learned or stored elements."""

    total: int
    parameters_by_dtype: Mapping[str, int]


@dataclass(frozen=True)
class TensorMetadata:
    """One selected checkpoint tensor, without its weight payload.

    Shape () denotes one scalar element. Stored bytes describe the tensor's
    payload in its shard, not its eventual runtime allocation.
    """

    name: str
    shape: tuple[int, ...]
    dtype: str
    stored_bytes: int


@dataclass(frozen=True)
class TensorInventory:
    """Complete header inventory for one selected checkpoint, across its shards."""

    tensors: tuple[TensorMetadata, ...]


@dataclass(frozen=True)
class ParameterEstimate:
    """Adapter-interpreted learned count and its qualifications.

    An unknown result must explain why in warnings. Recognized checkpoint
    buffers are reported separately and excluded from learned_parameter_count;
    their runtime allocation remains unmodelled.
    """

    learned_parameter_count: int | None
    source: ParameterSource
    assumptions: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    checkpoint_buffers: tuple[TensorMetadata, ...] = ()

    def __post_init__(self) -> None:
        if self.source not in (
            "hub_safetensors",
            "safetensors_headers",
            "architecture",
            "unknown",
        ):
            raise ValueError(f"Unknown parameter source: {self.source}")
        count = self.learned_parameter_count
        if count is None:
            if self.source != "unknown" or not self.warnings:
                raise ValueError(
                    "An unknown count requires source='unknown' and a warning."
                )
        elif type(count) is not int or count <= 0 or self.source == "unknown":
            raise ValueError(
                "A known count requires a positive integer and a known source."
            )
