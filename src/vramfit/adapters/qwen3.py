"""Dense Qwen3 semantics from Transformers v4.51.0's Qwen3 modules."""

from typing import Mapping

from vramfit.adapters._config import boolean
from vramfit.adapters._dense import DenseDecoderAdapter
from vramfit.errors import InvalidModelConfigError, UnsupportedArchitectureError
from vramfit.models import DecoderSpec


class Qwen3Adapter(DenseDecoderAdapter):
    model_type = "qwen3"
    architecture = "Qwen3ForCausalLM"
    default_kv_heads = 32
    default_head_dim = 128

    def _check_features(self, config: Mapping[str, object]) -> None:
        super()._check_features(config)
        if "head_dim" in config and config["head_dim"] is None:
            raise InvalidModelConfigError("Qwen3 head_dim must not be null.")
        if boolean(config, "mlp_bias", False):
            raise UnsupportedArchitectureError(self.model_type, (
                "Qwen3 MLP biases are not supported.",
            ))

    def _uses_sliding_window(self, config: Mapping[str, object]) -> bool:
        return boolean(config, "use_sliding_window", False)

    def _has_bias(self, config: Mapping[str, object], projection: str) -> bool:
        return projection.startswith("self_attn") and boolean(config, "attention_bias", False)

    def _tensor_shapes(self, config: Mapping[str, object], spec: DecoderSpec):
        weights, buffers = super()._tensor_shapes(config, spec)
        for layer in range(spec.num_layers):
            for name in ("q_norm", "k_norm"):
                weights[f"model.layers.{layer}.self_attn.{name}.weight"] = (spec.head_dim,)
        return weights, buffers
