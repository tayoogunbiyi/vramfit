"""Full-attention Mistral semantics (Transformers v4.51.0 Mistral modules)."""

from typing import Mapping

from vramfit.adapters._config import boolean
from vramfit.adapters._dense import DenseDecoderAdapter
from vramfit.errors import UnsupportedArchitectureError


class MistralAdapter(DenseDecoderAdapter):
    model_type = "mistral"
    architecture = "MistralForCausalLM"
    default_kv_heads = 8

    def _check_features(self, config: Mapping[str, object]) -> None:
        super()._check_features(config)
        if boolean(config, "attention_bias", False) or boolean(config, "mlp_bias", False):
            raise UnsupportedArchitectureError(self.model_type, (
                "Mistral attention and MLP projections must be bias-free.",
            ))

    def _uses_sliding_window(self, config: Mapping[str, object]) -> bool:
        # MistralConfig defaults to 4096: only explicit null disables the window.
        return config.get("sliding_window", 4096) is not None or boolean(
            config, "use_sliding_window", False
        )

    def _has_bias(self, config: Mapping[str, object], projection: str) -> bool:
        return False
