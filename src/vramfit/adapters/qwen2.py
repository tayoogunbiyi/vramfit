"""Qwen2/Qwen2.5 semantics from Transformers v4.51.0's Qwen2 modules."""

from typing import Mapping

from vramfit.adapters._config import boolean
from vramfit.adapters._dense import DenseDecoderAdapter
from vramfit.errors import UnsupportedArchitectureError


class Qwen2Adapter(DenseDecoderAdapter):
    model_type = "qwen2"
    architecture = "Qwen2ForCausalLM"
    default_kv_heads = 32

    def _check_features(self, config: Mapping[str, object]) -> None:
        super()._check_features(config)
        if "attention_bias" in config or boolean(config, "mlp_bias", False):
            raise UnsupportedArchitectureError(self.model_type, (
                "Qwen2 requires fixed Q/K/V biases, no output or MLP biases; "
                "custom bias settings are unverified.",
            ))

    def _uses_sliding_window(self, config: Mapping[str, object]) -> bool:
        # A configured window is inert unless this family-specific flag enables it.
        return boolean(config, "use_sliding_window", False)

    def _has_bias(self, config: Mapping[str, object], projection: str) -> bool:
        return projection in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")
