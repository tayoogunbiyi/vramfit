"""Standard LlamaForCausalLM semantics (Transformers v4.51.0)."""

from typing import Mapping

from vramfit.adapters._config import boolean
from vramfit.adapters._dense import DenseDecoderAdapter


class LlamaAdapter(DenseDecoderAdapter):
    model_type = "llama"
    architecture = "LlamaForCausalLM"

    def _uses_sliding_window(self, config: Mapping[str, object]) -> bool:
        return config.get("sliding_window") is not None or boolean(
            config, "use_sliding_window", False
        )

    def _has_bias(self, config: Mapping[str, object], projection: str) -> bool:
        key = "attention_bias" if projection.startswith("self_attn") else "mlp_bias"
        return boolean(config, key, False)
