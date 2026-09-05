"""Shared mechanics for dense rotary decoders; family policy stays in adapters."""

from math import prod
from typing import Mapping

from vramfit.adapters._config import boolean, positive_int
from vramfit.errors import InvalidModelConfigError, UnsupportedArchitectureError
from vramfit.models import (
    DecoderSpec,
    ParameterEstimate,
    ParameterSummary,
    TensorInventory,
)

FLOAT_DTYPES = frozenset({"F16", "BF16", "F32", "F64"})
EMBEDDING = "model.embed_tokens.weight"
OUTPUT = "lm_head.weight"


class DenseDecoderAdapter:
    model_type: str
    architecture: str
    default_kv_heads: int | None = None

    def normalize(self, config: Mapping[str, object]) -> DecoderSpec:
        if config.get("model_type") != self.model_type:
            raise InvalidModelConfigError(f"Expected model_type={self.model_type!r}.")
        self._check_features(config)
        fields = {
            "num_layers": "num_hidden_layers",
            "hidden_size": "hidden_size",
            "num_attention_heads": "num_attention_heads",
            "max_context_length": "max_position_embeddings",
        }
        values = {name: positive_int(config, key) for name, key in fields.items()}
        assumptions = []
        q_heads = values["num_attention_heads"]
        if config.get("num_key_value_heads") is None:
            values["num_key_value_heads"] = (
                self.default_kv_heads if "num_key_value_heads" not in config
                and self.default_kv_heads is not None else q_heads
            )
            fields["num_key_value_heads"] = f"{self.model_type} default: {values['num_key_value_heads']}"
            assumptions.append(f"Missing/null KV heads use {self.model_type}'s family default.")
        else:
            values["num_key_value_heads"] = positive_int(config, "num_key_value_heads")
            fields["num_key_value_heads"] = "num_key_value_heads"
        if config.get("head_dim") is None:
            if values["hidden_size"] % q_heads:
                raise InvalidModelConfigError(
                    "hidden_size must divide evenly by attention heads to derive head_dim."
                )
            values["head_dim"] = values["hidden_size"] // q_heads
            fields["head_dim"] = "hidden_size / num_attention_heads"
        else:
            values["head_dim"] = positive_int(config, "head_dim")
            fields["head_dim"] = "head_dim"
        if values["num_key_value_heads"] == 1 and q_heads > 1:
            raise UnsupportedArchitectureError(
                self.model_type, ("Multi-query attention is outside v1 support.",)
            )
        if values["head_dim"] % 2:
            raise InvalidModelConfigError("Rotary attention requires an even head_dim.")
        positive_int(config, "vocab_size")
        positive_int(config, "intermediate_size")
        for key in ("tie_word_embeddings", "attention_bias", "mlp_bias"):
            boolean(config, key, False)
        if config.get("rope_scaling") is not None:
            assumptions.append(
                "Context uses max_position_embeddings as configured; RoPE scaling "
                "does not multiply this limit again."
            )
        try:
            spec = DecoderSpec(
                model_type=self.model_type, **values,
                provenance=fields, assumptions=tuple(assumptions),
            )
        except ValueError as error:
            raise InvalidModelConfigError(str(error)) from error
        layer_types = config.get("layer_types")
        if layer_types is not None and len(layer_types) != spec.num_layers:
            raise InvalidModelConfigError("layer_types must have one entry per layer.")
        return spec

    def _check_features(self, config: Mapping[str, object]) -> None:
        reasons = []
        if config.get("architectures") != [self.architecture]:
            reasons.append(f"Requires an explicit {self.architecture} architecture.")
        for name in ("text_config", "vision_config", "audio_config", "encoder", "decoder"):
            if config.get(name) is not None:
                reasons.append(f"Contains a separate {name} component.")
        if boolean(config, "is_encoder_decoder", False):
            reasons.append("Encoder-decoder models are not supported.")
        if boolean(config, "add_cross_attention", False):
            reasons.append("Cross-attention is not supported.")
        if config.get("auto_map"):
            reasons.append("Custom model-code mappings have unverified semantics.")
        if config.get("quantization_config") is not None:
            reasons.append("Quantized checkpoints are not supported.")
        if self._uses_sliding_window(config):
            reasons.append("Sliding-window attention is not supported by this adapter.")
        if config.get("dual_chunk_attention_config") is not None:
            reasons.append("Dual-chunk attention is not supported.")
        for name in ("num_local_experts", "num_experts", "n_routed_experts", "n_shared_experts"):
            if config.get(name) not in (None, 0):
                reasons.append(f"Expert layers ({name}) are not supported.")
        layer_types = config.get("layer_types")
        if layer_types is not None:
            if not isinstance(layer_types, list):
                raise InvalidModelConfigError("layer_types must be a list.")
            if any(kind != "full_attention" for kind in layer_types):
                reasons.append("Only uniform full-attention layers are supported.")
        if config.get("attention_type") not in (None, "full_attention"):
            reasons.append("The declared attention type is not supported.")
        rope = config.get("rope_scaling")
        if rope is not None:
            if not isinstance(rope, dict):
                raise InvalidModelConfigError("rope_scaling must be an object.")
            if rope.get("rope_type", rope.get("type")) not in (
                "default", "linear", "dynamic", "yarn", "longrope", "llama3"
            ):
                reasons.append("Unrecognized RoPE scaling semantics.")
        if reasons:
            raise UnsupportedArchitectureError(self.model_type, tuple(reasons))

    def needs_headers(
        self, config: Mapping[str, object], summary: ParameterSummary | None
    ) -> bool:
        self.normalize(config)
        return (
            boolean(config, "tie_word_embeddings", False)
            or summary is None
            or summary.total <= 0
            or any(dtype not in FLOAT_DTYPES for dtype in summary.parameters_by_dtype)
        )

    def resolve_parameters(
        self,
        config: Mapping[str, object],
        summary: ParameterSummary | None,
        headers: TensorInventory | None,
    ) -> ParameterEstimate:
        spec = self.normalize(config)
        if headers is None:
            if summary is not None and not self.needs_headers(config, summary):
                return ParameterEstimate(
                    summary.total, "hub_safetensors",
                    assumptions=(
                        "Hub floating-element summary represents the standard untied "
                        f"{self.model_type} weights; individual tensors have not been inspected.",
                    ),
                )
            return self._unknown(
                "Tensor headers are required but unavailable; no config formula fallback."
            )

        expected, buffer_shapes = self._tensor_shapes(config, spec)
        tensors = {t.name: t for t in headers.tensors}
        if len(tensors) != len(headers.tensors):
            return self._unknown("Duplicate tensor names in the checkpoint inventory.")
        tied = boolean(config, "tie_word_embeddings", False)
        if tied:
            present_embeddings = {EMBEDDING, OUTPUT} & tensors.keys()
            if not present_embeddings:
                return self._unknown("The checkpoint contains neither tied embedding alias.")
            for name in {EMBEDDING, OUTPUT} - present_embeddings:
                del expected[name]
        missing = expected.keys() - tensors.keys()
        if missing:
            return self._unknown(f"Missing expected {self.model_type} tensor: {min(missing)}.")
        buffers = []
        for name, tensor in tensors.items():
            shape = expected.get(name, buffer_shapes.get(name))
            if shape is None or tensor.shape != shape:
                return self._unknown(f"Unrecognized {self.model_type} tensor or shape: {name}.")
            if tensor.dtype not in FLOAT_DTYPES:
                return self._unknown(f"Unsupported checkpoint dtype {tensor.dtype}: {name}.")
            if name in buffer_shapes:
                buffers.append(tensor)
        count = sum(prod(tensors[name].shape) for name in expected)
        assumptions = []
        if tied:
            if EMBEDDING in tensors and OUTPUT in tensors:
                count -= prod(tensors[EMBEDDING].shape)
            assumptions.append(
                "The runtime honors tie_word_embeddings and shares embedding/output weights."
            )
        warnings = []
        stored_count = sum(prod(t.shape) for t in headers.tensors)
        if summary is not None and summary.total != stored_count:
            warnings.append(
                "Hub summary differs from header elements; the validated headers were used."
            )
        if buffers:
            warnings.append(
                "Checkpoint rotary buffers are excluded; their runtime memory is unmodelled."
            )
        return ParameterEstimate(
            count, "safetensors_headers", assumptions=tuple(assumptions),
            warnings=tuple(warnings), checkpoint_buffers=tuple(buffers),
        )

    @staticmethod
    def _unknown(reason: str) -> ParameterEstimate:
        return ParameterEstimate(None, "unknown", warnings=(reason,))

    def _tensor_shapes(
        self, config: Mapping[str, object], spec: DecoderSpec
    ) -> tuple[dict[str, tuple[int, ...]], dict[str, tuple[int, ...]]]:
        """Expected layout for header validation, not a config-only count fallback."""
        h, d = spec.hidden_size, spec.head_dim
        q, kv = spec.num_attention_heads * d, spec.num_key_value_heads * d
        v, inter = positive_int(config, "vocab_size"), positive_int(config, "intermediate_size")
        weights = {EMBEDDING: (v, h), OUTPUT: (v, h), "model.norm.weight": (h,)}
        buffers = {"model.rotary_emb.inv_freq": (d // 2,)}
        for layer in range(spec.num_layers):
            prefix = f"model.layers.{layer}."
            weights[prefix + "input_layernorm.weight"] = (h,)
            weights[prefix + "post_attention_layernorm.weight"] = (h,)
            projections = {
                "self_attn.q_proj": (q, h), "self_attn.k_proj": (kv, h),
                "self_attn.v_proj": (kv, h), "self_attn.o_proj": (h, q),
                "mlp.gate_proj": (inter, h), "mlp.up_proj": (inter, h),
                "mlp.down_proj": (h, inter),
            }
            for name, shape in projections.items():
                weights[prefix + name + ".weight"] = shape
                if self._has_bias(config, name):
                    weights[prefix + name + ".bias"] = (shape[0],)
            buffers[prefix + "self_attn.rotary_emb.inv_freq"] = (d // 2,)
        return weights, buffers

    def _uses_sliding_window(self, config: Mapping[str, object]) -> bool:
        raise NotImplementedError

    def _has_bias(self, config: Mapping[str, object], projection: str) -> bool:
        raise NotImplementedError
