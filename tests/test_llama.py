import copy
import json
import unittest
from dataclasses import replace
from math import prod
from pathlib import Path

from vramfit.adapters.llama import LlamaAdapter
from vramfit.adapters.registry import default_registry
from vramfit.errors import InvalidModelConfigError, UnsupportedArchitectureError
from vramfit.models import ParameterSummary, TensorInventory, TensorMetadata

FIXTURES = Path(__file__).parent / "fixtures" / "llama"


def fixture(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class LlamaConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = LlamaAdapter()
        self.config = fixture("tinyllama.json")

    def test_tinyllama_geometry_and_summary_through_registry(self) -> None:
        adapter = default_registry().get("llama")
        spec = adapter.normalize(self.config)
        self.assertEqual(
            (spec.num_layers, spec.hidden_size, spec.num_attention_heads,
             spec.num_key_value_heads, spec.head_dim, spec.max_context_length),
            (22, 2048, 32, 4, 64, 2048),
        )
        evidence = fixture("provenance.json")["models"][0]["hub_summary"]
        summary = ParameterSummary(**evidence)
        self.assertFalse(adapter.needs_headers(self.config, summary))
        estimate = adapter.resolve_parameters(self.config, summary, None)
        self.assertEqual(estimate.learned_parameter_count, 1_100_048_384)
        self.assertEqual(estimate.source, "hub_safetensors")
        self.assertTrue(estimate.assumptions)
        self.assertEqual(spec.provenance["head_dim"], "hidden_size / num_attention_heads")

    def test_missing_and_null_kv_heads_use_documented_mha_default(self) -> None:
        self.config.pop("num_key_value_heads")
        for config in (self.config, self.config | {"num_key_value_heads": None}):
            with self.subTest(config=config):
                spec = self.adapter.normalize(config)
                self.assertEqual(spec.num_key_value_heads, 32)
                self.assertIn("default", spec.provenance["num_key_value_heads"])

    def test_explicit_head_dimension_overrides_hidden_division(self) -> None:
        spec = self.adapter.normalize(self.config | {"head_dim": 128})
        self.assertEqual(spec.head_dim, 128)
        self.assertEqual(spec.provenance["head_dim"], "head_dim")

    def test_context_does_not_double_apply_rope_scaling(self) -> None:
        spec = self.adapter.normalize(self.config | {
            "max_position_embeddings": 131072,
            "rope_scaling": {"rope_type": "llama3", "factor": 8.0},
        })
        self.assertEqual(spec.max_context_length, 131072)
        self.assertTrue(spec.assumptions)

    def test_tied_smollm_requires_headers_even_with_hub_summary(self) -> None:
        config = fixture("smollm2.json")
        summary = ParameterSummary(1_711_376_384, {"BF16": 1_711_376_384})
        self.assertTrue(self.adapter.needs_headers(config, summary))
        self.assertEqual(self.adapter.normalize(config).num_key_value_heads, 32)
        result = self.adapter.resolve_parameters(config, summary, None)
        self.assertIsNone(result.learned_parameter_count)

    def test_invalid_fields_fail_without_coercion(self) -> None:
        for updates in (
            {"model_type": "qwen2"}, {"hidden_size": None},
            {"num_hidden_layers": True}, {"num_attention_heads": "32"},
            {"num_key_value_heads": 3}, {"num_key_value_heads": 64},
            {"hidden_size": 2049}, {"head_dim": 0}, {"head_dim": 63},
            {"tie_word_embeddings": "false"}, {"vocab_size": 0},
            {"attention_bias": None}, {"layer_types": "full_attention"},
            {"layer_types": ["full_attention"]}, {"rope_scaling": []},
        ):
            with self.subTest(updates=updates):
                with self.assertRaises(InvalidModelConfigError):
                    self.adapter.normalize(self.config | updates)

    def test_unsupported_variants_have_specific_reasons(self) -> None:
        for updates, reason in (
            ({"architectures": ["LlamaForSequenceClassification"]}, "LlamaForCausalLM"),
            ({"vision_config": {}}, "vision_config"),
            ({"text_config": {}}, "text_config"),
            ({"quantization_config": {"quant_method": "awq"}}, "Quantized"),
            ({"sliding_window": 4096}, "Sliding-window"),
            ({"use_sliding_window": True}, "Sliding-window"),
            ({"num_local_experts": 8}, "Expert"),
            ({"layer_types": ["sliding_attention"] * 22}, "full-attention"),
            ({"attention_type": "latent"}, "attention type"),
            ({"num_key_value_heads": 1}, "Multi-query"),
            ({"is_encoder_decoder": True}, "Encoder-decoder"),
            ({"add_cross_attention": True}, "Cross-attention"),
            ({"auto_map": {"AutoModel": "custom.Model"}}, "Custom"),
            ({"rope_scaling": {"rope_type": "custom"}}, "RoPE"),
        ):
            with self.subTest(updates=updates):
                with self.assertRaisesRegex(UnsupportedArchitectureError, reason):
                    self.adapter.normalize(self.config | updates)

    def test_config_is_not_mutated(self) -> None:
        original = copy.deepcopy(self.config)
        self.adapter.normalize(self.config)
        self.adapter.needs_headers(self.config, None)
        self.adapter.resolve_parameters(self.config, None, None)
        self.assertEqual(self.config, original)


class LlamaParameterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = LlamaAdapter()
        data = fixture("synthetic-tied.json")
        self.config = data["config"]
        self.tensors = tuple(
            TensorMetadata(name, tuple(shape), "BF16", 2 * prod(shape))
            for name, shape in data["tensor_shapes"].items()
        )

    def resolve(self, tensors=None, config=None, summary=None):
        return self.adapter.resolve_parameters(
            self.config if config is None else config, summary,
            TensorInventory(self.tensors if tensors is None else tensors),
        )

    def test_tied_matrix_counts_once_with_one_or_both_aliases(self) -> None:
        for omitted in (None, "model.embed_tokens.weight", "lm_head.weight"):
            with self.subTest(omitted=omitted):
                result = self.resolve(tuple(t for t in self.tensors if t.name != omitted))
                self.assertEqual(result.learned_parameter_count, 744)
                self.assertEqual(result.source, "safetensors_headers")
                self.assertIn("runtime", result.assumptions[0])

    def test_untied_embeddings_count_separately(self) -> None:
        result = self.resolve(config=self.config | {"tie_word_embeddings": False})
        self.assertEqual(result.learned_parameter_count, 824)

    def test_header_shapes_honor_explicit_head_dimension(self) -> None:
        shapes = {
            "model.layers.0.self_attn.q_proj.weight": (16, 8),
            "model.layers.0.self_attn.k_proj.weight": (16, 8),
            "model.layers.0.self_attn.v_proj.weight": (16, 8),
            "model.layers.0.self_attn.o_proj.weight": (8, 16),
        }
        tensors = tuple(
            replace(t, shape=shapes[t.name], stored_bytes=256)
            if t.name in shapes else t for t in self.tensors
        )
        result = self.resolve(tensors, self.config | {"head_dim": 8})
        self.assertEqual(result.learned_parameter_count, 1000)

    def test_biases_are_included_when_declared(self) -> None:
        biases = {
            "self_attn.q_proj": 8, "self_attn.k_proj": 8,
            "self_attn.v_proj": 8, "self_attn.o_proj": 8,
            "mlp.gate_proj": 16, "mlp.up_proj": 16, "mlp.down_proj": 8,
        }
        tensors = self.tensors + tuple(
            TensorMetadata(f"model.layers.0.{name}.bias", (size,), "BF16", size * 2)
            for name, size in biases.items()
        )
        result = self.resolve(tensors, self.config | {"attention_bias": True, "mlp_bias": True})
        self.assertEqual(result.learned_parameter_count, 816)  # 744 + 32 + 40

    def test_known_rotary_buffers_are_disclosed_not_learned(self) -> None:
        buffer = TensorMetadata("model.rotary_emb.inv_freq", (2,), "F32", 8)
        result = self.resolve(self.tensors + (buffer,))
        self.assertEqual(result.learned_parameter_count, 744)
        self.assertEqual(result.checkpoint_buffers, (buffer,))
        self.assertTrue(result.warnings)

    def test_incomplete_unrecognized_or_quantized_headers_return_unknown(self) -> None:
        cases = (
            (), self.tensors[2:], self.tensors[:-1], self.tensors + self.tensors[:1],
            (replace(self.tensors[0], shape=(9, 8)),) + self.tensors[1:],
            (replace(self.tensors[0], dtype="I32"),) + self.tensors[1:],
            self.tensors + (TensorMetadata("unknown.weight", (2,), "BF16", 4),),
        )
        for tensors in cases:
            with self.subTest(names=[t.name for t in tensors]):
                result = self.resolve(tensors)
                self.assertIsNone(result.learned_parameter_count)
                self.assertEqual(result.source, "unknown")
                self.assertTrue(result.warnings)

    def test_no_metadata_has_no_config_formula_fallback(self) -> None:
        result = self.adapter.resolve_parameters(self.config, None, None)
        self.assertIsNone(result.learned_parameter_count)

    def test_unexpected_summary_dtype_requests_headers(self) -> None:
        config = self.config | {"tie_word_embeddings": False}
        summary = ParameterSummary(825, {"BF16": 824, "U8": 1})
        self.assertTrue(self.adapter.needs_headers(config, summary))
        result = self.adapter.resolve_parameters(config, summary, None)
        self.assertIsNone(result.learned_parameter_count)

    def test_summary_discrepancy_is_reported(self) -> None:
        result = self.resolve(summary=ParameterSummary(999, {"BF16": 999}))
        self.assertEqual(result.learned_parameter_count, 744)
        self.assertTrue(any("differs" in warning for warning in result.warnings))


if __name__ == "__main__":
    unittest.main()
