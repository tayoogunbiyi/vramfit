import copy
import json
import unittest
from math import prod
from pathlib import Path

from vramfit.adapters.qwen2 import Qwen2Adapter
from vramfit.adapters.registry import default_registry
from vramfit.errors import InvalidModelConfigError, UnsupportedArchitectureError
from vramfit.models import ParameterSummary, TensorInventory, TensorMetadata

FIXTURES = Path(__file__).parent / "fixtures"


def fixture(name):
    return json.loads((FIXTURES / "qwen2" / name).read_text())


class Qwen2Tests(unittest.TestCase):
    def setUp(self):
        self.adapter = Qwen2Adapter()
        self.config = fixture("qwen2.5-7b.json")

    def test_pinned_size_ladder_and_variants(self):
        entries = fixture("provenance.json")["models"]
        self.assertEqual(len(entries), 15)
        for entry in entries:
            if entry["file"].endswith(("-awq.json", "-1m.json")):
                continue
            with self.subTest(repo=entry["repo"]):
                self.assertRegex(entry["revision"], r"^[a-f0-9]{40}$")
                config = fixture(entry["file"])
                spec = default_registry().get("qwen2").normalize(config)
                self.assertEqual(spec.head_dim, config["hidden_size"] // config["num_attention_heads"])
                self.assertEqual(spec.num_key_value_heads, config["num_key_value_heads"])
                self.assertEqual(spec.max_context_length, config["max_position_embeddings"])
                summary = ParameterSummary(100, {"BF16": 100})
                self.assertEqual(self.adapter.needs_headers(config, summary), config["tie_word_embeddings"])

    def test_window_is_controlled_by_enabling_flag(self):
        self.adapter.normalize(self.config | {"sliding_window": 4096})
        no_flag = dict(self.config)
        no_flag.pop("use_sliding_window")
        self.adapter.normalize(no_flag)
        with self.assertRaisesRegex(UnsupportedArchitectureError, "Sliding-window"):
            self.adapter.normalize(self.config | {"use_sliding_window": True})
        with self.assertRaises(InvalidModelConfigError):
            self.adapter.normalize(self.config | {"use_sliding_window": "false"})

    def test_kv_defaults_are_family_specific(self):
        config = self.config | {"num_attention_heads": 32, "head_dim": 128}
        config.pop("num_key_value_heads")
        self.assertEqual(self.adapter.normalize(config).num_key_value_heads, 32)
        config.update(num_attention_heads=40, num_key_value_heads=None)
        self.assertEqual(self.adapter.normalize(config).num_key_value_heads, 40)

    def test_quantized_and_dual_chunk_checkpoints_rejected(self):
        for name, reason in (("qwen2.5-7b-instruct-awq.json", "Quantized"),
                             ("qwen2.5-7b-instruct-1m.json", "Dual-chunk")):
            with self.subTest(name=name), self.assertRaisesRegex(UnsupportedArchitectureError, reason):
                self.adapter.normalize(fixture(name))

    def test_nonstandard_bias_policy_rejected(self):
        for updates in ({"attention_bias": False}, {"mlp_bias": True}):
            with self.subTest(updates=updates), self.assertRaisesRegex(UnsupportedArchitectureError, "bias"):
                self.adapter.normalize(self.config | updates)

    def test_synthetic_headers_count_qkv_biases_and_ties(self):
        # Independent hand-specified Llama layout: 744 tied weights; add 24 QKV biases.
        data = json.loads((FIXTURES / "llama" / "synthetic-tied.json").read_text())
        config = data["config"] | {"model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"]}
        shapes = data["tensor_shapes"] | {
            f"model.layers.0.self_attn.{p}_proj.bias": [8] for p in ("q", "k", "v")
        }
        tensors = tuple(TensorMetadata(n, tuple(s), "BF16", prod(s) * 2) for n, s in shapes.items())
        for omitted in (None, "lm_head.weight", "model.embed_tokens.weight"):
            headers = TensorInventory(tuple(t for t in tensors if t.name != omitted))
            result = self.adapter.resolve_parameters(config, None, headers)
            self.assertEqual(result.learned_parameter_count, 768)
        untied = self.adapter.resolve_parameters(config | {"tie_word_embeddings": False}, None, TensorInventory(tensors))
        self.assertEqual(untied.learned_parameter_count, 848)
        missing_bias = TensorInventory(tensors[:-1])
        self.assertIsNone(self.adapter.resolve_parameters(config, None, missing_bias).learned_parameter_count)
        unexpected = TensorMetadata("model.layers.0.self_attn.o_proj.bias", (8,), "BF16", 16)
        self.assertIsNone(self.adapter.resolve_parameters(config, None, TensorInventory(tensors + (unexpected,))).learned_parameter_count)

    def test_summary_and_unknown_paths_do_not_mutate_config(self):
        original = copy.deepcopy(self.config)
        summary = ParameterSummary(7_615_616_512, {"BF16": 7_615_616_512})
        result = self.adapter.resolve_parameters(self.config, summary, None)
        self.assertEqual(result.source, "hub_safetensors")
        self.assertEqual(result.learned_parameter_count, summary.total)
        self.assertIsNone(self.adapter.resolve_parameters(self.config, None, None).learned_parameter_count)
        self.assertEqual(self.config, original)
