import json
import unittest
from dataclasses import replace
from math import prod
from pathlib import Path

from vramfit.adapters.qwen3 import Qwen3Adapter
from vramfit.adapters.registry import default_registry
from vramfit.errors import InvalidModelConfigError, UnsupportedArchitectureError
from vramfit.models import ParameterSummary, TensorInventory, TensorMetadata

FIXTURES = Path(__file__).parent / "fixtures"


def fixture(name):
    return json.loads((FIXTURES / "qwen3" / name).read_text())


class Qwen3Tests(unittest.TestCase):
    def setUp(self):
        self.adapter = Qwen3Adapter()

    def test_all_six_pinned_dense_sizes(self):
        entries = fixture("provenance.json")["models"]
        self.assertEqual(len(entries), 6)
        mismatches = []
        for entry in entries:
            with self.subTest(repo=entry["repo"]):
                self.assertRegex(entry["revision"], r"^[a-f0-9]{40}$")
                config = fixture(entry["file"])
                spec = default_registry().get("qwen3").normalize(config)
                self.assertEqual(spec.head_dim, 128)
                self.assertEqual(spec.provenance["head_dim"], "head_dim")
                self.assertEqual(spec.num_key_value_heads, config["num_key_value_heads"])
                if spec.hidden_size // spec.num_attention_heads != 128:
                    mismatches.append(entry["repo"].split("-")[-1])
                tied = entry["repo"].split("-")[-1] in ("0.6B", "1.7B", "4B")
                self.assertEqual(config["tie_word_embeddings"], tied)
                self.assertEqual(self.adapter.needs_headers(config, ParameterSummary(100, {"BF16": 100})), tied)
        self.assertEqual(mismatches, ["0.6B", "4B", "32B"])

    def test_missing_head_dim_uses_qwen3_default_not_hidden_division(self):
        config = fixture("qwen3-4b.json")
        config.pop("head_dim")
        spec = self.adapter.normalize(config)
        self.assertEqual(spec.head_dim, 128)
        self.assertIn("default", spec.provenance["head_dim"])
        with self.assertRaises(InvalidModelConfigError):
            self.adapter.normalize(config | {"head_dim": None})

    def test_other_qwen3_families_remain_unregistered(self):
        for model_type in ("qwen3_moe", "qwen3_next", "qwen3_vl"):
            with self.subTest(model_type=model_type), self.assertRaisesRegex(UnsupportedArchitectureError, "No adapter"):
                default_registry().get(model_type)

    def test_feature_boundary(self):
        config = fixture("qwen3-4b.json")
        self.adapter.normalize(config | {"sliding_window": 4096, "use_sliding_window": False})
        for updates, reason in (
            ({"use_sliding_window": True}, "Sliding-window"),
            ({"mlp_bias": True}, "MLP biases"),
            ({"num_experts": 8}, "Expert"),
            ({"quantization_config": {}}, "Quantized"),
            ({"layer_types": ["sliding_attention"] * 36}, "full-attention"),
        ):
            with self.subTest(updates=updates), self.assertRaisesRegex(UnsupportedArchitectureError, reason):
                self.adapter.normalize(config | updates)

    def test_norm_weights_are_per_head_dim_and_required(self):
        data = json.loads((FIXTURES / "llama" / "synthetic-tied.json").read_text())
        config = data["config"] | {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"], "head_dim": 4}
        shapes = data["tensor_shapes"] | {
            "model.layers.0.self_attn.q_norm.weight": [4],
            "model.layers.0.self_attn.k_norm.weight": [4],
        }
        tensors = tuple(TensorMetadata(n, tuple(s), "BF16", prod(s) * 2) for n, s in shapes.items())
        def resolve(items, overrides=None):
            return self.adapter.resolve_parameters(config | (overrides or {}), None, TensorInventory(items))
        for omitted in (None, "lm_head.weight", "model.embed_tokens.weight"):
            self.assertEqual(resolve(tuple(t for t in tensors if t.name != omitted)).learned_parameter_count, 752)
        self.assertEqual(resolve(tensors, {"tie_word_embeddings": False}).learned_parameter_count, 832)
        self.assertIsNone(resolve(tensors[:-1]).learned_parameter_count)
        wrong = tensors[:-1] + (replace(tensors[-1], shape=(8,)),)
        self.assertIsNone(resolve(wrong).learned_parameter_count)
        biases = tuple(TensorMetadata(f"model.layers.0.self_attn.{p}_proj.bias", (8,), "BF16", 16) for p in ("q", "k", "v", "o"))
        self.assertEqual(resolve(tensors + biases, {"attention_bias": True}).learned_parameter_count, 784)

    def test_no_evidence_stays_unknown(self):
        self.assertIsNone(self.adapter.resolve_parameters(fixture("qwen3-4b.json"), None, None).learned_parameter_count)
