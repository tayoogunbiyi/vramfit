import copy
import json
import unittest
from dataclasses import replace
from math import prod
from pathlib import Path

from vramfit.adapters.mistral import MistralAdapter
from vramfit.adapters.registry import default_registry
from vramfit.errors import InvalidModelConfigError, UnsupportedArchitectureError
from vramfit.models import ParameterSummary, TensorInventory, TensorMetadata

FIXTURES = Path(__file__).parent / "fixtures"


def fixture(name):
    return json.loads((FIXTURES / "mistral" / name).read_text())


class MistralTests(unittest.TestCase):
    def setUp(self):
        self.adapter = MistralAdapter()
        self.config = fixture("mistral-nemo-instruct-2407.json")

    def test_five_pinned_full_attention_checkpoints(self):
        entries = fixture("provenance.json")["models"][:5]
        self.assertEqual(len(entries), 5)
        for entry in entries:
            with self.subTest(repo=entry["repo"]):
                self.assertRegex(entry["revision"], r"^[a-f0-9]{40}$")
                config = fixture(entry["file"])
                spec = default_registry().get("mistral").normalize(config)
                self.assertIsNone(config["sliding_window"])
                self.assertEqual(spec.head_dim, 128)
                self.assertEqual(spec.num_key_value_heads, 8)
                self.assertEqual(spec.max_context_length, config["max_position_embeddings"])
                if spec.hidden_size == 5120:
                    self.assertNotEqual(spec.head_dim, spec.hidden_size // spec.num_attention_heads)
                    self.assertEqual(spec.provenance["head_dim"], "head_dim")
                self.assertFalse(self.adapter.needs_headers(config, ParameterSummary(100, {"BF16": 100})))

    def test_absent_window_is_not_explicit_null(self):
        config = dict(self.config)
        config.pop("sliding_window")
        with self.assertRaisesRegex(UnsupportedArchitectureError, "Sliding-window"):
            self.adapter.normalize(config)
        self.adapter.normalize(config | {"sliding_window": None})
        with self.assertRaisesRegex(UnsupportedArchitectureError, "Sliding-window"):
            self.adapter.normalize(config | {"sliding_window": 4096, "use_sliding_window": False})

    def test_kv_heads_default_to_eight_but_explicit_null_to_mha(self):
        config = dict(self.config)
        config.pop("num_key_value_heads")
        self.assertEqual(self.adapter.normalize(config).num_key_value_heads, 8)
        self.assertEqual(self.adapter.normalize(config | {"num_key_value_heads": None}).num_key_value_heads, 32)

    def test_sliding_mixed_and_composite_rejections(self):
        for filename, reason in (("mistral-7b-v0.1.json", "Sliding-window"),
                                 ("ministral-8b-instruct-2410.json", "full-attention")):
            with self.subTest(filename=filename), self.assertRaisesRegex(UnsupportedArchitectureError, reason):
                self.adapter.normalize(fixture(filename))
        composite = fixture("mistral-small-3.2-24b-instruct-2506.json")
        with self.assertRaisesRegex(UnsupportedArchitectureError, "No adapter"):
            default_registry().get(composite["model_type"])
        with self.assertRaises(InvalidModelConfigError):
            self.adapter.normalize(composite)
        with self.assertRaisesRegex(UnsupportedArchitectureError, "text_config"):
            self.adapter.normalize(composite | {"model_type": "mistral"})

    def test_projection_biases_rejected(self):
        for key in ("attention_bias", "mlp_bias"):
            with self.subTest(key=key), self.assertRaisesRegex(UnsupportedArchitectureError, "bias-free"):
                self.adapter.normalize(self.config | {key: True})

    def test_headers_use_explicit_dimension_and_no_biases(self):
        data = json.loads((FIXTURES / "llama" / "synthetic-tied.json").read_text())
        config = data["config"] | {"model_type": "mistral", "architectures": ["MistralForCausalLM"], "sliding_window": None, "head_dim": 8}
        shapes = data["tensor_shapes"] | {
            "model.layers.0.self_attn.q_proj.weight": [16, 8],
            "model.layers.0.self_attn.k_proj.weight": [16, 8],
            "model.layers.0.self_attn.v_proj.weight": [16, 8],
            "model.layers.0.self_attn.o_proj.weight": [8, 16],
        }
        tensors = tuple(TensorMetadata(n, tuple(s), "BF16", prod(s) * 2) for n, s in shapes.items())
        result = self.adapter.resolve_parameters(config, None, TensorInventory(tensors))
        self.assertEqual(result.learned_parameter_count, 1000)
        wrong = (replace(tensors[0], shape=(1,)),) + tensors[1:]
        self.assertIsNone(self.adapter.resolve_parameters(config, None, TensorInventory(wrong)).learned_parameter_count)

    def test_missing_evidence_and_input_immutability(self):
        original = copy.deepcopy(self.config)
        self.assertIsNone(self.adapter.resolve_parameters(self.config, None, None).learned_parameter_count)
        self.assertEqual(self.config, original)
