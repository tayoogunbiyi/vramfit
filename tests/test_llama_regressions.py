"""Pinned real-family regressions, independent of repository-name heuristics."""

import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from vramfit.adapters.registry import default_registry
from vramfit.cli import main
from vramfit.hub import ModelSnapshot
from vramfit.memory import GPUCapacity, Workload, calculate_memory
from vramfit.models import ParameterSummary, TensorInventory, TensorMetadata

FIXTURES = Path(__file__).parent / "fixtures" / "llama"


def fixture(name):
    return json.loads((FIXTURES / name).read_text())


def llama2_headers():
    data = fixture("llama2-mirror-headers.json")
    rows = data["root_tensors"] + [
        row | {"name": f"model.layers.{layer}.{row['name']}"}
        for layer in range(data["repeat_layers"]) for row in data["layer_tensors"]
    ]
    return TensorInventory(tuple(
        TensorMetadata(row["name"], tuple(row["shape"]), row["dtype"], row["stored_bytes"])
        for row in rows
    ))


class LlamaRegressionTests(unittest.TestCase):
    def test_distinct_configs_use_existing_adapter_and_expected_geometry(self):
        expected = {
            # layers, hidden width, Q heads, KV heads, head dim, context, KV bytes/token
            "llama2-mirror.json": (32, 4096, 32, 32, 128, 4096, 524288),
            "llama31-mirror.json": (32, 4096, 32, 8, 128, 131072, 131072),
            "distill-llama8b.json": (32, 4096, 32, 8, 128, 131072, 131072),
            "falcon3-7b.json": (28, 3072, 12, 4, 256, 32768, 114688),
        }
        adapter = default_registry().get("llama")
        for entry in fixture("extended-provenance.json")["models"]:
            with self.subTest(model=entry["model_id"]):
                self.assertRegex(entry["revision"], r"^[a-f0-9]{40}$")
                config = fixture(entry["file"])
                original = copy.deepcopy(config)
                spec = adapter.normalize(config)
                geometry = (spec.num_layers, spec.hidden_size, spec.num_attention_heads,
                            spec.num_key_value_heads, spec.head_dim, spec.max_context_length)
                self.assertEqual(geometry, expected[entry["file"]][:6])
                summary = ParameterSummary(**entry["hub_summary"])
                headers = llama2_headers() if entry["file"] == "llama2-mirror.json" else None
                parameters = adapter.resolve_parameters(config, summary, headers)
                memory = calculate_memory(spec, parameters, Workload(1024, 512), GPUCapacity(24))
                self.assertEqual(memory.kv_bytes_per_token, expected[entry["file"]][6])
                self.assertTrue(memory.workload_fits)
                self.assertEqual(config, original)
                if config.get("rope_scaling"):
                    self.assertTrue(any("does not multiply" in a for a in spec.assumptions))
                if entry["file"] == "falcon3-7b.json":
                    self.assertEqual(spec.provenance["head_dim"], "head_dim")

    def test_llama2_mixed_float_summary_requires_header_reconciliation(self):
        entry = fixture("extended-provenance.json")["models"][0]
        config = fixture(entry["file"])
        summary = ParameterSummary(**entry["hub_summary"])
        adapter = default_registry().get("llama")
        self.assertTrue(adapter.needs_headers(config, summary))
        self.assertIsNone(adapter.resolve_parameters(config, summary, None).learned_parameter_count)
        inventory = llama2_headers()
        self.assertEqual(len(inventory.tensors), 323)
        parameters = adapter.resolve_parameters(config, summary, inventory)
        self.assertEqual(parameters.learned_parameter_count, 6_738_415_616)
        self.assertEqual(parameters.source, "safetensors_headers")
        self.assertEqual(len(parameters.checkpoint_buffers), 32)
        self.assertEqual(sum(t.stored_bytes for t in parameters.checkpoint_buffers), 8192)
        self.assertEqual(summary.total - parameters.learned_parameter_count, 2048)

    def test_pinned_repositories_reach_cli_without_identity_substitution(self):
        expected = {
            "llama2-mirror.json": 6_738_415_616,
            "llama31-mirror.json": 8_030_261_248,
            "distill-llama8b.json": 8_030_261_248,
            "falcon3-7b.json": 7_455_550_464,
        }
        for entry in fixture("extended-provenance.json")["models"]:
            with self.subTest(model=entry["model_id"]):
                snapshot = ModelSnapshot(entry["model_id"], entry["revision"],
                                         FIXTURES / entry["file"], fixture(entry["file"]),
                                         ParameterSummary(**entry["hub_summary"]))
                with patch("vramfit.inspection.load_model_snapshot", return_value=snapshot) as load, \
                     patch("vramfit.inspection.load_tensor_inventory", return_value=llama2_headers()) as headers:
                    result = CliRunner().invoke(main, [entry["model_id"], "--revision", entry["revision"],
                        "--detailed", "--vram", "24", "--prompt-length", "1024", "--max-output-length", "512"])
                self.assertEqual(result.exit_code, 0, result.output)
                load.assert_called_once_with(entry["model_id"], revision=entry["revision"])
                self.assertIn(f"Hugging Face model: {entry['model_id']}", result.output)
                self.assertIn(f"Resolved revision: {entry['revision']}", result.output)
                self.assertIn(f"Learned parameters: {expected[entry['file']]:,}", result.output)
                self.assertIn("Adapter: llama", result.output)
                if entry["file"] == "llama2-mirror.json":
                    headers.assert_called_once_with(snapshot)
                    self.assertIn("Excluded checkpoint buffers: 32 tensors", result.output)
                else:
                    headers.assert_not_called()
                    self.assertIn("Parameter evidence: hub_safetensors", result.output)
