import json
import unittest
from dataclasses import replace
from math import prod
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from vramfit.cli import main
from vramfit.hub import ModelConfigDownloadError, ModelHubError, ModelSnapshot
from vramfit.models import ParameterSummary, TensorInventory, TensorMetadata

FIXTURES = Path(__file__).parent / "fixtures"
SHA = "a" * 40
MODEL = "publisher/model"
ARGS = [MODEL, "--vram", "24", "--prompt-length", "1024", "--max-output-length", "512"]


def config_fixture(family, name):
    return json.loads((FIXTURES / family / name).read_text())


class CLITests(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()
        self.snapshot = ModelSnapshot(MODEL, SHA, Path("/mock-cache/config.json"),
                                      config_fixture("llama", "tinyllama.json"),
                                      ParameterSummary(1_100_048_384, {"BF16": 1_100_048_384}))
        self.load = self.enterContext(patch("vramfit.inspection.load_model_snapshot", return_value=self.snapshot))
        self.headers = self.enterContext(patch("vramfit.inspection.load_tensor_inventory", return_value=None))

    def invoke(self, options=()):
        return self.runner.invoke(main, ARGS + ["--detailed"] + list(options))

    def test_all_four_adapters_reach_memory_output_without_unneeded_headers(self):
        for family, file in (("llama", "tinyllama.json"), ("qwen2", "qwen2.5-7b.json"),
                             ("qwen3", "qwen3-8b.json"), ("mistral", "mistral-nemo-instruct-2407.json")):
            with self.subTest(family=family):
                self.load.return_value = replace(self.snapshot, config=config_fixture(family, file))
                result = self.invoke()
                self.assertEqual(result.exit_code, 0, result.output)
                self.assertIn(f"Adapter: {family}", result.output)
                self.assertIn("Parameter evidence: hub_safetensors", result.output)
                self.assertIn("Known workload fits: yes, within the calculated budget", result.output)
                self.assertIn("not a runtime guarantee", result.output)
                self.assertIn("individual tensors have not been inspected", result.output)
        self.headers.assert_not_called()

    def test_revision_identity_and_geometry_provenance(self):
        self.load.return_value = replace(self.snapshot, config=config_fixture("qwen3", "qwen3-8b.json"))
        result = self.invoke(["--revision", "release"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.load.assert_called_once_with(MODEL, revision="release")
        self.assertIn(f"Hugging Face model: {MODEL}", result.output)
        self.assertIn(f"Resolved revision: {SHA}", result.output)
        self.assertIn("head_dim: 128 [from head_dim]", result.output)
        self.assertIn("Config: /mock-cache/config.json", result.output)

    def test_default_revision_dtype_and_headroom(self):
        result = self.invoke()
        self.load.assert_called_once_with(MODEL, revision="main")
        self.assertIn("Dtype (weights and KV): float16", result.output)
        self.assertIn("Reserved headroom (10%)", result.output)
        self.assertIn("Physical VRAM: 24.0000 GiB", result.output)
        self.assertIn("Usable VRAM: 21.6000 GiB", result.output)
        self.assertIn("Tokens per request: 1,536 (1,024 prompt + 512 output)", result.output)
        self.assertIn("Target concurrency: 1", result.output)

    def test_tied_headers_are_targeted_and_aliases_count_once(self):
        data = config_fixture("llama", "synthetic-tied.json")
        for family, architecture, extras, count in (
            ("llama", "LlamaForCausalLM", {}, 744),
            ("qwen2", "Qwen2ForCausalLM", {f"model.layers.0.self_attn.{p}_proj.bias": [8] for p in ("q", "k", "v")}, 768),
            ("qwen3", "Qwen3ForCausalLM", {f"model.layers.0.self_attn.{p}_norm.weight": [4] for p in ("q", "k")}, 752),
        ):
            with self.subTest(family=family):
                config = data["config"] | {"model_type": family, "architectures": [architecture],
                                           "head_dim": 4, "max_position_embeddings": 2048}
                snapshot = replace(self.snapshot, config=config)
                self.load.return_value = snapshot
                self.headers.reset_mock()
                self.headers.return_value = TensorInventory(tuple(
                    TensorMetadata(n, tuple(s), "BF16", prod(s) * 2)
                    for n, s in (data["tensor_shapes"] | extras).items()
                ))
                result = self.invoke()
                self.assertEqual(result.exit_code, 0, result.output)
                self.headers.assert_called_once_with(snapshot)
                self.assertIn(f"Learned parameters: {count}", result.output)
                self.assertIn("Parameter evidence: safetensors_headers", result.output)
                self.assertIn("runtime honors tie_word_embeddings", result.output)
                self.assertIn("Hub summary differs", result.output)

    def test_unknown_parameters_preserve_known_kv_but_not_fit(self):
        self.load.return_value = replace(self.snapshot, parameter_summary=None)
        result = self.invoke()
        self.assertEqual(result.exit_code, 0, result.output)
        self.headers.assert_called_once_with(self.load.return_value)
        self.assertIn("Learned parameters: unknown", result.output)
        self.assertIn("KV per token: 22,528 bytes", result.output)
        self.assertIn("Known workload fits: unknown", result.output)
        self.assertIn("Theoretical maximum concurrency: unknown", result.output)
        self.assertIn("Tensor headers are required but unavailable", result.output)
        self.assertNotIn("fits: yes", result.output)

    def test_weights_exceeding_budget_are_a_valid_negative_estimate(self):
        result = self.invoke(["--vram", "1"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Weights fit: no, exceeds", result.output)
        self.assertIn("Known workload fits: no, exceeds", result.output)
        self.assertIn("Theoretical maximum concurrency: 0", result.output)

    def test_kv_can_make_workload_fail_even_if_weights_fit(self):
        result = self.invoke(["--target-concurrency", "10000"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Weights fit: yes", result.output)
        self.assertIn("Known workload fits: no", result.output)

    def test_invalid_context_is_actionable(self):
        result = self.invoke(["--max-output-length", "1025"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("exceeds the configured context limit (2048 tokens)", result.output)
        self.assertNotIn("Known workload fits:", result.output)

    def test_unsupported_architecture_or_features_fail_before_headers(self):
        for config, reason in (
            ({"model_type": "qwen3_moe"}, "No adapter"),
            (config_fixture("qwen2", "qwen2.5-7b-instruct-awq.json"), "Quantized"),
            (config_fixture("mistral", "mistral-7b-v0.1.json"), "Sliding-window"),
            (config_fixture("mistral", "mistral-small-3.2-24b-instruct-2506.json"), "mistral3"),
        ):
            with self.subTest(reason=reason):
                self.load.return_value = replace(self.snapshot, config=config)
                result = self.invoke()
                self.assertNotEqual(result.exit_code, 0)
                self.assertIn(reason, result.output)
                self.assertNotIn("Traceback", result.output)
                self.assertNotIn("Known workload fits:", result.output)
        self.headers.assert_not_called()

    def test_malformed_model_type_is_not_a_registry_crash(self):
        for value in (None, [], {}, 12, " "):
            with self.subTest(value=value):
                self.load.return_value = replace(self.snapshot, config={"model_type": value})
                result = self.invoke()
                self.assertNotEqual(result.exit_code, 0)
                self.assertIn("non-empty string model_type", result.output)
        self.headers.assert_not_called()

    def test_access_errors_do_not_produce_estimates_or_substitute_repos(self):
        self.load.side_effect = ModelConfigDownloadError("Manual approval required for config.json.", status_code=403, gated="manual")
        result = self.invoke()
        self.assertEqual(result.exit_code, 1)
        self.assertIn("Manual approval required", result.output)
        self.assertNotIn("Known workload fits:", result.output)
        self.load.assert_called_once_with(MODEL, revision="main")
        self.headers.assert_not_called()

    def test_header_transport_failure_is_not_an_unknown_success(self):
        self.load.return_value = replace(self.snapshot, parameter_summary=None)
        self.headers.side_effect = ModelHubError("Could not inspect tensor headers.")
        result = self.invoke()
        self.assertEqual(result.exit_code, 1)
        self.assertIn("Could not inspect tensor headers", result.output)
        self.assertNotIn("Known workload fits:", result.output)

    def test_invalid_workload_and_nonfinite_capacity_fail_before_network(self):
        for options in (["--vram", "nan"], ["--vram", "inf"], ["--headroom", "nan"],
                        ["--headroom", "100"], ["--target-concurrency", "0"],
                        ["--prompt-length", "0"], ["--dtype", "int8"]):
            with self.subTest(options=options):
                result = self.invoke(options)
                self.assertNotEqual(result.exit_code, 0)
                self.assertNotIn("Traceback", result.output)
        self.load.assert_not_called()

    def test_case_insensitive_dtype_is_normalized(self):
        result = self.invoke(["--dtype", "BFLOAT16"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Dtype (weights and KV): bfloat16", result.output)

    def test_help_and_version_do_not_access_hub(self):
        for option in ("--help", "--version"):
            result = self.runner.invoke(main, [option])
            self.assertEqual(result.exit_code, 0)
        self.load.assert_not_called()

    def test_buffers_are_disclosed_separately_from_weight_memory(self):
        data = config_fixture("llama", "synthetic-tied.json")
        config = data["config"] | {"max_position_embeddings": 2048}
        self.load.return_value = replace(self.snapshot, config=config)
        tensors = tuple(TensorMetadata(n, tuple(s), "BF16", prod(s) * 2) for n, s in data["tensor_shapes"].items())
        buffer = TensorMetadata("model.rotary_emb.inv_freq", (2,), "F32", 8)
        self.headers.return_value = TensorInventory(tensors + (buffer,))
        result = self.invoke()
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Learned weight memory: 0.0000 GiB (1,488 bytes)", result.output)
        self.assertIn("Excluded checkpoint buffers: 1 tensors, 8 stored bytes", result.output)
        self.assertIn("model.rotary_emb.inv_freq: shape=(2,), dtype=F32", result.output)


class SummaryCLITests(unittest.TestCase):
    setUp = CLITests.setUp

    def summary(self, options=(), width=80):
        return self.runner.invoke(main, ARGS + list(options),
                                  env={"COLUMNS": str(width), "NO_COLOR": "1"})

    def test_summary_table_and_hidden_details(self):
        result = self.summary()
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("FITS ESTIMATED BUDGET", result.output)
        self.assertEqual(result.output.splitlines()[0], f"{MODEL} · FITS ESTIMATED BUDGET")
        self.assertIn("Estimated memory", result.output)
        self.assertIn("10% reserved", result.output)
        self.assertIn("not a runtime guarantee", result.output)
        self.assertIn("--detailed", result.output)
        for hidden in ("Resolved revision:", "Config:", "Geometry", "Theoretical maximum", " bytes)"):
            self.assertNotIn(hidden, result.output)
        self.assertNotIn("\x1b[", result.output)

    def test_table_width_and_borders_with_long_unicode_values(self):
        from rich.cells import cell_len
        self.load.return_value = replace(self.snapshot, model_id="模型/" + "long-name-" * 15)
        for width in (60, 80, 120):
            with self.subTest(width=width):
                result = self.summary(["--target-concurrency", "123456789"], width)
                self.assertEqual(result.exit_code, 0, result.output)
                lines = result.output.splitlines()
                self.assertTrue(all(cell_len(line) <= width for line in lines))
                borders = [line for line in lines if line.startswith(("┌", "│", "└"))]
                self.assertGreater(len(borders), 6)
                self.assertEqual(len({cell_len(line) for line in borders}), 1)
                self.assertTrue(all(line.endswith("│") for line in borders[1:-1]))

    def test_narrow_terminal_uses_stacked_rows(self):
        for width in (30, 59):
            result = self.summary(width=width)
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertIn("Estimated memory:", result.output)
            self.assertNotIn("┌", result.output)
            self.assertTrue(all(len(line) <= width for line in result.output.splitlines()))

    def test_summary_deficits_and_unknown(self):
        result = self.summary(["--vram", "1"])
        self.assertIn("EXCEEDS ESTIMATED BUDGET", result.output)
        self.assertIn("Budget deficit", result.output)
        self.assertIn("Weights alone exceed", result.output)
        result = self.summary(["--target-concurrency", "10000"])
        self.assertIn("Weights fit, but KV cache", result.output)
        self.load.return_value = replace(self.snapshot, parameter_summary=None)
        result = self.summary()
        self.assertIn("FIT UNKNOWN", result.output)
        self.assertIn("Tensor headers are required but unavailable", result.output)
        self.assertNotIn("FITS ESTIMATED", result.output)

    def test_tiny_deficit_is_not_rounded_to_zero(self):
        from vramfit.memory import GIB
        # TinyLlama fixture: float16 weights plus KV for 1,536 tokens.
        known_bytes = 1_100_048_384 * 2 + 22_528 * 1536
        result = self.summary(["--headroom", "0", "--vram", str((known_bytes - 1) / GIB)])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("EXCEEDS ESTIMATED BUDGET", result.output)
        self.assertRegex(result.output, r"Budget deficit\s*│ <0.01 GiB")
