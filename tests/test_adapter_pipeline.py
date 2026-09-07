"""A test-only family exercises the real pipeline with only Hub I/O replaced.

The toy checkpoint has one 3x4 weight and one 3-element bias: 15 learned
parameters. Its independent cache geometry gives 64 FP16 KV bytes/token.
It deliberately uses neither Llama config fields nor Llama tensor names.
"""

import json
import unittest
from pathlib import Path
from unittest.mock import call, patch

from click.testing import CliRunner
from httpx import ConnectError
from huggingface_hub import ModelInfo
from huggingface_hub.errors import NotASafetensorsRepoError
from huggingface_hub.utils import SafetensorsFileMetadata, SafetensorsRepoMetadata, TensorInfo

from vramfit.adapters.registry import AdapterRegistry, default_registry
from vramfit.cli import main
from vramfit.errors import InvalidModelConfigError, UnsupportedArchitectureError
from vramfit.hub import REQUEST_TIMEOUT
from vramfit.inspection import inspect_model
from vramfit.memory import GPUCapacity, Workload, calculate_memory
from vramfit.models import DecoderSpec, ParameterEstimate, ParameterSummary, TensorInventory

MODEL = "test-publisher/unrelated-name"
SHA = "b" * 40
CONFIG_PATH = Path("/mock-cache/test-decoder/config.json")


class ToyAdapter:
    """Independent protocol implementation, not a built-in adapter subclass."""

    model_type = "test_decoder"

    def normalize(self, config):
        if config.get("model_type") != self.model_type:
            raise InvalidModelConfigError("Expected the test decoder family.")
        if config.get("vision"):
            raise UnsupportedArchitectureError(self.model_type, ("Toy vision mode is unsupported.",))
        fields = dict(num_layers="blocks", hidden_size="width", num_attention_heads="q_heads",
                      num_key_value_heads="kv_heads", head_dim="channel", max_context_length="limit")
        try:
            return DecoderSpec(self.model_type, **{key: config[source] for key, source in fields.items()},
                               provenance=fields, assumptions=("Toy full-attention decoder.",))
        except (KeyError, ValueError) as error:
            raise InvalidModelConfigError("Invalid toy geometry.") from error

    def needs_headers(self, config, summary):
        return summary is None or summary.total <= 0 or set(summary.parameters_by_dtype) != {"BF16"}

    def resolve_parameters(self, config, summary, headers):
        if headers is None:
            if summary is not None and not self.needs_headers(config, summary):
                return ParameterEstimate(summary.total, "hub_safetensors",
                                         assumptions=("Toy summary represents learned weights.",))
            return ParameterEstimate(None, "unknown", warnings=("Toy tensor evidence is unavailable.",))
        tensors = {tensor.name: tensor for tensor in headers.tensors}
        expected = {"projection.weight": ((3, 4), "BF16"), "projection.bias": ((3,), "BF16")}
        optional = {"cache_hint": ((2,), "U8")}
        if (len(tensors) != len(headers.tensors) or not expected.keys() <= tensors.keys()
                or any((t.shape, t.dtype) != (expected | optional).get(n) for n, t in tensors.items())):
            return ParameterEstimate(None, "unknown", warnings=("Toy checkpoint layout is unrecognized.",))
        # Hand-specified toy layout, independent of the application's shape/count helpers.
        buffers = (tensors["cache_hint"],) if "cache_hint" in tensors else ()
        return ParameterEstimate(15, "safetensors_headers", checkpoint_buffers=buffers,
                                 warnings=("Toy cache hint is not a learned weight.",) if buffers else ())


class AdapterPipelineTests(unittest.TestCase):
    def setUp(self):
        self.config = dict(model_type="test_decoder", blocks=2, width=8, q_heads=4,
                           kv_heads=2, channel=4, limit=64)
        self.adapter = ToyAdapter()
        self.registry = AdapterRegistry([self.adapter])
        # Registration is the only application wiring replaced; the Hub boundary
        # returns fake responses, while inspection/calculation/rendering stay real.
        self.enterContext(patch("vramfit.inspection.default_registry", return_value=self.registry))
        self.api = self.enterContext(patch("vramfit.hub.HfApi", autospec=True)).return_value
        self.resolved = ModelInfo(id=MODEL, sha=SHA)
        self.info = ModelInfo(id=MODEL, safetensors={"total": 15, "parameters": {"BF16": 15}})
        self.api.model_info.side_effect = [self.resolved, self.info]
        self.download = self.enterContext(patch("vramfit.hub.hf_hub_download", return_value=str(CONFIG_PATH)))
        self.enterContext(patch.object(Path, "read_text", side_effect=lambda **kwargs: json.dumps(self.config)))
        self.file = SafetensorsFileMetadata({}, {
            "projection.weight": TensorInfo("BF16", [3, 4], (0, 24)),
            "projection.bias": TensorInfo("BF16", [3], (24, 30)),
            "cache_hint": TensorInfo("U8", [2], (30, 32)),
        })
        self.api.get_safetensors_metadata.return_value = SafetensorsRepoMetadata(
            None, False, {name: "model.safetensors" for name in self.file.tensors},
            {"model.safetensors": self.file},
        )
        self.normalize = self.enterContext(patch.object(self.adapter, "normalize", wraps=self.adapter.normalize))
        self.needs = self.enterContext(patch.object(self.adapter, "needs_headers", wraps=self.adapter.needs_headers))
        self.resolve = self.enterContext(patch.object(self.adapter, "resolve_parameters", wraps=self.adapter.resolve_parameters))

    def invoke(self):
        return CliRunner().invoke(main, [MODEL, "--revision", "release", "--vram", "1", "--headroom", "0",
                                       "--prompt-length", "10", "--max-output-length", "5",
                                       "--target-concurrency", "3"])

    def require_headers(self):
        self.info.safetensors.parameters = {"BF16": 15, "U8": 2}
        self.info.safetensors.total = 17

    def assert_pinned_evidence(self):
        self.assertEqual(self.api.model_info.call_args_list, [
            call(MODEL, revision="release", expand=["sha", "gated"], timeout=REQUEST_TIMEOUT),
            call(MODEL, revision=SHA, expand=["safetensors"], timeout=REQUEST_TIMEOUT),
        ])
        self.assertEqual(self.download.call_count, 1)
        self.assertEqual(self.download.call_args.kwargs["repo_id"], MODEL)
        self.assertEqual(self.download.call_args.kwargs["revision"], SHA)
        self.assertEqual(self.download.call_args.kwargs["filename"], "config.json")

    def test_explicit_registry_supplies_geometry_and_parameters_to_shared_math(self):
        inspection = inspect_model(MODEL, revision="release", registry=self.registry)
        self.assert_pinned_evidence()
        self.assertEqual(inspection.spec.head_dim, 4)  # width / q_heads would incorrectly give 2
        self.assertEqual(inspection.spec.provenance["head_dim"], "channel")
        self.assertEqual(inspection.parameters.learned_parameter_count, 15)
        result = calculate_memory(inspection.spec, inspection.parameters, Workload(10, 5, 3), GPUCapacity(1, 0))
        self.assertEqual(result.weight_bytes, 30)
        self.assertEqual(result.kv_bytes_per_token, 64)
        self.assertEqual(result.kv_bytes_per_request, 960)
        self.assertEqual(result.workload_kv_bytes, 2880)
        self.assertEqual(result.known_memory_bytes, 2910)
        self.assertEqual(result.known_headroom_bytes, 1073738914)
        self.assertEqual(result.theoretical_max_concurrency, 1118481)
        self.assertTrue(result.workload_fits)
        self.assertIn("Toy full-attention decoder.", result.assumptions)
        self.normalize.assert_called_once_with(self.config)
        self.resolve.assert_called_once_with(self.config, ParameterSummary(15, {"BF16": 15}), None)
        self.api.get_safetensors_metadata.assert_not_called()

    def test_registered_family_reaches_real_cli_via_summary_without_headers(self):
        result = self.invoke()
        self.assertEqual(result.exit_code, 0, result.output)
        self.assert_pinned_evidence()
        for expected in (f"Hugging Face model: {MODEL}", f"Resolved revision: {SHA}",
                         "Adapter: test_decoder", "Parameter evidence: hub_safetensors",
                         "head_dim: 4 [from channel]", "Learned parameters: 15",
                         "Learned weight memory: 0.0000 GiB (30 bytes)", "KV per token: 64 bytes",
                         "Workload KV memory: 0.0000 GiB (2,880 bytes)",
                         "Known memory (weights + KV): 0.0000 GiB (2,910 bytes)",
                         "Theoretical maximum concurrency: 1,118,481", "Known workload fits: yes",
                         "Assumption: Toy summary represents learned weights.", "not a runtime guarantee"):
            self.assertIn(expected, result.output)
        self.api.get_safetensors_metadata.assert_not_called()

    def test_adapter_requests_headers_and_controls_buffer_accounting(self):
        self.require_headers()
        result = self.invoke()
        self.assertEqual(result.exit_code, 0, result.output)
        self.assert_pinned_evidence()
        self.api.get_safetensors_metadata.assert_called_once_with(MODEL, revision=SHA, timeout=REQUEST_TIMEOUT)
        self.assertIsInstance(self.resolve.call_args.args[2], TensorInventory)
        self.assertEqual(self.resolve.call_args.args[1], ParameterSummary(17, {"BF16": 15, "U8": 2}))
        self.assertIn("Learned parameters: 15", result.output)
        self.assertIn("Parameter evidence: safetensors_headers", result.output)
        self.assertIn("Excluded checkpoint buffers: 1 tensors, 2 stored bytes", result.output)
        self.assertIn("Warning: Toy cache hint is not a learned weight.", result.output)
        self.assertIn("Known memory (weights + KV): 0.0000 GiB (2,910 bytes)", result.output)

    def test_missing_summary_can_be_resolved_by_headers(self):
        self.info.safetensors = None
        result = self.invoke()
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIsNone(self.resolve.call_args.args[1])
        self.assertIn("Parameter evidence: safetensors_headers", result.output)
        self.assertIn("Learned parameters: 15", result.output)

    def test_unavailable_evidence_stays_unknown_through_output(self):
        self.info.safetensors = None
        self.api.get_safetensors_metadata.side_effect = NotASafetensorsRepoError()
        result = self.invoke()
        self.assertEqual(result.exit_code, 0, result.output)
        self.resolve.assert_called_once_with(self.config, None, None)
        for expected in ("Parameter evidence: unknown", "Known workload fits: unknown",
                         "Theoretical maximum concurrency: unknown", "KV per token: 64 bytes",
                         "Warning: Toy tensor evidence is unavailable."):
            self.assertIn(expected, result.output)

    def test_unrecognized_tensor_layout_preserves_adapter_warning(self):
        self.require_headers()
        self.file.tensors["projection.weight"].shape = [2, 6]
        result = self.invoke()
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Known workload fits: unknown", result.output)
        self.assertIn("Toy checkpoint layout is unrecognized", result.output)

    def test_unsupported_features_stop_before_parameter_resolution(self):
        self.config["vision"] = True
        result = self.invoke()
        self.assertEqual(result.exit_code, 1)
        self.assertIn("Toy vision mode is unsupported", result.output)
        self.needs.assert_not_called()
        self.resolve.assert_not_called()
        self.api.get_safetensors_metadata.assert_not_called()
        self.assertNotIn("Known workload fits:", result.output)

    def test_invalid_geometry_is_not_a_successful_estimate(self):
        self.config["channel"] = 0
        result = self.invoke()
        self.assertEqual(result.exit_code, 1)
        self.assertIn("Invalid toy geometry", result.output)
        self.resolve.assert_not_called()
        self.assertNotIn("Known workload fits:", result.output)

    def test_header_transport_error_does_not_become_missing_evidence(self):
        self.require_headers()
        self.api.get_safetensors_metadata.side_effect = ConnectError("unavailable")
        result = self.invoke()
        self.assertEqual(result.exit_code, 1)
        self.assertIn("inspect tensor headers", result.output)
        self.resolve.assert_not_called()
        self.assertNotIn("Known workload fits:", result.output)

    def test_test_family_is_not_registered_in_production(self):
        with self.assertRaises(UnsupportedArchitectureError):
            default_registry().get("test_decoder")
