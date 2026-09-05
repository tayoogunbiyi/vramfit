import json
import unittest
from dataclasses import replace
from pathlib import Path

from vramfit.adapters.registry import default_registry
from vramfit.memory import GIB, GPUCapacity, Workload, calculate_memory
from vramfit.models import DecoderSpec, ParameterEstimate, TensorMetadata

FIXTURES = Path(__file__).parent / "fixtures"


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.spec = DecoderSpec("example", 2, 32, 4, 2, 8, 4096)
        self.parameters = ParameterEstimate(1000, "safetensors_headers")
        self.workload = Workload(10, 5, 3, "float16")

    def calculate(self, **overrides):
        args = dict(spec=self.spec, parameters=self.parameters,
                    workload=self.workload, gpu=GPUCapacity(1, 0)) | overrides
        return calculate_memory(**args)

    def test_exact_components(self):
        result = self.calculate()
        self.assertEqual(result.weight_bytes, 2000)
        self.assertEqual(result.kv_bytes_per_token, 128)
        self.assertEqual(result.kv_bytes_per_request, 1920)
        self.assertEqual(result.workload_kv_bytes, 5760)
        self.assertEqual(result.known_memory_bytes, 7760)
        self.assertEqual(result.physical_vram_bytes, GIB)
        self.assertEqual(result.reserved_headroom_bytes, 0)
        self.assertEqual(result.known_headroom_bytes, GIB - 7760)
        self.assertTrue(result.weights_fit)
        self.assertTrue(result.workload_fits)
        self.assertEqual(result.theoretical_max_concurrency, (GIB - 2000) // 1920)

    def test_dtype_applies_to_both_weights_and_cache(self):
        half = self.calculate()
        for dtype, factor in (("float16", 1), ("bfloat16", 1), ("float32", 2)):
            result = self.calculate(workload=replace(self.workload, dtype=dtype))
            self.assertEqual(result.weight_bytes, half.weight_bytes * factor)
            self.assertEqual(result.workload_kv_bytes, half.workload_kv_bytes * factor)

    def test_physical_capacity_and_reserved_headroom(self):
        result = self.calculate(gpu=GPUCapacity(24, 12.5))
        self.assertEqual(result.physical_vram_bytes, 24 * GIB)
        self.assertEqual(result.reserved_headroom_bytes, 3 * GIB)
        self.assertEqual(result.usable_vram_bytes, 21 * GIB)
        result = self.calculate(gpu=GPUCapacity(0.1, 10))
        self.assertEqual(result.physical_vram_bytes, 107374182)
        self.assertEqual(result.reserved_headroom_bytes, 10737419)
        self.assertEqual(result.usable_vram_bytes, 96636763)

    def test_exact_fit_and_one_byte_short(self):
        parameters = ParameterEstimate((GIB - 5760) // 2, "safetensors_headers")
        for reserve, fits in ((0, True), (100 / GIB, False)):
            result = self.calculate(parameters=parameters, gpu=GPUCapacity(1, reserve))
            self.assertEqual(result.workload_fits, fits)
            self.assertTrue(result.weights_fit)
            self.assertEqual(result.known_headroom_bytes, 0 if fits else -1)
            self.assertEqual(result.theoretical_max_concurrency, 3 if fits else 2)

    def test_weights_alone_exceed_budget(self):
        result = self.calculate(gpu=GPUCapacity(1000 / GIB, 0))
        self.assertFalse(result.weights_fit)
        self.assertFalse(result.workload_fits)
        self.assertEqual(result.theoretical_max_concurrency, 0)

    def test_unknown_weights_do_not_become_zero_or_a_fit(self):
        unknown = ParameterEstimate(None, "unknown", warnings=("No usable evidence.",))
        result = self.calculate(parameters=unknown)
        for name in ("weight_bytes", "known_memory_bytes", "known_headroom_bytes",
                     "weights_fit", "workload_fits", "theoretical_max_concurrency"):
            self.assertIsNone(getattr(result, name))
        self.assertEqual(result.workload_kv_bytes, 5760)
        self.assertIn("No usable evidence.", result.warnings)

    def test_context_limit_includes_prompt_and_output(self):
        self.calculate(workload=Workload(4000, 96))
        with self.assertRaisesRegex(ValueError, "exceeds the configured context limit"):
            self.calculate(workload=Workload(4000, 97))

    def test_workload_validation(self):
        for updates in ({"prompt_length": 0}, {"prompt_length": True},
                        {"max_output_length": -1}, {"target_concurrency": 1.5},
                        {"dtype": "int8"}, {"dtype": None}):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                replace(self.workload, **updates)

    def test_capacity_validation(self):
        for capacity, reserve in ((0, 0), (-1, 0), (float("nan"), 0),
                                  (float("inf"), 0), (1, float("nan")),
                                  (1, float("inf")), (1, -1), (1, 100), (True, 0)):
            with self.subTest(capacity=capacity, reserve=reserve), self.assertRaises(ValueError):
                GPUCapacity(capacity, reserve)

    def test_real_family_geometries_have_independent_kv_expectations(self):
        cases = (
            ("llama", "tinyllama.json", 22528),
            ("llama", "smollm2.json", 196608),
            ("qwen2", "qwen2.5-7b.json", 57344),
            ("qwen3", "qwen3-4b.json", 147456),
            ("mistral", "mistral-nemo-instruct-2407.json", 163840),
        )
        for family, file, expected in cases:
            with self.subTest(file=file):
                config = json.loads((FIXTURES / family / file).read_text())
                spec = default_registry().get(family).normalize(config)
                self.assertEqual(self.calculate(spec=spec).kv_bytes_per_token, expected)

    def test_checkpoint_buffers_are_not_runtime_weight_allocation(self):
        buffer = TensorMetadata("model.rotary_emb.inv_freq", (4,), "F32", 16)
        result = self.calculate(parameters=replace(self.parameters, checkpoint_buffers=(buffer,),
                                                  assumptions=("Example evidence assumption.",)))
        self.assertEqual(result.weight_bytes, 2000)
        self.assertIn("Example evidence assumption.", result.assumptions)
        self.assertTrue(any("buffers are unmodelled" in w for w in result.warnings))
        self.assertTrue(any("not a throughput" in a for a in result.assumptions))

    def test_long_context_and_high_concurrency_use_integer_bytes(self):
        spec = replace(self.spec, max_context_length=10**12)
        result = self.calculate(spec=spec, workload=Workload(10**9, 10**9, 10**9))
        self.assertEqual(result.workload_kv_bytes, 256 * 10**18)
        self.assertIsInstance(result.workload_kv_bytes, int)
