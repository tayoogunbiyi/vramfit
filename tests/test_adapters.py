"""Contract checks that do not need a model library or network access."""

import unittest
from dataclasses import replace
from typing import Mapping

from vramfit.adapters.base import ModelAdapter
from vramfit.adapters.registry import (
    AdapterRegistry,
    DuplicateAdapterError,
    default_registry,
)
from vramfit.errors import InvalidModelConfigError, UnsupportedArchitectureError
from vramfit.models import (
    DecoderSpec,
    ParameterEstimate,
    ParameterSummary,
    TensorInventory,
)


class ExampleAdapter:
    """A test-only family with geometry supplied directly in its config."""

    model_type = "example"

    def normalize(self, config: Mapping[str, object]) -> DecoderSpec:
        if config.get("model_type") != self.model_type:
            raise InvalidModelConfigError("Expected model_type='example'.")
        if config.get("vision_config") is not None:
            raise UnsupportedArchitectureError(
                self.model_type, ("Contains a vision component.",)
            )
        try:
            return DecoderSpec(**config)
        except (TypeError, ValueError) as error:
            raise InvalidModelConfigError(str(error)) from error

    def needs_headers(
        self, config: Mapping[str, object], summary: ParameterSummary | None
    ) -> bool:
        return False

    def resolve_parameters(
        self,
        config: Mapping[str, object],
        summary: ParameterSummary | None,
        headers: TensorInventory | None,
    ) -> ParameterEstimate:
        if summary is None:
            return ParameterEstimate(
                None, "unknown", warnings=("No summary or implemented fallback.",)
            )
        return ParameterEstimate(summary.total, "hub_safetensors")


def example_config() -> dict[str, object]:
    return {
        "model_type": "example",
        "num_layers": 28,
        "hidden_size": 1024,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "max_context_length": 40960,
    }


class RegistryTests(unittest.TestCase):
    def test_custom_adapter_uses_shared_contract(self) -> None:
        adapter: ModelAdapter = ExampleAdapter()
        registry = AdapterRegistry([adapter])
        selected = registry.get("example")
        config = example_config()
        summary = ParameterSummary(1000, {"BF16": 1000})

        decoder = selected.normalize(config)
        self.assertIs(selected, adapter)
        self.assertEqual(decoder.head_dim, 128)
        self.assertFalse(selected.needs_headers(config, summary))
        self.assertEqual(
            selected.resolve_parameters(config, summary, None),
            ParameterEstimate(1000, "hub_safetensors"),
        )
        self.assertEqual(config, example_config())

    def test_duplicate_registration_preserves_original(self) -> None:
        original = ExampleAdapter()
        registry = AdapterRegistry([original])
        with self.assertRaises(DuplicateAdapterError):
            registry.register(ExampleAdapter())
        self.assertIs(registry.get("example"), original)

    def test_unknown_family_does_not_fall_back(self) -> None:
        registry = AdapterRegistry([ExampleAdapter()])
        with self.assertRaises(UnsupportedArchitectureError) as caught:
            registry.get("example_variant")
        self.assertEqual(caught.exception.model_type, "example_variant")
        self.assertEqual(
            caught.exception.reasons,
            ("No adapter is registered for this model type.",),
        )

    def test_default_registries_do_not_share_registration_state(self) -> None:
        first, second = default_registry(), default_registry()
        first.register(ExampleAdapter())
        with self.assertRaises(UnsupportedArchitectureError):
            second.get("example")

    def test_empty_model_type_is_rejected(self) -> None:
        adapter = ExampleAdapter()
        adapter.model_type = " "
        with self.assertRaises(ValueError):
            AdapterRegistry([adapter])


class ContractTests(unittest.TestCase):
    def test_explicit_head_dimension_need_not_equal_hidden_over_heads(self) -> None:
        spec = ExampleAdapter().normalize(example_config())
        self.assertNotEqual(spec.head_dim, spec.hidden_size // spec.num_attention_heads)

    def test_invalid_geometry_is_distinct_from_unsupported_architecture(self) -> None:
        adapter = ExampleAdapter()
        for updates in (
            {"num_layers": 0},
            {"num_layers": True},
            {"head_dim": 64.0},
            {"num_key_value_heads": 3},
            {"max_context_length": -1},
        ):
            with self.subTest(updates=updates):
                with self.assertRaises(InvalidModelConfigError):
                    adapter.normalize(example_config() | updates)
        with self.assertRaises(UnsupportedArchitectureError) as caught:
            adapter.normalize(example_config() | {"vision_config": {}})
        self.assertEqual(caught.exception.reasons, ("Contains a vision component.",))

    def test_missing_evidence_is_explained_unknown_not_zero(self) -> None:
        result = ExampleAdapter().resolve_parameters(example_config(), None, None)
        self.assertIsNone(result.learned_parameter_count)
        self.assertEqual(result.source, "unknown")
        self.assertTrue(result.warnings)

    def test_parameter_result_rejects_contradictory_evidence(self) -> None:
        known = ParameterEstimate(1000, "hub_safetensors")
        for updates in (
            {"learned_parameter_count": None},
            {"learned_parameter_count": 0},
            {"learned_parameter_count": True},
            {"source": "unknown"},
            {"source": "model_name"},
        ):
            with self.subTest(updates=updates):
                with self.assertRaises(ValueError):
                    replace(known, **updates)
        with self.assertRaises(ValueError):
            ParameterEstimate(None, "unknown")


if __name__ == "__main__":
    unittest.main()
