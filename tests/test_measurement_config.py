import copy
import unittest
from vramfit.measurement.config import matrix, PILOT, validate
from vramfit.measurement.backends import Backend, ProtocolError, endpoint


class ConfigTests(unittest.TestCase):
    def test_pilot_has_four_cases(self):
        config = matrix(copy.deepcopy(PILOT))
        self.assertEqual(len(config['pairs']) * len(config['concurrency']), 4)
        for value in (0, True, float('nan'), '4'):
            bad = copy.deepcopy(config)
            bad['concurrency'] = [value]
            with self.assertRaises(ValueError):
                matrix(bad)

    def test_root_endpoint_validation(self):
        self.assertEqual(endpoint('http://localhost:8000/'), 'http://localhost:8000')
        for url in ('ftp://host', 'https://secret@host', 'http://host/v1', 'http://host?token=secret'):
            with self.assertRaises(ValueError):
                endpoint(url)

    def test_engine_payloads_and_usage(self):
        vllm = Backend('vllm', 'model')
        sglang = Backend('sglang', 'model')
        path, body = vllm.request([42] * 256, 128)
        self.assertEqual(path, '/v1/completions')
        self.assertEqual(len(body['prompt']), 256)
        self.assertTrue(body['ignore_eos'])
        # Metal rejects min_tokens; ignore_eos plus count validation is sufficient.
        self.assertNotIn('min_tokens', body)
        path, body = sglang.request([42] * 256, 128)
        self.assertEqual(path, '/generate')
        self.assertEqual(body['sampling_params']['min_new_tokens'], 128)
        self.assertEqual(vllm.usage({'usage': {'prompt_tokens': 256, 'completion_tokens': 128},
                                   'choices': [{'finish_reason': 'length'}]})['output_tokens'], 128)
        self.assertEqual(sglang.usage({'meta_info': {'prompt_tokens': 256, 'completion_tokens': 128}})['output_tokens'], 128)
        for backend in (vllm, sglang):
            with self.assertRaises(ProtocolError):
                backend.usage({})


class LaunchTests(unittest.TestCase):
    def test_commands_pin_model_and_match_pilot(self):
        from measurement_fixture import manifest
        from vramfit.measurement.launch import launch_command
        import shlex
        for engine in ('vllm', 'sglang'):
            data = manifest()
            data['model']['served_name'] = 'model; echo untrusted'
            plan = launch_command(data, engine)
            self.assertEqual(shlex.split(plan['command']), plan['argv'])
            self.assertIn('a' * 40, plan['argv'])
            self.assertIn('model; echo untrusted', plan['argv'])
            self.assertEqual(plan['environment']['max_model_len'], 2048)
            self.assertFalse(plan['environment']['prefix_caching'])
            self.assertEqual(plan['environment']['graph_mode'], 'eager')


class PrepareTests(unittest.TestCase):
    def test_prepare_pins_revision_and_roundtrips_exact_estimate(self):
        import json
        from pathlib import Path
        from unittest.mock import patch
        from vramfit.hub import ModelSnapshot
        from vramfit.models import ParameterSummary
        from vramfit.measurement.config import prepare, estimate
        fixture = Path(__file__).parent / 'fixtures/llama/tinyllama.json'
        config = json.loads(fixture.read_text())
        snapshot = ModelSnapshot('fixture/model', 'a' * 40, fixture, config,
                                 ParameterSummary(1_100_048_384, {'BF16': 1_100_048_384}))
        with patch('vramfit.inspection.load_model_snapshot', return_value=snapshot) as load:
            data = prepare('fixture/model', 'release', 'alias', 'bfloat16', 24, 10, copy.deepcopy(PILOT))
        load.assert_called_once_with('fixture/model', revision='release')
        data = validate(json.loads(json.dumps(data)))
        self.assertEqual(data['model']['revision'], 'a' * 40)
        self.assertEqual(data['model']['served_name'], 'alias')
        self.assertEqual(estimate(data, 1024, 512, 1)['weight_bytes'], 2_200_096_768)
        self.assertNotIn(config['eos_token_id'], data['token_ids'])

    def test_unpinned_or_malformed_manifest_is_rejected(self):
        from measurement_fixture import manifest
        for bad in (None, [], {'schema_version': 99}):
            with self.assertRaises(ValueError):
                validate(bad)
        bad = manifest()
        bad['model']['revision'] = 'main'
        with self.assertRaises(ValueError):
            validate(bad)
        bad = manifest()
        bad['token_ids'] = [1000]
        with self.assertRaises(ValueError):
            validate(bad)
