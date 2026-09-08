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
        path, body = sglang.request([42] * 256, 128)
        self.assertEqual(path, '/generate')
        self.assertEqual(body['sampling_params']['min_new_tokens'], 128)
        self.assertEqual(vllm.usage({'usage': {'prompt_tokens': 256, 'completion_tokens': 128},
                                   'choices': [{'finish_reason': 'length'}]})['output_tokens'], 128)
        self.assertEqual(sglang.usage({'meta_info': {'prompt_tokens': 256, 'completion_tokens': 128}})['output_tokens'], 128)
        for backend in (vllm, sglang):
            with self.assertRaises(ProtocolError):
                backend.usage({})
