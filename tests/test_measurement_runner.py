import asyncio
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from click.testing import CliRunner

from measurement_fixture import FixtureServer, manifest
from vramfit.measurement.cli import main
from vramfit.measurement.config import validate
from vramfit.measurement.runner import Settings, run, summarize_samples
from vramfit.measurement.telemetry import device_snapshot, engine_metrics




class RunnerTests(unittest.TestCase):
    def start_server(self, engine='vllm', mode='normal', token=''):
        server = FixtureServer(engine, mode, token)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return server

    def execute(self, engine='vllm', mode='normal', **overrides):
        server = self.start_server(engine, mode, 'test-secret')
        directory = self.enterContext(tempfile.TemporaryDirectory())
        output = Path(directory) / 'run'
        settings = Settings(engine, server.url, server.url, sample_interval=.05,
                            token='test-secret', telemetry_token='test-secret', synthetic=True, **overrides)
        report = asyncio.run(run(manifest(), settings, output))
        return server, output, report

    def test_both_engines_real_http_and_durable_artifacts(self):
        for engine in ('vllm', 'sglang'):
            with self.subTest(engine=engine):
                server, output, report = self.execute(engine)
                self.assertEqual(report['status'], 'completed', report)
                self.assertTrue(report['synthetic'])
                self.assertEqual(server.calls, 15)
                self.assertEqual(server.maximum_running, 4)
                self.assertEqual(len(report['phases']), 6)
                self.assertEqual(report['loaded_idle']['sampled_device_peak_bytes'], 4 * 1024**3)
                measured = [p for p in report['phases'] if p['kind'] == 'measured']
                self.assertEqual(len(measured), 4)
                self.assertTrue(all(p['status'] == 'ok' for p in measured))
                self.assertEqual(measured[-1]['observations']['observed_running_requests_max'], 4)
                self.assertEqual(measured[-1]['observations']['sampled_device_peak_bytes'], 8 * 1024**3)
                raw = (output / 'samples.jsonl').read_text()
                self.assertNotIn('test-secret', raw + (output / 'run.json').read_text())
                self.assertTrue(all(json.loads(line)['phase'] for line in raw.splitlines()))
                self.assertEqual(json.loads((output / 'run.json').read_text())['status'], 'completed')
                with self.assertRaises(FileExistsError):
                    asyncio.run(run(manifest(), Settings(engine, server.url), output))

    def test_wrong_model_fails_before_inference(self):
        server, output, report = self.execute(mode='identity')
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(server.calls, 0)

    def test_short_outputs_and_oom_fail_without_retries(self):
        for mode in ('short', 'oom'):
            server, output, report = self.execute(mode=mode)
            self.assertEqual(report['status'], 'failed')
            self.assertEqual(server.calls, 1)
            request = report['phases'][0]['requests'][0]
            self.assertEqual(request['status'], 'token_mismatch' if mode == 'short' else 'failed')
            if mode == 'oom':
                self.assertEqual(request['error'], 'out_of_memory')
                self.assertNotIn('secret-do-not-save', (output / 'run.json').read_text())

    def test_missing_metrics_still_reports_device_and_missing_engine(self):
        _, _, report = self.execute(mode='no_metrics')
        self.assertEqual(report['status'], 'completed')
        observations = report['phases'][0]['observations']
        self.assertIsNone(observations['observed_running_requests_max'])
        self.assertIsNotNone(observations['sampled_device_peak_bytes'])
        self.assertGreater(observations['collection_error_count'], 0)

    def test_deadline_preserves_partial_report(self):
        _, output, report = self.execute(mode='slow', max_duration=.15)
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['error'], 'TimeoutError')
        self.assertTrue((output / 'samples.jsonl').exists())

    def test_request_timeout_recorded(self):
        _, _, report = self.execute(mode='slow', request_timeout=.05)
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['phases'][0]['requests'][0]['error'], 'ReadTimeout')

    def test_cli_preflight_rejects_real_run_without_environment(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / 'manifest.json'
            output_path = Path(directory) / 'results'
            manifest_path.write_text(json.dumps(manifest()))
            result = runner.invoke(main, ['run', '--manifest', str(manifest_path), '--engine', 'vllm',
                                          '--endpoint', 'http://localhost:8000', '--output', str(output_path)])
            self.assertEqual(result.exit_code, 1)
            self.assertIn('--environment', result.output)
            self.assertFalse(output_path.exists())
