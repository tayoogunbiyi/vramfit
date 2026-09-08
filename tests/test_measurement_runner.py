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
                self.assertEqual(len(report['client']['source_sha256']), 64)
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

    def test_real_preflight_requires_device_and_scheduler_before_requests(self):
        server = self.start_server(mode='no_metrics')
        with tempfile.TemporaryDirectory() as directory:
            report = asyncio.run(run(manifest(), Settings('vllm', server.url, server.url),
                                     Path(directory) / 'run', {'engine_version': 'test'}))
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(server.calls, 0)
        self.assertIn('metrics unavailable', report['error_detail'])

    def test_auth_failure_does_not_send_work_or_persist_token(self):
        server = self.start_server(token='expected')
        with tempfile.TemporaryDirectory() as directory:
            report = asyncio.run(run(manifest(), Settings('vllm', server.url, token='private-secret', synthetic=True),
                                     Path(directory) / 'run'))
            content = (Path(directory) / 'run' / 'run.json').read_text()
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(server.calls, 0)
        self.assertNotIn('private-secret', content)

    def test_summary_excludes_warmup_and_keeps_missing_memory_empty(self):
        from vramfit.measurement.report import summarize
        server = self.start_server()
        with tempfile.TemporaryDirectory() as directory:
            report = asyncio.run(run(manifest(), Settings('vllm', server.url, synthetic=True), Path(directory) / 'run'))
        rows = summarize(report)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row['successful_repetitions'] == 2 for row in rows))
        self.assertTrue(all(row['sampled_device_peak_bytes_max'] is None for row in rows))
        self.assertFalse(report['coverage']['real_device_and_scheduler_evidence_available'])

    def test_environment_attestation_matches_manifest_and_no_credentials(self):
        from vramfit.measurement.cli import environment_metadata
        from vramfit.measurement.launch import launch_command
        data = launch_command(manifest(), 'vllm')['environment']
        data.update(engine_version='test-version', gpu_model='test-gpu')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'env.json'
            path.write_text(json.dumps(data))
            self.assertEqual(environment_metadata(path, manifest(), 'vllm'), data)
            for patch_data in ({'dtype': 'float16'}, {'prefix_caching': True}, {'tensor_parallel_size': 2},
                               {'model_revision': 'b' * 40}, {'memory_settings': {}},
                               {'launch_command': 'vllm serve --api-key secret'}):
                path.write_text(json.dumps(data | patch_data))
                with self.assertRaises(ValueError):
                    environment_metadata(path, manifest(), 'vllm')

    def test_cancelled_run_retains_interrupted_status(self):
        server = self.start_server(mode='slow')
        async def cancel_run(path):
            task = asyncio.create_task(run(manifest(), Settings('vllm', server.url, synthetic=True), path))
            await asyncio.sleep(.1)
            task.cancel()
            return await task
        with tempfile.TemporaryDirectory() as directory:
            report = asyncio.run(cancel_run(Path(directory) / 'run'))
        self.assertEqual(report['status'], 'interrupted')

    def test_baseline_command_uses_authenticated_real_collector(self):
        from http.server import ThreadingHTTPServer
        from vramfit.measurement.collector import Handler
        from vramfit.measurement.telemetry import device_snapshot
        with ThreadingHTTPServer(('127.0.0.1', 0), Handler) as server:
            server.token, server.gpu_index = 'collector-secret', 0
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / 'baseline.json'
                    args = ['baseline', '--telemetry-endpoint', f'http://127.0.0.1:{server.server_port}', '--output', str(path)]
                    result = CliRunner().invoke(main, args, env={'VRAMFIT_COLLECTOR_TOKEN': ''})
                    self.assertEqual(result.exit_code, 1)
                    self.assertFalse(path.exists())
                    # Pin unavailable-device behavior independent of the CI host GPU.
                    with patch('vramfit.measurement.collector.device_snapshot', return_value={
                        'schema_version': 1, 'observed_at': '2026-09-08T00:00:00+00:00',
                        'device': None, 'unavailable_reason': 'test host has no GPU'}):
                        result = CliRunner().invoke(main, args, env={'VRAMFIT_COLLECTOR_TOKEN': 'collector-secret'})
                    self.assertEqual(result.exit_code, 0, result.output)
                    self.assertIsNone(json.loads(path.read_text())['device'])
                    self.assertNotIn('collector-secret', path.read_text())
            finally:
                server.shutdown()
                thread.join()

    def test_collector_refuses_unauthenticated_public_bind(self):
        from vramfit.measurement.collector import main as collector_main
        result = CliRunner().invoke(collector_main, ['--host', '0.0.0.0'], env={'VRAMFIT_COLLECTOR_TOKEN': ''})
        self.assertEqual(result.exit_code, 1)
        self.assertIn('requires a token', result.output)

    def test_mixed_gpu_samples_do_not_produce_a_peak(self):
        from vramfit.measurement.runner import summarize_samples
        samples = [{'device_snapshot': {'device': {'uuid': uuid, 'total_bytes': 100, 'used_bytes': 50}},
                    'engine': None, 'errors': []} for uuid in ('a', 'b')]
        self.assertIsNone(summarize_samples(samples)['sampled_device_peak_bytes'])

    def test_nonfinite_settings_fail_before_endpoint_requests(self):
        for value in (0, float('nan'), float('inf'), True):
            with self.assertRaises(ValueError):
                Settings('vllm', 'http://localhost:8000', max_duration=value)

    def test_real_requests_only_mode_is_not_gpu_validation(self):
        server = self.start_server()
        with tempfile.TemporaryDirectory() as directory:
            report = asyncio.run(run(manifest(), Settings('vllm', server.url, requests_only=True),
                                     Path(directory) / 'run', {'engine_version': 'test'}))
        self.assertEqual(report['status'], 'completed')
        self.assertFalse(report['synthetic'])
        self.assertEqual(report['measurement_mode'], 'requests_only')
        self.assertFalse(report['coverage']['real_device_and_scheduler_evidence_available'])
