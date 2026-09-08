import unittest
from unittest.mock import patch
from vramfit.measurement.telemetry import device_snapshot, engine_metrics


class TelemetryTests(unittest.TestCase):
    def test_unavailable_is_not_zero(self):
        with patch('vramfit.measurement.telemetry.subprocess.run', side_effect=FileNotFoundError):
            self.assertIsNone(device_snapshot()['device'])

    def test_nvidia_units_and_bad_values(self):
        from subprocess import CompletedProcess
        with patch('vramfit.measurement.telemetry.subprocess.run', return_value=CompletedProcess([], 0, 'GPU-1, RTX 4090, 550.1, 24564, 1024\n')):
            self.assertEqual(device_snapshot()['device']['used_bytes'], 1024**3)
        with patch('vramfit.measurement.telemetry.subprocess.run', return_value=CompletedProcess([], 0, 'GPU-1, RTX, 550, NaN, -1\n')):
            self.assertIsNone(device_snapshot()['device'])

    def test_metrics_labels_aliases_and_ambiguous_workers(self):
        text = 'vllm:gpu_cache_usage_perc{model_name="m",engine="0"} 0.2\nvllm:num_requests_running{model_name="other"} 99\n'
        self.assertEqual(engine_metrics(text, 'vllm', 'm')['values']['cache_usage_ratio'], 0.2)
        self.assertIsNone(engine_metrics(text, 'vllm', 'm')['values']['running_requests'])
        text += 'vllm:gpu_cache_usage_perc{model_name="m",engine="1"} 0.3\n'
        self.assertIsNone(engine_metrics(text, 'vllm', 'm')['values']['cache_usage_ratio'])

