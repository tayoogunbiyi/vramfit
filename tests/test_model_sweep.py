"""Run the real Bash sweep against a fake CLI, without live Hub requests."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "model-sweep.sh"
FAKE_CLI = Path(__file__).parent / "fixtures" / "sweep_cli.py"


class ModelSweepTests(unittest.TestCase):
    def run_sweep(self, mode="normal", *, bad_report=False):
        with tempfile.TemporaryDirectory(prefix="vramfit-sweep-test-") as directory:
            report = Path(directory) / "report with spaces.txt"
            if bad_report:
                report = Path(directory) / "missing" / "report.txt"
            env = os.environ | {
                "VRAMFIT_CMD": f"{sys.executable} {FAKE_CLI}",
                "SWEEP_TEST_MODE": mode,
                # Avoid reading the user's .env or sending their real token to the fake.
                "HF_TOKEN": "test-only-token",
            }
            result = subprocess.run(["bash", str(SCRIPT), "-o", str(report)],
                                    env=env, capture_output=True, text=True, timeout=30)
            content = report.read_text() if report.exists() else ""
        self.assertNotIn("test-only-token", result.stdout + result.stderr + content)
        return result, content

    def test_expected_acceptances_rejections_and_gate_skip(self):
        result, report = self.run_sweep()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("assertions : 19 passed, 0 failed, 1 skipped", report)
        self.assertIn("models run : 20", report)
        self.assertIn("expected architecture rejection", report)
        self.assertIn("architecture not verified", report)
        self.assertIn("Known memory (weights + KV)", report)
        self.assertIn("exit status : 1", report)
        self.assertIn("Report written to:", result.stderr)

    def test_gated_repo_with_access_must_produce_known_estimate(self):
        result, report = self.run_sweep("gate_success")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("assertions : 20 passed, 0 failed, 0 skipped", report)

    def test_gated_401_is_skipped_not_passed(self):
        result, report = self.run_sweep("gate_401")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("assertions : 19 passed, 0 failed, 1 skipped", report)

    def test_known_estimate_need_not_fit_requested_gpu(self):
        result, report = self.run_sweep("no_fit")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Known workload fits: no, exceeds", report)
        self.assertIn("assertions : 19 passed, 0 failed, 1 skipped", report)

    def test_incomplete_wrong_or_unknown_supported_results_fail_and_continue(self):
        for mode in ("empty", "unknown", "wrong_family", "wrong_identity", "bad_revision", "missing_memory"):
            with self.subTest(mode=mode):
                result, report = self.run_sweep(mode)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn("assertions : 18 passed, 1 failed, 1 skipped", report)
                self.assertIn("models run : 20", report)

    def test_access_and_network_failures_on_public_models_fail(self):
        for mode in ("network", "public_auth"):
            with self.subTest(mode=mode):
                result, report = self.run_sweep(mode)
                self.assertEqual(result.returncode, 1)
                self.assertIn("assertions : 18 passed, 1 failed, 1 skipped", report)

    def test_arbitrary_failure_or_unexpected_success_is_not_expected_rejection(self):
        for mode in ("wrong_reason", "rejection_network", "rejection_exit2", "unexpected_accept"):
            with self.subTest(mode=mode):
                result, report = self.run_sweep(mode)
                self.assertEqual(result.returncode, 1)
                self.assertIn("assertions : 18 passed, 1 failed, 1 skipped", report)

    def test_gated_network_and_unknown_outcomes_are_not_access_skips(self):
        for mode in ("gate_network", "gate_unknown"):
            with self.subTest(mode=mode):
                result, report = self.run_sweep(mode)
                self.assertEqual(result.returncode, 1)
                self.assertIn("assertions : 19 passed, 1 failed, 0 skipped", report)

    def test_broken_command_is_setup_failure(self):
        result, report = self.run_sweep("version_failure")
        self.assertEqual(result.returncode, 2)
        self.assertIn("Cannot invoke CLI", result.stderr)
        self.assertEqual(report, "")

    def test_unwritable_report_is_setup_failure(self):
        result, report = self.run_sweep(bad_report=True)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(report, "")
