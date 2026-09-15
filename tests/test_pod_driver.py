"""Exercise the real drivers with a local SSH transport and lightweight pod tools."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PodDriverTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="vramfit-pod-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        self.remote = self.root / "pod"
        self.bin = self.root / "bin"
        self.bin.mkdir()
        shutil.copytree(ROOT / "validation", self.repo / "validation",
                        ignore=shutil.ignore_patterns("runs*", "__pycache__"))
        shutil.copy(ROOT / ".gitignore", self.repo)
        shutil.copy(ROOT / "Makefile", self.repo)
        self.write_tool("ssh", '''#!/usr/bin/env python3
import os, sys
os.execvp("bash", ["bash", "-c", sys.argv[-1].replace("/workspace/vramfit", os.environ["TEST_POD"])])
''')
        self.write_tool("flock", '''#!/usr/bin/env python3
import fcntl, sys
try:
    fcntl.flock(int(sys.argv[-1]), fcntl.LOCK_EX | (fcntl.LOCK_NB if "-n" in sys.argv else 0))
except BlockingIOError:
    sys.exit(1)
''')
        self.write_tool("setsid", '''#!/usr/bin/env python3
import os, sys
os.setsid()
os.execvp(sys.argv[1], sys.argv[1:])
''')
        self.write_tool("sleep", "#!/bin/bash\n/bin/sleep 0.02\n")
        self.env = os.environ | {"PATH": f"{self.bin}:{os.environ['PATH']}",
                                "TEST_POD": str(self.remote), "POLL_INTERVAL": "0.02"}
        self.git("init", "-q")
        self.git("add", ".")
        self.git("-c", "user.name=Test", "-c", "user.email=test@example.com",
                 "commit", "-qm", "fixture")

    def write_tool(self, name, body):
        path = self.bin / name
        path.write_text(body)
        path.chmod(0o755)

    def git(self, *args):
        return subprocess.run(["git", "-C", str(self.repo), *args], check=True,
                              capture_output=True, text=True).stdout.strip()

    def pod(self, command, *, ok=True):
        result = subprocess.run(["bash", str(self.repo / "validation/pod.sh"),
                                 command, "fake", "22"], env=self.env,
                                capture_output=True, text=True, timeout=20)
        if ok:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def prepare(self, run_body='date > "$2/finished-at.txt"'):
        self.pod("ship")
        validation = self.remote / "validation"
        (validation / "runs").mkdir()
        for name in ("nvidia-smi.txt", "pip-freeze.txt"):
            (validation / "runs" / name).write_text("test environment\n")
        (validation / "serve.sh").write_text('''#!/bin/bash
mkdir -p "$3"
sleep 1000 &
echo $! > "$3/server.pid"
echo started > "$3/server.log"
exit "${TEST_STARTUP_STATUS:-0}"
'''.replace("sleep 1000", "/bin/sleep 1000"))
        (validation / "run.sh").write_text('#!/bin/bash\nset -eu\n' + run_body + '\n')

    def test_ship_tracks_commit_and_rejects_untracked_or_modified_scripts(self):
        self.pod("ship")
        self.assertEqual((self.remote / "COMMIT.txt").read_text().strip(),
                         self.git("rev-parse", "HEAD"))
        extra = self.repo / "validation/new.sh"
        extra.write_text("exit 1\n")
        self.assertIn("uncommitted", self.pod("ship", ok=False).stderr)
        extra.unlink()
        with (self.repo / "validation/run.sh").open("a") as output:
            output.write("# changed\n")
        self.pod("ship", ok=False)

    def test_smoke_fetches_records_and_replaces_stale_local_snapshot(self):
        self.prepare('test -r "$CASES_FILE"\ndate > "$2/finished-at.txt"')
        self.pod("smoke")
        local = self.repo / "validation/runs"
        model = local / "smoke/Qwen3-4B"
        self.assertTrue((model / "finished-at.txt").exists())
        self.assertTrue((model / "COMMIT.txt").exists())
        self.assertIn("C 8 4096 256 60", (local / "smoke/cases.txt").read_text())
        (model / "stale.txt").touch()
        self.pod("fetch")
        self.assertFalse((model / "stale.txt").exists())
        self.assertTrue(list((self.repo / "validation").glob("runs.old-*")))

    def test_full_resumes_but_rejects_different_commit(self):
        self.prepare('test -z "${CASES_FILE:-}"\ndate > "$2/finished-at.txt"')
        self.env |= {"YES": "1", "CASES_FILE": "/incorrect/table"}
        self.pod("run")
        self.assertEqual(self.pod("run").stdout.count("already finished"), 2)
        (self.remote / "COMMIT.txt").write_text("another commit\n")
        self.assertIn("different commit", self.pod("run", ok=False).stdout)

    def test_failed_startup_cleans_server_and_fetches_evidence(self):
        self.prepare()
        self.env["TEST_STARTUP_STATUS"] = "1"
        self.pod("smoke", ok=False)
        model = self.repo / "validation/runs/smoke/Qwen3-4B"
        self.assertTrue((model / "server.log").exists())
        self.assertFalse((model / "finished-at.txt").exists())
        pid = int((model / "server.pid").read_text())
        status = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)],
                                capture_output=True, text=True).stdout.strip()
        self.assertTrue(not status or status.startswith("Z"), status)

    def test_detached_job_reattaches_and_blocks_other_mode_and_shipping(self):
        self.prepare('/bin/sleep 2\ndate > "$2/finished-at.txt"')
        (self.remote / "jobs").mkdir()
        (self.remote / "jobs/full.pid").write_text("99999999\n")
        command = ["bash", str(self.repo / "validation/pod.sh"), "smoke", "fake", "22"]
        client = subprocess.Popen(command, env=self.env, stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL)
        self.addCleanup(lambda: client.poll() is None and client.kill())
        for _ in range(200):
            if (self.remote / "jobs/smoke.pid").exists():
                break
            time.sleep(0.01)
        else:
            self.fail("detached job never started")
        client.terminate()
        client.wait(timeout=5)
        self.env["YES"] = "1"
        self.assertIn("still running", self.pod("run", ok=False).stderr)
        # Repeating make's ship/setup preserves the active job's files/environment.
        self.pod("ship")
        self.pod("setup")
        self.git("-c", "user.name=Test", "-c", "user.email=test@example.com",
                 "commit", "--allow-empty", "-qm", "new commit")
        self.assertIn("Cannot ship", self.pod("ship", ok=False).stderr)
        self.assertIn("Reattaching", self.pod("smoke").stdout)

    def test_failed_case_is_fetched_and_partial_model_is_preserved(self):
        self.prepare('echo failed > "$2/guidellm.log"\nexit 1')
        self.env["YES"] = "1"
        self.pod("run", ok=False)
        run = self.remote / "validation/runs/Qwen3-4B"
        self.assertTrue((self.repo / "validation/runs/Qwen3-4B/guidellm.log").exists())
        (self.remote / "validation/run.sh").write_text('date > "$2/finished-at.txt"\n')
        self.pod("run")
        self.assertTrue((run / "finished-at.txt").exists())
        old = list(run.parent.glob("Qwen3-4B.old-*"))
        self.assertEqual(len(old), 1)
        self.assertEqual((old[0] / "guidellm.log").read_text(), "failed\n")

    def test_setup_installs_once_and_records_environment(self):
        self.pod("ship")
        # Keep the SSH/flock helpers' Python separate from the fake pod Python.
        self.write_tool("python3", """#!/bin/bash
printf '%s\\n' "$*" >> "$TEST_POD/install-calls"
if [[ $* == '-m uv venv --python 3.12 .venv' ]]; then mkdir -p .venv; fi
if [[ $* == '-m uv pip freeze --python .venv/bin/python' ]]; then echo 'vllm==0.11.0'; fi
""")
        import sys
        for name in ("ssh", "flock", "setsid"):
            path = self.bin / name
            path.write_text(path.read_text().replace("#!/usr/bin/env python3", f"#!{sys.executable}"))
        self.write_tool("nvidia-smi", "#!/bin/bash\necho 'test GPU'\n")
        self.pod("setup")
        self.pod("setup")
        calls = (self.remote / "install-calls").read_text()
        self.assertEqual(calls.count("-m uv pip install"), 1)
        self.assertIn("transformers==4.57.6", calls)
        self.assertEqual((self.remote / "validation/runs/COMMIT.txt").read_text().strip(),
                         self.git("rev-parse", "HEAD"))

    def test_declined_full_run(self):
        result = subprocess.run(["bash", str(self.repo / "validation/pod.sh"),
                                 "run", "fake", "22"], input="n\n", env=self.env,
                                capture_output=True, text=True, timeout=5)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Cancelled", result.stdout)
        self.assertFalse(self.remote.exists())
