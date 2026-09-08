#!/usr/bin/env python3
"""Exercise the full pilot against both synthetic engine protocols on localhost.

Runs the real CLI in child processes and the real device collector on this host.
No models are loaded. On a Mac, NVIDIA memory is correctly unavailable.
"""

import argparse
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
from measurement_fixture import FixtureServer, manifest
from vramfit.measurement.collector import Handler
from vramfit.measurement.config import PILOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New directory for smoke artifacts')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    document = manifest()
    document['workloads'] = PILOT
    manifest_path = args.output / 'manifest.json'
    manifest_path.write_text(json.dumps(document, indent=2))
    with ThreadingHTTPServer(('127.0.0.1', 0), Handler) as collector:
        collector.token, collector.gpu_index = '', 0
        thread = threading.Thread(target=collector.serve_forever, daemon=True)
        thread.start()
        try:
            for engine in ('vllm', 'sglang'):
                with FixtureServer(engine) as server:
                    serving = threading.Thread(target=server.serve_forever, daemon=True)
                    serving.start()
                    try:
                        command = [sys.executable, '-m', 'vramfit.measurement', 'run',
                                   '--manifest', str(manifest_path), '--engine', engine,
                                   '--endpoint', server.url, '--synthetic',
                                   '--telemetry-endpoint', f'http://127.0.0.1:{collector.server_port}',
                                   '--sample-interval', '0.05', '--max-duration', '60',
                                   '--output', str(args.output / engine)]
                        subprocess.run(command, check=True, timeout=75, cwd=ROOT)
                        report = json.loads((args.output / engine / 'run.json').read_text())
                        assert report['status'] == 'completed' and report['synthetic']
                        assert len(report['phases']) == 16
                        assert server.calls == 40
                        assert server.maximum_running == 4
                        assert all(row['status'] == 'ok' for row in report['phases'])
                        print(f'{engine}: full pilot verified; 40 synthetic requests, 12 measured batches', flush=True)
                    finally:
                        server.shutdown()
                        serving.join()
        finally:
            collector.shutdown()
            thread.join()
    print(f'SYNTHETIC local verification complete: {args.output}')


if __name__ == '__main__':
    main()
