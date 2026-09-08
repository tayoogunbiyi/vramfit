"""Read-only GPU telemetry endpoint. Bind to loopback and access via SSH tunnel."""

from functools import partial
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os

import click

from vramfit.measurement.telemetry import device_snapshot


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.server.token and not hmac.compare_digest(
            self.headers.get("Authorization", ""), "Bearer " + self.server.token
        ):
            self.send_error(401)
            return
        if self.path != "/snapshot":
            self.send_error(404)
            return
        payload = json.dumps(device_snapshot(self.server.gpu_index), allow_nan=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


@click.command()
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", type=click.IntRange(1, 65535), default=9109, show_default=True)
@click.option("--gpu-index", type=click.IntRange(min=0), default=0, show_default=True)
@click.option("--token-env", default="VRAMFIT_COLLECTOR_TOKEN", show_default=True)
def main(host, port, gpu_index, token_env):
    """Expose NVIDIA device memory; no inference engine dependencies required."""
    token = os.environ.get(token_env, "")
    if host not in ("127.0.0.1", "localhost", "::1") and not token:
        raise click.ClickException("Non-loopback binding requires a token environment variable")
    try:
        with ThreadingHTTPServer((host, port), Handler) as server:
            server.token, server.gpu_index = token, gpu_index
            click.echo(f"Collector listening on {host}:{port}; GPU index {gpu_index}")
            server.serve_forever()
    except KeyboardInterrupt:
        pass
    except OSError as exc:
        raise click.ClickException(str(exc)) from exc
