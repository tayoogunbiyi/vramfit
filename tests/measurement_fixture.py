"""Synthetic protocol server for Mac/CI testing. Does not load or simulate an LLM."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time


class FixtureServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, engine='vllm', mode='normal', token=''):
        super().__init__(('127.0.0.1', 0), FixtureHandler)
        self.engine, self.mode, self.token = engine, mode, token
        self.running = self.maximum_running = self.calls = 0
        self.lock = threading.Lock()
        self.payloads = []

    @property
    def url(self):
        return f'http://127.0.0.1:{self.server_port}'


class FixtureHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, data, status=200, content_type='application/json'):
        raw = (json.dumps(data) if content_type == 'application/json' else data).encode()
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        try:
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def authenticated(self):
        if self.server.token and self.headers.get('Authorization') != 'Bearer ' + self.server.token:
            self.reply({'error': 'unauthorized'}, 401)
            return False
        return True

    def do_GET(self):
        if not self.authenticated():
            return
        if self.path == '/v1/models':
            self.reply({'data': [{'id': 'wrong' if self.server.mode == 'identity' else 'fixture/model'}]})
        elif self.path == '/snapshot':
            self.reply({'schema_version': 1, 'observed_at': '2026-09-08T00:00:00+00:00',
                        'source': 'synthetic', 'device': {'uuid': 'SYNTHETIC-GPU', 'name': 'Synthetic device',
                        'total_bytes': 24 * 1024**3, 'used_bytes': (4 + self.server.running) * 1024**3}})
        elif self.path == '/metrics' and self.server.mode != 'no_metrics':
            running = 'vllm:num_requests_running' if self.server.engine == 'vllm' else 'sglang:num_running_reqs'
            cache = 'vllm:kv_cache_usage_perc' if self.server.engine == 'vllm' else 'sglang:token_usage'
            waiting = 'vllm:num_requests_waiting' if self.server.engine == 'vllm' else 'sglang:num_queue_reqs'
            self.reply(f'{running}{{model_name="fixture/model"}} {self.server.running}\n{cache} 0.25\n{waiting} 0\n',
                       content_type='text/plain')
        else:
            self.reply({'error': 'not found'}, 404)

    def do_POST(self):
        if not self.authenticated():
            return
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        with self.server.lock:
            self.server.calls += 1
            self.server.running += 1
            self.server.maximum_running = max(self.server.maximum_running, self.server.running)
            self.server.payloads.append(body)
        try:
            time.sleep(0.2 if self.server.mode != 'slow' else 1)
            if self.server.mode == 'oom':
                self.reply({'error': 'CUDA out of memory. secret-do-not-save'}, 500)
                return
            if self.server.engine == 'vllm':
                if self.path != '/v1/completions':
                    self.reply({}, 404)
                    return
                prompt, output = len(body['prompt']), body['max_tokens']
                if self.server.mode == 'short':
                    output -= 1
                self.reply({'usage': {'prompt_tokens': prompt, 'completion_tokens': output},
                            'choices': [{'text': 'not retained', 'finish_reason': 'length'}]})
            else:
                if self.path != '/generate':
                    self.reply({}, 404)
                    return
                self.reply({'text': 'not retained', 'meta_info': {'prompt_tokens': len(body['input_ids']),
                            'completion_tokens': body['sampling_params']['max_new_tokens'],
                            'finish_reason': {'type': 'length'}}})
        finally:
            with self.server.lock:
                self.server.running -= 1


def manifest():
    return {'schema_version': 1,
            'model': {'id': 'fixture/model', 'served_name': 'fixture/model', 'revision': 'a' * 40, 'dtype': 'bfloat16'},
            'gpu': {'vram_gib': 24, 'headroom_percent': 10},
            'decoder': {'model_type': 'llama', 'num_layers': 2, 'hidden_size': 8,
                        'num_attention_heads': 2, 'num_key_value_heads': 1, 'head_dim': 4, 'max_context_length': 2048},
            'parameters': {'learned_parameter_count': 1000, 'source': 'safetensors_headers'},
            'vocab_size': 1000, 'token_ids': list(range(100, 200)),
            'workloads': {'pairs': [[16, 8]], 'concurrency': [1, 4], 'warmups': 1, 'repetitions': 2}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', choices=['vllm', 'sglang'], required=True)
    args = parser.parse_args()
    with FixtureServer(args.engine) as server:
        print(f'SYNTHETIC protocol fixture: {server.url}', flush=True)
        server.serve_forever()
