"""Generate reviewable commands; never provision a Pod or start paid compute."""

import shlex

from vramfit.measurement.config import finite, positive, validate


def launch_command(manifest, engine, port=8000, memory_fraction=.9):
    validate(manifest)
    positive(port, 'port')
    if port > 65535:
        raise ValueError('port must be <= 65535')
    finite(memory_fraction, 'memory_fraction')
    if memory_fraction >= 1:
        raise ValueError('memory_fraction must be below 1')
    model = manifest['model']
    context = min(manifest['decoder']['max_context_length'],
                  max(2048, max(sum(pair) for pair in manifest['workloads']['pairs'])))
    concurrency = max(manifest['workloads']['concurrency'])
    common = ['--revision', model['revision'], '--served-model-name', model['served_name'],
              '--dtype', model['dtype'], '--kv-cache-dtype', 'auto', '--host', '127.0.0.1', '--port', str(port)]
    if engine == 'vllm':
        argv = ['vllm', 'serve', model['id'], *common,
                '--tensor-parallel-size', '1', '--max-model-len', str(context),
                '--max-num-seqs', str(concurrency), '--max-num-batched-tokens', str(context),
                '--gpu-memory-utilization', str(memory_fraction), '--no-enable-prefix-caching',
                '--enforce-eager', '--generation-config', 'vllm']
        memory = {'gpu_memory_utilization': memory_fraction}
        scheduler = {'max_num_seqs': concurrency, 'max_num_batched_tokens': context}
    elif engine == 'sglang':
        argv = ['python', '-m', 'sglang.launch_server', '--model-path', model['id'], *common,
                '--tp-size', '1', '--context-length', str(context), '--max-running-requests', str(concurrency),
                '--chunked-prefill-size', str(context), '--mem-fraction-static', str(memory_fraction),
                '--disable-radix-cache', '--disable-cuda-graph', '--enable-metrics', '--sampling-defaults', 'openai']
        memory = {'mem_fraction_static': memory_fraction}
        scheduler = {'max_running_requests': concurrency, 'chunked_prefill_size': context}
    else:
        raise ValueError('engine must be vllm or sglang')
    return {'command': shlex.join(argv), 'argv': argv,
            'environment': {'engine': engine, 'engine_version': 'REPLACE_WITH_INSTALLED_VERSION',
                            'model_revision': model['revision'], 'dtype': model['dtype'], 'kv_dtype': model['dtype'],
                            'tensor_parallel_size': 1, 'prefix_caching': False, 'max_model_len': context,
                            'launch_command': shlex.join(argv), 'image_digest': None,
                            'gpu_model': 'REPLACE_WITH_ACTUAL_GPU', 'cuda_version': None, 'pytorch_version': None,
                            'graph_mode': 'eager', 'memory_settings': memory, 'scheduler_settings': scheduler}}
