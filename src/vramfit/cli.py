"""Command-line entry point for vramfit."""

import click

from vramfit import __version__
from vramfit.errors import ModelInspectionError
from vramfit.hub import ModelHubError
from vramfit.inspection import ModelInspection, inspect_model
from vramfit.memory import GIB, GPUCapacity, MemoryEstimate, Workload, calculate_memory


@click.command(
    name="vramfit",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.argument("model_id", metavar="HF_MODEL_ID")
@click.option("--revision", default="main", show_default=True,
              help="Hub branch, tag or commit; all evidence is pinned to its resolved SHA.")
@click.option(
    "--vram",
    type=click.FloatRange(min=0, min_open=True),
    required=True,
    metavar="GiB",
    help="Physical capacity of one GPU, in GiB, before reserving headroom.",
)
@click.option(
    "--dtype",
    type=click.Choice(("float32", "float16", "bfloat16"), case_sensitive=False),
    default="float16",
    show_default=True,
    help="Precision used for both learned weights and KV cache.",
)
@click.option(
    "--target-concurrency",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="Number of concurrent inference requests to support.",
)
@click.option(
    "--headroom",
    type=click.FloatRange(min=0, max=100, max_open=True),
    default=10,
    show_default=True,
    metavar="%",
    help="Percentage of GPU memory to leave unused.",
)
@click.option(
    "--prompt-length",
    type=click.IntRange(min=1),
    required=True,
    metavar="TOKENS",
    help="Prompt length per request, in tokens.",
)
@click.option(
    "--max-output-length",
    type=click.IntRange(min=1),
    required=True,
    metavar="TOKENS",
    help="Maximum generated length per request, in tokens.",
)
@click.version_option(version=__version__, prog_name="vramfit")
def main(
    model_id: str,
    revision: str,
    vram: float,
    dtype: str,
    target_concurrency: int,
    headroom: float,
    prompt_length: int,
    max_output_length: int,
) -> None:
    """Estimate whether a Hugging Face model will fit in GPU memory.

    HF_MODEL_ID is a Hugging Face model repository ID.
    """
    try:
        gpu = GPUCapacity(vram, headroom)
        workload = Workload(prompt_length, max_output_length, target_concurrency, dtype)
        inspection = inspect_model(model_id, revision=revision)
        estimate = calculate_memory(inspection.spec, inspection.parameters, workload, gpu)
    except (ModelHubError, ModelInspectionError, ValueError) as error:
        raise click.ClickException(str(error)) from error

    _render(inspection, estimate, workload, gpu)


def _memory(value: int | None) -> str:
    return "unknown" if value is None else f"{value / GIB:.4f} GiB ({value:,} bytes)"


def _fit(value: bool | None) -> str:
    if value is None:
        return "unknown"
    return "yes, within the calculated budget" if value else "no, exceeds the calculated budget"


def _render(
    inspection: ModelInspection, estimate: MemoryEstimate, workload: Workload, gpu: GPUCapacity,
) -> None:
    snapshot, spec, parameters = inspection.snapshot, inspection.spec, inspection.parameters
    click.echo(f"Hugging Face model: {snapshot.model_id}")
    click.echo(f"Resolved revision: {snapshot.revision}")
    click.echo(f"Config: {snapshot.config_path}")
    click.echo(f"Adapter: {spec.model_type} (uniform full attention)")
    count = parameters.learned_parameter_count
    click.echo(f"Learned parameters: {'unknown' if count is None else f'{count:,}'}")
    click.echo(f"Parameter evidence: {parameters.source}")
    click.echo("Geometry (config fields or adapter defaults):")
    for name in ("num_layers", "hidden_size", "num_attention_heads",
                 "num_key_value_heads", "head_dim", "max_context_length"):
        click.echo(f"  {name}: {getattr(spec, name)} [from {spec.provenance.get(name, 'adapter')}]")
    click.echo(f"Dtype (weights and KV): {workload.dtype}")
    click.echo(f"Tokens per request: {workload.tokens_per_request:,} "
               f"({workload.prompt_length:,} prompt + {workload.max_output_length:,} output)")
    click.echo(f"Target concurrency: {workload.target_concurrency:,}")
    click.echo(f"Physical VRAM: {_memory(estimate.physical_vram_bytes)}")
    click.echo(f"Reserved headroom ({gpu.headroom_percent:g}%): {_memory(estimate.reserved_headroom_bytes)}")
    click.echo(f"Usable VRAM: {_memory(estimate.usable_vram_bytes)}")
    click.echo(f"Learned weight memory: {_memory(estimate.weight_bytes)}")
    click.echo(f"KV per token: {estimate.kv_bytes_per_token:,} bytes")
    click.echo(f"KV per request: {_memory(estimate.kv_bytes_per_request)}")
    click.echo(f"Workload KV memory: {_memory(estimate.workload_kv_bytes)}")
    click.echo(f"Known memory (weights + KV): {_memory(estimate.known_memory_bytes)}")
    click.echo(f"Remaining known budget (negative means deficit): {_memory(estimate.known_headroom_bytes)}")
    click.echo(f"Weights fit: {_fit(estimate.weights_fit)}")
    click.echo(f"Known workload fits: {_fit(estimate.workload_fits)}")
    concurrency = estimate.theoretical_max_concurrency
    click.echo(f"Theoretical maximum concurrency: {'unknown' if concurrency is None else f'{concurrency:,}'}")
    if parameters.checkpoint_buffers:
        buffers = parameters.checkpoint_buffers
        click.echo(f"Excluded checkpoint buffers: {len(buffers)} tensors, "
                   f"{sum(t.stored_bytes for t in buffers):,} stored bytes (not runtime allocation)")
        for tensor in buffers:
            click.echo(f"  {tensor.name}: shape={tensor.shape}, dtype={tensor.dtype}")
    for assumption in estimate.assumptions:
        click.echo(f"Assumption: {assumption}")
    for warning in estimate.warnings:
        click.echo(f"Warning: {warning}")
