"""Command-line entry point for vramfit."""

import click
from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

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
@click.option("--detailed", is_flag=True, help="Show the full memory breakdown, model evidence and assumptions.")
def main(
    model_id: str,
    revision: str,
    vram: float,
    dtype: str,
    target_concurrency: int,
    headroom: float,
    prompt_length: int,
    max_output_length: int,
    detailed: bool,
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

    if detailed:
        _render(inspection, estimate, workload, gpu)
    else:
        _render_summary(inspection, estimate, workload, gpu)


def _compact_memory(value: int | None) -> str:
    if value is None:
        return "unknown"
    if 0 < abs(value) < GIB / 100:
        return "<0.01 GiB"
    return f"{abs(value) / GIB:,.2f} GiB"


def _render_summary(
    inspection: ModelInspection, estimate: MemoryEstimate, workload: Workload, gpu: GPUCapacity,
    *, console: Console | None = None, detailed: bool = False,
) -> None:
    console = console or Console(markup=False, highlight=False, emoji=False)
    verdict, style = {
        True: ("FITS ESTIMATED BUDGET", "green"),
        False: ("EXCEEDS ESTIMATED BUDGET", "red"),
        None: ("FIT UNKNOWN", "yellow"),
    }[estimate.workload_fits]
    remaining_label = "Budget deficit" if estimate.workload_fits is False else "Remaining budget"
    rows = [
        ("Estimated memory", f"{_compact_memory(estimate.known_memory_bytes)} / "
         f"{_compact_memory(estimate.usable_vram_bytes)} usable"),
        (remaining_label, _compact_memory(estimate.known_headroom_bytes)),
        ("GPU", f"{gpu.vram_gib:g} GiB · {gpu.headroom_percent:g}% reserved"),
        ("Precision", f"{workload.dtype} weights + KV"),
        ("Workload", f"{workload.target_concurrency:,} concurrent "
         f"{'request' if workload.target_concurrency == 1 else 'requests'}"),
        ("Tokens / request", f"{workload.prompt_length:,} prompt + "
         f"{workload.max_output_length:,} max output"),
    ]
    heading = Text(f"{inspection.snapshot.model_id} · ")
    heading.append(verdict, style=style)
    console.print(heading)
    console.print()
    _render_rows(console, rows)
    console.print()
    if estimate.workload_fits is False:
        console.print("Weights fit, but KV cache at the requested concurrency exceeds the budget."
                      if estimate.weights_fit else "Weights alone exceed the usable budget.")
    if not detailed:
        # Keep model-specific uncertainty visible; the generic calculation assumptions
        # and Hub-summary provenance remain in the detailed report.
        for assumption in inspection.spec.assumptions:
            console.print(Text(f"Assumption: {assumption}"))
        if inspection.parameters.source != "hub_safetensors":
            for assumption in inspection.parameters.assumptions:
                console.print(Text(f"Assumption: {assumption}"))
        for warning in inspection.parameters.warnings:
            console.print(Text(f"Warning: {warning}"))
    console.print("Runtime overhead excluded; not a runtime guarantee.")
    if not detailed:
        console.print("Use --detailed for memory breakdown and model evidence.")


def _render_rows(console: Console, rows: list[tuple[str, str]]) -> None:
    if console.width < 60:
        for label, value in rows:
            console.print(Text(f"{label}: {value}"))
    else:
        table = Table(box=box.SQUARE, show_header=False)
        table.add_column(overflow="fold", max_width=32)
        table.add_column(overflow="fold")
        for label, value in rows:
            table.add_row(Text(label), Text(value))
        console.print(table)


def _memory(value: int | None) -> str:
    return "unknown" if value is None else f"{value / GIB:.4f} GiB ({value:,} bytes)"


def _fit(value: bool | None) -> str:
    if value is None:
        return "unknown"
    return "yes, within the calculated budget" if value else "no, exceeds the calculated budget"


def _render(
    inspection: ModelInspection, estimate: MemoryEstimate, workload: Workload, gpu: GPUCapacity,
) -> None:
    console = Console(markup=False, highlight=False, emoji=False)
    snapshot, spec, parameters = inspection.snapshot, inspection.spec, inspection.parameters
    _render_summary(inspection, estimate, workload, gpu, console=console, detailed=True)

    def section(title: str, rows: list[tuple[str, str]]) -> None:
        console.print()
        console.print(Text(title))
        if console.is_terminal:
            _render_rows(console, rows)
        else:
            # Preserve complete labelled evidence in redirected reports, without
            # terminal-width wrapping or borders that scripts would need to strip.
            for label, value in rows:
                click.echo(f"{label}: {value}")

    concurrency = estimate.theoretical_max_concurrency
    section("Memory breakdown", [
        ("Physical VRAM", _memory(estimate.physical_vram_bytes)),
        (f"Reserved headroom ({gpu.headroom_percent:g}%)", _memory(estimate.reserved_headroom_bytes)),
        ("Usable VRAM", _memory(estimate.usable_vram_bytes)),
        ("Learned weight memory", _memory(estimate.weight_bytes)),
        ("KV per token", f"{estimate.kv_bytes_per_token:,} bytes"),
        ("KV per request", _memory(estimate.kv_bytes_per_request)),
        ("Workload KV memory", _memory(estimate.workload_kv_bytes)),
        ("Known memory (weights + KV)", _memory(estimate.known_memory_bytes)),
        ("Remaining known budget (negative means deficit)", _memory(estimate.known_headroom_bytes)),
        ("Weights fit", _fit(estimate.weights_fit)),
        ("Known workload fits", _fit(estimate.workload_fits)),
        ("Theoretical maximum concurrency", "unknown" if concurrency is None else f"{concurrency:,}"),
    ])
    count = parameters.learned_parameter_count
    section("Model evidence", [
        ("Hugging Face model", snapshot.model_id),
        ("Resolved revision", snapshot.revision),
        ("Config", str(snapshot.config_path)),
        ("Adapter", f"{spec.model_type} (uniform full attention)"),
        ("Learned parameters", "unknown" if count is None else f"{count:,}"),
        ("Parameter evidence", parameters.source),
    ])
    section("Geometry (config fields or adapter defaults)", [
        (name, f"{getattr(spec, name)} [from {spec.provenance.get(name, 'adapter')}]")
        for name in ("num_layers", "hidden_size", "num_attention_heads",
                     "num_key_value_heads", "head_dim", "max_context_length")
    ])
    if parameters.checkpoint_buffers:
        buffers = parameters.checkpoint_buffers
        section("Checkpoint buffers", [
            ("Excluded checkpoint buffers", f"{len(buffers)} tensors, "
             f"{sum(t.stored_bytes for t in buffers):,} stored bytes (not runtime allocation)"),
            *((tensor.name, f"shape={tensor.shape}, dtype={tensor.dtype}") for tensor in buffers),
        ])
    section("Assumptions and warnings", [
        *(("Assumption", assumption) for assumption in estimate.assumptions),
        *(("Warning", warning) for warning in estimate.warnings),
    ])
