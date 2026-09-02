"""Command-line entry point for vramfit."""

import click

from vramfit import __version__
from vramfit.hub import ModelConfigDownloadError, download_model_config


@click.command(
    name="vramfit",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.argument("model_id", metavar="HF_MODEL_ID")
@click.option(
    "--vram",
    type=click.FloatRange(min=0, min_open=True),
    required=True,
    metavar="GiB",
    help="GPU memory available for the model, in GiB.",
)
@click.version_option(version=__version__, prog_name="vramfit")
def main(model_id: str, vram: float) -> None:
    """Estimate whether a Hugging Face model will fit in GPU memory.

    HF_MODEL_ID is a Hugging Face model repository ID.
    """
    try:
        config_path = download_model_config(model_id)
    except ModelConfigDownloadError as error:
        raise click.ClickException(str(error)) from error

    click.echo(f"Hugging Face model: {model_id}")
    click.echo(f"Config: {config_path}")
    click.echo(f"Available VRAM: {vram:g} GiB")
    click.echo("Estimation is not implemented yet.")
