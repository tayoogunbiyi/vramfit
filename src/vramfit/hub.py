"""Hugging Face Hub integration."""

from pathlib import Path

from huggingface_hub import hf_hub_download, logging as hf_logging
from huggingface_hub.errors import (
    HFValidationError,
    HfHubHTTPError,
    LocalEntryNotFoundError,
    RemoteEntryNotFoundError,
    RepositoryNotFoundError,
)
from huggingface_hub.utils import disable_progress_bars

from vramfit import __version__

CONFIG_FILENAME = "config.json"


class ModelConfigDownloadError(Exception):
    """Raised when a model config cannot be downloaded from the Hub."""


def download_model_config(model_id: str) -> Path:
    """Download and cache a model's config, returning its local path."""
    hf_logging.set_verbosity_error()
    disable_progress_bars()

    try:
        config_path = hf_hub_download(
            repo_id=model_id,
            filename=CONFIG_FILENAME,
            repo_type="model",
            library_name="vramfit",
            library_version=__version__,
            etag_timeout=5,
        )
    except HFValidationError as error:
        raise ModelConfigDownloadError(
            f"'{model_id}' is not a valid Hugging Face model ID."
        ) from error
    except RemoteEntryNotFoundError as error:
        raise ModelConfigDownloadError(
            f"Model '{model_id}' does not contain {CONFIG_FILENAME}."
        ) from error
    except RepositoryNotFoundError as error:
        raise ModelConfigDownloadError(
            f"Model '{model_id}' could not be accessed. Check the model ID, "
            "request access if it is gated, or authenticate with "
            "'hf auth login' if it is private."
        ) from error
    except (LocalEntryNotFoundError, HfHubHTTPError, OSError) as error:
        raise ModelConfigDownloadError(
            f"Could not download or cache {CONFIG_FILENAME} for '{model_id}'. "
            "Check your connection and cache permissions, then try again."
        ) from error

    return Path(config_path)
