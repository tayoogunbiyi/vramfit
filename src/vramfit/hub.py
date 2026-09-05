"""Hugging Face Hub integration."""

import json
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping

from httpx import RequestError
from huggingface_hub import HfApi, hf_hub_download, logging as hf_logging
from huggingface_hub.errors import (
    GatedRepoError,
    HFValidationError,
    HfHubHTTPError,
    LocalEntryNotFoundError,
    NotASafetensorsRepoError,
    RemoteEntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
    SafetensorsParsingError,
)
from huggingface_hub.utils import disable_progress_bars

from vramfit import __version__
from vramfit.errors import InvalidModelConfigError
from vramfit.models import ParameterSummary, TensorInventory, TensorMetadata

CONFIG_FILENAME = "config.json"
REQUEST_TIMEOUT = 10


class ModelHubError(Exception):
    """Model evidence could not be retrieved from the Hub."""

    def __init__(self, message: str, *, status_code: int | None = None,
                 gated: str | bool | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.gated = gated


class ModelConfigDownloadError(ModelHubError):
    """Raised when a model config cannot be downloaded from the Hub."""


class ModelMetadataError(ModelHubError):
    """Hub evidence is malformed or inconsistent."""


@dataclass(frozen=True)
class ModelSnapshot:
    model_id: str
    revision: str
    config_path: Path
    config: Mapping[str, object]
    parameter_summary: ParameterSummary | None
    gated: str | bool | None = None


@contextmanager
def _hub_request(
    model_id: str,
    revision: str,
    action: str,
    error_type: type[ModelHubError] = ModelHubError,
    *,
    gated: str | bool | None = None,
) -> Iterator[None]:
    """Translate transport failures without treating them as absent evidence."""

    try:
        yield
    except HFValidationError as error:
        raise error_type(
            f"'{model_id}' is not a valid Hugging Face model ID."
        ) from error
    except HfHubHTTPError as error:
        status = error.response.status_code if error.response is not None else None
        gate = gated if gated is not None else (True if isinstance(error, GatedRepoError) else None)
        context = f"Could not {action} for '{model_id}' at '{revision}' (HTTP {status}). "
        if status == 401:
            message = context + "Authenticate with 'hf auth login' or supply a valid HF_TOKEN."
        elif status == 403 and (gate or isinstance(error, GatedRepoError)):
            message = context + f"Request or check access at https://huggingface.co/{model_id}. "
            if gate == "manual":
                message += "This repository requires manual approval; wait for the publisher's approval. "
            else:
                message += "Complete the repository's access requirements. "
            message += "If already approved, check the token's permissions. Authentication alone does not grant access."
        elif status == 403:
            message = context + "Access is forbidden. Check token permissions and repository or organization restrictions."
        elif isinstance(error, RevisionNotFoundError):
            message = f"Revision '{revision}' was not found for model '{model_id}'."
        elif isinstance(error, RepositoryNotFoundError):
            message = f"Model '{model_id}' could not be accessed. Check the model ID and credentials for private repositories."
        elif isinstance(error, RemoteEntryNotFoundError):
            message = f"A required file is missing while trying to {action} for '{model_id}' at '{revision}'."
        else:
            message = context + "Check Hub availability and try again."
        raise error_type(message, status_code=status, gated=gate) from error
    except (LocalEntryNotFoundError, RequestError, OSError) as error:
        raise error_type(
            f"Could not {action} for '{model_id}' at '{revision}'. "
            "Check your connection and cache permissions, then try again."
        ) from error


def _api() -> HfApi:
    hf_logging.set_verbosity_error()
    disable_progress_bars()
    return HfApi(library_name="vramfit", library_version=__version__)


def download_model_config(
    model_id: str, *, revision: str = "main", gated: str | bool | None = None
) -> Path:
    """Download and cache a config; retain the existing CLI entry point."""
    hf_logging.set_verbosity_error()
    disable_progress_bars()
    with _hub_request(
        model_id, revision, "download config.json", ModelConfigDownloadError, gated=gated
    ):
        return Path(
            hf_hub_download(
                repo_id=model_id,
                filename=CONFIG_FILENAME,
                repo_type="model",
                revision=revision,
                library_name="vramfit",
                library_version=__version__,
                etag_timeout=5,
            )
        )


def load_model_snapshot(model_id: str, *, revision: str = "main") -> ModelSnapshot:
    """Resolve once, then fetch config and summary at that immutable commit."""
    api = _api()
    with _hub_request(model_id, revision, "resolve model revision"):
        resolved = api.model_info(
            model_id, revision=revision, expand=["sha", "gated"], timeout=REQUEST_TIMEOUT
        )
    sha = resolved.sha
    if not isinstance(sha, str) or re.fullmatch(r"[0-9a-fA-F]{40}", sha) is None:
        raise ModelMetadataError("The Hub did not return an immutable commit SHA.")

    config_path = download_model_config(model_id, revision=sha, gated=resolved.gated)
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as error:
        raise InvalidModelConfigError(
            "config.json is not valid UTF-8 JSON."
        ) from error
    except OSError as error:
        raise ModelConfigDownloadError(
            "Could not read the cached config.json."
        ) from error
    if not isinstance(config, dict):
        raise InvalidModelConfigError("config.json must contain a JSON object.")

    with _hub_request(model_id, sha, "fetch parameter summary", gated=resolved.gated):
        info = api.model_info(
            model_id, revision=sha, expand=["safetensors"], timeout=REQUEST_TIMEOUT
        )
    summary = None
    if info.safetensors is not None:
        raw = info.safetensors
        counts = raw.parameters
        if (
            not isinstance(counts, dict)
            or any(not isinstance(k, str) for k in counts)
            or any(type(n) is not int or n < 0 for n in counts.values())
            or type(raw.total) is not int
            or raw.total < 0
            or raw.total != sum(counts.values())
        ):
            raise ModelMetadataError("The Hub returned an invalid parameter summary.")
        summary = ParameterSummary(raw.total, dict(counts))
    return ModelSnapshot(model_id, sha, config_path, config, summary, resolved.gated)


def load_tensor_inventory(snapshot: ModelSnapshot) -> TensorInventory | None:
    """Inspect the standard SafeTensors checkpoint only when requested.

    The Hub client selects model.safetensors, or follows the shard index when
    there is no single file, and fetches headers using HTTP byte ranges. None
    means neither standard checkpoint layout is available. Access, network and
    corrupt-checkpoint errors remain failures rather than missing evidence.
    """
    api = _api()
    with _hub_request(snapshot.model_id, snapshot.revision, "inspect tensor headers", gated=snapshot.gated):
        try:
            metadata = api.get_safetensors_metadata(
                snapshot.model_id,
                revision=snapshot.revision,
                timeout=REQUEST_TIMEOUT,
            )
        except NotASafetensorsRepoError:
            return None
        except (SafetensorsParsingError, ValueError, KeyError, TypeError) as error:
            raise ModelMetadataError("Could not parse the checkpoint metadata.") from error

    tensors: dict[str, TensorMetadata] = {}
    for filename, file_metadata in sorted(metadata.files_metadata.items()):
        for name, tensor in sorted(file_metadata.tensors.items()):
            if name in tensors or metadata.weight_map.get(name) != filename:
                raise ModelMetadataError(
                    "Tensor headers disagree with the checkpoint index."
                )
            shape = tuple(tensor.shape)
            offsets = tensor.data_offsets
            if (
                any(type(n) is not int or n < 0 for n in shape)
                or len(offsets) != 2
                or any(type(n) is not int or n < 0 for n in offsets)
                or offsets[1] < offsets[0]
            ):
                raise ModelMetadataError(
                    f"Invalid shape or byte offsets for tensor '{name}'."
                )
            tensors[name] = TensorMetadata(
                name, shape, tensor.dtype, offsets[1] - offsets[0]
            )
    if not tensors or tensors.keys() != metadata.weight_map.keys():
        raise ModelMetadataError("The checkpoint tensor inventory is incomplete.")
    return TensorInventory(tuple(tensors[name] for name in sorted(tensors)))
