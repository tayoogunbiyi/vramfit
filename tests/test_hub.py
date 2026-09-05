import json
import unittest
from pathlib import Path
from unittest.mock import call, mock_open, patch

from httpx import ConnectError, Request, Response
from huggingface_hub import HfApi, ModelInfo
from huggingface_hub.errors import (
    GatedRepoError,
    HFValidationError,
    HfHubHTTPError,
    NotASafetensorsRepoError,
    RemoteEntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
    SafetensorsParsingError,
)
from huggingface_hub.utils import (
    SafetensorsFileMetadata,
    SafetensorsRepoMetadata,
    TensorInfo,
)

from vramfit.errors import InvalidModelConfigError
from vramfit.hub import (
    REQUEST_TIMEOUT,
    ModelConfigDownloadError,
    ModelHubError,
    ModelMetadataError,
    ModelSnapshot,
    download_model_config,
    load_model_snapshot,
    load_tensor_inventory,
)
from vramfit.models import ParameterSummary, TensorMetadata

MODEL = "example/model"
SHA = "a" * 40
CONFIG_PATH = Path("/mock-cache/config.json")
ERROR_RESPONSE = Response(403, request=Request("GET", "https://huggingface.co/mock"))
NOT_FOUND_RESPONSE = Response(404, request=Request("GET", "https://huggingface.co/mock"))


class SnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.api = self.enterContext(patch("vramfit.hub.HfApi", autospec=True)).return_value
        self.download = self.enterContext(patch(
            "vramfit.hub.hf_hub_download", return_value=str(CONFIG_PATH)
        ))
        self.read = self.enterContext(patch.object(
            Path, "read_text", return_value='{"model_type": "example"}'
        ))
        self.resolved = ModelInfo(id=MODEL, sha=SHA)
        self.info = ModelInfo(
            id=MODEL, safetensors={"total": 1001, "parameters": {"BF16": 1000, "U8": 1}}
        )
        self.api.model_info.side_effect = [self.resolved, self.info]

    def test_config_and_summary_use_resolved_commit_not_moving_branch(self) -> None:
        snapshot = load_model_snapshot(MODEL, revision="release")
        self.assertEqual(snapshot.revision, SHA)
        self.assertEqual(snapshot.config_path, CONFIG_PATH)
        self.assertEqual(snapshot.config, {"model_type": "example"})
        self.assertEqual(
            snapshot.parameter_summary, ParameterSummary(1001, {"BF16": 1000, "U8": 1})
        )
        self.assertEqual(self.api.model_info.call_args_list, [
            call(MODEL, revision="release", expand=["sha", "gated"], timeout=REQUEST_TIMEOUT),
            call(MODEL, revision=SHA, expand=["safetensors"], timeout=REQUEST_TIMEOUT),
        ])
        self.assertEqual(self.download.call_args.kwargs["revision"], SHA)
        self.assertEqual(self.download.call_args.kwargs["filename"], "config.json")
        self.api.get_safetensors_metadata.assert_not_called()

    def test_missing_summary_stays_none(self) -> None:
        self.api.model_info.side_effect = [self.resolved, ModelInfo(id=MODEL)]
        snapshot = load_model_snapshot(MODEL)
        self.assertIsNone(snapshot.parameter_summary)
        self.assertEqual(self.api.model_info.call_args_list[0].kwargs["revision"], "main")

    def test_bad_revision_resolution_stops_before_download(self) -> None:
        self.api.model_info.side_effect = [ModelInfo(id=MODEL, sha="main")]
        with self.assertRaises(ModelMetadataError):
            load_model_snapshot(MODEL)
        self.download.assert_not_called()

    def test_invalid_config_is_not_a_transport_error(self) -> None:
        for content in ("not json", "[]", "null"):
            with self.subTest(content=content):
                self.read.return_value = content
                self.api.model_info.side_effect = [self.resolved, self.info]
                with self.assertRaises(InvalidModelConfigError):
                    load_model_snapshot(MODEL)

    def test_inconsistent_summary_is_not_accepted(self) -> None:
        self.info.safetensors.total = 99
        with self.assertRaises(ModelMetadataError):
            load_model_snapshot(MODEL)

    def test_resolution_failures_have_actionable_messages(self) -> None:
        for error, message in (
            (HFValidationError("bad id"), "not a valid"),
            (GatedRepoError("denied", response=ERROR_RESPONSE), "Request or check access"),
            (RevisionNotFoundError("missing", response=NOT_FOUND_RESPONSE), "Revision 'release'"),
            (ConnectError("offline"), "resolve model revision"),
        ):
            with self.subTest(error=type(error).__name__):
                self.api.model_info.side_effect = error
                with self.assertRaisesRegex(ModelHubError, message) as caught:
                    load_model_snapshot(MODEL, revision="release")
                self.assertIs(caught.exception.__cause__, error)
        self.download.assert_not_called()

    def test_summary_network_failure_is_not_missing_metadata(self) -> None:
        self.api.model_info.side_effect = [self.resolved, ConnectError("offline")]
        with self.assertRaisesRegex(ModelHubError, "fetch parameter summary"):
            load_model_snapshot(MODEL)

    def test_missing_config_preserves_existing_download_error(self) -> None:
        self.download.side_effect = RemoteEntryNotFoundError(
            "missing", response=NOT_FOUND_RESPONSE
        )
        with self.assertRaisesRegex(ModelConfigDownloadError, "config.json"):
            load_model_snapshot(MODEL)

    def test_existing_download_entrypoint_still_works(self) -> None:
        self.assertEqual(download_model_config(MODEL), CONFIG_PATH)
        self.api.model_info.assert_not_called()

    def test_manual_gate_with_public_metadata_and_denied_config(self) -> None:
        # Visible metadata does not imply access to architecture or weight files.
        self.resolved.gated = "manual"
        self.resolved.safetensors = self.info.safetensors
        error = HfHubHTTPError("denied", response=ERROR_RESPONSE)
        self.download.side_effect = error
        with self.assertRaises(ModelConfigDownloadError) as caught:
            load_model_snapshot(MODEL)
        self.assertEqual(caught.exception.status_code, 403)
        self.assertEqual(caught.exception.gated, "manual")
        self.assertIs(caught.exception.__cause__, error)
        message = str(caught.exception)
        self.assertIn("manual approval", message)
        self.assertIn("https://huggingface.co/example/model", message)
        self.assertNotIn("hf auth login", message)
        self.read.assert_not_called()
        self.api.get_safetensors_metadata.assert_not_called()
        self.download.assert_called_once()
        self.assertEqual(self.download.call_args.kwargs["repo_id"], MODEL)

    def test_401_requests_authentication_even_on_gated_repo(self) -> None:
        response = Response(401, request=ERROR_RESPONSE.request)
        self.resolved.gated = "manual"
        self.download.side_effect = GatedRepoError("unauthorized", response=response)
        with self.assertRaisesRegex(ModelConfigDownloadError, "hf auth login") as caught:
            load_model_snapshot(MODEL)
        self.assertEqual(caught.exception.status_code, 401)
        self.assertNotIn("wait", str(caught.exception))

    def test_generic_403_is_not_mislabelled_as_gating(self) -> None:
        self.resolved.gated = False
        self.download.side_effect = HfHubHTTPError("forbidden", response=ERROR_RESPONSE)
        with self.assertRaisesRegex(ModelConfigDownloadError, "token permissions") as caught:
            load_model_snapshot(MODEL)
        self.assertEqual(caught.exception.status_code, 403)
        self.assertIs(caught.exception.gated, False)
        self.assertNotIn("approval", str(caught.exception))
        self.assertNotIn("Request or check access", str(caught.exception))

    def test_automatic_gate_does_not_claim_manual_approval(self) -> None:
        self.download.side_effect = GatedRepoError("denied", response=ERROR_RESPONSE)
        with self.assertRaisesRegex(ModelConfigDownloadError, "access requirements") as caught:
            download_model_config(MODEL, gated="auto")
        self.assertNotIn("manual", str(caught.exception))

    def test_missing_repository_is_not_claimed_to_be_gated(self) -> None:
        self.api.model_info.side_effect = RepositoryNotFoundError("missing", response=NOT_FOUND_RESPONSE)
        with self.assertRaisesRegex(ModelHubError, "Check the model ID") as caught:
            load_model_snapshot(MODEL)
        self.assertEqual(caught.exception.status_code, 404)
        self.assertIsNone(caught.exception.gated)

    def test_gate_context_survives_successful_snapshot(self) -> None:
        self.resolved.gated = "manual"
        self.assertEqual(load_model_snapshot(MODEL).gated, "manual")


class TensorInventoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.api = self.enterContext(patch("vramfit.hub.HfApi", autospec=True)).return_value
        self.snapshot = ModelSnapshot(MODEL, SHA, CONFIG_PATH, {}, None)
        self.file = SafetensorsFileMetadata({}, {
            "weight": TensorInfo("BF16", [2, 3], (0, 12)),
            "masked_bias": TensorInfo("F16", [], (12, 14)),
        })
        self.metadata = SafetensorsRepoMetadata(
            None, False,
            {name: "model.safetensors" for name in self.file.tensors},
            {"model.safetensors": self.file},
        )
        self.api.get_safetensors_metadata.return_value = self.metadata

    def test_headers_preserve_scalar_shapes_and_stored_bytes(self) -> None:
        inventory = load_tensor_inventory(self.snapshot)
        self.assertEqual(inventory.tensors, (
            TensorMetadata("masked_bias", (), "F16", 2),
            TensorMetadata("weight", (2, 3), "BF16", 12),
        ))
        self.api.get_safetensors_metadata.assert_called_once_with(
            MODEL, revision=SHA, timeout=REQUEST_TIMEOUT
        )
        self.api.model_info.assert_not_called()

    def test_no_standard_checkpoint_returns_none(self) -> None:
        self.api.get_safetensors_metadata.side_effect = NotASafetensorsRepoError()
        self.assertIsNone(load_tensor_inventory(self.snapshot))

    def test_header_access_failure_preserves_gate_context(self) -> None:
        from dataclasses import replace
        snapshot = replace(self.snapshot, gated="manual")
        self.api.get_safetensors_metadata.side_effect = HfHubHTTPError("denied", response=ERROR_RESPONSE)
        with self.assertRaisesRegex(ModelHubError, "manual approval") as caught:
            load_tensor_inventory(snapshot)
        self.assertEqual(caught.exception.status_code, 403)

    def test_header_failures_are_not_silently_missing_evidence(self) -> None:
        for error, expected in (
            (GatedRepoError("denied", response=ERROR_RESPONSE), ModelHubError),
            (ConnectError("offline"), ModelHubError),
            (SafetensorsParsingError("invalid"), ModelMetadataError),
            (RemoteEntryNotFoundError("missing shard", response=ERROR_RESPONSE), ModelHubError),
        ):
            with self.subTest(error=type(error).__name__):
                self.api.get_safetensors_metadata.side_effect = error
                with self.assertRaises(expected):
                    load_tensor_inventory(self.snapshot)

    def test_missing_or_misplaced_index_tensor_is_rejected(self) -> None:
        for weight_map in (
            {"weight": "model.safetensors", "missing": "model.safetensors"},
            {"weight": "wrong-shard.safetensors", "masked_bias": "model.safetensors"},
        ):
            with self.subTest(weight_map=weight_map):
                self.metadata.weight_map = weight_map
                with self.assertRaises(ModelMetadataError):
                    load_tensor_inventory(self.snapshot)

    def test_duplicate_names_across_shards_are_rejected(self) -> None:
        self.metadata.files_metadata["duplicate.safetensors"] = self.file
        with self.assertRaises(ModelMetadataError):
            load_tensor_inventory(self.snapshot)

    def test_invalid_offsets_are_rejected(self) -> None:
        self.file.tensors["weight"].data_offsets = (12, 0)
        with self.assertRaises(ModelMetadataError):
            load_tensor_inventory(self.snapshot)

    def test_hub_client_follows_only_selected_index_shards_at_same_sha(self) -> None:
        # Exercise the real library's index selection, mocking only its I/O.
        api = HfApi()
        index = {
            "weight_map": {"first": "one.safetensors", "second": "two.safetensors"}
        }
        with (
            patch("vramfit.hub.HfApi", return_value=api),
            patch.object(api, "file_exists", side_effect=[False, True]),
            patch.object(api, "hf_hub_download", return_value="/mock-index") as download,
            patch.object(api, "parse_safetensors_file_metadata") as parse,
            patch("builtins.open", mock_open(read_data=json.dumps(index))),
        ):
            def header(**kwargs):
                name = "first" if kwargs["filename"] == "one.safetensors" else "second"
                return SafetensorsFileMetadata({}, {
                    name: TensorInfo("BF16", [2], (0, 4))
                })
            parse.side_effect = header
            inventory = load_tensor_inventory(self.snapshot)
        self.assertEqual([t.name for t in inventory.tensors], ["first", "second"])
        self.assertEqual(download.call_args.kwargs["revision"], SHA)
        self.assertEqual(download.call_args.kwargs["filename"], "model.safetensors.index.json")
        self.assertEqual({c.kwargs["filename"] for c in parse.call_args_list},
                         {"one.safetensors", "two.safetensors"})
        self.assertTrue(all(c.kwargs["revision"] == SHA for c in parse.call_args_list))
        self.assertEqual(parse.call_count, 2)


if __name__ == "__main__":
    unittest.main()
