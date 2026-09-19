import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.data_parsers.parser_factory import DataSourceConfig, ParserFactory, prepare_text_dataset, safe_destination
from src.inference.server_builder import InferenceConfig, InferenceServerBuilder
from src.inference.runtime import handler_for
from src.model_repo.hf_client import HuggingFaceClient
from src.training.qlora_pipeline import QLoRAPipeline, TrainingConfig


def test_local_source_normalization(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.txt").write_text("first")
    (source / "b.json").write_text('[{"text":"second"}]')
    raw = tmp_path / "raw"
    stats = ParserFactory.parse_data(DataSourceConfig("local", str(source)), raw)
    assert stats["files_processed"] == 2
    output = tmp_path / "prepared" / "data.jsonl"
    assert prepare_text_dataset(raw, output) == 2
    assert [json.loads(line)["text"] for line in output.read_text().splitlines()] == ["first", "second"]


def test_output_inside_source_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        ParserFactory.parse_data(DataSourceConfig("local", str(tmp_path)), tmp_path / "output")


@pytest.mark.parametrize("key", ["../escape", "/tmp/escape", "folder/../../escape"])
def test_cloud_object_cannot_escape_destination(tmp_path, key):
    with pytest.raises(ValueError, match="escapes"):
        safe_destination(tmp_path, key)


def test_factory_cloud_kwargs_are_specific():
    s3 = ParserFactory.create_parser("s3", oci_config_path="ignored", region_name="us-west-2")
    assert s3.region_name == "us-west-2"
    oci = ParserFactory.create_parser("oci", region_name="ignored", oci_config_path="config")
    assert oci.oci_config_path == "config"


def test_s3_files_are_downloaded_and_errors_reported(tmp_path):
    parser = ParserFactory.create_parser("s3")
    client = MagicMock()
    client.get_paginator.return_value.paginate.return_value = [
        {"Contents": [{"Key": "data/a.txt", "Size": 5}, {"Key": "../escape", "Size": 1}]}]
    client.download_file.side_effect = lambda bucket, key, destination: Path(destination).write_text("hello")
    parser._client = client
    stats = parser.parse(DataSourceConfig("s3", "", s3_bucket="bucket"), tmp_path)
    assert stats["files_processed"] == 1
    assert stats["errors"]
    assert (tmp_path / "data/a.txt").read_text() == "hello"


def test_oci_namespace_and_pagination(tmp_path):
    import sys
    sdk, object_storage = MagicMock(), MagicMock()
    sdk.pagination.list_call_get_all_results.return_value.data.objects = [MagicMock(size=5)]
    sdk.pagination.list_call_get_all_results.return_value.data.objects[0].name = "a.txt"
    client = object_storage.ObjectStorageClient.return_value
    client.get_object.return_value.data.read.return_value = b"hello"
    with patch.dict(sys.modules, {"oci": sdk, "oci.object_storage": object_storage}):
        parser = ParserFactory.create_parser("oci")
        parser._client = {"user": "user-is-not-namespace"}
        stats = parser.parse(DataSourceConfig("oci", "", oci_namespace="namespace", oci_bucket="bucket"), tmp_path)
    assert stats["files_processed"] == 1
    assert sdk.pagination.list_call_get_all_results.call_args.kwargs["namespace"] == "namespace"
    assert (tmp_path / "a.txt").read_bytes() == b"hello"


def test_training_rejects_bad_records(tmp_path):
    (tmp_path / "a.json").write_text('[{"wrong":"column"}]')
    with pytest.raises(ValueError, match="text string"):
        prepare_text_dataset(tmp_path, tmp_path.parent / "output.jsonl")


def test_training_requires_optional_dependencies():
    with patch("src.training.qlora_pipeline.torch", None):
        with pytest.raises(RuntimeError, match="requirements-training"):
            QLoRAPipeline().load_model()


def test_prepare_data_accepts_config_without_legacy_path(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.json").write_text('[{"text":"one"}, {"text":"two"}]')
    pipeline = QLoRAPipeline(TrainingConfig(output_dir=str(tmp_path / "out"),
        data_source_config=DataSourceConfig("local", str(source))))
    assert (pipeline.prepare_data() / "train.jsonl").is_file()
    with pytest.raises(ValueError, match="new output"):
        pipeline.prepare_data()


def test_model_card_reads_offline(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"model_storage_path": str(tmp_path / "models")}))
    client = HuggingFaceClient(str(config))
    model = tmp_path / "models" / "model"
    model.mkdir()
    (model / "README.md").write_text("# Saved model card")
    client._update_manifest("test/model", {"local_path": str(model), "downloaded_at": "now", "model_info": {}})
    with patch("src.model_repo.hf_client.hf_hub_download", side_effect=AssertionError("Network forbidden")):
        assert client.get_model_card("test/model") == "# Saved model card"


def test_generated_compose_builds_offline_cpu_runtime(tmp_path):
    model = tmp_path / 'model with "quotes"'
    model.mkdir()
    config = InferenceConfig("test/model", str(model), device="cpu")
    service = json.loads(InferenceServerBuilder().generate_config(config))["services"]["inference-server"]
    assert Path(service["build"]["context"], service["build"]["dockerfile"]).is_file()
    assert service["volumes"][0]["source"] == str(model)
    assert service["volumes"][0]["read_only"] is True
    assert service["environment"]["HF_HUB_OFFLINE"] == "1"
    assert service["ports"] == ["127.0.0.1:8000:8000"]
    assert "deploy" not in service


def test_gpu_compose_has_reservation(tmp_path):
    config = InferenceConfig("test", str(tmp_path), device="cuda")
    service = json.loads(InferenceServerBuilder().generate_config(config))["services"]["inference-server"]
    assert service["deploy"]["resources"]["reservations"]["devices"][0]["capabilities"] == ["gpu"]


def test_compose_failure_is_reported(tmp_path):
    builder = InferenceServerBuilder(str(tmp_path / "compose.json"))
    with patch.object(builder, "_get_compose_command", return_value=["docker", "compose"]), patch(
        "src.inference.server_builder.subprocess.run", return_value=MagicMock(returncode=1, stdout="", stderr="build failed")):
        result = builder.build_container(InferenceConfig("model", str(tmp_path), device="cpu"))
    assert result["success"] is False
    assert result["stderr"] == "build failed"


def http_request(raw_request, generate):
    class Connection:
        def __init__(self):
            self.output = io.BytesIO()
        def makefile(self, *args):
            return io.BytesIO(raw_request)
        def sendall(self, data):
            self.output.write(data)
    connection = Connection()
    handler_for(generate)(connection, ("127.0.0.1", 12345), MagicMock())
    return connection.output.getvalue()


def test_runtime_health_and_generation():
    generate = MagicMock(return_value="answer")
    health = http_request(b"GET /health HTTP/1.0\r\n\r\n", generate)
    assert b"200 OK" in health
    body = b'{"prompt":"Hello","max_new_tokens":4}'
    response = http_request(b"POST /generate HTTP/1.0\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body, generate)
    assert b'"text": "answer"' in response
    generate.assert_called_once_with("Hello", 4)


@pytest.mark.parametrize("body", [b"[]", b'{"prompt":""}', b'{"prompt":"x","max_new_tokens":-1}', b"not json"])
def test_runtime_rejects_invalid_requests(body):
    generate = MagicMock()
    response = http_request(b"POST /generate HTTP/1.0\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body, generate)
    assert b"400 Bad Request" in response
    generate.assert_not_called()


def test_local_model_registration_persists_and_preserves_source(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"model_storage_path": str(tmp_path / "repository")}))
    source = tmp_path / "existing-model"
    source.mkdir()
    (source / "config.json").write_text('{}')
    (source / "README.md").write_text('# Local model')
    client = HuggingFaceClient(str(config))
    entry = client.add_local_model(str(source))
    assert entry["local_path"] == str(source.resolve())
    assert client.add_local_model(str(source)) == entry
    reopened = HuggingFaceClient(str(config))
    assert len(reopened.list_models()) == 1
    assert reopened.get_model_card(entry["model_id"]) == '# Local model'
    assert reopened.delete_model(entry["model_id"])
    assert (source / "config.json").exists()
    assert reopened.list_models() == []


def test_local_model_rejects_non_model_directory(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"model_storage_path": str(tmp_path / "repository")}))
    client = HuggingFaceClient(str(config))
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="model folder"):
        client.add_local_model(str(empty))
    assert client.list_models() == []
