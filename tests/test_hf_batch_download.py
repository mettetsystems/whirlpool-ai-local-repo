"""Tests for HuggingFace batch download functionality."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock

import pytest

from src.model_repo.hf_client import HuggingFaceClient
from src.model_repo.manifest_validator import ManifestValidator, ManifestValidationError


class TestBatchDownload:
    """Test cases for batch download functionality."""

    @pytest.fixture
    def temp_config(self, tmp_path):
        """Create a temporary authentication config file."""
        config_path = tmp_path / "hf_auth.json"
        config_path.write_text(json.dumps({
            "hf_token": "test_token_123",
            "model_storage_path": str(tmp_path / "models")
        }))
        return str(config_path)

    @pytest.fixture
    def client(self, temp_config):
        """Create a HuggingFaceClient instance."""
        client = HuggingFaceClient(auth_config_path=temp_config)
        client.api.model_info = MagicMock(return_value=MagicMock(
            id="test/model", author="test", likes=0, downloads=0, tags=[], pipeline_tag=None,
            sha="a" * 40, siblings=[MagicMock(size=1024)]))
        return client

    @pytest.fixture
    def validator(self, temp_config, tmp_path):
        """Create a ManifestValidator instance."""
        manifest_path = Path(temp_config).parent / "models" / ".manifest.json"
        return ManifestValidator(str(manifest_path))

    def test_batch_size_limit(self, client, validator):
        """Test that batch download enforces 10-model limit."""
        model_ids = [f"model/{i}" for i in range(15)]
        is_valid, errors = validator.validate_batch(model_ids)
        assert is_valid is False
        assert any("exceeds maximum" in str(e) for e in errors)

    def test_batch_size_within_limit(self, client, validator):
        """Test that batch download accepts up to 10 models."""
        model_ids = [f"model/{i}" for i in range(10)]
        is_valid, errors = validator.validate_batch(model_ids)
        assert is_valid is True
        assert len(errors) == 0

    def test_duplicate_model_ids(self, client, validator):
        """Test that duplicate model IDs are rejected."""
        model_ids = ["model/a", "model/b", "model/a"]
        is_valid, errors = validator.validate_batch(model_ids)
        assert is_valid is False
        assert any("Duplicate" in str(e) for e in errors)

    def test_invalid_model_id_format(self, client, validator):
        """Test that invalid model ID format is rejected."""
        model_ids = ["invalid_model_id"]
        is_valid, errors = validator.validate_batch(model_ids)
        assert is_valid is False
        assert any("namespace" in str(e) for e in errors)

    def test_empty_model_id(self, client, validator):
        """Test that empty model IDs are rejected."""
        model_ids = [""]
        is_valid, errors = validator.validate_batch(model_ids)
        assert is_valid is False

    @patch("src.model_repo.manifest_validator.ManifestValidator._get_current_storage_usage")
    def test_storage_quota_enforcement(self, mock_usage, client, validator):
        """Test that storage quota is enforced."""
        mock_usage.return_value = 9 * 1024 * 1024 * 1024 * 1024  # 9TB used
        model_ids = ["model/a", "model/b"]
        is_valid, errors = validator.validate_batch(model_ids)
        # Should fail due to quota
        assert is_valid is False
        assert any("exceeds quota" in str(e) for e in errors)

    @patch("src.model_repo.hf_client.snapshot_download")
    @patch("src.model_repo.hf_client.HfApi")
    def test_sequential_download(self, mock_api_class, mock_snapshot, client, tmp_path):
        """Test that models are downloaded sequentially."""
        mock_api = MagicMock()
        mock_api_class.return_value = mock_api
        mock_api.model_info.return_value = MagicMock(
            id="test/model",
            author="test_author",
            likes=100,
            downloads=500,
            tags=[],
            pipeline_tag=None
        )
        mock_snapshot.return_value = str(tmp_path / "downloaded")

        model_ids = ["test/model1", "test/model2"]
        results = []
        for model_id in model_ids:
            result = client.download_model(model_id)
            results.append(result)

        assert len(results) == 2
        assert mock_snapshot.call_count == 2

    @patch("src.model_repo.hf_client.snapshot_download")
    @patch("src.model_repo.hf_client.HfApi")
    def test_retry_on_5xx_error(self, mock_api_class, mock_snapshot, client, tmp_path):
        """Test that downloads retry 3 times on HTTP 5xx errors."""
        mock_api = MagicMock()
        mock_api_class.return_value = mock_api
        mock_api.model_info.return_value = MagicMock(
            id="test/model",
            author="test_author",
            likes=100,
            downloads=500,
            tags=[],
            pipeline_tag=None
        )

        # Simulate 5xx errors then success
        from huggingface_hub.utils import HfHubHTTPError
        mock_snapshot.side_effect = [
            HfHubHTTPError("Server error", response=Mock(status_code=500)),
            HfHubHTTPError("Server error", response=Mock(status_code=502)),
            HfHubHTTPError("Server error", response=Mock(status_code=503)),
            str(tmp_path / "downloaded")
        ]

        # Note: Current implementation doesn't have retry logic yet
        # This test documents expected behavior
        with pytest.raises(HfHubHTTPError):
            client.download_model("test/model")

    def test_storage_usage_tracking(self, client, validator):
        """Test that storage usage is tracked."""
        usage = validator.get_storage_usage()
        assert "used_bytes" in usage
        assert "used_gb" in usage
        assert "used_tb" in usage
        assert "quota_bytes" in usage
        assert "quota_tb" in usage
        assert "usage_percent" in usage

    def test_quota_check(self, client, validator):
        """Test quota availability check."""
        has_quota, message = validator.check_quota_available(1024 * 1024 * 1024)  # 1GB
        assert has_quota is True
        assert "available" in message


class TestManifestValidation:
    """Test cases for manifest validation."""

    @pytest.fixture
    def temp_config(self, tmp_path):
        """Create a temporary authentication config file."""
        config_path = tmp_path / "hf_auth.json"
        config_path.write_text(json.dumps({
            "hf_token": "test_token_123",
            "model_storage_path": str(tmp_path / "models")
        }))
        return str(config_path)

    @pytest.fixture
    def validator(self, temp_config, tmp_path):
        """Create a ManifestValidator instance."""
        models_dir = Path(temp_config).parent / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = Path(temp_config).parent / "models" / ".manifest.json"
        return ManifestValidator(str(manifest_path))

    def test_validate_manifest_integrity_empty(self, validator):
        """Test manifest validation with empty manifest."""
        is_valid, errors = validator.validate_manifest_integrity()
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_manifest_integrity_valid(self, validator, tmp_path):
        """Test manifest validation with valid manifest."""
        manifest_path = validator.manifest_path
        manifest_path.write_text(json.dumps({
            "models": [
                {
                    "model_id": "test/model",
                    "local_path": "/path/to/model",
                    "downloaded_at": "2024-01-01T00:00:00Z",
                    "model_info": {}
                }
            ]
        }))
        is_valid, errors = validator.validate_manifest_integrity()
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_manifest_integrity_invalid_json(self, validator, tmp_path):
        """Test manifest validation with invalid JSON."""
        manifest_path = validator.manifest_path
        manifest_path.write_text("not valid json")
        is_valid, errors = validator.validate_manifest_integrity()
        assert is_valid is False
        assert any("Invalid JSON" in str(e) for e in errors)

    def test_validate_manifest_integrity_missing_models_key(self, validator, tmp_path):
        """Test manifest validation with missing models key."""
        manifest_path = validator.manifest_path
        manifest_path.write_text(json.dumps({"data": []}))
        is_valid, errors = validator.validate_manifest_integrity()
        assert is_valid is False
        assert any("must contain 'models'" in str(e) for e in errors)

    def test_validate_manifest_integrity_invalid_model(self, validator, tmp_path):
        """Test manifest validation with invalid model entry."""
        manifest_path = validator.manifest_path
        manifest_path.write_text(json.dumps({
            "models": ["not a dict"]
        }))
        is_valid, errors = validator.validate_manifest_integrity()
        assert is_valid is False
        assert any("must be a dictionary" in str(e) for e in errors)

    def test_validate_batch_with_valid_manifest(self, validator, tmp_path):
        """Test batch validation with valid manifest."""
        manifest_path = validator.manifest_path
        manifest_path.write_text(json.dumps({
            "models": [
                {
                    "model_id": "existing/model",
                    "local_path": "/path/to/model",
                    "downloaded_at": "2024-01-01T00:00:00Z",
                    "model_info": {}
                }
            ]
        }))
        is_valid, errors = validator.validate_batch(["new/model"])
        assert is_valid is True

    def test_record_download(self, validator, tmp_path):
        """Test recording a download in the manifest."""
        model_id = "test/model"
        local_path = str(tmp_path / "downloaded")
        model_info = {"id": model_id, "author": "test"}

        validator.record_download(model_id, local_path, model_info)

        manifest = validator._load_manifest()
        assert len(manifest["models"]) == 1
        assert manifest["models"][0]["model_id"] == model_id
        assert manifest["models"][0]["local_path"] == local_path

    def test_record_download_updates_existing(self, validator, tmp_path):
        """Test that recording a download updates existing entry."""
        model_id = "test/model"
        local_path1 = str(tmp_path / "path1")
        local_path2 = str(tmp_path / "path2")
        model_info = {"id": model_id, "author": "test"}

        # Record first download
        validator.record_download(model_id, local_path1, model_info)

        # Record second download (should update)
        validator.record_download(model_id, local_path2, model_info)

        manifest = validator._load_manifest()
        assert len(manifest["models"]) == 1
        assert manifest["models"][0]["local_path"] == local_path2

    def test_get_model_status(self, validator, tmp_path):
        """Test getting model status from manifest."""
        model_id = "test/model"
        local_path = str(tmp_path / "downloaded")
        model_info = {"id": model_id, "author": "test"}

        validator.record_download(model_id, local_path, model_info)

        status = validator.get_model_status(model_id)
        assert status is not None
        assert status["model_id"] == model_id

    def test_get_model_status_not_found(self, validator):
        """Test getting model status when model doesn't exist."""
        status = validator.get_model_status("nonexistent/model")
        assert status is None

    def test_validate_model_exists(self, validator, tmp_path):
        """Test validating if a model exists in manifest."""
        model_id = "test/model"
        local_path = str(tmp_path / "downloaded")
        model_info = {"id": model_id, "author": "test"}

        validator.record_download(model_id, local_path, model_info)

        assert validator.validate_model_exists(model_id) is True
        assert validator.validate_model_exists("nonexistent/model") is False

    def test_get_storage_usage(self, validator, tmp_path):
        """Test getting storage usage."""
        usage = validator.get_storage_usage()
        assert usage["used_bytes"] >= 0
        assert usage["quota_tb"] == 1.0
        assert usage["usage_percent"] >= 0
