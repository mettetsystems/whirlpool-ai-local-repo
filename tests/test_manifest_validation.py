"""Tests for manifest validation functionality."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.model_repo.manifest_validator import ManifestValidator, ManifestValidationError


class TestManifestValidator:
    """Test cases for ManifestValidator."""

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

    def test_init_creates_validator(self, temp_config, tmp_path):
        """Test that validator initializes correctly."""
        manifest_path = Path(temp_config).parent / "models" / ".manifest.json"
        validator = ManifestValidator(str(manifest_path))
        assert validator.manifest_path == manifest_path

    def test_max_models_per_batch(self, validator):
        """Test that max models per batch is 10."""
        assert validator.MAX_MODELS_PER_BATCH == 10

    def test_max_storage_quota(self, validator):
        """Test that max storage quota is 1 TiB."""
        assert validator.MAX_STORAGE_QUOTA_BYTES == 1024 ** 4

    def test_validate_batch_empty(self, validator):
        """Test validating empty batch."""
        is_valid, errors = validator.validate_batch([])
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_batch_single_model(self, validator):
        """Test validating single model."""
        is_valid, errors = validator.validate_batch(["test/model"])
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_batch_exactly_10_models(self, validator):
        """Test validating exactly 10 models (should pass)."""
        model_ids = [f"model/{i}" for i in range(10)]
        is_valid, errors = validator.validate_batch(model_ids)
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_batch_11_models(self, validator):
        """Test validating 11 models (should fail)."""
        model_ids = [f"model/{i}" for i in range(11)]
        is_valid, errors = validator.validate_batch(model_ids)
        assert is_valid is False
        assert any("exceeds maximum" in str(e) for e in errors)

    def test_validate_batch_duplicate(self, validator):
        """Test validating batch with duplicates."""
        model_ids = ["model/a", "model/b", "model/a"]
        is_valid, errors = validator.validate_batch(model_ids)
        assert is_valid is False
        assert any("Duplicate" in str(e) for e in errors)

    def test_validate_batch_invalid_format(self, validator):
        """Test validating batch with invalid model ID format."""
        model_ids = ["invalid"]
        is_valid, errors = validator.validate_batch(model_ids)
        assert is_valid is False
        assert any("namespace" in str(e) for e in errors)

    def test_validate_batch_mixed_valid_invalid(self, validator):
        """Test validating batch with mixed valid and invalid IDs."""
        model_ids = ["valid/model", "invalid"]
        is_valid, errors = validator.validate_batch(model_ids)
        assert is_valid is False
        assert any("namespace" in str(e) for e in errors)

    def test_get_current_storage_usage_empty(self, validator):
        """Test getting storage usage with empty manifest."""
        usage = validator._get_current_storage_usage()
        assert usage == 0

    def test_estimate_batch_size(self, validator):
        """Test estimating batch size."""
        model_ids = ["model/a", "model/b", "model/c"]
        estimated = validator._estimate_batch_size(model_ids)
        # Should be 3 * 1GB = 3GB
        assert estimated == 3 * 1024 * 1024 * 1024

    def test_get_model_status_not_found(self, validator):
        """Test getting model status when not found."""
        status = validator.get_model_status("nonexistent/model")
        assert status is None

    def test_record_download_new_model(self, validator, tmp_path):
        """Test recording a new model download."""
        model_id = "test/model"
        local_path = str(tmp_path / "downloaded")
        model_info = {"id": model_id, "author": "test"}

        validator.record_download(model_id, local_path, model_info)

        manifest = validator._load_manifest()
        assert len(manifest["models"]) == 1
        assert manifest["models"][0]["model_id"] == model_id
        assert manifest["models"][0]["local_path"] == local_path
        assert "downloaded_at" in manifest["models"][0]
        assert manifest["models"][0]["model_info"] == model_info

    def test_record_download_updates_existing(self, validator, tmp_path):
        """Test that recording updates existing model."""
        model_id = "test/model"
        local_path1 = str(tmp_path / "path1")
        local_path2 = str(tmp_path / "path2")
        model_info = {"id": model_id, "author": "test"}

        # Record first
        validator.record_download(model_id, local_path1, model_info)
        # Record second (should update)
        validator.record_download(model_id, local_path2, model_info)

        manifest = validator._load_manifest()
        assert len(manifest["models"]) == 1
        assert manifest["models"][0]["local_path"] == local_path2

    def test_get_storage_usage(self, validator):
        """Test getting storage usage info."""
        usage = validator.get_storage_usage()
        assert "used_bytes" in usage
        assert "used_gb" in usage
        assert "used_tb" in usage
        assert "quota_bytes" in usage
        assert "quota_gb" in usage
        assert "quota_tb" in usage
        assert "usage_percent" in usage
        assert "available_bytes" in usage

    def test_check_quota_available(self, validator):
        """Test checking quota availability."""
        has_quota, message = validator.check_quota_available(1024 * 1024 * 1024)  # 1GB
        assert has_quota is True
        assert "available" in message

    def test_check_quota_exceeded(self, validator, tmp_path, monkeypatch):
        """Test checking quota when exceeded."""
        (tmp_path / 'models').mkdir(parents=True, exist_ok=True)
        # Create a fake large file to simulate high usage
        model_dir = tmp_path / "large_model"
        model_dir.mkdir()
        large_file = model_dir / "large_file.bin"
        # Exercise real file accounting with a 1 KiB fixture and a scaled quota.
        large_file.write_bytes(b"x" * 1024)
        monkeypatch.setattr(validator, "MAX_STORAGE_QUOTA_BYTES", 1024)

        # Update manifest with this model
        validator.record_download(
            "large/model",
            str(model_dir),
            {"id": "large/model"}
        )

        has_quota, message = validator.check_quota_available(1)
        assert has_quota is False
        assert "Insufficient storage" in message

    def test_validate_manifest_integrity_valid(self, validator, tmp_path):
        """Test validating a valid manifest."""
        manifest_path = validator.manifest_path
        manifest_path.write_text(json.dumps({
            "models": [
                {
                    "model_id": "test/model",
                    "local_path": "/path/to/model",
                    "downloaded_at": "2024-01-01T00:00:00Z",
                    "model_info": {"id": "test/model"}
                }
            ]
        }))
        is_valid, errors = validator.validate_manifest_integrity()
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_manifest_integrity_invalid_json(self, validator, tmp_path):
        """Test validating invalid JSON manifest."""
        manifest_path = validator.manifest_path
        manifest_path.write_text("not valid json")
        is_valid, errors = validator.validate_manifest_integrity()
        assert is_valid is False
        assert any("Invalid JSON" in str(e) for e in errors)

    def test_validate_manifest_integrity_missing_models(self, validator, tmp_path):
        """Test validating manifest without models key."""
        manifest_path = validator.manifest_path
        manifest_path.write_text(json.dumps({"data": []}))
        is_valid, errors = validator.validate_manifest_integrity()
        assert is_valid is False
        assert any("must contain 'models'" in str(e) for e in errors)

    def test_validate_manifest_integrity_invalid_models_type(self, validator, tmp_path):
        """Test validating manifest with non-list models."""
        manifest_path = validator.manifest_path
        manifest_path.write_text(json.dumps({"models": "not a list"}))
        is_valid, errors = validator.validate_manifest_integrity()
        assert is_valid is False
        assert any("must be a list" in str(e) for e in errors)

    def test_validate_manifest_integrity_invalid_model_entry(self, validator, tmp_path):
        """Test validating manifest with invalid model entry."""
        manifest_path = validator.manifest_path
        manifest_path.write_text(json.dumps({
            "models": ["not a dict"]
        }))
        is_valid, errors = validator.validate_manifest_integrity()
        assert is_valid is False
        assert any("must be a dictionary" in str(e) for e in errors)

    def test_validate_manifest_integrity_missing_model_id(self, validator, tmp_path):
        """Test validating manifest with missing model_id."""
        manifest_path = validator.manifest_path
        manifest_path.write_text(json.dumps({
            "models": [{"local_path": "/path"}]
        }))
        is_valid, errors = validator.validate_manifest_integrity()
        assert is_valid is False
        assert any("missing 'model_id'" in str(e) for e in errors)

    def test_validate_manifest_integrity_missing_local_path(self, validator, tmp_path):
        """Test validating manifest with missing local_path."""
        manifest_path = validator.manifest_path
        manifest_path.write_text(json.dumps({
            "models": [{"model_id": "test/model"}]
        }))
        is_valid, errors = validator.validate_manifest_integrity()
        assert is_valid is False
        assert any("missing 'local_path'" in str(e) for e in errors)

    def test_validate_model_exists_true(self, validator, tmp_path):
        """Test validate_model_exists returns True."""
        model_id = "test/model"
        local_path = str(tmp_path / "downloaded")
        validator.record_download(model_id, local_path, {"id": model_id})
        assert validator.validate_model_exists(model_id) is True

    def test_validate_model_exists_false(self, validator):
        """Test validate_model_exists returns False."""
        assert validator.validate_model_exists("nonexistent/model") is False

    def test_load_manifest_empty(self, validator):
        """Test loading empty manifest."""
        manifest = validator._load_manifest()
        assert manifest == {"models": []}

    def test_save_manifest(self, validator, tmp_path):
        """Test saving manifest."""
        manifest = {"models": [{"model_id": "test/model"}]}
        validator._save_manifest(manifest)
        saved = validator._load_manifest()
        assert saved == manifest
