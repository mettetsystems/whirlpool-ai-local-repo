"""Tests for HuggingFace client."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.model_repo.hf_client import HuggingFaceClient


class TestHuggingFaceClient:
    """Test cases for HuggingFaceClient."""

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

    def test_init_creates_storage_dir(self, temp_config, tmp_path):
        """Test that initialization creates the storage directory."""
        client = HuggingFaceClient(auth_config_path=temp_config)
        storage_path = Path(temp_config).parent / "models"
        assert storage_path.exists()

    def test_init_creates_manifest(self, temp_config, tmp_path):
        """Test that initialization creates the manifest file."""
        client = HuggingFaceClient(auth_config_path=temp_config)
        manifest_path = Path(temp_config).parent / "models" / ".manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        assert "models" in manifest

    def test_set_token(self, client):
        """Test setting the HuggingFace token."""
        client.set_token("new_token")
        assert client.get_token() == "new_token"

    def test_get_token_returns_config_token(self, client):
        """Test that get_token returns the token from config."""
        assert client.get_token() == "test_token_123"

    def test_list_models_empty(self, client):
        """Test listing models when no models are stored."""
        models = client.list_models()
        assert models == []

    @patch("src.model_repo.hf_client.snapshot_download")
    @patch("src.model_repo.hf_client.HfApi")
    def test_download_model(self, mock_api_class, mock_snapshot, client, tmp_path):
        """Test downloading a model."""
        # Mock the API
        mock_api = MagicMock()
        mock_api_class.return_value = mock_api
        mock_api.model_info.return_value = MagicMock(
            id="test/model",
            author="test_author",
            likes=100,
            downloads=500,
            tags=["transformers", "pytorch"],
            pipeline_tag="text-generation"
        )

        # Mock snapshot_download
        mock_snapshot.return_value = str(tmp_path / "downloaded")

        # Download model
        metadata = client.download_model("test/model")

        assert metadata["model_id"] == "test/model"
        assert "local_path" in metadata
        assert "downloaded_at" in metadata
        assert "model_info" in metadata

    @patch("src.model_repo.hf_client.HuggingFaceClient._get_manifest")
    def test_get_model_info_found(self, mock_get_manifest, client):
        """Test getting model info when model exists."""
        mock_get_manifest.return_value = {
            "models": [{
                "model_id": "test/model",
                "local_path": "/path/to/model",
                "downloaded_at": "2024-01-01",
                "model_info": {}
            }]
        }

        info = client.get_model_info("test/model")
        assert info is not None
        assert info["model_id"] == "test/model"

    @patch("src.model_repo.hf_client.HuggingFaceClient._get_manifest")
    def test_get_model_info_not_found(self, mock_get_manifest, client):
        """Test getting model info when model doesn't exist."""
        mock_get_manifest.return_value = {"models": []}

        info = client.get_model_info("nonexistent/model")
        assert info is None

    def test_model_exists_true(self, client):
        """Test model_exists returns True when model exists."""
        with patch.object(client, "get_model_info", return_value={"model_id": "test/model"}):
            assert client.model_exists("test/model") is True

    def test_model_exists_false(self, client):
        """Test model_exists returns False when model doesn't exist."""
        with patch.object(client, "get_model_info", return_value=None):
            assert client.model_exists("nonexistent/model") is False

    @patch("src.model_repo.hf_client.HuggingFaceClient._get_manifest")
    @patch("src.model_repo.hf_client.shutil")
    @patch("src.model_repo.hf_client.Path")
    def test_delete_model(self, mock_path_class, mock_shutil, mock_get_manifest, client, tmp_path):
        """Test deleting a model."""
        expected_path_str = str(tmp_path / "test_model")

        # Create a mock Path instance
        mock_path_instance = MagicMock(spec=Path)
        mock_path_instance.exists.return_value = True
        mock_path_instance.__str__ = lambda self: expected_path_str

        # Setup mock_get_manifest to return string path
        mock_get_manifest.return_value = {
            "models": [{
                "model_id": "test/model",
                "local_path": expected_path_str,
                "downloaded_at": "2024-01-01",
                "model_info": {}
            }]
        }

        # Configure Path() to return our mock instance
        mock_path_class.return_value = mock_path_instance

        result = client.delete_model("test/model")

        assert result is True

        # Verify shutil.rmtree was called with a Path object
        mock_shutil.rmtree.assert_called_once()
        call_args = mock_shutil.rmtree.call_args[0][0]
        assert isinstance(call_args, Path)
        assert str(call_args) == expected_path_str

    @patch("src.model_repo.hf_client.HuggingFaceClient._get_manifest")
    def test_delete_model_not_found(self, mock_get_manifest, client):
        """Test deleting a model that doesn't exist."""
        mock_get_manifest.return_value = {"models": []}

        result = client.delete_model("nonexistent/model")
        assert result is False

    @patch("src.model_repo.hf_client.HfApi")
    def test_search_models(self, mock_api_class, client):
        """Test searching for models."""
        mock_api = MagicMock()
        mock_api_class.return_value = mock_api

        # Create a single model mock object
        mock_model = MagicMock()
        mock_model.id = "model1"
        mock_model.author = "author1"
        mock_model.likes = 100
        mock_model.downloads = 500
        mock_model.tags = ["transformers"]
        mock_model.pipeline_tag = "text-classification"

        # Return exactly one model result
        mock_api.list_models.return_value = [mock_model]

        # Patch the api attribute to return our mock
        client.api = mock_api

        results = client.search_models("test", limit=5)

        assert len(results) == 1
        assert results[0]["id"] == "model1"

    def test_search_models_no_token(self, client):
        """Test that search_models raises error when no token is set."""
        client.set_token("")
        with pytest.raises(ValueError, match="HuggingFace token not set"):
            client.search_models("test")

    def test_download_model_no_token(self, client):
        """Test that download_model raises error when no token is set."""
        client.set_token("")
        with pytest.raises(ValueError, match="HuggingFace token not set"):
            client.download_model("test/model")

    @patch("src.model_repo.hf_client.HuggingFaceClient._get_manifest")
    def test_verify_model_integrity_found(self, mock_get_manifest, client, tmp_path):
        """Test verify_model_integrity when model exists."""
        model_dir = tmp_path / "test_model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")

        mock_get_manifest.return_value = {
            "models": [{
                "model_id": "test/model",
                "local_path": str(model_dir),
                "downloaded_at": "2024-01-01",
                "model_info": {}
            }]
        }

        assert client.verify_model_integrity("test/model") is True

    @patch("src.model_repo.hf_client.HuggingFaceClient._get_manifest")
    def test_verify_model_integrity_not_found(self, mock_get_manifest, client):
        """Test verify_model_integrity when model doesn't exist."""
        mock_get_manifest.return_value = {"models": []}

        assert client.verify_model_integrity("nonexistent/model") is False

    @patch("src.model_repo.hf_client.HuggingFaceClient._get_manifest")
    def test_verify_model_integrity_no_files(self, mock_get_manifest, client, tmp_path):
        """Test verify_model_integrity when model directory is empty."""
        model_dir = tmp_path / "empty_model"
        model_dir.mkdir()

        mock_get_manifest.return_value = {
            "models": [{
                "model_id": "test/model",
                "local_path": str(model_dir),
                "downloaded_at": "2024-01-01",
                "model_info": {}
            }]
        }

        assert client.verify_model_integrity("test/model") is False
