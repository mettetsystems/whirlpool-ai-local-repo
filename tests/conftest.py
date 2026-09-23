"""Pytest configuration and fixtures for HuggingFace mocking."""

import argparse
from unittest.mock import MagicMock, patch
import pytest


def pytest_addoption(parser):
    """Add --mock-hf command line option."""
    parser.addoption(
        "--mock-hf",
        action="store_true",
        default=False,
        help="Mock HuggingFace API calls for testing"
    )


@pytest.fixture(autouse=True)
def mock_hf_api(request):
    """Automatically mock HuggingFace API when --mock-hf is passed."""
    if request.config.getoption("--mock-hf"):
        with patch("src.model_repo.hf_client.HfApi") as mock_api_class:
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
            with patch("src.model_repo.hf_client.snapshot_download") as mock_snapshot:
                mock_snapshot.return_value = "/tmp/mock_download"
                yield mock_api, mock_snapshot
    else:
        yield None, None


@pytest.fixture
def mock_hf_snapshot(tmp_path):
    """Provide a mock for snapshot_download."""
    with patch("src.model_repo.hf_client.snapshot_download") as mock_snapshot:
        mock_snapshot.return_value = str(tmp_path / "downloaded")
        yield mock_snapshot


@pytest.fixture
def mock_hf_api_instance():
    """Provide a mock HfApi instance."""
    with patch("src.model_repo.hf_client.HfApi") as mock_api_class:
        mock_api = MagicMock()
        mock_api.model_info.return_value = MagicMock(
            id="test/model",
            author="test_author",
            likes=100,
            downloads=500,
            tags=[],
            pipeline_tag=None
        )
        mock_api_class.return_value = mock_api
        yield mock_api
