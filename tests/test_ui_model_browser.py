"""Tests for the Model Browser UI."""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from PyQt5.QtWidgets import QApplication, QListWidgetItem
from PyQt5.QtCore import Qt

from src.ui.model_browser import (
    ModelBrowserWindow,
    ModelCardWidget,
    ModelInfoThread,
    DownloadModelDialog,
)
from src.model_repo.hf_client import HuggingFaceClient


@pytest.fixture
def app():
    """Create a QApplication instance for testing."""
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture
def temp_config(tmp_path):
    """Create a temporary HF auth config file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        config = {
            "hf_token": "test_token_123",
            "model_storage_path": str(tmp_path / "models")
        }
        json.dump(config, f)
        config_path = f.name

    yield config_path

    # Cleanup
    if os.path.exists(config_path):
        os.unlink(config_path)


@pytest.fixture
def hf_client(temp_config):
    """Create a HuggingFaceClient instance."""
    return HuggingFaceClient(auth_config_path=temp_config)


@pytest.fixture
def model_browser_window(hf_client, app):
    """Create a ModelBrowserWindow instance."""
    with patch('src.ui.model_browser.HuggingFaceClient'):
        window = ModelBrowserWindow(hf_client)
        yield window
        window.close()


class TestModelCardWidget:
    """Tests for ModelCardWidget."""

    def test_initialization(self, app):
        """Test that ModelCardWidget initializes correctly."""
        model_data = {
            "model_id": "test-model",
            "local_path": "/tmp/test",
            "model_info": {
                "author": "test_author",
                "likes": 100,
                "downloads": 500,
                "pipeline_tag": "text-generation"
            }
        }

        widget = ModelCardWidget(model_data)

        assert widget.model_data == model_data
        assert widget.is_expanded == False
        assert widget.model_id_label.text() == "test-model"

    def test_expand_toggle(self, app):
        """Test expanding and collapsing the model card."""
        model_data = {
            "model_id": "test-model",
            "local_path": "/tmp/test",
            "model_info": {}
        }

        widget = ModelCardWidget(model_data)

        # Initially collapsed
        assert widget.is_expanded == False
        assert widget.info_text.maximumHeight() == 0

        # Expand
        widget._toggle_expand()
        assert widget.is_expanded == True
        assert widget.info_text.maximumHeight() == 500
        assert widget.expand_button.text() == "Hide Details"

        # Collapse
        widget._toggle_expand()
        assert widget.is_expanded == False
        assert widget.info_text.maximumHeight() == 0
        assert widget.expand_button.text() == "Show Details"


class TestModelInfoThread:
    """Tests for ModelInfoThread."""

    def test_model_info_fetched(self, hf_client):
        """Test that model info is correctly fetched and emitted."""
        mock_model_info = MagicMock()
        mock_model_info.id = "test-model"
        mock_model_info.author = "test_author"
        mock_model_info.likes = 100
        mock_model_info.downloads = 500
        mock_model_info.tags = ["text-generation", "transformers"]
        mock_model_info.pipeline_tag = "text-generation"
        mock_model_info.created_at = "2024-01-01"

        with patch.object(hf_client.api, 'model_info', return_value=mock_model_info):
            thread = ModelInfoThread(hf_client, "test-model")

            info_received = {}
            def on_info_fetched(info):
                info_received.update(info)

            thread.model_info_fetched.connect(on_info_fetched)
            thread.run()

            assert "test-model" in info_received.get("id", "")
            assert info_received.get("author") == "test_author"
            assert info_received.get("likes") == 100

    def test_error_handling(self, hf_client):
        """Test that errors are correctly emitted."""
        with patch.object(hf_client.api, 'model_info', side_effect=Exception("Test error")):
            thread = ModelInfoThread(hf_client, "test-model")

            error_received = ""
            def on_error(error):
                nonlocal error_received
                error_received = error

            thread.error_occurred.connect(on_error)
            thread.run()

            assert "Test error" in error_received


class TestModelBrowserWindow:
    """Tests for ModelBrowserWindow."""

    def test_initialization(self, model_browser_window):
        """Test that ModelBrowserWindow initializes correctly."""
        assert model_browser_window.windowTitle() == "Whirlpool Model Browser"
        assert model_browser_window.model_list is not None
        assert model_browser_window.refresh_button is not None

    def test_refresh_model_list_empty(self, model_browser_window, hf_client):
        """Test refreshing model list when empty."""
        with patch.object(hf_client, 'list_models', return_value=[]):
            model_browser_window._refresh_model_list()

            assert model_browser_window.model_list.count() == 1
            assert "No models found" in model_browser_window.model_list.item(0).text()

    def test_refresh_model_list_with_models(self, model_browser_window, hf_client):
        """Test refreshing model list with models."""
        models = [
            {
                "model_id": "test-model-1",
                "local_path": "/tmp/model1",
                "model_info": {}
            },
            {
                "model_id": "test-model-2",
                "local_path": "/tmp/model2",
                "model_info": {}
            }
        ]

        with patch.object(hf_client, 'list_models', return_value=models):
            model_browser_window._refresh_model_list()

            assert model_browser_window.model_list.count() == 2
            assert "test-model-1" in model_browser_window.model_list.item(0).text()
            assert "test-model-2" in model_browser_window.model_list.item(1).text()

    def test_on_model_selected(self, model_browser_window, hf_client):
        """Test selecting a model from the list."""
        model_data = {
            "model_id": "test-model",
            "local_path": "/tmp/test",
            "model_info": {}
        }

        # Add item to list - create QListWidgetItem explicitly
        item = QListWidgetItem("test-model")
        item.setData(Qt.UserRole, model_data)
        model_browser_window.model_list.addItem(item)

        # Select the item
        model_browser_window._on_model_selected(item)

        # Check that model card was created
        assert model_browser_window.model_card is not None
        assert model_browser_window.model_card.model_data["model_id"] == "test-model"


class TestDownloadModelDialog:
    """Tests for DownloadModelDialog."""

    def test_initialization(self, hf_client, app):
        """Test that DownloadModelDialog initializes correctly."""
        dialog = DownloadModelDialog(hf_client)

        assert dialog.windowTitle() == "Download Model"
        assert dialog.model_id_input is not None
        assert dialog.search_results is not None
        assert dialog.download_button.isEnabled() == False

    def test_get_model_id(self, hf_client, app):
        """Test getting model ID from dialog."""
        dialog = DownloadModelDialog(hf_client)
        dialog.model_id_input.setText("test-model-id")

        assert dialog.get_model_id() == "test-model-id"

    def test_search_models(self, hf_client, app):
        """Test searching for models."""
        dialog = DownloadModelDialog(hf_client)

        mock_results = [
            {
                "id": "test-model-1",
                "author": "test_author",
                "likes": 100
            },
            {
                "id": "test-model-2",
                "author": "test_author2",
                "likes": 50
            }
        ]

        with patch.object(hf_client, 'search_models', return_value=mock_results):
            dialog.model_id_input.setText("test")
            dialog._search_models()
            from PyQt5.QtCore import QEventLoop, QTimer
            loop = QEventLoop()
            dialog.search_thread.finished.connect(loop.quit)
            QTimer.singleShot(3000, loop.quit)
            loop.exec_()
            app.processEvents()
            assert dialog.search_thread is None

            assert dialog.search_results.count() == 2
            assert "test-model-1" in dialog.search_results.item(0).text()

    def test_select_search_result(self, hf_client, app):
        """Test selecting a search result."""
        dialog = DownloadModelDialog(hf_client)

        mock_result = {
            "id": "selected-model",
            "author": "test_author",
            "likes": 100
        }

        # Create QListWidgetItem explicitly
        item = QListWidgetItem("selected-model")
        item.setData(Qt.UserRole, mock_result)
        dialog.search_results.addItem(item)

        dialog._select_search_result(item)

        assert dialog.model_id_input.text() == "selected-model"
        assert dialog.download_button.isEnabled() == True


class TestModelBrowserIntegration:
    """Integration tests for ModelBrowserWindow."""

    def test_auto_refresh_after_download(self, model_browser_window, hf_client, app):
        """Test that model list auto-refreshes after download."""
        # Mock download_model to return metadata
        mock_metadata = {
            "model_id": "downloaded-model",
            "local_path": "/tmp/downloaded",
            "downloaded_at": "2024-01-01",
            "model_info": {}
        }

        # Mock list_models to return the downloaded model after download
        with patch.object(hf_client, 'download_model', return_value=mock_metadata):
            with patch.object(hf_client, 'list_models', return_value=[mock_metadata]):
                # Simulate download
                from PyQt5.QtCore import QEventLoop, QTimer
                with patch("src.ui.model_browser.QMessageBox.information"):
                    model_browser_window._download_model("downloaded-model")
                    loop = QEventLoop()
                    model_browser_window.active_task.finished.connect(loop.quit)
                    QTimer.singleShot(3000, loop.quit)
                    loop.exec_()
                    app.processEvents()
                assert model_browser_window.active_task is None

                # Check that list was refreshed
                assert model_browser_window.model_list.count() == 1
                assert "downloaded-model" in model_browser_window.model_list.item(0).text()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_pasted_model_id_enables_download(hf_client, app):
    dialog = DownloadModelDialog(hf_client)
    dialog.model_id_input.setText("organization/model")
    assert dialog.download_button.isEnabled()
    dialog.model_id_input.setText("   ")
    assert not dialog.download_button.isEnabled()
    dialog.close()


def test_model_selection_does_not_fetch_remote_metadata(model_browser_window, hf_client):
    model = {"model_id": "test/offline", "local_path": "/tmp/offline", "model_info": {}}
    item = QListWidgetItem("offline")
    item.setData(Qt.UserRole, model)
    with patch.object(hf_client.api, "model_info", side_effect=AssertionError("Unexpected network")):
        model_browser_window._on_model_selected(item)
        assert model_browser_window.current_model_info_thread is None
        assert model_browser_window.model_card.is_expanded


def test_background_failure_is_reported(model_browser_window, app):
    from PyQt5.QtCore import QEventLoop, QTimer
    def fail():
        raise RuntimeError("test failure")
    with patch("src.ui.model_browser.QMessageBox.critical") as error:
        model_browser_window._run_task("Test", fail)
        loop = QEventLoop()
        model_browser_window.active_task.finished.connect(loop.quit)
        QTimer.singleShot(3000, loop.quit)
        loop.exec_()
        app.processEvents()
        error.assert_called_once()
        assert "test failure" in error.call_args.args[-1]
    assert model_browser_window.active_task is None


def test_add_local_model_picker(model_browser_window, hf_client, tmp_path):
    source = tmp_path / "local-model"
    source.mkdir()
    (source / "config.json").write_text('{}')
    with patch('src.ui.model_browser.QFileDialog.getExistingDirectory', return_value=str(source)):
        model_browser_window.local_model_button.click()
    selected = model_browser_window.model_list.currentItem().data(Qt.UserRole)
    assert selected['local_path'] == str(source)
    assert selected['managed_files'] is False
    assert model_browser_window.model_card.is_expanded
    with patch('src.ui.model_browser.QFileDialog.getExistingDirectory', return_value=''):
        model_browser_window.local_model_button.click()
    assert len(hf_client.list_models()) == 1


def test_training_source_directory_picker(model_browser_window, tmp_path):
    from PyQt5.QtWidgets import QDialog, QPushButton, QLineEdit, QComboBox
    source = tmp_path / "training-data"
    source.mkdir()
    model = {'model_id': 'local/model', 'local_path': str(tmp_path)}
    def inspect_dialog(dialog):
        browse = next(button for button in dialog.findChildren(QPushButton) if button.text() == 'Browse…')
        with patch('src.ui.model_browser.QFileDialog.getExistingDirectory', return_value=str(source)):
            browse.click()
        assert any(edit.text() == str(source) for edit in dialog.findChildren(QLineEdit))
        dialog.findChild(QComboBox).setCurrentText('s3')
        assert not browse.isEnabled()
        return QDialog.Rejected
    with patch.object(model_browser_window, '_selected_model', return_value=model), patch.object(QDialog, 'exec_', inspect_dialog):
        model_browser_window._show_training_dialog()


def test_settings_tab_saves_defaults(model_browser_window, hf_client, tmp_path):
    from src.settings import PATH_KEYS
    tab = model_browser_window.settings_tab
    assert model_browser_window.tabs.tabText(1) == 'Settings'
    assert tab.token.echoMode() == tab.token.Password
    for key in PATH_KEYS:
        tab.fields[key].setText(str(tmp_path / key))
    tab.token.setText('test-secret')
    tab.image.setText('whirlpool/custom:v2')
    with patch('src.credentials.backend') as backend:
        backend.return_value.get_password.return_value = 'test-secret'
        tab.save_button.click()
    assert 'Settings saved' in tab.status.text()
    assert hf_client.settings['inference_image'] == 'whirlpool/custom:v2'
    assert hf_client.settings['image_export_path'] == str(tmp_path / 'image_export_path')
    assert 'test-secret' not in hf_client.auth_config_path.read_text()


def test_settings_cannot_change_during_task(model_browser_window):
    model_browser_window.active_task = MagicMock()
    with patch('src.ui.settings_tab.QMessageBox.information') as message, patch.object(model_browser_window.hf_client, 'update_settings') as update:
        model_browser_window.settings_tab.save_button.click()
        message.assert_called_once()
        update.assert_not_called()
    model_browser_window.active_task = None


def test_inference_uses_saved_paths_and_image(model_browser_window, tmp_path):
    from PyQt5.QtWidgets import QDialog
    model_browser_window.hf_client.settings.update(inference_output_path=str(tmp_path / 'deploy'),
        image_export_path=str(tmp_path / 'images'), inference_image='custom/model:v2', export_inference_image=True)
    def execute(title, operation):
        operation()
    with patch.object(model_browser_window, '_selected_model', return_value={'model_id':'test', 'local_path':str(tmp_path)}), patch.object(QDialog, 'exec_', return_value=QDialog.Accepted), patch.object(model_browser_window, '_run_task', side_effect=execute), patch('src.inference.server_builder.InferenceServerBuilder') as builder:
        builder.return_value.get_model_recommendations.return_value.device = 'cpu'
        builder.return_value.build_container.return_value = {'success':True}
        model_browser_window._show_inference_dialog()
        assert builder.call_args.args[0] == str(tmp_path / 'deploy' / 'generated-inference.json')
        config = builder.return_value.build_container.call_args.args[0]
        assert config.inference_image == 'custom/model:v2'
        builder.return_value.export_image.assert_called_once_with(config, str(tmp_path / 'images'))
