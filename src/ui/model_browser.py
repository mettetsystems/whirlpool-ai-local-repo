"""Model Browser UI for displaying and managing local models."""

import json
from html import escape
import os
from datetime import datetime
from uuid import uuid4
from pathlib import Path
from typing import Dict, List, Optional, Any

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSettings
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QListWidget,
    QListWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QTextEdit,
    QPushButton,
    QLineEdit,
    QLabel,
    QFrame,
    QScrollArea,
    QGroupBox,
    QFormLayout,
    QDialog,
    QMessageBox,
    QProgressBar,
    QProgressDialog,
    QComboBox,
    QSpinBox,
    QFileDialog,
    QTabWidget,
    QAbstractItemView,
)
from PyQt5.QtGui import QFont, QColor, QPalette

from src.model_repo.hf_client import HuggingFaceClient


class TaskThread(QThread):
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, operation, parent=None):
        super().__init__(parent)
        self.operation = operation

    def run(self):
        try:
            self.succeeded.emit(self.operation())
        except Exception as exc:
            self.failed.emit(str(exc))


class ModelInfoThread(QThread):
    """Thread for fetching model information without blocking UI."""

    model_info_fetched = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, hf_client: HuggingFaceClient, model_id: str):
        super().__init__()
        self.hf_client = hf_client
        self.model_id = model_id

    def run(self):
        """Fetch model information from HuggingFace."""
        try:
            model_info = self.hf_client.api.model_info(
                self.model_id,
                token=self.hf_client.get_token()
            )
            info_dict = {
                "id": model_info.id,
                "author": model_info.author,
                "likes": model_info.likes,
                "downloads": model_info.downloads,
                "tags": list(model_info.tags) if model_info.tags else [],
                "pipeline_tag": model_info.pipeline_tag,
                "created_at": str(model_info.created_at) if model_info.created_at else None,
            }
            self.model_info_fetched.emit(info_dict)
        except Exception as e:
            self.error_occurred.emit(str(e))


from src.ui.model_card import ModelCardView


class ModelCardWidget(QWidget):
    """Widget displaying a single model card with expandable metadata."""

    def __init__(self, model_data: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.model_data = model_data
        self.is_expanded = False
        self._setup_ui()

    def _setup_ui(self):
        """Set up the model card UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Main card frame
        self.card_frame = QFrame()
        self.card_frame.setObjectName("ModelCard")
        card_layout = QVBoxLayout(self.card_frame)
        card_layout.setContentsMargins(10, 10, 10, 10)

        # Model ID header
        self.model_id_label = QLabel(self.model_data.get("model_id", "Unknown Model"))
        self.model_id_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.model_id_label.setStyleSheet("color: #2196F3;")
        card_layout.addWidget(self.model_id_label)

        # Local path
        local_path = self.model_data.get("local_path", "")
        self.path_label = QLabel(f"Path: {local_path}")
        self.path_label.setFont(QFont("Arial", 10))
        self.path_label.setStyleSheet("color: #666666;")
        card_layout.addWidget(self.path_label)

        # Expandable metadata section
        self.metadata_section = QGroupBox("Model Details")
        self.metadata_section.setCheckable(False)
        metadata_layout = QVBoxLayout(self.metadata_section)
        metadata_layout.setContentsMargins(5, 5, 5, 5)

        # Model info display
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(0)  # Hidden by default
        self.info_text.setFont(QFont("Arial", 9))
        metadata_layout.addWidget(self.info_text)

        # Expand/collapse button
        self.expand_button = QPushButton("Show Details")
        self.expand_button.clicked.connect(self._toggle_expand)
        card_layout.addWidget(self.expand_button)

        card_layout.addWidget(self.metadata_section)
        layout.addWidget(self.card_frame)
        self.metadata_section.hide()

        # Populate initial info
        self._update_info_display()

    def _update_info_display(self):
        """Update the info display with model metadata."""
        model_info = self.model_data.get("model_info", {})

        info_text = f"""
        <b>Author:</b> {escape(str(model_info.get('author', 'N/A')))}<br>
        <b>Likes:</b> {escape(str(model_info.get('likes', 0)))}<br>
        <b>Downloads:</b> {escape(str(model_info.get('downloads', 0)))}<br>
        <b>Pipeline:</b> {escape(str(model_info.get('pipeline_tag', 'N/A')))}<br>
        """

        tags = model_info.get('tags', [])
        if tags:
            info_text += f"<b>Tags:</b> {escape(', '.join(str(tag) for tag in tags[:5]))}"
            if len(tags) > 5:
                info_text += f" (+{len(tags) - 5} more)"

        self.info_text.setHtml(info_text)

    def _toggle_expand(self):
        """Toggle the expandable metadata section."""
        self.is_expanded = not self.is_expanded

        if self.is_expanded:
            self.info_text.setMaximumHeight(500)
            self.expand_button.setText("Hide Details")
            self.metadata_section.setVisible(True)
        else:
            self.info_text.setMaximumHeight(0)
            self.expand_button.setText("Show Details")
            self.metadata_section.setVisible(False)


class ModelBrowserWindow(QMainWindow):
    """Main window for browsing and managing local models."""

    models_updated = pyqtSignal()

    def __init__(self, hf_client: HuggingFaceClient):
        super().__init__()
        self.hf_client = hf_client
        self.current_model_info_thread: Optional[ModelInfoThread] = None
        self.active_task = None
        self._setup_ui()
        self._load_settings()
        self._refresh_model_list()

    def _setup_ui(self):
        """Set up the main UI."""
        self.setWindowTitle("Whirlpool Model Browser")
        self.setMinimumSize(1000, 700)

        # Central widget
        central_widget = QWidget()
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self.tabs.addTab(central_widget, "Models")

        main_layout = QVBoxLayout(central_widget)

        # Toolbar
        toolbar = self._create_toolbar()
        main_layout.addWidget(toolbar)

        # Splitter for model list and details
        splitter = QSplitter(Qt.Horizontal)

        # Left panel - Model list
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.model_list = QListWidget()
        self.model_list.itemClicked.connect(self._on_model_selected)
        self.model_list.setStyleSheet("""
            QListWidget {
                background-color: #FFFFFF;
                border: 1px solid #DDDDDD;
                border-radius: 4px;
                padding: 5px;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #EEEEEE;
            }
            QListWidget::item:selected {
                background-color: #E3F2FD;
                color: #1976D2;
            }
        """)
        left_layout.addWidget(self.model_list)

        splitter.addWidget(left_panel)

        # Right panel - Model details
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.details_scroll = QScrollArea()
        self.details_scroll.setWidgetResizable(True)
        self.details_scroll.setStyleSheet("""
            QScrollArea {
                background-color: #FAFAFA;
                border: 1px solid #DDDDDD;
                border-radius: 4px;
            }
        """)

        self.details_content = QWidget()
        self.details_layout = QVBoxLayout(self.details_content)
        self.details_layout.setContentsMargins(15, 15, 15, 15)

        self.details_label = QLabel("Select a model to view details")
        self.details_label.setFont(QFont("Arial", 12))
        self.details_label.setStyleSheet("color: #999999;")
        self.details_label.setAlignment(Qt.AlignCenter)
        self.details_layout.addWidget(self.details_label)

        self.details_scroll.setWidget(self.details_content)
        right_layout.addWidget(self.details_scroll)

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        main_layout.addWidget(splitter)
        from src.ui.settings_tab import SettingsTab
        self.settings_tab = SettingsTab(self.hf_client, lambda: self.active_task is not None, self)
        self.settings_tab.saved.connect(self._refresh_model_list)
        self.tabs.addTab(self.settings_tab, "Settings")
        from src.ui.downloads_tab import DownloadsTab
        self.downloads_tab = DownloadsTab(self.hf_client, self)
        self.downloads_tab.resume_requested.connect(self._start_downloads)
        self.settings_tab.saved.connect(self.downloads_tab.reload)
        self.tabs.addTab(self.downloads_tab, "Downloads")

    def _create_toolbar(self) -> QWidget:
        """Create the toolbar with action buttons."""
        toolbar = QWidget()
        layout = QHBoxLayout(toolbar)
        layout.setContentsMargins(5, 5, 5, 5)

        # Refresh button
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self._refresh_model_list)
        layout.addWidget(self.refresh_button)

        # Download button
        self.download_button = QPushButton("Download Model")
        self.download_button.clicked.connect(self._show_download_dialog)
        layout.addWidget(self.download_button)
        self.batch_download_button = QPushButton("Batch Download")
        self.batch_download_button.clicked.connect(self._show_batch_download_dialog)
        layout.addWidget(self.batch_download_button)

        self.local_model_button = QPushButton("Add Local Model")
        self.local_model_button.clicked.connect(self._add_local_model)
        layout.addWidget(self.local_model_button)

        # Delete button
        self.delete_button = QPushButton("Delete Model")
        self.delete_button.clicked.connect(self._delete_selected_model)
        layout.addWidget(self.delete_button)

        self.inference_button = QPushButton("Build Inference Server")
        self.inference_button.clicked.connect(self._show_inference_dialog)
        layout.addWidget(self.inference_button)
        self.train_button = QPushButton("Quick Train")
        self.train_button.clicked.connect(self._show_training_dialog)
        layout.addWidget(self.train_button)
        layout.addStretch()
        return toolbar

    def _add_local_model(self):
        directory = QFileDialog.getExistingDirectory(self, "Select model source directory")
        if not directory:
            return
        try:
            model = self.hf_client.add_local_model(directory)
            self._refresh_model_list()
            for index in range(self.model_list.count()):
                item = self.model_list.item(index)
                if item.data(Qt.UserRole).get("model_id") == model["model_id"]:
                    self.model_list.setCurrentItem(item)
                    self._on_model_selected(item)
                    break
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Cannot Add Model", str(exc))

    def _load_settings(self):
        """Load window settings from QSettings."""
        settings = QSettings("Whirlpool", "ModelBrowser")
        geometry = settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)

        window_state = settings.value("windowState")
        if window_state:
            self.restoreState(window_state)

    def _save_settings(self):
        """Save window settings to QSettings."""
        settings = QSettings("Whirlpool", "ModelBrowser")
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("windowState", self.saveState())

    def closeEvent(self, event):
        """Handle window close event."""
        if self.active_task is not None and self.active_task.isRunning():
            QMessageBox.information(self, "Task running", "Wait for the current operation before closing.")
            event.ignore()
            return
        self._save_settings()
        event.accept()

    def _refresh_model_list(self):
        """Refresh the model list from the manifest."""
        self.model_list.clear()

        models = self.hf_client.list_models()

        if not models:
            self.model_list.addItem("No models found. Click 'Download Model' to add one.")
            return

        for model in models:
            model_id = model.get("model_id", "Unknown")
            local_path = model.get("local_path", "")
            item_text = f"{model_id}\n{local_path}"
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, model)
            self.model_list.addItem(item)

    def _on_model_selected(self, item: QListWidgetItem):
        """Handle model selection in the list."""
        model_data = item.data(Qt.UserRole)
        if not model_data:
            return

        # Clear previous details
        while self.details_layout.count():
            child = self.details_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Create model card widget
        self.model_card = ModelCardWidget(model_data, self.details_content)
        self.details_layout.addWidget(self.model_card)

        self.model_card._toggle_expand()
        card_text = self.hf_client.get_model_card(model_data["model_id"])
        if card_text:
            published_card = ModelCardView(card_text)
            self.details_layout.addWidget(published_card)

    def _on_model_info_fetched(self, info: Dict[str, Any]):
        """Handle fetched model information."""
        if hasattr(self, 'model_card'):
            self.model_card.model_data["model_info"] = info
            self.model_card._update_info_display()

    def _on_model_info_error(self, error: str):
        """Handle model info fetch error."""
        print(f"Error fetching model info: {error}")

    def _show_download_dialog(self):
        """Show dialog to download a new model."""
        dialog = DownloadModelDialog(self.hf_client, self)
        if dialog.exec_() == QDialog.Accepted:
            model_id = dialog.get_model_id()
            if model_id:
                self._download_model(model_id)

    def _show_batch_download_dialog(self):
        dialog = DownloadModelDialog(self.hf_client, self, batch=True)
        if dialog.exec_() == QDialog.Accepted:
            self._start_downloads(dialog.get_model_ids())

    def _download_model(self, model_id: str):
        self._start_downloads([model_id])

    def _start_downloads(self, model_ids):
        if self.active_task is not None:
            QMessageBox.information(self, "Task running", "Wait for the current operation to finish.")
            return
        model_ids = list(dict.fromkeys(model_ids))
        if not model_ids:
            return
        if len(model_ids) > 10:
            QMessageBox.warning(self, "Batch too large", "Select at most 10 models per batch.")
            return
        from src.model_repo.download_state import DownloadState
        from src.ui.downloads_tab import DownloadThread
        state = DownloadState(self.hf_client._model_storage_path)
        try:
            for model_id in model_ids:
                if not state.get(model_id):
                    state.save({'model_id':model_id, 'status':'queued'})
        except OSError as exc:
            QMessageBox.critical(self, "Cannot save queue", str(exc))
            return
        worker = DownloadThread(self.hf_client, model_ids, self)
        self.active_task = worker
        self.downloads_tab.begin(worker)
        worker.model_ready.connect(lambda metadata: self._refresh_model_list())
        def finished():
            self.active_task = None
            worker.deleteLater()
        worker.finished.connect(finished)
        self.tabs.setCurrentWidget(self.downloads_tab)
        worker.start()

    def _run_task(self, title, operation, on_success=None):
        if self.active_task is not None:
            QMessageBox.information(self, "Task running", "Wait for the current operation to finish.")
            return
        progress = QProgressDialog(title + "…", "", 0, 0, self)
        progress.setCancelButton(None)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        task = TaskThread(operation, self)
        self.active_task = task
        def succeeded(result):
            progress.close()
            if on_success:
                on_success(result)
            QMessageBox.information(self, title, "Completed successfully.")
        def failed(error):
            progress.close()
            QMessageBox.critical(self, title, error)
        def finished():
            self.active_task = None
            task.deleteLater()
        task.succeeded.connect(succeeded)
        task.failed.connect(failed)
        task.finished.connect(finished)
        task.start()

    def _selected_model(self):
        item = self.model_list.currentItem()
        model = item.data(Qt.UserRole) if item else None
        if not model:
            QMessageBox.warning(self, "Select a model", "Select a downloaded model first.")
        return model

    def _show_inference_dialog(self):
        model = self._selected_model()
        if not model:
            return
        from src.inference.server_builder import InferenceServerBuilder, InferenceConfig
        dialog = QDialog(self)
        dialog.setWindowTitle("Build Inference Server")
        form = QFormLayout(dialog)
        form.addRow(QLabel("Local Transformers runtime • offline model loading"))
        device = QComboBox()
        device.addItems(["cpu", "cuda", "auto"])
        recommended = InferenceServerBuilder().get_model_recommendations(model)
        device.setCurrentText(recommended.device)
        port = QSpinBox()
        port.setRange(1024, 65535)
        port.setValue(8000)
        form.addRow("Device", device)
        form.addRow("Local port", port)
        build = QPushButton("Build and start")
        build.clicked.connect(dialog.accept)
        form.addRow(build)
        if dialog.exec_() != QDialog.Accepted:
            return
        config = InferenceConfig(model_id=model["model_id"], model_path=model["local_path"],
                                 host_port=port.value(), device=device.currentText(),
                                 gpu_count=0 if device.currentText() == "cpu" else 1,
                                 vllm_enabled=False,
                                 inference_image=self.hf_client.settings["inference_image"])
        def deploy():
            builder = InferenceServerBuilder(str(Path(self.hf_client.settings["inference_output_path"]) / "generated-inference.json"))
            result = builder.build_container(config)
            if not result.get("success"):
                raise RuntimeError(result.get("error") or result.get("stderr"))
            builder.wait_until_healthy(config)
            builder.smoke_test(config)
            if self.hf_client.settings["export_inference_image"]:
                builder.export_image(config, self.hf_client.settings["image_export_path"])
            return result
        self._run_task("Build inference server", deploy)

    def _show_training_dialog(self):
        model = self._selected_model()
        if not model:
            return
        from src.training.qlora_pipeline import TrainingConfig, QLoRAPipeline
        from src.data_parsers.parser_factory import DataSourceConfig
        dialog = QDialog(self)
        dialog.setWindowTitle("Quick Train")
        form = QFormLayout(dialog)
        form.addRow(QLabel("QLoRA requires NVIDIA CUDA and the optional training dependencies."))
        source_type = QComboBox()
        source_type.addItems(["local", "s3", "oci"])
        source = QLineEdit()
        source.setPlaceholderText("Local folder, or cloud object prefix")
        browse_source = QPushButton("Browse…")
        def choose_source():
            directory = QFileDialog.getExistingDirectory(self, "Select training data directory", source.text())
            if directory:
                source.setText(directory)
        browse_source.clicked.connect(choose_source)
        source_type.currentTextChanged.connect(lambda kind: browse_source.setEnabled(kind == "local"))
        source_row = QWidget()
        source_layout = QHBoxLayout(source_row)
        source_layout.setContentsMargins(0, 0, 0, 0)
        source_layout.addWidget(source)
        source_layout.addWidget(browse_source)
        bucket = QLineEdit()
        namespace = QLineEdit()
        run_name = datetime.now().strftime("run-%Y%m%d-%H%M%S-") + uuid4().hex[:6]
        output = QLineEdit(str(Path(self.hf_client.settings["training_output_path"]) / run_name))
        epochs = QSpinBox()
        epochs.setRange(1, 100)
        epochs.setValue(1)
        for label, widget in [("Source type", source_type), ("Folder / prefix", source_row),
                              ("Cloud bucket", bucket), ("OCI namespace", namespace),
                              ("New output directory", output), ("Epochs", epochs)]:
            form.addRow(label, widget)
        start = QPushButton("Start training")
        start.clicked.connect(dialog.accept)
        form.addRow(start)
        if dialog.exec_() != QDialog.Accepted:
            return
        kind = source_type.currentText()
        data = DataSourceConfig(kind, source.text().strip(), s3_bucket=bucket.text().strip(),
                                s3_prefix=source.text().strip(), oci_bucket=bucket.text().strip(),
                                oci_prefix=source.text().strip(), oci_namespace=namespace.text().strip())
        config = TrainingConfig(base_model_id=model["local_path"], data_source_config=data,
                                output_dir=output.text().strip(), num_train_epochs=epochs.value())
        def train():
            pipeline = QLoRAPipeline(config)
            try:
                result = pipeline.train()
                pipeline.save_training_state()
                return result
            finally:
                pipeline.cleanup()
        self._run_task("Quick Train", train)

    def _delete_selected_model(self):
        """Delete the currently selected model."""
        current_item = self.model_list.currentItem()
        if not current_item:
            QMessageBox.warning(
                self,
                "No Selection",
                "Please select a model to delete."
            )
            return

        model_data = current_item.data(Qt.UserRole)
        if not model_data:
            return

        model_id = model_data.get("model_id", "")
        external = not model_data.get("managed_files", True)

        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            (f"Remove '{model_id}' from the list? Its source files will be kept." if external else
             f"Are you sure you want to delete model '{model_id}'?\nThis action cannot be undone."),
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            try:
                success = self.hf_client.delete_model(model_id)
                if success:
                    self._refresh_model_list()
                    QMessageBox.information(
                        self,
                        "Delete Complete",
                        (f"Model '{model_id}' removed from the list. Source files were kept." if external else
                         f"Model '{model_id}' deleted successfully.")
                    )
                else:
                    QMessageBox.warning(
                        self,
                        "Delete Failed",
                        f"Could not find model '{model_id}'."
                    )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Delete Failed",
                    f"Failed to delete model: {str(e)}"
                )


class DownloadModelDialog(QDialog):
    """Search and explicitly choose one model, or check models for a batch."""

    def __init__(self, hf_client, parent=None, batch=False):
        super().__init__(parent)
        self.hf_client = hf_client
        self.batch = batch
        self.checked = {}
        self.search_thread = None
        self.searched = False
        self.setWindowTitle("Batch Download" if batch else "Download Model")
        self.setMinimumSize(800, 400)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Search Hugging Face or paste an exact model ID"))
        self.model_id_input = QLineEdit()
        self.model_id_input.setPlaceholderText("organization/model or search terms")
        layout.addWidget(self.model_id_input)
        self.search_button = QPushButton("Search")
        self.search_button.clicked.connect(self._search_models)
        layout.addWidget(self.search_button)
        self.search_results = QTreeWidget()
        self.search_results.setRootIsDecorated(False)
        self.search_results.setHeaderLabels(['Model', 'Downloads', 'Projected download size'])
        self.search_results.setColumnWidth(0, 380)
        self.search_results.setColumnWidth(2, 180)
        self.search_results.headerItem().setToolTip(2, 'Full repository file size, including all weight variants. Cached files may reduce the actual transfer. Unknown means metadata is unavailable.')
        self.search_results.itemClicked.connect(self._select_search_result)
        self.search_results.itemDoubleClicked.connect(self._select_search_result)
        self.search_results.itemChanged.connect(self._checked_changed)
        self.search_results.itemSelectionChanged.connect(self._selection_changed)
        layout.addWidget(self.search_results)
        self.selection_label = QLabel("Check up to 10 models; choices are kept across searches." if batch else "Select a result to download it.")
        self.selection_label.setWordWrap(True)
        layout.addWidget(self.selection_label)
        if batch:
            add = QPushButton("Add entered model ID to batch")
            add.clicked.connect(self._add_entered)
            layout.addWidget(add)
            clear = QPushButton("Clear batch selection")
            clear.clicked.connect(self._clear_checked)
            layout.addWidget(clear)
        row = QHBoxLayout()
        self.download_button = QPushButton("Download Checked Models" if batch else "Download Selected Model")
        self.download_button.setEnabled(False)
        self.download_button.clicked.connect(self.accept)
        self.model_id_input.textChanged.connect(self._input_changed)
        row.addWidget(self.download_button)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        row.addWidget(cancel)
        layout.addLayout(row)

    def _input_changed(self, text):
        if not self.batch:
            self.download_button.setEnabled(bool(text.strip()) and (not self.searched or '/' in text))

    def get_model_id(self):
        return self.model_id_input.text().strip()

    def get_model_ids(self):
        return list(self.checked) if self.batch else [self.get_model_id()]

    def _search_models(self):
        query = self.model_id_input.text().strip()
        if not query or self.search_thread is not None:
            return
        self.searched = True
        self.search_results.clear()
        if not self.batch:
            self.download_button.setEnabled(False)
        self.search_button.setEnabled(False)
        self.selection_label.setText("Searching and checking download sizes…")
        task = TaskThread(lambda: self.hf_client.search_models(query, limit=30, include_download_size=True), self)
        self.search_thread = task
        task.succeeded.connect(self._search_ready)
        task.failed.connect(lambda error: self.selection_label.setText("Search failed: " + error))
        def finished():
            self.search_thread = None
            self.search_button.setEnabled(True)
            task.deleteLater()
        task.finished.connect(finished)
        task.start()

    def _search_ready(self, results):
        self.search_results.blockSignals(True)
        for result in results:
            from src.ui.downloads_tab import size_text
            size = result.get('download_size_bytes')
            size_label = size_text(size) if type(size) is int and size >= 0 else 'Unknown'
            item = QTreeWidgetItem([result['id'], str(result.get('downloads') or 0), size_label])
            item.setToolTip(2, f'{size:,} bytes across all repository files' if size_label != 'Unknown' else 'Size metadata unavailable; download remains selectable.')
            item.setData(0, Qt.UserRole, result)
            if self.batch:
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(0, Qt.Checked if result['id'] in self.checked else Qt.Unchecked)
            self.search_results.addTopLevelItem(item)
        self.search_results.blockSignals(False)
        if self.batch:
            self._update_checked()
        else:
            self.selection_label.setText("Select one result to download." if results else "No models found.")

    def _selection_changed(self):
        if not self.batch and self.search_results.currentItem():
            self._select_search_result(self.search_results.currentItem())

    def _select_search_result(self, item, column=0):
        if self.batch:
            return
        result = item.data(0, Qt.UserRole)
        if result:
            self.model_id_input.setText(result['id'])
            self.download_button.setEnabled(True)

    def _checked_changed(self, item, column=0):
        if not self.batch:
            return
        model_id = item.data(0, Qt.UserRole)['id']
        if item.checkState(0) == Qt.Checked:
            self.checked[model_id] = True
        else:
            self.checked.pop(model_id, None)
        self._update_checked()

    def _update_checked(self):
        self.selection_label.setText(f"{len(self.checked)}/10 selected: " + ', '.join(self.checked))
        self.download_button.setEnabled(0 < len(self.checked) <= 10)

    def _add_entered(self):
        from huggingface_hub.utils import validate_repo_id
        try:
            model_id = self.get_model_id()
            validate_repo_id(model_id)
        except ValueError as exc:
            self.selection_label.setText(str(exc))
            return
        self.checked[model_id] = True
        for index in range(self.search_results.topLevelItemCount()):
            item = self.search_results.topLevelItem(index)
            if item.data(0, Qt.UserRole)['id'] == model_id:
                item.setCheckState(0, Qt.Checked)
        self._update_checked()

    def _clear_checked(self):
        self.checked.clear()
        for index in range(self.search_results.topLevelItemCount()):
            self.search_results.topLevelItem(index).setCheckState(0, Qt.Unchecked)
        self._update_checked()

    def done(self, result):
        if self.search_thread is not None:
            self.selection_label.setText("Wait for the search to finish before closing.")
            return
        super().done(result)


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)

    # Initialize HF client
    hf_client = HuggingFaceClient()

    # Create and show the model browser
    window = ModelBrowserWindow(hf_client)
    window.show()

    sys.exit(app.exec_())
