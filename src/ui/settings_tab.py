"""Settings form shared with the model browser's existing client."""
import os
from PyQt5.QtWidgets import QWidget, QFormLayout, QLineEdit, QPushButton, QHBoxLayout, QFileDialog, QLabel, QCheckBox, QMessageBox
from PyQt5.QtCore import pyqtSignal
from src.settings import PATH_KEYS


class SettingsTab(QWidget):
    saved = pyqtSignal()

    def __init__(self, client, is_busy, parent=None):
        super().__init__(parent)
        self.client = client
        self.is_busy = is_busy
        self.fields = {}
        form = QFormLayout(self)
        self.token = QLineEdit(client.settings.get("hf_token", ""))
        self.token.setEchoMode(QLineEdit.Password)
        form.addRow("Hugging Face token", self.token)
        show = QCheckBox("Show token")
        show.toggled.connect(lambda checked: self.token.setEchoMode(QLineEdit.Normal if checked else QLineEdit.Password))
        form.addRow(show)
        if os.environ.get("HF_TOKEN"):
            form.addRow(QLabel("HF_TOKEN is set in the environment and overrides the saved token."))
        labels = {
            "model_storage_path": "Model downloads",
            "training_output_path": "Training runs and adapters",
            "inference_output_path": "Inference deployment files",
            "image_export_path": "Exported inference images",
        }
        for key in PATH_KEYS:
            edit = QLineEdit(client.settings[key])
            self.fields[key] = edit
            row = QWidget()
            layout = QHBoxLayout(row)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(edit)
            browse = QPushButton("Browse…")
            browse.clicked.connect(lambda checked=False, field=edit: self.browse(field))
            layout.addWidget(browse)
            form.addRow(labels[key], row)
        self.image = QLineEdit(client.settings["inference_image"])
        form.addRow("Inference image name / tag", self.image)
        self.export = QCheckBox("Export a .tar image archive after a successful inference build")
        self.export.setChecked(client.settings["export_inference_image"])
        form.addRow(self.export)
        note = QLabel("Built images live in Docker/Podman. Exported archives go to the selected folder; saving settings does not publish to a registry.\n"
                      "Tokens are stored in Fedora’s desktop keyring (GNOME Keyring). Changing folders leaves existing files in place.")
        note.setWordWrap(True)
        form.addRow(note)
        self.save_button = QPushButton("Save Settings")
        self.save_button.clicked.connect(self.save)
        form.addRow(self.save_button)
        self.status = QLabel(client.settings.get("credential_error", ""))
        form.addRow(self.status)

    def browse(self, field):
        directory = QFileDialog.getExistingDirectory(self, "Select default directory", field.text())
        if directory:
            field.setText(directory)

    def save(self):
        if self.is_busy():
            QMessageBox.information(self, "Task running", "Wait for the current operation before changing settings.")
            return
        values = {key: edit.text().strip() for key, edit in self.fields.items()}
        values.update(hf_token=self.token.text(), inference_image=self.image.text(), export_inference_image=self.export.isChecked())
        try:
            self.client.update_settings(values)
        except (OSError, ValueError, RuntimeError) as exc:
            QMessageBox.critical(self, "Settings not saved", str(exc))
            return
        self.status.setText("Settings saved. Defaults apply to new operations.")
        self.saved.emit()
