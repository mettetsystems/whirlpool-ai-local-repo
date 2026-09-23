"""Queued model transfers, with progress and explicit pause/resume controls."""
import threading
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTreeWidget, QTreeWidgetItem, QProgressBar, QAbstractItemView
from src.model_repo.download_state import DownloadState
from src.model_repo.download_process import DownloadPaused


def size_text(value):
    value = float(value)
    for unit in ('B', 'KiB', 'MiB', 'GiB', 'TiB'):
        if value < 1024 or unit == 'TiB':
            return f'{value:.1f} {unit}'
        value /= 1024


class DownloadThread(QThread):
    updated = pyqtSignal(str, dict)
    model_ready = pyqtSignal(object)

    def __init__(self, client, model_ids, parent=None):
        super().__init__(parent)
        self.client = client
        self.model_ids = list(dict.fromkeys(model_ids))
        self.stop_event = threading.Event()

    def run(self):
        state = DownloadState(self.client._model_storage_path)
        for model_id in self.model_ids:
            if self.stop_event.is_set():
                break
            self.updated.emit(model_id, {'status':'downloading'})
            try:
                metadata = self.client.download_model(model_id, progress_callback=lambda event, mid=model_id: self.updated.emit(mid, event), stop_event=self.stop_event)
                record = state.get(model_id) or {}
                self.updated.emit(model_id, {**record, 'status':'completed'})
                self.model_ready.emit(metadata)
            except Exception as exc:
                status = 'paused' if isinstance(exc, DownloadPaused) else 'failed'
                error = str(exc)
                if self.client.get_token():
                    error = error.replace(self.client.get_token(), '[redacted]')
                record = state.get(model_id) or {'model_id':model_id}
                record.update(status=status, error=error)
                state.save(record)
                self.updated.emit(model_id, record)
                if status == 'paused':
                    break


class DownloadsTab(QWidget):
    resume_requested = pyqtSignal(list)

    def __init__(self, client, parent=None):
        super().__init__(parent)
        self.client = client
        self.worker = None
        self.rows = {}
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel('Downloads are resumable. Select interrupted models and click Resume Selected.'))
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(['Model', 'Status', 'Progress', 'Downloaded', 'Speed', 'ETA', 'File / error'])
        self.tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        layout.addWidget(self.tree)
        buttons = QHBoxLayout()
        self.pause_button = QPushButton('Pause downloads')
        self.pause_button.setEnabled(False)
        self.pause_button.clicked.connect(self.pause)
        self.resume_button = QPushButton('Resume Selected')
        self.resume_button.clicked.connect(self.resume)
        buttons.addWidget(self.pause_button)
        buttons.addWidget(self.resume_button)
        layout.addLayout(buttons)
        self.summary = QLabel('Ready')
        layout.addWidget(self.summary)
        self.reload()

    def reload(self):
        self.tree.clear()
        self.rows.clear()
        for record in DownloadState(self.client._model_storage_path).list():
            event = dict(record)
            if event.get('status') == 'downloading':
                event['status'] = 'interrupted'
            self.update_row(record['model_id'], event)

    def update_row(self, model_id, event):
        if model_id not in self.rows:
            item = QTreeWidgetItem([model_id, 'queued', '', '', '', '', ''])
            item.setData(0, Qt.UserRole, model_id)
            self.tree.addTopLevelItem(item)
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            self.tree.setItemWidget(item, 2, bar)
            self.rows[model_id] = (item, bar)
        item, bar = self.rows[model_id]
        if 'status' in event:
            item.setText(1, event['status'])
        if event.get('status') == 'downloading' and not event.get('total'):
            bar.setRange(0, 0)
        if 'bytes' in event:
            total = event.get('total', 0)
            bar.setRange(0, 100)
            bar.setValue(min(100, int(event['bytes'] * 100 / total)) if total else 0)
            item.setText(3, f"{size_text(event['bytes'])} / {size_text(total)}")
        if event.get('status') == 'completed':
            bar.setRange(0, 100)
            bar.setValue(100)
            item.setText(6, 'Complete')
        if event.get('status') in ('completed','failed','paused','interrupted'):
            bar.setRange(0, 100)
            item.setText(4, '')
            item.setText(5, '')
        if 'speed' in event:
            item.setText(4, size_text(event['speed']) + '/s')
            seconds = event.get('eta')
            item.setText(5, f'{int(seconds)//60}m {int(seconds)%60}s' if seconds is not None else '—')
        if event.get('file'):
            item.setText(6, event['file'] + (f" — {event['files_done']}/{event['files_total']} files" if 'files_done' in event else ''))
        if event.get('error'):
            item.setText(6, event['error'])
        item.setToolTip(6, item.text(6))
        self.tree.resizeColumnToContents(0)

    def begin(self, worker):
        self.worker = worker
        self.pause_button.setEnabled(True)
        self.resume_button.setEnabled(False)
        self.summary.setText(f'{len(worker.model_ids)} model(s) queued')
        for model_id in worker.model_ids:
            self.update_row(model_id, {'status':'queued'})
        worker.updated.connect(self.update_row)
        worker.finished.connect(self.finished)

    def pause(self):
        if self.worker:
            self.worker.stop_event.set()
            self.pause_button.setEnabled(False)
            self.summary.setText('Pausing; partial downloads will be kept…')

    def finished(self):
        self.worker = None
        self.pause_button.setEnabled(False)
        self.resume_button.setEnabled(True)
        self.summary.setText('Queue stopped. Completed models are available in Models; paused or failed items can be resumed.')

    def resume(self):
        selected = [item.data(0, Qt.UserRole) for item in self.tree.selectedItems() if item.text(1) != 'completed']
        if selected:
            self.resume_requested.emit(selected)
        else:
            self.summary.setText('Select a queued, paused, interrupted, or failed model first.')
