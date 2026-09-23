"""Token-free, restart-safe download records."""
import hashlib
import json
from pathlib import Path
from src.settings import private_json


class DownloadState:
    def __init__(self, storage):
        self.directory = Path(storage) / '.downloads'

    def path(self, model_id):
        return self.directory / (hashlib.sha256(model_id.encode()).hexdigest() + '.json')

    def get(self, model_id):
        path = self.path(model_id)
        return json.loads(path.read_text()) if path.exists() else None

    def save(self, record):
        private_json(self.path(record['model_id']), record)

    def list(self):
        result = []
        for path in sorted(self.directory.glob('*.json')):
            try:
                result.append(json.loads(path.read_text()))
            except (OSError, ValueError):
                continue
        return result
