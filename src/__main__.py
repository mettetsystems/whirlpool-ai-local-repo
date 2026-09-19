"""Launch the desktop app with python -m src."""
import argparse
import json
import sys
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Whirlpool local model browser")
    parser.add_argument("--config", default="config/hf_auth.json")
    parser.add_argument("--smoke-test", action="store_true", help="Open and close an isolated empty window")
    args = parser.parse_args()
    from PyQt5.QtCore import QTimer
    from PyQt5.QtWidgets import QApplication
    from src.model_repo.hf_client import HuggingFaceClient
    from src.ui.model_browser import ModelBrowserWindow
    app = QApplication(sys.argv[:1])
    with tempfile.TemporaryDirectory(prefix="whirlpool-launch-") as temporary:
        config = args.config
        if args.smoke_test:
            config = str(Path(temporary) / "config.json")
            Path(config).write_text(json.dumps({"model_storage_path": str(Path(temporary) / "models")}))
        try:
            window = ModelBrowserWindow(HuggingFaceClient(config))
        except (OSError, ValueError) as exc:
            parser.exit(1, f"Cannot launch Whirlpool: {exc}\nSee README.md for configuration.\n")
        window.show()
        if args.smoke_test:
            QTimer.singleShot(100, app.quit)
        return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
