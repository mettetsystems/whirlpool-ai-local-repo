"""Persistent application defaults and private local credential storage."""
import json
import os
import tempfile
from pathlib import Path
from src import credentials

DEFAULTS = {
    "model_storage_path": "./models",
    "training_output_path": "./training_output",
    "inference_output_path": "./inference_output",
    "image_export_path": "./inference_images",
    "inference_image": "whirlpool/inference-server:latest",
    "export_inference_image": False,
}
PATH_KEYS = tuple(key for key in DEFAULTS if key.endswith('_path'))


def read_settings(config_path):
    path = Path(config_path)
    config = json.loads(path.read_text()) if path.exists() else {}
    values = {**DEFAULTS, **config}
    for key in PATH_KEYS:
        values[key] = str(Path(values[key]).expanduser().resolve())
    if config.get("credential_backend") == "secret-service":
        try:
            values["hf_token"] = credentials.get_token(path)
        except RuntimeError as exc:
            values["hf_token"] = ""
            values["credential_error"] = str(exc)
    else:
        # Read old configurations for migration; only Save writes credentials.
        values["hf_token"] = config.get("hf_token", "")
    return values



def private_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".whirlpool-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_settings(config_path, updates):
    values = {**read_settings(config_path), **updates}
    for key in PATH_KEYS:
        if not str(values[key]).strip():
            raise ValueError(f"Choose a directory for {key}")
        directory = Path(values[key]).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        values[key] = str(directory)
    image = values["inference_image"].strip()
    if not image or image.startswith('-') or any(char.isspace() for char in image):
        raise ValueError("Enter an inference image name without spaces")
    values["inference_image"] = image
    token = values.pop("hf_token", "").strip()
    if values.get("credential_error") and not token:
        raise RuntimeError(values["credential_error"])
    if token or values.get("credential_backend") == "secret-service":
        credentials.set_token(config_path, token)
        values["credential_backend"] = "secret-service"
    values.pop("credential_error", None)
    values.pop("credentials_path", None)
    private_json(config_path, values)
    return {**values, "hf_token": token}
