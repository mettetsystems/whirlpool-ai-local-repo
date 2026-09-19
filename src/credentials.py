"""Desktop credentials: Linux Secret Service (GNOME Keyring on Fedora)."""
import hashlib
from pathlib import Path

SERVICE = "Whirlpool AI"


def account(config_path):
    profile = hashlib.sha256(str(Path(config_path).resolve()).encode()).hexdigest()[:16]
    return f"huggingface:{profile}"


def backend():
    # Explicit backend: never select a plaintext, third-party, or null fallback.
    from keyring.backends.SecretService import Keyring
    return Keyring()


def get_token(config_path):
    try:
        return backend().get_password(SERVICE, account(config_path)) or ""
    except Exception as exc:
        raise RuntimeError("Cannot access the desktop keyring. Unlock GNOME Keyring and run Whirlpool in your desktop session.") from exc


def set_token(config_path, token):
    try:
        backend().set_password(SERVICE, account(config_path), token)
    except Exception as exc:
        raise RuntimeError("Cannot save the token to the desktop keyring. Install the project dependencies, unlock GNOME Keyring, and try again. No plaintext token was saved.") from exc
