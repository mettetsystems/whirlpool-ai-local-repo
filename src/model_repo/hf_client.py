"""HuggingFace model client for downloading and managing models locally."""

import json
import hashlib
from datetime import datetime, timezone
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any

import requests
from huggingface_hub import HfApi, snapshot_download, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError, validate_repo_id
from src.model_repo.manifest_validator import ManifestValidator, ManifestValidationError
from src.settings import read_settings, save_settings
from src.model_repo.download_state import DownloadState
from src.model_repo.download_process import run_download, DownloadPaused


class HuggingFaceClient:
    """Client for interacting with HuggingFace Hub for model management."""

    def __init__(self, auth_config_path: str = "config/hf_auth.json"):
        """Initialize the HuggingFace client with authentication configuration.

        Args:
            auth_config_path: Path to the HF authentication JSON config file.
        """
        self.auth_config_path = Path(auth_config_path)
        self.api = HfApi()
        self._token: Optional[str] = None
        self._model_storage_path: Path = Path("./models")
        self._manifest_path: Path = Path("./models/.manifest.json")
        self._load_config()

    def _load_config(self) -> None:
        """Load authentication configuration from JSON file."""
        config = read_settings(self.auth_config_path)
        self.settings = config

        self._token = os.environ.get("HF_TOKEN") or config.get("hf_token", "")
        storage_path = config.get("model_storage_path", "./models")
        self._model_storage_path = Path(storage_path).resolve()

        # Update manifest path to be relative to storage path
        self._manifest_path = self._model_storage_path / ".manifest.json"

        # Ensure storage directory exists
        self._model_storage_path.mkdir(parents=True, exist_ok=True)

        # Ensure manifest file exists
        if not self._manifest_path.exists():
            self._manifest_path.write_text(json.dumps({"models": []}))

    def update_settings(self, updates):
        # Preserve existing registrations when changing the destination for new downloads.
        previous = self.list_models()
        destination = Path(updates.get("model_storage_path", self.settings["model_storage_path"])).expanduser().resolve()
        manifest_path = destination / ".manifest.json"
        merged = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"models": []}
        by_id = {model["model_id"]: model for model in merged["models"]}
        for model in previous:
            existing = by_id.get(model["model_id"])
            if existing and Path(existing["local_path"]).resolve() != Path(model["local_path"]).resolve():
                raise ValueError("The destination contains a different copy of " + model["model_id"])
            by_id[model["model_id"]] = model
        save_settings(self.auth_config_path, updates)
        self._load_config()
        self._save_manifest({"models": list(by_id.values())})

    def _get_manifest(self) -> Dict[str, Any]:
        """Load the model manifest from disk."""
        with open(self._manifest_path, "r") as f:
            return json.load(f)

    def _save_manifest(self, manifest: Dict[str, Any]) -> None:
        """Save the model manifest to disk."""
        with open(self._manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    def _update_manifest(self, model_id: str, metadata: Dict[str, Any]) -> None:
        """Add or update a model entry in the manifest."""
        manifest = self._get_manifest()

        # Check if model already exists
        for i, model in enumerate(manifest["models"]):
            if model["model_id"] == model_id:
                manifest["models"][i] = {
                    "model_id": model_id,
                    "local_path": str(metadata["local_path"]),
                    "downloaded_at": metadata["downloaded_at"],
                    "model_info": metadata["model_info"],
                    "managed_files": metadata.get("managed_files", True),
                }
                break
        else:
            # Add new model entry
            manifest["models"].append({
                "model_id": model_id,
                "local_path": str(metadata["local_path"]),
                "downloaded_at": metadata["downloaded_at"],
                "model_info": metadata["model_info"],
                "managed_files": metadata.get("managed_files", True),
            })

        self._save_manifest(manifest)

    def add_local_model(self, directory: str) -> Dict[str, Any]:
        """Register an existing model in place without taking ownership of its files."""
        path = Path(directory).expanduser().resolve()
        if not path.is_dir():
            raise ValueError("Select an existing model directory.")
        if not (path / "config.json").is_file() and not any(path.glob("*.gguf")):
            raise ValueError("Select a model folder containing config.json or a GGUF model file.")
        for model in self.list_models():
            if Path(model["local_path"]).resolve() == path:
                return model
        suffix = hashlib.sha256(str(path).encode()).hexdigest()[:12]
        metadata = {
            "model_id": f"local/{path.name}-{suffix}",
            "local_path": str(path),
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
            "model_info": {},
            "managed_files": False,
        }
        self._update_manifest(metadata["model_id"], metadata)
        return metadata

    def set_token(self, token: str) -> None:
        """Set the HuggingFace API token.

        Args:
            token: HuggingFace API token string.
        """
        self._token = token

    def get_token(self) -> Optional[str]:
        """Get the current HuggingFace API token."""
        return self._token

    def list_models(self) -> List[Dict[str, Any]]:
        """List all locally stored models from the manifest.

        Returns:
            List of model metadata dictionaries.
        """
        manifest = self._get_manifest()
        return manifest.get("models", [])

    def download_model(
        self,
        model_id: str,
        local_dir: Optional[str] = None,
        progress_callback=None,
        stop_event=None,
        **kwargs
    ) -> Dict[str, Any]:
        """Download a model from HuggingFace Hub.

        Args:
            model_id: The model identifier (e.g., 'meta-llama/Llama-2-7b').
            local_dir: Optional local directory to download to.
            **kwargs: Additional arguments passed to snapshot_download.

        Returns:
            Dictionary containing model metadata.

        Raises:
            RepositoryNotFoundError: If the model does not exist.
            HfHubHTTPError: If there's an authentication or network error.
        """
        if not self._token:
            raise ValueError(
                "HuggingFace token not set. Please configure hf_auth.json or call set_token()."
            )

        validate_repo_id(model_id)
        state = DownloadState(self._model_storage_path)
        previous = state.get(model_id)
        existing = self.get_model_info(model_id)
        if local_dir:
            model_local_dir = Path(local_dir).expanduser().resolve()
        elif previous and previous.get("local_path") and previous.get("status") != "completed":
            model_local_dir = Path(previous["local_path"]).resolve()
        elif existing and existing.get("managed_files", True):
            model_local_dir = Path(existing["local_path"]).resolve()
        else:
            # Hash the full ID: replacing slashes with underscores is not injective.
            digest = hashlib.sha256(model_id.encode()).hexdigest()
            model_local_dir = self._model_storage_path / digest
        for entry in self.list_models():
            other = Path(entry["local_path"]).resolve()
            if entry["model_id"] != model_id and (
                other.is_relative_to(model_local_dir) or model_local_dir.is_relative_to(other)
            ):
                raise ValueError("Download destination overlaps another model. Choose a new directory.")
        resumable = previous and previous.get("revision") and Path(previous["local_path"]).resolve() == model_local_dir
        if model_local_dir.exists() and any(model_local_dir.iterdir()) and not resumable and (
            not existing or Path(existing["local_path"]).resolve() != model_local_dir
            or not existing.get("managed_files", True)
        ):
            raise ValueError("Download destination must be empty or owned by this downloaded model")

        # Require Hub file metadata rather than guessing model size. Pin the revision
        # so the downloaded snapshot has the same contents as the quota estimate.
        revision = previous["revision"] if resumable and previous.get("status") != "completed" else kwargs.get("revision")
        if resumable and kwargs.get("revision") and kwargs["revision"] != revision:
            raise ValueError("Finish or remove the pending download before changing its revision")
        info = self.api.model_info(model_id, token=self._token, files_metadata=True, revision=revision)
        if not info.siblings or any(type(file.size) is not int or file.size < 0 for file in info.siblings):
            raise ManifestValidationError("Cannot determine model size; download was not started")
        if not isinstance(info.sha, str) or not info.sha:
            raise ManifestValidationError("Cannot resolve model revision; download was not started")
        total = sum(file.size for file in info.siblings)
        completed = 0
        if resumable and previous["revision"] == info.sha:
            for file in info.siblings:
                name = getattr(file, 'rfilename', None)
                if isinstance(name, str):
                    path = (model_local_dir / name).resolve()
                    if path.is_relative_to(model_local_dir) and path.is_file() and path.stat().st_size == file.size:
                        completed += file.size
        # Existing partial chunks already count in storage usage; reserve only
        # remaining snapshot bytes on a pinned resume.
        retained = completed
        if resumable and previous["revision"] == info.sha:
            cache = model_local_dir / '.cache' / 'huggingface' / 'download'
            retained += sum(path.stat().st_size for path in cache.rglob('*.incomplete') if path.is_file())
        allowed, reason = ManifestValidator(str(self._manifest_path)).check_quota_available(max(0, total - min(total, retained)))
        if not allowed:
            raise ManifestValidationError(reason)
        kwargs["revision"] = info.sha
        model_local_dir.mkdir(parents=True, exist_ok=True)

        record = {"model_id":model_id, "local_path":str(model_local_dir), "revision":info.sha,
                  "status":"downloading", "bytes":completed, "total":total, "error":""}
        state.save(record)
        last_save = [0.0]
        def report(event):
            import time
            record.update(bytes=event['bytes'], total=event['total'])
            if time.monotonic() - last_save[0] > 1:
                state.save(record)
                last_save[0] = time.monotonic()
            progress_callback(event)
        try:
            if progress_callback is not None or stop_event is not None:
                if progress_callback is None:
                    progress_callback = lambda event: None
                options = dict(kwargs)
                options.pop('revision', None)
                if options.get('force_download'):
                    raise ValueError("Force download cannot be combined with resumable UI downloads")
                downloaded_files = run_download({
                    'model_id':model_id, 'local_dir':str(model_local_dir), 'revision':info.sha,
                    'token':self._token, 'total_bytes':total, 'completed_bytes':completed,
                    'files_total':len(info.siblings), 'options':options,
                }, report, stop_event)
            else:
                downloaded_files = snapshot_download(repo_id=model_id, local_dir=str(model_local_dir),
                                                     token=self._token, **kwargs)
        except Exception as exc:
            persisted = 0
            for file in info.siblings:
                name = getattr(file, 'rfilename', None)
                if isinstance(name, str):
                    path = (model_local_dir / name).resolve()
                    if path.is_relative_to(model_local_dir) and path.is_file() and path.stat().st_size == file.size:
                        persisted += file.size
            cache = model_local_dir / '.cache' / 'huggingface' / 'download'
            persisted += sum(path.stat().st_size for path in cache.rglob('*.incomplete') if path.is_file())
            record.update(bytes=min(total, persisted), status='paused' if isinstance(exc, DownloadPaused) else 'failed',
                          error=str(exc).replace(self._token, '[redacted]'))
            state.save(record)
            raise

        # Get model info
        try:
            model_info = info
            model_info_dict = {
                "id": model_info.id,
                "author": model_info.author,
                "likes": model_info.likes,
                "downloads": model_info.downloads,
                "tags": list(model_info.tags) if model_info.tags else [],
                "pipeline_tag": model_info.pipeline_tag,
            }
        except Exception:
            model_info_dict = {}

        # Create metadata
        metadata = {
            "model_id": model_id,
            "local_path": str(model_local_dir),
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
            "model_info": model_info_dict,
        }

        # Update manifest
        self._update_manifest(model_id, metadata)
        record.update(status="completed", bytes=total, error="")
        state.save(record)

        return metadata

    def cleanup_download(self, model_id):
        """Remove a stopped transfer's owned files, or clear completed history.

        The caller must ensure the download worker has finished first. Custom
        directories and registered models are never deleted by this operation.
        """
        import shutil
        state = DownloadState(self._model_storage_path)
        record = state.get(model_id)
        if not record:
            return
        if record.get('status') != 'completed' and record.get('local_path'):
            destination = Path(record['local_path'])
            expected = self._model_storage_path / hashlib.sha256(model_id.encode()).hexdigest()
            registered = [Path(model['local_path']).resolve() for model in self.list_models()]
            resolved = destination.resolve()
            if any(path.is_relative_to(resolved) or resolved.is_relative_to(path) for path in registered):
                raise ValueError('These files belong to a registered model. Manage them from Models.')
            if destination.is_symlink() or destination.absolute() != expected.absolute() or resolved != expected.absolute():
                raise ValueError('Cleanup only removes app-owned download folders. This custom location must be managed manually.')
            if destination.exists():
                shutil.rmtree(destination)
        state.path(model_id).unlink(missing_ok=True)

    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a locally stored model.

        Args:
            model_id: The model identifier.

        Returns:
            Model metadata if found, None otherwise.
        """
        manifest = self._get_manifest()
        for model in manifest.get("models", []):
            if model["model_id"] == model_id:
                return model
        return None

    def model_exists(self, model_id: str) -> bool:
        """Check if a model is already stored locally.

        Args:
            model_id: The model identifier.

        Returns:
            True if the model exists locally, False otherwise.
        """
        return self.get_model_info(model_id) is not None

    def delete_model(self, model_id: str) -> bool:
        """Delete a locally stored model.

        Args:
            model_id: The model identifier.

        Returns:
            True if the model was deleted, False if not found.
        """
        model_info = self.get_model_info(model_id)
        if not model_info:
            return False

        # Remove local directory
        local_path = Path(model_info["local_path"]) if isinstance(model_info["local_path"], str) else model_info["local_path"]
        # Old versions could register multiple IDs against one directory. Never
        # recursively delete a directory overlapping another registration.
        shared = any(
            entry["model_id"] != model_id and (
                Path(entry["local_path"]).resolve().is_relative_to(local_path.resolve())
                or local_path.resolve().is_relative_to(Path(entry["local_path"]).resolve())
            ) for entry in self.list_models()
        )
        if model_info.get("managed_files", True) and not shared and local_path.exists():
            shutil.rmtree(local_path)

        # Update manifest
        manifest = self._get_manifest()
        manifest["models"] = [
            m for m in manifest["models"] if m["model_id"] != model_id
        ]
        self._save_manifest(manifest)

        return True

    def search_models(
        self,
        query: str,
        limit: int = 10,
        include_download_size: bool = False,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search for models on HuggingFace Hub.

        Args:
            query: Search query string.
            limit: Maximum number of results to return.
            **kwargs: Additional arguments for the search.

        Returns:
            List of model search results.
        """
        if not self._token:
            raise ValueError(
                "HuggingFace token not set. Please configure hf_auth.json or call set_token()."
            )

        models = self.api.list_models(
            search=query,
            limit=limit,
            token=self._token,
            **kwargs
        )

        results = [
            {
                "id": model.id,
                "author": model.author,
                "likes": model.likes,
                "downloads": model.downloads,
                "tags": list(model.tags) if model.tags else [],
                "pipeline_tag": model.pipeline_tag,
            }
            for model in models
        ]

        if include_download_size:
            # Metadata only: no model files are downloaded. Keep requests bounded
            # and run this search from the UI worker rather than the GUI thread.
            from concurrent.futures import ThreadPoolExecutor
            def estimate(result):
                try:
                    info = self.api.model_info(result["id"], token=self._token,
                                               files_metadata=True, timeout=10)
                    files = info.siblings
                    if not files or any(type(file.size) is not int or file.size < 0 for file in files):
                        return None
                    return sum(file.size for file in files)
                except Exception:
                    # Gated/offline/missing metadata must not prevent selection.
                    return None
            with ThreadPoolExecutor(max_workers=4) as pool:
                for result, size in zip(results, pool.map(estimate, results)):
                    result["download_size_bytes"] = size
        return results

    def get_model_card(self, model_id: str) -> Optional[str]:
        """Get the model card (README) for a model.

        Args:
            model_id: The model identifier.

        Returns:
            Model card content as string, or None if not found.
        """
        info = self.get_model_info(model_id)
        if info:
            card = Path(info["local_path"]) / "README.md"
            if card.is_file():
                return card.read_text(encoding="utf-8")
        # Browsing downloaded models must never require a network connection.
        return None

    def verify_model_integrity(self, model_id: str) -> bool:
        """Verify the integrity of a locally stored model.

        Args:
            model_id: The model identifier.

        Returns:
            True if the model files exist and are accessible, False otherwise.
        """
        model_info = self.get_model_info(model_id)
        if not model_info:
            return False

        local_path = Path(model_info["local_path"])
        if not local_path.exists():
            return False

        # Check for essential model files
        essential_files = ["config.json", "pytorch_model.bin", "model.safetensors"]
        for file in essential_files:
            if (local_path / file).exists():
                return True

        # If no essential files found, consider it valid if directory has content
        return any(local_path.iterdir())
