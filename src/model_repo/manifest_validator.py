"""Manifest validation for HuggingFace model downloads."""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone


class ManifestValidationError(Exception):
    """Raised when manifest validation fails."""
    pass


class ManifestValidator:
    """Validates model manifests before download execution."""

    MAX_MODELS_PER_BATCH = 10
    MAX_STORAGE_QUOTA_BYTES = 1024 ** 4  # 1 TiB

    def __init__(self, manifest_path: str = "./models/.manifest.json"):
        """Initialize the manifest validator.

        Args:
            manifest_path: Path to the manifest JSON file.
        """
        self.manifest_path = Path(manifest_path)

    def _load_manifest(self) -> Dict[str, Any]:
        """Load the manifest from disk."""
        if not self.manifest_path.exists():
            return {"models": []}
        with open(self.manifest_path, "r") as f:
            return json.load(f)

    def _save_manifest(self, manifest: Dict[str, Any]) -> None:
        """Save the manifest to disk."""
        Path(self.manifest_path).parent.mkdir(parents=True, exist_ok=True)

        with open(self.manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    def validate_batch(self, model_ids: List[str]) -> Tuple[bool, List[str]]:
        """Validate a batch of model IDs before download.

        Args:
            model_ids: List of model IDs to validate.

        Returns:
            Tuple of (is_valid, list of error messages).
        """
        errors = []

        # Check batch size limit
        if len(model_ids) > self.MAX_MODELS_PER_BATCH:
            errors.append(
                f"Batch size {len(model_ids)} exceeds maximum of {self.MAX_MODELS_PER_BATCH} models"
            )

        # Check for duplicates
        if len(model_ids) != len(set(model_ids)):
            duplicates = [mid for mid in model_ids if model_ids.count(mid) > 1]
            errors.append(f"Duplicate model IDs in batch: {set(duplicates)}")

        # Check storage quota
        current_usage = self._get_current_storage_usage()
        estimated_new_usage = self._estimate_batch_size(model_ids)
        total_usage = current_usage + estimated_new_usage

        if total_usage > self.MAX_STORAGE_QUOTA_BYTES:
            errors.append(
                f"Estimated storage usage {total_usage / (1024**3):.2f}GB exceeds quota of 1 TiB"
            )

        # Validate each model ID format
        for model_id in model_ids:
            if not model_id or not isinstance(model_id, str):
                errors.append(f"Invalid model ID: {model_id}")
            elif "/" not in model_id:
                errors.append(f"Model ID must include namespace: {model_id}")

        return len(errors) == 0, errors

    def _get_current_storage_usage(self) -> int:
        """Calculate current storage usage from manifest."""
        manifest = self._load_manifest()
        total_size = 0
        seen = set()
        # Include partial/unregistered downloads in the repository as well.
        roots = [self.manifest_path.parent] + [Path(model["local_path"]) for model in manifest.get("models", []) if model.get("local_path")]
        for root in roots:
            if not root.is_dir():
                continue
            for file in root.rglob("*"):
                if not file.is_file() or file.resolve() == self.manifest_path.resolve():
                    continue
                stat = file.stat()
                identity = (stat.st_dev, stat.st_ino)
                if identity not in seen:
                    seen.add(identity)
                    total_size += stat.st_size
        return total_size

    def _estimate_batch_size(self, model_ids: List[str]) -> int:
        """Estimate storage size for a batch of models.

        Args:
            model_ids: List of model IDs to estimate.

        Returns:
            Estimated size in bytes.
        """
        # Use average model size as estimate (conservative)
        # Typical models range from 100MB to 10GB
        average_model_size = 1 * 1024 * 1024 * 1024  # 1GB average
        return average_model_size * len(model_ids)

    def validate_model_exists(self, model_id: str) -> bool:
        """Check if a model is already in the manifest.

        Args:
            model_id: The model ID to check.

        Returns:
            True if model exists in manifest, False otherwise.
        """
        manifest = self._load_manifest()
        return any(m["model_id"] == model_id for m in manifest.get("models", []))

    def get_model_status(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a model from the manifest.

        Args:
            model_id: The model ID to check.

        Returns:
            Model metadata if found, None otherwise.
        """
        manifest = self._load_manifest()
        for model in manifest.get("models", []):
            if model["model_id"] == model_id:
                return model
        return None

    def record_download(self, model_id: str, local_path: str, model_info: Dict[str, Any]) -> None:
        """Record a successful download in the manifest.

        Args:
            model_id: The model identifier.
            local_path: The local path where the model was downloaded.
            model_info: Model metadata from HuggingFace.
        """
        manifest = self._load_manifest()

        # Check if model already exists
        for i, model in enumerate(manifest["models"]):
            if model["model_id"] == model_id:
                manifest["models"][i] = {
                    "model_id": model_id,
                    "local_path": local_path,
                    "downloaded_at": datetime.now(timezone.utc).isoformat(),
                    "model_info": model_info,
                    "managed_files": True,
                }
                break
        else:
            # Add new model entry
            manifest["models"].append({
                "model_id": model_id,
                "local_path": local_path,
                "downloaded_at": datetime.now(timezone.utc).isoformat(),
                "model_info": model_info,
                "managed_files": True,
            })

        self._save_manifest(manifest)

    def get_storage_usage(self) -> Dict[str, Any]:
        """Get current storage usage information.

        Returns:
            Dictionary with usage statistics.
        """
        current_usage = self._get_current_storage_usage()
        return {
            "used_bytes": current_usage,
            "used_gb": current_usage / (1024 ** 3),
            "used_tb": current_usage / (1024 ** 4),
            "quota_bytes": self.MAX_STORAGE_QUOTA_BYTES,
            "quota_gb": self.MAX_STORAGE_QUOTA_BYTES / (1024 ** 3),
            "quota_tb": self.MAX_STORAGE_QUOTA_BYTES / (1024 ** 4),
            "usage_percent": (current_usage / self.MAX_STORAGE_QUOTA_BYTES) * 100,
            "available_bytes": self.MAX_STORAGE_QUOTA_BYTES - current_usage,
        }

    def check_quota_available(self, estimated_size: int) -> Tuple[bool, str]:
        """Check if there is enough storage quota for a download.

        Args:
            estimated_size: Estimated size of the model in bytes.

        Returns:
            Tuple of (has_quota, message).
        """
        current_usage = self._get_current_storage_usage()
        new_usage = current_usage + estimated_size

        if new_usage > self.MAX_STORAGE_QUOTA_BYTES:
            return False, (
                f"Insufficient storage: {new_usage / (1024**3):.2f}GB required, "
                f"only {self.MAX_STORAGE_QUOTA_BYTES / (1024**3):.2f}GB quota available"
            )

        return True, f"Storage quota available: {self.MAX_STORAGE_QUOTA_BYTES / (1024**3):.2f}GB"

    def validate_manifest_integrity(self) -> Tuple[bool, List[str]]:
        """Validate the integrity of the manifest file.

        Returns:
            Tuple of (is_valid, list of validation errors).
        """
        errors = []

        if not self.manifest_path.exists():
            return True, []  # Empty manifest is valid

        try:
            manifest = self._load_manifest()
        except json.JSONDecodeError as e:
            return False, [f"Invalid JSON in manifest: {str(e)}"]

        if not isinstance(manifest, dict):
            return False, ["Manifest must be a JSON object"]

        if "models" not in manifest:
            return False, ["Manifest must contain 'models' key"]

        if not isinstance(manifest["models"], list):
            return False, ["'models' must be a list"]

        for i, model in enumerate(manifest["models"]):
            if not isinstance(model, dict):
                errors.append(f"Model at index {i} must be a dictionary")
                continue

            if "model_id" not in model:
                errors.append(f"Model at index {i} missing 'model_id'")

            if "local_path" not in model:
                errors.append(f"Model at index {i} missing 'local_path'")

        return len(errors) == 0, errors
