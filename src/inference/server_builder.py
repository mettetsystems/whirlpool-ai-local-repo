"""Inference server builder for deploying models as containerized endpoints."""

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict


@dataclass
class InferenceConfig:
    """Configuration for inference server deployment."""
    model_id: str
    model_path: str
    host_port: int = 8000
    device: str = "auto"
    max_batch_size: int = 1
    max_seq_len: int = 2048
    quantization: str = "none"
    vllm_enabled: bool = False
    gpu_count: int = 1
    gpu_driver: str = "nvidia"
    log_dir: str = "./logs"
    container_name: str = "inference-server"
    inference_image: str = "whirlpool/inference-server:latest"


class InferenceServerBuilder:
    """Builder for creating and managing inference server containers."""

    def __init__(self, compose_path: str = "docker/inference-compose.yaml"):
        """Initialize the inference server builder.

        Args:
            compose_path: Path to the Docker/Podman compose file.
        """
        self.compose_path = Path(compose_path)
        self._env_vars: Dict[str, str] = {}

    def set_env_var(self, key: str, value: str) -> "InferenceServerBuilder":
        """Set an environment variable for the compose file.

        Args:
            key: Environment variable key.
            value: Environment variable value.

        Returns:
            Self for method chaining.
        """
        self._env_vars[key] = value
        return self

    def generate_config(self, config: InferenceConfig) -> str:
        """Generate Docker/Podman compose configuration from model card.

        Args:
            config: Inference configuration object.

        Returns:
            Compose file content as string.
        """
        model_path = Path(config.model_path).expanduser().resolve()
        if not model_path.is_dir():
            raise ValueError(f"Model directory does not exist: {model_path}")
        if not 1024 <= config.host_port <= 65535:
            raise ValueError("Port must be between 1024 and 65535")
        if config.device not in ("cpu", "cuda", "auto"):
            raise ValueError("Device must be cpu, cuda, or auto")
        if config.quantization != "none":
            raise ValueError("The bundled runtime currently supports unquantized Transformers models")
        if config.vllm_enabled:
            raise ValueError("The bundled runtime uses Transformers; set vllm_enabled=False")
        root = Path(__file__).resolve().parents[2]
        service = {
            "image": config.inference_image,
            "build": {"context": str(root), "dockerfile": "docker/Dockerfile"},
            "container_name": config.container_name,
            "ports": [f"127.0.0.1:{config.host_port}:8000"],
            "environment": {"MODEL_PATH": "/model", "DEVICE": config.device,
                            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
            "volumes": [{"type": "bind", "source": str(model_path), "target": "/model", "read_only": True}],
            "healthcheck": {"test": ["CMD", "python", "-c",
                "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"],
                "interval": "10s", "timeout": "5s", "retries": 30, "start_period": "60s"},
            "restart": "unless-stopped",
            "logging": {"driver": "json-file", "options": {"max-size": "10m", "max-file": "3"}},
        }
        if config.device != "cpu" and config.gpu_count > 0:
            service["deploy"] = {"resources": {"reservations": {"devices": [
                {"driver": config.gpu_driver, "count": config.gpu_count, "capabilities": ["gpu"]}]}}}
        # JSON is valid YAML and safely quotes paths and user-provided settings.
        return json.dumps({"services": {"inference-server": service}}, indent=2)

    def save_config(self, config: InferenceConfig, output_path: Optional[str] = None) -> Path:
        """Save the generated compose configuration to a file.

        Args:
            config: Inference configuration object.
            output_path: Optional path to save the config. Defaults to self.compose_path.

        Returns:
            Path to the saved configuration file.
        """
        output_path = Path(output_path) if output_path else self.compose_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        compose_content = self.generate_config(config)
        output_path.write_text(compose_content)
        return output_path

    def build_container(self, config: InferenceConfig, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Build the inference server container.

        Args:
            config: Inference configuration object.
            output_path: Optional path to the compose file.

        Returns:
            Dictionary with build status and output.
        """
        compose_file = self.save_config(config, output_path)

        # Determine which compose tool to use
        compose_cmd = self._get_compose_command()

        if compose_cmd is None:
            return {
                "success": False,
                "error": "No compose tool found. Please install Docker or Podman with compose support.",
            }

        try:
            result = subprocess.run(
                compose_cmd + ["-f", str(compose_file), "up", "--build", "-d"],
                capture_output=True,
                text=True,
                timeout=1800,
            )

            return {
                "success": result.returncode == 0,
                "compose_tool": compose_cmd[0],
                "stdout": result.stdout,
                "stderr": result.stderr,
                "compose_file": str(compose_file),
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Container build timed out after 30 minutes.",
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    def start_container(self, config: InferenceConfig, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Start the inference server container.

        Args:
            config: Inference configuration object.
            output_path: Optional path to the compose file.

        Returns:
            Dictionary with start status and output.
        """
        compose_file = self.save_config(config, output_path)
        compose_cmd = self._get_compose_command()

        if compose_cmd is None:
            return {
                "success": False,
                "error": "No compose tool found.",
            }

        try:
            result = subprocess.run(
                compose_cmd + ["-f", str(compose_file), "up", "-d"],
                capture_output=True,
                text=True,
                timeout=120,
            )

            return {
                "success": result.returncode == 0,
                "compose_tool": compose_cmd[0],
                "stdout": result.stdout,
                "stderr": result.stderr,
                "compose_file": str(compose_file),
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Container start timed out.",
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    def stop_container(self, config: InferenceConfig, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Stop the inference server container.

        Args:
            config: Inference configuration object.
            output_path: Optional path to the compose file.

        Returns:
            Dictionary with stop status and output.
        """
        compose_file = self.save_config(config, output_path)
        compose_cmd = self._get_compose_command()

        if compose_cmd is None:
            return {
                "success": False,
                "error": "No compose tool found.",
            }

        try:
            result = subprocess.run(
                compose_cmd + ["-f", str(compose_file), "down"],
                capture_output=True,
                text=True,
                timeout=60,
            )

            return {
                "success": result.returncode == 0,
                "compose_tool": compose_cmd[0],
                "stdout": result.stdout,
                "stderr": result.stderr,
                "compose_file": str(compose_file),
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Container stop timed out.",
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    def _get_compose_command(self) -> Optional[List[str]]:
        """Detect and return the correct compose command.

        Returns:
            List of command arguments for the compose tool, or None if not found.
        """
        # Check for Podman Compose first (preferred on this system)
        try:
            result = subprocess.run(
                ["podman-compose", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return ["podman-compose"]
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # Check for Docker Compose (v2)
        try:
            result = subprocess.run(
                ["docker", "compose", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return ["docker", "compose"]
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # Check for docker-compose (legacy)
        try:
            result = subprocess.run(
                ["docker-compose", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return ["docker-compose"]
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        return None

    def get_model_recommendations(self, model_card: Dict[str, Any]) -> InferenceConfig:
        """Generate recommended inference settings based on model card.

        Args:
            model_card: Model card dictionary with model metadata.

        Returns:
            InferenceConfig with recommended settings.
        """
        # Extract model information
        model_id = model_card.get("model_id") or model_card.get("id", "unknown")
        metadata = model_card.get("model_info", model_card)
        tags = metadata.get("tags", [])
        pipeline_tag = model_card.get("pipeline_tag", "text-generation")

        # Determine device based on model type
        device = "cuda" if any("cuda" in tag or "gpu" in tag for tag in tags) else "cpu"

        # Set quantization based on model size hints
        quantization = "none"
        if any("quantized" in tag or "bitsandbytes" in tag for tag in tags):
            quantization = "4bit"
        elif any("fp16" in tag for tag in tags):
            quantization = "fp16"

        # Set sequence length based on model type
        max_seq_len = 2048
        if "long-context" in tags or "seq2seq" in tags:
            max_seq_len = 4096
        elif "embedding" in tags:
            max_seq_len = 512

        # Set batch size based on model type
        max_batch_size = 1
        if "batch" in tags or "serving" in tags:
            max_batch_size = 4

        # Determine if VLLM is suitable
        vllm_enabled = True
        if "embedding" in tags or "feature-extraction" in tags:
            vllm_enabled = False

        return InferenceConfig(
            model_id=model_id,
            model_path=model_card.get("local_path", f"/models/{model_id}"),
            device=device,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            quantization="none",
            vllm_enabled=False,
            gpu_count=0 if device == "cpu" else 1,
        )

    def run_health_check(self, config: InferenceConfig) -> Dict[str, Any]:
        """Run a health check on the inference server.

        Args:
            config: Inference configuration object.

        Returns:
            Dictionary with health check status.
        """
        try:
            import requests
            response = requests.get(
                f"http://localhost:{config.host_port}/health",
                timeout=10,
            )
            return {
                "healthy": response.status_code == 200,
                "status_code": response.status_code,
                "response": response.text,
            }
        except requests.exceptions.ConnectionError:
            return {
                "healthy": False,
                "status_code": None,
                "response": "Connection refused",
            }
        except Exception as e:
            return {
                "healthy": False,
                "status_code": None,
                "response": str(e),
            }

    def export_config(self, config: InferenceConfig, output_path: str) -> Path:
        """Export the configuration to a JSON file.

        Args:
            config: Inference configuration object.
            output_path: Path to save the JSON file.

        Returns:
            Path to the exported configuration file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        config_dict = asdict(config)
        output_path.write_text(json.dumps(config_dict, indent=2))
        return output_path

    def load_config(self, config_path: str) -> InferenceConfig:
        """Load configuration from a JSON file.

        Args:
            config_path: Path to the JSON configuration file.

        Returns:
            InferenceConfig object.
        """
        config_path = Path(config_path)
        config_dict = json.loads(config_path.read_text())
        return InferenceConfig(**config_dict)


    def wait_until_healthy(self, config: InferenceConfig, timeout: float = 300):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.run_health_check(config)["healthy"]:
                return
            time.sleep(2)
        raise RuntimeError("Container started but the model did not become ready; inspect compose logs")

    def smoke_test(self, config: InferenceConfig, requests_count: int = 3):
        import requests
        for _ in range(requests_count):
            response = requests.post(f"http://localhost:{config.host_port}/generate",
                                     json={"prompt": "Hello", "max_new_tokens": 4}, timeout=120)
            response.raise_for_status()
            if not isinstance(response.json().get("text"), str):
                raise RuntimeError("Inference smoke test returned an invalid response")

    def export_image(self, config: InferenceConfig, directory: str) -> Path:
        """Export a built image archive without changing the engine's own storage."""
        from uuid import uuid4
        compose = self._get_compose_command()
        if compose is None:
            raise RuntimeError("No container engine found for image export")
        engine = "podman" if compose[0] == "podman-compose" else "docker"
        destination = Path(directory).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        output = destination / f"inference-{uuid4().hex[:12]}.tar"
        temporary = output.with_suffix(".tar.partial")
        try:
            result = subprocess.run([engine, "save", "--output", str(temporary), config.inference_image],
                                    capture_output=True, text=True, timeout=1800)
            if result.returncode != 0:
                raise RuntimeError("Image export failed: " + result.stderr)
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
        return output
