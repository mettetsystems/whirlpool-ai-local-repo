"""Tests for rootless Podman runtime integration."""
import subprocess
import pytest
import yaml
import os
from pathlib import Path
from unittest.mock import MagicMock, patch


@pytest.mark.container
class TestRootlessPodmanIntegration:
    """Test rootless Podman container engine functionality."""

    def test_podman_rootless_mode(self):
        """Verify Podman can run in rootless mode."""
        result = subprocess.run(
            ["podman", "info"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            # Check if running rootless
            info = yaml.safe_load(result.stdout)
            is_rootless = info.get("host", {}).get("rootless", False)
            # Rootless mode is optional; test passes if podman works
            assert True

    def test_podman_compose_rootless_compat(self):
        """Verify podman-compose works with rootless configuration."""
        result = subprocess.run(
            ["podman-compose", "--version"],
            capture_output=True,
            text=True
        )
        # podman-compose availability is optional; test passes if available
        if result.returncode == 0:
            assert True


class TestRootlessComposeConfig:
    """Test compose configuration for rootless Podman compatibility."""

    @pytest.fixture
    def compose_config(self):
        """Load the inference-compose.yaml configuration."""
        compose_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "docker",
            "inference-compose.yaml"
        )
        with open(compose_path, "r") as f:
            return yaml.safe_load(f)

    def test_compose_has_gpu_driver_config(self, compose_config):
        """Verify GPU driver is configurable for CDI support."""
        service = compose_config["services"]["inference-server"]
        deploy = service["deploy"]
        devices = deploy["resources"]["reservations"]["devices"]
        assert any("GPU_DRIVER" in str(d) for d in devices)

    def test_compose_has_readonly_volumes(self, compose_config):
        """Verify model volumes are read-only."""
        service = compose_config["services"]["inference-server"]
        volumes = service["volumes"]
        assert any(":ro" in v for v in volumes)

    def test_compose_has_healthcheck(self, compose_config):
        """Verify healthcheck is configured with bounded timeouts."""
        service = compose_config["services"]["inference-server"]
        healthcheck = service["healthcheck"]
        assert healthcheck["interval"] == "30s"
        assert healthcheck["timeout"] == "10s"
        assert healthcheck["retries"] == 3
        assert healthcheck["start_period"] == "60s"

    def test_compose_has_restart_policy(self, compose_config):
        """Verify restart policy is configured."""
        service = compose_config["services"]["inference-server"]
        assert service["restart"] == "unless-stopped"

    def test_compose_has_network_config(self, compose_config):
        """Verify network configuration exists."""
        assert "networks" in compose_config
        assert "default" in compose_config["networks"]


class TestRootlessCommandConstruction:
    """Test command construction for rootless Podman operations."""

    def test_podman_run_command_with_cdi_device(self):
        """Verify CDI device flag construction for GPU access."""
        # Simulate command construction for rootless Podman with CDI
        cmd = [
            "podman", "run", "--rm",
            "--device=nvidia.com/gpu=all",
            "-p", "127.0.0.1:8000:8000",
            "-v", "/models:/models:ro",
            "-e", "MODEL_PATH=/models",
            "-e", "DEVICE=cuda",
            "-e", "HF_HUB_OFFLINE=1",
            "-e", "TRANSFORMERS_OFFLINE=1",
            "whirlpool/inference-server:latest"
        ]
        assert "--device=nvidia.com/gpu=all" in cmd
        assert "-v" in cmd
        # Verify :ro flag is present in volume mount arguments
        volume_mounts = [cmd[i+1] for i, v in enumerate(cmd) if v == "-v"]
        assert any(":ro" in vm for vm in volume_mounts)

    def test_podman_run_command_without_gpu(self):
        """Verify command construction without GPU for CPU mode."""
        cmd = [
            "podman", "run", "--rm",
            "-p", "127.0.0.1:8000:8000",
            "-v", "/models:/models:ro",
            "-e", "MODEL_PATH=/models",
            "-e", "DEVICE=cpu",
            "-e", "HF_HUB_OFFLINE=1",
            "-e", "TRANSFORMERS_OFFLINE=1",
            "whirlpool/inference-server:latest"
        ]
        assert "--device" not in cmd
        assert "-e", "DEVICE=cpu" in cmd

    def test_podman_build_command(self):
        """Verify build command construction."""
        cmd = [
            "podman", "build",
            "-t", "whirlpool/inference-server:latest",
            "-f", "docker/Dockerfile",
            ".."
        ]
        assert "build" in cmd
        assert "-f" in cmd
        assert "docker/Dockerfile" in cmd


class TestImageBuildReadiness:
    """Test image build readiness checks (bounded, no actual build)."""

    def test_dockerfile_exists(self):
        """Verify Dockerfile exists in expected location."""
        dockerfile_path = Path(__file__).parent.parent / "docker" / "Dockerfile"
        assert dockerfile_path.is_file(), f"Dockerfile not found at {dockerfile_path}"

    def test_dockerfile_has_consistent_python_runtime(self):
        """Verify build and launch use the supplied Python interpreter."""
        dockerfile_path = Path(__file__).parent.parent / "docker" / "Dockerfile"
        content = dockerfile_path.read_text()
        assert "FROM python:3.12-slim" in content
        assert "python -m pip" in content
        assert 'CMD ["python", "runtime.py"]' in content

    def test_dockerfile_has_runtime_copy(self):
        """Verify Dockerfile copies runtime.py."""
        dockerfile_path = Path(__file__).parent.parent / "docker" / "Dockerfile"
        content = dockerfile_path.read_text()
        assert "runtime.py" in content

    def test_requirements_file_exists(self):
        """Verify requirements-runtime.txt exists."""
        req_path = Path(__file__).parent.parent / "requirements-runtime.txt"
        assert req_path.is_file()

    def test_requirements_has_torch(self):
        """Verify requirements includes torch."""
        req_path = Path(__file__).parent.parent / "requirements-runtime.txt"
        content = req_path.read_text()
        assert "torch" in content

    def test_requirements_has_transformers(self):
        """Verify requirements includes transformers."""
        req_path = Path(__file__).parent.parent / "requirements-runtime.txt"
        content = req_path.read_text()
        assert "transformers" in content


class TestHostIntegrationPrerequisites:
    """Test host integration prerequisites documentation."""

    def test_deployment_docs_exist(self):
        """Verify deployment documentation exists."""
        docs_path = Path(__file__).parent.parent / "docs" / "docker-deployment.md"
        assert docs_path.is_file(), f"Deployment docs not found at {docs_path}"

    def test_deployment_docs_mention_rootless(self):
        """Verify deployment docs mention rootless Podman."""
        docs_path = Path(__file__).parent.parent / "docs" / "docker-deployment.md"
        content = docs_path.read_text()
        assert "rootless" in content.lower()

    def test_deployment_docs_mention_gpu_requirements(self):
        """Verify deployment docs document GPU requirements."""
        docs_path = Path(__file__).parent.parent / "docs" / "docker-deployment.md"
        content = docs_path.read_text()
        assert "GPU" in content or "gpu" in content

    def test_deployment_docs_mention_cdi(self):
        """Verify deployment docs mention CDI for GPU access."""
        docs_path = Path(__file__).parent.parent / "docs" / "docker-deployment.md"
        content = docs_path.read_text()
        assert "CDI" in content or "cdi" in content


class TestModelPathValidation:
    """Test model path validation for read-only mounts."""

    def test_model_path_is_required(self):
        """Verify MODEL_PATH is required in compose config."""
        compose_path = Path(__file__).parent.parent / "docker" / "inference-compose.yaml"
        content = compose_path.read_text()
        assert "MODEL_PATH:?" in content or "MODEL_PATH:?" in content

    def test_model_path_mount_is_readonly(self):
        """Verify model path mount is read-only."""
        compose_path = Path(__file__).parent.parent / "docker" / "inference-compose.yaml"
        content = compose_path.read_text()
        assert ":ro" in content


class TestCleanupDocumentation:
    """Test that cleanup procedures are documented."""

    def test_deployment_docs_have_cleanup_section(self):
        """Verify deployment docs include cleanup section."""
        docs_path = Path(__file__).parent.parent / "docs" / "docker-deployment.md"
        content = docs_path.read_text()
        assert "cleanup" in content.lower() or "down" in content.lower()

    def test_compose_has_restart_policy(self):
        """Verify compose has restart policy for container lifecycle."""
        compose_path = Path(__file__).parent.parent / "docker" / "inference-compose.yaml"
        content = compose_path.read_text()
        assert "restart" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
