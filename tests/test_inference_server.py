"""Tests for inference server Podman runtime validation."""
import subprocess
import pytest
import yaml
import os


@pytest.mark.container
class TestPodmanRuntime:
    """Test Podman container engine functionality."""

    def test_podman_api_accessible(self):
        """Verify Podman API is accessible."""
        result = subprocess.run(
            ["podman", "info"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, f"Podman API not accessible: {result.stderr}"

    def test_podman_compose_available(self):
        """Verify podman-compose is available."""
        result = subprocess.run(
            ["podman-compose", "--version"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, f"podman-compose not available: {result.stderr}"


class TestInferenceComposeConfig:
    """Test inference-compose.yaml configuration validity."""

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

    def test_compose_version_valid(self, compose_config):
        """Verify compose file has valid version."""
        assert "version" in compose_config
        assert compose_config["version"] in ["3.8", "3.9"]

    def test_inference_server_service_exists(self, compose_config):
        """Verify inference-server service is defined."""
        assert "services" in compose_config
        assert "inference-server" in compose_config["services"]

    def test_inference_server_has_image(self, compose_config):
        """Verify inference-server has image configured."""
        service = compose_config["services"]["inference-server"]
        assert "image" in service

    def test_inference_server_has_ports(self, compose_config):
        """Verify inference-server has ports configured."""
        service = compose_config["services"]["inference-server"]
        assert "ports" in service
        assert len(service["ports"]) > 0

    def test_inference_server_has_environment(self, compose_config):
        """Verify inference-server has environment variables."""
        service = compose_config["services"]["inference-server"]
        assert "environment" in service
        assert len(service["environment"]) > 0

    def test_inference_server_has_volumes(self, compose_config):
        """Verify inference-server has volumes configured."""
        service = compose_config["services"]["inference-server"]
        assert "volumes" in service

    def test_inference_server_has_healthcheck(self, compose_config):
        """Verify inference-server has healthcheck configured."""
        service = compose_config["services"]["inference-server"]
        assert "healthcheck" in service
        healthcheck = service["healthcheck"]
        assert "test" in healthcheck
        assert "interval" in healthcheck
        assert "timeout" in healthcheck
        assert "retries" in healthcheck

    def test_inference_server_has_restart_policy(self, compose_config):
        """Verify inference-server has restart policy."""
        service = compose_config["services"]["inference-server"]
        assert "restart" in service

    def test_volumes_defined(self, compose_config):
        """Verify volumes are defined at root level."""
        assert "volumes" in compose_config
        assert "models" in compose_config["volumes"]
        assert "logs" in compose_config["volumes"]

    def test_gpu_resources_configured(self, compose_config):
        """Verify GPU resources are configured."""
        service = compose_config["services"]["inference-server"]
        assert "deploy" in service
        assert "resources" in service["deploy"]
        assert "reservations" in service["deploy"]["resources"]
        assert "devices" in service["deploy"]["resources"]["reservations"]


class TestInferenceServerEnvironmentVariables:
    """Test inference server environment variable configuration."""

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

    def test_model_id_env_var(self, compose_config):
        """Verify MODEL_ID environment variable is configured."""
        env = compose_config["services"]["inference-server"]["environment"]
        assert any("MODEL_ID" in e for e in env)

    def test_model_path_env_var(self, compose_config):
        """Verify MODEL_PATH environment variable is configured."""
        env = compose_config["services"]["inference-server"]["environment"]
        assert any("MODEL_PATH" in e for e in env)

    def test_device_env_var(self, compose_config):
        """Verify DEVICE environment variable is configured."""
        env = compose_config["services"]["inference-server"]["environment"]
        assert any("DEVICE" in e for e in env)

    def test_max_batch_size_env_var(self, compose_config):
        """Verify MAX_BATCH_SIZE environment variable is configured."""
        env = compose_config["services"]["inference-server"]["environment"]
        assert any("MAX_BATCH_SIZE" in e for e in env)

    def test_max_seq_len_env_var(self, compose_config):
        """Verify MAX_SEQ_LEN environment variable is configured."""
        env = compose_config["services"]["inference-server"]["environment"]
        assert any("MAX_SEQ_LEN" in e for e in env)

    def test_quantization_env_var(self, compose_config):
        """Verify QUANTIZATION environment variable is configured."""
        env = compose_config["services"]["inference-server"]["environment"]
        assert any("QUANTIZATION" in e for e in env)

    def test_vllm_enabled_env_var(self, compose_config):
        """Verify VLLM_ENABLED environment variable is configured."""
        env = compose_config["services"]["inference-server"]["environment"]
        assert any("VLLM_ENABLED" in e for e in env)

    def test_gpu_count_env_var(self, compose_config):
        """Verify GPU_COUNT environment variable is configured."""
        env = compose_config["services"]["inference-server"]["environment"]
        assert any("GPU_COUNT" in e for e in env)

    def test_host_port_env_var(self, compose_config):
        """Verify HOST_PORT environment variable is configured."""
        service = compose_config["services"]["inference-server"]
        ports = service["ports"]
        assert any("HOST_PORT" in p for p in ports)

    def test_container_name_env_var(self, compose_config):
        """Verify CONTAINER_NAME environment variable is configured."""
        service = compose_config["services"]["inference-server"]
        assert "container_name" in service
        assert "CONTAINER_NAME" in service["container_name"]

    def test_inference_image_env_var(self, compose_config):
        """Verify INFERENCE_IMAGE environment variable is configured."""
        service = compose_config["services"]["inference-server"]
        assert "image" in service
        assert "INFERENCE_IMAGE" in service["image"]

    def test_log_dir_env_var(self, compose_config):
        """Verify LOG_DIR environment variable is configured."""
        service = compose_config["services"]["inference-server"]
        volumes = service["volumes"]
        assert any("LOG_DIR" in v for v in volumes)

    def test_gpu_driver_env_var(self, compose_config):
        """Verify GPU_DRIVER environment variable is configured."""
        service = compose_config["services"]["inference-server"]
        deploy = service["deploy"]
        resources = deploy["resources"]
        reservations = resources["reservations"]
        devices = reservations["devices"]
        assert any("GPU_DRIVER" in str(d) for d in devices)


class TestInferenceServerDefaults:
    """Test inference server default values."""

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

    def test_default_host_port(self, compose_config):
        """Verify default HOST_PORT is 8000."""
        service = compose_config["services"]["inference-server"]
        ports = service["ports"]
        assert any("8000" in p for p in ports)

    def test_default_container_name(self, compose_config):
        """Verify default CONTAINER_NAME is inference-server."""
        service = compose_config["services"]["inference-server"]
        assert "inference-server" in service["container_name"]

    def test_default_device(self, compose_config):
        """Verify default DEVICE is auto."""
        env = compose_config["services"]["inference-server"]["environment"]
        assert any("DEVICE:-auto" in e for e in env)

    def test_default_max_batch_size(self, compose_config):
        """Verify default MAX_BATCH_SIZE is 1."""
        env = compose_config["services"]["inference-server"]["environment"]
        assert any("MAX_BATCH_SIZE:-1" in e for e in env)

    def test_default_max_seq_len(self, compose_config):
        """Verify default MAX_SEQ_LEN is 2048."""
        env = compose_config["services"]["inference-server"]["environment"]
        assert any("MAX_SEQ_LEN:-2048" in e for e in env)

    def test_default_quantization(self, compose_config):
        """Verify default QUANTIZATION is none."""
        env = compose_config["services"]["inference-server"]["environment"]
        assert any("QUANTIZATION:-none" in e for e in env)

    def test_default_vllm_enabled(self, compose_config):
        """Verify default VLLM_ENABLED is false."""
        env = compose_config["services"]["inference-server"]["environment"]
        assert any("VLLM_ENABLED:-false" in e for e in env)

    def test_default_gpu_count(self, compose_config):
        """Verify default GPU_COUNT is 1."""
        env = compose_config["services"]["inference-server"]["environment"]
        assert any("GPU_COUNT:-1" in e for e in env)

    def test_default_gpu_driver(self, compose_config):
        """Verify default GPU_DRIVER is nvidia."""
        service = compose_config["services"]["inference-server"]
        deploy = service["deploy"]
        resources = deploy["resources"]
        reservations = resources["reservations"]
        devices = reservations["devices"]
        assert any("nvidia" in str(d) for d in devices)

    def test_default_model_path(self, compose_config):
        """Verify default MODEL_PATH includes /models/."""
        service = compose_config["services"]["inference-server"]
        volumes = service["volumes"]
        assert any("/models" in v for v in volumes)

    def test_default_log_dir(self, compose_config):
        """Verify default LOG_DIR includes ./logs."""
        service = compose_config["services"]["inference-server"]
        volumes = service["volumes"]
        assert any("./logs" in v for v in volumes)

    def test_healthcheck_url(self, compose_config):
        """Verify healthcheck uses correct URL."""
        service = compose_config["services"]["inference-server"]
        healthcheck = service["healthcheck"]
        test = healthcheck["test"]
        assert "localhost:8000" in " ".join(test)
        assert "/health" in " ".join(test)

    def test_healthcheck_interval(self, compose_config):
        """Verify healthcheck interval is 30s."""
        service = compose_config["services"]["inference-server"]
        healthcheck = service["healthcheck"]
        assert healthcheck["interval"] == "30s"

    def test_healthcheck_timeout(self, compose_config):
        """Verify healthcheck timeout is 10s."""
        service = compose_config["services"]["inference-server"]
        healthcheck = service["healthcheck"]
        assert healthcheck["timeout"] == "10s"

    def test_healthcheck_retries(self, compose_config):
        """Verify healthcheck retries is 3."""
        service = compose_config["services"]["inference-server"]
        healthcheck = service["healthcheck"]
        assert healthcheck["retries"] == 3

    def test_healthcheck_start_period(self, compose_config):
        """Verify healthcheck start_period is 60s."""
        service = compose_config["services"]["inference-server"]
        healthcheck = service["healthcheck"]
        assert healthcheck["start_period"] == "60s"

    def test_restart_policy(self, compose_config):
        """Verify restart policy is unless-stopped."""
        service = compose_config["services"]["inference-server"]
        assert service["restart"] == "unless-stopped"


class TestInferenceServerCapabilities:
    """Test inference server capability configuration."""

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

    def test_gpu_driver_capability(self, compose_config):
        """Verify GPU driver capability is configured."""
        service = compose_config["services"]["inference-server"]
        deploy = service["deploy"]
        resources = deploy["resources"]
        reservations = resources["reservations"]
        devices = reservations["devices"]
        assert any("gpu" in str(d).lower() for d in devices)

    def test_gpu_count_reservation(self, compose_config):
        """Verify GPU count reservation is configured."""
        service = compose_config["services"]["inference-server"]
        deploy = service["deploy"]
        resources = deploy["resources"]
        reservations = resources["reservations"]
        devices = reservations["devices"]
        assert any("count" in str(d) for d in devices)

    def test_model_volume_mount(self, compose_config):
        """Verify model volume mount is configured."""
        service = compose_config["services"]["inference-server"]
        volumes = service["volumes"]
        assert any("/models" in v for v in volumes)

    def test_log_volume_mount(self, compose_config):
        """Verify log volume mount is configured."""
        service = compose_config["services"]["inference-server"]
        volumes = service["volumes"]
        assert any("/logs" in v for v in volumes)

    def test_port_mapping(self, compose_config):
        """Verify port mapping is configured."""
        service = compose_config["services"]["inference-server"]
        ports = service["ports"]
        assert any(":8000" in p for p in ports)


class TestInferenceServerSecurity:
    """Test inference server security configuration."""

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

    def test_container_name_is_set(self, compose_config):
        """Verify container name is explicitly set."""
        service = compose_config["services"]["inference-server"]
        assert "container_name" in service

    def test_healthcheck_provides_liveness(self, compose_config):
        """Verify healthcheck provides liveness probe."""
        service = compose_config["services"]["inference-server"]
        healthcheck = service["healthcheck"]
        assert "urllib.request" in " ".join(healthcheck["test"])
        assert "http" in " ".join(healthcheck["test"])

    def test_healthcheck_provides_readiness(self, compose_config):
        """Verify healthcheck provides readiness probe."""
        service = compose_config["services"]["inference-server"]
        healthcheck = service["healthcheck"]
        assert "urlopen" in " ".join(healthcheck["test"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
