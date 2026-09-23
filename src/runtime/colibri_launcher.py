#!/usr/bin/env python3
"""Colibri runtime launcher with Docker/Podman support."""

import os
import subprocess
import sys
import shutil
from pathlib import Path


def detect_container_runtime():
    """Detect available container runtime (Docker or Podman)."""
    for runtime in ("podman", "docker"):
        if shutil.which(runtime):
            return runtime
    return None


def run_command(cmd, cwd=None, env=None):
    """Execute a command and return the result."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            timeout=1800,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {' '.join(cmd)}")
        print(f"stderr: {e.stderr}")
        raise


def build_image(runtime, image_name, dockerfile_path, context_path):
    """Build the Colibri runtime image."""
    cmd = [
        runtime,
        "build",
        "-t",
        image_name,
        "-f",
        dockerfile_path,
        context_path
    ]
    return run_command(cmd)


def run_container(runtime, compose_file, env_vars=None):
    """Run the Colibri inference server using compose."""
    if runtime not in ("podman", "docker"):
        raise ValueError("Runtime must be podman or docker")
    compose = ["podman-compose"] if runtime == "podman" and shutil.which("podman-compose") else [runtime, "compose"]
    environment = {**os.environ, **(env_vars or {})}
    return run_command(compose + ["-f", str(compose_file), "up", "--detach"], env=environment)


def health_check(runtime, host="localhost", port=8000):
    """Check the health endpoint of the running service."""
    import urllib.request
    url = f"http://{host}:{port}/healthz"
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            if response.status == 200:
                return True
    except Exception:
        pass
    return False


def launch_colibri(runtime=None, image_name="colibri-runtime",
                   compose_file=None, env_vars=None):
    """Launch the Colibri runtime with automatic runtime detection."""
    if runtime is None:
        runtime = detect_container_runtime()

    if runtime is None:
        raise RuntimeError("No container runtime found (Docker or Podman)")

    project_root = Path(__file__).parent.parent.parent
    dockerfile_path = project_root / "docker" / "Dockerfile"
    environment = {**os.environ, **(env_vars or {}), "INFERENCE_IMAGE": image_name}
    if compose_file is None:
        from src.inference.server_builder import InferenceConfig, InferenceServerBuilder
        if not environment.get("MODEL_PATH"):
            raise ValueError("Set MODEL_PATH to a downloaded model directory")
        config = InferenceConfig(
            model_id=environment.get("MODEL_ID", "local"), model_path=environment["MODEL_PATH"],
            inference_image=image_name, container_runtime=runtime,
            device=environment.get("DEVICE", "cpu"), host_port=int(environment.get("HOST_PORT", "8000")),
            gpu_count=int(environment.get("GPU_COUNT", "1")),
        )
        compose_file = project_root / "inference_output" / "colibri.json"
        InferenceServerBuilder().save_config(config, str(compose_file))
    compose_file = Path(compose_file)

    if not dockerfile_path.exists():
        raise FileNotFoundError(f"Dockerfile not found: {dockerfile_path}")

    if not compose_file.exists():
        raise FileNotFoundError(f"Compose file not found: {compose_file}")

    print(f"Using container runtime: {runtime}")
    print(f"Building image: {image_name}")

    build_image(runtime, image_name, str(dockerfile_path), str(project_root))

    print(f"Starting container from: {compose_file}")
    run_container(runtime, str(compose_file), environment)

    print("Waiting for health check...")
    for _ in range(30):
        if health_check(runtime, port=int(environment.get("HOST_PORT", os.environ.get("HOST_PORT", "8000")))):
            print("Service is healthy and ready")
            return True
        import time
        time.sleep(2)

    raise RuntimeError("Service failed health check")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Launch Colibri runtime")
    parser.add_argument("--runtime", choices=["docker", "podman"],
                        help="Container runtime to use")
    parser.add_argument("--image", default="colibri-runtime",
                        help="Image name")
    parser.add_argument("--compose", help="Compose file path")
    args = parser.parse_args()

    launch_colibri(
        runtime=args.runtime,
        image_name=args.image,
        compose_file=args.compose
    )
