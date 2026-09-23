# Docker/Podman deployment

Whirlpool is optimized for **Fedora 40+**, with Fedora 44 as the development
host. Other systems with compatible Python/Qt and container engines can run the
app, but platform-specific support varies. The container carries Python 3.12,
so it does not depend on the host Python version or Ubuntu package repositories.

## Fedora setup

Install Podman and its Compose provider using Fedora packages:

```bash
sudo dnf install podman podman-compose
```

Run the application as your desktop user, without sudo. The GUI prefers
podman-compose and falls back to Docker Compose. Tokens use GNOME Keyring.

For NVIDIA GPUs, install a compatible host driver and NVIDIA Container Toolkit,
then check the available CDI devices:

```bash
nvidia-smi
nvidia-ctk cdi list
```

The generated Podman configuration requests `nvidia.com/gpu=0` (and additional
indices for multiple GPUs). Docker uses NVIDIA GPU resource reservations.
CPU mode requests neither. The host driver must support the CUDA version in
the installed PyTorch wheel; the requirements currently permit new wheel releases.
NVIDIA's [CDI documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html)
covers device setup. SELinux policy must also permit GPU device access; the app
does not disable SELinux or change host policy automatically.

Generated model mounts are read-only and use shared SELinux relabeling (`z`).
Select a dedicated model folder, since relabeling applies to that folder's files.

## Build and launch

The recommended path is **Models → Build Inference Server** in the UI. It
uses saved defaults, detects the engine, creates the appropriate Compose file,
waits for health, and runs three short generation requests.

Standalone launcher (from the repository root):

```bash
MODEL_PATH=/absolute/path/to/model DEVICE=cpu \
  .venv/bin/python -m src.runtime.colibri_launcher --runtime podman \
  --image whirlpool/inference-server:latest
```

Use `DEVICE=cuda` for GPU inference, or `--runtime docker` for Docker. The launcher
writes `inference_output/colibri.json` with engine-specific GPU settings. `HOST_PORT`
sets the port and health-check target. `--compose` accepts a custom Compose file;
that file must honor INFERENCE_IMAGE and include appropriate model mounts and
GPU settings for the selected engine.

Manual image build (the final dot is the repository root build context):

```bash
podman build -t whirlpool/inference-server:latest -f docker/Dockerfile .
# Docker alternative:
docker build -t whirlpool/inference-server:latest -f docker/Dockerfile .
```

The static `docker/inference-compose.yaml` remains a **Docker/NVIDIA GPU example**.
Use the GUI or launcher for CPU-only or rootless Podman deployment. Building
requires network access for the base image and dependencies; the loaded model
runs offline once available locally. Both /health and /healthz report readiness.

## Logs and cleanup

```bash
podman-compose -f inference_output/colibri.json logs
podman-compose -f inference_output/colibri.json down
```

For GUI deployments substitute your configured deployment folder and
`generated-inference.json`. For Docker use `docker compose`. Stopping containers
does not remove models or exported archives. Inspect logs if readiness times out.
