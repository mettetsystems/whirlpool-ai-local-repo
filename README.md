# Whirlpool AI

A PyQt desktop application for downloading Hugging Face models, browsing their
local files and published cards, starting a local Transformers inference server,
and fine-tuning local causal language models with QLoRA.

**Optimized for Fedora 40+, with Fedora 44 as the current development host.**
The app can run on other operating systems with compatible Python/Qt and a
container engine; credential integration currently requires Linux Secret Service.
Other operating systems are not covered by the Fedora verification checks.

Fedora deployments prefer rootless Podman, NVIDIA CDI devices for GPU access,
and SELinux-labelled read-only model mounts. See [deployment instructions](docs/docker-deployment.md).
The inference container uses its own Python 3.12 environment, independently of
the host distribution's Python version.

## Install and launch

Use Python 3.10+ for the desktop app. Python 3.12 is recommended for the optional
ML dependencies; the container uses Python 3.12. The desktop tests were run on
Python 3.14. The original Python 3.8 target is not currently supported.

```bash
make setup
# Only on first setup, if config/hf_auth.json does not already exist:
cp -n config/hf_auth.example.json config/hf_auth.json
make run
```

Run from the repository root. Open **Settings** to set your Hugging Face token
and default folders. Tokens are stored in the Fedora desktop keyring (GNOME
Keyring via Secret Service), not in the configuration file. `HF_TOKEN` in the
environment overrides the saved token; the UI shows when this applies.
An alternate config can be supplied with:

```bash
.venv/bin/python -m src --config /absolute/path/to/config.json
```

Paste `organization/model` into Download Model and click Download. Downloaded
models appear in the list with their local paths. Selecting a model expands its
metadata and displays its saved README, without a network request. Download,
container build, training, and Hub search run in background threads; closing is deferred
while an operation is active. Pause downloads before closing the app.

Already have a model on disk? Click **Add Local Model** and select its source
directory (the folder containing `config.json` or a GGUF file). The app registers
the existing path without copying or downloading files, then selects it in the
model list. Removing an added local model from the list preserves its source
files. GGUF folders can be catalogued, but the bundled Transformers runtime does
not run GGUF models.

## Download progress, resume, and batches

- **Download Model** searches Hugging Face asynchronously. Click a result once to
  select that exact model, then **Download Selected Model**. You can also paste
  an exact ID directly. Searching never downloads all results.
- **Batch Download** is a separate toolbar button. Check individual results,
  optionally search again to add more, then click **Download Checked Models**.
  Checks persist across searches; Clear batch selection resets them. You can
  also add exact IDs directly. Each batch allows up to 10 unique models and runs
  sequentially; failures do not prevent the other checked models from downloading.
- The **Downloads** tab shows each model's status, percentage, received bytes,
  transfer rate, ETA, current file, and file counts. Rates are averages for the
  current transfer process; cached data can make progress jump forward. Errors
  appear on the affected row (hover for the full message).
- **Pause downloads** stops the current transfer and leaves remaining models
  queued. Select paused/failed/queued rows and click **Resume Selected**. Hold
  Ctrl/Shift to select several rows. Pause can wait for metadata requests to
  finish; transfer workers are stopped once that phase completes.
- Download records survive restarts. An unexpected exit appears as interrupted
  and can be resumed. The revision is pinned, completed files are reused, and
  Hugging Face resumes cached chunks when the transfer backend supports it.
  Received bytes not yet flushed to disk may need to be transferred again.

State lives under the selected model directory's `.downloads/` folder and contains
no credentials. Keep the downloaded folder and its `.cache/huggingface` contents
for resume to work. Tokens reach disposable transfer workers over stdin, never
through command-line arguments or state files. Unknown partial directories from
older app versions remain protected from automatic overwrite.

The byte-progress adapter is tested against `huggingface_hub` 0.36.x, which is
pinned in `requirements.txt`. After upgrading, install dependencies with
`.venv/bin/python -m pip install -r requirements.txt`.

## Model storage quota

The download quota is **1 TiB (1,099,511,627,776 bytes)**. Before a download, the
client reads file sizes from Hugging Face, resolves the revision, and rejects a
snapshot that would exceed the quota. Missing size metadata fails explicitly.
Downloads are pinned to the checked revision. Partial downloads and registered
external model files count toward usage; overlapping registrations are counted
once. A pinned resume reserves only the remaining snapshot bytes, accounting for completed files and retained partial chunks.

This is an application preflight check, not a filesystem quota: avoid concurrent
app instances writing to the same repository. External writes and download-cache
overhead are not constrained by this check. Training outputs are separate.
New downloads use a hash of the full model ID; existing paths stay unchanged.
Legacy shared folders cannot be overwritten by re-download, and deleting a
registration preserves files still referenced by another model.

## Settings

The **Settings** tab saves defaults for:

- Hugging Face authentication, with a masked token field and Show token toggle.
- The destination for new model downloads.
- Training runs and adapters (each new run gets its own directory).
- Generated inference deployment files.
- Exported inference image archives and the container image name/tag.

Click **Save Settings** to apply these defaults to new operations. Existing model
registrations remain visible when you change the download destination; files are
not moved. Settings cannot change while a download, build, or training job runs.

Tokens use the explicit Secret Service backend under the service **Whirlpool AI**,
with an account scoped to the configuration file's absolute path. There is no
plaintext fallback. A legacy `hf_token` field is migrated into the keyring and
removed from the config when you save successfully. If the keyring is locked or
unavailable, token saving fails and the existing config remains unchanged.
Clearing the token field and saving replaces the saved credential with an empty
value. Run the app inside your logged-in GNOME desktop session and unlock the
keyring if prompted. Python dependencies are included in `requirements.txt`.

Docker/Podman manages built image storage. Enable **Export a .tar image archive**
to save a portable image archive in your selected directory after a successful
build and inference smoke test. This does not push images to a remote registry.
The base model files remain separate from the exported runtime image.

Validation for this addition: 131 portable tests passed; the desktop keyring was
verified by writing, reading, and removing a unique temporary test credential.
No existing credentials were accessed by that integration check.

## Local inference

Select a downloaded **Transformers causal language model** and click **Build
Inference Server**. Review the device recommendation, choose CPU/CUDA/auto and
a local port, then build. This writes `generated-inference.json` in your configured inference deployment
folder, builds
the included Dockerfile, starts the service, waits for readiness, and runs three
small generation requests. The initial image build requires internet access.
Once dependencies and model files are local, inference loads with
`local_files_only=True`, `HF_HUB_OFFLINE=1`, and remote model code disabled.

The service binds to localhost, mounts the model read-only, and exposes:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Hello","max_new_tokens":32}'
```

Use your installed Compose engine to inspect or stop the generated service:

```bash
podman-compose -f inference_output/generated-inference.json logs
podman-compose -f inference_output/generated-inference.json down
# Docker alternative:
docker compose -f inference_output/generated-inference.json down
```

The builder prefers podman-compose, then Docker Compose. GPU containers require
host GPU drivers and a correctly configured NVIDIA container runtime. CPU builds
do not request GPU devices. The static `docker/inference-compose.yaml` is a GPU
example: set MODEL_PATH to the absolute directory of one downloaded model.

The bundled server processes requests serially. It does not implement vLLM,
llama.cpp/GGUF, OpenLLM, embeddings, multimodal inference, or adapter serving.
Quantization overrides and vLLM requests fail explicitly. Large models may not
fit the selected device; metadata-based device recommendations are only hints.
The model directory is mounted, not embedded into the image; copy both to move
an offline deployment. This is a local development server, not a public API.

## Quick Train

Install optional components into an environment with compatible CUDA PyTorch:

```bash
.venv/bin/python -m pip install -r requirements-training.txt
# Only when using S3 or OCI:
.venv/bin/python -m pip install -r requirements-cloud.txt
# Only when parsing documents/images:
.venv/bin/python -m pip install -r requirements-documents.txt
```

Select a local base model, click **Quick Train**, choose a data source and a new
output directory, then start. QLoRA requires a supported NVIDIA CUDA GPU. The
default LoRA target modules suit Llama-like models; other architectures may
require changing TrainingConfig in Python. QLoRA produces an adapter, not a
standalone replacement for the base model.

- **Local:** enter a folder containing text, Markdown, JSON or JSONL files.
  Use **Browse…** next to Folder / prefix to choose the source directory.
- **S3:** enter a bucket and optional prefix; use standard AWS credentials.
- **OCI:** enter a bucket, namespace and optional prefix; configure `~/.oci/config`.
- JSON/JSONL records must contain a string `text` field. Supply at least two
  nonempty records. Text/Markdown files each become one record.
- PDF, DOCX, PPTX, HTML and supported image formats use optional Docling.
  Docling may download its own parsing models on first use; pre-cache them for
  disconnected use. Video parsing is not implemented and unsupported files fail
  with an explicit error.

All sources are copied/downloaded locally and normalized to
`prepared_data/train.jsonl`. Output also includes `source_data/`, training
checkpoints, `adapter/`, and `training_state.json`. Reusing a populated source-data
output is rejected to prevent mixing different runs. Parsing currently holds
normalized records in memory, so use modest datasets.

## Validation

```bash
make test   # portable code/UI tests, no model download or container engine needed
make smoke  # opens and closes an isolated offscreen desktop window
.venv/bin/python -m pytest -q -m container  # host Podman prerequisite checks
```

Current repair validation: **116 portable tests and 2 host prerequisite tests
passed**, plus the desktop launch smoke test and Python compilation. ML/cloud
boundaries use mocks in unit tests. Real model image builds, generation, CUDA
training, cloud credentials, Docling parsing, and cross-platform installs have
not been demonstrated in this repair.

## Review fixes verified September 23, 2026

- 236 portable tests passed, including directory collision/deletion protection,
  quota rejection before download, strict OCI method signatures and streaming,
  launcher environment handling, and Fedora Podman CDI configuration.
- Desktop launch smoke test passed.
- The inference Dockerfile built successfully using rootless Podman as
  `localhost/whirlpool-inference:review-fixes`.
- With networking disabled, that image loaded a tiny locally created causal model,
  served its health endpoint, and completed three real CPU generation requests.
- Generated Docker and Podman Compose configurations passed schema validation.

GPU execution and real OCI account access were not exercised. The 1 TiB limit is
validated with scaled fixtures; the quota test now writes only 1 KiB rather than
allocating/writing 11 GiB. Host/container prerequisite tests are separate from
the portable test count.

## Remaining original product scope

This is a repair of the core workflows, not the complete original product.
Agent/tool/harness construction, video parsing, offline synchronization conflict
resolution, exact 30-day log retention, and alternative
inference engines remain unimplemented. Generated container logs have size-based
rotation. Training dependencies are optional and version-ranged, not a fully
locked cross-platform environment. Hardware/model compatibility needs real
workload validation. The single inference service name means building another
model replaces the configured service.

API references used for the training repair:
[Transformers Trainer](https://huggingface.co/docs/transformers/main_classes/trainer)
and [PEFT model parameter counts](https://huggingface.co/docs/peft/package_reference/peft_model).

Model cards use a readable documentation view that hides Hub YAML metadata and badge/image placeholders while preserving headings, tables, and code examples. Use **Show original model card source** to inspect the complete unmodified README.
