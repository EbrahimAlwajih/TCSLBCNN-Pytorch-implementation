# TCSLBCNN (PyTorch)

PyTorch implementation of **TCS-LBCNN** with a modernized repository layout (packaged module, tests, CI, and Docker support).

## Features

- Train and evaluate on:
  - **MNIST**
  - **CIFAR10** (optionally resized to 28x28 as configured in code)
- Reproducible training artifacts:
  - Checkpoints under `artifacts/models/`
  - TensorBoard runs under `artifacts/runs/`
- Developer tooling:
  - Ruff linting + formatting
  - Pytest unit tests
  - GitHub Actions workflows (Quality Gate, Security Scan, Container Build, Release, GHCR publish, Coverage)

---

## Repository structure

- `src/tcslbcnn/` — installable Python package
- `tests/` — unit tests
- `scripts/` — developer helper scripts
- `artifacts/` — runtime outputs (created at runtime; not committed)

---

## Local setup (Windows)

> Use the venv Python explicitly to avoid mixing with Anaconda/system Python.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
````

### Run lint + tests

```powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest -q
```

### Dev helper script (Windows)

```powershell
.\scripts\dev.ps1 -Task lint
.\scripts\dev.ps1 -Task test
.\scripts\dev.ps1 -Task smoke
```

Linux/macOS:

```bash
./scripts/dev.sh lint
./scripts/dev.sh test
./scripts/dev.sh smoke
```

---

## Training and evaluation

### Train (MNIST smoke example)

```powershell
.\.venv\Scripts\python.exe -c "from tcslbcnn.training import train; train(dataset='mnist', n_epochs=1, batch_size=64, use_compile=False)"
```

### Train (CIFAR10 example)

```powershell
.\.venv\Scripts\python.exe -c "from tcslbcnn.training import train; train(dataset='cifar10', n_epochs=1, batch_size=64, use_compile=False)"
```

### Evaluate a checkpoint

```powershell
.\.venv\Scripts\python.exe -c "from tcslbcnn.training import test; test(checkpoint_path='artifacts/models/tcslbcnn_best.pt')"
```

---

## TensorBoard

```powershell
.\.venv\Scripts\python.exe -m tensorboard --logdir artifacts/runs
```

---

## Docker (local build)

### Build image

```powershell
docker build -t tcslbcnn:local .
```

### Verify package import inside the container

```powershell
docker run --rm tcslbcnn:local python -c "import tcslbcnn; print('tcslbcnn ok')"
```

### Run unit tests in the container (CI-like)

```powershell
docker run --rm tcslbcnn:local python -m pytest -q
```

### Training smoke in the container (downloads MNIST inside the container)

```powershell
docker run --rm tcslbcnn:local python -c "from tcslbcnn.training import train; train(dataset='mnist', n_epochs=1, batch_size=64, use_compile=False)"
```

### Persist data and artifacts (recommended)

```powershell
mkdir .\data -Force | Out-Null
mkdir .\artifacts -Force | Out-Null

docker run --rm `
  -v "${PWD}\data:/app/data" `
  -v "${PWD}\artifacts:/app/artifacts" `
  tcslbcnn:local `
  python -c "from tcslbcnn.training import train; train(dataset='mnist', n_epochs=1, batch_size=64, run_dir='artifacts/runs', checkpoint_path='artifacts/models/tcslbcnn_best.pt', use_compile=False)"
```

> If your Dockerfile uses a different `WORKDIR` than `/app`, adjust the mount paths accordingly.

---

## GHCR (published container)

After the **Container Publish** workflow runs, pull and run the published image:

```powershell
docker pull ghcr.io/ebrahimalwajih/tcslbcnn-pytorch-implementation:latest
docker run --rm ghcr.io/ebrahimalwajih/tcslbcnn-pytorch-implementation:latest python -m pytest -q
```

---

## Release workflow (tags)

To create a release, tag and push:

```powershell
git tag v0.1.0
git push origin v0.1.0
```

This triggers the Release workflow to publish a GitHub Release and attach build artifacts.

---

## Troubleshooting

### `pytest` can’t import `tcslbcnn`

You are likely running pytest using a different Python (e.g., Anaconda). Use the venv Python:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pytest -q
```

### PowerShell cannot activate venv (ExecutionPolicy)

Instead of activating, use the venv python directly:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

---

## License

MIT (see `LICENSE` if present).
