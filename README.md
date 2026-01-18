# TCSLBCNN (PyTorch)

A PyTorch implementation of **TCS-LBCNN**, a lightweight convolutional neural network that uses fixed sparse local binary pattern–inspired filters combined with learnable 1×1 convolutions.

This repository provides:
- A clean, installable Python package
- Ready-to-use Docker images
- Reproducible training and evaluation workflows

---

## What is TCS-LBCNN?

TCS-LBCNN replaces standard 3×3 convolution kernels with **fixed, sparse, directional contrast filters** (two non-zero weights per kernel) followed by learnable channel mixing.

**Benefits**
- Fewer trainable parameters
- Strong inductive bias for local texture and edge patterns
- Suitable for lightweight or constrained environments

---

## Supported datasets

- **MNIST**
- **CIFAR-10** (optionally resized to 28×28 as configured)

---

## Quick start (Docker – recommended)

The easiest way to use this project is via Docker.
No Python installation required.

### Pull the latest image

```bash
docker pull ghcr.io/ebrahimalwajih/tcslbcnn-pytorch-implementation:latest
````

### Verify installation

```bash
docker run --rm ghcr.io/ebrahimalwajih/tcslbcnn-pytorch-implementation:latest \
  python -c "import tcslbcnn; print('tcslbcnn ready')"
```

### Train (MNIST example)

```bash
docker run --rm \
  ghcr.io/ebrahimalwajih/tcslbcnn-pytorch-implementation:latest \
  python -c "from tcslbcnn.training import train; train(dataset='mnist', n_epochs=1, batch_size=64, use_compile=False)"
```

---

## Running locally (Python)

### Requirements

* Python ≥ 3.10
* pip

### Install

```bash
pip install -r requirements.txt
pip install -e .
```

### Train

```bash
python -c "from tcslbcnn.training import train; train(dataset='mnist', n_epochs=1, batch_size=64, use_compile=False)"
```

---

## Output artifacts

During training, the following directories are created:

* `artifacts/models/` — model checkpoints
* `artifacts/runs/` — TensorBoard logs

To view training metrics:

```bash
tensorboard --logdir artifacts/runs
```

---

## Project structure

```
src/tcslbcnn/      Core Python package
tests/             Unit tests
artifacts/          Training outputs (not committed)
```

---

## Reproducibility

* Fixed initialization for TCS-LBP filters
* Deterministic training options
* Docker images ensure environment consistency

---

## License

MIT License.

---

## Citation

If you use this implementation in academic or research work, please cite the following paper:

> **E. Al-wajih and R. Ghazali**,
> *Threshold center-symmetric local binary convolutional neural networks for bilingual handwritten digit recognition*,
> **Knowledge-Based Systems**, vol. 259, Article 110079, 2023.
> https://doi.org/10.1016/j.knosys.2022.110079

### BibTeX
```bibtex
@article{ALWAJIH2023110079,
  title   = {Threshold center-symmetric local binary convolutional neural networks for bilingual handwritten digit recognition},
  author  = {Al-wajih, Ebrahim and Ghazali, Rozaida},
  journal = {Knowledge-Based Systems},
  volume  = {259},
  pages   = {110079},
  year    = {2023},
  issn    = {0950-7051},
  doi     = {10.1016/j.knosys.2022.110079},
  url     = {https://www.sciencedirect.com/science/article/pii/S0950705122011753}
}
