# NeuralCAE Experiments

Physical AI exploration by a CAE engineer: MeshGraphNet, FNO, PyG experiments, and NPM (Neural Percolation Model) validation.

[中文版 README](README_CN.md)

---

## NPM Framework Study Series

| # | Title | Status | Link |
|---|-------|--------|------|
| #1 | Predicting Emergence Order | Published | [Zenodo (DOI: 10.5281/zenodo.19440042)](https://doi.org/10.5281/zenodo.19440042) |
| #2 | Physical Interpretation of Critical Exponent | Planned | — |
| #3 | Emergence Curve Across Critical Threshold | Planned | — |
| #4 | Operationalizing Data Quality Parameter | Planned | — |

**NPM main framework:** [Zenodo (DOI: 10.5281/zenodo.19209722)](https://doi.org/10.5281/zenodo.19209722)

### Validation Data & Code (`validation/`)

- `scripts/` — NPM fitting, Brier Score computation
- `results/` — MMLU evaluation results for 9 models (Pythia 70M–12B + Phi-2)
- `figures/` — Accuracy vs Brier Score scatter plots (CN/EN)

---

## MeshGraphNets Experiments with PhysicsNeMo

API validation, training pipeline reproduction, and documentation issue reporting for MeshGraphNet models under the NVIDIA PhysicsNeMo framework.

> Based on DeepMind's 2021 [MeshGraphNets](https://github.com/deepmind/deepmind-research/tree/master/meshgraphnets) paper, using the model implementation provided by NVIDIA PhysicsNeMo.

---

### Environment

| Package | Purpose |
|---------|---------|
| PyTorch | Deep learning framework |
| PyTorch Geometric (PyG) | Graph neural network backend (recommended) |
| DGL | Graph neural network backend (required by older PhysicsNeMo) |
| PhysicsNeMo | NVIDIA physics simulation AI framework, provides MeshGraphNet models |
| h5py | Reading HDF5 format CFD simulation data |
| TensorFlow 1.x | Data preprocessing only (`triangles_to_edges` function) |
| torch_scatter | PyG message passing dependency |
| matplotlib | Visualization and animation generation |

---

### Directory Structure

```
NeuralCAE-experiments/
├── 01-meshgraphnets.py          # PhysicsNeMo API test (DGL backend)
├── 01-meshgraphnets-3.py        # PhysicsNeMo API test (PyG backend) + bug report
├── 01-MeshGraphNets_PyG.py      # Full training pipeline (PyG, based on DeepMind paper)
├── 02-pyg.py                    # DGL vs PyG graph creation comparison
├── validation/                  # NPM validation experiments
│   ├── scripts/                 # NPM fitting & Brier Score scripts
│   ├── results/                 # MMLU evaluation results (9 models)
│   └── figures/                 # Scatter plots (CN/EN)
├── datasets/                    # Datasets (excluded by .gitignore)
├── best_models/                 # Training outputs
├── animations/                  # Velocity field prediction animations (GIF)
├── 2d_loss_plots/               # Loss curve plots (PDF)
└── .gitignore
```

---

### Scripts

#### 1. `01-meshgraphnets.py` — API Test (DGL Backend)

Tests PhysicsNeMo MeshGraphNet models using DGL graph format:

- **MeshGraphNet**: Base model, 100-node toy graph forward pass
- **MeshGraphKAN**: KAN (Kolmogorov-Arnold Networks) variant, `num_harmonics=5`
- **BiStrideMeshGraphNet**: Multi-scale variant, requires `BistrideMultiLayerGraphDataset`

#### 2. `01-meshgraphnets-3.py` — API Test (PyG Backend) + Bug Report

Tests all 4 model variants using PyG `Data` objects, with documentation issue reports (see [Findings](#findings) section).

#### 3. `02-pyg.py` — DGL vs PyG Graph Creation

Compares graph creation approaches between DGL and PyG for the same 3-node directed graph.

#### 4. `01-MeshGraphNets_PyG.py` — Full Training Pipeline

Complete MeshGraphNet training pipeline based on the DeepMind paper, implemented with PyG:

| Module | Function |
|--------|----------|
| Data Loading | Reads CFD simulation data from `valid.h5`, builds PyG graphs |
| `ProcessorLayer` | Custom MessagePassing layer (edge MLP → node MLP, with residual connections) |
| `MeshGraphNet` | Encoder-Processor-Decoder architecture (multi-layer ProcessorLayer) |
| `train()` / `test()` | Training loop + acceleration loss + velocity validation RMSE |
| `visualize()` | Generates Ground Truth / Prediction / Error animations via matplotlib |

---

### Experiments

| num_layers | batch_size | hidden_dim | epochs | lr | weight_decay | shuffle | train/test | Note |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|------|
| 10 | 16 | 10 | 5000 | 0.001 | 5e-4 | False | 45/10 | Control group |
| 10 | 16 | 10 | 5000 | 0.001 | 5e-4 | True | 45/10 | Shuffle comparison |
| 10 | 16 | 10 | 5000 | 0.001 | 5e-4 | True | 85/15 | Increased data |
| 10 | 16 | **30** | 5000 | 0.001 | 5e-4 | True | **420/80** | Largest scale |

---

### Findings

Three issues found in NVIDIA PhysicsNeMo official documentation during API testing:

**Bug 1: Input type error.** Documentation passes `edge_index` (Tensor), but the model requires a PyG `Data` object.

**Bug 2: HybridMeshGraphNet edge structure.** Documentation does not mention that mesh edges and world edges must be **merged** into a single graph.

**Bug 3: BiStrideMeshGraphNet example unusable.** The simple example in the documentation cannot run. The model requires `BistrideMultiLayerGraphDataset` preprocessing to generate correct multi-scale graph structures.

---

### Usage

```bash
# API test (PyG backend, recommended)
python 01-meshgraphnets-3.py

# Full training pipeline (requires datasets/valid.h5)
python 01-MeshGraphNets_PyG.py
```

---

### Data

Dataset files are large (~4.9GB total) and **not included in the repository**.

| File | Size | Description |
|------|------|-------------|
| `valid.h5` | 1.2 GB | Raw CFD simulation data (HDF5) |
| `meshgraphnets_miniset5traj_vis.pt` | 1.3 GB | 5 trajectories, full timesteps |
| `meshgraphnets_miniset100traj25ts_vis.pt` | 1.1 GB | 100 trajectories, 25 timesteps |
| `meshgraphnets_miniset30traj5ts_vis.pt` | 59 MB | 30 trajectories, 5 timesteps (lightweight) |
| `test_processed_set.pt` | 1.8 MB | 2 trajectories, 2 timesteps (toy example) |

Data source: DeepMind MeshGraphNets cylinder flow dataset.

---

**License:** MIT | **Author:** Tiexin Ding | **NeuralCAE** | All work conducted in personal time.
