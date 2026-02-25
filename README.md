# FEA Reduced-Order Model (ROM)

A proof-of-concept that uses **Machine Learning to replace expensive FEA solvers**.
A neural network is trained on data generated from Euler-Bernoulli beam theory, then used to predict full displacement and stress fields in milliseconds.

> 📄 See [docs/Overview.md](docs/Overview.md) for a detailed explanation of the physics, architecture, and training strategy.
> 📐 See [docs/GCNs.md](docs/GCNs.md) for an in-depth primer on Graph Convolutional Networks.

---

## Workflow

1. **Data Generation** — `MockFEASolver` analytically solves a cantilever beam using Euler-Bernoulli beam theory (Steel: E=210,000 MPa, ν=0.29). Outputs displacement and stress fields on a 21×6×6 hex mesh.
2. **Training** — A TensorFlow/Keras neural network (MLP or GCN) learns the mapping from `[Length, Width, Depth, Load]` → full field results.
3. **Visualisation** — PyVista renders GT vs Predicted side-by-side with max values and % errors.

---

## Setup

Project is managed with `uv`.

```bash
uv sync
```

---

## Usage

```bash
# Full pipeline (generate → train → visualize)
uv run .\main.py

# Generate 500 samples with a specific sampling strategy
uv run main.py --generate --samples 500 --sampling taguchi

# Train with MLP
uv run .\main.py --train --model mlp

# Train with GCN (default)
uv run .\main.py --train --model gcn

# Visualize predictions vs ground truth
uv run .\main.py --visualize

# Save visualization to file
uv run .\main.py --visualize --screenshot output.png
```

### Arguments

| Argument             | Description                                          | Default   |
| -------------------- | ---------------------------------------------------- | --------- |
| `--generate`         | Generate synthetic dataset                           | —         |
| `--train`            | Train the ROM model                                  | —         |
| `--visualize`        | Launch 3D visualizer                                 | —         |
| `--samples N`        | Number of samples to generate                        | `500`     |
| `--sampling`         | Sampling strategy: `random`, `lhs`, `sobol`, `taguchi` | `lhs`   |
| `--model`            | Model type: `mlp` or `gcn`                           | `gcn`     |
| `--screenshot PATH`  | Save screenshot instead of opening window            | —         |

> If no stage flags (`--generate`, `--train`, `--visualize`) are provided, the full pipeline runs end-to-end.

---

## Models

### MLP (Multi-Layer Perceptron)

Deep Dense network: `4 → 256 → 512 → 512 → 256 → output`

- Swish (SiLU) activations, BatchNorm, Dropout (15%)
- Compiled with Adam (lr=1e-3) and MSE loss

### GCN (Graph Convolutional Network)

Treats the beam mesh as a graph. Performs spectral convolution over the 21×6×6 node adjacency.

- Lifts global params → per-node features via `RepeatVector(756)` + `Dense(32)`
- 6 GCN message-passing layers (`GCNLayer(128)`) with alternating ReLU / LeakyReLU
- `GlobalAveragePooling1D` → Dense decoder head (256 → 512 → output)
- Compiled with Adam (lr=1e-3) and MSE loss

---

## Inputs & Physics

| Parameter | Symbol | Unit | Range          |
| --------- | ------ | ---- | -------------- |
| Length    | L      | mm   | 5 – 20         |
| Width     | w      | mm   | 1 – 3          |
| Depth     | d      | mm   | 1 – 3          |
| Load      | P      | N    | -500 – 500     |

**Material:** Steel — E = 210,000 MPa, ν = 0.29

**Mesh:** Structured hexahedral grid, 21 × 6 × 6 = 756 nodes

---

## Project Structure

```
fea-ml-training/
├── main.py                    ← CLI entry point
├── docs/
│   ├── Overview.md            ← Detailed physics & architecture docs
│   └── GCNs.md                ← GCN deep-dive reference
├── src/
│   ├── data_generator.py      ← MockFEASolver (Euler-Bernoulli physics)
│   ├── generate_dataset.py    ← Batch sample generation (multiple sampling strategies)
│   ├── rom_model/             ← ROM model package
│   │   ├── __init__.py        ← Re-exports all public symbols
│   │   ├── layers.py          ← GCNLayer (custom Keras layer)
│   │   ├── adjacency.py       ← Normalised adjacency matrix builder
│   │   ├── architectures.py   ← MLP & GCN model factories
│   │   └── trainer.py         ← ROMTrainer (training pipeline)
│   └── visualizer.py          ← PyVista 3D visualizer (predicted vs ground truth)
├── tests/                     ← Batch scripts for end-to-end testing
├── mock_data/<sampling>/      ← Generated .npy and .vtk samples
└── models/<sampling>/         ← Saved .keras models and .npy scalers
```
