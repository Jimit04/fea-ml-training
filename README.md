# FEA Reduced-Order Model (ROM)

A proof-of-concept that uses **Machine Learning to replace expensive FEA solvers**.
A neural network is trained on data generated from Euler-Bernoulli beam theory, then used to predict full displacement and stress fields in milliseconds.

> 📄 See [docs/Overview.md](docs/Overview.md) for detailed explanation of the physics, architecture, and training strategy.

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

# Generate 500 samples
uv run .\main.py --generate --samples 500

# Train with MLP (default)
uv run .\main.py --train --model mlp

# Train with GCN
uv run .\main.py --train --model gcn

# Visualize predictions vs ground truth
uv run .\main.py --visualize

# Save visualization to file
uv run .\main.py --visualize --screenshot output.png
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--generate` | Generate synthetic dataset | — |
| `--train` | Train the ROM model | — |
| `--visualize` | Launch 3D visualizer | — |
| `--samples N` | Number of samples to generate | 500 |
| `--model` | Model type: `mlp` or `gcn` | `mlp` |
| `--screenshot PATH` | Save screenshot instead of opening window | — |

---

## Models

### MLP (Multi-Layer Perceptron)
Deep Dense network: `4 → 256 → 512 → 512 → 256 → output`
- Swish activations, BatchNorm, Dropout
- ~723,700 parameters

### GCN (Graph Convolutional Network)
Treats the beam mesh as a graph. Performs spectral convolution over the 21×6×6 node adjacency.
- 3 GCN layers (64 → 128 → 64), global average pool, Dense decoder head
- ~554,900 parameters

---

## Inputs & Physics

| Parameter | Symbol | Unit | Range |
|-----------|--------|------|-------|
| Length | L | mm | 5 – 20 |
| Width  | w | mm | 0.5 – 3 |
| Depth  | d | mm | 0.1 – 0.5 |
| Load   | P | N  | 1,000 – 50,000 |

**Material:** Steel — E = 210,000 MPa, ν = 0.29

---

## Project Structure

```
fea-ml-training/
├── main.py                    ← CLI entry point
├── DOCUMENTATION.md           ← Detailed documentation
├── src/
│   ├── data_generator.py      ← MockFEASolver (Euler-Bernoulli physics)
│   ├── generate_dataset.py    ← Batch sample generation
│   ├── rom_model.py           ← GCNLayer, MLP/GCN builders, ROMTrainer
│   └── visualizer.py          ← PyVista 3D visualizer
├── mock_data/                 ← Generated .npy and .vtk samples
└── models/                    ← Saved .keras models and .npy scalers
```
