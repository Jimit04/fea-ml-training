# FEA Reduced-Order Model (ROM)

> **Project Goal:** Use Machine Learning to *replace* expensive Finite Element Analysis (FEA) solvers.
> Instead of running a full FEA simulation every time, we train a neural network on pre-computed FEA data,
> then use it to predict results in milliseconds.

---

## Table of Contents
1. [Problem Statement](#1-problem-statement)
2. [Dataset & Physics](#2-dataset--physics)
3. [Data Preprocessing](#3-data-preprocessing)
4. [Model 1 — MLP (Multi-Layer Perceptron)](#4-model-1--mlp-multi-layer-perceptron)
5. [Model 2 — GCN (Graph Convolutional Network)](#5-model-2--gcn-graph-convolutional-network)
6. [Training Strategy](#6-training-strategy)
7. [Evaluation & Visualisation](#7-evaluation--visualisation)
8. [How to Run](#8-how-to-run)

---

## 1. Problem Statement

A cantilever beam is fixed at one end (x = 0) and has a point load **P** applied at the free tip (x = L).
Given four input parameters:

| Symbol | Name | Unit | Range |
|--------|------|------|-------|
| L | Length | mm | 5 – 20 |
| w | Width  | mm | 0.5 – 3 |
| d | Depth  | mm | 0.1 – 0.5 |
| P | Load   | N  | 1 000 – 50 000 |

We want to predict the **full displacement field** (a 3D vector at every mesh node) and the
**full stress field** (σ_xx at every mesh node) — without running a traditional FEA solver.

---

## 2. Dataset & Physics

### 2.1 Mesh

The beam is discretised into a structured **21 × 6 × 6** hexahedral grid:

```
nx=21 nodes along X (length direction)
ny=6  nodes along Y (width direction)
nz=6  nodes along Z (depth / height direction)
Total nodes = 21 × 6 × 6 = 756
```

### 2.2 Euler-Bernoulli Beam Theory (MockFEASolver)

We use classical beam theory as the "ground truth" solver. For Steel:

```
E  = 210,000 MPa   (Young's Modulus)
ν  = 0.29           (Poisson's Ratio)
```

**Moment of Inertia** for a rectangular cross-section:
```
I = (w × d³) / 12
```

**Vertical displacement** at any point x along the beam (cantilever, tip load P):
```
v(x) = - (P × x²) / (6 × E × I) × (3L - x)
```

**Bending stress** σ_xx (tension at top surface z > 0, compression at bottom z < 0):
```
M(x)     = P × (L - x)         ← Bending moment at position x
σ_xx(x,z) = M(x) × z / I
```

> **Key intuition:** The beam bends most at the tip. Stress is highest near the fixed end (x = 0)
> and zero at the free tip. The top surface (z > 0) is in tension, the bottom (z < 0) is in compression.

Each sample is saved as:
- `sample_N_params.npy`  — `[L, w, d, P]`
- `sample_N_disp.npy`    — shape `(756, 3)` displacement vectors
- `sample_N_stress.npy`  — shape `(756,)` σ_xx at every node
- `sample_N.vtk`         — full mesh file for 3D visualisation

---

## 3. Data Preprocessing

Before feeding data into any neural network, inputs must be **normalised** so each feature has
roughly equal influence on the gradient updates.

We use **Standardisation** (zero mean, unit variance):

```
X_scaled = (X - μ) / σ

where:
  μ = mean of each feature over all training samples
  σ = standard deviation of each feature over all training samples
```

The scaler is fitted **only on training data**, then applied to test data — this prevents
*data leakage* (the model must not "see" test statistics during training).

After training, `μ` and `σ` are saved as `models/scaler_mean.npy` and `models/scaler_std.npy`
so the visualiser can normalise new inputs at inference time.

---

## 4. Model 1 — MLP (Multi-Layer Perceptron)

### 4.1 What is an MLP?

An MLP is the most fundamental neural network: a series of **Dense (fully-connected) layers**
where every neuron in layer k is connected to every neuron in layer k+1.

```
Input → [Linear Transform → Activation → Regularisation] × N → Output
```

### 4.2 Architecture

```
Input: [L, w, d, P]  →  shape (4,)

Dense(256)  → BatchNorm → Swish → Dropout(0.15)
Dense(512)  → BatchNorm → Swish → Dropout(0.15)
Dense(512)  → BatchNorm → Swish
Dense(256)  → Swish
Dense(output_dim)          ← output_dim = 2268 for displacement, 756 for stress
```

**Total trainable parameters ≈ 723,700** (~2.76 MB)

### 4.3 Building Blocks Explained

#### Dense Layer
Performs the linear transformation:
```
y = x W + b
```
- `x` — input vector of shape (batch, in_features)
- `W` — weight matrix of shape (in_features, out_features),  *learned*
- `b` — bias vector of shape (out_features),  *learned*
- `y` — output of shape (batch, out_features)

#### Batch Normalisation
After each Dense layer, we normalise the activations within a mini-batch:
```
x̂ = (x - μ_batch) / √(σ²_batch + ε)
y  = γ × x̂ + β
```
This keeps activations from exploding or vanishing, and allows higher learning rates.

#### Swish / SiLU Activation
```
Swish(x) = x × sigmoid(x) = x / (1 + e^{-x})
```
Unlike ReLU (which hard-clips negatives to 0), Swish is smooth everywhere and allows
small negative values to pass through, which tends to give better gradient flow in deep networks.

#### Dropout
During training, randomly sets a fraction (here 15%) of neuron outputs to **zero**.
This forces the network not to rely on any single neuron and acts as regularisation,
reducing overfitting.

### 4.4 Why does MLP work here?

The 4 input parameters fully describe the beam geometry and load.
Since Euler-Bernoulli theory gives a deterministic, smooth mapping from `[L, w, d, P]`
to the displacement and stress fields, a deep MLP can approximate this mapping without
needing any graph structure.

**Limitation:** The MLP treats the output as a flat vector and does not exploit the spatial
structure of the mesh. It must learn the geometry implicitly from the data.

---

## 5. Model 2 — GCN (Graph Convolutional Network)

### 5.1 Why Graph Networks for FEA?

FEA meshes are naturally graphs: nodes are connected to their neighbours by element edges.
A GCN respects this topology — each node's feature is updated based on its own state *and*
the state of its immediate neighbours (message passing).

> **Analogy:** Think of each mesh node as a person in a network.
> In each "round" (GCN layer), each person averages their own opinion with their neighbours' opinions.
> After several rounds, information has propagated across the whole network.

### 5.2 Building the Graph (Beam Adjacency)

The 21 × 6 × 6 structured grid has a simple neighbourhood rule:
two nodes are adjacent if they differ by exactly 1 step along any single axis (X, Y, or Z).

```
Node index:  n = i × ny × nz + j × nz + k
Neighbours:  (i±1, j, k), (i, j±1, k), (i, j, k±1)   — if within bounds
```

We also add **self-loops** (each node is its own neighbour) so every node retains
its own information during aggregation.

#### Symmetric Normalisation

Raw adjacency A is normalised to prevent scale issues:
```
Â = D^{-1/2} (A + I) D^{-1/2}

where D is the diagonal degree matrix:  D_{ii} = Σ_j A_{ij}
```
This ensures that nodes with many neighbours don't dominate the aggregation.
Â has all eigenvalues in [-1, 1], keeping gradients stable.

### 5.3 GCN Layer Forward Pass

```python
# H:     (batch, N, F)   — current node feature matrix
# A_hat: (batch, N, N)   — precomputed normalised adjacency
# W:     (F, F')         — learnable weight matrix
# b:     (F',)           — learnable bias

support = H @ W          # (batch, N, F')  — feature transform
H'      = Â @ support + b  # (batch, N, F')  — neighbourhood aggregation
H'      = activation(H')
```

Each GCN layer:
1. **Transforms** each node's features with a weight matrix W
2. **Aggregates** features from all neighbours (via Â matrix multiplication)
3. **Activates** the result non-linearly (ReLU)

### 5.4 Full GCN Architecture

```
Step 1 — Lift global params to per-node features:
  Input: [L, w, d, P]  →  shape (4,)
  RepeatVector(756)    →  shape (756, 4)   ← same params broadcast to all 756 nodes
  Dense(32, swish)     →  shape (756, 32)  ← initial node embeddings

Step 2 — Graph message passing (3 GCN layers):
  GCNLayer(64,  ReLU)  →  shape (756, 64)
  GCNLayer(128, ReLU)  →  shape (756, 128)
  GCNLayer(64,  ReLU)  →  shape (756, 64)

Step 3 — Global readout (pool over all nodes):
  GlobalAveragePooling1D() →  shape (64,)  ← one vector summarising the whole mesh

Step 4 — Decode to output field:
  Dense(256, swish)    →  shape (256,)
  Dropout(0.1)
  Dense(512, swish)    →  shape (512,)
  Dense(output_dim)    →  shape (2268,) or (756,)
```

**Total trainable parameters ≈ 554,900** (~2.12 MB)

### 5.5 Why GCN might outperform MLP

| Aspect | MLP | GCN |
|--------|-----|-----|
| Exploits mesh topology | ✗ | ✓ |
| Parameter count | Higher | Lower |
| Generalises to new meshes | ✗ | ✓ (with same connectivity) |
| Training speed | Faster | Slower |
| Interpretability | Black-box | Node-level features inspectable |

> **Note:** Because we're still using a *fixed* `21×6×6` mesh and the physics is smooth,
> the MLP often performs competitively with the GCN at this scale.
> GCNs really shine when geometry varies (unstructured meshes) or when very good
> generalisation across topologies is needed.

---

## 6. Training Strategy

Both models are trained **separately** for displacement and stress:

### 6.1 Loss Function
```
MSE = (1/N) × Σ (ŷ_i - y_i)²
```
Mean Squared Error penalises large errors more than small ones, which is appropriate
for regression over physical fields.

### 6.2 Optimiser: Adam
Adam (Adaptive Moment Estimation) maintains per-parameter learning rates:
```
m_t = β₁ × m_{t-1} + (1 - β₁) × ∇L   ← first moment (mean)
v_t = β₂ × v_{t-1} + (1 - β₂) × ∇L²  ← second moment (variance)
θ   = θ - α × m̂_t / (√v̂_t + ε)

Default: α=0.001, β₁=0.9, β₂=0.999
```

### 6.3 Callbacks

#### Early Stopping
Monitors validation loss. If it does not improve for **20 consecutive epochs**,
training stops and the best weights are restored. This prevents overfitting.

#### ReduceLROnPlateau
If validation loss doesn't improve for **10 epochs**, the learning rate is halved
(`factor=0.5`). This allows the optimiser to make smaller, more precise updates
as it approaches a minimum.

```
Epoch 1:   LR = 0.001
Epoch 50:  LR = 0.001  (improving)
Epoch 60:  LR = 0.0005 (plateau detected → halved)
Epoch 70:  LR = 0.00025
...
```

### 6.4 Train / Test Split
- **80%** training samples
- **20%** held-out test samples (never seen during training)
- `random_state=42` for reproducibility

---

## 7. Evaluation & Visualisation

### 7.1 Metrics
- **Test MSE** — Mean Squared Error on the held-out set
- **Test MAE** — Mean Absolute Error (easier to interpret in physical units)

### 7.2 Visualiser

The `ROMVisualizer` class:
1. Loads saved `.keras` models and `.npy` scalers
2. Normalises the input `[L, w, d, P]` using saved μ / σ
3. Runs the model to predict displacement and stress fields
4. Also runs the `MockFEASolver` to get the ground truth
5. Renders both side-by-side using PyVista with:
   - **Jet colormap** for stress (blue = low, red = high)
   - **Warped geometry** showing the deflected shape
   - **Max displacement & stress** annotations on both panels
   - **% error** relative to ground truth on the predicted panel

### 7.3 Interpreting % Error

```
Disp Error (%)   = |max_pred_disp  - max_gt_disp|  / max_gt_disp  × 100
Stress Error (%) = |max_pred_stress - max_gt_stress| / max_gt_stress × 100
```

Errors below **10%** are generally acceptable for engineering ROM applications.
To improve accuracy: train on more samples (≥ 1000) and train for more epochs.

---

## 8. How to Run

```bash
# 1. Generate 500 training samples
uv run .\main.py --generate --samples 500

# 2. Train with MLP (default)
uv run .\main.py --train --model mlp

# 3. Train with GCN
uv run .\main.py --train --model gcn

# 4. Visualise (uses the last trained model)
uv run .\main.py --visualize

# 5. Run entire pipeline at once
uv run .\main.py --generate --train --visualize --samples 500 --model mlp

# Save a screenshot instead of opening a window
uv run .\main.py --visualize --screenshot output.png
```

### File Structure

```
fea-ml-training/
├── main.py                    ← Entry point, CLI
├── src/
│   ├── data_generator.py      ← MockFEASolver (Euler-Bernoulli physics)
│   ├── generate_dataset.py    ← Batch sample generation loop
│   ├── rom_model.py           ← GCNLayer, MLP/GCN builders, ROMTrainer
│   └── visualizer.py          ← ROMVisualizer (PyVista rendering)
├── mock_data/                 ← Generated .npy and .vtk samples
├── models/                    ← Saved .keras models + scaler .npy files
└── DOCUMENTATION.md           ← This file
```

---

*This project demonstrates how ML-based Reduced Order Models can approximate physics simulations.
The same pattern is used in industry for aerodynamics, structural analysis, and thermal simulations.*
