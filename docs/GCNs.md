# Graph Convolutional Networks (GCNs) — A Deep Dive

## 1. Motivation: Why Graphs?

Most classical deep learning architectures (CNNs, RNNs, MLPs) assume data lives on a **regular, Euclidean** domain:

| Architecture | Domain assumption |
|---|---|
| MLP | Fixed-size flat vector |
| CNN | Regular pixel grid (2-D/3-D) |
| RNN | Ordered sequence |

Many real-world datasets, however, have an **irregular, relational structure** better modelled as a **graph**:

- Social networks (users = nodes, friendships = edges)
- Molecular structures (atoms = nodes, bonds = edges)
- Finite-Element Analysis meshes ← *our use-case*
- Knowledge graphs, citation networks, …

A graph **G = (V, E)** is defined by:
- **V** — set of *N* nodes, each carrying a feature vector **x_i ∈ ℝ^F**
- **E** — set of edges encoding pairwise relationships

The node features are stacked into a matrix **X ∈ ℝ^{N×F}**, and the topology is encoded in the **adjacency matrix A ∈ {0,1}^{N×N}**.

---

## 2. Core Idea: Message Passing / Neighbourhood Aggregation

The central mechanism of any GNN (Graph Neural Network) — including GCNs — is called **message passing**. At each layer, every node:

1. **Gathers** ("aggregates") feature information from its immediate neighbours.
2. **Transforms** the aggregated information alongside its own features.
3. **Updates** its own embedding.

After *K* such layers, a node's representation captures information from its **K-hop neighbourhood**.

```
Layer 0 (raw features)
  Node i: h_i^(0) = x_i

Layer k+1
  h_i^(k+1) = σ( W^(k) · AGGREGATE({ h_j^(k) : j ∈ N(i) ∪ {i} }) )
```

Different GNN variants differ only in how they implement `AGGREGATE`.

---

## 3. The GCN Formulation (Kipf & Welling, 2017)

### 3.1 Spectral Motivation

Classical convolutional filters are defined in the **frequency domain** via the Fourier transform.  
For graphs, the analogue is the **Graph Laplacian**:

```
L = D - A          (combinatorial Laplacian)
L_sym = I - D^{-1/2} A D^{-1/2}   (symmetric normalised)
```

where **D** is the diagonal degree matrix: `D_ii = Σ_j A_ij`.

Spectral GCNs approximate a generalised convolution in the eigenbasis of *L*. Kipf & Welling simplified this to a single first-order approximation:

### 3.2 Layer-wise Propagation Rule

$$
H^{(k+1)} = \sigma\!\left(\tilde{D}^{-\frac{1}{2}}\,\tilde{A}\,\tilde{D}^{-\frac{1}{2}}\, H^{(k)}\, W^{(k)}\right)
$$

In plain math / code terms:

```
Ã  = A + I_N              # add self-loops so each node aggregates itself
D̃  = diag(Ã · 1)         # degree matrix of Ã
Â  = D̃^{-½} · Ã · D̃^{-½}  # symmetric normalisation
H' = σ(Â · H · W)        # propagate → linear transform → activate
```

where:

| Symbol | Meaning |
|---|---|
| `A` | Adjacency matrix `(N × N)` |
| `I_N` | Identity (adds self-loops) |
| `Â` | Normalised adjacency |
| `H^(k)` | Node feature matrix at layer *k* `(N × F_k)` |
| `W^(k)` | Trainable weight matrix `(F_k × F_{k+1})` |
| `σ` | Non-linear activation (e.g. ReLU) |

### 3.3 Why Normalise?

Without normalisation, nodes with many neighbours dominate the sum. The symmetric normalisation `D̃^{-½} Ã D̃^{-½}` rescales each entry by `1 / sqrt(d_i * d_j)`, giving each node an equally weighted contribution regardless of degree.

---

## 4. Building a Multi-layer GCN

A typical GCN for **node-level regression** (our FEA ROM task):

```
Input:  X  ∈ ℝ^{N × F_in}    (node features: coordinates, loads, BCs, …)
        Â  ∈ ℝ^{N × N}        (pre-computed, fixed)

GCN Layer 1:  H1 = ReLU(Â · X  · W1)   # (N × 64)
GCN Layer 2:  H2 = ReLU(Â · H1 · W2)   # (N × 32)
Output Layer: Ŷ  =       Â · H2 · W3   # (N × F_out)  ← no final activation for regression
```

Each layer's weight matrix `W` is shared across **all nodes** (analogous to weight sharing in CNNs), making GCNs highly parameter-efficient even on large meshes.

---

## 5. GCN in TensorFlow / Keras

The actual implementation in this project uses a **custom `GCNLayer`** registered with Keras for serialisation:

### 5.1 Custom GCNLayer Implementation

```python
import tensorflow as tf
import keras
from keras import layers

@keras.saving.register_keras_serializable(package="GCNLayer")
class GCNLayer(layers.Layer):
    """Spectral Graph Convolutional Layer.
    
    Applies: ``H' = σ(Â @ H @ W + b)``
    where ``Â`` is the pre-computed symmetrically normalised adjacency matrix.
    """
    
    def __init__(self, units, activation="relu", **kwargs):
        """Create a GCN layer.
        
        Parameters
        ----------
        units : int
            Dimensionality of the output feature space.
        activation : str or callable, optional
            Activation function applied after the graph convolution (default "relu").
        """
        super().__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)
    
    def build(self, input_shape):
        """Create weight matrix W and bias b.
        
        Parameters
        ----------
        input_shape : list of TensorShape
            [H_shape, A_hat_shape] where H_shape = (batch, N, F)
        """
        feature_dim = input_shape[0][-1]
        self.W = self.add_weight(
            name="W",
            shape=(feature_dim, self.units),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="b",
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)
    
    def call(self, inputs):
        """Forward pass: H' = activation(Â @ H @ W + b).
        
        Parameters
        ----------
        inputs : list of tf.Tensor
            [H, A_hat] — node features (B, N, F) and normalised adjacency (B, N, N).
        
        Returns
        -------
        tf.Tensor
            Updated node features of shape (B, N, units).
        """
        H, A_hat = inputs              # H: (batch, N, F), A_hat: (batch, N, N)
        support = tf.matmul(H, self.W)     # (batch, N, units)
        output = tf.matmul(A_hat, support) + self.b  # (batch, N, units)
        return self.activation(output)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "activation": keras.activations.serialize(self.activation),
        })
        return config
```

### 5.2 Adjacency Matrix Normalisation

The normalised adjacency **Â** is precomputed once and reused during training/inference:

```python
def build_beam_adjacency(nx=21, ny=6, nz=6):
    """Build normalised adjacency matrix for a structured hex mesh.
    
    For a 21 × 6 × 6 grid:
      N = 21 × 6 × 6 = 756 nodes
      Two nodes are adjacent if they differ by 1 step along any single axis.
    
    Returns Â = D^{-1/2} (A + I) D^{-1/2}
    """
    N = nx * ny * nz
    
    def idx(i, j, k):
        return i * ny * nz + j * nz + k
    
    rows, cols = [], []
    # Add self-loops and neighbours
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                n = idx(i, j, k)
                rows.append(n); cols.append(n)  # self-loop
                # Check 6 neighbours (±x, ±y, ±z)
                for di, dj, dk in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
                    ni, nj, nk = i+di, j+dj, k+dk
                    if 0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz:
                        rows.append(n); cols.append(idx(ni, nj, nk))
    
    A = np.zeros((N, N), dtype=np.float32)
    A[rows, cols] = 1.0
    
    # Symmetric normalisation
    deg = A.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(deg, 1e-9)))
    A_hat = D_inv_sqrt @ A @ D_inv_sqrt
    return A_hat.astype(np.float32)
```

### 5.3 Full GCN Model Architecture (Production Code)

```python
def build_gcn(input_dim: int, output_dim: int, A_hat: np.ndarray) -> keras.Model:
    """Build a GCN-based ROM model.
    
    The model receives global scalar parameters [L, w, d, P] and lifts them
    to per-node features, then performs graph convolution, and finally decodes
    to the full output field.
    """
    N = A_hat.shape[0]  # 756
    
    # Inputs
    params_inp = keras.Input(shape=(input_dim,), name="params")   # (B, 4)
    a_inp      = keras.Input(shape=(N, N),      name="A_hat")    # (B, N, N)
    
    # 1. Lift global params → per-node features
    broadcast = layers.RepeatVector(N)(params_inp)               # (B, N, 4)
    node_init = layers.Dense(32, activation="swish")(broadcast)  # (B, N, 32)
    
    # 2. GCN message passing (6 layers with alternating activations)
    x = GCNLayer(128, activation="relu",       name="gcn_1")([node_init, a_inp])
    x = GCNLayer(128, activation="leaky_relu", name="gcn_2")([x, a_inp])
    x = GCNLayer(128, activation="relu",       name="gcn_3")([x, a_inp])
    x = GCNLayer(128, activation="leaky_relu", name="gcn_4")([x, a_inp])
    x = GCNLayer(128, activation="relu",       name="gcn_5")([x, a_inp])
    x = GCNLayer(128, activation="leaky_relu", name="gcn_6")([x, a_inp])
    # x: (B, N, 128)
    
    # 3. Global average pool → graph-level summary
    pooled = layers.GlobalAveragePooling1D()(x)  # (B, 128)
    
    # 4. Dense decoder head
    h = layers.Dense(256, activation="swish")(pooled)
    h = layers.Dropout(0.1)(h)
    h = layers.Dense(512, activation="swish")(h)
    out = layers.Dense(output_dim, name="output")(h)
    
    model = keras.Model(inputs=[params_inp, a_inp], outputs=out, name="GCN_ROM")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )
    return model
```

### 5.4 Training with ROMTrainer

```python
from src.rom_model.trainer import ROMTrainer

# Train a GCN model
trainer = ROMTrainer(
    data_dir="mock_data/lhs",    # or "random", "sobol", "taguchi"
    model_dir="models/lhs",
    model_type="gcn"              # Select GCN architecture
)

trainer.train()  # Trains displacement and stress models separately
```

The trainer handles:
- Loading `.npy` samples
- 80/15/5 train/test/validation split
- Input standardisation (preserves scaler for inference)
- Training with early stopping and learning rate scheduling
- Saving `.keras` models + scalers for deployment

---

## 6. How GCNs Learn on FEA Meshes

In a Finite-Element Analysis (FEA) context, the graph structure comes **directly from the mesh**:

```
Nodes  →  mesh nodes (integration points / DOFs)
Edges  →  element connectivity (shared edge/face between elements)
Features → coordinates (x, y, z), applied loads, boundary conditions
Labels  → displacement / stress / strain at each node
```

Because nearby mesh nodes have highly correlated mechanical responses, the neighbourhood aggregation in GCNs is physically meaningful — it learns to propagate load effects through the mesh topology.

### Why GCNs outperform MLPs on meshes

| Criterion | MLP | GCN | Transformer |
|---|---|---|---|
| Uses mesh topology | ❌ (flattens nodes) | ✅ (adjacency matrix) | ✅ (self-attention) |
| Permutation invariant | ❌ | ✅ | ❌ (needs positional encoding) |
| Generalises to different mesh sizes | ❌ | ✅ | ❌ (fixed N) |
| Parameters scale with mesh size | Yes (O(N)) | No (O(F²) per layer) | No (O(N²) attention) |
| Training speed on 756-node mesh | Fastest | Slower | Slowest |

> **In practice:** For the fixed `21×6×6` mesh, all three models train successfully.
> MLP is fastest but ignores topology. GCN is parameter-efficient and respects mesh structure.
> Transformer can learn flexible attention patterns but has higher computational cost.

---

## 6. Training GCN Models in This Project

The `ROMTrainer` class in `src/rom_model/trainer.py` provides an end-to-end pipeline
supporting three architectures: **MLP**, **GCN**, and **Transformer**.

### 6.1 Initialising the Trainer

```python
from src.rom_model.trainer import ROMTrainer

# Create trainer for GCN model
trainer = ROMTrainer(
    data_dir="mock_data/lhs",       # Directory with *.npy samples
    model_dir="models/lhs",          # Where to save trained .keras files
    model_type="gcn"                 # Choose: "mlp", "gcn", or "transformer"
)
```

### 6.2 Full Training Pipeline

```python
trainer.train()
```

This does:

1. **Load data** — All `*_params.npy`, `*_disp.npy`, `*_stress.npy` files
2. **Split** — 80% train, 15% test, 5% validation (random state 42)
3. **Standardise** — Fit `StandardScaler` on training data, save for inference
4. **Train displacement model** — Separate GCN for 2268-D displacement field
5. **Train stress model** — Separate GCN for 756-D stress field
6. **Save artefacts**:
   - `models/lhs/rom_disp.keras` — Trained displacement model
   - `models/lhs/rom_stress.keras` — Trained stress model
   - `models/lhs/scaler_mean.npy`, `scaler_std.npy` — For input normalisation
   - `models/lhs/model_type.txt` — Records `"gcn"`
   - `models/lhs/metrics.json` — R² scores on test set

### 6.3 Callbacks and Regularisation

Both models use:

- **Early Stopping** — Monitors validation loss, stops after 50 epochs without improvement
- **ReduceLROnPlateau** — Halves learning rate if validation loss doesn't improve for 20 epochs
- **TensorBoard logging** — Histograms, graphs, and loss curves saved under `logs/<timestamp>/`

### 6.4 Inference with Saved Models

```python
from src.rom_model.visualizer import ROMVisualizer

visualizer = ROMVisualizer(
    model_dir="models/lhs",
    model_type="gcn"  # Must match the training model_type
)

# Predict on new input [L, w, d, P]
visualizer.visualise(L=15.0, w=2.0, d=2.0, P=250.0)
```

---

## 7. Key Hyperparameters

| Hyperparameter | Typical range | Effect |
|---|---|---|
| Number of GCN layers | 2–5 | Receptive field depth (K-hop neighbourhood) |
| Hidden units per layer | 32–256 | Representational capacity |
| Activation | ReLU, ELU, LeakyReLU | Non-linearity |
| Dropout (on H) | 0.0–0.5 | Regularisation |
| Learning rate | 1e-3 – 1e-4 | Optimisation speed |
| Loss function | MSE (regression), CrossEntropy (classification) | Task dependent |

---

## 8. Variants & Extensions

| Variant | Key idea |
|---|---|
| **GAT** (Graph Attention Network) | Learns attention weights per edge instead of fixed normalisation |
| **GraphSAGE** | Samples a fixed-size neighbourhood; scales to millions of nodes |
| **ChebNet** | Uses higher-order Chebyshev polynomials for wider spectral filters |
| **GIN** (Graph Isomorphism Net) | Provably most expressive 1-WL GNN |
| **MPNN** (Message Passing NN) | General framework unifying most GNN variants |
| **GCN + Pooling** | Hierarchical coarsening (DiffPool, MinCutPool) for graph-level tasks |

---

## 9. Limitations of Vanilla GCNs

1. **Over-smoothing** — With too many layers, all node embeddings converge to the same value. Typically ≤ 4 layers work best.
2. **Fixed graph** — The adjacency must be known at training time; purely inductive settings need GraphSAGE-style sampling.
3. **Scalability** — The full `Â · H` multiplication is `O(N²)` on dense graphs. Use sparse operations for large meshes.
4. **Depth vs. breadth trade-off** — Increasing layers increases the receptive field but risks over-smoothing.

---

## 10. Quick Reference: Equations Summary

```
# Preprocessing (done once)
Ã  = A + I

D̃_ii = Σ_j Ã_ij

Â = D̃^{-½} · Ã · D̃^{-½}

# Forward pass (per layer k)
H^{(k+1)} = σ( Â · H^{(k)} · W^{(k)} )

# Final output (regression)
Ŷ = Â · H^{(K)} · W^{(K)}    (no σ)
```

---

## References

- Kipf, T. N. & Welling, M. (2017). *Semi-Supervised Classification with Graph Convolutional Networks*. ICLR 2017. [arXiv:1609.02907](https://arxiv.org/abs/1609.02907)
- Hamilton, W. et al. (2017). *Inductive Representation Learning on Large Graphs* (GraphSAGE). NeurIPS 2017.
- Veličković, P. et al. (2018). *Graph Attention Networks* (GAT). ICLR 2018.
- Bronstein, M. et al. (2021). *Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges*. [arXiv:2104.13478](https://arxiv.org/abs/2104.13478)
