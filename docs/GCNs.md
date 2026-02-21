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

Below is a minimal self-contained GCN implementation using TensorFlow:

```python
import tensorflow as tf
import numpy as np
import scipy.sparse as sp

# ── Preprocessing ──────────────────────────────────────────────
def normalise_adj(A: np.ndarray) -> np.ndarray:
    """Compute  Â = D̃^{-½} (A + I) D̃^{-½}  as a dense array."""
    A_tilde = A + np.eye(A.shape[0])
    D_tilde = np.diag(A_tilde.sum(axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D_tilde.diagonal()))
    return D_inv_sqrt @ A_tilde @ D_inv_sqrt          # Â


# ── Custom Keras Layer ──────────────────────────────────────────
class GCNLayer(tf.keras.layers.Layer):
    """A single  H' = σ(Â · H · W)  layer."""

    def __init__(self, units: int, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        # input_shape[0] → H,  input_shape[1] → Â  (ignored for weight shape)
        F_in = input_shape[0][-1]
        self.W = self.add_weight(
            name="W", shape=(F_in, self.units),
            initializer="glorot_uniform", trainable=True
        )

    def call(self, inputs):
        H, A_hat = inputs                    # node features, normalised adj
        # (N,F)·(F,units) → (N,units) ;  then (N,N)·(N,units)
        support = tf.matmul(H, self.W)
        output  = tf.matmul(A_hat, support)
        return self.activation(output) if self.activation else output


# ── Full GCN Model ──────────────────────────────────────────────
def build_gcn(n_nodes: int, f_in: int, f_out: int) -> tf.keras.Model:
    H_in    = tf.keras.Input(shape=(f_in,),   name="node_features")   # (N, F_in)
    A_in    = tf.keras.Input(shape=(n_nodes,), name="adj_matrix")      # (N, N)

    h = GCNLayer(64, activation="relu")([H_in, A_in])
    h = GCNLayer(32, activation="relu")([h,    A_in])
    out = GCNLayer(f_out)([h, A_in])          # linear output for regression

    model = tf.keras.Model(inputs=[H_in, A_in], outputs=out)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model
```

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

| Criterion | MLP | GCN |
|---|---|---|
| Uses mesh topology | ❌ (flattens nodes) | ✅ (adjacency matrix) |
| Permutation invariant | ❌ | ✅ |
| Generalises to different mesh sizes | ❌ | ✅ |
| Parameters scale with mesh size | Yes (O(N)) | No (O(F²) per layer) |

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
