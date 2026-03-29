# Transformers for FEA ROM — A Deep Dive

## 1. Motivation: Beyond Neighbourhood Aggregation

Graph Convolutional Networks (GCNs) use **local message passing**: each node aggregates features
only from its *immediate neighbours*, and information propagates layer-by-layer across the mesh.

```
GCN receptive field growth:
  Layer 1:  Each node sees its 1-hop neighbours
  Layer 2:  Each node sees its 2-hop neighbours
  Layer 3:  Each node sees its 3-hop neighbours
  ...
```

For the `21×6×6` FEA mesh (756 nodes), reaching one end from the other requires ~10–20 GCN layers,
risking **over-smoothing** (all node embeddings converge to the same value).

**Transformers** solve this differently using **self-attention**: every node can directly "look at"
every other node in a single layer, learning which relationships are important.

```
Transformer receptive field:
  Layer 1:  Each node attends to ALL 756 nodes simultaneously
  Layer 2:  Each node attends to ALL 756 nodes (with updated features)
  ...
```

---

## 2. Self-Attention: The Core Mechanism

### 2.1 Scaled Dot-Product Attention

Given a query **q**, key **k**, and value **v** vectors:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

where:
- **Q** (Queries) — "What am I looking for?" `(N, d_k)`
- **K** (Keys) — "What am I offering to match?" `(N, d_k)`
- **V** (Values) — "What should I pass forward?" `(N, d_v)`
- **d_k** — Query/Key dimension (scaling prevents gradient explosion)

**Intuition:**

1. For each node **i**, compute a dot product with every other node **j**: `Q_i · K_j^T`
2. Softmax normalises these scores → attention weights that sum to 1.0
3. Multiply attention weights by values: each node gets a weighted sum of all other nodes' features

$$
\text{output}_i = \sum_{j=1}^{N} \alpha_{ij} V_j \quad \text{where} \quad \alpha_{ij} = \frac{e^{Q_i \cdot K_j / \sqrt{d_k}}}{\sum_k e^{Q_i \cdot K_k / \sqrt{d_k}}}
$$

### 2.2 Attention Weights Are Learned

Unlike GCNs (which use fixed adjacency `Â`), attention weights are **dynamically computed**
based on node features. Nodes learn which other nodes are "relevant" for their task.

**Example in FEA context:**
- Nodes near the load point learn to attend strongly to the load application node
- Nodes near fixed supports learn to attend to boundary condition nodes
- Nodes in free-body regions ignore fixed boundary conditions

---

## 3. Multi-Head Attention

Computing a single attention head limits expressivity. **Multi-head attention** computes
attention in *parallel* over different representation subspaces:

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O
$$

where each head uses its own query, key, and value projections:

$$
\text{head}_i = \text{Attention}\left(Q W_i^Q, K W_i^K, V W_i^V\right)
$$

**Benefit:** Each head focuses on different relationships. In FEA:
- Head 1 might learn spatial proximity
- Head 2 might learn stress concentration patterns
- Head 3 might learn load transmission paths

---

## 4. Positional Encoding

A critical limitation of self-attention: it is **permutation-invariant**. If we shuffle the node order,
the output doesn't change (each node still attends to the same peers, just in different positions).

For FEA meshes, position is *critical*. Node (0,0,0) has very different physics than node (20,5,5).

### 4.1 The PositionalEmbedding Layer (This Project)

```python
@keras.saving.register_keras_serializable(name="PositionalEmbedding")
class PositionalEmbedding(layers.Layer):
    """Learnable positional embeddings for mesh nodes."""
    
    def __init__(self, max_seq_len, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.max_seq_len = max_seq_len      # e.g., 756
        self.embed_dim = embed_dim          # e.g., 64
    
    def build(self, input_shape):
        # Learnable embedding table: (max_seq_len, embed_dim)
        self.embedding = layers.Embedding(
            input_dim=self.max_seq_len,
            output_dim=self.embed_dim
        )
        super().build(input_shape)
    
    def call(self, inputs):
        # inputs: (B, N, F)
        N_seq = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=N_seq, delta=1)    # [0, 1, 2, ..., N-1]
        # Look up learnable embeddings for each position
        pos_embed = self.embedding(positions)                   # (N, embed_dim)
        # Add positional embeddings to node features
        return inputs + pos_embed                               # (B, N, F)
```

**Instead of hardcoded Fourier features** (e.g., sin/cos patterns), we **learn** the positional
embeddings from data. The model discovers which position coordinates matter for FEA prediction.

---

## 5. Transformer Encoder Block

A single **Transformer encoder block** consists of:

1. **Multi-head Self-Attention** (with residual connection + layer norm)
2. **Feed-Forward Network** (with residual connection + layer norm)

```python
# Input: x  ∈ ℝ^{(B, N, F)}

# Self-Attention
attn_out = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
x = Add()([x, attn_out])                 # Residual connection
x = LayerNormalization(epsilon=1e-6)(x)

# Feed-Forward (2-layer MLP)
ffn = Dense(128, activation="swish")(x)  # Expand
ffn = Dense(64)(ffn)                     # Project back to original dim
x = Add()([x, ffn])                      # Residual connection
x = LayerNormalization(epsilon=1e-6)(x)

# Output: x  ∈ ℝ^{(B, N, F)}
```

Stacking 3 of these blocks (6 attention computations total) allows the model to progressively
refine node features with access to global context.

---

## 6. Full Transformer ROM Architecture (Production Code)

### 6.1 Model Definition

```python
def build_transformer(input_dim: int, output_dim: int, A_hat: np.ndarray) -> keras.Model:
    """Build a Transformer-based ROM model treating the mesh as a sequence.
    
    The architecture:
      1. Lifts scalar global parameters to per-node features.
      2. Adds learnable positional embeddings.
      3. Three Transformer encoder blocks.
      4. Global average pooling → graph-level summary.
      5. Dense decoder to output field.
    """
    N = A_hat.shape[0]  # 756
    
    # ── Input ──
    params_inp = keras.Input(shape=(input_dim,), name="params")  # (B, 4)
    
    # ── 1. Lift params to per-node features ──
    broadcast = layers.RepeatVector(N)(params_inp)               # (B, N, 4)
    node_init = layers.Dense(64, activation="swish")(broadcast)  # (B, N, 64)
    
    # ── 2. Positional encoding ──
    x = PositionalEmbedding(max_seq_len=N, embed_dim=64)(node_init)  # (B, N, 64)
    
    # ── 3. Transformer encoder blocks ──
    num_heads = 4
    embed_dim = 64
    ff_dim = 128
    
    for block_idx in range(3):
        # Multi-head self-attention
        attn_out = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim
        )(x, x)
        x = layers.Add()([x, attn_out])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Feed-forward network
        ffn_out = layers.Dense(ff_dim, activation="swish")(x)
        ffn_out = layers.Dense(embed_dim)(ffn_out)
        x = layers.Add()([x, ffn_out])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    # ── 4. Global average pooling ──
    pooled = layers.GlobalAveragePooling1D()(x)  # (B, 64)
    
    # ── 5. Dense decoder head ──
    h = layers.Dense(256, activation="swish")(pooled)  # (B, 256)
    h = layers.Dropout(0.1)(h)
    h = layers.Dense(512, activation="swish")(h)       # (B, 512)
    out = layers.Dense(output_dim, name="output")(h)   # (B, 2268) or (B, 756)
    
    model = keras.Model(inputs=params_inp, outputs=out, name="Transformer_ROM")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )
    return model
```

### 6.2 Training with ROMTrainer

```python
from src.rom_model.trainer import ROMTrainer

# Train a Transformer model
trainer = ROMTrainer(
    data_dir="mock_data/lhs",
    model_dir="models/lhs",
    model_type="transformer"  # Select Transformer architecture
)

trainer.train()  # Trains displacement and stress models separately
```

### 6.3 Inference with Transformer

```python
from src.rom_model.visualizer import ROMVisualizer

visualizer = ROMVisualizer(
    model_dir="models/lhs",
    model_type="transformer"  # Must match training model_type
)

# Predict on new input
visualizer.visualise(L=15.0, w=2.0, d=2.0, P=250.0)
```

---

## 7. Why Attention Works for FEA

### 7.1 Long-Range Dependencies

Transformers excel when nodes need to consider information far away:
- A load at node (20, 2, 2) affects displacement all the way to the fixed end (0, 0, 0)
- GCNs need many layers for this information to propagate
- Transformers capture it in layer 1 via attention

### 7.2 Learned Importance Weights

Unlike GCNs (fixed topology), each Transformer head learns:
- Which nodes are "sources" of information (high in query)
- Which nodes are "sinks" (high in key)
- What information to extract (value projection)

This flexibility can discover non-obvious patterns in FEA data.

### 7.3 Flexibility in Mesh Order

While positional embeddings ensure position awareness, the flexibility of attention
means the model doesn't *require* a specific node ordering — any permutation works
(as long as positional encoding is consistent).

---

## 8. Comparison: MLP vs GCN vs Transformer

| Aspect | MLP | GCN | Transformer |
|---|---|---|---|
| **Complexity** | O(N) | O(N²) per layer | O(N²) per layer |
| **Receptive field** | Global (1 layer) | Grows with layers | Global (1 layer) |
| **Mesh topology** | ❌ | ✅ (fixed adjacency) | ❌ (needs pos. encoding) |
| **Attention/Importance** | Implicit | Fixed (Â) | ✅ Learned dynamics |
| **Depth risk** | None | Over-smoothing | Stable (ResNets help) |
| **Inference speed** | Fastest | Medium | Slowest |
| **Parameters** | ~723K | ~554K | ~550K |
| **For 756-node mesh** | ✅ Works | ✅ Works | ✅ Works |

---

## 9. Empirical Trade-offs on Small Meshes

For the `21×6×6` mesh:

- **MLP** is fastest and often *competitive* in accuracy — the physics is smooth, so a dense network
  learns the mapping efficiently.
- **GCN** exploits topology but adds overhead for a small mesh. Advantage grows with mesh size.
- **Transformer** has the highest capacity (adaptive attention) but also highest training cost.
  Useful if the model discovers important non-local patterns.

---

## 10. Limitations of Transformers for FEA

1. **Memory & Compute** — O(N²) attention scales poorly to millions of nodes. GCNs (O(degree²)) are more scalable.
2. **Fixed Mesh Size** — Positional embeddings are tied to N=756. Can't easily transfer to different mesh sizes.
3. **No Inductive Structure** — Unlike GCNs, Transformers don't *encode* the mesh structure; they must learn it.
4. **Position Dependency** — Sensitive to node ordering; positional encoding design is critical.

---

## 11. Extensions & Variants

| Variant | Key Idea | Use Case |
|---|---|---|
| **Local Attention** | Attend only to nearby nodes (e.g., within distance k) | Large meshes |
| **Sparse Attention** | Pattern-based attention (e.g., every k-th node) | Scaling to millions |
| **Hierarchical** | Coarsened mesh representation + upsampling | Multi-scale physics |
| **Cross-Attention** | Attend between different feature spaces (e.g., load vs stress) | Multi-task learning |
| **Positional Bias** | Encode mesh coordinates directly in attention <math>Q_i \cdot K_j / \sqrt{d_k} + \text{bias}(i,j)</math> | Explicit geometry |

---

## 12. Quick Reference: Transformer Equations

```
# Input processing
x^{(0)} = Dense(params)  →  (B, N, 64)
x^{(0)} ← x^{(0)} + PosEmbed(indices)

# Transformer block k (repeated 3 times)
Q, K, V = Linear(x^{(k)})
attn = softmax(Q K^T / √d_k) V
x^{(k+0.5)} = x^{(k)} + attn
x^{(k+0.5)} = LayerNorm(x^{(k+0.5)})

ffn = Dense_2(ReLU(Dense_1(x^{(k+0.5)})))
x^{(k+1)} = x^{(k+0.5)} + ffn
x^{(k+1)} = LayerNorm(x^{(k+1)})

# Readout
graph_vec = GlobalMeanPool(x^{(3)})
output = Dense(graph_vec)  →  (B, output_dim)
```

---

## References

- Vaswani, A. et al. (2017). *Attention is All You Need*. NeurIPS 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Velic, M. et al. (2023). *Graph Attention Networks*. ICLR 2018.
- Liang, P. P. et al. (2022). *What Makes Training Multi-modal Classification Networks Hard?* — explores architectures for complex data.
- Dwivedi, V. & Bresson, X. (2021). *A Generalization of Transformer Networks to Graphs*. [arXiv:2012.09136](https://arxiv.org/abs/2012.09136)
