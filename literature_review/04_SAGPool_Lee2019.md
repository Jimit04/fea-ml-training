# Self-Attention Graph Pooling (SAGPool) - Lee et al., 2019

**arXiv:** https://arxiv.org/abs/1904.08187
**DOI:** 10.48550/arXiv.1904.08187
**Status:** Published in ICML 2019

## Summary

SAGPool introduces **learned, hierarchical graph pooling** using self-attention. It allows neural networks to compress graphs (remove less important nodes) while maintaining learned representations — perfect for handling variable mesh densities.

## The Core Problem

Your mesh can have 100 nodes or 5000 nodes. How do you process both with the same network?

```
Standard approach:
  Pad small graphs to 5000 nodes ✗ (wastes memory)
  Train separate models ✗ (inefficient)
  Fixed resolution only ✗ (not mesh-independent)

SAGPool approach:
  Learn which nodes are important
  Compress graph intelligently
  Process at multiple scales ✓
```

## Key Idea: Attention-Based Node Selection

```python
# For each node, compute importance score
scores = self_attention(node_features)  # (N,) scores

# Keep top-K% nodes (learnable K)
keep_mask = top_k(scores, k=int(0.5 * N))  # Keep 50% of nodes

# Remove unimportant nodes, keep edges
graph_pooled = graph[keep_mask]
```

This is differentiable! The network learns which nodes to keep during backprop.

## Architecture Overview

```
Layer 1:
  [5000 node mesh] → GNN → compute scores
                              ↓
                        select top 50% nodes (2500)
                              ↓
                        [Pooled graph: 2500 nodes]

Layer 2:
  [2500 nodes] → GNN → compute scores
                            ↓
                      select top 50% nodes (1250)
                            ↓
                      [Compressed: 1250 nodes]

Layer 3:
  [1250 nodes] → GNN → compute scores
                            ↓
                      select top 50% nodes (625)
                            ↓
                      [Final: 625 nodes]

Then: Global aggregation (max/mean pooling) for graph-level prediction
```

## Mathematical Formulation

```
Given graph G = (V, E, X) with |V| = N nodes

Step 1: Attention scoring
  z_i = X_i · a  (or MLP(X_i) · a)  for each node i
  p_i = sigmoid(z_i)                  (importance score)

Step 2: Node selection
  idx = top_k(p, k)                   (keep k nodes)

Step 3: Subgraph extraction
  X' = X[idx]                         (selected nodes)
  E' = E[filter(E, idx)]              (selected edges)
  
Step 4: Coarsen adjacency
  A' = A[idx][:, idx]                 (submatrix for selected nodes)
```

## Why This Works for Variable Mesh FEA

### Problem 1: Different Mesh Densities

```
Train mesh: 21×6×6 = 756 nodes
New mesh:   30×8×8 = 1920 nodes

Without pooling:
  Network expects 756 → 2268 params
  Can't process 1920 nodes!

With SAGPool:
  Pool 1920 → 960 nodes (keep 50%) ✓
  Pool 960  → 480 nodes (keep 50%) ✓
  Process at consistent scale ✓
```

### Problem 2: Computational Cost

```
Large mesh: 5000 nodes
GCN pass: 5000² = 25M operations (if fully connected)
SAGPool: 5000² → (2500)² → (1250)² ✓ (efficient pyramidal)
```

## Integration with Your Current Architecture

```python
# Current: [GCN × 6] → GAP → Decoder → Output

# With SAGPool: [GCN + Pool + GCN + Pool] → GAP → Decoder
#
#  Input (N nodes)
#    ↓ [GCN layer] → (N, 128)
#    ↓ [SAGPool 50%] → (N/2, 128) ← learned compression
#    ↓ [GCN layer]  → (N/2, 128)
#    ↓ [SAGPool 50%] → (N/4, 128) ← further compression
#    ↓ [GCN layer]  → (N/4, 128)
#    ↓ [Global Average Pooling] → (128,) ← graph summary
#    ↓ [Dense decoder] → Output
```

## Implementation (PyTorch Geometric)

```python
from torch_geometric.nn import GCNConv, SAGPooling
from torch_scatter import scatter

class PoolingGCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
      
        # Convolution layers
        self.conv1 = GCNConv(input_dim, 128)
        self.conv2 = GCNConv(128, 128)
        self.conv3 = GCNConv(128, 128)
      
        # Pooling layers
        self.pool1 = SAGPooling(128, ratio=0.5)  # Keep 50%
        self.pool2 = SAGPooling(128, ratio=0.5)  # Keep 50%
      
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.Linear(512, output_dim)
        )
  
    def forward(self, x, edge_index, batch):
        # First block
        x = self.conv1(x, edge_index).relu()
        x, edge_index, _, batch, _, _ = self.pool1(
            x, edge_index, batch=batch
        )
      
        # Second block
        x = self.conv2(x, edge_index).relu()
        x, edge_index, _, batch, _, _ = self.pool2(
            x, edge_index, batch=batch
        )
      
        # Final convolution + global pooling
        x = self.conv3(x, edge_index)
        x = scatter(x, batch, dim=0, reduce="mean")  # Mean pooling
      
        # Decode
        out = self.decoder(x)
        return out
```

## Key Benefits for Your Project

| Benefit                            | Impact                                           |
| ---------------------------------- | ------------------------------------------------ |
| **Multi-scale processing**   | Handles coarse and fine meshes naturally         |
| **Learned compression**      | Network decides what's important                 |
| **Computational efficiency** | O(N log N) instead of O(N²)                     |
| **Permutation invariant**    | Works with any node ordering                     |
| **Differentiable**           | End-to-end training                              |
| **Flexible K**               | Can learn different compression ratios per layer |

## Comparison to Other Pooling Methods

| Method                   | How it works                   | Pros           | Cons                  |
| ------------------------ | ------------------------------ | -------------- | --------------------- |
| **Max Pooling**    | Keep max feature value         | Simple         | Loses spatial info    |
| **DiffPool**       | Learn soft cluster assignments | End-to-end     | Expensive (O(N³))    |
| **SAGPool**        | Attention scores + top-K       | Fast + learned | May lose nearby nodes |
| **Global Pooling** | Average/sum all nodes          | Very simple    | No hierarchy          |

**For mesh FEA: SAGPool is best** (fast + learned + interpretable)

## Experimental Results from Paper

- **Graph classification**: Beats baseline by 2-3%
- **Speed**: 10× faster than DiffPool on large graphs
- **Scalability**: Works on graphs with 100K+ nodes
- **Interpretability**: Can visualize which nodes were important

## Challenges & Solutions

### Challenge 1: Which compression ratio?

```python
# Experiment: try different ratios
for ratio in [0.3, 0.5, 0.7, 0.9]:
    pool = SAGPooling(dim, ratio=ratio)
    # Test on different mesh sizes
```

### Challenge 2: How many pooling layers?

```python
# Guideline: Log scale
# Example: 1920 nodes
#   Pool 50%: 960 nodes
#   Pool 50%: 480 nodes
#   Pool 50%: 240 nodes ← usually stop here
```

### Challenge 3: Node features before pooling?

```python
# Best practice: Concatenate node features with scores
# This helps pooling make informed decisions
x = torch.cat([node_features, attention_scores], dim=1)
```

## Roadmap Integration

### Stage 1: Add pooling to existing GCN

Replace:

```python
x = GCNLayer(128)([node_init, a_inp])  # ×6
```

With:

```python
x = GCNLayer(128)(...)
x, a_inp, batch, _ = SAGPool(0.5)(x, a_inp, batch)  # Compress

x = GCNLayer(128)(...)
x, a_inp, batch, _ = SAGPool(0.5)(x, a_inp, batch)  # Compress again

x = torch.scatter(x, batch, reduce='mean')  # Global pool
```

### Stage 2: Multi-scale features

Connect features from different levels back to output (deeper networks).

### Stage 3: Adaptive pooling

Learn the compression ratio per layer:

```python
ratio = self.ratio_mlp(node_features)  # Learns ratio per node
```

## Required Dependencies

```bash
pip install torch-geometric
# Already have: torch, numpy, keras
```

## Papers to Read Next

1. **SAGPool original** - Method section for details
2. **DiffPool** - Alternative (more expensive but no top-K)
3. **Graph U-Net** - Unpooling to recover resolution

## When to Use SAGPool

✅ **Use SAGPool if**:

- You have variable graph sizes
- You want multi-scale processing
- Computational efficiency matters
- Interpretability is desired (see which nodes kept)

❌ **Don't use if**:

- You need to preserve exact node positions
- Lossy compression unacceptable
- Connectivity information is critical

---

**Key Insight**: SAGPool = differentiable "summarization" of graphs. Learns what matters!
