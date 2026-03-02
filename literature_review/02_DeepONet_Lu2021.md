# DeepONet: Learning Nonlinear Operators - Lu et al., 2021

**arXiv:** https://arxiv.org/abs/1910.03193
**DOI:** 10.48550/arXiv.1910.03193
**Status:** Published in Nature Machine Intelligence, 2021

## Summary

DeepONet learns operators by decomposing them into an **operator network** (branch) that processes parameters and a **function network** (trunk) that encodes spatial locations. The two networks interact to produce output values.

## Key Innovation: The Two-Network Architecture

```
Parameters → [Branch Network] → operator embedding (e.g., 128-dim)
                                    ↓ (multiply/inner product)
Spatial locations (x,y,z) → [Trunk Network] → basis embeddings (128-dim)
                                    ↓
                            Output at that location
```

This decomposition enables:

1. **Arbitrary domains**: Trunk evaluates at any spatial point
2. **Transfer learning**: Train branch on one domain, use trunk for different domain
3. **Generalization**: Branch captures parameter effects, trunk captures spatial variations

## Architecture Details

### Branch Network

```python
# Takes: parameter vector (4-dim: length, width, depth, load)
# Outputs: embedding vector (e.g., 128-dim)
# Structure: Dense → Dense → Dense (fully connected)
```

### Trunk Network

```python
# Takes: spatial coordinates (3-dim: x, y, z)
# Outputs: basis embeddings (128-dim)
# Structure: Dense → Dense → Dense with positional encoding
```

### Combining Networks

```python
# Operator output at location (x,y,z) with parameters p:
# u(x,y,z; p) = sum_j [ branch_j(p) * trunk_j(x,y,z) ]
# Or: u(x,y,z; p) = branch(p) · trunk(x,y,z)  [inner/outer product]
```

## Why It Works for Your Problem

### **Arbitrary Mesh Support**

- Trunk evaluates at ANY spatial coordinates
- Don't need structured grid - works with scattered point clouds
- Can evaluate at non-grid points (interpolation points)

### **Compositional Structure**

- Branch learns: "how parameters affect the solution"
- Trunk learns: "spatial basis functions"
- Separated concerns = better generalization

### **Flexible Resolution**

- Evaluate trunk at arbitrary number of spatial points
- Train on 756 points, evaluate at 5000 points (or 100 points)
- No retraining needed

## Comparison to Your Current GCN

| Aspect            | GCN                   | DeepONet                     |
| ----------------- | --------------------- | ---------------------------- |
| Mesh Type         | Structured graphs     | ANY point set                |
| Node Count        | Fixed                 | Variable (flexible)          |
| Implementation    | Message passing       | Two networks + output layer  |
| Physics awareness | Explicit connectivity | Implicit in branch network   |
| Generalization    | What you're building  | Excellent across resolutions |

## Comparison to FNO

| Aspect           | FNO                      | DeepONet                       |
| ---------------- | ------------------------ | ------------------------------ |
| Mesh Type        | Regular grids            | Arbitrary points               |
| Assumptions      | Translational invariance | None                           |
| Complexity       | Very complex (FFT)       | Simpler (Dense layers)         |
| Speed            | Very fast                | Fast                           |
| Super-resolution | Built-in                 | Needs evaluation at new points |
| Implementation   | JAX/PyTorch              | Any framework                  |

## Example Implementation (PyTorch)

```python
class DeepONet(nn.Module):
    def __init__(self, param_dim=4, coord_dim=3, hidden_dim=128, output_dim=2):
        super().__init__()
        # Branch: parameters → basis coefficients
        self.branch = nn.Sequential(
            nn.Linear(param_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim)
        )
        # Trunk: coordinates → basis functions
        self.trunk = nn.Sequential(
            nn.Linear(coord_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim)
        )
        # Output layer
        self.out = nn.Linear(hidden_dim, output_dim)
  
    def forward(self, params, coords):
        # params: (batch, 4)
        # coords: (batch, n_points, 3)
        branch_out = self.branch(params)  # (batch, hidden)
        trunk_out = self.trunk(coords)    # (batch, n_points, hidden)
      
        # Expand for broadcasting
        branch_out = branch_out.unsqueeze(1)  # (batch, 1, hidden)
      
        # Hadamard product (element-wise multiply)
        combined = branch_out * trunk_out  # (batch, n_points, hidden)
      
        output = self.out(combined)  # (batch, n_points, output_dim)
        return output
```

## Strengths

1. **Simplicity**: Just Dense layers, no graph operations
2. **Universality**: Works on ANY spatial discretization
3. **Scalability**: O(N) in number of points (not N²)
4. **Flexibility**: Easy to extend (add more trunk basis functions)
5. **Transfer Learning**: Pre-train branch on one geometry, use with different trunk

## Weaknesses

1. **No Spatial Connectivity**: Doesn't explicitly use mesh connectivity
2. **Less Physics-Aware**: Branch doesn't see spatial structure
3. **Scaling Challenge**: Hidden dimension must grow with complexity
4. **Generalization**: May not generalize far beyond training distribution

## Recommended Experiment

```python
# Train DeepONet on standard 21×6×6 mesh
# Test on:
#   - Same mesh (baseline)
#   - Different mesh: 30×8×8 (extra points evaluated at same locations)
#   - Interpolated mesh: evaluate trunk at new random points
#   - Downsampled: evaluate at subset of original points
```

## When to Use DeepONet

✅ **Use DeepONet if**:

- You need unstructured mesh support
- You want lightweight, fast inference
- Your geometries have variable shapes/features
- You need flexible evaluation (any # of points)

❌ **Don't use if**:

- You have complex mesh topology (curved surfaces)
- You need explicit connectivity (contact, interactions)

## Implementation Frameworks

- **TensorFlow**: Simple with Keras layers ✓ (your preference)
- **PyTorch**: Also straightforward
- **JAX**: Efficient for large batches

## Papers to Read Next

1. **DeepONet original** - Section 3: Architecture details
2. **FNO paper** - for comparison
3. **Transformer for point clouds** - for advanced version

## Roadmap Integration

- **Phase 1**: Understand the concept (done!)
- **Phase 2**: Implement simple DeepONet
- **Phase 3**: Compare with GCN/FNO
- **Phase 4**: Hybrid approach (DeepONet branch + GCN trunk)

---

**My Recommendation**: **Start with DeepONet** for your generalization goal. It's:

- Easier to implement than GCN/FNO
- More flexible with mesh types
- Perfect for your "any mesh" requirement
- Naturally handles variable node counts

**Then compare** with current GCN to see which generalizes better.
