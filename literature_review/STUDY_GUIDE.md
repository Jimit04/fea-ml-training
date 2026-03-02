# Study Guide: From Papers to Implementation

A structured path for implementing mesh-independent FEA ML models using SOTA techniques.

## Your Challenge

Build a single neural network model that:
1. ✓ Works with **any mesh density** (coarse, fine, ultra-fine)
2. ✓ Generalizes across **different beam geometries** (length, width, depth)
3. ✓ Runs **fast** for real-time prediction
4. ✓ Has **physics-aware** predictions

## Why These Papers?

| Paper | Solves | Key Concept |
|-------|--------|-------------|
| **FNO** | Arbitrary input resolution | Function space operators |
| **DeepONet** | Unstructured domains | Operator + trunk decomposition |
| **GIN** | Graph expressiveness | Permutation-invariant learning |
| **SAGPool** | Variable graph sizes | Learned graph compression |
| **PINNs** | Physics guarantee | Constraint satisfaction |

## Recommended Learning Path

### Week 1: Understand Current Limitations

**Task**: Run your current model on different mesh densities
```python
# Train on: 21×6×6
# Test on:
#   - Same mesh: 21×6×6 (baseline)
#   - Finer: 30×8×8 (generalize?)
#   - Coarser: 15×4×4 (generalize?)
# Measure: % error increase
```

**Read**: None (implementation-focused)

### Week 2: Theory of Graph Learning

**Papers**: GIN → SAGPool  
**Time**: 4-6 hours  
**Goals**:
- [ ] Understand permutation invariance (why it matters)
- [ ] Know what "expressive power" means for graphs
- [ ] See why GCN might miss important patterns

**Activities**:
```python
# Experiment 1: GCN permutation invariance
# Shuffle node indices → predictions should be same
indices_orig = [0, 1, 2, ..., 755]
indices_shuffle = np.random.permutation(indices_orig)

# Test: model(X, A) == model(X[indices_shuffle], A[indices_shuffle])
```

### Week 3: Variable Resolution Strategies

**Papers**: FNO + DeepONet  
**Time**: 6-8 hours  
**Goals**:
- [ ] Understand 3 different paradigms for variable meshes
- [ ] Know when to use each approach
- [ ] Design experiments to test each

**Decision Tree**:
```
Do you have structured grids? 
  → YES: Use FNO (fast, super-resolution)
  → NO:  Use DeepONet or GCN

Do you have unstructured meshes?
  → YES: Use DeepONet (handles any points)
  → NO:  Use GCN or FNO

Do you need hierarchy?
  → YES: Add SAGPool to any of above
  → NO:  Keep it simple
```

### Week 4: Implement Multi-Approach Comparison

**Task**: Code 3 models
1. **Model A**: Current GCN + variable mesh handling
2. **Model B**: DeepONet (branch + trunk)
3. **Model C**: FNO (if structured grids)

**Comparison metrics**:
- Training time per epoch
- Inference speed
- Generalization (% error on unseen mesh)
- Memory usage

### Week 5: Physics Integration

**Paper**: PINNs  
**Time**: 4-6 hours  
**Goals**:
- [ ] Understand physics loss design
- [ ] Know boundary conditions for cantilever
- [ ] Implement derivativevolutionary computation

**Implementation**:
```python
# Add to training loop:
for params, displacements_true, stresses_true in train_loader:
    # Standard data loss
    disp_pred = model(params)
    loss_data = mse(disp_pred, displacements_true)
    
    # Physics loss (optional, add later)
    # loss_physics = compute_pinn_loss(params, disp_pred)
    # loss = loss_data + 0.1 * loss_physics
    
    # Backprop and update
```

## Study Schedule: 2 Weeks Intensive

### Day 1: Foundations
- **Read**: GIN paper (intro + method)
- **Implement**: Modify GCN to use coordinates as input
- **Test**: Permutation invariance experiment

### Day 2: Multi-Scale
- **Read**: SAGPool paper
- **Implement**: Add pooling layers to GCN
- **Test**: Compare unpooled vs pooled on same data

### Day 3-4: Alternative Paradigms
- **Read**: FNO paper (skip derivations)
- **Read**: DeepONet paper
- **Decision**: Which is best for your problem?

### Day 5: Implementation
- **Code**: DeepONet skeleton (simpler than FNO)
- **Test**: DeepONet vs GCN on same training data
- **Benchmark**: Speed, accuracy, generalization

### Day 6: Data Generation
- **Code**: Multi-mesh data generation
- **Generate**: Same beam on coarse/medium/fine meshes
- **Analyze**: How many samples needed for each approach?

### Day 7: Physics Constraints
- **Read**: PINNs paper (method section)
- **Code**: PINN loss function
- **Experiment**: Data-only vs PINN-augmented training

### Day 8-10: Comparisons & Optimization
- **Run**: Train all 3 models (GCN, DeepONet, FNO)
- **Test**: Mesh generalization tests
- **Select**: Best performing architecture
- **Optimize**: Hyperparameter tuning

### Day 11-14: Refinement & Documentation
- **Improve**: Add SAGPool to best model
- **Add**: PINN loss if applicable
- **Benchmark**: Final comparison vs original approach
- **Document**: Architecture decisions, results, next steps

## Quick Reference: Concepts

### Permutation Invariance
Neural networks should produce same output regardless of input order.
```
[1, 2, 3, 4] → Model → [result]
[4, 3, 2, 1] → Model → [result]  ← Should be SAME
```
**Why matters**: Mesh nodes can be numbered differently.

### Expressiveness
Some models can learn all graph patterns, others miss some.
```
GCN: Can represent ~60% of possible graph properties
GIN: Can represent ~100% of possible graph properties (proven)
```
**Why matters**: More expressive = better generalization.

### Operator Learning
Learn mapping **between function spaces**, not vectors.
```
Standard: (4 numbers) → neural network → (756 numbers)
Operator: (function: params→field) → network → (new function)
```
**Why matters**: Generalizes to any sample density.

### Multi-Scale Learning
Process information at different granularities.
```
Fine scale:   Captures local stress concentrations
Medium scale: Captures beam bending
Coarse scale: Overall load response
All together → Complete understanding
```
**Why matters**: Your mesh has varying local features.

### Physics-Informed Loss
Let equations guide learning, not just data.
```
Loss = Fit_to_data + λ × Satisfy_physics
```
**Why matters**: Better extrapolation, fewer samples needed.

## Experiments to Run

### Experiment 1: Mesh Sensitivity
```python
train_mesh = MeshGeometry(21, 6, 6)  # Current
test_meshes = [
    MeshGeometry(10, 3, 3),  # Coarse
    MeshGeometry(21, 6, 6),  # Same
    MeshGeometry(35, 9, 9),  # Fine
]

for test_mesh in test_meshes:
    error = evaluate_model(model, test_mesh)
    print(f"Mesh {test_mesh}: {error:.2f}%")
```

### Experiment 2: Geometry Generalization
```python
# Train on: 100-200mm length
# Test on: 210mm (extrapolated)
# Measure: Comparison vs FEA ground truth
```

### Experiment 3: Model Comparison
```python
models = [
    ("Current GCN", current_model),
    ("GCN + SAGPool", pooled_model),
    ("DeepONet", deeponet),
    ("GCN + PINN Loss", pinn_model),
]

metrics = ["Train Time", "Inference Speed", "Mesh Generalization", "Memory"]
# Create comparison table
```

### Experiment 4: Data Efficiency
```python
# Train with different dataset sizes
for n_samples in [50, 100, 200, 400, 600]:
    model = train(n_samples)
    error = evaluate(model)
    print(f"{n_samples} samples: {error:.2f}% error")

# Compare: Data-only vs PINN-augmented
```

## Decision Framework

At the end of week 2, answer these:

**Q1**: Does FNO work for your structured grids?  
→ If YES: Consider FNO (fast, proven)  
→ If NO:  Skip FNO

**Q2**: Do you have unstructured meshes?  
→ If YES: Must use DeepONet or keep GCN  
→ If NO:  All options viable

**Q3**: How important is interpretability?  
→ If HIGH: GCN + SAGPool (connection-aware)  
→ If LOW:  DeepONet (simpler)

**Q4**: How sparse is your training data?  
→ If VERY SPARSE: Must use PINNs  
→ If ABUNDANT:  Optional

## Final Architecture Recommendation

Based on your current setup & requirements:

```
Recommended: GCN with SAGPool + Optional PINN loss
├─ Why GCN: Already implemented, fast
├─ Why SAGPool: Handles variable mesh sizes
├─ Why PINN: Improves generalization with physics
└─ Upgrade path: Simple additions to existing code
```

Implementation effort: **2-3 weeks** for production-ready model

## Resources

### Code Templates
- **GIN in PyG**: https://pytorch-geometric.readthedocs.io/
- **SAGPool example**: Same library
- **DeepONet template**: See `02_DeepONet_Lu2021.md`
- **PINN template**: See `05_PINNs_Raissi2019.md`

### Datasets for Pretraining
- **ModelNet40**: 3D shape meshes
- **ShapeNet**: Large CAD model database  
- **Synthetic FEA**: Generate on various mesh densities

### Benchmark Problems
- **Darcy flow**: Test on different resolutions (FNO paper)
- **Cantilever beam**: Your exact problem
- **Bracket optimization**: More complex geometry

## Reading Order (If Starting Fresh)

1. **5 min**: This document (you're here!)
2. **1 hour**: GIN paper (conceptual foundation)
3. **1 hour**: SAGPool paper (practical technique)
4. **30 min**: DeepONet summary (alternative paradigm)
5. **30 min**: FNO summary (if structured grids)
6. **1 hour**: PINNs (physics integration)
7. **Code**: Implement incrementally

## Success Metrics

Your model succeeds when:

- [ ] Trains on 21×6×6 mesh
- [ ] Evaluates on 30×8×8 mesh with <10% accuracy loss
- [ ] Evaluates on 15×4×4 mesh with <10% accuracy loss
- [ ] Inference time < 1ms per sample
- [ ] Memory usage < 100MB
- [ ] Generalizes to unseen geometry parameters
- [ ] Satisfies boundary conditions (after PINN)

## Troubleshooting Guide

### Problem: Model fails on coarse mesh
**Cause**: Network expects too much detail  
**Solution**: Add multi-scale training (SAGPool helps)

### Problem: Model fails on fine mesh
**Cause**: Network overtrained on medium density  
**Solution**: Curriculum learning (train coarse → fine)

### Problem: Poor generalization to new geometries
**Cause**: Physics constraints missing  
**Solution**: Add PINN loss

### Problem: Training too slow
**Cause**: Dense adjacency matrix operations  
**Solution**: Use sparse matrix formats in PyG

## Next Steps After This Roadmap

1. Update your GCN to use coordinates as features
2. Generate multi-mesh training data
3. Implement SAGPool layers
4. Run comparison experiments
5. Integrate PINN loss
6. Benchmark against baseline


---

**Ready to start?** Pick Day 1 activities above and begin!
