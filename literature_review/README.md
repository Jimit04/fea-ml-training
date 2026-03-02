# Literature Review: Mesh-Independent & Shape-Independent ML for FEA

This folder contains curated research papers and resources for building general-purpose neural network-based surrogate models for finite element analysis (FEA).

## Quick Download Links

All papers are freely available on arXiv. Use these direct links:

### **Core Papers (Essential Reading)**

1. **Fourier Neural Operator for Parametric PDEs** - *Li et al., 2021*
   - arXiv: https://arxiv.org/abs/2010.08895
   - DOI: 10.48550/arXiv.2010.08895
   - Key: Resolution-agnostic operator learning, super-resolution capabilities

2. **DeepONet: Learning Nonlinear Operators for Identification and Control of PDEs** - *Lu et al., 2021*
   - arXiv: https://arxiv.org/abs/1910.03193
   - DOI: 10.48550/arXiv.1910.03193
   - Key: Arbitrary domain support, function-operator pairs

3. **Graph Isomorphism Networks** - *Xu et al., 2019*
   - arXiv: https://arxiv.org/abs/1810.00826
   - DOI: 10.48550/arXiv.1810.00826
   - Key: Permutation invariance, expressive power for graph-structured data

4. **Self-Attention Graph Pooling** (SAGPool) - *Lee et al., 2019*
   - arXiv: https://arxiv.org/abs/1904.08187
   - DOI: 10.48550/arXiv.1904.08187
   - Key: Hierarchical graph learning, multi-scale processing

### **Supporting Papers (Recommended)**

5. **Physics-Informed Neural Networks (PINNs)** - *Raissi et al., 2019*
   - arXiv: https://arxiv.org/abs/1711.10561
   - DOI: 10.48550/arXiv.1711.10561
   - Key: Incorporating physics constraints, learning governing equations

6. **SciANN: Physics-Informed Deep Learning for Scientific Computing** - *Haghighat & Juanes, 2020*
   - arXiv: https://arxiv.org/abs/2005.08803
   - DOI: 10.48550/arXiv.2005.08803
   - Key: Keras-based framework, practical implementation guidelines

7. **Attention is All You Need (Transformers)** - *Vaswani et al., 2017*
   - arXiv: https://arxiv.org/abs/1706.03762
   - DOI: 10.48550/arXiv.1706.03762
   - Key: Foundation for graph transformers, permutation-invariant architectures

8. **Spectral Networks and Locally Connected Networks on Graphs** - *Bruna et al., 2014*
   - arXiv: https://arxiv.org/abs/1312.6203
   - DOI: 10.48550/arXiv.1312.6203
   - Key: Early graph convolution theory, spectral methods

### **Industry Application Paper (Real-World Implementation)**

9. **AI-Enhanced CAE Simulations: A Revolutionary Approach to Automotive Design and Engineering** - *Patil & Sonavane, 2025*
   - DOI: https://doi.org/10.4271/2025-01-8241
   - Publisher: SAE International (WCX World Congress Experience)
   - Access: https://saemobilus.sae.org/papers/ai-enhanced-cae-simulations-a-revolutionary-approach-automotive-design-engineering-2025-01-8241
   - Key: Practical ML integration into automotive CAE, 30% speedup, durability/thermal predictions, variable geometries

## Reading Guide by Phase

### For Phase 2 (Variable Mesh Data Generation)
- Start with: **Graph Isomorphism Networks** #3
- Then: **SAGPool** #4
- Reference: **Spectral Networks** #8

### For Phase 3 (Model Architecture)
- Core: **FNO** #1 + **DeepONet** #2
- Implement: **Graph Isomorphism Networks** #3
- Enhance: **Self-Attention Graph Pooling** #4

### For Phase 4-6 (Training & Production)
- Physics guidance: **PINNs** #5 + **SciANN** #6
- Attention mechanisms: **Transformers** #7

## How to Download PDFs

### **Option 1: Direct from arXiv** (Recommended)
```bash
# Click "pdf" link on arXiv page
# Or use curl:
curl -o paper.pdf https://arxiv.org/pdf/2010.08895.pdf
```

### **Option 2: Using arXiv-dl (Python)**
```bash
pip install arxiv-dl
arxiv-dl 2010.08895  # Downloads to current directory
```

### **Option 3: Institutional Access**
- If your institution has access to IEEE Xplore or other repositories
- Check your university library for extended access

## Key Concepts from SOTA

| Concept | Paper | Relevance |
|---------|-------|-----------|
| **Operator Learning** | FNO, DeepONet | Learn mappings between function spaces (params→fields) |
| **Mesh Independence** | FNO | Super-resolution & different input resolutions |
| **Permutation Invariance** | GIN | Graph order doesn't matter, robust to mesh reordering |
| **Hierarchical Processing** | SAGPool | Coarse-to-fine representations for variable mesh sizes |
| **Physics Constraints** | PINNs | Incorporate BC/material laws into loss function |
| **Geometry Awareness** | All | Embed coordinates & connectivity as input features |

## Implementation References

- **PyTorch Geometric** (for GNNs): https://pytorch-geometric.readthedocs.io/
- **JAX** (for FNO): https://github.com/google/jax
- **TensorFlow Keras** (your current stack): https://keras.io/api/layers/

## Citation Format (BibTeX)

See `references.bib` in this folder for ready-to-use citations.

## Notes

- All papers are published/preprints and freely available on arXiv
- Dates shown are submission dates; many have newer versions
- Focus on papers from 2015-2023 for current SOTA alignment
- Some papers may require reading related work sections for context

---

**Last Updated**: March 2, 2026
**Curated for**: Mesh/Shape-Independent FEA ML Models
