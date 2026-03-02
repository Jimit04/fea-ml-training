# Literature Review Folder - Navigation Guide

You now have a complete, curated literature review resource for mesh-independent FEA ML models. Here's how to navigate it:

## 📚 Files in This Folder

### Start Here 👈

1. **README.md** - Overview of all papers with direct arXiv links
   - Quick reference table
   - Download links
   - Citation formats
   - Which paper to read when

2. **DOWNLOAD.md** - How to get the PDFs
   - Multiple download methods (curl, Python, wget)
   - Citation management tools
   - Folder organization tips
   - Troubleshooting

3. **STUDY_GUIDE.md** - Your learning roadmap
   - 2-week intensive study plan
   - Day-by-day schedule
   - Experiments to run
   - Decision framework
   - Success metrics

### Paper Summaries (Detailed Explanations)

4. **01_FNO_Li2021.md** - Fourier Neural Operators
   - For: Structured grids, super-resolution
   - Complexity: Medium
   - TL;DR: Learn in frequency domain

5. **02_DeepONet_Lu2021.md** - Operator Learning
   - For: Any point cloud/mesh type
   - Complexity: Low
   - TL;DR: Branch (params) + Trunk (coords)

6. **03_GIN_Xu2019.md** - Graph Isomorphism Networks
   - For: Permutation-invariant learning
   - Complexity: Medium
   - TL;DR: GCN upgrade with proven expressiveness

7. **04_SAGPool_Lee2019.md** - Hierarchical Graph Pooling
   - For: Variable mesh sizes
   - Complexity: Medium
   - TL;DR: Learned graph compression

8. **05_PINNs_Raissi2019.md** - Physics-Informed Networks
   - For: Constraint satisfaction
   - Complexity: Medium
   - TL;DR: Add physics to loss function

### References

9. **references.bib** - BibTeX for all citations
   - Ready to copy-paste into your LaTeX docs
   - All 10+ papers included
   - Update as you read more

---

## 🗺️ Quick Navigation by Goal

### "I want to understand the big picture"
→ Start with: **README.md** (5 min) → **STUDY_GUIDE.md** (30 min)

### "I'm implementing DeepONet"
→ Read: **02_DeepONet_Lu2021.md** (1 hour) → Code examples included

### "I need to handle variable mesh sizes"
→ Read: **04_SAGPool_Lee2019.md** (1 hour) + **03_GIN_Xu2019.md** (45 min)

### "I want physics constraints"
→ Read: **05_PINNs_Raissi2019.md** (1 hour) → Implementation code provided

### "I'm choosing which architecture to implement"
→ Read: **STUDY_GUIDE.md** section "Decision Framework"

### "I want the papers as PDFs to read"
→ Use: **DOWNLOAD.md** → Follow curl/Python/wget commands

### "I need to cite these papers"
→ Copy from: **references.bib** → Paste into your LaTeX/Markdown

---

## ⏱️ Reading Time Estimates

| Document | Time | Purpose |
|----------|------|---------|
| README.md | 5 min | Overview |
| DOWNLOAD.md | 5 min | Getting PDFs |
| STUDY_GUIDE.md | 1 hour | Full learning path |
| 01_FNO | 1 hour | Understanding FNO |
| 02_DeepONet | 1 hour | Understanding DeepONet |
| 03_GIN | 45 min | Understanding GIN |
| 04_SAGPool | 1 hour | Understanding SAGPool |
| 05_PINNs | 1 hour | Understanding PINNs |
| **Total** | **~6.5 hours** | Complete understanding |

---

## 🎯 Recommended Reading Orders

### Path A: GCN Improvement (Minimal Changes)
1. **03_GIN_Xu2019.md** (upgrade your GCN)
2. **04_SAGPool_Lee2019.md** (add variable mesh support)
3. **05_PINNs_Raissi2019.md** (improve generalization)
4. **STUDY_GUIDE.md** (experiments section)

**Effort**: 2-3 weeks | Uses existing TensorFlow stack

### Path B: Complete Redesign (Better Results)
1. **02_DeepONet_Lu2021.md** (understand new architecture)
2. **01_FNO_Li2021.md** (understand alternatives)
3. Implement DeepONet
4. Compare with current GCN
5. **03_GIN_Xu2019.md** (upgrade if keeping GCN)

**Effort**: 3-4 weeks | More complex, potentially better performance

### Path C: Physics-First (For Safety-Critical Apps)
1. **05_PINNs_Raissi2019.md** (understand physics loss)
2. **03_GIN_Xu2019.md** (better graph model)
3. Implement GCN + SAGPool + PINN loss
4. Validate physics satisfaction

**Effort**: 2-3 weeks | Best generalization

---

## 💡 What Each Paper Solves

```
Your Problem Areas          ← Solution Papers
─────────────────────────────────────────────
Fixed mesh resolution       → FNO, DeepONet
Variable mesh density       → SAGPool, DeepONet
Generalization to new geo   → PINNs, FNO
Slow inference              → GIN (same speed as GCN)
Graph expressiveness        → GIN, SAGPool
Physics constraints         → PINNs
Unstructured meshes         → DeepONet
```

---

## 📖 Paper Summaries Content

Each paper summary (01-05) includes:

- **What it is** - Plain English explanation
- **Key innovation** - What makes it special
- **Why for your problem** - Specific relevance
- **Architecture details** - How it works
- **Pros vs cons** - Honest assessment
- **Code examples** - Implementation snippets
- **When to use** - Decision criteria
- **Next steps** - How to implement
- **Comparison table** - vs your current approach

---

## 🔍 Cross-Reference Index

### By Concept

**Graph Neural Networks**:
- Current baseline: 02_DeepONet_Lu2021, 03_GIN_Xu2019
- Variable sizes: 04_SAGPool_Lee2019
- Theory: 03_GIN_Xu2019

**Resolution Independence**:
- Super-resolution: 01_FNO_Li2021
- Arbitrary points: 02_DeepONet_Lu2021
- Hierarchical: 04_SAGPool_Lee2019

**Physics Constraints**:
- Main reference: 05_PINNs_Raissi2019
- Integration with GNNs: STUDY_GUIDE.md

**Implementation**:
- Keras/TensorFlow: 02_DeepONet_Lu2021, 05_PINNs_Raissi2019
- PyTorch: All papers have PT code
- PyG: 03_GIN_Xu2019, 04_SAGPool_Lee2019

### By Mathematical Topic

**Fourier Analysis**: 01_FNO_Li2021  
**Operator Theory**: 02_DeepONet_Lu2021  
**Graph Theory**: 03_GIN_Xu2019  
**Calculus/Autodiff**: 05_PINNs_Raissi2019  
**Spectral Methods**: 01_FNO_Li2021  

---

## 🚀 Next Actions

### Immediate (Today)
1. Skim this navigation guide
2. Read README.md
3. Bookmark STUDY_GUIDE.md

### This Week
4. Choose one paper from 01-05
5. Read its summary (30-60 min)
6. Check if implementation makes sense
7. Download PDF if needed (using DOWNLOAD.md)

### Next Week
8. Start implementation based on summary
9. Reference the code examples provided
10. Run experiments from STUDY_GUIDE.md

### Following Week
11. Compare results to your baseline
12. Iterate & optimize
13. Read next paper in sequence

---

## ❓ FAQ

**Q: Do I need to read all the papers?**  
A: No! Start with your most relevant goal and read that summary. You can skip others.

**Q: Can I implement without reading the papers?**  
A: Yes, use code examples in each summary. But understanding helps you debug.

**Q: Which is best for my problem?**  
A: Use STUDY_GUIDE.md "Decision Framework" section.

**Q: How do I stay current?**  
A: See DOWNLOAD.md "Staying Updated" section for conferences, arXiv, Twitter.

**Q: Can I share these summaries?**  
A: Yes! Summaries are original. Papers are from arXiv (open access).

**Q: What if I want to read the actual papers?**  
A: Use DOWNLOAD.md for direct download links.

**Q: These are new to you/outdated?**  
A: Papers span 2014-2021, core concepts are timeless. Check dates in README.md.

---

## 📊 Study Statistics

| Metric | Value |
|--------|-------|
| Papers included | 5 core + 10 supporting |
| Total reading time | 6-8 hours (core) |
| Implementation examples | 20+ code snippets |
| Experiments to try | 8+ detailed experiments |
| BibTeX entries | 15 papers |
| Markdown content | ~15,000 words |

---

## 🏆 What You'll Know After This

✓ Why your current model has limitations  
✓ 3-5 different architectures for mesh independence  
✓ How to handle variable mesh sizes  
✓ How to incorporate physics constraints  
✓ Which approach is best for your specific goal  
✓ How to implement & test each approach  
✓ How to benchmark against baselines  

---

**Created**: March 2, 2026  
**Status**: Ready to use  
**Updates**: Check this folder for new papers as you discover them  

**Questions?** Reference the paper summaries first. If stuck, read related section in STUDY_GUIDE.md.

**Ready to dive in?** Start with **README.md** → Pick a goal → Read corresponding summary!
