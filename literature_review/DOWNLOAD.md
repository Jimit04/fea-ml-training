# Quick Download Commands

## Download All Papers as PDFs

### Option 1: Using `curl` (Windows PowerShell)

```powershell
# Navigate to literature_review folder
cd literature_review

# FNO - Fourier Neural Operator
curl -o 01_FNO_Li2021.pdf https://arxiv.org/pdf/2010.08895.pdf

# DeepONet - Learning Nonlinear Operators
curl -o 02_DeepONet_Lu2021.pdf https://arxiv.org/pdf/1910.03193.pdf

# GIN - Graph Isomorphism Networks
curl -o 03_GIN_Xu2019.pdf https://arxiv.org/pdf/1810.00826.pdf

# SAGPool - Self-Attention Graph Pooling
curl -o 04_SAGPool_Lee2019.pdf https://arxiv.org/pdf/1904.08187.pdf

# PINNs - Physics-Informed Neural Networks
curl -o 05_PINNs_Raissi2019.pdf https://arxiv.org/pdf/1711.10561.pdf
```

### Option 2: Using Python `urllib`

```python
import urllib.request
import os

papers = {
    "01_FNO_Li2021.pdf": "https://arxiv.org/pdf/2010.08895.pdf",
    "02_DeepONet_Lu2021.pdf": "https://arxiv.org/pdf/1910.03193.pdf",
    "03_GIN_Xu2019.pdf": "https://arxiv.org/pdf/1810.00826.pdf",
    "04_SAGPool_Lee2019.pdf": "https://arxiv.org/pdf/1904.08187.pdf",
    "05_PINNs_Raissi2019.pdf": "https://arxiv.org/pdf/1711.10561.pdf",
}

os.makedirs("literature_review", exist_ok=True)
os.chdir("literature_review")

for filename, url in papers.items():
    print(f"Downloading {filename}...")
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"  ✓ {filename}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
```

Run this script from your workspace root:

```bash
python download_papers.py
```

### Option 3: Using `wget` (if installed)

```bash
cd literature_review
wget https://arxiv.org/pdf/2010.08895.pdf -O 01_FNO_Li2021.pdf
wget https://arxiv.org/pdf/1910.03193.pdf -O 02_DeepONet_Lu2021.pdf
wget https://arxiv.org/pdf/1810.00826.pdf -O 03_GIN_Xu2019.pdf
wget https://arxiv.org/pdf/1904.08187.pdf -O 04_SAGPool_Lee2019.pdf
wget https://arxiv.org/pdf/1711.10561.pdf -O 05_PINNs_Raissi2019.pdf
```

## Supporting Papers (Optional but Recommended)

### Graph Neural Networks Foundations

```bash
# Semi-Supervised Classification with Graph Convolutional Networks
# Kipf & Welling (2017) - Your current GCN is based on this
wget https://arxiv.org/pdf/1609.02907.pdf -O 06_GCN_Kipf2017.pdf

# Relational Inductive Biases, Deep Learning, and Graph Networks
# Battaglia et al. (2018) - Comprehensive survey
wget https://arxiv.org/pdf/1806.01261.pdf -O 07_GraphNetworks_Battaglia2018.pdf

# Inductive Representation Learning on Large Graphs (GraphSAGE)
# Hamilton et al. (2017) - Scalable inductive learning
wget https://arxiv.org/pdf/1706.02216.pdf -O 08_GraphSAGE_Hamilton2017.pdf
```

### Physics-Informed & Scientific ML

```bash
# SciANN: Physics-Informed Deep Learning (Keras/TensorFlow)
# Haghighat & Juanes (2020) - Practical implementation
wget https://arxiv.org/pdf/2005.08803.pdf -O 09_SciANN_Haghighat2020.pdf

# Hamiltonian Neural Networks
# Greydanus et al. (2019) - Physics-preserving networks
wget https://arxiv.org/pdf/1906.04341.pdf -O 10_HNN_Greydanus2019.pdf
```

### Attention & Transformers

```bash
# Attention is All You Need
# Vaswani et al. (2017) - Foundation for graph transformers
wget https://arxiv.org/pdf/1706.03762.pdf -O 11_Transformers_Vaswani2017.pdf

# Spectral Networks and Locally Connected Networks on Graphs
# Bruna et al. (2014) - Early spectral graph convolution theory
wget https://arxiv.org/pdf/1312.6203.pdf -O 12_SpectralNetworks_Bruna2014.pdf
```

### Point Cloud & Mesh Learning

```bash
# Dynamic Graph CNN for Learning on Point Clouds
# Wang et al. (2019) - Point cloud processing
wget https://arxiv.org/pdf/1801.07829.pdf -O 13_DGCNN_Wang2019.pdf

# MeshCNN: A Network with an Edge
# Hanocka et al. (2019) - Direct mesh processing
wget https://arxiv.org/pdf/1809.05910.pdf -O 14_MeshCNN_Hanocka2019.pdf
```

## PDF Reading Tips for Academic Papers

### Efficient Reading Strategy

1. **Title & Abstract** (5 min): Get the big idea
2. **Figures & Tables** (10 min): See what they achieved
3. **Introduction** (10 min): Problem motivation
4. **Method** (20-30 min): Core contribution (read carefully)
5. **Experiments** (10 min): How they validated
6. **References** (2 min): See what built on what

**Total**: 1-2 hours per paper

### Note-Taking Template

```
# Paper: [Title]
## Problem
- What problem does it solve?
- Why is it important?

## Solution
- Key innovation (1-2 sentences)
- Main algorithm (pseudocode)

## Results
- Best metrics achieved
- Comparison to baselines

## For Your Project
- How applicable? (1-5 scale)
- What can you adopt?
- Related work to explore
```

## Browser Tools for PDFs

### Free Annotation

- **PDF.js** (Firefox built-in)
- **Okular** (Linux, free)
- **Preview** (macOS, built-in)
- **Windows 10 Edge** (built-in annotation)

### Advanced (Linux/macOS)

```bash
# Install Zathura (lightweight, fast)
sudo apt install zathura

# Or Skim (macOS)
brew install skim
```

## Citation Management

### BibTeX (Already provided: `references.bib`)

Use in LaTeX documents:

```latex
\documentclass{article}
\bibliography{references.bib}

% In text:
\cite{li2021fourier}  % FNO
\cite{lu2021deeponet}  % DeepONet
```

### Other Tools

- **Zotero**: Free, opensource, web-based https://www.zotero.org/
- **Mendeley**: Freemium, large community
- **Papers3**: Professional, macOS/iOS

## Organizing Your Literature Review

### Folder Structure

```
literature_review/
├── README.md              ← Start here
├── STUDY_GUIDE.md         ← Learning path
├── DOWNLOAD.md            ← This file
├── references.bib         ← All citations
│
├── 01_FNO_Li2021.md       ← Summary (already created)
├── 02_DeepONet_Lu2021.md  ← Summary (already created)
├── 03_GIN_Xu2019.md       ← Summary (already created)
├── 04_SAGPool_Lee2019.md  ← Summary (already created)
├── 05_PINNs_Raissi2019.md ← Summary (already created)
│
├── pdfs/                  ← Downloaded papers
│   ├── 01_FNO_Li2021.pdf
│   ├── 02_DeepONet_Lu2021.pdf
│   ├── ...
│
├── notes/                 ← Your annotations
│   ├── FNO_notes.md
│   ├── DeepONet_notes.md
│   ├── ...
│
└── implementations/       ← Code from papers
    ├── fno_pytorch.py
    ├── deeponet_keras.py
    ├── ...
```

## Implementing Papers Yourself

### Suggested Order (Start Simple)

1. **DeepONet** - Just Dense layers, easy to implement in Keras ✓
2. **GIN** - Replace GCN aggregation, straightforward ✓
3. **SAGPool** - Add to existing GCN, modular ✓
4. **PINN** - Add loss term, no new architecture
5. **FNO** - Most complex, skip if time-limited

### Code Template Locations

- **PyTorch Geometric** (GNN frameworks): https://pytorch-geometric.readthedocs.io/
- **DeepONet implementations**: Various authors on GitHub
- **FNO code**: https://github.com/zongyi-li/fourier_neural_operator
- **SAGPool**: PyTorch Geometric has it built-in

## Staying Updated

### Key Conferences (with OpenReview repos)

- **NeurIPS**: https://openreview.net/group?id=NeurIPS
- **ICML**: https://openreview.net/group?id=ICML
- **ICLR**: https://openreview.net/group?id=ICLR
- **CVPR**: https://openreview.net/group?id=CVPR

### Alerts

- **arXiv daily digest**: https://arxiv.org/help/subscribe
- **Papers with Code**: https://paperswithcode.com/
- **Twitter/X research accounts**: Follow #machinelearning #NeuralNetworks

## Troubleshooting Downloads

### If `curl` fails:

```powershell
# Check internet connection
Test-NetConnection -ComputerName arxiv.org -Port 443

# Try using browser directly: https://arxiv.org/pdf/2010.08895.pdf
```

### If PDFs are corrupted:

```python
import urllib.request
import ssl

# Create SSL context (if certificate issues)
ssl_context = ssl._create_unverified_context()
urllib.request.urlretrieve(url, filename, context=ssl_context)
```

### If rate-limited by arXiv:

- Wait 30 seconds between requests
- Use a VPN if needed
- All papers are free & legal to download

## Quick Checklist

- [X] Downloaded all 5 core papers (PDF folder)
- [ ] Read summaries in markdown (01-05 files)
- [ ] Completed STUDY_GUIDE.md reading order
- [ ] Chose your implementation architecture
- [ ] Set up note-taking system
- [ ] Joined relevant research communities
- [ ] Bookmarked PyTorch Geometric documentation

---

**You're ready to deep dive!** Pick the paper that's most relevant to your next implementation task and start reading the corresponding markdown summary.
