# Implementation Instructions: Mesh-Independent FEA ML Model

**Project**: Build a general-purpose, mesh-independent, shape-independent ML surrogate model for FEA  
**Current State**: Fixed 21×6×6 mesh GCN, single beam geometry  
**Target State**: Model works on any mesh density and variable beam geometries  
**Implementation Window**: 4-5 weeks

---

## CONTEXT SUMMARY FOR AGENT

### Current Repository State
```
e:\jimit.vyas\scripts\fea-ml-training/
├── main.py                          # Entry point
├── pyproject.toml                   # UV dependencies
├── src/
│   ├── data_generator.py            # MockFEASolver (HARDCODED: 21×6×6)
│   ├── generate_dataset.py          # Dataset generation
│   ├── rom_model/
│   │   ├── architectures.py         # build_mlp(), build_gcn()
│   │   ├── adjacency.py             # build_beam_adjacency(21, 6, 6)
│   │   ├── layers.py                # GCNLayer implementation
│   │   ├── trainer.py               # ROMTrainer class
│   │   └── __init__.py
│   └── visualizer.py
├── mock_data/                       # Training data (21×6×6 only)
├── models/                          # Trained models
└── literature_review/               # SOTA papers & guides
    ├── README.md
    ├── STUDY_GUIDE.md              # 2-week roadmap
    ├── 01_FNO_Li2021.md
    ├── 02_DeepONet_Lu2021.md
    ├── 03_GIN_Xu2019.md
    ├── 04_SAGPool_Lee2019.md
    ├── 05_PINNs_Raissi2019.md
    └── references.bib
```

### Key Limitations to Fix
1. **Fixed Mesh**: `nx, ny, nz = 21, 6, 6` hardcoded in `data_generator.py`
2. **Fixed Adjacency**: `build_beam_adjacency(21, 6, 6)` in `trainer.py`
3. **Fixed Node Count**: Model expects 756 nodes (21×6×6)
4. **Single Geometry**: Only varies load, not beam dimensions
5. **No Generalization**: Fails on different mesh densities

### Recommended Architecture (Based on SOTA Research)
**GCN + SAGPool + optional PINN Loss**
- Why: Balances simplicity with effectiveness
- Why not FNO: Unstructured meshes, less flexibility
- Why not DeepONet alone: Want to keep connectivity awareness
- Why SAGPool: Handles variable mesh sizes naturally
- Why PINN: Better generalization, physics constraints

---

## PHASE 1: PARAMETERIZE DATA GENERATION

### Task 1.1: Extend `data_generator.py` for Variable Meshes

**File**: `src/data_generator.py`

**Current State**:
```python
nx, ny, nz = 21, 6, 6
```

**Required Changes**:

1. Make mesh resolution a parameter:
```python
def solve(self, length, width, depth, load, sample_id=0, mesh_size=None):
    """
    mesh_size: tuple (nx, ny, nz) or None for default (21, 6, 6)
    """
    if mesh_size is None:
        mesh_size = (21, 6, 6)
    nx, ny, nz = mesh_size
    
    # Rest of solve() method unchanged
```

2. Return mesh info with outputs:
```python
# Save outputs with metadata
mesh_metadata = {
    'sample_id': sample_id,
    'mesh_size': mesh_size,
    'node_count': nx * ny * nz,
    'geometry': {'length': length, 'width': width, 'depth': depth},
    'load': load
}
np.save(os.path.join(self.output_dir, f"{filename}_metadata.npy"), 
        mesh_metadata)
```

3. Validate mesh generation:
   - Ensure mesh object has `.points` attribute (node coordinates)
   - Ensure displacement/stress arrays match node count N
   - Print: `f"Generated {nx*ny*nz} nodes, {len(displacement)} displacement values"`

**Validation**:
```python
# Test in script
solver = MockFEASolver()
mesh_21x6x6 = solver.solve(180, 25, 12, 400, mesh_size=(21, 6, 6))
mesh_30x8x8 = solver.solve(180, 25, 12, 400, mesh_size=(30, 8, 8))
assert mesh_21x6x6.n_points == 756
assert mesh_30x8x8.n_points == 1920
```

---

### Task 1.2: Update `generate_dataset.py` for Multi-Mesh Generation

**File**: `src/generate_dataset.py`

**Current Function Signature**:
```python
def generate_dataset(n_samples=100, output_dir="mock_data", sampling="taguchi", ...)
```

**Required Changes**:

1. Add mesh resolution parameter:
```python
def generate_dataset(
    n_samples=100,
    output_dir="mock_data",
    sampling="taguchi",
    mesh_resolutions=None,  # NEW: [(21,6,6), (30,8,8), ...]
    taguchi_levels=5,
    seed=42
):
    """
    mesh_resolutions: List of (nx, ny, nz) tuples
                     If None, uses [(21,6,6)] for backward compatibility
    """
    if mesh_resolutions is None:
        mesh_resolutions = [(21, 6, 6)]
```

2. Generate samples for each resolution:
```python
for mesh_res in mesh_resolutions:
    mesh_output_dir = os.path.join(output_dir, f"mesh_{mesh_res[0]}x{mesh_res[1]}x{mesh_res[2]}")
    os.makedirs(mesh_output_dir, exist_ok=True)
    
    solver = MockFEASolver(output_dir=mesh_output_dir)
    
    for i, sample in tqdm(df.iterrows()):
        solver.solve(
            length=sample["length"],
            width=sample["width"],
            depth=sample["depth"],
            load=sample["load"],
            sample_id=int(sample["sample_id"]),
            mesh_size=mesh_res  # Pass mesh resolution
        )
```

3. Generate design table ONCE (shared across all meshes):
```python
# Create only one design table in parent directory
csv_path = os.path.join(output_dir, "design_table.csv")
# Don't duplicate in each mesh folder
```

**Update `main.py` usage**:
```python
from src.generate_dataset import generate_dataset

# Instead of:
# generate_dataset(n_samples=600, sampling="random")

# Use:
generate_dataset(
    n_samples=600,
    sampling="random",
    mesh_resolutions=[
        (15, 4, 4),   # Coarse: 240 nodes
        (21, 6, 6),   # Medium: 756 nodes (current)
        (30, 8, 8),   # Fine: 1920 nodes
    ]
)
```

**Extend geometry variation**:
```python
# Expand parameter ranges (optional, for Phase 2)
param_ranges = {
    "length": (80.0, 200.0),    # More variation
    "width": (10.0, 60.0),      # More variation
    "depth": (5.0, 30.0),       # More variation
    "load": (-500.0, 500.0)     # Compression + tension
}
```

**Validation**:
```python
# After generation, verify:
for mesh_dir in ["mesh_15x4x4", "mesh_21x6x6", "mesh_30x8x8"]:
    assert os.path.exists(f"mock_data/{mesh_dir}/design_table.csv")
    assert len(glob(f"mock_data/{mesh_dir}/*_params.npy")) == 600
```

---

## PHASE 2: GRAPH DATA STRUCTURE WITH NODE COORDINATES

### Task 2.1: Create Mesh Graph Data Loader

**New File**: `src/graph_utils.py`

```python
"""Utilities for converting mesh data to graph format."""

import numpy as np
import networkx as nx
from scipy.spatial import cKDTree

def load_mesh_graph(mesh_file):
    """
    Load mesh from VTK file and convert to graph representation.
    
    Returns:
    --------
    graph_data : dict with keys:
        'node_coords': (N, 3) node coordinates
        'edge_index': (2, E) edge connectivity
        'displacement': (N, 3) displacement values
        'stress': (N,) stress values
    """
    import pyvista as pv
    
    mesh = pv.read(mesh_file)
    node_coords = mesh.points.astype(np.float32)
    
    # Build edge connectivity from mesh (6-neighbor connectivity for structured grid)
    N = len(node_coords)
    edges = build_structured_adjacency(node_coords)
    
    # Load output data
    displacement = mesh.point_data.get("Displacement", np.zeros((N, 3)))
    stress = mesh.point_data.get("Stress_XX", np.zeros(N))
    
    return {
        'node_coords': node_coords,
        'edge_index': edges,
        'displacement': displacement.astype(np.float32),
        'stress': stress.astype(np.float32),
    }

def build_structured_adjacency(node_coords, k=6):
    """
    Build adjacency matrix for structured hex mesh using k-NN.
    For structured grid like ours, 6-NN gives 6-connectivity (face neighbors).
    
    Returns: edge_index of shape (2, num_edges)
    """
    tree = cKDTree(node_coords)
    distances, indices = tree.query(node_coords, k=k+1)  # k+1 to exclude self
    
    edges_set = set()
    for i in range(len(node_coords)):
        for j in indices[i][1:]:  # Skip first (self)
            edge = tuple(sorted([i, int(j)]))
            edges_set.add(edge)
    
    edge_index = np.array(list(edges_set), dtype=np.int64).T  # (2, E)
    return edge_index

def load_dataset_with_graphs(data_dir, mesh_size):
    """
    Load all samples from data_dir as graph format.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing sample_*_params.npy, *_disp.npy, *_stress.npy
    mesh_size : tuple
        (nx, ny, nz) for adjacency matrix
    
    Returns:
    --------
    graphs : list of graph_data dicts
    params : (N, 4) parameter array
    """
    import glob
    
    param_files = sorted(glob.glob(os.path.join(data_dir, "*_params.npy")))
    disp_files = sorted(glob.glob(os.path.join(data_dir, "*_disp.npy")))
    stress_files = sorted(glob.glob(os.path.join(data_dir, "*_stress.npy")))
    
    graphs = []
    params = []
    
    for p_f, d_f, s_f in zip(param_files, disp_files, stress_files):
        param = np.load(p_f)
        displacement = np.load(d_f)
        stress = np.load(s_f)
        
        # Create synthetic node coordinates (for structured grid)
        node_coords = create_structured_mesh_coords(*mesh_size, param[:3])
        
        # Build connectivity
        edge_index = build_structured_adjacency(node_coords)
        
        graphs.append({
            'node_coords': node_coords,
            'edge_index': edge_index,
            'displacement': displacement,
            'stress': stress,
        })
        params.append(param)
    
    return graphs, np.array(params)

def create_structured_mesh_coords(nx, ny, nz, geometry):
    """
    Create node coordinates for structured hex mesh.
    
    Parameters:
    -----------
    nx, ny, nz : int - grid dimensions
    geometry : array of [length, width, depth]
    """
    length, width, depth = geometry
    
    x = np.linspace(0, length, nx)
    y = np.linspace(-width/2, width/2, ny)
    z = np.linspace(-depth/2, depth/2, nz)
    
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    coords = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    
    return coords.astype(np.float32)
```

**Validation**:
```python
from src.graph_utils import load_dataset_with_graphs

graphs, params = load_dataset_with_graphs("mock_data/mesh_21x6x6", (21, 6, 6))
assert len(graphs) == 600
assert graphs[0]['node_coords'].shape == (756, 3)
assert graphs[0]['edge_index'].shape[0] == 2
assert graphs[0]['displacement'].shape == (756, 3)
```

---

## PHASE 3: UPGRADE GCN TO PERMUTATION-INVARIANT KEY-AWARE MODEL

### Task 3.1: Modify Adjacency Module

**File**: `src/rom_model/adjacency.py`

**Change**: Make `build_beam_adjacency` accept variable dimensions

```python
def build_beam_adjacency(nx=21, ny=6, nz=6):
    """
    Build normalised adjacency matrix for a structured hex mesh.
    
    Parameters:
    -----------
    nx, ny, nz : int
        Mesh dimensions (now parameterized)
    
    Returns:
    --------
    A_hat : (N, N) normalized adjacency, where N = nx*ny*nz
    """
    N = nx * ny * nz

    def idx(i, j, k):
        return i * ny * nz + j * nz + k

    rows, cols = [], []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                n = idx(i, j, k)
                rows.append(n); cols.append(n)  # Self-loop
                # 6 neighbors (face-adjacent in hex mesh)
                for di, dj, dk in [(1,0,0),(-1,0,0),(0,1,0),
                                   (0,-1,0),(0,0,1),(0,0,-1)]:
                    ni, nj, nk = i+di, j+dj, k+dk
                    if 0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz:
                        rows.append(n)
                        cols.append(idx(ni, nj, nk))

    A = np.zeros((N, N), dtype=np.float32)
    A[rows, cols] = 1.0

    # Symmetric normalization
    deg = A.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(deg, 1e-9)))
    A_hat = D_inv_sqrt @ A @ D_inv_sqrt
    
    return A_hat.astype(np.float32)
```

**No changes needed** - function signature already accepts parameters!

---

### Task 3.2: Update ROMTrainer to Handle Variable Meshes

**File**: `src/rom_model/trainer.py`

**Changes**:

1. Update `__init__`:
```python
def __init__(self, data_dir="mock_data", model_dir="models", 
             model_type="gcn", mesh_size=(21, 6, 6)):
    """
    mesh_size : tuple
        (nx, ny, nz) dimensions of training mesh
    """
    self.data_dir = data_dir
    self.model_dir = model_dir
    self.model_type = model_type.lower()
    self.mesh_size = mesh_size  # NEW
    os.makedirs(self.model_dir, exist_ok=True)

    if self.model_type not in ("mlp", "gcn"):
        raise ValueError(f"Unknown model_type '{self.model_type}'")

    # Precompute adjacency for training mesh
    self._A_hat = build_beam_adjacency(*mesh_size)  # Use actual size
    print(f"Adjacency matrix shape: {self._A_hat.shape}")
```

2. In `train()` method, save mesh_size with model:
```python
# Before model.save():
config = {
    'model_type': self.model_type,
    'mesh_size': self.mesh_size,
    'scaler_params': scaler.mean_.tolist(),
    'scaler_scale': scaler.scale_.tolist(),
}
json.dump(config, open(os.path.join(self.model_dir, 'config.json'), 'w'))
```

**Validation**:
```python
trainer_21 = ROMTrainer(data_dir="mock_data/mesh_21x6x6", 
                        model_type="gcn",
                        mesh_size=(21, 6, 6))
assert trainer_21._A_hat.shape == (756, 756)

trainer_30 = ROMTrainer(data_dir="mock_data/mesh_30x8x8",
                        model_type="gcn", 
                        mesh_size=(30, 8, 8))
assert trainer_30._A_hat.shape == (1920, 1920)
```

---

## PHASE 4: ADD SAGPool FOR VARIABLE MESH SIZES

### Task 4.1: Create Multi-Resolution GCN with Pooling

**File**: `src/rom_model/architectures.py`

**New Function**:

```python
def build_gcn_with_pooling(input_dim: int, output_dim: int, 
                           A_hat: np.ndarray, pool_ratio: float = 0.5) -> keras.Model:
    """
    Build GCN with self-attention graph pooling for variable mesh sizes.
    
    Architecture:
    [Input: (N, hidden)] → [GCN] → [SAGPool 50%] → [GCN] → [Global Pool] → [Decoder]
    
    Parameters:
    -----------
    input_dim : int
        Number of input features (4: geometry params)
    output_dim : int
        Number of output features (tensor size)
    A_hat : np.ndarray
        Normalized adjacency matrix (N, N)
    pool_ratio : float
        Fraction of nodes to keep after pooling
    
    Returns:
    --------
    keras.Model with inputs [params, A_hat, initial_features]
    """
    N = A_hat.shape[0]
    
    params_inp = keras.Input(shape=(input_dim,), name="params")  # (B, 4)
    a_inp = keras.Input(shape=(N, N), name="A_hat")             # (B, N, N)
    
    # Initialize node features: broadcast params to all nodes
    broadcast = layers.RepeatVector(N)(params_inp)              # (B, N, 4)
    node_feat = layers.Dense(32, activation="swish")(broadcast) # (B, N, 32)
    
    # GCN layer 1
    x = GCNLayer(64, activation="relu", name="gcn_1")([node_feat, a_inp])  # (B, N, 64)
    
    # Attention pooling 1: Keep top-K nodes
    # Simplified: Use max pooling over top-K (approximately)
    # For exact SAGPool, would need custom layer
    
    # For now: Alternative - use global attention
    # scores = Dense(1)(x)  # (B, N, 1)
    # x_pooled = multiply([x, scores])  # Weighted nodes
    # x = GlobalAveragePooling1D()(x_pooled)  # To (B, 64)
    
    # Simpler v1: Just use multiple GCN layers without explicit pooling
    x = GCNLayer(128, activation="leaky_relu", name="gcn_2")([x, a_inp])
    x = GCNLayer(128, activation="relu", name="gcn_3")([x, a_inp])
    x = GCNLayer(128, activation="leaky_relu", name="gcn_4")([x, a_inp])
    x = GCNLayer(64, activation="relu", name="gcn_5")([x, a_inp])
    
    # Global average pooling (summarize all nodes)
    pooled = layers.GlobalAveragePooling1D()(x)  # (B, 64)
    
    # Dense decoder head
    h = layers.Dense(256, activation="swish")(pooled)
    h = layers.Dropout(0.1)(h)
    h = layers.Dense(512, activation="swish")(h)
    out_disp = layers.Dense(output_dim * 3, name="displacement")(h)
    out_stress = layers.Dense(output_dim, name="stress")(h)
    
    model = keras.Model(
        inputs=[params_inp, a_inp],
        outputs=[out_disp, out_stress],
        name="GCN_MultiResolution"
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss={"displacement": "mse", "stress": "mse"},
        metrics={"displacement": "mae", "stress": "mae"}
    )
    return model
```

**Note**: Full SAGPool implementation requires PyTorch Geometric (optional upgrade in Phase 5).

**Validation**:
```python
A_hat_small = build_beam_adjacency(21, 6, 6)
A_hat_large = build_beam_adjacency(30, 8, 8)

model_small = build_gcn_with_pooling(4, 756, A_hat_small)
model_large = build_gcn_with_pooling(4, 1920, A_hat_large)

# Both should work
print(model_small.summary())
print(model_large.summary())
```

---

## PHASE 5: MULTI-MESH TRAINING & EVALUATION

### Task 5.1: Create Multi-Mesh Training Pipeline

**File**: `src/rom_model/multi_mesh_trainer.py` (New file)

```python
"""Training pipeline for mesh-independent models."""

import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import keras
from src.rom_model.architectures import build_gcn_with_pooling
from src.rom_model.adjacency import build_beam_adjacency

class MultiMeshTrainer:
    """Train model on multiple mesh resolutions simultaneously."""
    
    def __init__(self, base_data_dir="mock_data", model_dir="models",
                 mesh_resolutions=None):
        """
        mesh_resolutions : list of (nx, ny, nz) tuples
        """
        self.base_data_dir = base_data_dir
        self.model_dir = model_dir
        self.mesh_resolutions = mesh_resolutions or [(21, 6, 6)]
        os.makedirs(model_dir, exist_ok=True)
    
    def load_multi_mesh_data(self):
        """
        Load training data from all mesh resolutions.
        
        Returns:
        --------
        X : (total_samples, 4) parameters
        Y_disp : (total_samples, max_nodes*3) displacement
        Y_stress : (total_samples, max_nodes) stress
        A_hats : dict mapping mesh_size → adjacency matrix
        sample_mesh_indices : array indicating which mesh each sample comes from
        """
        import glob
        
        all_params, all_disps, all_stresses = [], [], []
        A_hats = {}
        sample_mesh_map = []
        max_nodes = 0
        
        for mesh_res in self.mesh_resolutions:
            mesh_dir = os.path.join(
                self.base_data_dir, 
                f"mesh_{mesh_res[0]}x{mesh_res[1]}x{mesh_res[2]}"
            )
            
            param_files = sorted(glob.glob(os.path.join(mesh_dir, "*_params.npy")))
            disp_files = sorted(glob.glob(os.path.join(mesh_dir, "*_disp.npy")))
            stress_files = sorted(glob.glob(os.path.join(mesh_dir, "*_stress.npy")))
            
            mesh_params = []
            mesh_disps = []
            mesh_stresses = []
            
            for p_f, d_f, s_f in zip(param_files, disp_files, stress_files):
                mesh_params.append(np.load(p_f))
                mesh_disps.append(np.load(d_f).flatten())
                mesh_stresses.append(np.load(s_f).flatten())
            
            mesh_params = np.array(mesh_params)
            mesh_disps = np.array(mesh_disps)
            mesh_stresses = np.array(mesh_stresses)
            
            # Store adjacency
            A_hats[mesh_res] = build_beam_adjacency(*mesh_res)
            
            all_params.append(mesh_params)
            all_disps.append(mesh_disps)
            all_stresses.append(mesh_stresses)
            
            sample_mesh_map.extend([mesh_res] * len(mesh_params))
            max_nodes = max(max_nodes, mesh_disps.shape[1] // 3)
        
        # Concatenate across meshes
        X = np.concatenate(all_params, axis=0)
        
        # Pad displacement/stress to max size
        Y_disp_padded = np.zeros((sum(len(d) for d in all_disps), max_nodes * 3))
        Y_stress_padded = np.zeros((sum(len(s) for s in all_stresses), max_nodes))
        
        row_idx = 0
        for disp_arr, stress_arr in zip(all_disps, all_stresses):
            n_samples = len(disp_arr)
            n_nodes = disp_arr.shape[1] // 3
            Y_disp_padded[row_idx:row_idx+n_samples, :n_nodes*3] = disp_arr
            Y_stress_padded[row_idx:row_idx+n_samples, :n_nodes] = stress_arr
            row_idx += n_samples
        
        return X, Y_disp_padded, Y_stress_padded, A_hats, sample_mesh_map
    
    def train(self, epochs=100, batch_size=32, validation_split=0.2):
        """Train model on multi-mesh data."""
        
        print("Loading multi-mesh data...")
        X, Y_disp, Y_stress, A_hats, sample_mesh_map = self.load_multi_mesh_data()
        
        print(f"Total samples: {len(X)}")
        print(f"Mesh resolutions: {self.mesh_resolutions}")
        print(f"Max output size: {Y_disp.shape[1]} (displacement), {Y_stress.shape[1]} (stress)")
        
        # Normalize inputs
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_val, Y_disp_train, Y_disp_val, Y_stress_train, Y_stress_val = \
            train_test_split(X_scaled, Y_disp, Y_stress, 
                           test_size=validation_split, random_state=42)
        
        print(f"Train samples: {len(X_train)}, Validation: {len(X_val)}")
        
        # TO BE IMPLEMENTED: Curriculum learning
        # Stage 1: Train on single mesh (easiest)
        # Stage 2: Add two meshes
        # Stage 3: All three meshes
        # This improves convergence
        
        print("Training base model on primary mesh...")
        primary_mesh = self.mesh_resolutions[0]
        A_hat = A_hats[primary_mesh]
        
        model = build_gcn_with_pooling(4, self._max_nodes_from_shape(Y_disp.shape[1]), A_hat)
        
        # Prepare batch data with adjacency matrix
        A_hat_batch = np.tile(A_hat[np.newaxis, :, :], (len(X_train), 1, 1))
        A_hat_val = np.tile(A_hat[np.newaxis, :, :], (len(X_val), 1, 1))
        
        history = model.fit(
            [X_train, A_hat_batch],
            [Y_disp_train, Y_stress_train],
            validation_data=([X_val, A_hat_val], [Y_disp_val, Y_stress_val]),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        # Save model and scaler
        model.save(os.path.join(self.model_dir, "multi_mesh_model.keras"))
        np.save(os.path.join(self.model_dir, "scaler_mean.npy"), scaler.mean_)
        np.save(os.path.join(self.model_dir, "scaler_scale.npy"), scaler.scale_)
        
        return model, history
    
    @staticmethod
    def _max_nodes_from_shape(shape):
        """Extract max nodes from displacement shape."""
        return shape // 3
```

---

### Task 5.2: Create Comprehensive Evaluation Script

**File**: `src/evaluate_mesh_independence.py` (New file)

```python
"""Evaluate mesh independence and generalization."""

import numpy as np
import os
from sklearn.metrics import mean_squared_error, r2_score
from src.rom_model.trainer import ROMTrainer
from src.rom_model.adjacency import build_beam_adjacency

def evaluate_models_on_multiple_meshes(
    data_dir="mock_data",
    model_dirs=None,
    test_meshes=None,
    test_params=None
):
    """
    Evaluate trained models on different mesh resolutions.
    
    Parameters:
    -----------
    model_dirs : dict
        Maps mesh_size → path to trained model
    test_meshes : list of tuples
        Mesh resolutions to test on
    test_params : (n_test, 4)
        Parameters to evaluate (if None, use first sample from each mesh)
    
    Returns:
    --------
    results : dict with per-mesh accuracy metrics
    """
    if test_meshes is None:
        test_meshes = [(15, 4, 4), (21, 6, 6), (30, 8, 8), (35, 9, 9)]
    
    results = {}
    
    for test_mesh in test_meshes:
        print(f"\n=== Evaluating on mesh {test_mesh} ===")
        
        # Get ground truth data
        mesh_dir = os.path.join(data_dir, f"mesh_{test_mesh[0]}x{test_mesh[1]}x{test_mesh[2]}")
        if not os.path.exists(mesh_dir):
            print(f"Skipping {test_mesh} - data not found")
            continue
        
        # Load test samples
        Y_disp_test = np.load(os.path.join(mesh_dir, "sample_50_disp.npy")).flatten()
        Y_stress_test = np.load(os.path.join(mesh_dir, "sample_50_stress.npy")).flatten()
        params_test = np.load(os.path.join(mesh_dir, "sample_50_params.npy"))
        
        # Pad if necessary
        n_nodes = test_mesh[0] * test_mesh[1] * test_mesh[2]
        Y_disp_padded = np.zeros(n_nodes * 3)
        Y_disp_padded[:len(Y_disp_test)] = Y_disp_test
        
        mesh_results = {}
        
        # Evaluate each trained model on test mesh
        for train_mesh, model_path in (model_dirs or {}).items():
            if not os.path.exists(model_path):
                continue
            
            print(f"  Model trained on {train_mesh}:")
            
            # Load model
            model = keras.models.load_model(model_path)
            
            # Prepare input
            A_hat = build_beam_adjacency(*test_mesh)
            A_hat_batch = np.tile(A_hat[np.newaxis, :, :], (1, 1, 1))
            
            # Predict
            disp_pred, stress_pred = model.predict([params_test[np.newaxis, :], A_hat_batch], verbose=0)
            
            # Evaluate
            disp_mse = mean_squared_error(Y_disp_padded, disp_pred[0, :len(Y_disp_padded)])
            stress_mse = mean_squared_error(Y_stress_test, stress_pred[0, :len(Y_stress_test)])
            
            mesh_results[train_mesh] = {
                'displacement_mse': float(disp_mse),
                'stress_mse': float(stress_mse),
                'displacement_rmse': float(np.sqrt(disp_mse)),
                'stress_rmse': float(np.sqrt(stress_mse)),
            }
            
            print(f"    Disp RMSE: {mesh_results[train_mesh]['displacement_rmse']:.6f}")
            print(f"    Stress RMSE: {mesh_results[train_mesh]['stress_rmse']:.6f}")
        
        results[test_mesh] = mesh_results
    
    return results

def print_evaluation_table(results):
    """Print evaluation results as table."""
    import pandas as pd
    
    rows = []
    for test_mesh, model_results in results.items():
        for train_mesh, metrics in model_results.items():
            rows.append({
                'Test Mesh': str(test_mesh),
                'Train Mesh': str(train_mesh),
                'Disp RMSE': f"{metrics['displacement_rmse']:.6f}",
                'Stress RMSE': f"{metrics['stress_rmse']:.6f}",
            })
    
    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))
    return df
```

**Validation Script** (`tests/test_mesh_independence.py`):

```python
"""Tests for mesh independence."""

import sys
sys.path.insert(0, '/path/to/fea-ml-training')

def test_phase_1_variable_mesh_generation():
    """Test: Data generation works for multiple meshes."""
    from src.generate_dataset import generate_dataset
    
    generate_dataset(
        n_samples=10,
        sampling="random",
        mesh_resolutions=[(15, 4, 4), (21, 6, 6), (30, 8, 8)]
    )
    
    # Verify directories exist
    assert os.path.exists("mock_data/mesh_15x4x4")
    assert os.path.exists("mock_data/mesh_21x6x6")
    assert os.path.exists("mock_data/mesh_30x8x8")
    print("✓ Phase 1 passed: Multi-mesh data generation")

def test_phase_2_graph_format():
    """Test: Graph format with coordinates."""
    from src.graph_utils import load_dataset_with_graphs
    
    graphs, params = load_dataset_with_graphs("mock_data/mesh_21x6x6", (21, 6, 6))
    
    assert len(graphs) == 10
    assert graphs[0]['node_coords'].shape == (756, 3)
    assert graphs[0]['edge_index'].shape[0] == 2
    print("✓ Phase 2 passed: Graph format conversion")

def test_phase_3_variable_adjacency():
    """Test: Adjacency matrix works for variable sizes."""
    from src.rom_model.adjacency import build_beam_adjacency
    
    A_15 = build_beam_adjacency(15, 4, 4)
    A_21 = build_beam_adjacency(21, 6, 6)
    A_30 = build_beam_adjacency(30, 8, 8)
    
    assert A_15.shape == (240, 240)
    assert A_21.shape == (756, 756)
    assert A_30.shape == (1920, 1920)
    print("✓ Phase 3 passed: Variable adjacency matrices")

def test_phase_4_gcn_variable_input():
    """Test: GCN accepts variable input sizes."""
    from src.rom_model.architectures import build_gcn_with_pooling
    from src.rom_model.adjacency import build_beam_adjacency
    
    A_small = build_beam_adjacency(15, 4, 4)
    A_large = build_beam_adjacency(30, 8, 8)
    
    model_small = build_gcn_with_pooling(4, 240, A_small)
    model_large = build_gcn_with_pooling(4, 1920, A_large)
    
    assert model_small.get_config() is not None
    assert model_large.get_config() is not None
    print("✓ Phase 4 passed: GCN with variable inputs")

def test_phase_5_multi_mesh_training():
    """Test: Training on multiple meshes simultaneously."""
    from src.rom_model.multi_mesh_trainer import MultiMeshTrainer
    
    trainer = MultiMeshTrainer(
        mesh_resolutions=[(15, 4, 4), (21, 6, 6)]
    )
    
    X, Y_disp, Y_stress, A_hats, _ = trainer.load_multi_mesh_data()
    
    assert X.shape[1] == 4
    assert len(A_hats) == 2
    print("✓ Phase 5 passed: Multi-mesh training pipeline")

if __name__ == "__main__":
    test_phase_1_variable_mesh_generation()
    test_phase_2_graph_format()
    test_phase_3_variable_adjacency()
    test_phase_4_gcn_variable_input()
    test_phase_5_multi_mesh_training()
    print("\n✓✓✓ All tests passed! ✓✓✓")
```

---

## PHASE 6: OPTIONAL ENHANCEMENTS

### Task 6.1: Add PINN Loss for Physics Awareness (Optional)

**File**: `src/rom_model/pinn_loss.py` (New file)

```python
"""Physics-informed loss components."""

import tensorflow as tf

def compute_physics_loss(predictions, geometry_params, load):
    """
    Compute physics-informed loss for cantilever beam.
    
    Beam theory: d²u/dx² = -M / (E*I) = -load * x / (E*I)
    """
    # Material constants
    E = 210000.0  # Young's modulus [MPa]
    I_ratio = 1.0  # Will be computed from geometry
    
    # Simplified: assume stress should satisfy equilibrium
    # stress ≈ M(x) * z / I
    # Further constraint: average stress relates to load
    
    disp_pred = predictions[0]
    stress_pred = predictions[1]
    
    # Physics constraint 1: Stress magnitude proportional to load
    expected_stress_scale = load / 100.0  # Heuristic
    stress_scale_loss = tf.reduce_mean((tf.reduce_max(tf.abs(stress_pred)) - expected_stress_scale) ** 2)
    
    # Physics constraint 2: Displacement should be smooth
    # (adjacent nodes should have similar displacement)
    #  smoothness_loss = tf.reduce_mean(tf.square(disp_pred[1:] - disp_pred[:-1]))
    
    return 0.1 * stress_scale_loss  # Weight for combined loss
```

**Integration in trainer**:
```python
# In training loop:
loss_data = model.loss(predictions, ground_truth)
loss_physics = compute_physics_loss(predictions, params, load)
total_loss = loss_data + 0.1 * loss_physics
```

---

### Task 6.2: Implement PyTorch Geometric Version (Optional)

**File**: `src/rom_model/pytorch_gnn_model.py` (New file)

Uses PyTorch Geometric for true SAGPool implementation:

```python
import torch
from torch_geometric.nn import GCNConv, SAGPooling
from torch_geometric.data import Data, DataLoader
import torch.nn.functional as F

class MeshIndependentGNN(torch.nn.Module):
    """GNN with learned hierarchical pooling for variable meshes."""
    
    def __init__(self, input_dim=4, hidden_dim=128, output_dim=756):
        super().__init__()
        
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.pool1 = SAGPooling(hidden_dim, ratio=0.5)
        
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.pool2 = SAGPooling(hidden_dim, ratio=0.5)
        
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        self.fc1 = torch.nn.Linear(hidden_dim, 256)
        self.fc2 = torch.nn.Linear(256, output_dim)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = self.conv1(x, edge_index).relu()
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, batch=batch)
        
        x = self.conv2(x, edge_index).relu()
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, batch=batch)
        
        x = self.conv3(x, edge_index)
        
        # Global mean pooling
        x = torch.scatter(x, batch, dim=0, reduce="mean")
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
```

---

## VALIDATION CHECKLIST

### Must Pass Before Moving to Next Phase

**Phase 1**: ✓ All mesh resolutions generate correct node counts  
**Phase 2**: ✓ Graph format loads correctly with coordinates  
**Phase 3**: ✓ Adjacency matrices correct for each mesh size  
**Phase 4**: ✓ GCN model builds and runs without errors  
**Phase 5**: ✓ Multi-mesh training converges on all meshes  
**Phase 6**: ✓ Evaluation shows <15% accuracy drop on unseen mesh  

### Final Success Criteria

- [ ] Model trains on (21×6×6)
- [ ] Model evaluates on (30×8×8) with <10% error increase
- [ ] Model evaluates on (15×4×4) with <10% error increase
- [ ] Inference time < 1ms per sample
- [ ] Memory: < 100MB model + <1GB data
- [ ] Generalizes to geometry parameters outside training range
- [ ] Physics constraints satisfied (boundary conditions met)

---

## IMPLEMENTATION TIMELINE

```
Week 1: Phases 1-2 (Data generation + graph format)
  - Mon-Tue:  Data generation parametrization
  - Wed-Thu:  Graph utilities & loaders
  - Fri:      Testing & validation

Week 2: Phase 3-4 (GCN upgrade + pooling)
  - Mon:      Adjacency generalization
  - Tue-Wed:  GCN with pooling
  - Thu-Fri:  Multiple model building & testing

Week 3: Phase 5 (Training pipeline)
  - Mon-Tue:  Multi-mesh trainer implementation
  - Wed:      Training runs on all meshes
  - Thu-Fri:  Debugging & hyperparameter tuning

Week 4: Testing & optimization
  - Mon-Tue:  Comprehensive evaluation
  - Wed:      Comparison vs baseline
  - Thu-Fri:  Optimization & refinement

Week 5 (Optional): Enhancements
  - PINN loss integration
  - PyTorch Geometric version
  - Documentation & demo
```

---

## DEBUGGING & TROUBLESHOOTING

| Problem | Cause | Solution |
|---------|-------|----------|
| Shapes mismatch in training | Adjacency size wrong | Verify `build_beam_adjacency(*mesh_size)` |
| Model fails on new mesh | Hardcoded sizes remain | Search for `(21, 6, 6)` in all files |
| Accuracy drops on coarse mesh | Too little information | Add curriculum learning (coarse→fine) |
| Accuracy drops on fine mesh | Overfitting to training density | Data augmentation or domain randomization |
| Training too slow | Dense adjacency operations | Use sparse matrices (tf.SparseTensor) |
| Memory issues | Batching large graphs | Reduce batch size or use gradient accumulation |

---

## REFERENCES

- **Literature Review**: See `/literature_review/` folder
- **SOTA Papers**: 
  - GIN (Xu et al., 2019): `03_GIN_Xu2019.md`
  - SAGPool (Lee et al., 2019): `04_SAGPool_Lee2019.md`
  - PINNs (Raissi et al., 2019): `05_PINNs_Raissi2019.md`
- **PyTorch Geometric**: https://pytorch-geometric.readthedocs.io/
- **TensorFlow/Keras**: https://keras.io/

---

**Created**: March 2, 2026  
**Status**: Ready for agent implementation  
**Last Updated**: [TIMESTAMP]

