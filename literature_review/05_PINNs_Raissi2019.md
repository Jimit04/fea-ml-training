# Physics-Informed Neural Networks (PINNs) - Raissi et al., 2019

**arXiv:** https://arxiv.org/abs/1711.10561
**DOI:** 10.1016/j.jcp.2018.10.045
**Status:** Published in Journal of Computational Physics, 2019

## Summary

PINNs encode physics constraints (PDEs, boundary conditions, conservation laws) directly into the neural network training loss function. Instead of purely data-driven learning, the network learns solutions that satisfy both data AND physics.

## Key Innovation: Physics as Regularization

```
Standard supervised learning:
  Loss = || y_pred - y_true ||²
  → Learns to fit data, ignores physics

Physics-informed learning:
  Loss = α·|| y_pred - y_true ||² + β·|| PDE_residual ||²
  → Learns to fit data AND satisfy equations
```

The PDE constraint is automatically differentiated via backprop!

## How It Works for Your FEA Problem

### Current Approach

```python
loss = mse(predictions, ground_truth)
# Network learns patterns from data
# Doesn't guarantee physics satisfaction
```

### PINN Approach

```python
# Data loss (supervised)
loss_data = mse(predictions, ground_truth)

# Physics loss (unsupervised, from laws)
# For Euler-Bernoulli beam:
#   d²u/dx² + load/EI = 0  (governing PDE)
#   u(0) = 0, du/dx(0) = 0  (BCs)

# Compute derivatives via autodiff
u_xx = grad(grad(u, x), x)
physics_residual = u_xx + load / (E*I)

loss_physics = mse(physics_residual, 0)

# Combined loss
total_loss = loss_data + λ * loss_physics
```

## Architecture for FEA Problem

```python
class PhysicsInformedNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Standard MLP, but with monitoring of derivatives
        self.net = nn.Sequential(
            nn.Linear(4, 256),  # (length, width, depth, load)
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3)   # (u_x, u_y, u_z)
        )
  
    def forward(self, params):
        return self.net(params)
  
    def compute_physics_loss(self, params, load, E, I):
        # Enable gradient tracking for second derivatives
        params_var = params.clone().detach().requires_grad_(True)
      
        # Forward pass
        u = self.forward(params_var)  # Displacement
      
        # First derivative
        u_x = torch.autograd.grad(
            u.sum(), params_var, create_graph=True
        )[0][:, 0]  # du/dL (length sensitivity)
      
        # Second derivative
        u_xx = torch.autograd.grad(
            u_x.sum(), params_var
        )[0][:, 0]
      
        # Physics: d²u/dx² + load/(E*I) = 0
        # (simplified cantilever beam equation)
        physics_residual = u_xx + load / (E * I)
      
        # Physics loss
        loss_physics = (physics_residual ** 2).mean()
        return loss_physics
```

## Benefits for Your Project

### 1. Reduced Data Requirements

```
Purely data-driven:
  Need 1000 samples of diverse beams to learn well

Physics-informed:
  Need only 100 samples
  + Physics constraints compensate for sparse data
```

### 2. Better Generalization

```
Data domain:     Load 0-500N, Length 100-200mm
Test domain:     Load 600N (extrapolated)

Data-only:       Fails (out of distribution)
Physics-PINN:    Works (physics constrains behavior)
```

### 3. Constraint Satisfaction

```
Standard NN:     May violate boundary conditions
                 Displacement might not be 0 at fixed end
                 Stress might violate equilibrium

PINN:            Automatically satisfies:
                 - Fixed boundary (u=0 at x=0)
                 - Free boundary (σ=0 at x=L)
                 - Equilibrium everywhere
```

### 4. Physics Discovery

```
You can infer unknown physics:
- Train PINN with unknown material property E
- Let network discover E from data
- Compare with actual material properties

loss = || predictions - data ||² + 
       || unknown_E * predicted_stress - actual_stress ||²
```

## Example for Cantilever Beam

```python
import torch
from torch.autograd import Variable

class BeamPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, 256),
            nn.Tanh(),  # Better for PDEs than ReLU
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1)  # Output: displacement u(x)
        )
  
    def forward(self, x):
        return self.mlp(x)
  
    def loss(self, params, displacements_true, length, load, E, I):
        # 1. Data loss
        disp_pred = self.forward(params)
        loss_data = ((disp_pred - displacements_true) ** 2).mean()
      
        # 2. Physics loss
        params_var = Variable(params, requires_grad=True)
        u = self.forward(params_var)
      
        # Compute d²u/dx² where x is length coordinate
        dudu = torch.autograd.grad(
            u, params_var,
            grad_outputs=torch.ones_like(u),
            create_graph=True
        )[0]  # du/d(params)
      
        # For simplicity: assume only x (length) contributes to curvature
        du_dx = dudu[:, 0]  # Sensitivity to length changes
        d2u_dx2 = torch.autograd.grad(
            du_dx, params_var,
            grad_outputs=torch.ones_like(du_dx),
            create_graph=True
        )[0][:, 0]
      
        # Euler-Bernoulli: d²u/dx² = -M / (E*I) = -P*x / (E*I)
        # Residual: d²u/dx² + load*x / (E*I)
        physics_residual = d2u_dx2 + load * length / (E * I)
      
        loss_physics = (physics_residual ** 2).mean()
      
        # Combined
        return loss_data + 0.1 * loss_physics
```

## Physics Constraints for Your Problem

### Boundary Conditions

```python
# Fixed at x=0:
loss_bc_fixed = (displacement_at_x0 ** 2).mean()
loss_bc_slope_zero = (du_dx_at_x0 ** 2).mean()

# Free at x=L:
loss_bc_free = (stress_at_xL ** 2).mean()
```

### Equilibrium Constraints

```python
# Force equilibrium: σ·A = P (constant along beam)
# Can encode as: d(σ·A)/dx = 0
```

### Energy Conservation

```python
# Strain energy: U = ∫ M²/(2EI) dx
# Work: W = P·u(L)
# Conservation: U = W
# Can be penalty term in loss
```

## Key Insight: Loss Function Design

```python
# Weights determine importance
loss = (
    1.0 * loss_data +           # Most important: fit observations
    0.1 * loss_pde_residual +   # Physics: satisfy equations
    0.01 * loss_boundaries      # Boundary conditions
)

# You need to balance these via hyperparameter tuning
# Not obvious what weights should be!
```

## Advantages

✅ **Reduced data**: Combine sparse measurements with physics
✅ **Generalization**: Extrapolation beyond training domain
✅ **Constraint satisfaction**: Guarantees boundary conditions
✅ **Interpretability**: Know which physics constraints matter
✅ **Few-shot learning**: Learn with very few samples

## Disadvantages

❌ **Complex loss design**: Balancing data vs physics is non-trivial
❌ **Hyperparameter sensitivity**: λ (weight) needs tuning
❌ **Expensive derivatives**: Computing d²u/dx² adds computation
❌ **PDE complexity**: Must explicitly code all physical laws
❌ **Debugging harder**: Hard to tell if issue is data or physics loss

## When to Use PINNs

✅ **Use if**:

- You have physics equations (you do: Euler-Bernoulli!)
- You want better generalization
- Data is sparse/expensive
- Safety-critical (need physics guarantees)

❌ **Don't use if**:

- Unknown governing equations
- Data is abundant & diverse
- Real-time inference critical (expensive backprop)

## Roadmap Integration

### Stage 1: Add PINN to your GCN model

```python
# Current loss:
loss = mse(predictions, ground_truth)

# PINN loss:
loss = mse(predictions, ground_truth) + 
       0.1 * pinn_physics_loss(predictions, params)
```

### Stage 2: Physics-aware features

Embed physics knowledge into node features:

```python
# Instead of just coordinates:
# node_features = [x, y, z, stress_analytical, curvature_analytical]
```

### Stage 3: Hybrid architecture

- Use GCN for learning mesh variations
- Use PINN loss to ensure physics satisfaction
- Best of both worlds!

## Implementation in TensorFlow

```python
# Your current stack supports this well
import tensorflow as tf
from tensorflow.python.ops import math_ops

@tf.function
def pinn_loss(model, params, displacements_true, load):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(params)
        disp_pred = model(params)
      
        # Data loss
        loss_data = tf.reduce_mean((disp_pred - displacements_true)**2)
      
        # Physics loss (derivatives)
        # d²disp/d²params  
        with tf.GradientTape() as tape2:
            tape2.watch(params)
            disp = model(params)
        d1 = tape2.gradient(disp, params)
      
        d2 = tape.gradient(d1, params)  # Second derivative
      
        # PDE residual
        residual = d2 + load / (E * I)
        loss_physics = tf.reduce_mean(residual**2)
  
    return loss_data + 0.1 * loss_physics
```

## Papers to Read Next

1. **PINNs original** (Raissi et al.) - This one
2. **DeepXDE** - Framework for PINNs
3. **Hamiltonian Neural Networks** - Physics-preserving architectures

---

## Recommendation for Your Project

**Hybrid approach**:

1. Train base GCN/DeepONet on your synthetic data (weeks 1-2)
2. Add PINN loss as regularization (week 3)
3. Compare: GCN-only vs GCN+PINN on generalization tests
4. Use PINN version for production (better safety guarantees)
