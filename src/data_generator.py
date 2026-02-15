import pyvista as pv
import numpy as np
import os

class MockFEASolver:
    def __init__(self, output_dir="data"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def solve(self, length, width, depth, load, sample_id=0):
        """
        Generates a synthetic FEA result (Cantilever beam bending) using Euler-Bernoulli Beam Theory.
        Geometry: Box(length, width, depth)
        BC: Fixed at x=0
        Load: Point load at x=length, y=0, z=0 (Applied at tip center)
        Material: Steel (E=210GPa, nu=0.29)
        
        Physics (Euler-Bernoulli):
        Displacement v(x) = - (P * x^2) / (6 * E * I) * (3*L - x)
        Stress sigma_xx(x, z) = M(x) * z / I
        Moment M(x) = P * (L - x)
        Moment of Inertia I = width * depth^3 / 12
        """
        
        # Constants
        E = 210000.0  # MPa
        nu = 0.29
        
        # 1. Create Mesh
        # Resolution (Fixed for ROM compatibility)
        nx, ny, nz = 21, 6, 6 # Fixed node count
        
        # Create StructuredGrid
        x = np.linspace(0, length, nx)
        y = np.linspace(-width/2, width/2, ny)
        z = np.linspace(-depth/2, depth/2, nz)
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        
        mesh = pv.StructuredGrid(xx, yy, zz)
        
        # 2. Compute Fields
        points = mesh.points
        # x coordinates are in column 0, z in column 2
        px = points[:, 0]
        pz = points[:, 2]
        
        # Moment of Inertia for rectangular cross-section
        I = width * (depth**3) / 12.0
        
        # Displacement Vector (3D)
        # Vertical displacement v(x) (downward if load is positive but convention typically P is load magnitude)
        # Current code had u_z = - alpha * x^2. 
        # Euler-Bernoulli beam with point load P at tip:
        # v(x) = - (P * x^2) / (6 * E * I) * (3*length - x) for 0 <= x <= length
        
        # Assuming 'load' is the force magnitude P acting downwards (negative z)
        # So v(x) should be negative.
        u_z = - (load * px**2) / (6 * E * I) * (3 * length - px)
        
        displacement = np.zeros_like(points)
        displacement[:, 2] = u_z
        
        # Stress Tensor (S_xx)
        # Bending Moment M(x) = P * (length - x)
        # Stress sigma_xx = M(x) * z / I
        # z is the distance from neutral axis. 
        # If load is downwards (producing tension at top z>0, compression at bottom z<0),
        # Actually standard sign convention: M is sagging positive. 
        # Downward load creates hogging (convex up) -> Negative Moment? 
        # Let's stick to magnitude logic: 
        # Top fibers (z>0) under tension if load is upwards? No, downward load -> top fibers extended? 
        # Cantilever with downward load: Top is in Tension (+), Bottom is in Compression (-).
        # M(x) = -P(L-x). sigma = M*z/I.
        # If P is magnitude of downward load, Moment is negative?
        # Let's maximize simple intuition: Downward load -> Tip simply moves down. 
        # Top surface (z > 0) stretches -> Stress > 0.
        # Bottom surface (z < 0) compresses -> Stress < 0.
        # So sigma ~ (L-x) * z * positive_constant.
        
        moment = load * (length - px)
        s_xx = moment * pz / I 
        
        # Add noise to make it realistic for ML
        noise = np.random.normal(0, 0.01 * np.max(np.abs(s_xx)) if np.max(np.abs(s_xx)) > 1e-9 else 1e-9, s_xx.shape)
        s_xx += noise
        
        # Add data to mesh
        mesh.point_data["Displacement"] = displacement
        mesh.point_data["Stress_XX"] = s_xx
        
        # 3. Save Data
        filename = f"sample_{sample_id}"
        if sample_id == "temp": # Don't save files for temp visualization
             return mesh
             
        # Save VTK for visualization/verification
        mesh.save(os.path.join(self.output_dir, f"{filename}.vtk"))
        
        # Save Features and Targets for ML
        # Input: [length, width, depth, load]
        # Target: Flattened Displacement or sampled points. 
        np.save(os.path.join(self.output_dir, f"{filename}_disp.npy"), displacement)
        np.save(os.path.join(self.output_dir, f"{filename}_stress.npy"), s_xx)
        # Saving new parameters
        np.save(os.path.join(self.output_dir, f"{filename}_params.npy"), np.array([length, width, depth, load]))
        
        return mesh

if __name__ == "__main__":
    solver = MockFEASolver(output_dir="mock_data")
    print("Generating sample 0...")
    solver.solve(length=10.0, width=2.0, load=100.0, stiffness=210.0, sample_id=0)
    print("Done.")
