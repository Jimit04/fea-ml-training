import pyvista as pv
import numpy as np
import os

class MockFEASolver:
    """Analytical cantilever-beam solver based on Euler-Bernoulli beam theory.

    Generates synthetic FEA-like results (displacement and stress fields) on a
    structured 21×6×6 hexahedral mesh.  Used to create labelled training data
    for the ROM neural networks.

    Attributes
    ----------
    output_dir : str
        Directory where ``.vtk`` meshes and ``.npy`` arrays are saved.
    """

    def __init__(self, output_dir="data"):
        """Initialise the solver and ensure the output directory exists.

        Parameters
        ----------
        output_dir : str, optional
            Path to the directory for saving generated samples (default ``"data"``).
        """
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
        
        # Material constants (Steel)
        E = 210000.0  # Young's Modulus [MPa]
        nu = 0.29     # Poisson's ratio

        # Create structured hex mesh (fixed resolution for ROM compatibility)
        nx, ny, nz = 21, 6, 6
        

        x = np.linspace(0, length, nx)
        y = np.linspace(-width/2, width/2, ny)
        z = np.linspace(-depth/2, depth/2, nz)
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        
        mesh = pv.StructuredGrid(xx, yy, zz)
        
        # Extract node coordinates
        points = mesh.points
        px = points[:, 0]  # x coordinates
        pz = points[:, 2]  # z coordinates (distance from neutral axis)
        
        # Moment of inertia for rectangular cross-section: I = w·d³/12
        I = width * (depth**3) / 12.0

        # Euler-Bernoulli tip-loaded cantilever displacement (z-direction)
        # v(x) = -(P·x²)/(6·E·I) · (3L - x)   →   negative = downward
        u_z = - (load * px**2) / (6 * E * I) * (3 * length - px)
        
        displacement = np.zeros_like(points)
        displacement[:, 2] = u_z
        
        # Bending stress σ_xx(x,z) = M(x)·z / I
        # M(x) = P·(L-x).  Positive load → top (z>0) in tension, bottom in compression.
        
        moment = load * (length - px)
        s_xx = moment * pz / I 
        
        # Add small noise to make data more realistic for ML training
        noise = np.random.normal(0, 0.01 * np.max(np.abs(s_xx)) if np.max(np.abs(s_xx)) > 1e-9 else 1e-9, s_xx.shape)
        s_xx += noise
        
        # Attach fields to mesh as point data
        mesh.point_data["Displacement"] = displacement
        mesh.point_data["Stress_XX"] = s_xx
        
        # Save outputs
        filename = f"sample_{sample_id}"
        if sample_id == "temp":  # Temp samples are for visualisation only — skip file I/O
             return mesh
             
        # Save VTK for visualization/verification
        mesh.save(os.path.join(self.output_dir, f"{filename}.vtk"))
        
        # Save ML arrays: params, displacement, and stress
        np.save(os.path.join(self.output_dir, f"{filename}_disp.npy"), displacement)
        np.save(os.path.join(self.output_dir, f"{filename}_stress.npy"), s_xx)
        np.save(os.path.join(self.output_dir, f"{filename}_params.npy"), np.array([length, width, depth, load]))
        
        return mesh

if __name__ == "__main__":
    solver = MockFEASolver(output_dir="mock_data")
    print("Generating sample 0...")
    solver.solve(length=10.0, width=2.0, depth=1.0, load=100.0, sample_id=0)
    print("Done.")
