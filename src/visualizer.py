import pyvista as pv
import numpy as np
import joblib
import os
from src.data_generator import MockFEASolver

class ROMVisualizer:
    def __init__(self, model_dir="models"):
        self.model_disp = joblib.load(os.path.join(model_dir, "rom_disp.pkl"))
        self.model_stress = joblib.load(os.path.join(model_dir, "rom_stress.pkl"))
        self.scaler_x = joblib.load(os.path.join(model_dir, "scaler_x.pkl"))
        
        # We need to know original mesh shape to reconstruct
        # Usually we save metadata. For POC we hardcode or query MockFEASolver logic
        # MockFEASolver uses fixed 21x6x6
        self.nx, self.ny, self.nz = 21, 6, 6
        
    def predict_and_plot(self, length, width, depth, load, screenshot=None):
        # Prepare Input
        params = np.array([[length, width, depth, load]])
        params_s = self.scaler_x.transform(params)
        
        # Predict
        pred_disp_flat = self.model_disp.predict(params_s)
        pred_stress_flat = self.model_stress.predict(params_s)
        
        # Reshape
        # Disp: (N_points, 3)
        n_points = self.nx * self.ny * self.nz
        pred_disp = pred_disp_flat.reshape((n_points, 3))
        pred_stress = pred_stress_flat.reshape((n_points,)) # Scalar stress
        
        # Create Dummy Mesh to host data
        x = np.linspace(0, length, self.nx)
        y = np.linspace(-width/2, width/2, self.ny)
        z = np.linspace(-depth/2, depth/2, self.nz)
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        mesh = pv.StructuredGrid(xx, yy, zz)
        
        # Attach Data
        mesh.point_data["Predicted_Displacement"] = pred_disp
        mesh.point_data["Predicted_Stress"] = pred_stress
        
        # Warp
        warped = mesh.warp_by_vector("Predicted_Displacement", factor=1.0)
        
        # Plot
        pl = pv.Plotter(shape=(1, 2), off_screen=(screenshot is not None))
        
        pl.subplot(0, 0)
        pl.add_text("Predicted Stress (Warped)", font_size=10)
        pl.add_mesh(warped, scalars="Predicted_Stress", cmap="jet", show_edges=True)
        pl.show_grid()
        
        pl.subplot(0, 1)
        pl.add_text("Ground Truth (Simulated)", font_size=10)
        
        # Run solver for GT
        solver = MockFEASolver(output_dir="temp_gt")
        gt_mesh = solver.solve(length, width, depth, load, sample_id="temp")
        gt_warped = gt_mesh.warp_by_vector("Displacement", factor=1.0)
        pl.add_mesh(gt_warped, scalars="Stress_XX", cmap="jet", show_edges=True)
        pl.show_grid()
        
        pl.link_views()
        if screenshot:
            pl.show(screenshot=screenshot)
        else:
            pl.show()

if __name__ == "__main__":
    viz = ROMVisualizer()
    # Test case
    viz.predict_and_plot(length=12.0, width=1.5, depth=0.2, load=25000.0)
