import pyvista as pv
import numpy as np
import os
import keras
from src.data_generator import MockFEASolver
from src.rom_model import build_beam_adjacency, GCNLayer  # GCNLayer import triggers @register_keras_serializable


class ROMVisualizer:
    def __init__(self, model_dir="models"):
        # Load TF models
        self.model_disp   = keras.models.load_model(
            os.path.join(model_dir, "rom_disp.keras"),
            compile=False
        )
        self.model_stress = keras.models.load_model(
            os.path.join(model_dir, "rom_stress.keras"),
            compile=False
        )

        # Load numpy-based scaler
        self.scaler_mean = np.load(os.path.join(model_dir, "scaler_mean.npy"))
        self.scaler_std  = np.load(os.path.join(model_dir, "scaler_std.npy"))

        # Detect model type (mlp or gcn)
        model_type_arr = np.load(os.path.join(model_dir, "model_type.npy"), allow_pickle=True)
        self.model_type = str(model_type_arr[0])

        # Fixed mesh resolution (matches MockFEASolver)
        self.nx, self.ny, self.nz = 21, 6, 6

        # Precompute adjacency for GCN
        if self.model_type == "gcn":
            self._A_hat = build_beam_adjacency(self.nx, self.ny, self.nz)

    # ── Normalise input ────────────────────────
    def _scale(self, params: np.ndarray) -> np.ndarray:
        return ((params - self.scaler_mean) / self.scaler_std).astype(np.float32)

    # ── Build model input(s) ───────────────────
    def _make_inputs(self, params_s: np.ndarray):
        if self.model_type == "mlp":
            return params_s
        # GCN: tile adjacency to batch size 1
        A_batch = self._A_hat[np.newaxis]   # (1, N, N)
        return [params_s, A_batch]

    def predict_and_plot(self, length, width, depth, load, screenshot=None):
        # Prepare & scale input
        params   = np.array([[length, width, depth, load]], dtype=np.float32)
        params_s = self._scale(params)
        inputs   = self._make_inputs(params_s)

        # Predict
        pred_disp_flat   = self.model_disp.predict(inputs, verbose=0)
        pred_stress_flat = self.model_stress.predict(inputs, verbose=0)

        # Reshape
        n_points = self.nx * self.ny * self.nz
        pred_disp   = pred_disp_flat.reshape((n_points, 3))
        pred_stress = pred_stress_flat.reshape((n_points,))

        # Ground truth
        solver   = MockFEASolver(output_dir="temp_gt")
        gt_mesh  = solver.solve(length, width, depth, load, sample_id="temp")
        gt_disp   = np.array(gt_mesh.point_data["Displacement"])
        gt_stress = np.array(gt_mesh.point_data["Stress_XX"])

        # ── Scalar summaries ──────────────────────────────────────────────
        pred_max_disp   = float(np.max(np.linalg.norm(pred_disp,   axis=1)))
        pred_max_stress = float(np.max(np.abs(pred_stress)))
        gt_max_disp     = float(np.max(np.linalg.norm(gt_disp,     axis=1)))
        gt_max_stress   = float(np.max(np.abs(gt_stress)))

        def pct_err(pred, gt):
            if abs(gt) < 1e-12:
                return float("inf")
            return abs(pred - gt) / abs(gt) * 100.0

        disp_err   = pct_err(pred_max_disp,   gt_max_disp)
        stress_err = pct_err(pred_max_stress, gt_max_stress)

        # ── Shared font / text params ─────────────────────────────────────
        FONT      = "arial"
        TITLE_SZ  = 11
        ANNOT_SZ  = 9

        # ── Create meshes ─────────────────────────────────────────────────
        x = np.linspace(0, length, self.nx)
        y = np.linspace(-width/2, width/2, self.ny)
        z = np.linspace(-depth/2, depth/2, self.nz)
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        mesh = pv.StructuredGrid(xx, yy, zz)

        mesh.point_data["Predicted_Displacement"] = pred_disp
        mesh.point_data["Predicted_Stress"]       = pred_stress
        warped    = mesh.warp_by_vector("Predicted_Displacement", factor=1.0)
        gt_warped = gt_mesh.warp_by_vector("Displacement", factor=1.0)

        # ── Plot ──────────────────────────────────────────────────────────
        pl = pv.Plotter(shape=(1, 2), off_screen=(screenshot is not None))

        # ── Subplot 0: Predicted ─────────────────────────
        pl.subplot(0, 0)
        pred_title = (
            f"Predicted  [Disp err: {disp_err:.1f}%  |  Stress err: {stress_err:.1f}%]"
        )
        pl.add_text(pred_title, font_size=TITLE_SZ, font=FONT, position="upper_edge")
        pl.add_mesh(warped, scalars="Predicted_Stress", cmap="jet", show_edges=True)
        pl.show_grid()
        pl.show_axes()
        pl.add_text(
            f"Max |Disp|   = {pred_max_disp:.4e} mm\n"
            f"Max |Stress| = {pred_max_stress:.4e} MPa",
            font_size=ANNOT_SZ, font=FONT, position="lower_left",
        )

        # ── Subplot 1: Ground Truth ──────────────────────
        pl.subplot(0, 1)
        pl.add_text("Ground Truth (Simulated)", font_size=TITLE_SZ, font=FONT, position="upper_edge")
        pl.add_mesh(gt_warped, scalars="Stress_XX", cmap="jet", show_edges=True)
        pl.show_grid()
        pl.show_axes()
        pl.add_text(
            f"Max |Disp|   = {gt_max_disp:.4e} mm\n"
            f"Max |Stress| = {gt_max_stress:.4e} MPa",
            font_size=ANNOT_SZ, font=FONT, position="lower_left",
        )

        pl.link_views()
        if screenshot:
            pl.show(screenshot=screenshot)
        else:
            pl.show()


if __name__ == "__main__":
    viz = ROMVisualizer()
    viz.predict_and_plot(length=12.0, width=1.5, depth=0.2, load=25000.0)
