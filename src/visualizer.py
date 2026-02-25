"""3-D visualiser comparing ROM predictions against analytical ground truth.

Uses PyVista to render side-by-side subplots of predicted and ground-truth
displacement / stress fields on a warped beam mesh.
"""

import pyvista as pv
import numpy as np
import json
import os
import keras
from src.data_generator import MockFEASolver
from src.rom_model import build_beam_adjacency, GCNLayer  # GCNLayer import registers the custom layer for Keras deserialization


class ROMVisualizer:
    """Load trained ROM models and visualise predictions vs. ground truth.

    Supports both MLP and GCN model types.  The model type is detected
    automatically from the saved ``model_type.npy`` file.

    Attributes
    ----------
    model_disp : keras.Model
        Loaded displacement model.
    model_stress : keras.Model
        Loaded stress model.
    model_type : str
        ``"mlp"`` or ``"gcn"``.
    nx, ny, nz : int
        Mesh resolution (must match ``MockFEASolver``).
    """

    def __init__(self, model_dir="models"):
        """Load models, scaler, and adjacency matrix from *model_dir*.

        Parameters
        ----------
        model_dir : str, optional
            Directory containing ``rom_disp.keras``, ``rom_stress.keras``,
            ``scaler_mean.npy``, ``scaler_std.npy``, and ``model_type.npy``
            (default ``"models"``).
        """
        # Load Keras models (compile=False since we only need inference)
        self.model_disp   = keras.models.load_model(
            os.path.join(model_dir, "rom_disp.keras"),
            compile=False
        )
        self.model_stress = keras.models.load_model(
            os.path.join(model_dir, "rom_stress.keras"),
            compile=False
        )

        # Load saved scaler statistics (mean and std from training set)
        self.scaler_mean = np.load(os.path.join(model_dir, "scaler_mean.npy"))
        self.scaler_std  = np.load(os.path.join(model_dir, "scaler_std.npy"))

        # Detect model type from saved metadata
        model_type_arr = np.load(os.path.join(model_dir, "model_type.npy"), allow_pickle=True)
        self.model_type = str(model_type_arr[0])

        # Mesh resolution (must match MockFEASolver)
        self.nx, self.ny, self.nz = 21, 6, 6

        # Build adjacency matrix for GCN inference
        if self.model_type == "gcn":
            self._A_hat = build_beam_adjacency(self.nx, self.ny, self.nz)

        # Load training metrics (R² scores) if available
        metrics_path = os.path.join(model_dir, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                self.metrics = json.load(f)
        else:
            self.metrics = {}

    # ── Normalise input ────────────────────────
    def _scale(self, params: np.ndarray) -> np.ndarray:
        """Standardise *params* using the saved training-set statistics.

        Parameters
        ----------
        params : np.ndarray, shape (1, 4)
            Raw input ``[length, width, depth, load]``.

        Returns
        -------
        np.ndarray
            Scaled parameters as ``float32``.
        """
        return ((params - self.scaler_mean) / self.scaler_std).astype(np.float32)

    # ── Build model input(s) ───────────────────
    def _make_inputs(self, params_s: np.ndarray):
        """Prepare model input(s) depending on model type.

        For MLP models the scaled parameters are returned as-is.  For GCN
        models the normalised adjacency matrix is tiled to batch dimension
        and returned alongside the parameters.

        Parameters
        ----------
        params_s : np.ndarray
            Scaled input parameters.

        Returns
        -------
        np.ndarray or list[np.ndarray]
            Model-ready input(s).
        """
        if self.model_type == "mlp":
            return params_s
        # GCN: add adjacency as second input with batch dim
        A_batch = self._A_hat[np.newaxis]   # (1, N, N)
        return [params_s, A_batch]

    def predict_and_plot(self, length, width, depth, load, screenshot=None):
        """Predict displacement / stress fields and render a comparison plot.

        Runs the trained ROM for the given beam parameters, computes the
        analytical ground truth via ``MockFEASolver``, and displays both
        side-by-side in a PyVista plotter with error annotations.

        Parameters
        ----------
        length, width, depth : float
            Beam geometry in mm.
        load : float
            Applied tip load in N.
        screenshot : str or None, optional
            If given, save the visualisation to this file path instead of
            opening an interactive window.
        """
        # Scale input parameters
        params   = np.array([[length, width, depth, load]], dtype=np.float32)
        params_s = self._scale(params)
        inputs   = self._make_inputs(params_s)

        # Run ROM inference
        pred_disp_flat   = self.model_disp.predict(inputs, verbose=0)
        pred_stress_flat = self.model_stress.predict(inputs, verbose=0)

        # Reshape flat predictions back to mesh layout
        n_points = self.nx * self.ny * self.nz
        pred_disp   = pred_disp_flat.reshape((n_points, 3))
        pred_stress = pred_stress_flat.reshape((n_points,))

        # Compute analytical ground truth for comparison
        solver   = MockFEASolver(output_dir="temp_gt")
        gt_mesh  = solver.solve(length, width, depth, load, sample_id="temp")
        gt_disp   = np.array(gt_mesh.point_data["Displacement"])
        gt_stress = np.array(gt_mesh.point_data["Stress_XX"])

        # Compute max displacement / stress and % errors
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

        # Plot configuration
        FONT      = "arial"
        TITLE_SZ  = 11
        ANNOT_SZ  = 9

        # Build mesh and attach predicted fields
        x = np.linspace(0, length, self.nx)
        y = np.linspace(-width/2, width/2, self.ny)
        z = np.linspace(-depth/2, depth/2, self.nz)
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        mesh = pv.StructuredGrid(xx, yy, zz)

        mesh.point_data["Predicted_Displacement"] = pred_disp
        mesh.point_data["Predicted_Stress"]       = pred_stress
        warped    = mesh.warp_by_vector("Predicted_Displacement", factor=1.0)
        gt_warped = gt_mesh.warp_by_vector("Displacement", factor=1.0)

        # Render side-by-side comparison
        pl = pv.Plotter(shape=(1, 2), off_screen=(screenshot is not None))

        # Left panel: ROM prediction
        pl.subplot(0, 0)
        pred_title = (
            f"Predicted  [Disp err: {disp_err:.1f}%  |  Stress err: {stress_err:.1f}%]"
        )
        pl.add_text(pred_title, font_size=TITLE_SZ, font=FONT, position="upper_edge")
        pl.add_mesh(warped, scalars="Predicted_Stress", cmap="jet", show_edges=True)
        pl.show_grid()
        pl.show_axes()
        # R² annotation (from training metrics)
        r2_disp   = self.metrics.get("r2_displacement")
        r2_stress = self.metrics.get("r2_stress")
        r2_text = ""
        if r2_disp is not None:
            r2_text += f"\nR² Disp   = {r2_disp:.4f}"
        if r2_stress is not None:
            r2_text += f"\nR² Stress = {r2_stress:.4f}"

        pl.add_text(
            f"Max |Disp|   = {pred_max_disp:.4e} mm\n"
            f"Max |Stress| = {pred_max_stress:.4e} MPa"
            f"{r2_text}",
            font_size=ANNOT_SZ, font=FONT, position="lower_left",
        )

        # Right panel: analytical ground truth
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
            pl.show(screenshot=screenshot, window_size=(1920, 1080))
        else:
            pl.show()


if __name__ == "__main__":
    viz = ROMVisualizer()
    viz.predict_and_plot(length=12.0, width=1.5, depth=0.2, load=25000.0)
