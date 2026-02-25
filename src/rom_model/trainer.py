"""End-to-end training pipeline for ROM models."""

import datetime
import json
import numpy as np
import os
import glob
import keras
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.rom_model.adjacency import build_beam_adjacency
from src.rom_model.architectures import build_mlp, build_gcn


class ROMTrainer:
    """End-to-end training pipeline for MLP or GCN ROM models.

    Loads ``.npy`` samples produced by ``generate_dataset``, trains separate
    displacement and stress models, and persists the trained ``.keras`` files
    together with the input scaler.

    Attributes
    ----------
    data_dir : str
        Path to the directory containing ``*_params.npy``, ``*_disp.npy``,
        and ``*_stress.npy`` files.
    model_dir : str
        Path where trained models and scaler arrays are saved.
    model_type : str
        Either ``"mlp"`` or ``"gcn"``.
    """

    def __init__(self, data_dir="mock_data", model_dir="models", model_type="gcn"):
        """Initialise the trainer.

        Parameters
        ----------
        data_dir : str, optional
            Directory containing the training data (default ``"mock_data"``).
        model_dir : str, optional
            Directory for saving trained artefacts (default ``"models"``).
        model_type : str, optional
            Model architecture — ``"mlp"`` or ``"gcn"`` (default ``"gcn"``).

        Raises
        ------
        ValueError
            If *model_type* is not ``"mlp"`` or ``"gcn"``.
        """
        self.data_dir   = data_dir
        self.model_dir  = model_dir
        self.model_type = model_type.lower()
        os.makedirs(self.model_dir, exist_ok=True)

        if self.model_type not in ("mlp", "gcn"):
            raise ValueError(f"Unknown model_type '{self.model_type}'. Choose 'mlp' or 'gcn'.")

        # Precompute adjacency matrix (used only by GCN, but cheap to build)
        self._A_hat = build_beam_adjacency(nx=21, ny=6, nz=6)  # (756, 756)

    # ── Data loading ───────────────────────────
    def load_data(self):
        """Load all ``*_params``, ``*_disp``, and ``*_stress`` ``.npy`` files.

        Returns
        -------
        X : np.ndarray, shape (n_samples, 4)
            Input parameters ``[length, width, depth, load]``.
        Y_disp : np.ndarray, shape (n_samples, 2268)
            Flattened displacement vectors.
        Y_stress : np.ndarray, shape (n_samples, 756)
            Flattened stress scalars.

        Raises
        ------
        FileNotFoundError
            If no ``*_params.npy`` files are found in ``self.data_dir``.
        """
        print("Loading data...")
        param_files  = sorted(glob.glob(os.path.join(self.data_dir, "*_params.npy")))
        disp_files   = sorted(glob.glob(os.path.join(self.data_dir, "*_disp.npy")))
        stress_files = sorted(glob.glob(os.path.join(self.data_dir, "*_stress.npy")))

        if not param_files:
            raise FileNotFoundError("No data found. Run generate_dataset.py first.")

        X, Y_disp, Y_stress = [], [], []
        for p_f, d_f, s_f in zip(param_files, disp_files, stress_files):
            X.append(np.load(p_f))
            Y_disp.append(np.load(d_f).flatten())
            Y_stress.append(np.load(s_f).flatten())

        return np.array(X, dtype=np.float32), \
               np.array(Y_disp, dtype=np.float32), \
               np.array(Y_stress, dtype=np.float32)

    # ── Training callbacks ─────────────────────
    def _callbacks(self, monitor="val_loss", patience=20, log_dir="logs"):
        """Build the list of Keras training callbacks.

        Includes ``EarlyStopping``, ``ReduceLROnPlateau``, and a
        ``TensorBoard`` logger with a timestamped run directory.

        Parameters
        ----------
        monitor : str, optional
            Metric to monitor (default ``"val_loss"``).
        patience : int, optional
            Early-stopping patience in epochs (default ``20``).
        log_dir : str, optional
            Root directory for TensorBoard logs (default ``"logs"``).

        Returns
        -------
        list[keras.callbacks.Callback]
        """
        # Create timestamped log directory
        run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tb_log_dir = os.path.join(log_dir, run_id)

        tensorboard_cb = keras.callbacks.TensorBoard(
            log_dir=tb_log_dir,
            histogram_freq=1,          # log weight histograms
            write_graph=True,
            write_images=False,
            update_freq="epoch",       # or "batch" for finer granularity
            profile_batch=0            # set to (100,110) to profile specific batches
        )

        return [
            keras.callbacks.EarlyStopping(
                monitor=monitor,
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor=monitor,
                factor=0.5,
                patience=10,
                min_lr=1e-6,
                verbose=1
            ),
            tensorboard_cb
        ]

    # ── Prepare inputs for GCN ─────────────────
    def _gcn_inputs(self, X_scaled: np.ndarray):
        """Tile A_hat to batch dimension."""
        B = X_scaled.shape[0]
        A_batch = np.tile(self._A_hat[np.newaxis], (B, 1, 1))  # (B, N, N)
        return [X_scaled, A_batch]

    # ── Single model train helper ──────────────
    def _train_one(self, model_name: str, X_train_s, X_val_s, X_test_s,
                   y_train, y_val, y_test, output_dim: int):
        """Build, train, evaluate, and score a single model.

        Parameters
        ----------
        model_name : str
            Human-readable label used in log messages (e.g. ``"Displacement"``).
        X_train_s, X_val_s, X_test_s : np.ndarray
            Scaled input features for train / validation / test splits.
        y_train, y_val, y_test : np.ndarray
            Target arrays for train / validation / test splits.
        output_dim : int
            Number of output neurons.

        Returns
        -------
        tuple[keras.Model, float]
            The trained model (best weights restored) and R² score on the
            held-out test set.
        """

        if self.model_type == "mlp":
            model = build_mlp(input_dim=X_train_s.shape[1], output_dim=output_dim)
            train_in, val_in, test_in = X_train_s, X_val_s, X_test_s
        else:  # gcn
            model = build_gcn(input_dim=X_train_s.shape[1], output_dim=output_dim,
                               A_hat=self._A_hat)
            train_in = self._gcn_inputs(X_train_s)
            val_in   = self._gcn_inputs(X_val_s)
            test_in  = self._gcn_inputs(X_test_s)

        print(model.summary())
        history = model.fit(
            train_in, y_train,
            validation_data=(val_in, y_val),
            epochs=300,
            batch_size=min(64, len(y_train)),
            callbacks=self._callbacks(),
            verbose=1,
        )

        # Evaluate on held-out test set
        results = model.evaluate(test_in, y_test, verbose=0)
        print(f"  [{model_name}] Test MSE: {results[0]:.6f}  MAE: {results[1]:.6f}")

        # Compute R² on test set
        y_pred = model.predict(test_in, verbose=0)
        r2 = float(r2_score(y_test, y_pred))
        print(f"  [{model_name}] Test R²:  {r2:.6f}")

        return model, r2

    # ── Main train ─────────────────────────────
    def train(self):
        """Run the full training pipeline.

        1. Loads data via :meth:`load_data`.
        2. Splits into train (80%), test (15%), and validate (5%) sets.
        3. Scales inputs with ``StandardScaler``.
        4. Trains displacement and stress models sequentially.
        5. Computes R² on the held-out test set for each model.
        6. Saves ``.keras`` models, scaler arrays, model-type flag, and
           ``metrics.json`` to ``self.model_dir``.
        """
        print(f"Training with model: {self.model_type.upper()}")
        X, Y_disp, Y_stress = self.load_data()
        print(f"Data shape: X={X.shape}, Y_disp={Y_disp.shape}, Y_stress={Y_stress.shape}")

        # 80/15/5 split
        (X_train, X_test, X_val,
         yd_train, yd_test, yd_val,
         ys_train, ys_test, ys_val) = self._split_all(X, Y_disp, Y_stress)
        print(f"Split: train={len(X_train)}, test={len(X_test)}, val={len(X_val)}")

        # Standardise inputs
        scaler_x = StandardScaler()
        X_train_s = scaler_x.fit_transform(X_train).astype(np.float32)
        X_test_s  = scaler_x.transform(X_test).astype(np.float32)
        X_val_s   = scaler_x.transform(X_val).astype(np.float32)

        # Train displacement model
        print("\n[1/2] Training Displacement Model...")
        model_disp, r2_disp = self._train_one(
            "Displacement", X_train_s, X_val_s, X_test_s,
            yd_train, yd_val, yd_test, output_dim=Y_disp.shape[1])
        model_disp.save(os.path.join(self.model_dir, "rom_disp.keras"))

        # Train stress model
        print("\n[2/2] Training Stress Model...")
        model_stress, r2_stress = self._train_one(
            "Stress", X_train_s, X_val_s, X_test_s,
            ys_train, ys_val, ys_test, output_dim=Y_stress.shape[1])
        model_stress.save(os.path.join(self.model_dir, "rom_stress.keras"))

        # Persist scaler statistics as .npy (avoids joblib dependency)
        np.save(os.path.join(self.model_dir, "scaler_mean.npy"), scaler_x.mean_.astype(np.float32))
        np.save(os.path.join(self.model_dir, "scaler_std.npy"),  scaler_x.scale_.astype(np.float32))
        # Save model type so the visualiser knows how to prepare inputs
        np.save(os.path.join(self.model_dir, "model_type.npy"),  np.array([self.model_type]))

        # Save metrics (R² scores + split sizes)
        metrics = {
            "r2_displacement": r2_disp,
            "r2_stress": r2_stress,
            "model_type": self.model_type,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_val": len(X_val),
        }
        metrics_path = os.path.join(self.model_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to: {metrics_path}")
        print(f"  R² Displacement: {r2_disp:.6f}")
        print(f"  R² Stress:       {r2_stress:.6f}")

        print("\nTraining complete. Models saved to:", self.model_dir)

    # ── Helper: 3-way split (80/15/5) ─────────
    @staticmethod
    def _split_all(X, Y_disp, Y_stress, train_frac=0.80, test_frac=0.15, seed=42):
        """Split X, Y_disp, and Y_stress into train/test/validate sets.

        Parameters
        ----------
        X, Y_disp, Y_stress : np.ndarray
            Full dataset arrays.
        train_frac : float, optional
            Fraction of data for training (default ``0.80``).
        test_frac : float, optional
            Fraction of data for testing (default ``0.15``).
            The remainder (``1 - train_frac - test_frac``) is used for
            validation (default ``0.05``).
        seed : int, optional
            Random seed for reproducibility (default ``42``).

        Returns
        -------
        tuple
            ``(X_train, X_test, X_val,
            Y_disp_train, Y_disp_test, Y_disp_val,
            Y_stress_train, Y_stress_test, Y_stress_val)``.
        """
        idx = np.arange(len(X))
        # First split: train vs (test + val)
        rest_frac = 1.0 - train_frac
        idx_train, idx_rest = train_test_split(
            idx, test_size=rest_frac, random_state=seed)
        # Second split: test vs val from the remainder
        val_frac_of_rest = 1.0 - (test_frac / rest_frac)  # 0.05/0.20 = 0.25
        idx_test, idx_val = train_test_split(
            idx_rest, test_size=val_frac_of_rest, random_state=seed)
        return (X[idx_train], X[idx_test], X[idx_val],
                Y_disp[idx_train], Y_disp[idx_test], Y_disp[idx_val],
                Y_stress[idx_train], Y_stress[idx_test], Y_stress[idx_val])


if __name__ == "__main__":
    trainer = ROMTrainer(model_type="mlp")
    trainer.train()
