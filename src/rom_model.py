"""ROM model definitions and training loop.

Contains the custom ``GCNLayer``, factory functions for building MLP and GCN
Keras models, and the ``ROMTrainer`` class that orchestrates data loading,
training, evaluation, and model persistence.
"""

import datetime
import numpy as np
import os
import glob
import tensorflow as tf
import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────
# Custom GCN Layer
# ─────────────────────────────────────────────
@keras.saving.register_keras_serializable(package="GCNLayer")
class GCNLayer(layers.Layer):
    """
    Spectral Graph Convolutional Layer.
    Applies: H' = sigma(A_hat @ H @ W)
    where A_hat is the symmetrically normalised adjacency matrix (precomputed).
    """
    def __init__(self, units, activation="relu", **kwargs):
        """Create a GCN layer.

        Parameters
        ----------
        units : int
            Dimensionality of the output feature space.
        activation : str or callable, optional
            Activation function applied after the graph convolution
            (default ``"relu"``).
        """
        super().__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        """Create the weight matrix ``W`` and bias ``b``.

        Parameters
        ----------
        input_shape : list of TensorShape
            ``[H_shape, A_hat_shape]`` where ``H_shape = (batch, N, F)``.
        """
        # input_shape: [(batch, N, F), (batch, N, N)]
        feature_dim = input_shape[0][-1]
        self.W = self.add_weight(
            name="W",
            shape=(feature_dim, self.units),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="b",
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        """Forward pass: ``H' = activation(A_hat @ H @ W + b)``.

        Parameters
        ----------
        inputs : list of tf.Tensor
            ``[H, A_hat]`` — node features ``(B, N, F)`` and normalised
            adjacency ``(B, N, N)``.

        Returns
        -------
        tf.Tensor
            Updated node features of shape ``(B, N, units)``.
        """
        H, A_hat = inputs          # H: (batch, N, F), A_hat: (batch, N, N)
        # H' = A_hat @ H @ W + b
        support = tf.matmul(H, self.W)          # (batch, N, units)
        output  = tf.matmul(A_hat, support) + self.b  # (batch, N, units)
        return self.activation(output)

    def get_config(self):
        """Return layer config for Keras serialisation."""

        config = super().get_config()
        config.update({"units": self.units, "activation": keras.activations.serialize(self.activation)})
        return config


# ─────────────────────────────────────────────
# Adjacency builder for structured 21×6×6 grid
# ─────────────────────────────────────────────
def build_beam_adjacency(nx=21, ny=6, nz=6):
    """
    Build normalised adjacency matrix A_hat = D^{-1/2} (A + I) D^{-1/2}
    for a structured hex mesh with nx*ny*nz nodes.
    Nodes are adjacent if they differ by 1 step along any single axis.
    """
    N = nx * ny * nz
    def idx(i, j, k):
        return i * ny * nz + j * nz + k

    rows, cols = [], []
    # Self-loops are already included (A + I)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                n = idx(i, j, k)
                rows.append(n); cols.append(n)  # self-loop
                for di, dj, dk in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
                    ni, nj, nk = i+di, j+dj, k+dk
                    if 0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz:
                        rows.append(n); cols.append(idx(ni, nj, nk))

    A = np.zeros((N, N), dtype=np.float32)
    A[rows, cols] = 1.0

    # Symmetric normalisation: A_hat = D^{-1/2} A D^{-1/2}
    deg = A.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(deg, 1e-9)))
    A_hat = D_inv_sqrt @ A @ D_inv_sqrt
    return A_hat.astype(np.float32)             # (N, N)


# ─────────────────────────────────────────────
# Model factories
# ─────────────────────────────────────────────
def build_mlp(input_dim: int, output_dim: int) -> keras.Model:
    """
    Deep MLP with SiLU activations, BatchNorm and Dropout.
    Architecture: 4 → 256 → 512 → 512 → 256 → output_dim
    """
    inp = keras.Input(shape=(input_dim,), name="params")
    x = layers.Dense(256)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)
    x = layers.Dropout(0.15)(x)

    x = layers.Dense(512)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)
    x = layers.Dropout(0.15)(x)

    x = layers.Dense(512)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)

    x = layers.Dense(256)(x)
    x = layers.Activation("swish")(x)

    out = layers.Dense(output_dim, name="output")(x)
    model = keras.Model(inputs=inp, outputs=out, name="MLP_ROM")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )
    return model


def build_gcn(input_dim: int, output_dim: int, A_hat: np.ndarray) -> keras.Model:
    """Build a GCN-based ROM model.

    Architecture
    ------------
    1. Lifts the scalar global parameters to per-node features via
       ``RepeatVector`` + ``Dense(32)``.
    2. Six ``GCNLayer(128)`` message-passing layers with alternating
       ReLU / LeakyReLU activations.
    3. ``GlobalAveragePooling1D`` to obtain a single graph-level vector.
    4. Dense decoder head (256 → 512 → ``output_dim``).

    Parameters
    ----------
    input_dim : int
        Number of global scalar parameters (typically 4).
    output_dim : int
        Flattened target size (e.g. 2268 for displacement, 756 for stress).
    A_hat : np.ndarray
        Pre-computed normalised adjacency matrix of shape ``(N, N)``.

    Returns
    -------
    keras.Model
        Compiled Keras model with Adam optimiser and MSE loss.

    Notes
    -----
    During training ``A_hat`` is treated as a constant broadcast over the
    batch dimension.
    """
    N = A_hat.shape[0]  # 756

    # Inputs
    params_inp = keras.Input(shape=(input_dim,),  name="params")   # (B, 4)
    a_inp      = keras.Input(shape=(N, N),         name="A_hat")    # (B, N, N)

    # Lift global params → per-node feature matrix (B, N, 32)
    broadcast = layers.RepeatVector(N)(params_inp)  # (B, N, 4)
    node_init = layers.Dense(32, activation="swish")(broadcast)   # (B, N, 32)

    # GCN stack
    x = GCNLayer(128,  activation="relu",       name="gcn_1")([node_init, a_inp])
    x = GCNLayer(128,  activation="leaky_relu", name="gcn_2")([x, a_inp])
    x = GCNLayer(128,  activation="relu",       name="gcn_3")([x, a_inp])
    x = GCNLayer(128,  activation="leaky_relu", name="gcn_4")([x, a_inp])
    x = GCNLayer(128,  activation="relu",       name="gcn_5")([x, a_inp])
    x = GCNLayer(128,  activation="leaky_relu", name="gcn_6")([x, a_inp])
    # x: (B, N, 128)

    # Global average pool → (B, 128)
    pooled = layers.GlobalAveragePooling1D()(x)

    # Dense decoder head
    h = layers.Dense(256, activation="swish")(pooled)
    h = layers.Dropout(0.1)(h)
    h = layers.Dense(512, activation="swish")(h)
    out = layers.Dense(output_dim, name="output")(h)

    model = keras.Model(inputs=[params_inp, a_inp], outputs=out, name="GCN_ROM")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )
    return model


# ─────────────────────────────────────────────
# ROMTrainer
# ─────────────────────────────────────────────
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
    def _train_one(self, model_name: str, X_train_s, X_test_s,
                   y_train, y_test, output_dim: int) -> keras.Model:
        """Build, train, and evaluate a single displacement or stress model.

        Parameters
        ----------
        model_name : str
            Human-readable label used in log messages (e.g. ``"Displacement"``).
        X_train_s, X_test_s : np.ndarray
            Scaled input features for train / test splits.
        y_train, y_test : np.ndarray
            Target arrays for train / test splits.
        output_dim : int
            Number of output neurons.

        Returns
        -------
        keras.Model
            The trained model with best weights restored.
        """

        if self.model_type == "mlp":
            model = build_mlp(input_dim=X_train_s.shape[1], output_dim=output_dim)
            train_in, test_in = X_train_s, X_test_s
        else:  # gcn
            model = build_gcn(input_dim=X_train_s.shape[1], output_dim=output_dim,
                               A_hat=self._A_hat)
            train_in = self._gcn_inputs(X_train_s)
            test_in  = self._gcn_inputs(X_test_s)

        print(model.summary())
        history = model.fit(
            train_in, y_train,
            validation_data=(test_in, y_test),
            epochs=300,
            batch_size=min(64, len(y_train)),
            callbacks=self._callbacks(),
            verbose=1,
        )

        # Evaluate
        results = model.evaluate(test_in, y_test, verbose=0)
        print(f"  [{model_name}] Test MSE: {results[0]:.6f}  MAE: {results[1]:.6f}")

        return model

    # ── Main train ─────────────────────────────
    def train(self):
        """Run the full training pipeline.

        1. Loads data via :meth:`load_data`.
        2. Scales inputs with ``StandardScaler``.
        3. Trains displacement and stress models sequentially.
        4. Saves ``.keras`` models, scaler arrays, and model-type flag to
           ``self.model_dir``.
        """
        print(f"Training with model: {self.model_type.upper()}")
        X, Y_disp, Y_stress = self.load_data()
        print(f"Data shape: X={X.shape}, Y_disp={Y_disp.shape}, Y_stress={Y_stress.shape}")

        # Standardise inputs
        scaler_x = StandardScaler()
        X_train, X_test, yd_train, yd_test, ys_train, ys_test = \
            self._split_all(X, Y_disp, Y_stress)

        X_train_s = scaler_x.fit_transform(X_train).astype(np.float32)
        X_test_s  = scaler_x.transform(X_test).astype(np.float32)

        # Train displacement model
        print("\n[1/2] Training Displacement Model...")
        model_disp = self._train_one("Displacement", X_train_s, X_test_s,
                                     yd_train, yd_test, output_dim=Y_disp.shape[1])
        model_disp.save(os.path.join(self.model_dir, "rom_disp.keras"))

        # Train stress model
        print("\n[2/2] Training Stress Model...")
        model_stress = self._train_one("Stress", X_train_s, X_test_s,
                                       ys_train, ys_test, output_dim=Y_stress.shape[1])
        model_stress.save(os.path.join(self.model_dir, "rom_stress.keras"))

        # Persist scaler statistics as .npy (avoids joblib dependency)
        np.save(os.path.join(self.model_dir, "scaler_mean.npy"), scaler_x.mean_.astype(np.float32))
        np.save(os.path.join(self.model_dir, "scaler_std.npy"),  scaler_x.scale_.astype(np.float32))
        # Save model type so the visualiser knows how to prepare inputs
        np.save(os.path.join(self.model_dir, "model_type.npy"),  np.array([self.model_type]))

        print("\nTraining complete. Models saved to:", self.model_dir)

    # ── Helper: consistent 3-way split ────────
    @staticmethod
    def _split_all(X, Y_disp, Y_stress, test_size=0.2, seed=42):
        """Split X, Y_disp, and Y_stress into train/test sets consistently.

        Returns
        -------
        tuple
            ``(X_train, X_test, Y_disp_train, Y_disp_test,
            Y_stress_train, Y_stress_test)``.
        """
        idx = np.arange(len(X))
        idx_train, idx_test = train_test_split(idx, test_size=test_size, random_state=seed)
        return (X[idx_train], X[idx_test],
                Y_disp[idx_train], Y_disp[idx_test],
                Y_stress[idx_train], Y_stress[idx_test])


if __name__ == "__main__":
    trainer = ROMTrainer(model_type="mlp")
    trainer.train()
