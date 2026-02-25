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
        super().__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
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
        H, A_hat = inputs          # H: (batch, N, F), A_hat: (batch, N, N)
        # H' = A_hat @ H @ W + b
        support = tf.matmul(H, self.W)          # (batch, N, units)
        output  = tf.matmul(A_hat, support) + self.b  # (batch, N, units)
        return self.activation(output)

    def get_config(self):
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
    # Self-loops included in A + I
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

    # Symmetric normalisation
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
    """
    GCN-based encoder followed by a Dense decoder head.
    - Lifts the scalar global parameters to per-node features.
    - 3 GCN message-passing layers.
    - Global mean-pool.
    - Dense head to predict the output field.

    Note: During training A_hat is treated as a constant broadcast over the batch.
    """
    N = A_hat.shape[0]  # 756

    # Inputs
    params_inp = keras.Input(shape=(input_dim,),  name="params")     # (B, 4)
    a_inp      = keras.Input(shape=(N, N),         name="A_hat")      # (B, N, N)

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
    # x: (B, N, 64)

    # Global average pool → (B, 64)
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
    def __init__(self, data_dir="mock_data", model_dir="models", model_type="gcn"):
        self.data_dir   = data_dir
        self.model_dir  = model_dir
        self.model_type = model_type.lower()
        os.makedirs(self.model_dir, exist_ok=True)

        if self.model_type not in ("mlp", "gcn"):
            raise ValueError(f"Unknown model_type '{self.model_type}'. Choose 'mlp' or 'gcn'.")

        # Precompute adjacency once (used only for GCN)
        self._A_hat = build_beam_adjacency(nx=21, ny=6, nz=6)  # (756, 756)

    # ── Data loading ───────────────────────────
    def load_data(self):
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
        print(f"Training with model: {self.model_type.upper()}")
        X, Y_disp, Y_stress = self.load_data()
        print(f"Data shape: X={X.shape}, Y_disp={Y_disp.shape}, Y_stress={Y_stress.shape}")

        # Scale inputs
        scaler_x = StandardScaler()
        X_train, X_test, yd_train, yd_test, ys_train, ys_test = \
            self._split_all(X, Y_disp, Y_stress)

        X_train_s = scaler_x.fit_transform(X_train).astype(np.float32)
        X_test_s  = scaler_x.transform(X_test).astype(np.float32)

        # 1. Displacement model
        print("\n[1/2] Training Displacement Model...")
        model_disp = self._train_one("Displacement", X_train_s, X_test_s,
                                     yd_train, yd_test, output_dim=Y_disp.shape[1])
        model_disp.save(os.path.join(self.model_dir, "rom_disp.keras"))

        # 2. Stress model
        print("\n[2/2] Training Stress Model...")
        model_stress = self._train_one("Stress", X_train_s, X_test_s,
                                       ys_train, ys_test, output_dim=Y_stress.shape[1])
        model_stress.save(os.path.join(self.model_dir, "rom_stress.keras"))

        # Save scaler as numpy arrays (no joblib dependency)
        np.save(os.path.join(self.model_dir, "scaler_mean.npy"), scaler_x.mean_.astype(np.float32))
        np.save(os.path.join(self.model_dir, "scaler_std.npy"),  scaler_x.scale_.astype(np.float32))
        # Also save model_type so visualizer can reconstruct GCN inputs if needed
        np.save(os.path.join(self.model_dir, "model_type.npy"),  np.array([self.model_type]))

        print("\nTraining complete. Models saved to:", self.model_dir)

    # ── Helper: consistent 3-way split ────────
    @staticmethod
    def _split_all(X, Y_disp, Y_stress, test_size=0.2, seed=42):
        idx = np.arange(len(X))
        idx_train, idx_test = train_test_split(idx, test_size=test_size, random_state=seed)
        return (X[idx_train], X[idx_test],
                Y_disp[idx_train], Y_disp[idx_test],
                Y_stress[idx_train], Y_stress[idx_test])


if __name__ == "__main__":
    trainer = ROMTrainer(model_type="mlp")
    trainer.train()
