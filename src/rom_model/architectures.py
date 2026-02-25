"""Model factory functions for MLP and GCN ROM architectures."""

import numpy as np
import keras
from keras import layers

from src.rom_model.layers import GCNLayer


def build_mlp(input_dim: int, output_dim: int) -> keras.Model:
    """Build a deep MLP ROM model.

    Architecture: ``input_dim → 256 → 512 → 512 → 256 → output_dim``
    with SiLU (Swish) activations, BatchNorm, and Dropout.

    Parameters
    ----------
    input_dim : int
        Number of input features (typically 4).
    output_dim : int
        Flattened target size (e.g. 2268 for displacement, 756 for stress).

    Returns
    -------
    keras.Model
        Compiled Keras model with Adam optimiser and MSE loss.
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
