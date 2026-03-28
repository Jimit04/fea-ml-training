"""Model factory functions for MLP and GCN ROM architectures."""

import numpy as np
import tensorflow as tf
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

@keras.saving.register_keras_serializable(name="PositionalEmbedding")
class PositionalEmbedding(layers.Layer):
    """Simple trainable positional embedding layer."""
    def __init__(self, max_seq_len, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.embedding = layers.Embedding(input_dim=self.max_seq_len, output_dim=self.embed_dim)
        super().build(input_shape)

    def call(self, inputs):
        # inputs shape: (B, N, F)
        N_seq = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=N_seq, delta=1)
        # Broadcast positions to match batch size
        return inputs + self.embedding(positions)

    def get_config(self):
        config = super().get_config()
        config.update({
            "max_seq_len": self.max_seq_len,
            "embed_dim": self.embed_dim,
        })
        return config


def build_transformer(input_dim: int, output_dim: int, A_hat: np.ndarray) -> keras.Model:
    """Build a Transformer-based ROM model treating the mesh as a sequence.

    Architecture
    ------------
    1. Lifts the scalar global parameters to per-node features.
    2. Adds a learned positional embedding since Transformers are permutation-invariant.
    3. Several MultiHeadAttention layers + FeedForward networks (Transformer encoder blocks).
    4. ``GlobalAveragePooling1D`` to obtain a single graph-level vector.
    5. Dense decoder head to final ``output_dim``.

    Parameters
    ----------
    input_dim : int
        Number of global scalar parameters (typically 4).
    output_dim : int
        Flattened target size (e.g. 2268 for displacement, 756 for stress).
    A_hat : np.ndarray
        Pre-computed normalised adjacency matrix of shape ``(N, N)``.
        (Note: For a pure Transformer, this may just be used to infer N or can be passed 
        for compatibility with the API).

    Returns
    -------
    keras.Model
        Compiled Keras model with Adam optimiser and MSE loss.
    """
    N = A_hat.shape[0]  # e.g., 756
    
    # Inputs:
    params_inp = keras.Input(shape=(input_dim,), name="params")   # (B, 4)

    # 1. Lift global params → per-node feature matrix
    broadcast = layers.RepeatVector(N)(params_inp)              # (B, N, 4)
    node_init = layers.Dense(64, activation="swish")(broadcast)   # (B, N, 64)

    # 2. Add Positional Embedding 
    x = PositionalEmbedding(max_seq_len=N, embed_dim=64)(node_init)

    # 3. Transformer Encoder Blocks
    num_heads = 4
    embed_dim = 64
    ff_dim = 128
    
    for _ in range(3): # 3 Blocks
        # Self attention
        attn_out = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x, x)
        x = layers.Add()([x, attn_out])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Feed Foward
        ffn_out = layers.Dense(ff_dim, activation="swish")(x)
        ffn_out = layers.Dense(embed_dim)(ffn_out)
        x = layers.Add()([x, ffn_out])
        x = layers.LayerNormalization(epsilon=1e-6)(x)

    # 4. Global average pool → (B, 64)
    pooled = layers.GlobalAveragePooling1D()(x)

    # 5. Dense decoder head
    h = layers.Dense(256, activation="swish")(pooled)
    h = layers.Dropout(0.1)(h)
    h = layers.Dense(512, activation="swish")(h)
    out = layers.Dense(output_dim, name="output")(h)

    model = keras.Model(inputs=params_inp, outputs=out, name="Transformer_ROM")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )
    return model

