"""Custom Keras layers for graph-based ROM models."""

import tensorflow as tf
import keras
from keras import layers


# ─────────────────────────────────────────────
# Custom GCN Layer
# ─────────────────────────────────────────────
@keras.saving.register_keras_serializable(package="GCNLayer")
class GCNLayer(layers.Layer):
    """Spectral Graph Convolutional Layer.

    Applies: ``H' = sigma(A_hat @ H @ W + b)``
    where ``A_hat`` is the symmetrically normalised adjacency matrix
    (precomputed).
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
        config.update({
            "units": self.units,
            "activation": keras.activations.serialize(self.activation),
        })
        return config
