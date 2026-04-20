"""Custom Bahdanau-style attention layer."""

from __future__ import annotations

import tensorflow as tf


class AttentionLayer(tf.keras.layers.Layer):
    """Bahdanau-style additive attention for sequence encoders.

    Args:
        units: Dimensionality of the attention space.
    """

    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        """Create trainable attention weights."""

        self.W = self.add_weight(
            name="attention_W",
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="attention_b",
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
        )
        self.v = self.add_weight(
            name="attention_v",
            shape=(self.units, 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs) -> tuple[tf.Tensor, tf.Tensor]:
        """Return the attention-weighted context vector and weights."""

        score = tf.nn.tanh(tf.matmul(inputs, self.W) + self.b)
        attention_weights = tf.nn.softmax(tf.matmul(score, self.v), axis=1)
        context_vector = tf.reduce_sum(inputs * attention_weights, axis=1)
        return context_vector, attention_weights

    def get_config(self) -> dict:
        """Return layer config for model serialization."""

        config = super().get_config()
        config.update({"units": self.units})
        return config
