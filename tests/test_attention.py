"""Unit tests for the custom attention layer."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from emotion_detector.model.attention import AttentionLayer


def test_attention_output_shapes() -> None:
    """The attention layer should emit the expected tensor shapes."""

    layer = AttentionLayer(units=8)
    inputs = tf.random.uniform(shape=(2, 100, 256))
    context_vector, attention_weights = layer(inputs)

    assert context_vector.shape == (2, 256)
    assert attention_weights.shape == (2, 100, 1)


def test_attention_weights_sum_to_one() -> None:
    """Attention weights should form a probability distribution over time."""

    layer = AttentionLayer(units=8)
    inputs = tf.random.uniform(shape=(2, 100, 256))
    _context_vector, attention_weights = layer(inputs)

    sums = tf.reduce_sum(attention_weights, axis=1).numpy()
    assert np.allclose(sums, np.ones((2, 1)), atol=1e-5)
