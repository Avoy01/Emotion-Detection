"""Model builders for the emotion detector."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from emotion_detector.config import DROPOUT_RATE, LEARNING_RATE
from emotion_detector.model.attention import AttentionLayer


def build_model(
    vocab_size: int,
    embedding_dim: int,
    embedding_matrix: np.ndarray,
    maxlen: int,
    lstm_units: int,
    attention_dim: int,
    dense_units: int,
    dropout_rate: float = DROPOUT_RATE,
    num_classes: int = 6,
    learning_rate: float = LEARNING_RATE,
) -> tf.keras.Model:
    """Construct and compile the BiLSTM-attention classifier."""

    inputs = tf.keras.Input(shape=(maxlen,), name="token_ids")
    x = tf.keras.layers.Embedding(
        input_dim=vocab_size + 1,
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        trainable=False,
        name="embedding",
    )(inputs)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            units=lstm_units,
            return_sequences=True,
            dropout=0.2,
            recurrent_dropout=0.2,
        ),
        name="bilstm",
    )(x)
    context_vector, _attention_weights = AttentionLayer(
        units=attention_dim,
        name="attention",
    )(x)
    x = tf.keras.layers.Dense(units=dense_units, activation="relu", name="dense")(context_vector)
    x = tf.keras.layers.Dropout(rate=dropout_rate, name="dropout")(x)
    outputs = tf.keras.layers.Dense(
        units=num_classes,
        activation="softmax",
        name="classifier",
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="emotion_detector")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_attention_extractor(model: tf.keras.Model) -> tf.keras.Model:
    """Build a helper model that returns attention weights."""

    attention_output = model.get_layer("attention").output[1]
    return tf.keras.Model(inputs=model.input, outputs=attention_output, name="attention_extractor")
