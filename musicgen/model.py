"""The character-level LSTM language model."""

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf


@dataclass
class ModelConfig:
    vocab_size: int
    embedding_dim: int = 256
    lstm_units: int = 1024
    num_lstm_layers: int = 1
    dropout: float = 0.0


def build_model(config: ModelConfig, batch_size: int) -> tf.keras.Model:
    """Embedding -> (LSTM + optional Dropout) x N -> Dense.

    batch_size has to be fixed since the LSTM layers are stateful (state
    carries over between calls instead of resetting each batch).
    """
    layers = [
        tf.keras.layers.Embedding(
            config.vocab_size,
            config.embedding_dim,
            batch_input_shape=[batch_size, None],
        )
    ]
    for _ in range(config.num_lstm_layers):
        layers.append(
            tf.keras.layers.LSTM(
                config.lstm_units,
                return_sequences=True,
                recurrent_initializer="glorot_uniform",
                stateful=True,
            )
        )
        if config.dropout > 0:
            layers.append(tf.keras.layers.Dropout(config.dropout))
    layers.append(tf.keras.layers.Dense(config.vocab_size))
    return tf.keras.Sequential(layers)
