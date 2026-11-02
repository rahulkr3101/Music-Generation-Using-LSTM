"""Train the character-level LSTM on the ABC-notation song corpus.

Usage::

    python -m musicgen.train --iterations 2000 --checkpoint-dir trained_weights
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import tensorflow as tf

from .data import DEFAULT_DATA_DIR, Vocabulary, build_corpus, get_batch
from .model import ModelConfig, build_model


def compute_loss(labels: tf.Tensor, logits: tf.Tensor) -> tf.Tensor:
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Directory of ABC .txt song files")
    p.add_argument("--checkpoint-dir", default="trained_weights", help="Where to save model weights")
    p.add_argument("--iterations", type=int, default=2000, help="Number of training steps")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--sequence-length", type=int, default=128)
    p.add_argument("--embedding-dim", type=int, default=256)
    p.add_argument("--lstm-units", type=int, default=1024)
    p.add_argument("--num-lstm-layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--learning-rate", type=float, default=5e-3)
    p.add_argument("--log-every", type=int, default=50, help="Print loss every N iterations")
    p.add_argument("--checkpoint-every", type=int, default=200,
                    help="Save weights every N iterations, in addition to the final save")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--history-csv", default=None,
                    help="Optional path to write per-iteration loss as CSV")
    return p.parse_args(argv)


def train(args: argparse.Namespace) -> list[float]:
    if args.seed is not None:
        tf.random.set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    _, vocab, encoded = build_corpus(args.data_dir)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    vocab.save(os.path.join(args.checkpoint_dir, "vocab.txt"))

    config = ModelConfig(
        vocab_size=vocab.size,
        embedding_dim=args.embedding_dim,
        lstm_units=args.lstm_units,
        num_lstm_layers=args.num_lstm_layers,
        dropout=args.dropout,
    )
    model = build_model(config, batch_size=args.batch_size)
    model.summary()

    optimizer = tf.keras.optimizers.Adam(args.learning_rate)
    weights_prefix = os.path.join(args.checkpoint_dir, "weights")

    @tf.function
    def train_step(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        with tf.GradientTape() as tape:
            y_hat = model(x)
            loss = compute_loss(y, y_hat)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    history: list[float] = []
    for i in range(args.iterations):
        x, y = get_batch(encoded, args.sequence_length, args.batch_size, rng=rng)
        loss = train_step(x, y)
        loss_value = float(tf.reduce_mean(loss).numpy())
        history.append(loss_value)

        if args.log_every and i % args.log_every == 0:
            print(f"iteration {i:5d} / {args.iterations}  loss={loss_value:.4f}")

        if args.checkpoint_every and (i + 1) % args.checkpoint_every == 0:
            model.save_weights(weights_prefix)

    model.save_weights(weights_prefix)
    print(f"Final weights saved to {weights_prefix}")

    if args.history_csv:
        with open(args.history_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["iteration", "loss"])
            writer.writerows(enumerate(history))
        print(f"Loss history written to {args.history_csv}")

    return history


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()
