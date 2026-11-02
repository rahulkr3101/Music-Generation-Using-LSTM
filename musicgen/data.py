"""Loading and vectorizing the ABC-notation song corpus."""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def load_songs(data_dir: str = DEFAULT_DATA_DIR) -> list[str]:
    """Read every .txt file in data_dir and split it into individual songs."""
    songs: list[str] = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*.txt"))):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        for song in text.split("\n\n\n"):
            song = song.lstrip("\n")
            if song.strip():
                songs.append(song)
    if not songs:
        raise FileNotFoundError(f"No ABC notation .txt files found in {data_dir!r}")
    return songs


@dataclass
class Vocabulary:
    """Bidirectional mapping between characters and integer ids."""

    chars: list[str]

    @classmethod
    def from_text(cls, text: str) -> "Vocabulary":
        return cls(chars=sorted(set(text)))

    def __post_init__(self) -> None:
        self.char_to_id = {c: i for i, c in enumerate(self.chars)}
        self.id_to_char = np.array(self.chars)

    @property
    def size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> np.ndarray:
        return np.array([self.char_to_id[c] for c in text], dtype=np.int32)

    def decode(self, ids) -> str:
        return "".join(self.id_to_char[i] for i in ids)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.chars))

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        with open(path, encoding="utf-8") as f:
            chars = f.read().split("\n")
        return cls(chars=chars)


def build_corpus(data_dir: str = DEFAULT_DATA_DIR) -> tuple[str, Vocabulary, np.ndarray]:
    """Load all songs, build the vocab, and encode the whole corpus as ids."""
    songs = load_songs(data_dir)
    corpus = "\n\n".join(songs)
    vocab = Vocabulary.from_text(corpus)
    encoded = vocab.encode(corpus)
    return corpus, vocab, encoded


def get_batch(encoded: np.ndarray, seq_length: int, batch_size: int,
              rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Grab a random batch of (input, target) sequences, target = input shifted by 1."""
    rng = rng or np.random.default_rng()
    n = len(encoded) - 1
    start_idxs = rng.choice(n - seq_length, batch_size)
    x = np.stack([encoded[i : i + seq_length] for i in start_idxs])
    y = np.stack([encoded[i + 1 : i + seq_length + 1] for i in start_idxs])
    return x, y
