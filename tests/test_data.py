import numpy as np

from musicgen.data import Vocabulary, get_batch


def test_vocabulary_roundtrip():
    vocab = Vocabulary.from_text("abcabc")
    encoded = vocab.encode("cab")
    assert vocab.decode(encoded) == "cab"


def test_vocabulary_save_load(tmp_path):
    vocab = Vocabulary.from_text("hello world")
    path = tmp_path / "vocab.txt"
    vocab.save(str(path))
    loaded = Vocabulary.load(str(path))
    assert loaded.chars == vocab.chars


def test_get_batch_shapes_and_shift():
    encoded = np.arange(1000)
    rng = np.random.default_rng(0)
    x, y = get_batch(encoded, seq_length=10, batch_size=4, rng=rng)
    assert x.shape == (4, 10)
    assert y.shape == (4, 10)
    # y is x shifted right by one position in the source array
    assert np.array_equal(y[:, :-1], x[:, 1:])
