"""Generate ABC tunes from a trained model and optionally render them to audio.

python -m musicgen.generate --checkpoint-dir trained_weights --num-songs 5 --render-audio
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import tempfile
import typing

from .data import Vocabulary

if typing.TYPE_CHECKING:
    import tensorflow as tf

# A song in the Nottingham-style corpus starts with an "X:<index>" header and
# runs until the next header (or end of text).
SONG_PATTERN = re.compile(r"(^|\n\n)(X:\d+[\s\S]*?)(?=\n\nX:|\Z)")


def extract_songs(text: str) -> list[str]:
    """Pull out individual well-formed ABC tunes from generated text."""
    return [m.group(2).strip() for m in SONG_PATTERN.finditer(text)]


def generate_text(
    model: "tf.keras.Model",
    vocab: Vocabulary,
    start_string: str = "X",
    length: int = 1000,
    temperature: float = 1.0,
) -> str:
    """Sample `length` characters from the model, feeding each prediction back in.

    Lower temperature = safer/closer to training data, higher = more novel
    but more likely to produce broken ABC syntax.
    """
    import tensorflow as tf

    input_eval = tf.expand_dims(vocab.encode(start_string), 0)
    model.reset_states()

    generated = []
    for _ in range(length):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0) / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        generated.append(predicted_id)

    return start_string + vocab.decode(generated)


def abc_to_wav(abc_song: str, out_path: str) -> bool:
    """Render one ABC tune to WAV via abc2midi + timidity. Returns False if
    those tools aren't installed."""
    if not (shutil.which("abc2midi") and shutil.which("timidity")):
        return False

    with tempfile.TemporaryDirectory() as tmp:
        abc_path = os.path.join(tmp, "song.abc")
        midi_path = os.path.join(tmp, "song.mid")
        with open(abc_path, "w", encoding="utf-8") as f:
            f.write(abc_song)

        subprocess.run(["abc2midi", abc_path, "-o", midi_path], check=True, capture_output=True)
        subprocess.run(["timidity", midi_path, "-Ow", "-o", out_path], check=True, capture_output=True)
    return True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-dir", default="trained_weights", help="Directory with weights.* and vocab.txt")
    p.add_argument("--output-dir", default="generated", help="Where to write generated songs")
    p.add_argument("--num-songs", type=int, default=5, help="Number of attempts to sample a complete tune from")
    p.add_argument("--start-string", default="X", help="Seed text to begin generation from")
    p.add_argument("--length", type=int, default=1000, help="Number of characters to sample per attempt")
    p.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature; higher = more novel/risky")
    p.add_argument("--embedding-dim", type=int, default=256)
    p.add_argument("--lstm-units", type=int, default=1024)
    p.add_argument("--num-lstm-layers", type=int, default=1)
    p.add_argument("--render-audio", action="store_true",
                    help="Also render each tune to WAV via abc2midi + timidity, if installed")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    import tensorflow as tf

    from .model import ModelConfig, build_model

    args = parse_args(argv)

    vocab = Vocabulary.load(os.path.join(args.checkpoint_dir, "vocab.txt"))
    config = ModelConfig(
        vocab_size=vocab.size,
        embedding_dim=args.embedding_dim,
        lstm_units=args.lstm_units,
        num_lstm_layers=args.num_lstm_layers,
    )
    model = build_model(config, batch_size=1)
    model.load_weights(tf.train.latest_checkpoint(args.checkpoint_dir))
    model.build(tf.TensorShape([1, None]))

    os.makedirs(args.output_dir, exist_ok=True)

    song_count = 0
    for attempt in range(args.num_songs):
        text = generate_text(model, vocab, args.start_string, args.length, args.temperature)
        songs = extract_songs(text)
        if not songs:
            print(f"attempt {attempt}: no complete tune found in sampled text, skipping")
            continue

        for song in songs:
            song_count += 1
            abc_path = os.path.join(args.output_dir, f"song_{song_count:03d}.abc")
            with open(abc_path, "w", encoding="utf-8") as f:
                f.write(song)
            print(f"wrote {abc_path}")

            if args.render_audio:
                wav_path = os.path.join(args.output_dir, f"song_{song_count:03d}.wav")
                if abc_to_wav(song, wav_path):
                    print(f"rendered {wav_path}")
                else:
                    print("abc2midi/timidity not found on PATH; skipping audio rendering "
                          "(install abcmidi + timidity, e.g. `apt-get install abcmidi timidity`)")

    if song_count == 0:
        print("No complete tunes were generated. Try a longer --length or lower --temperature.")


if __name__ == "__main__":
    main()
