# Music Generation Using LSTM

A character-level LSTM that learns to write folk tunes in ABC notation — trained on Irish jigs, hornpipes, waltzes, morris dances, carols, and Playford tunes from the Nottingham Music Database — and can turn what it writes into actual audio.

Originally a single Colab notebook (still in `legacy/`), split up here into a small package so training/generation can be run from the command line instead of a notebook.

🎧 Samples: [`samples/`](samples/)

## How it works

The corpus is a bunch of folk tunes in ABC notation (basically text). Every unique character gets mapped to an id, and an `Embedding -> LSTM -> Dense` model is trained to predict the next character given the ones before it. Once trained, you seed it with `X` (start of an ABC header) and let it generate character by character, feeding its own output back in. The result is ABC text, which gets converted to MIDI (`abc2midi`) and then WAV (`timidity`) so you can listen to it.

## Layout

```
musicgen/
├── data.py      - loading songs, vocab, batching
├── model.py     - the Embedding/LSTM/Dense model
├── train.py     - training CLI
└── generate.py  - sampling + abc -> midi -> wav
tests/           - a few unit tests
data/            - ABC notation songs (Nottingham Music Database)
trained_weights/ - pretrained weights + vocab, ready to generate from
samples/         - example generated audio
legacy/          - original notebook/script
```

## Setup

```bash
git clone https://github.com/rahulkr3101/Music-Generation-Using-LSTM.git
cd Music-Generation-Using-LSTM
pip install -r requirements.txt
```

Audio rendering needs `abc2midi` and `timidity` on your PATH (`apt-get install abcmidi timidity` on Ubuntu). Without them you still get the generated `.abc` text, just no `.wav`.

## Generating with the pretrained weights

```bash
python -m musicgen.generate --checkpoint-dir trained_weights --num-songs 5 --temperature 0.9 --render-audio
```

`--temperature` controls how safe vs. adventurous the sampling is — lower sticks closer to the training data, higher gets weirder and more likely to break ABC syntax. `--length` is how many characters to sample per attempt (longer = more chances of landing a full tune).

## Training from scratch

```bash
python -m musicgen.train --iterations 2000 --batch-size 32 --sequence-length 128 --checkpoint-dir trained_weights
```

Loads everything in `data/`, trains, and saves weights + the vocab (needed later for generation) to `--checkpoint-dir`.

## Changes from the original notebook

The original was one long Colab script. Breaking it up into `data.py`/`model.py`/`train.py`/`generate.py` made a few things easier to fix along the way:

- CLI args instead of hardcoded globals for batch size, sequence length, LSTM size, etc.
- The vocabulary wasn't being saved anywhere before, so generation depended on re-deriving it from the training corpus in memory — now it's written to `trained_weights/vocab.txt`.
- Song extraction used to depend on the `mitdeeplearning` teaching library; replaced with a small regex-based version that's actually testable.
- Audio generation no longer needs a Colab/Jupyter widget — it shells out to `abc2midi`/`timidity` directly and just saves `.abc` text if those aren't installed.
- Added a `--temperature` flag (the original always sampled at temperature 1.0).
- A couple of unit tests for the batching logic and song extraction, since those were the two places most likely to silently break.

## LSTM background

![LSTM block](https://miro.medium.com/max/1400/0*3sFqvDBejuRpjcNm.png)

LSTMs are RNNs with a "cell state" that runs through the whole sequence like a conveyor belt, plus gates (forget/input/output) that control what gets added or removed from it. That's what lets them hold onto context over longer sequences than a plain RNN, which is useful here since a tune should stay in key and keep its rhythm over many notes.

The LSTM layers here are stateful, meaning the hidden state carries over between training batches instead of resetting each time — basically treating consecutive batches as a continuation of the same sequence.

## Dataset

~1000 tunes from the [Nottingham Music Database](http://abc.sourceforge.net/NMD/), converted to ABC notation (jigs, hornpipes, waltzes, morris, carols, Playford — see `data/`).

## Results

Training loss over 2000 iterations for the included checkpoint:

![Loss vs epochs](https://miro.medium.com/max/960/1*v88bD3oGo95wwofUiEt95A.jpeg)

A few generated samples are in [`samples/`](samples/).

## Testing

```bash
pip install -r requirements-dev.txt
pytest
```

## Possible next steps

- Try a small Transformer instead of/alongside the LSTM and compare
- Check generated ABC is actually valid (e.g. with `abc2abc`) and see how temperature affects the valid-tune rate
- Condition generation on tune type (jig vs waltz vs carol)
- Package it properly with a console-script entry point

## References

1. [Nottingham Music Database](http://abc.sourceforge.net/NMD/)
2. Christopher Olah, [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
3. [Original Colab notebook](https://colab.research.google.com/drive/1OgF1atThRwYwrqe-mFuXOzAi0MOBU8-4)
4. [Challenging Beethoven with Artificial Intelligence](https://python.plainenglish.io/challenging-beethoven-with-artificial-intelligence-927749baf9e5)
