# track-lrc-align

> AI-powered lyrics-to-music alignment tool — produce timestamped LRC files with per-word or per-line timing from plain text lyrics and audio.

[中文文档](README_zh.md)

## Research training handoff

Training is paused at the user's request as of 2026-10-10. The latest state,
results, environment and data recovery commands, and remaining work are in
[the training handoff](docs/TRAINING_HANDOFF.md). P4 training completed with
the best checkpoint at epoch 26; P4 final acceptance and P5/P6 remain open.
The checkpoint contains model weights, not an optimizer resume state.
Run `git lfs pull` after cloning to retrieve archived weights and large reports.
Do not automatically start training or overwrite existing experiment outputs.

## Overview

`track-lrc-align` takes a lyrics text file and an audio file, then generates a synced LRC file. It uses a CNN-RNN acoustic model with DTW (Dynamic Time Warping) alignment to map phonemes to audio frames. Built-in vocal separation via [Demucs](https://github.com/adefossez/demucs) improves alignment accuracy by isolating the singing voice.

**Supported languages**: Chinese (zh), Japanese (ja), Korean (ko), English (en), Russian (ru)

## Installation

```bash
git clone https://github.com/liu-xiaoran/auto-lrc.git
cd auto-lrc

# Git LFS is required for model checkpoints
git lfs install
git lfs pull

pip install -r requirements.txt
```

> Python 3.9 / 3.10 is recommended. The project requires PyTorch and Demucs, so GPU support is recommended for reasonable performance.

Run the lightweight CPU-only regression suite with:

```bash
pip install -r requirements-data.txt  # Optional data reader, required for its tests
python -m pytest -q
```

The tests use synthetic data and do not load checkpoints or download Demucs models.

GitHub Actions runs the same suite on Linux/Python 3.10 using CPU-only PyTorch
wheels and [requirements-ci.txt](requirements-ci.txt). The optional CUDA workflow
is manually dispatched on a prepared GPU runner. See [CI setup and scope](docs/ci.md).

## Quick Start

```bash
python main.py demofile/original_txt.txt demofile/original_track.mp3
```

This runs the full pipeline with default settings: word-level alignment, Demucs vocal separation (mdx_extra ensemble), and prints the enhanced LRC (with per-word timestamps) to terminal. A standard LRC file (line-level timestamps only) is automatically saved to `demofile/original_track.lrc`.

### CLI Options

```
python main.py <lyrics_file> <audio_file> [options]

Options:
  -f, --format     Output format: lrc only (default)
  -l, --line_only  1 = line-level timestamps, 0 = word-level (default)
  -v, --vocalize   1 = separate vocals via Demucs (default), 0 = skip
  -m, --model      Demucs model: mdx | mdx_extra (default) | mdx_q | mdx_extra_q
  -i, --idx        Demucs model index: -1 = ensemble (default), 0-3 = single sub-model
  -o, --out_dir    Standard LRC output directory (default: demofile)
  --aligner       mtl (default) | mtl_bdr (adds line-boundary scoring)
  --json-output PATH          Save raw alignment diagnostics as JSON
  --enhanced-lrc-output PATH  Also save the current word/line LRC output
```

### Examples

```bash
# Word-level with vocal separation (default)
# Prints enhanced LRC to terminal, saves standard LRC to demofile/song.lrc
python main.py lyrics.txt song.mp3

# Line-level, no vocal separation (faster)
python main.py lyrics.txt song.mp3 -l 1 -v 0

# Use a lighter Demucs model
python main.py lyrics.txt song.mp3 -m mdx_q

# Specify a custom output directory for standard LRC
python main.py lyrics.txt song.mp3 -o output

# Compare the BDR baseline and retain unrounded alignment diagnostics
python main.py lyrics.txt song.mp3 -v 0 --aligner mtl_bdr -o output \
  --json-output output/alignment.json --enhanced-lrc-output output/enhanced.lrc
```

## Experimental data preparation

`scripts/fetch_gtsinger_samples.py` and `scripts/prepare_gtsinger.py` support pinned two-sample and 20-segment Chinese/English GTSinger subsets. `scripts/review_gtsinger.py` provides a local blind annotation workspace using original WAV samples. Independent human labels, reviewed corpus splits and accuracy evaluation remain incomplete. See [P1 data preparation](docs/p1-data-preparation.md) for commands and [P0 results](docs/p0-results.md) for completed baseline experiments.

## Python API

```python
from t2l.t2l import process

lrc = process(
    txt_lines,              # list[str]: lyrics lines
    "song.mp3",             # audio file path
    mtl_model="MTL",        # "Baseline" | "MTL" | "Baseline_BDR" | "MTL_BDR"
                            # or a pre-loaded model tuple from init_model
    demucs_model="mdx_extra",
    demucs_idx=-1,
    line_only=False,        # True for line-level, False for word-level
    out_file="output.lrc",  # optional: write to file
    verbose=True,
    vocalize=True,          # set False to skip Demucs and use raw audio
    format="lrc"
)
print(lrc)
```

The `format` argument currently accepts only `"lrc"`; unsupported values raise `ValueError`. When alignment runs directly on multichannel audio (`vocalize=False`), the first channel is used for compatibility with earlier output while avoiding channel concatenation.

Pass `return_details=True` to `process()` to receive a dictionary with `lrc`, `units`, and `alignment`. Units include text, line/word indices, phonemes, raw frame indices and unrounded seconds; missing alignments use `null`. The default return value remains an LRC string. The global path score is uncalibrated, and the nominal 34.83 ms frame grid has not been calibrated against physical singing onsets. See the [baseline methodology](docs/p0-methodology.md) for reproducible experiments and limitations.

To avoid repeated model loading across multiple calls, use `t2l.init_model`:

```python
from t2l.t2l import process
from t2l.init_model import load_mtl_model

model = load_mtl_model("MTL")  # load once
lrc1 = process(lines1, "song1.mp3", mtl_model=model)
lrc2 = process(lines2, "song2.mp3", mtl_model=model)
```

## How It Works

```
lyrics text + audio file
  │
  ├─ phonetic.py
  │   Language detection: regex for CJK/Cyrillic → fastText fallback (lid.176.ftz)
  │   Phonemization: convert characters to ASCII phonemes per language
  │     zh → pypinyin
  │     ja → pykakasi
  │     ko → kroman
  │     ru → cyrtranslit
  │     en → g2p_en
  │
  ├─ t2l.py
  │   Audio loading: torchaudio.load → librosa.load (fallback)
  │   Vocal separation: Demucs (mdx_extra ensemble, 4 sub-models)
  │   Resample to 22050 Hz
  │
  ├─ mtl/wrapper.py
  │   Mel-spectrogram (128 bins, FFT 512)
  │   CNN-RNN acoustic model → phoneme posteriorgram
  │   DTW alignment (utils.alignment / alignment_bdr)
  │
  └─ gen_lrc()
      Frame indices → [MM:SS.mmm] timestamps
      Output LRC with per-word <timestamp> tags
```

### Model Checkpoints

Stored in `./checkpoints/`, managed via Git LFS:

| Checkpoint | Type | Description |
|---|---|---|
| `checkpoint_Baseline` | Single-task | Acoustic model, 41 phoneme classes |
| `checkpoint_MTL` | Multi-task | Default model, classes (41, 47) |
| `checkpoint_BDR` | Boundary detection | 1.8 MB, used with `*_BDR` variants |

The `method` parameter in the alignment pipeline accepts `"Baseline"`, `"MTL"`, `"Baseline_BDR"`, or `"MTL_BDR"`. BDR variants add a learned boundary score at lyric-line starts; accuracy gains require evaluation against reference annotations.

### Key Parameters

| Parameter | Value | Notes |
|---|---|---|
| Sample rate | 22050 Hz | Target rate after resampling |
| Frame resolution | ~0.0348s | `256 / 22050 * 3` seconds per frame |
| Mel features | 128 | Used in `train_audio_transforms` |
| FFT size | 512 | Spectrogram computation |

## Output

The CLI produces two outputs:

1. **Enhanced LRC** — printed to terminal, with per-word `<mm:ss.xxx>` timestamps for detailed alignment
2. **Standard LRC** — saved to `demofile/` (or directory specified by `-o`), with only line-level `[mm:ss.xxx]` timestamps, compatible with standard LRC players

The standard LRC filename is derived from the audio filename, e.g. `song.mp3` → `demofile/song.lrc`.

## Output Format

### Word-level LRC

```lrc
[00:00.120]<00:00.120>Hel<00:00.240>lo<00:00.360> <00:00.480>World
[00:01.000]<00:01.000>这<00:01.120>是<00:01.240>一<00:01.360>首歌
```

Each word is prefixed with its start timestamp in angle brackets, and each line begins with a line-level timestamp in square brackets.

### Line-level LRC

```lrc
[00:00.120]Hello World
[00:01.000]这是一首歌
```

Only line-level timestamps are included.

## Requirements

Core dependencies (full list in `requirements.txt`):

- **PyTorch / torchaudio** — audio processing and model inference
- **Demucs** — vocal/instrumental separation
- **librosa** — audio loading fallback
- **fastText** — language detection for non-CJK text
- **pypinyin / pykakasi / kroman / cyrtranslit / g2p-en** — phonemization per language

> `pykakasi` (Japanese romanization) is GPL-licensed. Be aware of this if you plan to use this project under a different license.

## License

See [LICENSE](LICENSE) for details.
