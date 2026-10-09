#!/usr/bin/env python3
"""Make locally generated instrumental mixtures without changing sample positions."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_baseline import file_provenance
from scripts.prepare_gtsinger import publish_directory

PCM_SCALE = 2 ** 23


def read_pcm24_mono(path):
    with wave.open(str(path), "rb") as audio:
        if (audio.getcomptype() != "NONE" or audio.getsampwidth() != 3
                or audio.getnchannels() != 1 or audio.getframerate() != 48000
                or audio.getnframes() <= 0):
            raise ValueError(str(path) + ": expected non-empty mono 24-bit 48kHz PCM WAV")
        count = audio.getnframes()
        raw = audio.readframes(count)
    if len(raw) != count * 3:
        raise ValueError(str(path) + ": truncated PCM data")
    octets = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
    integers = octets[:, 0] | (octets[:, 1] << 8) | (octets[:, 2] << 16)
    integers = (integers ^ PCM_SCALE) - PCM_SCALE
    return integers.astype(np.float64) / PCM_SCALE


def write_pcm24_mono(path, samples):
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 1 or not len(samples) or not np.isfinite(samples).all():
        raise ValueError("Expected a finite non-empty mono signal")
    integers = np.rint(samples * PCM_SCALE).astype(np.int64)
    if np.any(integers < -PCM_SCALE) or np.any(integers >= PCM_SCALE):
        raise ValueError("Signal would clip in 24-bit PCM")
    packed = np.column_stack([integers & 255, (integers >> 8) & 255,
                              (integers >> 16) & 255]).astype(np.uint8).tobytes()
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(3)
        audio.setframerate(48000)
        audio.writeframes(packed)


def synthesize_accompaniment(num_samples, seed):
    """Original fixed instrumental pattern; independent of text/source boundaries."""
    if isinstance(num_samples, bool) or not isinstance(num_samples, int) or num_samples < 1:
        raise ValueError("num_samples must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2 ** 32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    rng = np.random.RandomState(seed)
    output = np.zeros(num_samples, dtype=np.float64)
    beat_samples = 30000  # 96 BPM at 48kHz, no rounding of beat positions.
    roots = (48, 53, 57, 55)
    for beat, start in enumerate(range(0, num_samples, beat_samples)):
        stop = min(start + beat_samples, num_samples)
        time = np.arange(stop - start, dtype=np.float64) / 48000
        root = roots[(beat // 4) % len(roots)]
        for semitones in (0, 4, 7):
            frequency = 440 * 2 ** ((root + semitones - 69) / 12)
            attack = np.minimum(time / 0.008, 1)
            output[start:stop] += 0.16 * attack * np.exp(-time * 4) * np.sin(2 * np.pi * frequency * time)
        bass = 440 * 2 ** ((root - 12 - 69) / 12)
        output[start:stop] += 0.2 * np.exp(-time * 5) * np.sin(2 * np.pi * bass * time)
        output[start:stop] += 0.3 * np.exp(-time * 25) * np.sin(2 * np.pi * (45 * time + 30 * (1 - np.exp(-time * 20)) / 20))
        output[start:stop] += 0.035 * np.exp(-time * 60) * rng.standard_normal(len(time))
    return output


def mix_same_timeline(source, accompaniment, snr_db=6.0):
    source = np.asarray(source, dtype=np.float64)
    accompaniment = np.asarray(accompaniment, dtype=np.float64)
    if (source.ndim != 1 or not len(source) or source.shape != accompaniment.shape
            or not np.isfinite(source).all() or not np.isfinite(accompaniment).all()):
        raise ValueError("Source and accompaniment must have the same finite mono sample grid")
    if isinstance(snr_db, bool) or not math.isfinite(snr_db) or not -20 <= snr_db <= 40:
        raise ValueError("snr_db must be finite and between -20 and 40")
    source_rms = float(np.sqrt(np.mean(source ** 2)))
    accompaniment_rms = float(np.sqrt(np.mean(accompaniment ** 2)))
    if source_rms <= 0 or accompaniment_rms <= 0:
        raise ValueError("Cannot define SNR for a silent source or accompaniment")
    accompaniment_gain = source_rms / accompaniment_rms * 10 ** (-snr_db / 20)
    scaled_backing = accompaniment * accompaniment_gain
    combined = source + scaled_backing
    peak = max(float(np.max(np.abs(combined))), float(np.max(np.abs(source))),
               float(np.max(np.abs(scaled_backing))))
    mixture_gain = min(1.0, 0.9 / peak)
    return combined * mixture_gain, accompaniment * accompaniment_gain * mixture_gain, {
        "snr_db": snr_db, "snr_definition": "Full-clip uncentered RMS of vocal and accompaniment components",
        "mixture_gain": mixture_gain, "accompaniment_gain": accompaniment_gain,
        "timeline": {"source_to_mixture_offset_samples": 0, "sample_rate": 48000,
                     "num_samples": len(source), "resampled": False, "filtered": False,
                     "padded": False, "cropped": False},
    }


def prepare_mixtures(segments_path, out_dir, snr_db=6.0, seed=7):
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2 ** 32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    segments_path = Path(segments_path).resolve()
    out_dir = Path(out_dir).absolute()
    if out_dir.exists() or out_dir.is_symlink():
        raise FileExistsError("Mixture directory already exists: " + str(out_dir))
    segments = [json.loads(line) for line in segments_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not segments or len({s["segment_id"] for s in segments}) != len(segments):
        raise ValueError("Expected non-empty segments with distinct IDs")
    source_manifest = segments_path.parent / "source-manifest.json"
    dataset = json.loads(source_manifest.read_text(encoding="utf-8"))
    if not dataset.get("license") or not dataset.get("revision"):
        raise ValueError("Prepared source manifest must retain its license and dataset revision")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".gtsinger-mixture-", dir=out_dir.parent))
    rows = []
    try:
        for segment in segments:
            segment_id = segment["segment_id"]
            source_path = Path(segment["audio_path"])
            source_info = file_provenance(source_path)
            if source_info["sha256"] != segment["sources"]["wav"]["sha256"]:
                raise ValueError("Source audio checksum mismatch: " + str(source_path))
            source = read_pcm24_mono(source_path)
            if segment["sample_rate"] != 48000 or segment["num_samples"] != len(source):
                raise ValueError("Source sample grid differs from normalized segment: " + segment_id)
            stem = hashlib.sha256(segment_id.encode("utf-8")).hexdigest()
            segment_seed = (seed + int(stem[:8], 16)) % 2 ** 32
            accompaniment = synthesize_accompaniment(len(source), segment_seed)
            mixture, backing, mixing = mix_same_timeline(source, accompaniment, snr_db)
            files = {"mixture_audio": stem + "-mixture.wav", "accompaniment_audio": stem + "-backing.wav",
                     "lyrics": stem + "-lyrics.txt"}
            write_pcm24_mono(staging / files["mixture_audio"], mixture)
            write_pcm24_mono(staging / files["accompaniment_audio"], backing)
            words = [u["text"] for u in segment["units"] if u["kind"] == "lyric"]
            if not words or not all(isinstance(word, str) and word.strip() for word in words):
                raise ValueError("No valid lyric units: " + segment_id)
            separator = "" if segment["language"] in ("zh", "Chinese") else " "
            (staging / files["lyrics"]).write_text(separator.join(words) + "\n", encoding="utf-8")
            decoded_mix = read_pcm24_mono(staging / files["mixture_audio"])
            decoded_backing = read_pcm24_mono(staging / files["accompaniment_audio"])
            residual = decoded_mix - decoded_backing - source * mixing["mixture_gain"]
            error = float(np.max(np.abs(residual)))
            if error > 1 / PCM_SCALE + 1e-14:
                raise ValueError("Decoded mix does not preserve the source grid within PCM quantization")
            decoded_backing_rms = float(np.sqrt(np.mean(decoded_backing ** 2)))
            if decoded_backing_rms <= 0:
                raise ValueError("Accompaniment became silent after PCM quantization")
            measured_snr = 20 * math.log10(float(np.sqrt(np.mean((source * mixing["mixture_gain"]) ** 2)))
                                         / decoded_backing_rms)
            if file_provenance(source_path) != source_info:
                raise ValueError("Source audio changed during mixing: " + str(source_path))
            generated = {}
            for name, filename in files.items():
                generated[name] = dict(file_provenance(staging / filename), path=str(out_dir / filename))
            rows.append({"segment_id": segment_id, "song_id": segment["song_id"],
                         "derived_from_segment_id": segment_id, "split": segment.get("split", "unassigned"),
                         "source_audio": source_info, **generated, **mixing,
                         "sample_rate": 48000, "num_samples": len(source), "channels": 1,
                         "generator_seed": segment_seed,
                         "checks": {"source_unchanged": True, "decoded_num_samples_unchanged": True,
                                    "linear_mix_max_abs_residual": error, "quantization_bound": 1 / PCM_SCALE,
                                    "quantization_bound_domain": "Decoded mixture minus backing compared to gain-scaled source",
                                    "reconstructed_source_max_abs_residual": error / mixing["mixture_gain"],
                                    "reconstructed_source_quantization_bound": 1 / PCM_SCALE / mixing["mixture_gain"],
                                    "decoded_component_snr_db": measured_snr},
                         "boundary_labels": "Inherited sample coordinates only; no new human labels"})
        result = {"schema_version": "gtsinger-mixtures-v1",
                  "created_at": datetime.now(timezone.utc).isoformat(),
                  "segments_source": file_provenance(segments_path),
                  "source_manifest": file_provenance(source_manifest), "source_license": dataset["license"],
                  "dataset_revision": dataset["revision"],
                  "generator": {"source": file_provenance(Path(__file__)),
                                "dependencies": [file_provenance(ROOT / "scripts" / name) for name in
                                                 ("prepare_gtsinger.py", "benchmark_baseline.py")],
                                "kind": "Original additive instrumental synthesis; no third-party recordings or composition",
                                "accompaniment_license": "CC0-1.0", "bpm": 96,
                                "limitation": "Synthetic backing is an engineering fixture, not realistic matched accompaniment or a licensed source for original vocals. Mixed vocals retain the dataset terms."},
                  "config": {"seed": seed, "snr_db": snr_db, "numpy": np.__version__,
                             "sample_rate": 48000, "pcm_bits": 24},
                  "accuracy": {"available": False, "reason": "Independent human labels remain pending"},
                  "split_policy": "Every mixture inherits its source song/segment split; no independent split assignment",
                  "rows": rows}
        (staging / "mixtures.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        (staging / "ACCOMPANIMENT-LICENSE.txt").write_text(
            "The standalone instrumental signals generated by this script are newly synthesized, "
            "contain no third-party recordings or copied compositions, and are dedicated under CC0 1.0.\n"
            "https://creativecommons.org/publicdomain/zero/1.0/\n"
            "This dedication does not apply to GTSinger vocals or their mixtures. Those retain "
            "the complete original dataset license, additional terms and attribution obligations.\n", encoding="utf-8")
        publish_directory(staging, out_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True, help="New local directory; do not redistribute dataset audio")
    parser.add_argument("--snr-db", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    if not 0 <= args.seed < 2 ** 32:
        parser.error("--seed must be in [0, 2**32)")
    try:
        result = prepare_mixtures(args.segments, args.out_dir, args.snr_db, args.seed)
        print(json.dumps({"out_dir": str(args.out_dir), "segments": len(result["rows"]),
                          "accuracy_available": False}, ensure_ascii=False))
    except (OSError, ValueError, KeyError, wave.Error) as error:
        parser.exit(1, "GTSinger mixing failed: %s\n" % error)


if __name__ == "__main__":
    main()
