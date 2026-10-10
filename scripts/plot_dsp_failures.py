#!/usr/bin/env python3
"""Render source-error distributions and waveform/spectrum failure diagnostics."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_coarse_gtsinger import digest, load_inputs
from scripts.benchmark_dsp_gtsinger import read_audio


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('segments', 'splits', 'audio-root', 'result', 'out-dir'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--max-failures', type=int, default=6)
    args = parser.parse_args(argv)
    if not 1 <= args.max_failures <= 20:
        parser.error('max-failures must be between 1 and 20')
    report = json.loads(args.result.read_text())
    if report.get('status') != 'completed' or report.get('stage') != 'evaluate':
        parser.error('Expected completed DSP evaluate report')
    for name, path in (('segments', args.segments), ('splits', args.splits)):
        if report['input_sha256'][name] != digest(path):
            parser.error('Input differs from evaluated ' + name)
    segments = {s['segment_id']: s for s in load_inputs(args.segments, args.splits, args.audio_root)}
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.signal import spectrogram
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    args.out_dir.mkdir(parents=True, exist_ok=False)
    colors = {'coarse': '#335d8c', 'dsp_trial': '#d14e3a'}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    test = [r for r in report['rows'] if r['split'] == 'test']
    for variant, label in (('coarse', 'MTL coarse'), ('dsp_trial', 'MTL + fixed DSP trial')):
        errors = [e['absolute_error_ms'] for r in test for e in r['runs']['mtl'][variant]['source_metrics']['errors']]
        signed = [e['signed_error_ms'] for r in test for e in r['runs']['mtl'][variant]['source_metrics']['errors']]
        ordered = np.sort(errors)
        axes[0].step(ordered, np.arange(1, len(ordered) + 1) / len(ordered), where='post',
                     label=label, color=colors[variant])
        axes[1].hist(signed, bins=np.linspace(-600, 600, 41), alpha=.5, label=label, color=colors[variant])
    axes[0].set(xlabel='Absolute source annotation error (ms)', ylabel='Fraction', xlim=(0, 500), ylim=(0, 1))
    axes[1].set(xlabel='Signed error: prediction - source (ms)', ylabel='Boundary count')
    for ax in axes:
        ax.grid(alpha=.15)
        ax.legend(fontsize=9)
    fig.suptitle('English-only development test: published annotation agreement, not human accuracy')
    fig.savefig(args.out_dir / 'test-error-distribution.png', dpi=160)
    plt.close(fig)
    cases = []
    for row in test:
        run = row['runs']['mtl']['dsp_trial']
        for error in run['source_metrics']['errors']:
            lyric_index = [i for i, u in enumerate(segments[row['segment_id']]['units']) if u['kind'] == 'lyric'].index(error['unit_index'])
            cases.append((error['absolute_error_ms'], row, error, lyric_index))
    entries = []
    for index, (_, row, error, unit_index) in enumerate(sorted(cases, key=lambda c: -c[0])[:args.max_failures], 1):
        segment = segments[row['segment_id']]
        audio = read_audio(segment)
        rate = segment['sample_rate']
        coarse = row['runs']['mtl']['coarse']['candidates'][unit_index]
        refined = row['runs']['mtl']['dsp_trial']['candidates'][unit_index]
        reference, start, prediction = error['reference_seconds'], coarse['start_seconds'], refined['start_seconds']
        lo, hi = max(0, min(reference, start, prediction) - .16), min(len(audio) / rate, max(reference, start, prediction) + .22)
        lower, upper = int(lo * rate), int(hi * rate)
        clip = audio[lower:upper]
        fig, axes = plt.subplots(3, 1, figsize=(10, 6), constrained_layout=True,
                                 gridspec_kw={'height_ratios': [1, 2, .8]}, sharex=True)
        time = np.arange(lower, upper) / rate
        axes[0].plot(time, clip, color='#29333d', linewidth=.5)
        axes[0].set_ylabel('Amplitude')
        f, t, spectrum = spectrogram(clip, rate, nperseg=min(960, len(clip)), noverlap=min(840, max(0, len(clip) - 1)))
        axes[1].pcolormesh(t + lower / rate, f / 1000, 10 * np.log10(np.maximum(spectrum, 1e-12)),
                           shading='auto', cmap='magma', vmin=-100, vmax=-35)
        axes[1].set(ylim=(0, min(8, rate / 2000)), ylabel='Frequency (kHz)')
        alternatives = refined['refinement'].get('candidates', [])
        if alternatives:
            axes[2].scatter([c['seconds'] for c in alternatives], [c['score'] for c in alternatives],
                            color=colors['dsp_trial'], s=18)
        axes[2].set(xlabel='Original decoded audio time (s)', ylabel='Local score', xlim=(lo, hi))
        for ax in axes:
            ax.axvline(reference, color='#218b62', linewidth=1.5, label='Published source annotation')
            ax.axvline(start, color=colors['coarse'], linestyle='--', linewidth=1.2, label='Coarse')
            ax.axvline(prediction, color=colors['dsp_trial'], linestyle=':', linewidth=1.5, label='DSP trial')
        axes[0].legend(loc='upper right', fontsize=8)
        detail = refined['refinement']
        fig.suptitle('Case {} | word: {} | source AE {:.1f} ms | {} | confidence: uncalibrated'.format(
            index, error['text'], error['absolute_error_ms'], detail['status']))
        name = 'failure-{:02d}.png'.format(index)
        fig.savefig(args.out_dir / name, dpi=160)
        plt.close(fig)
        entries.append({'file': name, 'segment_id': row['segment_id'], 'song_id': row['song_id'],
                        'split': row['split'], 'unit_index': error['unit_index'], 'text': error['text'],
                        'source_annotation_sample': error['reference_sample'],
                        'coarse_sample': coarse['start_sample'], 'dsp_sample': refined['start_sample'],
                        'absolute_error_ms': error['absolute_error_ms'], 'confidence': None,
                        'refinement_status': detail['status'], 'reason': detail['reason']})
    metadata = {'created_at': datetime.now(timezone.utc).isoformat(), 'result_sha256': digest(args.result),
                'script_sha256': digest(Path(__file__)), 'reference': 'published_dataset_annotation',
                'independent_human_accuracy': False, 'failures': entries,
                'images': {p.name: digest(p) for p in args.out_dir.glob('*.png')}}
    (args.out_dir / 'index.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    print(args.out_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(cli())
