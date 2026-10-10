#!/usr/bin/env python3
"""Compare original and constrained coarse candidates against dataset annotations."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import unicodedata
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text_key(text):
    return ''.join(c for c in unicodedata.normalize('NFKC', text).casefold() if c.isalnum())


def source_errors(segment, candidates):
    """Compare exact ordered text only; mismatch/missing must not disappear."""
    units = [(i, u) for i, u in enumerate(segment['units']) if u['kind'] == 'lyric']
    result = {'total': len(units), 'missing': 0, 'text_mismatch': 0, 'errors': []}
    if len(candidates) != len(units):
        result.update(text_mismatch=len(units), extra_or_missing_units=len(candidates) - len(units))
        return result
    for (index, unit), candidate in zip(units, candidates):
        if not text_key(unit['text']) or text_key(unit['text']) != text_key(candidate['text']):
            result['text_mismatch'] += 1
            continue
        predicted = candidate['start_seconds']
        if predicted is None or candidate.get('status') == 'unaligned':
            result['missing'] += 1
            continue
        if isinstance(predicted, bool) or not isinstance(predicted, (float, int)) or not math.isfinite(predicted):
            raise ValueError('Nonfinite or invalid prediction')
        boundary = unit['boundary']
        if boundary['source'] != 'textgrid:first_phone':
            raise ValueError('Expected explicit TextGrid annotation source')
        reference = boundary['candidate_sample'] / segment['sample_rate']
        error = (predicted - reference) * 1000
        result['errors'].append({'unit_index': index, 'text': unit['text'],
                                 'reference_sample': boundary['candidate_sample'],
                                 'reference_seconds': reference, 'predicted_seconds': predicted,
                                 'signed_error_ms': error, 'absolute_error_ms': abs(error)})
    return result


def summarize(rows):
    values = sorted(e['absolute_error_ms'] for r in rows for e in r['errors'])
    def quantile(fraction):
        if not values:
            return None
        pos = fraction * (len(values) - 1)
        lo, hi = math.floor(pos), math.ceil(pos)
        return values[lo] + (values[hi] - values[lo]) * (pos - lo)
    total = sum(r['total'] for r in rows)
    return {'reference': 'published_dataset_annotation', 'independent_human_accuracy': False,
            'total': total, 'evaluated': len(values),
            'missing': sum(r['missing'] for r in rows),
            'text_mismatch': sum(r['text_mismatch'] for r in rows),
            'mae_ms': sum(values) / len(values) if values else None,
            'median_ae_ms': quantile(.5), 'p90_ms': quantile(.9), 'p95_ms': quantile(.95),
            'coverage_of_evaluated': {str(ms): sum(v <= ms for v in values) / len(values) if values else None
                                      for ms in (5, 10, 20, 50)},
            'coverage_of_all': {str(ms): sum(v <= ms for v in values) / total if total else None
                                for ms in (5, 10, 20, 50)},
            'severe_over_500ms': sum(v > 500 for v in values)}


def windows_from_legacy(candidates, frame_count, radius_frames):
    """Use model-only occupancy bands; never annotation timestamps."""
    windows = []
    onsets = []
    for candidate in candidates:
        start, end = candidate['raw_frames']['start'], candidate['raw_frames']['end']
        if start is None or end is None:
            raise ValueError('Legacy path incomplete: cannot construct local bands')
        windows.append([max(0, start - radius_frames), min(frame_count, max(end, start + 1) + radius_frames)])
        onsets.append([max(0, start - radius_frames), min(frame_count, start + radius_frames + 1)])
    return {'mode': 'constrained', 'word_windows': windows, 'onset_windows': onsets,
            'fallback_to_legacy': True}


def load_inputs(segments_path, splits_path, audio_root):
    segments = [json.loads(line) for line in Path(segments_path).read_text().splitlines() if line.strip()]
    split = json.loads(Path(splits_path).read_text())
    by_id = {r['segment_id']: r for r in split['assignments']}
    if len(by_id) != len(split['assignments']) or len({s['segment_id'] for s in segments}) != len(segments):
        raise ValueError('Duplicate segment identity')
    if set(by_id) != {s['segment_id'] for s in segments}:
        raise ValueError('Split coverage must exactly match segments')
    songs, audios = defaultdict(set), defaultdict(set)
    validated = []
    for segment in segments:
        row = by_id[segment['segment_id']]
        if row['split'] not in ('train', 'val', 'test'):
            raise ValueError('Unassigned split')
        if split.get('identity_mode') == 'dataset' and row['canonical_song_id'] != segment['song_id']:
            raise ValueError('Dataset song identity changed')
        if segment.get('singer') in split.get('held_out_singers', []) and row['split'] != 'test':
            raise ValueError('Held-out singer appears outside test')
        songs[row['canonical_song_id']].add(row['split'])
        audios[segment['sources']['wav']['sha256']].add(row['split'])
        relative = Path(segment['segment_id'] + '.wav')
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Invalid dataset relative path')
        audio = (Path(audio_root) / relative).resolve()
        audio.relative_to(Path(audio_root).resolve())
        if digest(audio) != segment['sources']['wav']['sha256']:
            raise ValueError('WAV differs from frozen source: ' + str(audio))
        with wave.open(str(audio), 'rb') as wav:
            if (wav.getframerate(), wav.getnframes()) != (segment['sample_rate'], segment['num_samples']):
                raise ValueError('Original sample grid changed')
        segment = dict(segment, audio_path=str(audio), split=row['split'])
        validated.append(segment)
    if any(len(v) != 1 for v in list(songs.values()) + list(audios.values())):
        raise ValueError('Song or original audio crosses splits')
    return validated


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--segments', required=True, type=Path)
    parser.add_argument('--splits', required=True, type=Path)
    parser.add_argument('--audio-root', required=True, type=Path)
    parser.add_argument('--out-dir', required=True, type=Path)
    parser.add_argument('--band-radius-seconds', type=float, default=.35)
    args = parser.parse_args(argv)
    if not math.isfinite(args.band_radius_seconds) or args.band_radius_seconds <= 0:
        parser.error('band radius must be finite and positive')
    segments = list(load_inputs(args.segments, args.splits, args.audio_root))
    args.out_dir.mkdir(parents=True, exist_ok=False)
    report = {'schema_version': 1, 'status': 'running', 'started_at': datetime.now(timezone.utc).isoformat(),
              'reference': 'published_dataset_annotation', 'independent_human_accuracy': False,
              'seed': 7, 'band_radius_seconds': args.band_radius_seconds, 'rows': [],
              'input_sha256': {str(p): digest(p) for p in (args.segments, args.splits)},
              'limitations': ['No human boundary or source-alias verification.',
                             'Small development subset; split language coverage is incomplete.',
                             'No parameters are selected from test errors; fixed radius chosen before evaluation.',
                             'Confidence remains uncalibrated; no audio time shift is applied.']}
    target = args.out_dir / 'result.json'
    def save():
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    try:
        import torch
        from t2l.alignment.coarse import coarse_align
        from t2l.mtl.wrapper import load_mtl_model, FRAME_RESOLUTION_SECONDS
        from scripts.benchmark_gtsinger_mixtures import measured_cuda_call, cuda_forward_observations
        if not torch.cuda.is_available():
            raise RuntimeError('This benchmark requires the configured CUDA environment')
        torch.set_num_threads(2)
        torch.cuda.set_device(0)
        report['environment'] = {'python': sys.version, 'torch': torch.__version__,
                                  'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0)}
        report['runtime_sha256'] = {str(p.relative_to(ROOT)): digest(p) for p in sorted(
            list((ROOT / 't2l/alignment').glob('*.py')) + [Path(__file__), ROOT / 't2l/t2l.py',
                 ROOT / 't2l/phonetic.py', ROOT / 't2l/mtl/wrapper.py'])}
        report['checkpoints'] = {name: digest(ROOT / 'checkpoints' / name)
                                  for name in ('checkpoint_MTL', 'checkpoint_BDR')}
        models = load_mtl_model('MTL_BDR', cuda=True, verbose=False)
        variants = {'mtl': (models[0], None, models[2], False, models[4]), 'mtl_bdr': models}
        buckets = defaultdict(list)
        for segment in segments:
            language = {'Chinese': 'zh', 'English': 'en'}[segment['language']]
            words = [u['text'] for u in segment['units'] if u['kind'] == 'lyric']
            lyrics = [''.join(words) if language == 'zh' else ' '.join(words)]
            row = {'segment_id': segment['segment_id'], 'song_id': segment['song_id'],
                   'split': segment['split'], 'language': language, 'runs': {}}
            report['rows'].append(row)
            for method, model in variants.items():
                constraints = None
                for decoder in ('legacy', 'constrained'):
                    run = {'observed_forwards': [], 'constraints_input': constraints}
                    row['runs'][method + ':' + decoder] = run
                    roles = [('acoustic', model[0])] + ([('boundary', model[1])] if model[3] else [])
                    def invoke():
                        with cuda_forward_observations(torch, roles, run['observed_forwards']):
                            return [c.to_dict() for c in coarse_align(segment['audio_path'], lyrics,
                                    method=method, model=model, language=language, constraints=constraints)]
                    run['candidates'] = measured_cuda_call(torch, run, invoke)
                    run['source_metrics'] = source_errors(segment, run['candidates'])
                    buckets[(method, decoder, segment['split'])].append(run['source_metrics'])
                    if decoder == 'legacy':
                        # Existing centered STFT then pooling: floor((floor(N/256)+1)/3).
                        n = segment['num_samples'] * 22050 // segment['sample_rate']
                        frames = (n // 256 + 1) // 3
                        constraints = windows_from_legacy(run['candidates'], frames,
                                     max(1, math.ceil(args.band_radius_seconds / FRAME_RESOLUTION_SECONDS)))
                    save()
        report['summary'] = {':'.join(k): summarize(v) for k, v in buckets.items()}
        report['status'] = 'completed'
        save()
    except Exception as error:
        report.update(status='failed', error={'type': type(error).__name__, 'message': str(error)})
        save()
        raise
    print(target)
    return 0


if __name__ == '__main__':
    raise SystemExit(cli())
