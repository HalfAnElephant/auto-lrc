#!/usr/bin/env python3
"""Reuse frozen clean/mixed/separated inference for a fixed DSP diagnostic."""
import argparse
from collections import defaultdict
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.benchmark_coarse_gtsinger import digest, load_inputs, source_errors, summarize
from scripts.benchmark_dsp_gtsinger import runtime_digests, verify_selection
from t2l.alignment.coarse import candidates_from_details
from t2l.boundary.dsp import refine_boundaries
from t2l.boundary.features import extract_features


def reused_candidates(run, audio_rate, audio_samples, lyrics, language):
    if run.get('status') != 'completed':
        raise ValueError('Cannot reuse incomplete mixture inference')
    details = copy.deepcopy(run['result'])
    details['original_audio'] = {'sample_rate': audio_rate, 'num_samples': audio_samples}
    return [candidate.to_dict() for candidate in candidates_from_details(
        details, lyrics, method=details['alignment']['method'], language=language)]


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('segments', 'splits', 'audio-root', 'coarse-report', 'mixture-report', 'selection-file', 'out-dir'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args(argv)
    inputs = {'segments': digest(args.segments), 'splits': digest(args.splits),
              'coarse_report': digest(args.coarse_report)}
    selection = json.loads(args.selection_file.read_text())
    config = verify_selection(selection, inputs, runtime_digests())
    segments = {s['segment_id']: s for s in load_inputs(args.segments, args.splits, args.audio_root)}
    previous = json.loads(args.mixture_report.read_text())
    if previous.get('status') != 'completed':
        parser.error('Mixture report must be completed')
    args.out_dir.mkdir(parents=True, exist_ok=False)
    report = {'schema_version': 1, 'status': 'running', 'started_at': datetime.now(timezone.utc).isoformat(),
              'reference': 'published_dataset_annotation', 'independent_human_accuracy': False,
              'selection_sha256': digest(args.selection_file), 'config': config.to_dict(),
              'input_sha256': dict(inputs, mixture_report=digest(args.mixture_report)),
              'runtime_sha256': dict(runtime_digests(), benchmark=digest(Path(__file__))),
              'rows': [], 'limitations': ['Two training-song diagnostics, not an independent test.',
                  'Frozen validation DSP configuration; no mixture-specific tuning.',
                  'Synthetic accompaniment does not cover natural arrangements.',
                  'No time shift or resampling; native separated sample grid retained.',
                  'Prior model/separation inference reused, not rerun.']}
    target = args.out_dir / 'result.json'
    def save():
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    wall = time.perf_counter()
    buckets = defaultdict(list)
    try:
        import numpy as np
        import soundfile as sf
        report['environment'] = {'python': sys.version, 'numpy': np.__version__, 'soundfile': sf.__version__, 'device': 'cpu'}
        for old in previous['rows']:
            segment = segments[old['segment_id']]
            if old['song_id'] != segment['song_id']:
                raise ValueError('Mixture song identity differs from frozen split')
            language = {'Chinese': 'zh', 'English': 'en'}[segment['language']]
            words = [u['text'] for u in segment['units'] if u['kind'] == 'lyric']
            lyrics = [''.join(words) if language == 'zh' else ' '.join(words)]
            row = {'segment_id': segment['segment_id'], 'song_id': segment['song_id'],
                   'split': segment['split'], 'variants': {}}
            report['rows'].append(row)
            for variant in ('clean', 'mixture', 'separated'):
                runs = old['predictions'][variant]
                identity = runs['MTL']['audio']
                path = Path(identity['path'])
                if digest(path) != identity['sha256']:
                    raise ValueError('Frozen audio changed: ' + str(path))
                audio, rate = sf.read(path, dtype='float64', always_2d=True)
                if not np.isfinite(audio).all() or not len(audio):
                    raise ValueError('Invalid decoded audio')
                audio = audio.T
                features = extract_features(audio, rate, hop_samples=config.hop_samples)
                result = {'audio_sha256': identity['sha256'], 'sample_rate': rate,
                          'num_samples': audio.shape[1], 'feature_metadata': features['metadata'], 'runs': {}}
                row['variants'][variant] = result
                for name, run in runs.items():
                    if run['audio']['sha256'] != identity['sha256']:
                        raise ValueError('Models used different variant audio')
                    candidates = reused_candidates(run, rate, audio.shape[1], lyrics, language)
                    start = time.perf_counter()
                    refined = refine_boundaries(audio, rate, candidates, config, features)
                    result['runs'][name] = {'runtime_seconds': time.perf_counter() - start,
                                           'coarse': {'candidates': candidates, 'metrics': source_errors(segment, candidates)},
                                           'dsp': {'candidates': refined, 'metrics': source_errors(segment, refined)}}
                    for method in ('coarse', 'dsp'):
                        buckets[(variant, name, method)].append(result['runs'][name][method]['metrics'])
                save()
        report.update(status='completed', summary={':'.join(key): summarize(rows) for key, rows in buckets.items()})
    except Exception as error:
        report.update(status='failed', error={'type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        report['runtime_seconds'] = time.perf_counter() - wall
        save()
    print(target)
    return 0


if __name__ == '__main__':
    raise SystemExit(cli())
