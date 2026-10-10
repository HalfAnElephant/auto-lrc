#!/usr/bin/env python3
"""Select DSP on validation annotations, then evaluate one frozen trial on CPU."""
import argparse
from collections import Counter, defaultdict
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import time
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.benchmark_coarse_gtsinger import digest, load_inputs, source_errors, summarize
from t2l.boundary.dsp import DSPConfig, refine_boundaries
from t2l.boundary.features import extract_features

METHODS = ('mtl', 'mtl_bdr')
CONFIG_GRID = (
    DSPConfig(),
    DSPConfig(name='low_prior', prior_weight=.1, min_evidence=.25),
    DSPConfig(name='high_prior', prior_weight=.5, min_evidence=.5),
)
POLICY = {
    'selection_split': 'val', 'metric': 'pooled_mae_ms',
    'pooling': 'mtl:legacy and mtl_bdr:legacy; one configuration for both',
    'tie_break': 'declared_grid_order', 'strict_mae_improvement': True,
    'evaluated_must_not_decrease': True,
    'coverage_of_all_must_not_decrease_ms': [5, 10, 20, 50],
    'severe_over_500ms_must_not_increase': True,
    'reference': 'published_dataset_annotation', 'independent_human_accuracy': False,
}


def runtime_digests():
    return {name: digest(ROOT / name) for name in
            ('t2l/boundary/dsp.py', 't2l/boundary/features.py')}


def read_audio(segment):
    """Decode original integer PCM without changing its sample grid or channel policy."""
    path = Path(segment['audio_path'])
    if digest(path) != segment['sources']['wav']['sha256']:
        raise ValueError('WAV differs from frozen source: ' + str(path))
    with wave.open(str(path), 'rb') as wav:
        rate, frames = wav.getframerate(), wav.getnframes()
        channels, width = wav.getnchannels(), wav.getsampwidth()
        if wav.getcomptype() != 'NONE' or width not in (2, 3, 4) or channels < 1:
            raise ValueError('Expected uncompressed PCM16/24/32 WAV: ' + str(path))
        if (rate, frames) != (segment['sample_rate'], segment['num_samples']):
            raise ValueError('Original sample grid changed: ' + str(path))
        reuse_pcm24 = width == 3 and channels == 1 and rate == 48000
        raw = None if reuse_pcm24 else wav.readframes(frames)
    if reuse_pcm24:
        from scripts.prepare_gtsinger_mixtures import read_pcm24_mono
        return read_pcm24_mono(path)
    if len(raw) != frames * channels * width or frames < 1:
        raise ValueError('Empty or truncated PCM WAV: ' + str(path))
    if width == 3:
        octets = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        integers = octets[:, 0] | (octets[:, 1] << 8) | (octets[:, 2] << 16)
        integers = (integers ^ (1 << 23)) - (1 << 23)
    else:
        integers = np.frombuffer(raw, dtype='<i' + str(width))
    samples = integers.reshape(frames, channels)[:, 0].astype(np.float64)
    return samples / (1 << (width * 8 - 1))


def load_coarse_report(path, segments, input_sha256):
    report = json.loads(Path(path).read_text())
    if not isinstance(report, dict) or report.get('schema_version') != 1 or report.get('status') != 'completed':
        raise ValueError('Coarse report must be schema 1 and completed')
    recorded = report.get('input_sha256')
    # Content identities remain valid when a frozen report is copied between hosts.
    if (not isinstance(recorded, dict) or len(recorded) != 2
            or Counter(recorded.values()) != Counter(input_sha256[k] for k in ('segments', 'splits'))):
        raise ValueError('Coarse report input SHA256 differs from segments/splits')
    rows = report.get('rows')
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError('Coarse report rows must be a list')
    by_id = {row.get('segment_id'): row for row in rows}
    if len(by_id) != len(rows) or set(by_id) != {s['segment_id'] for s in segments}:
        raise ValueError('Coarse report coverage must exactly match segments')
    for segment in segments:
        row = by_id[segment['segment_id']]
        if row.get('song_id') != segment['song_id'] or row.get('split') != segment['split']:
            raise ValueError('Coarse report song/split mismatch: ' + segment['segment_id'])
        for method in METHODS:
            run = row.get('runs', {}).get(method + ':legacy', {})
            if run.get('status') != 'completed' or not isinstance(run.get('candidates'), list):
                raise ValueError('Incomplete coarse legacy run: ' + segment['segment_id'] + ':' + method)
            for candidate in run['candidates']:
                if (not isinstance(candidate, dict)
                        or candidate.get('original_sample_rate') != segment['sample_rate']
                        or candidate.get('original_num_samples') != segment['num_samples']):
                    raise ValueError('Coarse candidate sample grid differs: ' + segment['segment_id'])
    return by_id


def choose_configuration(validation):
    """Only validation summaries enter this rule; no test labels or errors are accepted."""
    baseline = validation['baseline']
    if baseline['mae_ms'] is None:
        raise ValueError('No evaluated validation boundaries: cannot select a DSP trial')
    available = [(trial['metrics']['mae_ms'], index, trial) for index, trial in enumerate(
                 validation['trials'].values()) if trial['metrics']['mae_ms'] is not None]
    if not available:
        raise ValueError('No evaluated DSP validation trial')
    best = min(available, key=lambda item: (item[0], item[1]))[2]
    metrics = best['metrics']
    gates = {
        'strict_mae_improvement': metrics['mae_ms'] < baseline['mae_ms'],
        'evaluated_not_decreased': metrics['evaluated'] >= baseline['evaluated'],
        'coverage_not_decreased': all(metrics['coverage_of_all'][str(ms)] is not None
                                      and metrics['coverage_of_all'][str(ms)] >= baseline['coverage_of_all'][str(ms)]
                                      for ms in (5, 10, 20, 50)),
        'severe_not_increased': metrics['severe_over_500ms'] <= baseline['severe_over_500ms'],
    }
    selected = 'dsp' if all(gates.values()) else 'none'
    return {'best_trial_config': best['config'], 'selected': selected,
            'selected_config': best['config'] if selected == 'dsp' else None,
            'decision': {'gates': gates, 'reason': 'all_validation_gates_passed' if selected == 'dsp'
                         else 'validation_gates_not_met; retain_coarse'}}


def _coarse_run(segment, candidates):
    return {'candidates': copy.deepcopy(candidates), 'source_metrics': source_errors(segment, candidates),
            'runtime_seconds': 0.0, 'cpu_seconds': 0.0, 'gpu': None,
            'runtime_scope': 'frozen candidates; excludes original model inference'}


def _dsp_run(segment, candidates, audio, features, config):
    wall, cpu = time.perf_counter(), time.process_time()
    refined = refine_boundaries(audio, segment['sample_rate'], candidates, config=config, features=features)
    result = {'runtime_seconds': time.perf_counter() - wall,
              'cpu_seconds': time.process_time() - cpu, 'gpu': None,
              'runtime_scope': 'refinement only; feature extraction recorded per segment',
              'candidates': refined, 'source_metrics': source_errors(segment, refined)}
    result['refinement_counts'] = dict(Counter(row['refinement']['status'] for row in refined))
    result['fallback_reasons'] = dict(Counter(row['refinement']['reason'] for row in refined
                                            if row['refinement']['status'] == 'fallback'))
    return result


def _audio_features(segment):
    audio = read_audio(segment)
    wall, cpu = time.perf_counter(), time.process_time()
    features = extract_features(audio, segment['sample_rate'], hop_samples=CONFIG_GRID[0].hop_samples)
    timing = {'runtime_seconds': time.perf_counter() - wall, 'cpu_seconds': time.process_time() - cpu,
              'gpu': None, 'metadata': features['metadata']}
    return audio, features, timing


def select_configuration(validation_segments, coarse_rows):
    """Callers must provide validation segments only; source labels never enter DSP."""
    if not validation_segments or any(segment['split'] != 'val' for segment in validation_segments):
        raise ValueError('Selection accepts nonempty validation segments only')
    if set(coarse_rows) != {s['segment_id'] for s in validation_segments}:
        raise ValueError('Selection coarse rows must cover validation only')
    rows, buckets = [], defaultdict(list)
    for segment in validation_segments:
        audio, features, timing = _audio_features(segment)
        row = {key: segment[key] for key in ('segment_id', 'song_id', 'split', 'language')}
        row.update(feature_extraction=timing, runs={})
        for method in METHODS:
            candidates = coarse_rows[segment['segment_id']]['runs'][method + ':legacy']['candidates']
            coarse = _coarse_run(segment, candidates)
            trials = {config.name: _dsp_run(segment, candidates, audio, features, config) for config in CONFIG_GRID}
            row['runs'][method] = {'coarse': coarse, 'trials': trials}
            buckets[(method, 'coarse', 'val')].append(coarse['source_metrics'])
            for name, run in trials.items():
                buckets[(method, name, 'val')].append(run['source_metrics'])
        rows.append(row)
    baseline = summarize([metrics for (method, variant, split), values in buckets.items()
                          if variant == 'coarse' for metrics in values])
    trials = {config.name: {'config': config.to_dict(),
                           'metrics': summarize([metrics for method in METHODS
                                                 for metrics in buckets[(method, config.name, 'val')]]),
                           'by_model': {method: summarize(buckets[(method, config.name, 'val')]) for method in METHODS}}
              for config in CONFIG_GRID}
    validation = {'baseline': baseline, 'trials': trials}
    return rows, {':'.join(key): summarize(values) for key, values in buckets.items()}, validation


def verify_selection(selection, input_sha256, runtime_sha256):
    if (not isinstance(selection, dict) or selection.get('protocol') != 'dsp_configuration_selection'
            or selection.get('schema_version') != 1 or selection.get('status') != 'completed'):
        raise ValueError('Expected completed DSP selection schema 1')
    if selection.get('input_sha256') != input_sha256:
        raise ValueError('Selection input SHA256 differs')
    if selection.get('runtime_sha256') != runtime_sha256:
        raise ValueError('Selection DSP/features SHA256 differs')
    if selection.get('config_grid') != [config.to_dict() for config in CONFIG_GRID] or selection.get('policy') != POLICY:
        raise ValueError('Selection configuration grid or policy differs')
    validation = selection.get('validation', {})
    trials = validation.get('trials', {})
    if list(trials) != [config.name for config in CONFIG_GRID] or any(
            trials[config.name].get('config') != config.to_dict() for config in CONFIG_GRID):
        raise ValueError('Selection validation trial identities differ')
    expected = choose_configuration(validation)
    if any(selection.get(key) != value for key, value in expected.items()):
        raise ValueError('Selection decision differs from frozen validation evidence')
    return DSPConfig(**selection['best_trial_config'])


def evaluate_configuration(segments, coarse_rows, selection):
    config = DSPConfig(**selection['best_trial_config'])
    rows, buckets = [], defaultdict(list)
    for segment in segments:
        audio, features, timing = _audio_features(segment)
        row = {key: segment[key] for key in ('segment_id', 'song_id', 'split', 'language')}
        row.update(feature_extraction=timing, runs={})
        for method in METHODS:
            candidates = coarse_rows[segment['segment_id']]['runs'][method + ':legacy']['candidates']
            coarse = _coarse_run(segment, candidates)
            trial = _dsp_run(segment, candidates, audio, features, config)
            variant = 'dsp_trial' if selection['selected'] == 'dsp' else 'coarse'
            runs = {'coarse': coarse, 'dsp_trial': trial,
                    'selected': {'variant': variant, 'source_metrics': (trial if variant == 'dsp_trial' else coarse)['source_metrics']}}
            row['runs'][method] = runs
            for name, run in runs.items():
                buckets[(method, name, segment['split'])].append(run['source_metrics'])
        rows.append(row)
    return rows, {':'.join(key): summarize(values) for key, values in buckets.items()}


def _diagnostics(rows):
    failures, errors = [], []
    for row in rows:
        for method, runs in row['runs'].items():
            variants = {'coarse': runs['coarse'], **runs.get('trials', {})}
            if 'dsp_trial' in runs:
                variants['dsp_trial'] = runs['dsp_trial']
            for variant, run in variants.items():
                metrics = run['source_metrics']
                identity = {key: row[key] for key in ('segment_id', 'song_id', 'split')}
                identity.update(method=method, variant=variant)
                if metrics['missing'] or metrics['text_mismatch']:
                    failures.append({**identity, 'missing': metrics['missing'], 'text_mismatch': metrics['text_mismatch']})
                errors.extend({**identity, **error} for error in metrics['errors'])
    return {'failure_count': len(failures), 'failures': failures[:50],
            'largest_errors': sorted(errors, key=lambda error: -error['absolute_error_ms'])[:50],
            'detail_limit': 50}


def _save(path, document):
    temporary = path.with_suffix(path.suffix + '.tmp')
    try:
        temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('segments', 'splits', 'audio-root', 'coarse-report', 'out-dir'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--selection-file', type=Path)
    args = parser.parse_args(argv)
    try:
        inputs = {'segments': digest(args.segments), 'splits': digest(args.splits),
                  'coarse_report': digest(args.coarse_report)}
        runtime = runtime_digests()
        segments = load_inputs(args.segments, args.splits, args.audio_root)
        coarse_rows = load_coarse_report(args.coarse_report, segments, inputs)
        selection = json.loads(args.selection_file.read_text()) if args.selection_file else None
        if args.selection_file is not None:
            verify_selection(selection, inputs, runtime)
        args.out_dir.mkdir(parents=True, exist_ok=False)
    except (OSError, ValueError, KeyError, TypeError, wave.Error) as error:
        parser.exit(1, str(error) + '\n')
    report = {'schema_version': 1, 'status': 'running',
              'stage': 'evaluate' if selection is not None else 'select',
              'started_at': datetime.now(timezone.utc).isoformat(),
              'reference': 'published_dataset_annotation', 'independent_human_accuracy': False,
              'input_sha256': inputs, 'runtime_sha256': runtime,
              'benchmark_sha256': digest(Path(__file__)),
              'config_grid': [config.to_dict() for config in CONFIG_GRID], 'policy': POLICY,
              'environment': {'python': sys.version, 'numpy': np.__version__, 'platform': platform.platform(),
                              'device': 'cpu', 'gpu': None},
              'rows': [], 'limitations': [
                  'Source annotation differences are not independently verified human accuracy.',
                  'No source alias confirmation or independent test-set claim.',
                  'Selection uses validation only; training metrics are diagnostic.',
                  'Only the best validation trial is evaluated on test, even when selected remains coarse.',
                  'No resampling, waveform shift, calibrated confidence, or model inference.']}
    target = args.out_dir / 'result.json'
    wall, cpu = time.perf_counter(), time.process_time()
    try:
        _save(target, report)
        if selection is None:
            validation_segments = [segment for segment in segments if segment['split'] == 'val']
            validation_coarse = {segment['segment_id']: coarse_rows[segment['segment_id']] for segment in validation_segments}
            rows, summary, validation = select_configuration(validation_segments, validation_coarse)
            selection = {'protocol': 'dsp_configuration_selection', 'schema_version': 1, 'status': 'completed',
                         'input_sha256': inputs, 'runtime_sha256': runtime,
                         'config_grid': [config.to_dict() for config in CONFIG_GRID],
                         'policy': POLICY, 'validation': validation, **choose_configuration(validation)}
        else:
            report['selection_file'] = {'path': str(args.selection_file), 'sha256': digest(args.selection_file)}
            rows, summary = evaluate_configuration(segments, coarse_rows, selection)
        if runtime_digests() != runtime or any(digest(path) != inputs[name] for name, path in
                (('segments', args.segments), ('splits', args.splits), ('coarse_report', args.coarse_report))):
            raise ValueError('Input or DSP/features changed during benchmark')
        report.update(status='completed', rows=rows, summary=summary, selection=selection,
                      runtime_seconds=time.perf_counter() - wall, cpu_seconds=time.process_time() - cpu,
                      gpu=None, **_diagnostics(rows))
        if args.selection_file is None:
            _save(args.out_dir / 'selection.json', selection)
        _save(target, report)
    except Exception as error:
        report.update(status='failed', error={'type': type(error).__name__, 'message': str(error)},
                      runtime_seconds=time.perf_counter() - wall, cpu_seconds=time.process_time() - cpu, gpu=None)
        _save(target, report)
        parser.exit(1, str(error) + '\n')
    print(target)
    return 0


if __name__ == '__main__':
    raise SystemExit(cli())
