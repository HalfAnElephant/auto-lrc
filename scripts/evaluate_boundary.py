#!/usr/bin/env python3
"""Evaluate one frozen neural checkpoint and a fixed DSP-plus-neural ablation."""
import argparse
from collections import Counter, defaultdict
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.benchmark_coarse_gtsinger import digest, load_inputs, source_errors, summarize
from scripts.benchmark_dsp_gtsinger import METHODS, load_coarse_report, read_audio
from scripts.train_boundary import runtime_digests as training_runtime_digests, write_json
from t2l.boundary.dsp import DSPConfig, refine_boundaries as refine_dsp
from t2l.boundary.features import extract_features
from t2l.evaluation.metrics import classified_records, grouped_summary, record_summary, song_bootstrap

DSP_TRIAL = DSPConfig(name='low_prior', prior_weight=.1, min_evidence=.25)
VARIANTS = ('coarse', 'dsp', 'neural', 'dsp_neural')


def runtime_digests():
    result = training_runtime_digests()
    result.update({name: digest(ROOT / name) for name in
                   ('scripts/evaluate_boundary.py', 't2l/boundary/dsp.py', 't2l/evaluation/metrics.py')})
    return result


def verify_training_checkpoint(path, input_sha256):
    """Require a completed training result, not a best weight left by a failed run."""
    import torch
    path = Path(path)
    if path.name != 'boundary.pt' or not path.is_file() or path.is_symlink():
        raise ValueError('Expected regular boundary.pt from its frozen training output directory')
    result_path = path.parent / 'result.json'
    report = json.loads(result_path.read_text())
    if (not isinstance(report, dict) or report.get('protocol') != 'boundary_training'
            or report.get('schema_version') != 1 or report.get('status') != 'completed'
            or report.get('selection_split') != 'val'):
        raise ValueError('Checkpoint requires a completed validation-selected training result')
    if report.get('checkpoint', {}).get('sha256') != digest(path):
        raise ValueError('Checkpoint SHA256 differs from completed training result')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get('schema_version') != 'boundary-checkpoint-v1':
        raise ValueError('Expected boundary-checkpoint-v1')
    metadata = checkpoint.get('training_metadata', {})
    if (not isinstance(metadata, dict) or metadata.get('selection_split') != 'val'
            or metadata.get('training_splits') != ['train', 'val']
            or metadata.get('reference') != 'published_dataset_annotation'
            or metadata.get('independent_human_accuracy') is not False):
        raise ValueError('Checkpoint lacks the train/validation-only source annotation provenance')
    if metadata.get('input_sha256') != report.get('input_sha256'):
        raise ValueError('Checkpoint and training input SHA256 differ')
    if any(metadata['input_sha256'].get(key) != input_sha256[key]
           for key in ('segments', 'splits', 'coarse_report')):
        raise ValueError('Evaluation input SHA256 differs from frozen training dataset/coarse predictions')
    expected = training_runtime_digests()
    if metadata.get('runtime_sha256') != expected or report.get('runtime_sha256') != expected:
        raise ValueError('Training/inference runtime SHA256 differs from frozen checkpoint')
    if (checkpoint.get('config') != report.get('config')
            or metadata.get('training') != report.get('training')
            or metadata.get('best_epoch') != report.get('best_epoch')
            or metadata.get('validation') != report.get('best_validation')):
        raise ValueError('Checkpoint configuration/selection differs from completed training evidence')
    return copy.deepcopy(metadata)


def measured_call(runtime, function):
    import torch
    device = torch.device(runtime['device'])
    is_cuda = device.type == 'cuda'
    if is_cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    wall, cpu = time.perf_counter(), time.process_time()
    value = function()
    if is_cuda:
        torch.cuda.synchronize(device)
    return value, {'runtime_seconds': time.perf_counter() - wall,
                   'cpu_seconds': time.process_time() - cpu,
                   'gpu': {'device': str(device), 'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
                           'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(device)} if is_cuda else None}


def _run(segment, candidates, timing=None):
    result = {'status': 'completed', 'candidates': candidates,
              'source_metrics': source_errors(segment, candidates),
              'runtime_seconds': 0., 'cpu_seconds': 0., 'gpu': None,
              'runtime_scope': 'refinement only; frozen coarse acoustic inference excluded'}
    if timing is not None:
        result.update(timing)
    else:
        result['runtime_scope'] = 'frozen coarse candidates; original model inference excluded'
    result['refinement_counts'] = dict(Counter(row.get('refinement', {}).get('status', 'coarse') for row in candidates))
    result['fallback_reasons'] = dict(Counter(row['refinement']['reason'] for row in candidates
                                            if row.get('refinement', {}).get('status') == 'fallback'))
    return result


def evaluate(segments, coarse_rows, runtime, *, batch_size=32):
    from t2l.boundary.inference import refine_boundaries
    rows, metrics, records, failures = [], defaultdict(list), defaultdict(list), []
    for segment in segments:
        row = {key: segment[key] for key in ('segment_id', 'song_id', 'split', 'language')}
        row['runs'] = {}
        rows.append(row)
        # Source labels are consumed only by reporting, never passed into a refiner.
        try:
            wall, cpu = time.perf_counter(), time.process_time()
            audio = read_audio(segment)
            features = extract_features(audio, segment['sample_rate'], hop_samples=DSP_TRIAL.hop_samples)
            row['audio_and_dsp_features'] = {'runtime_seconds': time.perf_counter() - wall,
                                           'cpu_seconds': time.process_time() - cpu, 'gpu': None,
                                           'metadata': features['metadata']}
            audio_error = None
        except Exception as error:
            audio_error = error
        for method in METHODS:
            candidates = copy.deepcopy(coarse_rows[segment['segment_id']]['runs'][method + ':legacy']['candidates'])
            row['runs'][method] = {'coarse': _run(segment, candidates)}
            dsp_candidates, dsp_timing, dsp_error = None, None, None
            for variant in ('dsp', 'neural', 'dsp_neural'):
                try:
                    if audio_error is not None:
                        raise audio_error
                    if variant == 'dsp':
                        intermediate, dsp_timing = measured_call({'device': 'cpu'}, lambda: refine_dsp(
                            audio, segment['sample_rate'], candidates, config=DSP_TRIAL, features=features))
                        dsp_candidates = intermediate
                        predictions, timing = intermediate, dsp_timing
                    elif variant == 'dsp_neural':
                        if dsp_error is not None:
                            raise dsp_error
                        intermediate = dsp_candidates
                        predictions, timing = measured_call(runtime, lambda: refine_boundaries(
                            audio, segment['sample_rate'], intermediate, loaded=runtime, device=runtime['device'],
                            batch_size=batch_size))
                    else:
                        intermediate = candidates
                        predictions, timing = measured_call(runtime, lambda: refine_boundaries(
                            audio, segment['sample_rate'], intermediate, loaded=runtime, device=runtime['device'],
                            batch_size=batch_size))
                    run = _run(segment, predictions, timing)
                    if variant == 'dsp_neural':
                        run.update(dsp_candidates=intermediate, dsp_timing=dsp_timing)
                        run['runtime_seconds'] += dsp_timing['runtime_seconds']
                        run['cpu_seconds'] += dsp_timing['cpu_seconds']
                    row['runs'][method][variant] = run
                except Exception as error:
                    if variant == 'dsp':
                        dsp_error = error
                    missing = copy.deepcopy(candidates)
                    for candidate in missing:
                        candidate.update(start_sample=None, start_seconds=None, status='unaligned')
                    run = _run(segment, missing)
                    run.update(status='failed', error={'type': type(error).__name__, 'message': str(error)},
                               runtime_scope='failed refinement; every source boundary retained in denominator')
                    row['runs'][method][variant] = run
                    failures.append({'segment_id': segment['segment_id'], 'method': method,
                                     'variant': variant, **run['error']})
            for variant, run in row['runs'][method].items():
                key = (method, variant, segment['split'])
                metrics[key].append(run['source_metrics'])
                records[key].extend(classified_records(segment, run['candidates']))
    return {'rows': rows, 'summary': {':'.join(key): summarize(value) for key, value in metrics.items()},
            'grouped': {':'.join(key): grouped_summary(value) for key, value in records.items()},
            'record_summary': {':'.join(key): record_summary(value) for key, value in records.items()},
            'song_bootstrap': {':'.join(key): song_bootstrap(value) for key, value in records.items() if key[2] == 'test'},
            'failure_count': len(failures), 'failure_examples': failures[:50]}


def cli(argv=None):
    cli_started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('segments', 'splits', 'audio-root', 'coarse-report', 'checkpoint', 'out-dir'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args(argv)
    report, created = None, False
    try:
        if args.batch_size < 1:
            raise ValueError('batch-size must be positive')
        inputs = {name: digest(path) for name, path in
                  (('segments', args.segments), ('splits', args.splits), ('coarse_report', args.coarse_report))}
        segments = load_inputs(args.segments, args.splits, args.audio_root)
        coarse = load_coarse_report(args.coarse_report, segments, inputs)
        metadata = verify_training_checkpoint(args.checkpoint, inputs)
        checkpoint_sha256 = digest(args.checkpoint)
        args.out_dir.mkdir(parents=True, exist_ok=False)
        created = True
        report = {'schema_version': 1, 'protocol': 'boundary_evaluation', 'status': 'running',
                  'started_at': datetime.now(timezone.utc).isoformat(), 'input_sha256': inputs,
                  'runtime_sha256': runtime_digests(), 'checkpoint_sha256': checkpoint_sha256,
                  'training_metadata': metadata, 'fixed_dsp_trial_config': DSP_TRIAL.to_dict(),
                  'reference': 'published_dataset_annotation', 'independent_human_accuracy': False,
                  'selection_split': 'val', 'test_used_for_selection': False,
                  'limitations': ['One frozen checkpoint; no test-driven adjustment or trial scan.',
                      'DSP plus neural is a fixed low_prior research ablation, not the default system.',
                      'Published annotation disagreement is not independently verified human onset accuracy.',
                      'Song bootstrap includes failures; few test songs give unstable intervals.',
                      'Runtime excludes original coarse acoustic inference; CPU decode/DSP features recorded per segment.']}
        write_json(args.out_dir / 'result.json', report)
        from t2l.boundary.inference import load_boundary_checkpoint
        import torch
        runtime = load_boundary_checkpoint(args.checkpoint, device=args.device)
        torch.set_num_threads(2)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        report['environment'] = {'python': sys.version, 'torch': str(torch.__version__),
                                 'torch_num_threads': torch.get_num_threads(),
                                 'torch_num_interop_threads': torch.get_num_interop_threads(),
                                 'cuda': torch.version.cuda, 'device': runtime['device'],
                                 'gpu': torch.cuda.get_device_name(runtime['device'])
                                        if torch.device(runtime['device']).type == 'cuda' else None}
        report['config'] = runtime['config'].to_dict()
        started = time.perf_counter()
        report.update(evaluate(segments, coarse, runtime, batch_size=args.batch_size))
        report['runtime_seconds'] = time.perf_counter() - started
        if (runtime_digests() != report['runtime_sha256'] or digest(args.checkpoint) != checkpoint_sha256
                or any(digest(path) != inputs[name] for name, path in
                       (('segments', args.segments), ('splits', args.splits), ('coarse_report', args.coarse_report)))):
            raise ValueError('Evaluation inputs/checkpoint/runtime changed during execution')
        report['status'] = 'completed' if not report['failure_count'] else 'failed'
        report['total_wall_seconds'] = time.perf_counter() - cli_started
        write_json(args.out_dir / 'result.json', report)
        if report['failure_count']:
            print('Refinement failures: ' + str(report['failure_count']), file=sys.stderr)
            return 1
    except Exception as error:
        if created and report is not None:
            report.update(status='failed', total_wall_seconds=time.perf_counter() - cli_started,
                          error={'type': type(error).__name__, 'message': str(error)})
            write_json(args.out_dir / 'result.json', report)
        print(type(error).__name__ + ': ' + str(error), file=sys.stderr)
        return 1
    print(args.out_dir / 'result.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(cli())
