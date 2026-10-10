#!/usr/bin/env python3
"""Cache legacy CUDA candidates for training, reusing verified prior predictions."""
import argparse
from collections import defaultdict
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_coarse_gtsinger import digest, load_inputs, source_errors, summarize

METHODS = ('mtl', 'mtl_bdr')


def input_identity(segment):
    return {
        'segment_id': segment['segment_id'], 'song_id': segment['song_id'],
        'split': segment['split'], 'language': segment['language'],
        'wav_sha256': segment['sources']['wav']['sha256'],
        'sample_rate': segment['sample_rate'], 'num_samples': segment['num_samples'],
        'lyrics': [u['text'] for u in segment['units'] if u['kind'] == 'lyric'],
    }


def reusable_runs(segment, previous_segment, previous_row):
    if input_identity(segment) != input_identity(previous_segment):
        raise ValueError('Prior prediction input differs: ' + segment['segment_id'])
    if (previous_row['segment_id'], previous_row['song_id'], previous_row['split']) != (
            segment['segment_id'], segment['song_id'], segment['split']):
        raise ValueError('Prior prediction identity differs')
    runs = {}
    for method in METHODS:
        key = method + ':legacy'
        run = previous_row['runs'][key]
        if run.get('status') != 'completed':
            raise ValueError('Prior prediction was not completed')
        for candidate in run['candidates']:
            if (candidate['original_sample_rate'], candidate['original_num_samples']) != (
                    segment['sample_rate'], segment['num_samples']):
                raise ValueError('Prior prediction sample grid differs')
        runs[key] = copy.deepcopy(run)
        runs[key]['source_metrics'] = source_errors(segment, runs[key]['candidates'])
        runs[key]['execution'] = 'reused_verified_prior_prediction'
    return runs


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n')
    temporary.replace(path)


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('segments', 'splits', 'audio-root', 'out-dir', 'reuse-report', 'reuse-segments', 'reuse-splits'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args(argv)
    segments = load_inputs(args.segments, args.splits, args.audio_root)
    old_segments = {s['segment_id']: s for s in load_inputs(args.reuse_segments, args.reuse_splits, args.audio_root)}
    previous = json.loads(args.reuse_report.read_text())
    if previous.get('status') != 'completed' or previous.get('schema_version') != 1:
        parser.error('Expected completed prior coarse report')
    if sorted(previous['input_sha256'].values()) != sorted([digest(args.reuse_segments), digest(args.reuse_splits)]):
        parser.error('Prior report input hashes differ')
    previous_rows = {r['segment_id']: r for r in previous['rows']}
    if len(previous_rows) != len(previous['rows']) or set(previous_rows) != set(old_segments):
        parser.error('Prior report does not exactly cover prior inputs')
    runtime = {name: digest(ROOT / name) for name in previous['runtime_sha256']
               if name != 'scripts/benchmark_coarse_gtsinger.py'}
    if any(runtime[name] != previous['runtime_sha256'][name] for name in runtime):
        parser.error('Coarse inference implementation changed: cannot reuse prior predictions')
    checkpoints = {name: digest(ROOT / 'checkpoints' / name) for name in ('checkpoint_MTL', 'checkpoint_BDR')}
    if checkpoints != previous['checkpoints']:
        parser.error('Coarse checkpoint changed: cannot reuse prior predictions')
    args.out_dir.mkdir(parents=True, exist_ok=False)
    target = args.out_dir / 'result.json'
    report = {
        'schema_version': 1, 'status': 'running', 'seed': 7,
        'started_at': datetime.now(timezone.utc).isoformat(), 'rows': [],
        'input_sha256': {str(p): digest(p) for p in (args.segments, args.splits)},
        'runtime_sha256': runtime, 'cache_script_sha256': digest(Path(__file__)),
        'checkpoints': checkpoints, 'reuse_report_sha256': digest(args.reuse_report),
        'reference': 'published_dataset_annotation', 'independent_human_accuracy': False,
        'limitations': ['Only legacy candidates; unchanged constrained runs are not repeated.',
                        'Published source annotations are not verified human truth.',
                        'Reused runs retain their original CUDA observations and timing.'],
    }
    save(target, report)
    try:
        import torch
        from t2l.alignment.coarse import coarse_align
        from t2l.mtl.wrapper import load_mtl_model
        from scripts.benchmark_gtsinger_mixtures import measured_cuda_call, cuda_forward_observations
        if not torch.cuda.is_available():
            raise RuntimeError('Configured CUDA environment required')
        torch.set_num_threads(2)
        torch.manual_seed(7)
        torch.cuda.set_device(0)
        report['environment'] = {'python': sys.version, 'torch': torch.__version__,
                                 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0)}
        models = load_mtl_model('MTL_BDR', cuda=True, verbose=False)
        variants = {'mtl': (models[0], None, models[2], False, models[4]), 'mtl_bdr': models}
        buckets = defaultdict(list)
        reused, executed = 0, 0
        for index, segment in enumerate(segments):
            row = {name: segment[name] for name in ('segment_id', 'song_id', 'split')}
            language = {'Chinese': 'zh', 'English': 'en'}[segment['language']]
            row.update(language=language, runs={})
            report['rows'].append(row)
            if segment['segment_id'] in old_segments:
                row['runs'] = reusable_runs(segment, old_segments[segment['segment_id']],
                                            previous_rows[segment['segment_id']])
                reused += len(METHODS)
            else:
                words = [u['text'] for u in segment['units'] if u['kind'] == 'lyric']
                lyrics = [''.join(words) if language == 'zh' else ' '.join(words)]
                for method, model in variants.items():
                    run = {'observed_forwards': [], 'constraints_input': None, 'execution': 'new_cuda_forward'}
                    row['runs'][method + ':legacy'] = run
                    roles = [('acoustic', model[0])] + ([('boundary', model[1])] if model[3] else [])
                    def invoke():
                        with cuda_forward_observations(torch, roles, run['observed_forwards']):
                            return [c.to_dict() for c in coarse_align(segment['audio_path'], lyrics,
                                    method=method, model=model, language=language)]
                    run['candidates'] = measured_cuda_call(torch, run, invoke)
                    run['source_metrics'] = source_errors(segment, run['candidates'])
                    executed += 1
            for key, run in row['runs'].items():
                buckets[(key, segment['split'])].append(run['source_metrics'])
            report['counts'] = {'reused_runs': reused, 'new_cuda_runs': executed}
            save(target, report)
            print('{}/{} {}'.format(index + 1, len(segments), segment['segment_id']), flush=True)
        report['summary'] = {':'.join(key): summarize(value) for key, value in buckets.items()}
        report['status'] = 'completed'
        save(target, report)
    except Exception as error:
        report.update(status='failed', error={'type': type(error).__name__, 'message': str(error)})
        save(target, report)
        raise
    print(target)
    return 0


if __name__ == '__main__':
    raise SystemExit(cli())
