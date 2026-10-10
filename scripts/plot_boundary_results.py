#!/usr/bin/env python3
"""Plot frozen boundary training/evaluation evidence without inference or tuning."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.benchmark_coarse_gtsinger import digest, load_inputs, text_key

METHODS = ('mtl', 'mtl_bdr')
BASE_VARIANTS = ('coarse', 'dsp', 'neural', 'dsp_neural')
COLORS = ('#32648e', '#da7143', '#7b4da3', '#15917c', '#aa4758')
REFERENCE = 'published_dataset_annotation'
CASE_POLICY = 'Largest absolute source error across test methods/variants; six unique segment/unit pairs; identity tie-break'


def _number(value, label, *, minimum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Expected finite number: ' + label)
    if minimum is not None and value < minimum:
        raise ValueError('Number below minimum: ' + label)
    return value


def _integer(value, label, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError('Expected integer >= {}: {}'.format(minimum, label))
    return value


def _sha(value, label):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdefABCDEF' for c in value):
        raise ValueError('Expected SHA256: ' + label)
    return value


def _completed(report, protocol):
    if (not isinstance(report, dict) or type(report.get('schema_version')) is not int
            or report['schema_version'] != 1 or report.get('protocol') != protocol
            or report.get('status') != 'completed' or report.get('reference') != REFERENCE
            or report.get('independent_human_accuracy') is not False
            or report.get('selection_split') != 'val'):
        raise ValueError('Expected completed validation-selected source report: ' + protocol)


def _read_json(path):
    def invalid_constant(value):
        raise ValueError('Nonfinite JSON constant in {}: {}'.format(path, value))
    return json.loads(Path(path).read_text(), parse_constant=invalid_constant)


def prepare_results(training, evaluation):
    """Extract plot data only; preserve reported failures in the CDF denominator."""
    _completed(training, 'boundary_training')
    _completed(evaluation, 'boundary_evaluation')
    if evaluation.get('test_used_for_selection') is not False:
        raise ValueError('Evaluation must explicitly exclude test-driven selection')
    inputs = training.get('input_sha256', {})
    evaluated_inputs = evaluation.get('input_sha256', {})
    for key in ('segments', 'splits', 'coarse_report'):
        if _sha(inputs.get(key), 'training ' + key) != _sha(evaluated_inputs.get(key), 'evaluation ' + key):
            raise ValueError('Training/evaluation input SHA256 differs: ' + key)
    if (_sha(training.get('checkpoint', {}).get('sha256'), 'training checkpoint')
            != _sha(evaluation.get('checkpoint_sha256'), 'evaluation checkpoint')):
        raise ValueError('Training/evaluation checkpoint SHA256 differs')
    metadata = evaluation.get('training_metadata')
    if (not isinstance(metadata, dict) or metadata.get('training_splits') != ['train', 'val']
            or metadata.get('selection_split') != 'val' or metadata.get('reference') != REFERENCE
            or metadata.get('independent_human_accuracy') is not False):
        raise ValueError('Evaluation lacks frozen train/validation-only provenance')
    for key in ('input_sha256', 'runtime_sha256', 'training', 'best_epoch'):
        if key not in training or metadata.get(key) != training[key]:
            raise ValueError('Training metadata differs: ' + key)
    if metadata.get('validation') != training.get('best_validation'):
        raise ValueError('Frozen validation result differs')
    if training.get('config') != evaluation.get('config'):
        raise ValueError('Training/evaluation feature/model configuration differs')

    history = training.get('history')
    if not isinstance(history, list) or not history:
        raise ValueError('Training history is empty')
    trajectory, previous = [], 0
    for entry in history:
        epoch = _integer(entry.get('epoch'), 'history epoch', minimum=1)
        if epoch <= previous:
            raise ValueError('History epochs must be strictly increasing')
        previous = epoch
        components = entry.get('loss_parts')
        if not isinstance(components, dict) or not components:
            raise ValueError('Missing training loss components')
        trajectory.append({'epoch': epoch, 'loss': _number(entry.get('loss'), 'training loss', minimum=0),
            'loss_parts': {name: _number(value, 'loss ' + name, minimum=0) for name, value in components.items()},
            'val_mae_ms': _number(entry.get('validation', {}).get('mae_ms'), 'validation MAE', minimum=0)})
    best = _integer(training.get('best_epoch'), 'best epoch', minimum=1)
    best_entries = [entry for entry in history if entry['epoch'] == best]
    if len(best_entries) != 1 or best_entries[0].get('validation') != training.get('best_validation'):
        raise ValueError('Best epoch/validation must match the frozen history')

    rows = evaluation.get('rows')
    if not isinstance(rows, list) or not rows:
        raise ValueError('Evaluation rows are empty')
    identities = set()
    for row in rows:
        identity = row.get('segment_id')
        if not isinstance(identity, str) or not identity or identity in identities:
            raise ValueError('Missing or duplicate evaluation segment identity')
        identities.add(identity)
        if row.get('split') not in ('train', 'val', 'test'):
            raise ValueError('Invalid evaluation split: ' + identity)
    test_rows = [row for row in rows if row['split'] == 'test']
    if not test_rows:
        raise ValueError('No test rows to plot')
    first_runs = test_rows[0].get('runs', {}).get(METHODS[0], {})
    extra = sorted(set(first_runs) - set(BASE_VARIANTS))
    if not set(BASE_VARIANTS).issubset(first_runs) or len(extra) > 1:
        raise ValueError('Expected four frozen variants, with at most one additional variant')
    variants = list(BASE_VARIANTS) + extra
    distributions = {method: {variant: {'total': 0, 'missing': 0, 'text_mismatch': 0, 'errors_ms': []}
                              for variant in variants} for method in METHODS}
    references, cases = {}, {}
    for row in test_rows:
        totals = set()
        for method in METHODS:
            runs = row.get('runs', {}).get(method, {})
            if set(runs) != set(variants):
                raise ValueError('Variant coverage differs: ' + row['segment_id'] + ':' + method)
            for variant in variants:
                run = runs[variant]
                if run.get('status') not in ('completed', 'failed'):
                    raise ValueError('Incomplete test run: ' + row['segment_id'])
                metrics = run.get('source_metrics', {})
                counts = {name: _integer(metrics.get(name), name) for name in ('total', 'missing', 'text_mismatch')}
                errors = metrics.get('errors')
                if not isinstance(errors, list) or len(errors) + counts['missing'] + counts['text_mismatch'] != counts['total']:
                    raise ValueError('Source metrics must retain every boundary in the denominator')
                totals.add(counts['total'])
                distribution = distributions[method][variant]
                for name, count in counts.items():
                    distribution[name] += count
                seen_units = set()
                for error in errors:
                    unit = _integer(error.get('unit_index'), 'source unit index')
                    if unit in seen_units:
                        raise ValueError('Duplicate evaluated unit: ' + row['segment_id'])
                    seen_units.add(unit)
                    absolute = _number(error.get('absolute_error_ms'), 'absolute source error', minimum=0)
                    signed = _number(error.get('signed_error_ms'), 'signed source error')
                    if not math.isclose(absolute, abs(signed), rel_tol=1e-9, abs_tol=1e-7):
                        raise ValueError('Signed/absolute source error differs')
                    signature = (_integer(error.get('reference_sample'), 'source sample'),
                                 _number(error.get('reference_seconds'), 'source seconds', minimum=0), error.get('text'))
                    if not isinstance(signature[2], str):
                        raise ValueError('Source error text must be a string')
                    key = (row['segment_id'], unit)
                    if key in references and references[key] != signature:
                        raise ValueError('Source annotation identity differs across variants')
                    references[key] = signature
                    distribution['errors_ms'].append(absolute)
                    candidate = {'segment_id': key[0], 'song_id': row.get('song_id'), 'unit_index': unit,
                        'text': signature[2], 'source_annotation_sample': signature[0], 'source_annotation_seconds': signature[1],
                        'worst_absolute_error_ms': absolute, 'worst_method': method, 'worst_variant': variant}
                    if key not in cases or absolute > cases[key]['worst_absolute_error_ms']:
                        cases[key] = candidate
        if len(totals) != 1:
            raise ValueError('Source denominators differ across methods/variants')
    for by_variant in distributions.values():
        for distribution in by_variant.values():
            distribution['errors_ms'].sort()
            total = distribution['total']
            if total < 1:
                raise ValueError('No test source boundaries')
            distribution['cdf_fraction_of_all'] = [(i + 1) / total for i in range(len(distribution['errors_ms']))]
    ranked = sorted(cases.values(), key=lambda case: (-case['worst_absolute_error_ms'], case['segment_id'], case['unit_index']))[:6]
    return {'trajectory': trajectory, 'best_epoch': best, 'variants': variants,
            'distributions': distributions, 'failure_cases': ranked,
            'test_song_count': len({row.get('song_id') for row in test_rows})}


def display_text(value):
    """Keep figure glyphs ASCII; exact original text remains in index.json."""
    return ''.join(char if 32 <= ord(char) < 127 else '[U+{:04X}]'.format(ord(char)) for char in str(value))


def _plotting():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    return plt, matplotlib.__version__


def render_overview(prepared, out_dir):
    plt, version = _plotting()
    history = prepared['trajectory']
    epochs = [entry['epoch'] for entry in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    axes[0].plot(epochs, [entry['loss'] for entry in history], label='Total loss', color='#263a45', linewidth=2)
    for component in sorted({name for entry in history for name in entry['loss_parts']}):
        axes[0].plot(epochs, [entry['loss_parts'].get(component, float('nan')) for entry in history], label=component, alpha=.75)
    axes[0].set(xlabel='Epoch', ylabel='Training objective (mixed units)', title='Train only; original + synthetic augmentation')
    axes[1].plot(epochs, [entry['val_mae_ms'] for entry in history], color='#32648e', label='Validation source MAE')
    axes[1].axvline(prepared['best_epoch'], color='#da7143', linestyle='--', label='Frozen best epoch')
    axes[1].set(xlabel='Epoch', ylabel='MAE against source annotation (ms)', title='Checkpoint selection: validation only')
    for ax in axes:
        ax.grid(alpha=.15)
        ax.legend(fontsize=8)
    fig.suptitle('Published dataset annotation agreement; not independently verified human accuracy')
    fig.savefig(out_dir / 'training-curves.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for ax, method in zip(axes, METHODS):
        for index, variant in enumerate(prepared['variants']):
            distribution = prepared['distributions'][method][variant]
            values = distribution['errors_ms']
            label = '{} ({}/{}, missing {}, mismatch {})'.format(variant, len(values), distribution['total'],
                        distribution['missing'], distribution['text_mismatch'])
            ax.step([0.] + values, [0.] + distribution['cdf_fraction_of_all'], where='post',
                    color=COLORS[index], label=label)
        for threshold in (5, 10, 20, 50):
            ax.axvline(threshold, color='#777777', linewidth=.6, alpha=.4)
        ax.set_xscale('symlog', linthresh=5)
        ax.set(xlim=(0, max(50., max((value for d in prepared['distributions'][method].values()
                                    for value in d['errors_ms']), default=0.) * 1.05)),
               ylim=(0, 1.02), xlabel='Absolute source error (ms; symlog beyond 5 ms)',
               ylabel='Fraction of ALL source boundaries', title=method.upper())
        ax.grid(alpha=.15)
        ax.legend(fontsize=7, loc='lower right')
    fig.suptitle('Frozen test: {} songs | source annotation agreement, not human accuracy'.format(prepared['test_song_count']))
    fig.savefig(out_dir / 'test-error-cdf.png', dpi=160)
    plt.close(fig)
    return {'matplotlib': version}


def case_evidence(case, row, segment, variants):
    """Associate ordered candidates with one exact original source unit."""
    if row.get('song_id') != segment['song_id'] or row.get('split') != segment['split'] or segment['split'] != 'test':
        raise ValueError('Failure case song/split differs from frozen inputs')
    index = case['unit_index']
    if index >= len(segment['units']):
        raise ValueError('Failure unit index outside source units')
    unit = segment['units'][index]
    if (unit['kind'] != 'lyric' or unit['text'] != case['text']
            or unit['boundary']['source'] != 'textgrid:first_phone'
            or unit['boundary']['candidate_sample'] != case['source_annotation_sample']
            or not math.isclose(case['source_annotation_seconds'], case['source_annotation_sample'] / segment['sample_rate'], abs_tol=1e-12)):
        raise ValueError('Failure source annotation differs from frozen input')
    lyric_indices = [i for i, item in enumerate(segment['units']) if item['kind'] == 'lyric']
    lyric_index = lyric_indices.index(index)
    evidence = {}
    for method in METHODS:
        evidence[method] = {}
        for variant in variants:
            candidates = row['runs'][method][variant].get('candidates')
            if not isinstance(candidates, list):
                raise ValueError('Missing frozen candidate list')
            if len(candidates) != len(lyric_indices) or text_key(candidates[lyric_index].get('text', '')) != text_key(unit['text']):
                evidence[method][variant] = {'status': 'text_mismatch', 'start_sample': None, 'start_seconds': None}
                continue
            candidate = candidates[lyric_index]
            if (candidate.get('original_sample_rate') != segment['sample_rate']
                    or candidate.get('original_num_samples') != segment['num_samples']):
                raise ValueError('Frozen candidate original sample grid differs')
            start = candidate.get('start_sample')
            seconds = candidate.get('start_seconds')
            if start is None or seconds is None or candidate.get('status') == 'unaligned':
                evidence[method][variant] = dict(candidate, start_sample=None, start_seconds=None)
                continue
            _integer(start, 'candidate original sample')
            _number(seconds, 'candidate original seconds', minimum=0)
            if start >= segment['num_samples'] or abs(seconds * segment['sample_rate'] - start) > 1.000001:
                raise ValueError('Frozen candidate sample/time differs or is out of range')
            evidence[method][variant] = candidate
    return dict(case, sample_rate=segment['sample_rate'], audio_path=segment['audio_path'],
                wav_sha256=segment['sources']['wav']['sha256'], methods=evidence)


def render_cases(prepared, evaluation, segments, out_dir):
    import numpy as np
    import scipy
    from scipy.signal import spectrogram
    from scripts.benchmark_dsp_gtsinger import read_audio
    plt, version = _plotting()
    by_id = {segment['segment_id']: segment for segment in segments}
    rows = {row['segment_id']: row for row in evaluation['rows']}
    if set(by_id) != set(rows):
        raise ValueError('Evaluation coverage differs from frozen segments')
    for identity, segment in by_id.items():
        if rows[identity]['song_id'] != segment['song_id'] or rows[identity]['split'] != segment['split']:
            raise ValueError('Evaluation song/split differs from frozen segments: ' + identity)
    cache, entries = {}, []
    for number, case in enumerate(prepared['failure_cases'], 1):
        identity = case['segment_id']
        segment = by_id[identity]
        evidence = case_evidence(case, rows[identity], segment, prepared['variants'])
        if identity not in cache:
            cache[identity] = read_audio(segment)
        audio, rate = cache[identity], segment['sample_rate']
        source = case['source_annotation_seconds']
        times = [source] + [candidate['start_seconds'] for runs in evidence['methods'].values()
                           for candidate in runs.values() if candidate['start_seconds'] is not None]
        times.extend(_number(alternative['seconds'], 'neural top candidate seconds', minimum=0)
                     for runs in evidence['methods'].values() for candidate in runs.values()
                     for alternative in candidate.get('refinement', {}).get('top_candidates', []))
        lower = max(0, math.floor((min(times) - .2) * rate))
        upper = min(len(audio), math.ceil((max(times) + .25) * rate))
        clip = audio[lower:upper]
        fig, axes = plt.subplots(3, 2, figsize=(14, 8), constrained_layout=True,
                                 gridspec_kw={'height_ratios': [1, 2, 1.3]}, sharex='col')
        notes = []
        for column, method in enumerate(METHODS):
            wave_ax, spectrum_ax, mass_ax = axes[:, column]
            wave_ax.plot(np.arange(lower, upper) / rate, clip, color='#29333d', linewidth=.5)
            wave_ax.set(ylabel='Original WAV amplitude', title=method.upper())
            window = min(960, len(clip))
            frequencies, seconds, spectrum = spectrogram(clip, rate, nperseg=window, noverlap=min(window - 1, window * 7 // 8))
            spectrum_ax.pcolormesh(seconds + lower / rate, frequencies / 1000,
                                  10 * np.log10(np.maximum(spectrum, 1e-12)), shading='auto', cmap='magma', vmin=-110, vmax=-35)
            spectrum_ax.set(ylabel='Frequency (kHz)', ylim=(0, min(8, rate / 2000)))
            for index, variant in enumerate(prepared['variants']):
                candidate = evidence['methods'][method][variant]
                if candidate['start_seconds'] is not None:
                    for ax in (wave_ax, spectrum_ax, mass_ax):
                        ax.axvline(candidate['start_seconds'], color=COLORS[index], linestyle='--',
                                   linewidth=1.1, alpha=.85, label=variant)
                detail = candidate.get('refinement', {})
                alternatives = detail.get('top_candidates', [])
                if alternatives:
                    mass_ax.scatter([item['seconds'] for item in alternatives], [item['heatmap_mass'] for item in alternatives],
                                    color=COLORS[index], s=24, marker='x' if variant == 'dsp_neural' else 'o')
                if 'confidence_logit' in detail:
                    notes.append('{}:{} logit={} | {}:{}'.format(method, variant,
                        '{:.3f}'.format(detail['confidence_logit']) if detail['confidence_logit'] is not None else 'null',
                        detail.get('status'), detail.get('reason')))
            for ax in (wave_ax, spectrum_ax, mass_ax):
                ax.axvline(source, color='#171717', linewidth=1.7, label='Published source annotation')
                ax.set_xlim(lower / rate, upper / rate)
            mass_ax.set(xlabel='Original WAV time (s; no resampling)', ylabel='Top-candidate mass\n(uncalibrated)', ylim=(-.02, 1.02))
            wave_ax.legend(loc='upper right', fontsize=7, ncol=2)
        title = display_text(case['text'][:12])
        heading = 'Case {} | {} | worst source AE {:.1f} ms ({}:{})'.format(
            number, title, case['worst_absolute_error_ms'], case['worst_method'], case['worst_variant'])
        fig.suptitle(textwrap.fill(heading, width=120) + '\nSource annotation is not human accuracy; mass/confidence are uncalibrated', fontsize=11)
        fig.supxlabel('\n'.join(textwrap.fill(display_text(note), width=170) for note in notes)
                      or 'No neural top-candidate/confidence evidence available', fontsize=7)
        name = 'failure-{:02d}.png'.format(number)
        fig.savefig(out_dir / name, dpi=160)
        plt.close(fig)
        entries.append(dict(evidence, file=name, spectrogram={'nperseg_samples': window,
            'noverlap_samples': min(window - 1, window * 7 // 8), 'time_convention': 'window centers in original WAV; no onset shift correction'},
            plotted_interval_samples=[lower, upper], display_text=title))
    return entries, {'matplotlib': version, 'numpy': np.__version__, 'scipy': scipy.__version__}


def _write_index(path, document):
    temporary = path.with_suffix('.json.tmp')
    try:
        temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('training-report', 'evaluation-report', 'out-dir'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--failure-cases', action='store_true')
    for name in ('segments', 'splits', 'audio-root'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args(argv)
    created, index = False, None
    try:
        supplied = [args.segments is not None, args.splits is not None, args.audio_root is not None]
        if any(supplied) != args.failure_cases or (args.failure_cases and not all(supplied)):
            raise ValueError('--failure-cases requires --segments --splits --audio-root together; overview requires no WAV arguments')
        sources = {'training_report': args.training_report, 'evaluation_report': args.evaluation_report,
                   'script': Path(__file__).resolve()}
        if args.failure_cases:
            sources.update(segments=args.segments, splits=args.splits)
        hashes = {name: digest(path) for name, path in sources.items()}
        training = _read_json(args.training_report)
        evaluation = _read_json(args.evaluation_report)
        prepared = prepare_results(training, evaluation)
        segments = None
        if args.failure_cases:
            for name in ('segments', 'splits'):
                if hashes[name] != evaluation['input_sha256'][name]:
                    raise ValueError('Failure plot input SHA256 differs: ' + name)
            segments = load_inputs(args.segments, args.splits, args.audio_root)
        args.out_dir.mkdir(parents=True, exist_ok=False)
        created = True
        index = {'schema_version': 1, 'protocol': 'boundary_diagnostics', 'status': 'running',
            'created_at': datetime.now(timezone.utc).isoformat(), 'input_sha256': hashes,
            'input_paths': {name: str(path.resolve()) for name, path in sources.items()},
            'training_input_sha256': training['input_sha256'],
            'evaluation_input_sha256': evaluation['input_sha256'],
            'training_runtime_sha256': training['runtime_sha256'],
            'evaluation_runtime_sha256': evaluation.get('runtime_sha256', {}),
            'checkpoint_sha256': evaluation['checkpoint_sha256'], 'reference': REFERENCE,
            'independent_human_accuracy': False, 'test_used_for_selection': False,
            'case_selection_policy': CASE_POLICY, 'failure_cases_requested': args.failure_cases,
            'variants': prepared['variants'], 'test_song_count': prepared['test_song_count'],
            'best_epoch': prepared['best_epoch'], 'cdf_denominator': 'all source boundaries, including missing and text mismatches',
            'text_rendering': 'ASCII U+ code points for non-ASCII glyphs; exact text preserved in this index',
            'distributions': {method: {variant: {key: value for key, value in distribution.items()
                if key not in ('errors_ms', 'cdf_fraction_of_all')} | {'evaluated': len(distribution['errors_ms'])}
                for variant, distribution in by_variant.items()} for method, by_variant in prepared['distributions'].items()},
            'failure_cases': [], 'images': {},
            'limitations': ['Published annotation disagreement is not independently verified human onset accuracy.',
                'Figures use frozen predictions; no test-driven parameter selection or inference.',
                'Largest-error ranking excludes missing/text-mismatch boundaries with no numerical error; CDF retains their denominator.',
                'Top-candidate heatmap mass and confidence logits are uncalibrated, not probabilities of correct onset.',
                'Spectrogram times denote centered analysis windows, not estimated onset times.']}
        _write_index(args.out_dir / 'index.json', index)
        index['environment'] = render_overview(prepared, args.out_dir)
        if args.failure_cases:
            entries, environment = render_cases(prepared, evaluation, segments, args.out_dir)
            index.update(failure_cases=entries, environment=environment)
        if any(digest(path) != hashes[name] for name, path in sources.items()):
            raise ValueError('Plot inputs/script changed during rendering')
        index['images'] = {path.name: digest(path) for path in sorted(args.out_dir.glob('*.png'))}
        index['status'] = 'completed'
        _write_index(args.out_dir / 'index.json', index)
    except Exception as error:
        if created and index is not None:
            index.update(status='failed', error={'type': type(error).__name__, 'message': str(error)})
            index['images'] = {path.name: digest(path) for path in sorted(args.out_dir.glob('*.png'))}
            _write_index(args.out_dir / 'index.json', index)
        print(type(error).__name__ + ': ' + str(error), file=sys.stderr)
        return 1
    print(args.out_dir / 'index.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(cli())
