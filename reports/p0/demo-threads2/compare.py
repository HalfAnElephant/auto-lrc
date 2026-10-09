"""Read saved P0 outputs, emit exact parity and layout shifts as JSON to stdout."""
import argparse
import json
import math
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, default=Path(__file__).resolve().parent.parent / 'original-threads2')
    parser.add_argument('--current', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    original_manifest = read(args.original / 'manifest.json')
    current_manifest = read(args.current / 'manifest.json')
    original_summary = read(args.original / 'summary.json')
    current_summary = read(args.current / 'summary.json')
    result = {
        'comparison_definition': 'Exact JSON value equality for each paired repeat; no rounding or tolerance.',
        'shift_definition': 'batch-first minus legacy start_frame, multiplied by frame_resolution_seconds; absolute-shift summaries are prediction differences, not reference-boundary errors.',
        'percentile_definition': 'Linear interpolation between ordered values at (n - 1) * fraction.',
        'original_config': original_manifest['config'],
        'current_config': current_manifest['config'],
        'legacy_parity': [], 'layout_ablation': [],
        'original_repeatability': original_summary['repeatability'],
        'current_repeatability': current_summary['repeatability'],
        'accuracy_metrics': current_summary['accuracy_metrics'],
    }
    original_files = {Path(row['path']).name: row for row in original_manifest['source_verification']}
    result['input_and_model_parity'] = [
        {'file': Path(row['path']).name,
         'identical_sha256': original_files[Path(row['path']).name]['sha256'] == row['sha256']}
        for row in current_manifest['inputs'] + current_manifest['models']
    ]
    for variant in ('mtl', 'mtl_bdr'):
        for repeat in (1, 2):
            original = read(args.original / '{}-r{}.json'.format(variant, repeat))
            current = read(args.current / '{}-legacy-r{}.json'.format(variant, repeat))
            alignment = current['result']['alignment']
            equal = {name: original[name] == alignment[name] for name in (
                'word_frames', 'score', 'phonemes', 'word_phoneme_indices', 'output_frames')}
            equal['enhanced_lrc'] = original['lrc'] == current['result']['lrc']
            result['legacy_parity'].append({'variant': variant, 'repeat': repeat,
                                           'equal': equal, 'all_equal': all(equal.values())})
        legacy = read(args.current / '{}-legacy-r1.json'.format(variant))
        corrected = read(args.current / '{}-batch-first-r1.json'.format(variant))
        old_units, new_units = legacy['result']['units'], corrected['result']['units']
        if [u['text'] for u in old_units] != [u['text'] for u in new_units]:
            raise ValueError('Unit texts are different; ordinal comparison would be invalid')
        resolution = legacy['result']['alignment']['frame_resolution_seconds']
        starts = [b['start_frame'] - a['start_frame'] for a, b in zip(old_units, new_units)]
        ends = [b['end_frame'] - a['end_frame'] for a, b in zip(old_units, new_units)]
        absolute_seconds = [abs(value) * resolution for value in starts]
        result['layout_ablation'].append({
            'variant': variant, 'unit_count': len(starts),
            'changed_start_count': sum(value != 0 for value in starts),
            'changed_end_count': sum(value != 0 for value in ends),
            'enhanced_lrc_identical': legacy['result']['lrc'] == corrected['result']['lrc'],
            'legacy_score': legacy['result']['alignment']['score'],
            'batch_first_score': corrected['result']['alignment']['score'],
            'mean_absolute_start_shift_seconds': statistics.mean(absolute_seconds),
            'median_absolute_start_shift_seconds': statistics.median(absolute_seconds),
            'p90_absolute_start_shift_seconds': percentile(absolute_seconds, 0.9),
            'maximum_absolute_start_shift_seconds': max(absolute_seconds),
            'minimum_signed_start_shift_frames': min(starts),
            'maximum_signed_start_shift_frames': max(starts),
            'state_dict_shapes_unchanged': all(x['state_dict_shapes_unchanged'] for x in corrected['lstm']),
            'all_observed_tensor_shapes_unchanged': all(
                corrected['timeline_audit']['observed_shapes'][key][field] == row[field]
                for key, row in legacy['timeline_audit']['observed_shapes'].items()
                for field in ('input', 'output')),
        })
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
